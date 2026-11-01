#include <python-frontend/json_utils.h>
#include <util/arith/arith_tools.h>
#include <python-frontend/numpy/numpy_call_expr.h>
#include <python-frontend/numpy/numpy_dtype.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type/type_utils.h>

#include <stdexcept>

static bool is_dtype_attribute(const nlohmann::json &node)
{
  return node.is_object() && node.value("_type", "") == "Attribute" &&
         node.value("attr", "") == "dtype";
}

// A probe of an operand: with the assignment's target in place, converting a
// slice would retype that target to a view.
exprt python_converter::get_expr_without_lhs(const nlohmann::json &node)
{
  exprt *const saved = current_lhs;
  current_lhs = nullptr;
  const exprt value = get_expr(node);
  current_lhs = saved;
  return value;
}

// The dtype of the array or view @p base denotes. Only a name or a subscript is
// read, since reading it twice must not repeat a call.
std::optional<numpy_dtypet>
python_converter::numpy_array_dtype(const nlohmann::json &base)
{
  const std::string kind = base.value("_type", "");
  if (
    (kind != "Name" && kind != "Subscript") ||
    json_utils::ast_contains_call(base))
    return std::nullopt;
  const exprt array = numpy_param_as_array(get_expr_without_lhs(base));
  const typet type = ns.follow(array.type());
  const bool is_view =
    type.is_pointer() && array.is_symbol() &&
    numpy_pointer_view_info_.count(array.identifier().as_string()) != 0;
  if (!type.is_array() && !is_view)
    return std::nullopt;
  const typet &storage = is_view ? type.subtype() : type;
  typet element = storage;
  while (element.is_array() && !type_utils::is_char_storage(element.subtype()))
    element = element.subtype();
  // A fixed-width string element: NumPy's `<U{n}` holds four bytes a character,
  // and the storage keeps a terminator after them.
  if (element.is_array())
  {
    if (element == storage)
      return std::nullopt;
    BigInt length;
    if (to_integer(to_array_type(element).size(), length))
      return std::nullopt;
    const unsigned width = length.to_uint64() - 1;
    return numpy_dtypet{numpy_dtypet::kindt::string, width * 32};
  }
  if (type_utils::is_char_storage(element))
    return std::nullopt;
  return numpy_dtype_of_element(element);
}

// `.itemsize` and `.nbytes` of an array or view; nbytes follows the logical
// shape, so a view reports its own bytes.
// A variable, parameter, attribute or container holds a Python number, so a
// narrow scalar leaving its array would stop wrapping at its width.
void python_converter::reject_numpy_scalar_crossing(
  const typet &type,
  const std::string &action) const
{
  // A signed 64-bit element has the model of a Python integer.
  const bool python_integer_width =
    type.is_signedbv() && to_signedbv_type(type).get_width() == 64;
  if (type_utils::is_numpy_scalar_type(type) && !python_integer_width)
    throw std::runtime_error(
      "TypeError: " + action +
      " a numpy scalar of a narrow dtype is not supported");
}

static const nlohmann::json &subscript_root(const nlohmann::json &node)
{
  const nlohmann::json *root = &node;
  while (root->value("_type", "") == "Subscript")
    root = &(*root)["value"];
  return *root;
}

// Whether the program may reach a string slice's source or copy other than by
// reading an element: a store, an alias, a parameter or a call argument.
static bool
names_one_of(const nlohmann::json &node, const std::set<std::string> &names)
{
  return node.value("_type", "") == "Name" &&
         names.count(node.value("id", "")) != 0;
}

static bool stores_or_aliases(
  const nlohmann::json &assign,
  const std::set<std::string> &names)
{
  const nlohmann::json targets = assign.value("_type", "") == "Assign"
                                   ? assign["targets"]
                                   : nlohmann::json::array({assign["target"]});
  for (const auto &target : targets)
    if (
      target.value("_type", "") == "Subscript" &&
      names_one_of(subscript_root(target), names))
      return true;
  return assign.contains("value") && names_one_of(assign["value"], names);
}

static bool
passes_to_callee(const nlohmann::json &call, const std::set<std::string> &names)
{
  static const std::set<std::string> readers = {
    "len", "list", "tuple", "print"};
  const auto &args = call.value("args", nlohmann::json::array());
  return readers.count(call["func"].value("id", "")) == 0 &&
         call["func"].value("attr", "") != "choice" &&
         std::any_of(
           args.begin(), args.end(), [&names](const nlohmann::json &arg) {
             return names_one_of(arg, names);
           });
}

static bool string_slice_may_diverge(
  const nlohmann::json &node,
  const std::set<std::string> &names)
{
  if (!node.is_object() && !node.is_array())
    return false;
  const std::string kind = node.is_object() ? node.value("_type", "") : "";
  if (
    ((kind == "Assign" || kind == "AugAssign" || kind == "AnnAssign") &&
     stores_or_aliases(node, names)) ||
    (kind == "arg" && names.count(node.value("arg", "")) != 0) ||
    (kind == "Call" && passes_to_callee(node, names)))
    return true;
  return std::any_of(
    node.begin(), node.end(), [&names](const nlohmann::json &child) {
      return string_slice_may_diverge(child, names);
    });
}

static bool calls_len_of(const nlohmann::json &node, const std::string &name)
{
  if (!node.is_object() && !node.is_array())
    return false;
  if (
    node.is_object() && node.value("_type", "") == "Call" &&
    node["func"].value("id", "") == "len" && !node["args"].empty() &&
    node["args"][0].value("id", "") == name)
    return true;
  for (const auto &child : node)
    if (calls_len_of(child, name))
      return true;
  return false;
}

static void collect_slice_holders(
  const nlohmann::json &node,
  const std::string &source,
  std::set<std::string> &holders)
{
  if (!node.is_object() && !node.is_array())
    return;
  const std::string kind = node.is_object() ? node.value("_type", "") : "";
  if (
    (kind == "Assign" || kind == "AnnAssign") && node.contains("value") &&
    node["value"].is_object() &&
    node["value"].value("_type", "") == "Subscript" &&
    node["value"]["slice"].value("_type", "") == "Slice" &&
    subscript_root(node["value"]).value("id", "") == source)
  {
    const nlohmann::json targets = kind == "Assign"
                                     ? node["targets"]
                                     : nlohmann::json::array({node["target"]});
    for (const auto &target : targets)
      if (target.value("_type", "") == "Name")
        holders.insert(target.value("id", ""));
  }
  for (const auto &child : node)
    collect_slice_holders(child, source, holders);
}

// NumPy's string slice shares storage; ours is a copy (ADR-NP-003 stage 1), so
// it is only sound when neither side can change.
void python_converter::reject_unsafe_numpy_string_slice(
  const nlohmann::json &array) const
{
  const std::string id = subscript_root(array).value("id", "");
  std::set<std::string> names = {id};
  collect_slice_holders(*ast_json, id, names);
  if (id.empty() || string_slice_may_diverge(*ast_json, names))
    throw std::runtime_error(
      "TypeError: a slice of a numpy string array that is written, aliased or "
      "passed to a function is not supported");
}

// `s = a[0]`: the name keeps the array's width and shares its storage, so only
// reads that see neither are sound.
void python_converter::reject_unsafe_numpy_string_binding(
  const nlohmann::json &value,
  const exprt &lhs)
{
  if (
    !lhs.is_symbol() || value.value("_type", "") != "Subscript" ||
    value["slice"].value("_type", "") == "Slice" ||
    json_utils::ast_contains_call(value))
    return;
  const auto dtype = numpy_array_dtype(value["value"]);
  if (!dtype || dtype->kind != numpy_dtypet::kindt::string)
    return;
  const std::string id = lhs.identifier().as_string();
  const std::string root = subscript_root(value).value("id", "");
  if (
    string_slice_may_diverge(*ast_json, {root}) ||
    calls_len_of(*ast_json, id.substr(id.rfind('@') + 1)))
    throw std::runtime_error(
      "TypeError: binding an element of a numpy string array that is written "
      "or measured with len() is not supported");
}

void python_converter::reject_numpy_scalars_in_container(
  const nlohmann::json &assign)
{
  if (!assign.contains("value") || !assign["value"].is_object())
    return;
  for (const char *field : {"values", "elts"})
    for (const auto &item :
         assign["value"].value(field, nlohmann::json::array()))
      if (item.is_object())
        reject_numpy_scalar_read(item, "storing in a container");
}

void python_converter::reject_numpy_scalar_return(const nlohmann::json &value)
{
  if (value.is_object())
    reject_numpy_scalar_read(value, "returning");
}

void python_converter::reject_numpy_scalar_read(
  const nlohmann::json &value,
  const std::string &action)
{
  if (
    value.value("_type", "") == "Subscript" &&
    !json_utils::ast_contains_call(value))
    reject_numpy_scalar_crossing(get_expr_without_lhs(value).type(), action);
}

void python_converter::reject_numpy_scalar_in_attribute(
  const exprt &lhs,
  const exprt &rhs) const
{
  if (lhs.id() == "member")
    reject_numpy_scalar_crossing(rhs.type(), "storing in an attribute");
}

void python_converter::reject_subscript_of_unmodelled_method(
  const nlohmann::json &element)
{
  if (
    element.value("_type", "") == "Subscript" &&
    element["value"].value("_type", "") == "Call")
    reject_unmodelled_array_method(element["value"]);
}

// A method of a string or complex array with no model would read its storage
// as numbers.
void python_converter::reject_unmodelled_array_method(
  const nlohmann::json &call)
{
  const nlohmann::json &func = call["func"];
  if (func.value("_type", "") != "Attribute")
    return;
  const nlohmann::json &root = subscript_root(func["value"]);
  if (
    root.value("_type", "") != "Name" ||
    json_utils::ast_contains_call(func["value"]) ||
    !numpy_array_symbols_.count(resolve_name_symbol_id(root.value("id", ""))))
    return;
  const std::string method = func.value("attr", "");
  if (func["value"].value("_type", "") == "Subscript")
    reject_numpy_scalar_crossing(
      get_expr_without_lhs(func["value"]).type(), "." + method + "() of");
  const auto dtype = numpy_array_dtype(func["value"]);
  if (
    dtype && method == "tolist" &&
    dtype->kind == numpy_dtypet::kindt::unsigned_integer && dtype->bits == 64)
    throw std::runtime_error(
      "TypeError: .tolist() of a numpy uint64 array is not supported");
  if (dtype && !numpy_consumer_is_modelled(dtype->kind, method))
    throw std::runtime_error(
      "TypeError: ." + method + "() of a numpy " +
      numpy_kind_name(dtype->kind) + " array is not supported");
}

std::optional<exprt> python_converter::try_get_numpy_storage_attr(
  const nlohmann::json &base,
  const std::string &attr)
{
  const auto dtype = numpy_array_dtype(base);
  if (!dtype)
    return std::nullopt;
  if (!numpy_consumer_is_modelled(dtype->kind, attr))
    throw std::runtime_error(
      "TypeError: ." + attr + " of a numpy " + numpy_kind_name(dtype->kind) +
      " array is not supported");
  if (attr != "itemsize" && attr != "nbytes" && attr != "dtype")
    return std::nullopt;
  if (attr == "dtype")
    throw std::runtime_error(
      "TypeError: a numpy dtype is only supported in a comparison");
  const exprt itemsize = from_integer(dtype->itemsize(), int_type());
  if (attr == "itemsize")
    return itemsize;
  const typet storage = ns.follow(numpy_param_as_array(get_expr(base)).type());
  if (storage.is_array())
  {
    std::size_t total = dtype->itemsize();
    for (const std::size_t extent : type_handler_.get_array_type_shape(storage))
      total *= extent;
    if (dtype->kind == numpy_dtypet::kindt::string)
      total /= to_array_type(storage).subtype().is_array()
                 ? type_handler_.get_array_type_shape(storage).back()
                 : 1;
    return from_integer(total, int_type());
  }
  nlohmann::json size = {
    {"_type", "Attribute"}, {"value", base}, {"attr", "size"}};
  const exprt count = typecast_exprt(get_expr(size), int_type());
  exprt nbytes("*", int_type());
  nbytes.copy_to_operands(count, itemsize);
  return nbytes;
}

// `a.dtype == np.int8`: decided here, as NumPy compares any spelling of a dtype
// and not the text of its name.
std::optional<exprt>
python_converter::try_numpy_dtype_compare(const nlohmann::json &element)
{
  if (element.value("_type", "") != "Compare" || element["ops"].size() != 1)
    return std::nullopt;
  const std::string op = element["ops"][0].value("_type", "");
  if (op != "Eq" && op != "NotEq")
    return std::nullopt;
  const nlohmann::json &left = element["left"];
  const nlohmann::json &right = element["comparators"][0];
  const auto side_name = [&](const nlohmann::json &side) -> std::string {
    if (!is_dtype_attribute(side))
      return extract_numpy_dtype_name(side, *ast_json);
    const auto dtype = numpy_array_dtype(side["value"]);
    if (!dtype)
      throw std::runtime_error(
        "TypeError: .dtype of this value is not supported");
    if (dtype->kind == numpy_dtypet::kindt::string)
      throw std::runtime_error(
        "TypeError: comparing the dtype of a numpy string array is not "
        "supported");
    return dtype->name();
  };
  if (!is_dtype_attribute(left) && !is_dtype_attribute(right))
    return std::nullopt;
  const bool same = side_name(left) == side_name(right);
  return gen_boolean(same == (op == "Eq"));
}

static std::size_t
stores_to(const nlohmann::json &node, const std::string &name)
{
  std::size_t stores = 0;
  if (
    node.is_object() && node.value("_type", "") == "Name" &&
    node.value("id", "") == name && node.contains("ctx") &&
    node["ctx"].value("_type", "") != "Load")
    ++stores;
  if (node.is_object() || node.is_array())
    for (const auto &child : node)
      stores += stores_to(child, name);
  return stores;
}

void python_converter::bind_numpy_scalar_name(
  const nlohmann::json &target,
  symbolt *symbol,
  exprt &lhs,
  const exprt &rhs)
{
  // The annotator reads `1j * a` or `-a` as a scalar from its literal operand.
  if (
    symbol && target.value("_type", "") == "Name" &&
    type_utils::is_numpy_complex_array(rhs.type()) && !lhs.type().is_array())
  {
    lhs.type() = rhs.type();
    symbol->set_type(rhs.type());
    return;
  }
  if (
    !symbol || target.value("_type", "") != "Name" ||
    !type_utils::is_numpy_scalar_type(rhs.type()) || lhs.type() == rhs.type())
    return;
  const std::string name = target.value("id", "");
  const std::string function = current_function_name();
  const nlohmann::json scope =
    function.empty()
      ? nlohmann::json()
      : json_utils::try_find_function((*ast_json)["body"], function);
  const nlohmann::json &body = scope.is_object() && scope.contains("body")
                                 ? scope["body"]
                                 : (*ast_json)["body"];
  if (stores_to(body, name) > 1)
    throw std::runtime_error(
      "TypeError: rebinding a name holding a narrow numpy scalar is not "
      "supported");
  lhs.type() = rhs.type();
  symbol->set_type(rhs.type());
}

// `-a` and `~a` over narrow integers and floats, wrapped at the operand's
// width. Nil for any other operand.
exprt python_converter::numpy_narrow_unary(
  const std::string &op,
  const exprt &operand)
{
  const typet leaf = type_utils::array_scalar_type(operand.type());
  const auto dtype = numpy_dtype_of_element(leaf);
  if (
    (op != "USub" && op != "Invert") || !dtype ||
    (op == "Invert" && !dtype->is_integer()) ||
    (!type_utils::is_numpy_scalar_type(leaf) &&
     !type_utils::is_numpy_complex_array(operand.type())))
    return nil_exprt();
  const exprt minus_one = from_integer(-1, long_long_int_type());
  return numpy_elementwise(
    op == "USub" ? "Mult" : "BitXor", operand, minus_one, operand.type());
}

// `x in a` over a numpy string array, as equality with each element in turn.
std::optional<exprt>
python_converter::try_numpy_string_membership(const nlohmann::json &element)
{
  if (element.value("_type", "") != "Compare" || element["ops"].size() != 1)
    return std::nullopt;
  const std::string op = element["ops"][0].value("_type", "");
  if (op != "In" && op != "NotIn")
    return std::nullopt;
  const nlohmann::json &array = element["comparators"][0];
  const nlohmann::json &root =
    array.value("_type", "") == "Subscript" ? array["value"] : array;
  const std::string id = root.value("_type", "") == "Name"
                           ? resolve_name_symbol_id(root.value("id", ""))
                           : std::string();
  if (
    numpy_array_symbols_.count(id) == 0 &&
    numpy_param_array_types_.count(id) == 0)
    return std::nullopt;
  const auto dtype = numpy_array_dtype(array);
  if (!dtype || dtype->kind != numpy_dtypet::kindt::string)
    return std::nullopt;
  if (
    ns.follow(numpy_param_as_array(get_expr(array)).type())
      .subtype()
      .subtype()
      .is_array())
    throw std::runtime_error(
      "TypeError: membership in a numpy string array of rank above 1 is not "
      "supported");
  if (json_utils::ast_contains_call(element["left"]))
    throw std::runtime_error(
      "TypeError: membership of a call's value in a numpy string array is not "
      "supported");
  const exprt storage = numpy_param_as_array(get_expr(array));
  BigInt length;
  to_integer(to_array_type(ns.follow(storage.type())).size(), length);
  exprt found = false_exprt();
  for (long long i = 0; i < length.to_int64(); ++i)
  {
    nlohmann::json item = {
      {"_type", "Subscript"},
      {"value", array},
      {"slice", {{"_type", "Constant"}, {"value", i}}}};
    nlohmann::json equal = element;
    equal["ops"][0]["_type"] = "Eq";
    equal["comparators"][0] = item;
    found = or_exprt(found, typecast_exprt(get_expr(equal), bool_type()));
  }
  return op == "In" ? found : exprt(not_exprt(found));
}
