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
  const exprt array = numpy_param_as_array(get_expr(base));
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
std::optional<exprt> python_converter::try_get_numpy_storage_attr(
  const nlohmann::json &base,
  const std::string &attr)
{
  if (attr != "itemsize" && attr != "nbytes" && attr != "dtype")
    return std::nullopt;
  const auto dtype = numpy_array_dtype(base);
  if (!dtype)
    return std::nullopt;
  if (attr == "dtype")
    throw std::runtime_error(
      "TypeError: a numpy dtype is only supported in a comparison");
  const exprt itemsize = from_integer(dtype->itemsize(), int_type());
  if (attr == "itemsize")
    return itemsize;
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
    (op != "USub" && op != "Invert") ||
    !type_utils::is_numpy_scalar_type(leaf) || !dtype ||
    (op == "Invert" && !dtype->is_integer()))
    return nil_exprt();
  const exprt minus_one = from_integer(-1, long_long_int_type());
  return numpy_elementwise(
    op == "USub" ? "Mult" : "BitXor", operand, minus_one, operand.type());
}
