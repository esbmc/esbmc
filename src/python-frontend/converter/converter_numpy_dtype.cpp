#include <python-frontend/json_utils.h>
#include <util/arith/arith_tools.h>
#include <python-frontend/numpy/numpy_call_expr.h>
#include <python-frontend/numpy/numpy_dtype.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/type/type_utils.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

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

static bool
passes_string_slice_to_callee(const nlohmann::json &node, const std::string &id)
{
  if (!node.is_object() && !node.is_array())
    return false;
  if (node.is_object() && node.value("_type", "") == "Call")
  {
    static const std::set<std::string> readers = {
      "len", "list", "tuple", "print"};
    if (
      readers.count(node["func"].value("id", "")) == 0 &&
      node["func"].value("attr", "") != "choice")
      for (const auto &arg : node.value("args", nlohmann::json::array()))
        if (
          arg.value("_type", "") == "Subscript" &&
          arg["slice"].value("_type", "") == "Slice" &&
          subscript_root(arg).value("id", "") == id)
          return true;
  }
  for (const auto &child : node)
    if (passes_string_slice_to_callee(child, id))
      return true;
  return false;
}

// NumPy's string slice shares storage; ours is a copy (ADR-NP-003 stage 1). A
// slice read where it is written sees the current elements; one held by a name
// is only sound when neither side can change.
void python_converter::reject_unsafe_numpy_string_slice(
  const nlohmann::json &array) const
{
  const std::string id = subscript_root(array).value("id", "");
  std::set<std::string> names = {id};
  collect_slice_holders(*ast_json, id, names);
  if (
    id.empty() || passes_string_slice_to_callee(*ast_json, id) ||
    (names.size() > 1 && string_slice_may_diverge(*ast_json, names)))
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

namespace
{
// The sites binding or parameterising each name across the program.
void count_name_bindings(
  const nlohmann::json &node,
  std::map<std::string, std::size_t> &bindings)
{
  if (!node.is_object() && !node.is_array())
    return;
  if (node.is_object())
  {
    const std::string kind = node.value("_type", "");
    if (
      kind == "Name" && node.contains("ctx") &&
      node["ctx"].value("_type", "") != "Load")
      ++bindings[node.value("id", "")];
    else if (kind == "arg")
      ++bindings[node.value("arg", "")];
    else if (kind == "FunctionDef" || kind == "ClassDef")
      ++bindings[node.value("name", "")];
  }
  for (const auto &child : node)
    count_name_bindings(child, bindings);
}

std::optional<long long> literal_index(const nlohmann::json &node)
{
  if (
    node.value("_type", "") == "Constant" && node["value"].is_number_integer())
    return node["value"].get<long long>();
  if (
    node.value("_type", "") == "UnaryOp" &&
    node["op"].value("_type", "") == "USub")
    if (const auto value = literal_index(node["operand"]))
      return -*value;
  return std::nullopt;
}

nlohmann::json located_index(long long value, const nlohmann::json &source)
{
  nlohmann::json index = {{"_type", "Constant"}, {"value", value}};
  for (const char *key :
       {"lineno", "col_offset", "end_lineno", "end_col_offset"})
    if (source.contains(key))
      index[key] = source[key];
  return index;
}

// The elements `base[slice]` selects of an axis of @p length, when literal.
std::optional<std::vector<long long>>
literal_slice_indices(const nlohmann::json &slice, long long length)
{
  const auto bound = [](const nlohmann::json &node, long long fallback) {
    return node.is_null() ? std::optional<long long>(fallback)
                          : literal_index(node);
  };
  const auto step = bound(slice["step"], 1);
  if (!step || *step == 0)
    return std::nullopt;
  const auto lower = bound(slice["lower"], *step > 0 ? 0 : length - 1);
  const auto upper = bound(slice["upper"], *step > 0 ? length : -length - 1);
  if (!lower || !upper)
    return std::nullopt;
  const auto clamp = [&](long long value) {
    if (value < 0)
      value += length;
    const long long low = *step > 0 ? 0 : -1;
    const long long high = *step > 0 ? length : length - 1;
    return std::min(std::max(value, low), high);
  };
  std::vector<long long> indices;
  const long long last = clamp(*upper);
  for (long long i = clamp(*lower); *step > 0 ? i < last : i > last; i += *step)
    indices.push_back(i);
  return indices;
}

struct string_view_aliast
{
  nlohmann::json value;
  // For a slice, the source index of each element it holds.
  std::optional<std::vector<long long>> indices;
};

bool replace_string_view_read(
  nlohmann::json &node,
  const std::string &name,
  const string_view_aliast &alias);

// Reads and writes through the name become reads and writes of its source.
void replace_string_view(
  nlohmann::json &node,
  const std::string &name,
  const string_view_aliast &alias)
{
  if (!node.is_object() && !node.is_array())
    return;
  if (
    node.is_object() && node.value("_type", "") == "Subscript" &&
    node["value"].value("id", "") == name &&
    replace_string_view_read(node, name, alias))
    return;
  if (
    node.is_object() && node.value("_type", "") == "Name" &&
    node.value("id", "") == name)
  {
    nlohmann::json source = alias.value;
    node = std::move(source);
    return;
  }
  for (auto &child : node)
    replace_string_view(child, name, alias);
}

bool replace_string_view_read(
  nlohmann::json &node,
  const std::string &name,
  const string_view_aliast &alias)
{
  if (node["value"].value("_type", "") != "Name")
    return false;
  if (!alias.indices)
  {
    replace_string_view(node["slice"], name, alias);
    nlohmann::json source = alias.value;
    source["ctx"] = {{"_type", "Load"}};
    node["value"] = std::move(source);
    return true;
  }
  const auto index = literal_index(node["slice"]);
  const long long length = alias.indices->size();
  const long long at =
    index && *index < 0 ? *index + length : index.value_or(-1);
  if (!index || at < 0 || at >= length)
    return false;
  node["value"] = alias.value["value"];
  node["slice"] = located_index((*alias.indices)[at], node["slice"]);
  return true;
}

const nlohmann::json *single_assign_target(const nlohmann::json &statement)
{
  const std::string kind = statement.value("_type", "");
  if (kind == "AnnAssign")
    return &statement["target"];
  if (kind == "Assign" && statement["targets"].size() == 1)
    return &statement["targets"][0];
  return nullptr;
}

void copy_statement_location(const nlohmann::json &from, nlohmann::json &to)
{
  for (const char *key :
       {"lineno", "col_offset", "end_lineno", "end_col_offset"})
    if (from.contains(key))
      to[key] = from[key];
}

bool is_literal_string_array_call(
  const nlohmann::json &ast,
  const nlohmann::json &value)
{
  return value.value("_type", "") == "Call" &&
         value["func"].value("attr", "") == "array" &&
         value["func"]["value"].value("_type", "") == "Name" &&
         is_imported_numpy_module_alias(
           ast, value["func"]["value"].value("id", "")) &&
         numpy_literal_string_width(value).has_value();
}

// The source array of a stable `name = source[...]` in @p body, and its
// literal.
const nlohmann::json *sole_string_array(
  const nlohmann::json &ast,
  const nlohmann::json &body,
  const std::string &source)
{
  for (const auto &statement : body)
  {
    const nlohmann::json *target = single_assign_target(statement);
    if (
      target && target->value("id", "") == source &&
      statement["value"].is_object() &&
      is_literal_string_array_call(ast, statement["value"]))
      return &statement["value"];
  }
  return nullptr;
}

// Whether `other = name` copies the array under a second name; the model keeps
// no alias between them, so a view of it cannot follow a write through the
// copy.
bool binds_name_elsewhere(const nlohmann::json &node, const std::string &name)
{
  if (!node.is_object() && !node.is_array())
    return false;
  const std::string kind = node.is_object() ? node.value("_type", "") : "";
  if (
    (kind == "Assign" || kind == "AnnAssign") && node.contains("value") &&
    node["value"].is_object() && node["value"].value("id", "") == name)
    return true;
  for (const auto &child : node)
    if (binds_name_elsewhere(child, name))
      return true;
  return false;
}

// `a[l:h][i]` over a literal string array reads `a[j]`, the element it selects.
void read_through_chained_slices(
  nlohmann::json &node,
  const nlohmann::json &ast,
  const nlohmann::json &body)
{
  if (!node.is_object() && !node.is_array())
    return;
  for (auto &child : node)
    read_through_chained_slices(child, ast, body);
  if (
    !node.is_object() || node.value("_type", "") != "Subscript" ||
    node["value"].value("_type", "") != "Subscript" ||
    node["value"]["slice"].value("_type", "") != "Slice" ||
    node["value"]["value"].value("_type", "") != "Name")
    return;
  const nlohmann::json *array =
    sole_string_array(ast, body, node["value"]["value"].value("id", ""));
  const auto index = literal_index(node["slice"]);
  if (!array || !index)
    return;
  const auto indices = literal_slice_indices(
    node["value"]["slice"], (*array)["args"][0]["elts"].size());
  if (!indices)
    return;
  const long long length = indices->size();
  const long long at = *index < 0 ? *index + length : *index;
  if (at < 0 || at >= length)
    return;
  const nlohmann::json source = node["value"]["value"];
  node["slice"] = located_index((*indices)[at], node["slice"]);
  node["value"] = source;
}

// The alias of `value`, a read of @p array, when it selects a known slice or
// row.
std::optional<string_view_aliast>
string_view_alias_of(const nlohmann::json &value, const nlohmann::json &array)
{
  string_view_aliast alias{value, std::nullopt};
  const auto &elements = array["args"][0]["elts"];
  if (value["slice"].value("_type", "") == "Slice")
  {
    alias.indices = literal_slice_indices(value["slice"], elements.size());
    return alias.indices ? std::optional(alias) : std::nullopt;
  }
  if (
    !literal_index(value["slice"]) || elements.empty() ||
    elements[0].value("_type", "") != "List")
    return std::nullopt;
  return alias;
}

bool is_name_bound_to_subscript(const nlohmann::json &statement)
{
  const nlohmann::json *target = single_assign_target(statement);
  return target && target->value("_type", "") == "Name" &&
         statement["value"].is_object() &&
         statement["value"].value("_type", "") == "Subscript" &&
         statement["value"]["value"].value("_type", "") == "Name";
}

void replace_string_view_uses(
  nlohmann::json &body,
  std::size_t binding,
  const std::string &name,
  const string_view_aliast &alias)
{
  for (std::size_t later = binding + 1; later < body.size(); ++later)
    replace_string_view(body[later], name, alias);
  for (std::size_t earlier = 0; earlier < binding; ++earlier)
    if (body[earlier].value("_type", "") == "FunctionDef")
      replace_string_view(body[earlier], name, alias);
}

void alias_string_views_in(const nlohmann::json &ast, nlohmann::json &body)
{
  read_through_chained_slices(body, ast, body);
  std::map<std::string, std::size_t> bindings;
  count_name_bindings(body, bindings);
  for (std::size_t i = 0; i < body.size(); ++i)
  {
    nlohmann::json &statement = body[i];
    if (!is_name_bound_to_subscript(statement))
      continue;
    const nlohmann::json &value = statement["value"];
    const std::string name = single_assign_target(statement)->value("id", "");
    const std::string source = value["value"].value("id", "");
    const nlohmann::json *array = sole_string_array(ast, body, source);
    if (
      !array || bindings[name] != 1 || bindings[source] != 1 ||
      binds_name_elsewhere(body, source))
      continue;
    const auto alias = string_view_alias_of(value, *array);
    if (!alias)
      continue;
    nlohmann::json evaluated = {{"_type", "Expr"}, {"value", value}};
    copy_statement_location(statement, evaluated);
    statement = std::move(evaluated);
    replace_string_view_uses(body, i, name, *alias);
  }
}

void alias_string_views_in_scopes(
  const nlohmann::json &ast,
  nlohmann::json &body)
{
  alias_string_views_in(ast, body);
  for (auto &statement : body)
    if (statement.value("_type", "") == "FunctionDef")
      alias_string_views_in_scopes(ast, statement["body"]);
}
} // namespace

// A stable name for a slice or row of a literal string array is an alias of its
// source (ADR-NP-003): the storage is one, whatever is written through it.
namespace
{
std::string decoded_bytes_literal(const nlohmann::json &node)
{
  static const std::string alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string text;
  unsigned buffer = 0;
  int bits = 0;
  for (const char c : node.value("encoded_bytes", std::string()))
  {
    const std::size_t digit = alphabet.find(c);
    if (digit == std::string::npos)
      break;
    buffer = (buffer << 6) | static_cast<unsigned>(digit);
    bits += 6;
    if (bits >= 8)
    {
      bits -= 8;
      text.push_back(static_cast<char>((buffer >> bits) & 0xFF));
    }
  }
  return text;
}

bool is_ascii_text(const std::string &text)
{
  return std::all_of(text.begin(), text.end(), [](char c) {
    return static_cast<unsigned char>(c) < 128 && c != 0;
  });
}

bool is_bytes_constant(const nlohmann::json &node)
{
  return node.value("_type", "") == "Constant" &&
         node.contains("encoded_bytes");
}

bool is_text_constant(const nlohmann::json &node)
{
  return node.value("_type", "") == "Constant" && node["value"].is_string() &&
         !node.contains("encoded_bytes");
}

// A bytes literal as the text its ASCII storage holds.
void bytes_literal_to_text(nlohmann::json &node)
{
  if (!is_bytes_constant(node))
    return;
  const std::string text = decoded_bytes_literal(node);
  if (!is_ascii_text(text))
    throw std::runtime_error(
      "TypeError: non-ASCII bytes in a numpy bytes array are not supported");
  node["value"] = text;
  node.erase("encoded_bytes");
  node.erase("esbmc_type_annotation");
  if (node.contains("kind"))
    node.erase("kind");
}

void texts_of_bytes_elements(nlohmann::json &node)
{
  if (node.is_array())
    for (auto &child : node)
      texts_of_bytes_elements(child);
  else if (node.value("_type", "") == "List")
    texts_of_bytes_elements(node["elts"]);
  else
    bytes_literal_to_text(node);
}

// `S<n>` as the fixed-width storage of `U<n>`, or nullopt for another dtype.
std::optional<std::string> bytes_dtype_as_text(const nlohmann::json &dtype)
{
  if (dtype.value("_type", "") != "Constant" || !dtype["value"].is_string())
    return std::nullopt;
  const std::string name = dtype["value"].get<std::string>();
  if (
    name.empty() || name[0] != 'S' ||
    !std::all_of(name.begin() + 1, name.end(), [](unsigned char c) {
      return std::isdigit(c) != 0;
    }))
    return std::nullopt;
  return name.size() == 1 ? std::string("U1") : "U" + name.substr(1);
}

bool reads_element_of(const nlohmann::json &node, const std::string &name)
{
  return node.value("_type", "") == "Subscript" &&
         subscript_root(node).value("id", "") == name;
}

// The uses a bytes array has a model for: an element compared with a literal or
// stored from one, and its shape. A str never equals bytes, which the storage
// alone would not show.
bool lower_bytes_compare(nlohmann::json &node, const std::string &name)
{
  if (node.value("_type", "") != "Compare" || node["ops"].size() != 1)
    return false;
  const std::string op = node["ops"][0].value("_type", "");
  nlohmann::json &left = node["left"];
  nlohmann::json &right = node["comparators"][0];
  const bool left_read = reads_element_of(left, name);
  if (
    (op != "Eq" && op != "NotEq") ||
    (!left_read && !reads_element_of(right, name)))
    return false;
  nlohmann::json &literal = left_read ? right : left;
  if (is_bytes_constant(literal))
  {
    bytes_literal_to_text(literal);
    return true;
  }
  if (!is_text_constant(literal))
    return false;
  nlohmann::json folded = {{"_type", "Constant"}, {"value", op == "NotEq"}};
  copy_statement_location(node, folded);
  node = std::move(folded);
  return true;
}

bool lower_bytes_store(nlohmann::json &node, const std::string &name)
{
  const nlohmann::json *target = single_assign_target(node);
  if (!target)
    return false;
  if (target->value("id", "") == name)
    return true;
  if (!reads_element_of(*target, name))
    return false;
  bytes_literal_to_text(node["value"]);
  if (!is_text_constant(node["value"]))
    throw std::runtime_error(
      "TypeError: storing a non-literal into a numpy bytes array is not "
      "supported");
  return true;
}

bool reads_bytes_array_shape(
  const nlohmann::json &node,
  const std::string &name)
{
  const std::string kind = node.value("_type", "");
  if (kind == "Call")
    return node["func"].value("id", "") == "len" && node["args"].size() == 1 &&
           node["args"][0].value("id", "") == name;
  if (kind != "Attribute" || node["value"].value("id", "") != name)
    return false;
  const std::string attr = node.value("attr", "");
  return attr == "shape" || attr == "size" || attr == "ndim";
}

void lower_bytes_array_uses(nlohmann::json &node, const std::string &name)
{
  if (!node.is_object() && !node.is_array())
    return;
  if (node.is_array())
  {
    for (auto &child : node)
      lower_bytes_array_uses(child, name);
    return;
  }
  const std::string kind = node.value("_type", "");
  if (
    lower_bytes_compare(node, name) || lower_bytes_store(node, name) ||
    reads_bytes_array_shape(node, name))
    return;
  if (kind == "Name" && node.value("id", "") == name)
    throw std::runtime_error(
      "TypeError: this use of a numpy bytes array is not supported");
  for (auto &child : node)
    lower_bytes_array_uses(child, name);
}

bool holds_bytes_constant(const nlohmann::json &node)
{
  if (node.value("_type", "") != "List")
    return is_bytes_constant(node);
  return std::any_of(
    node["elts"].begin(), node["elts"].end(), holds_bytes_constant);
}

// Only an explicit `S<n>` dtype has a lowering; NumPy's inference has none.
void reject_untyped_bytes_array(
  const nlohmann::json &ast,
  const nlohmann::json &call)
{
  const nlohmann::json &func = call["func"];
  if (
    func.value("_type", "") != "Attribute" ||
    func.value("attr", "") != "array" ||
    func["value"].value("_type", "") != "Name" ||
    !is_imported_numpy_module_alias(ast, func["value"].value("id", "")) ||
    call["args"].empty() || !holds_bytes_constant(call["args"][0]))
    return;
  for (const auto &keyword : call["keywords"])
    if (
      keyword.value("arg", "") == "dtype" &&
      bytes_dtype_as_text(keyword["value"]))
      return;
  throw std::runtime_error(
    "TypeError: numpy.array() of bytes without an S<n> dtype is not supported");
}

bool is_numpy_module_call(
  const nlohmann::json &ast,
  const nlohmann::json &value)
{
  return value.is_object() && value.value("_type", "") == "Call" &&
         value["func"].value("_type", "") == "Attribute" &&
         value["func"]["value"].value("_type", "") == "Name" &&
         is_imported_numpy_module_alias(
           ast, value["func"]["value"].value("id", ""));
}

// Lowers `name = np.f(..., dtype="S<n>")`, recording a name bound only there.
void lower_bytes_binding(
  const nlohmann::json &ast,
  nlohmann::json &statement,
  std::map<std::string, std::size_t> &bindings,
  std::vector<std::string> &names)
{
  const nlohmann::json *target = single_assign_target(statement);
  if (!target || !is_numpy_module_call(ast, statement["value"]))
    return;
  const std::string name = target->value("id", "");
  for (auto &keyword : statement["value"]["keywords"])
  {
    const auto text = keyword.value("arg", "") == "dtype"
                        ? bytes_dtype_as_text(keyword["value"])
                        : std::nullopt;
    if (!text || bindings[name] != 1)
      continue;
    keyword["value"]["value"] = *text;
    texts_of_bytes_elements(statement["value"]["args"]);
    names.push_back(name);
  }
}

void lower_bytes_arrays(nlohmann::json &ast)
{
  std::map<std::string, std::size_t> bindings;
  count_name_bindings(ast["body"], bindings);
  std::vector<std::string> names;
  const std::function<void(nlohmann::json &)> visit =
    [&](nlohmann::json &node) {
      if (!node.is_object() && !node.is_array())
        return;
      for (auto &child : node)
        visit(child);
      if (!node.is_object())
        return;
      if (node.value("_type", "") == "Call")
        reject_untyped_bytes_array(ast, node);
      lower_bytes_binding(ast, node, bindings, names);
    };
  visit(ast["body"]);
  for (const std::string &name : names)
    lower_bytes_array_uses(ast["body"], name);
}
} // namespace

// A comparison whose result is held or indexed is the array NumPy computes; one
// read for its truth is not.
static void mark_compares_read_as_arrays(nlohmann::json &node)
{
  if (!node.is_object() && !node.is_array())
    return;
  if (node.is_object())
  {
    const std::string kind = node.value("_type", "");
    if (
      (kind == "Assign" || kind == "AnnAssign" || kind == "Subscript") &&
      node.contains("value") && node["value"].is_object() &&
      node["value"].value("_type", "") == "Compare")
      node["value"]["numpy_array_result"] = true;
  }
  for (auto &child : node)
    mark_compares_read_as_arrays(child);
}

void alias_numpy_string_views(nlohmann::json &ast)
{
  lower_bytes_arrays(ast);
  mark_compares_read_as_arrays(ast["body"]);
  alias_string_views_in_scopes(ast, ast["body"]);
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
  const auto narrow = numpy_scalar_returns_.find(current_function_name());
  if (value.is_object() && narrow == numpy_scalar_returns_.end())
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
  if (attr == "name" && is_dtype_attribute(base))
    if (const auto described = numpy_array_dtype(base["value"]))
      return string_builder_->build_string_literal(described->name());
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

// A str is a 1-D char array, and is no numpy array.
static bool is_retyped_numpy_array(const typet &type)
{
  if (
    !type.is_array() || type_utils::is_char_storage(type.subtype()) ||
    type_utils::is_bytes_array(type))
    return false;
  const typet leaf = type_utils::array_scalar_type(type);
  return is_complex_type(leaf) || leaf.is_floatbv() || leaf.is_signedbv() ||
         leaf.is_unsignedbv() || leaf.is_bool();
}

void python_converter::bind_numpy_scalar_name(
  const nlohmann::json &target,
  symbolt *symbol,
  exprt &lhs,
  const exprt &rhs)
{
  if (!symbol || target.value("_type", "") != "Name")
    return;
  // The annotator reads `1j * a` or `-a` as a scalar from its literal operand,
  // and has no type for an operation over a view of rank above 1.
  if (
    !symbol->is_parameter && !lhs.type().is_array() &&
    is_retyped_numpy_array(rhs.type()))
  {
    lhs.type() = rhs.type();
    symbol->set_type(rhs.type());
    return;
  }
  if (!type_utils::is_numpy_scalar_type(rhs.type()) || lhs.type() == rhs.type())
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

// `v += x` writes the result to the storage v sees, element by element.
bool python_converter::store_through_numpy_view(
  const nlohmann::json &target,
  const exprt &lhs,
  const exprt &rhs,
  const locationt &location,
  codet &target_block)
{
  const auto *info = target.value("_type", "") == "Name"
                       ? find_numpy_pointer_view_info(target.value("id", ""))
                       : nullptr;
  if (!info || !lhs.type().is_pointer() || !rhs.type().is_array())
    return false;
  const auto extents = type_handler_.get_array_type_shape(rhs.type());
  if (!std::equal(
        extents.begin(), extents.end(), info->shape.begin(), info->shape.end()))
    throw std::runtime_error(
      "ValueError: the in-place result does not have the numpy view's shape");
  symbolt &result =
    create_tmp_symbol(target, "$view_store$", rhs.type(), exprt());
  code_declt decl(symbol_expr(result));
  decl.operands().push_back(rhs);
  decl.location() = location;
  target_block.copy_to_operands(decl);
  std::vector<std::pair<nlohmann::json, exprt>> cells = {
    {target, symbol_expr(result)}};
  for (const std::size_t extent : extents)
  {
    std::vector<std::pair<nlohmann::json, exprt>> next;
    for (const auto &[node, value] : cells)
      for (std::size_t i = 0; i < extent; ++i)
      {
        nlohmann::json item = {
          {"_type", "Subscript"},
          {"value", node},
          {"slice", {{"_type", "Constant"}, {"value", i}}},
          {"ctx", {{"_type", "Store"}}}};
        copy_location_fields_from_decl(target, item);
        copy_location_fields_from_decl(target, item["slice"]);
        next.emplace_back(
          item,
          index_exprt(
            value,
            from_integer(i, size_type()),
            to_array_type(value.type()).subtype()));
      }
    cells = std::move(next);
  }
  for (const auto &[node, value] : cells)
  {
    const exprt cell = get_expr(node);
    code_assignt store(
      cell,
      cell.type() == value.type() ? value : typecast_exprt(value, cell.type()));
    store.location() = location;
    target_block.copy_to_operands(store);
  }
  return true;
}

// `-a` and `~a` over narrow integers and floats, wrapped at the operand's
// width. Nil for any other operand.
exprt python_converter::numpy_narrow_unary(
  const std::string &op,
  const exprt &operand)
{
  const typet leaf = type_utils::array_scalar_type(operand.type());
  if (type_utils::is_numpy_complex_array(operand.type()))
    return numpy_elementwise(
      op, operand, from_integer(0, long_long_int_type()), operand.type());
  const auto dtype = numpy_dtype_of_element(leaf);
  if (
    (op != "USub" && op != "Invert") || !dtype ||
    (op == "Invert" && !dtype->is_integer()) ||
    !type_utils::is_numpy_scalar_type(leaf))
    return nil_exprt();
  const exprt minus_one = from_integer(-1, long_long_int_type());
  return numpy_elementwise(
    op == "USub" ? "Mult" : "BitXor", operand, minus_one, operand.type());
}

// `a == "x"` read for its elements: one bool an element, by string comparison.
std::optional<exprt> python_converter::numpy_string_elementwise_compare(
  const nlohmann::json &element,
  const exprt &lhs,
  const exprt &rhs)
{
  const bool left_array = type_utils::is_numpy_string_array(lhs.type());
  const bool right_array = type_utils::is_numpy_string_array(rhs.type());
  if (
    (!left_array && !right_array) || json_utils::ast_contains_call(element) ||
    (left_array && right_array && lhs.type() != rhs.type()))
    return std::nullopt;
  const typet &array = left_array ? lhs.type() : rhs.type();
  BigInt length;
  if (
    array.subtype().subtype().is_array() ||
    to_integer(to_array_type(array).size(), length))
    return std::nullopt;
  const auto item = [&](const nlohmann::json &node, long long i) {
    nlohmann::json read = {
      {"_type", "Subscript"},
      {"value", node},
      {"slice", {{"_type", "Constant"}, {"value", i}}}};
    copy_location_fields_from_decl(element, read);
    copy_location_fields_from_decl(element, read["slice"]);
    return read;
  };
  std::vector<exprt> cells;
  for (long long i = 0; i < length.to_int64(); ++i)
  {
    nlohmann::json compared = element;
    compared.erase("numpy_array_result");
    if (left_array)
      compared["left"] = item(element["left"], i);
    if (right_array)
      compared["comparators"][0] = item(element["comparators"][0], i);
    cells.push_back(get_expr(compared));
  }
  if (cells.empty())
    return std::nullopt;
  const bool was_fresh_copy = numpy_result_is_fresh_copy_;
  const exprt result =
    materialize_numpy_elements(element, {cells.size()}, cells, bool_type());
  numpy_result_is_fresh_copy_ = was_fresh_copy;
  return result;
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
