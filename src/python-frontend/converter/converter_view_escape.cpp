#include <python-frontend/json_utils.h>
#include <python-frontend/python_converter.h>

#include <functional>
#include <set>
#include <stdexcept>

// NumPy views that leave direct local use (ADR-NP-003): a call to a simple
// function is folded into the caller, where the view's metadata is known;
// any other call is an escape, after which only the view's shape may be
// queried.

namespace
{
const char *const view_use_error =
  "TypeError: passing a numpy view to a function that is not a single return "
  "expression over its parameters is not supported";
const char *const escaped_read_error =
  "TypeError: reading a numpy view after it escaped to an unknown call is "
  "not supported";
const char *const escaped_write_error =
  "TypeError: writing to a numpy view after it escaped to an unknown call is "
  "not supported";

bool is_name(const nlohmann::json &node)
{
  return node.is_object() && node.value("_type", "") == "Name" &&
         node.contains("id");
}

// Calls `visit` on every Name node under `node`.
void for_each_name(
  const nlohmann::json &node,
  const std::function<void(const nlohmann::json &)> &visit)
{
  if (!node.is_object() && !node.is_array())
    return;
  if (is_name(node))
    visit(node);
  for (const auto &child : node)
    for_each_name(child, visit);
}

bool references_name(const nlohmann::json &node, const std::string &name)
{
  bool found = false;
  for_each_name(node, [&](const nlohmann::json &id) {
    found = found || id["id"].get<std::string>() == name;
  });
  return found;
}

// Calls whose only use of an array argument is to read it.
bool is_value_builtin(const std::string &name)
{
  static const std::set<std::string> builtins = {"len", "bool", "int", "float"};
  return builtins.count(name) != 0;
}

bool is_named_call(const nlohmann::json &node)
{
  return node.is_object() && node.value("_type", "") == "Call" &&
         node.contains("func") && is_name(node["func"]) &&
         node.contains("args") && node["args"].is_array();
}

// `v.shape`, `v.ndim`, `v.size` and `len(v)` over a bare name: metadata
// only, never the storage's content.
bool is_shape_query(const nlohmann::json &node)
{
  if (!node.is_object())
    return false;
  if (node.value("_type", "") == "Attribute")
  {
    static const std::set<std::string> attrs = {"shape", "ndim", "size"};
    return attrs.count(node.value("attr", "")) != 0 && is_name(node["value"]);
  }
  return is_named_call(node) && node["func"]["id"] == "len" &&
         node["args"].size() == 1 && is_name(node["args"][0]);
}

// The nested statement lists of a compound statement: each gets its own
// check when its block is converted.
bool is_nested_block_key(const std::string &key)
{
  static const std::set<std::string> keys = {
    "body", "orelse", "finalbody", "handlers"};
  return keys.count(key) != 0;
}
} // namespace

std::string
python_converter::numpy_view_storage_root(const std::string &id) const
{
  std::string current = id;
  std::set<std::string> seen;
  while (seen.insert(current).second)
  {
    const auto pointer_view = numpy_pointer_view_info_.find(current);
    if (
      pointer_view != numpy_pointer_view_info_.end() &&
      !pointer_view->second.source_id.empty())
    {
      current = pointer_view->second.source_id;
      continue;
    }
    const auto copied_view = numpy_view_copy_sources_.find(current);
    if (copied_view == numpy_view_copy_sources_.end())
      break;
    current = copied_view->second;
  }
  return resolve_numpy_array_storage_alias_id(current);
}

bool python_converter::is_numpy_storage_escaped(
  const nlohmann::json &name) const
{
  if (numpy_escaped_storage_.empty())
    return false;
  const std::string id = resolve_name_symbol_id(name["id"].get<std::string>());
  return !id.empty() &&
         numpy_escaped_storage_.count(numpy_view_storage_root(id)) != 0;
}

// True when every name the return expression reads is a parameter, a module
// alias or a callee: substituting it into the caller then captures nothing.
bool python_converter::is_closed_return_expression(
  const nlohmann::json &value,
  const nlohmann::json &params) const
{
  std::set<std::string> callees;
  std::function<void(const nlohmann::json &)> collect_callees =
    [&](const nlohmann::json &node) {
      if (!node.is_object() && !node.is_array())
        return;
      if (is_named_call(node))
        callees.insert(node["func"]["id"].get<std::string>());
      for (const auto &child : node)
        collect_callees(child);
    };
  collect_callees(value);

  bool closed = true;
  for_each_name(value, [&](const nlohmann::json &id) {
    const std::string name = id["id"].get<std::string>();
    bool is_param = false;
    for (const auto &param : params)
      is_param = is_param || param.value("arg", "") == name;
    closed = closed && (is_param || imported_modules.count(name) != 0 ||
                        callees.count(name) != 0);
  });
  return closed;
}

std::optional<nlohmann::json>
python_converter::fold_numpy_view_call(const nlohmann::json &call)
{
  if (!is_named_call(call) || !contains_tracked_numpy_view_name(call["args"]))
    return std::nullopt;
  const std::string func_name = call["func"]["id"].get<std::string>();
  if (is_value_builtin(func_name))
    return std::nullopt;

  std::optional<nlohmann::json> value = select_return_value_for_call(call);
  if (!value)
    return std::nullopt;
  const nlohmann::json func_node =
    json_utils::try_find_function((*ast_json)["body"], func_name);
  // A recursive function would fold forever.
  if (
    references_name(*value, func_name) ||
    !is_closed_return_expression(*value, func_node["args"]["args"]))
    return std::nullopt;
  return substitute_call_arguments(*value, call);
}

void python_converter::track_numpy_view_call_escape(const nlohmann::json &call)
{
  if (!is_named_call(call) || !contains_tracked_numpy_view_name(call["args"]))
    return;
  const std::string func_name = call["func"]["id"].get<std::string>();
  if (is_value_builtin(func_name) || fold_numpy_view_call(call))
    return;

  // The callee's parameter carries no view metadata, so a body that touches
  // it would read the view through the wrong model.
  const nlohmann::json func_node =
    json_utils::try_find_function((*ast_json)["body"], func_name);
  if (!func_node.empty())
  {
    const nlohmann::json &params = func_node["args"]["args"];
    for (std::size_t i = 0; i < call["args"].size(); ++i)
      if (
        contains_tracked_numpy_view_name(call["args"][i]) &&
        (i >= params.size() ||
         references_name(func_node["body"], params[i].value("arg", ""))))
        throw std::runtime_error(view_use_error);
  }

  // Not assumed read-only: the callee may keep or mutate what it received.
  for_each_name(call["args"], [&](const nlohmann::json &name) {
    const std::string id =
      resolve_name_symbol_id(name["id"].get<std::string>());
    if (
      !id.empty() &&
      (numpy_array_symbols_.count(id) != 0 ||
       numpy_pointer_view_info_.count(id) != 0 || is_tracked_numpy_view_id(id)))
      numpy_escaped_storage_.insert(numpy_view_storage_root(id));
  });
}

void python_converter::reject_escaped_numpy_view_read(
  const nlohmann::json &node) const
{
  if ((!node.is_object() && !node.is_array()) || is_shape_query(node))
    return;
  if (is_name(node) && is_numpy_storage_escaped(node))
    throw std::runtime_error(escaped_read_error);
  for (auto it = node.begin(); it != node.end(); ++it)
    if (!node.is_object() || !is_nested_block_key(it.key()))
      reject_escaped_numpy_view_read(it.value());
}

// An assignment target is written, not read: a bare name is a rebind, a
// subscript stores into the storage of its root name.
void python_converter::reject_escaped_numpy_view_write(
  const nlohmann::json &target) const
{
  if (is_name(target))
    return;
  const std::string root = root_name_from_subscript(target);
  if (!root.empty() && is_numpy_storage_escaped({{"id", root}}))
    throw std::runtime_error(escaped_write_error);
  reject_escaped_numpy_view_read(target);
}

void python_converter::reject_escaped_numpy_view_use(
  const nlohmann::json &statement) const
{
  if (numpy_escaped_storage_.empty() || !statement.is_object())
    return;
  const std::string type = statement.value("_type", "");
  if (type == "FunctionDef" || type == "ClassDef")
    return;

  if (statement.contains("targets") && statement["targets"].is_array())
    for (const nlohmann::json &target : statement["targets"])
      reject_escaped_numpy_view_write(target);
  if (statement.contains("target"))
  {
    // `v += 1` writes through the name itself.
    if (type == "AugAssign" && is_numpy_storage_escaped(statement["target"]))
      throw std::runtime_error(escaped_write_error);
    reject_escaped_numpy_view_write(statement["target"]);
  }

  for (auto it = statement.begin(); it != statement.end(); ++it)
    if (
      it.key() != "targets" && it.key() != "target" &&
      !is_nested_block_key(it.key()))
      reject_escaped_numpy_view_read(it.value());
}
