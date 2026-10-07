#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_converter.h>

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

// NumPy views that leave direct local use (ADR-NP-003): a call to a simple
// function is folded into the caller, where the view's metadata is known;
// any other call is an escape, after which only the view's shape may be
// queried.

namespace
{
const char *const view_use_error =
  "TypeError: passing a numpy view to a function that is not a single return "
  "expression over its parameters is not supported";
const char *const returned_view_error =
  "TypeError: returning a copied numpy view is not supported";
const char *const multi_path_return_error =
  "TypeError: returning numpy views from more than one path is not "
  "supported";
const char *const container_index_error =
  "TypeError: a container holding numpy views supports only literal index "
  "access";
const char *const container_mutation_error =
  "TypeError: mutating a container that holds numpy views is not supported";
const char *const container_nested_error =
  "TypeError: storing a numpy view in a nested container is not supported";
const char *const container_branch_error =
  "TypeError: binding a container of numpy views inside a branch or loop is "
  "not supported";
const char *const unknown_function_error =
  "TypeError: passing a copied numpy view to an unknown function is not "
  "supported";
const char *const constructor_error =
  "TypeError: passing a numpy view to a class constructor is not supported";
const char *const callable_error =
  "TypeError: passing a numpy view to a callable that is not a plain "
  "function is not supported";
const char *const method_error =
  "TypeError: passing a numpy view to a method is not supported";
const char *const container_store_error =
  "TypeError: storing a numpy view in a container by append or item "
  "assignment (a list comprehension included) is not supported";
const char *const comprehension_error =
  "TypeError: numpy views in a comprehension or generator are not supported";
const char *const unfolded_call_error =
  "TypeError: this call over numpy views could not be folded into its "
  "caller";
const char *const inline_runtime_slice_error =
  "TypeError: an N-D numpy slice with run-time bounds must be assigned to a "
  "name";
const char *const view_rebind_error =
  "TypeError: rebinding a numpy view name to another view is not supported";
const char *const view_path_conflict_error =
  "TypeError: binding a numpy view name to views with a different shape, "
  "strides or storage on different paths is not supported";
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

bool contains_name(const nlohmann::json &node)
{
  bool found = false;
  for_each_name(node, [&](const nlohmann::json &) { found = true; });
  return found;
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

// The indices of a subscript chain, outermost axis first, and its root.
const nlohmann::json *
subscript_chain(const nlohmann::json &node, std::vector<nlohmann::json> &out)
{
  const nlohmann::json *current = &node;
  std::vector<std::vector<nlohmann::json>> groups;
  while (current->is_object() && current->value("_type", "") == "Subscript")
  {
    const nlohmann::json &slice = (*current)["slice"];
    if (slice.value("_type", "") == "Tuple")
      groups.emplace_back(slice["elts"].begin(), slice["elts"].end());
    else
      groups.push_back({slice});
    current = &(*current)["value"];
  }
  for (auto group = groups.rbegin(); group != groups.rend(); ++group)
    out.insert(out.end(), group->begin(), group->end());
  return current;
}

bool is_slice(const nlohmann::json &index)
{
  return index.value("_type", "") == "Slice";
}

bool contains_call(const nlohmann::json &node)
{
  if (!node.is_object() && !node.is_array())
    return false;
  if (node.is_object() && node.value("_type", "") == "Call")
    return true;
  for (const auto &child : node)
    if (contains_call(child))
      return true;
  return false;
}

using module_aliasest = std::unordered_map<std::string, std::string>;

// Calls that may have a side effect: anything but a function of an imported
// module (`np.transpose(a)`), which only computes a value.
std::size_t count_effectful_calls(
  const nlohmann::json &node,
  const module_aliasest &modules)
{
  if (!node.is_object() && !node.is_array())
    return 0;
  std::size_t count = 0;
  if (node.is_object() && node.value("_type", "") == "Call")
  {
    const nlohmann::json &func = node["func"];
    const bool module_function =
      func.value("_type", "") == "Attribute" && is_name(func["value"]) &&
      modules.count(func["value"]["id"].get<std::string>()) != 0;
    count = module_function ? 0 : 1;
  }
  for (const auto &child : node)
    count += count_effectful_calls(child, modules);
  return count;
}

// Replaces each local alias of a straight-line body by the expression it
// was bound to.
struct alias_inliner
{
  std::map<std::string, nlohmann::json> aliases;
  std::map<std::string, std::size_t> uses;

  nlohmann::json inlined(const nlohmann::json &node)
  {
    if (is_name(node))
    {
      const auto alias = aliases.find(node["id"].get<std::string>());
      if (alias == aliases.end())
        return node;
      ++uses[alias->first];
      return alias->second;
    }
    nlohmann::json copy = node;
    if (node.is_object() || node.is_array())
      for (auto it = copy.begin(); it != copy.end(); ++it)
        *it = inlined(*it);
    return copy;
  }

  // Inlining must run every call exactly once and in its original order: an
  // alias used twice would repeat its calls, an unused one would drop them
  // (and the exception they may raise), and an alias holding a call with side
  // effects would move it past any other such call in `result`.
  bool changes_evaluation(
    const nlohmann::json &result,
    const module_aliasest &modules) const
  {
    bool moves_an_effect = false;
    for (const auto &[name, value] : aliases)
    {
      const auto used = uses.find(name);
      if (!contains_call(value))
        continue;
      if (used == uses.end() || used->second != 1)
        return true;
      moves_an_effect =
        moves_an_effect || count_effectful_calls(value, modules);
    }
    return moves_an_effect && count_effectful_calls(result, modules) > 1;
  }
};

// The bare name a local alias statement binds, or nullptr. The annotation
// pre-pass turns an inferred `x = e` into `x: T = e`.
const nlohmann::json *alias_target(const nlohmann::json &stmt)
{
  const std::string type = stmt.value("_type", "");
  const nlohmann::json *target = nullptr;
  if (type == "Assign" && stmt["targets"].size() == 1)
    target = &stmt["targets"][0];
  else if (type == "AnnAssign" && !stmt["value"].is_null())
    target = &stmt["target"];
  return target && is_name(*target) ? target : nullptr;
}

// The return expression of a straight-line body -- local aliases followed by
// one `return` -- with each alias inlined; nullopt for any other body, or
// when inlining would change which calls run or their order.
std::optional<nlohmann::json> straight_line_return_value(
  const nlohmann::json &body,
  const module_aliasest &modules)
{
  alias_inliner inliner;
  for (const auto &stmt : body)
  {
    if (
      stmt.value("_type", "") == "Return" && stmt.contains("value") &&
      !stmt["value"].is_null())
    {
      nlohmann::json value = inliner.inlined(stmt["value"]);
      if (inliner.changes_evaluation(value, modules))
        return std::nullopt;
      return value;
    }
    // A name bound twice would lose what it was first bound to.
    const nlohmann::json *target = alias_target(stmt);
    if (!target || inliner.aliases.count((*target)["id"].get<std::string>()))
      return std::nullopt;
    inliner.aliases[(*target)["id"].get<std::string>()] =
      inliner.inlined(stmt["value"]);
  }
  return std::nullopt;
}

// Every `return` value under `node`, through nested blocks but not nested
// definitions.
void collect_return_values(
  const nlohmann::json &node,
  std::vector<nlohmann::json> &values)
{
  if (!node.is_object() && !node.is_array())
    return;
  const std::string type = node.is_object() ? node.value("_type", "") : "";
  if (type == "FunctionDef" || type == "ClassDef" || type == "Lambda")
    return;
  if (type == "Return" && node.contains("value") && !node["value"].is_null())
    values.push_back(node["value"]);
  for (const auto &child : node)
    collect_return_values(child, values);
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

// The value a call to a simple function returns, over the callee's own
// parameter names.
std::optional<nlohmann::json>
python_converter::simple_call_return_value(const nlohmann::json &call) const
{
  if (std::optional<nlohmann::json> value = select_return_value_for_call(call))
    return value;

  const nlohmann::json func_node = json_utils::try_find_function(
    (*ast_json)["body"], call["func"]["id"].get<std::string>());
  if (
    func_node.empty() ||
    call.value("keywords", nlohmann::json::array()).size() ||
    func_node["args"]["args"].size() != call["args"].size())
    return std::nullopt;
  for (const auto &arg : call["args"])
    if (!is_name(arg) && arg.value("_type", "") != "Constant")
      return std::nullopt;
  return straight_line_return_value(func_node["body"], imported_modules);
}

std::optional<nlohmann::json>
python_converter::fold_numpy_view_call(const nlohmann::json &call)
{
  if (!is_named_call(call))
    return std::nullopt;
  const std::string func_name = call["func"]["id"].get<std::string>();
  if (is_value_builtin(func_name))
    return std::nullopt;

  const std::optional<nlohmann::json> value = simple_call_return_value(call);
  if (!value)
    return std::nullopt;
  const nlohmann::json func_node =
    json_utils::try_find_function((*ast_json)["body"], func_name);
  // A recursive function would fold forever.
  if (
    references_name(*value, func_name) ||
    !is_closed_return_expression(*value, func_node["args"]["args"]))
    return std::nullopt;

  // Fold only where a view is involved: one passed in, or one returned (a
  // subscript that reads a single element returns no view).
  nlohmann::json folded = substitute_call_arguments(*value, call);
  // A subscript of a numpy call (`np.array(...)[0]`, a local array inlined)
  // is a view of that fresh array.
  std::vector<nlohmann::json> indices;
  const nlohmann::json *root = subscript_chain(folded, indices);
  const bool returns_view = indices.empty() || is_name(*root)
                              ? is_numpy_view_value(folded)
                              : is_numpy_module_call(*root);
  if (!contains_tracked_numpy_view_name(call["args"]) && !returns_view)
    return std::nullopt;
  return folded;
}

// `alias = view`: the alias is the same pointer, so it carries the same
// shape, strides, source and read-only state.
bool python_converter::bind_numpy_pointer_view_alias(
  const exprt &lhs,
  const std::string &lhs_id,
  const std::string &rhs_id)
{
  const auto view = numpy_pointer_view_info_.find(rhs_id);
  if (view == numpy_pointer_view_info_.end())
    return false;
  const numpy_scalar_pointer_view_infot info = view->second;
  clear_numpy_view_copy(lhs);
  clear_numpy_array_storage_aliases_for(lhs_id);
  numpy_param_shapes_.erase(lhs_id);
  numpy_pointer_view_info_[lhs_id] = info;
  if (const symbolt *symbol = symbol_table_.find_symbol(lhs_id))
    numpy_pointer_view_info_[symbol->id.as_string()] = info;
  numpy_array_symbols_.insert(lhs_id);
  return true;
}

void python_converter::get_folded_var_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  reject_numpy_view_container_store(ast_node);
  if (try_bind_numpy_view_container(ast_node, target_block))
    return;
  std::optional<nlohmann::json> folded =
    ast_node.contains("value") ? fold_numpy_view_call(ast_node["value"])
                               : std::nullopt;
  if (!folded)
    return get_var_assign(ast_node, target_block);
  nlohmann::json folded_assign = ast_node;
  folded_assign["value"] = std::move(*folded);
  get_var_assign(folded_assign, target_block);
}

void python_converter::reject_or_defer_numpy_view_return(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  const nlohmann::json func_node = json_utils::find_function_by_path(
    *ast_json, json_utils::split_function_path(current_func_name_));
  if (
    func_node.empty() ||
    !straight_line_return_value(func_node["body"], imported_modules))
    throw std::runtime_error(returned_view_error);

  // A folded call never runs this body and any other call is rejected where
  // it is converted; a path that reaches the body anyway fails here.
  numpy_view_return_functions_.insert(func_node["name"].get<std::string>());
  code_assertt unsupported(gen_boolean(false));
  unsupported.location() = get_location_from_decl(ast_node);
  unsupported.location().user_provided(true);
  unsupported.location().comment(
    "returning a numpy view outside a folded call is not supported");
  target_block.copy_to_operands(unsupported);
}

// A call that was not folded although its callee returns a view from one of
// several paths: the caller cannot tell which metadata the result carries.
void python_converter::reject_multi_path_numpy_view_return(
  const nlohmann::json &call)
{
  const nlohmann::json func_node = json_utils::try_find_function(
    (*ast_json)["body"], call["func"]["id"].get<std::string>());
  std::vector<nlohmann::json> values;
  if (!func_node.empty())
    collect_return_values(func_node["body"], values);
  if (values.size() < 2)
    return;
  for (const nlohmann::json &value : values)
  {
    if (is_numpy_view_value(substitute_call_arguments(value, call)))
      throw std::runtime_error(multi_path_return_error);
  }
}

// A non-folded call to a function whose body is only sound through a fold.
void python_converter::reject_unfoldable_numpy_view_call(
  const nlohmann::json &call)
{
  const std::string func_name = call["func"]["id"].get<std::string>();
  if (numpy_view_return_functions_.count(func_name) != 0)
    throw std::runtime_error(returned_view_error);
  if (numpy_fold_only_functions_.count(func_name) != 0)
    throw std::runtime_error(unfolded_call_error);
  reject_multi_path_numpy_view_return(call);
}

// Only a user function that never touches the parameter can receive a view
// without being folded: a builtin, an undefined name, or a body that uses
// the parameter would read the view through the wrong model.
void python_converter::reject_numpy_view_argument_use(
  const nlohmann::json &call)
{
  const nlohmann::json func_node = json_utils::try_find_function(
    (*ast_json)["body"], call["func"]["id"].get<std::string>());
  reject_numpy_view_callee(call);
  if (func_node.empty())
    throw std::runtime_error(unknown_function_error);
  const nlohmann::json &params = func_node["args"]["args"];
  for (std::size_t i = 0; i < call["args"].size(); ++i)
    if (
      contains_tracked_numpy_view_name(call["args"][i]) &&
      (i >= params.size() ||
       references_name(func_node["body"], params[i].value("arg", ""))))
      throw std::runtime_error(view_use_error);
}

void python_converter::track_numpy_view_call_escape(const nlohmann::json &call)
{
  if (!is_named_call(call))
    return;
  reject_unfoldable_numpy_view_call(call);
  if (
    !contains_tracked_numpy_view_name(call["args"]) ||
    is_value_builtin(call["func"]["id"].get<std::string>()) ||
    fold_numpy_view_call(call))
    return;
  reject_numpy_view_argument_use(call);

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

namespace
{
bool is_container_literal(const nlohmann::json &node)
{
  static const std::set<std::string> kinds = {"List", "Tuple", "Dict", "Set"};
  return node.is_object() && kinds.count(node.value("_type", "")) != 0;
}

// The literal a container is indexed with, as a lookup key; nullopt for a
// run-time index.
std::optional<std::string> literal_container_key(const nlohmann::json &slice)
{
  if (
    slice.is_object() && slice.value("_type", "") == "Constant" &&
    slice.contains("value") &&
    (slice["value"].is_number_integer() || slice["value"].is_string()))
    return slice["value"].dump();
  if (
    slice.is_object() && slice.value("_type", "") == "UnaryOp" &&
    slice["op"].value("_type", "") == "USub" &&
    slice["operand"].value("_type", "") == "Constant" &&
    slice["operand"]["value"].is_number_integer())
    return std::to_string(-slice["operand"]["value"].get<long long>());
  return std::nullopt;
}

bool is_container_mutator(const std::string &method)
{
  static const std::set<std::string> mutators = {
    "append",
    "extend",
    "insert",
    "remove",
    "pop",
    "clear",
    "sort",
    "reverse",
    "update",
    "setdefault",
    "popitem"};
  return mutators.count(method) != 0;
}

bool is_store_context(const nlohmann::json &node)
{
  const std::string ctx =
    node.contains("ctx") ? node["ctx"].value("_type", "") : "";
  return ctx == "Store" || ctx == "Del";
}

nlohmann::json named_like(const nlohmann::json &node, const std::string &name)
{
  nlohmann::json result = node;
  for (const char *key : {"value", "slice", "attr", "targets", "target"})
    result.erase(key);
  result["_type"] = "Name";
  result["id"] = name;
  result["ctx"] = {{"_type", "Load"}};
  return result;
}
} // namespace

const python_converter::numpy_view_containert *
python_converter::find_numpy_view_container(const nlohmann::json &node) const
{
  if (!is_name(node))
    return nullptr;
  const auto it = numpy_view_containers_.find(
    current_func_name_ + "@" + node["id"].get<std::string>());
  return it == numpy_view_containers_.end() ? nullptr : &it->second;
}

// The hidden view variable `container[slice]` denotes.
std::string python_converter::numpy_view_container_element(
  const numpy_view_containert &container,
  const nlohmann::json &slice) const
{
  const std::optional<std::string> key = literal_container_key(slice);
  if (!key)
    throw std::runtime_error(container_index_error);
  if (container.is_dict)
  {
    const auto element = container.keys.find(*key);
    if (element == container.keys.end())
      throw std::runtime_error("KeyError: " + *key);
    return container.elements[element->second];
  }
  if (
    slice.value("_type", "") != "UnaryOp" &&
    !slice["value"].is_number_integer())
    throw std::runtime_error(container_index_error);
  const long long size = static_cast<long long>(container.elements.size());
  long long index = std::stoll(*key);
  if (index < 0)
    index += size;
  if (index < 0 || index >= size)
    throw std::runtime_error("IndexError: list index out of range");
  return container.elements[static_cast<std::size_t>(index)];
}

// The replacement for a node that uses a view container directly: the
// element's variable for a literal-index read, an error for anything else;
// nullopt when the node is not such a use.
std::optional<nlohmann::json>
python_converter::resolve_numpy_view_container_use(
  const nlohmann::json &node) const
{
  const std::string type = node.value("_type", "");
  if (type == "Name")
  {
    if (find_numpy_view_container(node) && !is_store_context(node))
      throw std::runtime_error(container_index_error);
    return std::nullopt;
  }
  if (
    (type != "Subscript" && type != "Attribute") ||
    !find_numpy_view_container(node["value"]))
    return std::nullopt;
  if (type == "Attribute")
    throw std::runtime_error(
      is_container_mutator(node.value("attr", "")) ? container_mutation_error
                                                   : container_index_error);
  if (is_store_context(node))
    throw std::runtime_error(container_mutation_error);
  return named_like(
    node,
    numpy_view_container_element(
      *find_numpy_view_container(node["value"]), node["slice"]));
}

nlohmann::json python_converter::rewrite_numpy_view_container_reads(
  const nlohmann::json &node) const
{
  if (!node.is_object() && !node.is_array())
    return node;
  if (node.is_object())
  {
    const std::string type = node.value("_type", "");
    if (type == "FunctionDef" || type == "ClassDef")
      return node;
    if (
      std::optional<nlohmann::json> element =
        resolve_numpy_view_container_use(node))
      return *element;
  }

  nlohmann::json result = node;
  for (auto it = result.begin(); it != result.end(); ++it)
    if (!node.is_object() || !is_nested_block_key(it.key()))
      *it = rewrite_numpy_view_container_reads(*it);
  return result;
}

const nlohmann::json &python_converter::resolve_numpy_view_containers(
  const nlohmann::json &statement,
  nlohmann::json &rewritten) const
{
  if (numpy_view_containers_.empty())
    return statement;
  // `alias = box` is a binding, handled where the assignment is converted.
  const nlohmann::json *target = alias_target(statement);
  if (target && find_numpy_view_container(statement["value"]))
    return statement;
  rewritten = rewrite_numpy_view_container_reads(statement);
  // The annotation pre-pass typed `x: T = box[0]` from the container, not
  // from the view the element really is.
  if (
    target && statement.value("_type", "") == "AnnAssign" &&
    rewritten["value"] != statement["value"])
  {
    rewritten["_type"] = "Assign";
    rewritten["targets"] = nlohmann::json::array({rewritten["target"]});
    rewritten.erase("target");
    rewritten.erase("annotation");
  }
  return rewritten;
}

bool python_converter::try_bind_numpy_view_container(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  const nlohmann::json *target = alias_target(ast_node);
  if (!target)
    return false;
  const std::string key =
    current_func_name_ + "@" + (*target)["id"].get<std::string>();
  const nlohmann::json &value = ast_node["value"];

  if (const numpy_view_containert *aliased = find_numpy_view_container(value))
  {
    numpy_view_containers_[key] = numpy_view_containert(*aliased);
    return true;
  }
  numpy_view_containers_.erase(key);
  if (!is_container_literal(value) || value["_type"] == "Set")
    return false;

  numpy_view_containert container;
  container.is_dict = value["_type"] == "Dict";
  const nlohmann::json &elements =
    container.is_dict ? value["values"] : value["elts"];
  if (!std::any_of(
        elements.begin(), elements.end(), [this](const auto &element) {
          return is_numpy_view_value(element);
        }))
    return false;
  if (block_nesting_ != function_body_depth_ + 1)
    throw std::runtime_error(container_branch_error);
  const std::string prefix = (*target)["id"].get<std::string>() + "$view$" +
                             std::to_string(numpy_view_container_count_++) +
                             "$";
  for (std::size_t i = 0; i < elements.size(); ++i)
  {
    if (is_container_literal(elements[i]))
      throw std::runtime_error(container_nested_error);
    if (container.is_dict)
    {
      const std::optional<std::string> element_key =
        literal_container_key(value["keys"][i]);
      if (!element_key)
        throw std::runtime_error(container_index_error);
      container.keys[*element_key] = i;
    }
    const std::string element_name = prefix + std::to_string(i);
    nlohmann::json element_target = named_like(*target, element_name);
    element_target["ctx"] = {{"_type", "Store"}};
    nlohmann::json binding = ast_node;
    binding.erase("target");
    binding.erase("annotation");
    binding["_type"] = "Assign";
    binding["targets"] = nlohmann::json::array({element_target});
    binding["value"] = elements[i];
    get_folded_var_assign(binding, target_block);
    container.elements.push_back(element_name);
  }
  numpy_view_containers_[key] = std::move(container);
  return true;
}

namespace
{
bool is_comprehension(const nlohmann::json &node)
{
  static const std::set<std::string> kinds = {
    "ListComp", "SetComp", "DictComp", "GeneratorExp"};
  return node.is_object() && kinds.count(node.value("_type", "")) != 0;
}

bool is_literal_bound(const nlohmann::json &slice, const char *key)
{
  return !slice.contains(key) || slice[key].is_null() ||
         python_frontend::literal_int_value(slice[key]).has_value();
}

bool has_runtime_bound(const nlohmann::json &index)
{
  return is_slice(index) &&
         !(is_literal_bound(index, "lower") &&
           is_literal_bound(index, "upper") && is_literal_bound(index, "step"));
}
} // namespace

// True when `node` subscripts a numpy array or view and yields a view: it
// slices an axis, or indexes fewer axes than the source has.
bool python_converter::numpy_subscript_yields_view(
  const nlohmann::json &node) const
{
  std::vector<nlohmann::json> indices;
  const nlohmann::json *root = subscript_chain(node, indices);
  if (indices.empty() || !is_name(*root))
    return false;
  const std::string id =
    resolve_name_symbol_id((*root)["id"].get<std::string>());
  if (id.empty())
    return false;
  if (std::any_of(indices.begin(), indices.end(), is_slice))
    return numpy_array_symbols_.count(id) != 0 ||
           numpy_pointer_view_info_.count(id) != 0 ||
           is_tracked_numpy_view_id(id);

  // Indices only: a view is left when fewer axes are indexed than exist.
  std::optional<std::size_t> rank;
  const auto view = numpy_pointer_view_info_.find(id);
  if (view != numpy_pointer_view_info_.end())
    rank = view->second.shape.empty() ? 1 : view->second.shape.size();
  else if (numpy_array_symbols_.count(id) != 0)
  {
    const std::optional<std::vector<std::size_t>> shape =
      get_numpy_nditer_logical_shape(id);
    if (shape)
      rank = shape->size();
  }
  return rank ? indices.size() < *rank : is_tracked_numpy_view_id(id);
}

// True when `node` evaluates to a numpy view: a name bound to one, or a
// subscript that yields one. A scalar read out of a view is not a view.
bool python_converter::holds_numpy_view(const nlohmann::json &node) const
{
  if (!node.is_object() && !node.is_array())
    return false;
  if (is_name(node))
  {
    const std::string id =
      resolve_name_symbol_id(node["id"].get<std::string>());
    const symbolt *symbol =
      id.empty() ? nullptr : symbol_table_.find_symbol(id);
    if (!symbol)
      return false;
    const namespacet ns(symbol_table_);
    const typet type = ns.follow(symbol->get_type());
    return (type.is_pointer() || type.is_array()) &&
           (numpy_pointer_view_info_.count(id) != 0 ||
            is_tracked_numpy_view_id(id));
  }
  // A subscript is judged as a whole: `a[0][0]` of a 2-D array is a scalar
  // although its inner `a[0]` is a view.
  if (node.is_object() && node.value("_type", "") == "Subscript")
    return numpy_subscript_yields_view(node);
  for (const auto &child : node)
    if (holds_numpy_view(child))
      return true;
  return false;
}

// `np.<function>(...)` through an imported numpy alias.
bool python_converter::is_numpy_module_call(const nlohmann::json &node) const
{
  return node.is_object() && node.value("_type", "") == "Call" &&
         node["func"].value("_type", "") == "Attribute" &&
         is_name(node["func"]["value"]) &&
         is_imported_numpy_module_alias(
           *ast_json, node["func"]["value"]["id"].get<std::string>());
}

// True when `node` is a numpy view: one held by name or produced by a
// subscript, or a view function (`.T`, `broadcast_to`, ...) over a numpy
// array. An ordinary Python subscript or attribute is not.
bool python_converter::is_numpy_view_value(const nlohmann::json &node) const
{
  if (holds_numpy_view(node))
    return true;
  if (!is_numpy_view_copy_expr(node) || node.value("_type", "") == "Subscript")
    return false;
  const std::string root = root_name_from_numpy_view_copy_expr(node);
  const std::string id =
    root.empty() ? std::string() : resolve_name_symbol_id(root);
  return !id.empty() && (numpy_array_symbols_.count(id) != 0 ||
                         numpy_pointer_view_info_.count(id) != 0);
}

// A comprehension or generator builds its elements one by one into a plain
// container, so a numpy view placed in it would lose its aliasing.
void python_converter::reject_numpy_view_comprehension(
  const nlohmann::json &node) const
{
  if (!node.is_object() && !node.is_array())
    return;
  if (is_comprehension(node))
  {
    if (holds_numpy_view(node.contains("elt") ? node["elt"] : node["value"]))
      throw std::runtime_error(comprehension_error);
  }
  for (auto it = node.begin(); it != node.end(); ++it)
    if (!node.is_object() || !is_nested_block_key(it.key()))
      reject_numpy_view_comprehension(it.value());
}

void python_converter::check_numpy_view_statement(
  const nlohmann::json &statement) const
{
  if (numpy_array_symbols_.empty() || !statement.is_object())
    return;
  const std::string type = statement.value("_type", "");
  if (type == "FunctionDef" || type == "ClassDef")
    return;
  reject_numpy_view_comprehension(statement);
  reject_escaped_numpy_view_use(statement);
}

// A method call that passes a numpy view: allowed for a module function or
// a method of a numpy array, which read the view themselves.
void python_converter::reject_numpy_view_method_call(
  const nlohmann::json &func) const
{
  const std::string receiver = root_name_from_subscript(func["value"]);
  const std::string id =
    receiver.empty() ? std::string() : resolve_name_symbol_id(receiver);
  if (
    receiver.empty() || imported_modules.count(receiver) != 0 ||
    numpy_array_symbols_.count(id) != 0 ||
    numpy_pointer_view_info_.count(id) != 0)
    return;
  throw std::runtime_error(
    is_container_mutator(func.value("attr", "")) ? container_store_error
                                                 : method_error);
}

// A call that passes a numpy view where the callee is not a plain function:
// a method, a class constructor, or a name bound to some other callable.
void python_converter::reject_numpy_view_callee(const nlohmann::json &call)
{
  if (
    !call.is_object() || !call.contains("func") || !call.contains("args") ||
    !holds_numpy_view(call["args"]))
    return;
  const nlohmann::json &func = call["func"];
  if (func.value("_type", "") == "Attribute")
    return reject_numpy_view_method_call(func);
  if (!is_name(func))
    return;
  const std::string name = func["id"].get<std::string>();
  if (!json_utils::find_class((*ast_json)["body"], name).empty())
    throw std::runtime_error(constructor_error);
  const nlohmann::json decl =
    json_utils::find_var_decl(name, current_func_name_, *ast_json);
  const bool is_lambda = decl.is_object() && decl.contains("value") &&
                         decl["value"].is_object() &&
                         decl["value"].value("_type", "") == "Lambda";
  if (
    is_lambda ||
    (json_utils::try_find_function((*ast_json)["body"], name).empty() &&
     !decl.empty()))
    throw std::runtime_error(callable_error);
}

// An N-D slice with run-time bounds passed straight to a call has no name
// to carry its shape.
void python_converter::reject_inline_runtime_numpy_slice(
  const nlohmann::json &call) const
{
  if (!call.is_object() || !call.contains("args") || !call["args"].is_array())
    return;
  for (const nlohmann::json &arg : call["args"])
  {
    std::vector<nlohmann::json> indices;
    const nlohmann::json *root = subscript_chain(arg, indices);
    if (
      !std::any_of(indices.begin(), indices.end(), has_runtime_bound) ||
      !is_name(*root))
      continue;
    const std::string id =
      resolve_name_symbol_id((*root)["id"].get<std::string>());
    const std::optional<std::vector<std::size_t>> shape =
      id.empty() || numpy_array_symbols_.count(id) == 0
        ? std::nullopt
        : get_numpy_nditer_logical_shape(id);
    if (shape && shape->size() > 1)
      throw std::runtime_error(inline_runtime_slice_error);
  }
}

// `items[0] = view` on an ordinary container would store a copy.
void python_converter::reject_numpy_view_container_store(
  const nlohmann::json &ast_node)
{
  if (!ast_node.contains("targets") || !ast_node.contains("value"))
    return;
  if (!holds_numpy_view(ast_node["value"]))
    return;
  for (const nlohmann::json &target : ast_node["targets"])
  {
    if (target.value("_type", "") != "Subscript")
      continue;
    const std::string root = root_name_from_subscript(target);
    const std::string id =
      root.empty() ? std::string() : resolve_name_symbol_id(root);
    if (
      !id.empty() && numpy_array_symbols_.count(id) == 0 &&
      numpy_pointer_view_info_.count(id) == 0)
      throw std::runtime_error(container_store_error);
  }
}

namespace
{
// Calls to `name` under `node`, each with the function it appears in.
void collect_calls_to(
  const nlohmann::json &node,
  const std::string &name,
  const std::string &enclosing,
  std::vector<std::pair<const nlohmann::json *, std::string>> &out)
{
  if (!node.is_object() && !node.is_array())
    return;
  std::string scope = enclosing;
  if (node.is_object() && node.value("_type", "") == "FunctionDef")
    scope = node.value("name", "");
  if (is_named_call(node) && node["func"]["id"] == name)
    out.emplace_back(&node, scope);
  for (const auto &child : node)
    collect_calls_to(child, name, scope, out);
}
} // namespace

// Whether `value`, as written, denotes a numpy view: a slice, a partial
// index or a view function over a name that is bound to a numpy call.
bool python_converter::is_numpy_view_syntax(
  const nlohmann::json &value,
  const std::string &scope,
  std::size_t depth) const
{
  if (!value.is_object() || depth > 4)
    return false;
  const std::string type = value.value("_type", "");
  if (type == "Name")
  {
    const nlohmann::json decl = json_utils::find_var_decl(
      value["id"].get<std::string>(), scope, *ast_json);
    return decl.is_object() && decl.contains("value") &&
           is_numpy_view_syntax(decl["value"], scope, depth + 1);
  }
  if (type == "Attribute" && value.value("attr", "") == "T")
    return is_numpy_storage_syntax(value["value"], scope, depth + 1);
  // `np.transpose(a)` or `a.transpose()`: the array is the first argument
  // or the receiver.
  if (type == "Call" && is_numpy_view_copy_call_node(value))
    return is_numpy_storage_syntax(
      is_numpy_module_call(value) && !value["args"].empty()
        ? value["args"][0]
        : value["func"]["value"],
      scope,
      depth + 1);
  if (type != "Subscript")
    return false;

  std::vector<nlohmann::json> indices;
  const nlohmann::json *root = subscript_chain(value, indices);
  return std::any_of(indices.begin(), indices.end(), is_slice) &&
         is_numpy_storage_syntax(*root, scope, depth + 1);
}

// Whether `value` names numpy storage: a numpy call, or a view of one.
bool python_converter::is_numpy_storage_syntax(
  const nlohmann::json &value,
  const std::string &scope,
  std::size_t depth) const
{
  if (!is_name(value) || depth > 4)
    return false;
  const nlohmann::json decl =
    json_utils::find_var_decl(value["id"].get<std::string>(), scope, *ast_json);
  if (
    !decl.is_object() || !decl.contains("value") || !decl["value"].is_object())
    return false;
  const nlohmann::json &bound = decl["value"];
  // Only a constructor with literal arguments: a run-time shape is rejected
  // where the parameter is typed, which a skipped definition would bypass.
  const bool numpy_call =
    is_numpy_module_call(bound) && !contains_name(bound["args"]);
  return numpy_call || is_numpy_view_syntax(bound, scope, depth + 1);
}

// A simple module-level function every call of which passes a numpy view:
// each call is folded, so its body is never needed -- and, read on its own,
// its untyped parameter would be rejected by the numpy consumers it calls.
bool python_converter::is_fold_only_numpy_function(
  const nlohmann::json &function_node) const
{
  const std::optional<nlohmann::json> returned =
    straight_line_return_value(function_node["body"], imported_modules);
  if (!current_func_name_.empty() || !returned)
    return false;
  // A function that returns a view of its argument folds for any array.
  std::vector<nlohmann::json> indices;
  subscript_chain(*returned, indices);
  const bool returns_view =
    (returned->value("_type", "") == "Attribute" &&
     returned->value("attr", "") == "T") ||
    is_numpy_view_copy_call_node(*returned) ||
    std::any_of(indices.begin(), indices.end(), is_slice);
  const std::string name = function_node.value("name", "");
  std::vector<std::pair<const nlohmann::json *, std::string>> calls;
  collect_calls_to((*ast_json)["body"], name, "", calls);
  if (calls.empty())
    return false;
  for (const auto &[call, scope] : calls)
  {
    const nlohmann::json &args = (*call)["args"];
    if (
      scope == name ||
      std::none_of(args.begin(), args.end(), [&](const nlohmann::json &arg) {
        return is_name(arg) &&
               (is_numpy_view_syntax(arg, scope, 0) ||
                (returns_view && is_numpy_storage_syntax(arg, scope, 0)));
      }))
      return false;
  }
  return true;
}

void python_converter::get_unfolded_function_definition(
  const nlohmann::json &function_node)
{
  if (is_fold_only_numpy_function(function_node))
  {
    numpy_fold_only_functions_.insert(function_node["name"].get<std::string>());
    return;
  }
  get_function_definition(function_node);
}

// A view name bound again to a view the existing entry does not describe:
// on a conditional path the two would have to be merged, which one entry
// cannot represent.
void python_converter::reject_numpy_view_rebind() const
{
  throw std::runtime_error(
    block_nesting_ == function_body_depth_ + 1 ? view_rebind_error
                                               : view_path_conflict_error);
}

bool python_converter::accepts_numpy_view_binding(
  const std::string &lhs_id,
  const numpy_scalar_pointer_view_infot &candidate,
  const typet &view_ptr_type,
  const exprt &source) const
{
  const auto bound = numpy_pointer_view_info_.find(lhs_id);
  if (bound == numpy_pointer_view_info_.end())
    return true;
  const numpy_scalar_pointer_view_infot &current = bound->second;
  const symbolt *symbol = symbol_table_.find_symbol(lhs_id);
  const bool same_layout =
    symbol && symbol->get_type() == view_ptr_type && !current.is_symbolic() &&
    current.length == candidate.length && current.stride == candidate.stride &&
    current.readonly == candidate.readonly &&
    current.shape == candidate.shape && current.strides == candidate.strides;
  const bool same_storage =
    source.is_symbol() &&
    numpy_view_storage_root(source.identifier().as_string()) ==
      numpy_view_storage_root(lhs_id);
  if (same_layout && same_storage)
    return true;
  if (block_nesting_ != function_body_depth_ + 1)
    throw std::runtime_error(view_path_conflict_error);
  return false;
}
