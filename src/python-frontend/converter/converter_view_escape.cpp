#include <python-frontend/json_utils.h>
#include <python-frontend/python_converter.h>

#include <algorithm>
#include <functional>
#include <map>
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

  // An alias used twice would evaluate its calls twice.
  bool duplicates_a_call() const
  {
    for (const auto &[name, count] : uses)
      if (count > 1 && contains_call(aliases.at(name)))
        return true;
    return false;
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
// when inlining would evaluate a call more than once.
std::optional<nlohmann::json>
straight_line_return_value(const nlohmann::json &body)
{
  alias_inliner inliner;
  for (const auto &stmt : body)
  {
    if (
      stmt.value("_type", "") == "Return" && stmt.contains("value") &&
      !stmt["value"].is_null())
    {
      nlohmann::json value = inliner.inlined(stmt["value"]);
      if (inliner.duplicates_a_call())
        return std::nullopt;
      return value;
    }
    const nlohmann::json *target = alias_target(stmt);
    if (!target)
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
  return straight_line_return_value(func_node["body"]);
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

  // Fold only where a view is involved: one passed in, or one returned.
  nlohmann::json folded = substitute_call_arguments(*value, call);
  if (
    !contains_tracked_numpy_view_name(call["args"]) &&
    !is_numpy_view_copy_expr(folded))
    return std::nullopt;
  return folded;
}

void python_converter::get_folded_var_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
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
  if (func_node.empty() || !straight_line_return_value(func_node["body"]))
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
    const nlohmann::json returned = substitute_call_arguments(value, call);
    if (
      is_numpy_view_copy_expr(returned) ||
      contains_tracked_numpy_view_name(returned))
      throw std::runtime_error(multi_path_return_error);
  }
}

void python_converter::track_numpy_view_call_escape(const nlohmann::json &call)
{
  if (!is_named_call(call))
    return;
  const std::string func_name = call["func"]["id"].get<std::string>();
  if (numpy_view_return_functions_.count(func_name) != 0)
    throw std::runtime_error(returned_view_error);
  reject_multi_path_numpy_view_return(call);
  if (!contains_tracked_numpy_view_name(call["args"]))
    return;
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
  const bool holds_view =
    contains_tracked_numpy_view_name(value) ||
    std::any_of(elements.begin(), elements.end(), [this](const auto &element) {
      return is_numpy_view_copy_expr(element);
    });
  if (!holds_view)
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
