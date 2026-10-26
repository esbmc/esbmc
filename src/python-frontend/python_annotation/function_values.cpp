#include <python-frontend/json_utils.h>
#include <python-frontend/python_annotation/function_values.h>

#include <algorithm>
#include <fstream>

using json = nlohmann::json;

namespace
{
/// The depth of name, call and parameter indirections followed.
constexpr unsigned max_depth = 16;

std::string kind(const json &node)
{
  return node.is_object() ? node.value("_type", "") : "";
}

bool is_load(const json &name)
{
  return name.contains("ctx") && kind(name["ctx"]) == "Load";
}

bool is_name(const json &node, const std::string &id)
{
  return kind(node) == "Name" && node.value("id", "") == id;
}

/// Whether a node of this kind opens a scope for its `body`.
bool is_scope(const std::string &k)
{
  return k == "FunctionDef" || k == "AsyncFunctionDef" || k == "Lambda" ||
         k == "ClassDef";
}

/// Whether a node of this kind opens a scope for all of its children.
bool is_comprehension(const std::string &k)
{
  return k == "ListComp" || k == "SetComp" || k == "DictComp" ||
         k == "GeneratorExp";
}

/// Visits every node under `node` with the scopes enclosing it, innermost
/// last.
template <class Visitor>
void walk(
  const json &node,
  std::vector<const json *> &chain,
  const Visitor &visit)
{
  if (node.is_array())
  {
    for (const json &child : node)
      walk(child, chain, visit);
    return;
  }
  if (!node.is_object())
    return;
  visit(node, chain);
  const std::string k = kind(node);
  for (auto it = node.begin(); it != node.end(); ++it)
  {
    const bool inner =
      is_comprehension(k) || (is_scope(k) && it.key() == "body");
    if (inner)
      chain.push_back(&node);
    walk(it.value(), chain, visit);
    if (inner)
      chain.pop_back();
  }
}

bool same_range(const json &a, const json &b)
{
  for (const char *key :
       {"lineno", "col_offset", "end_lineno", "end_col_offset"})
    if (!a.contains(key) || !b.contains(key) || a[key] != b[key])
      return false;
  return kind(a) == kind(b);
}

bool contains(const json &node, const json *target)
{
  if (&node == target)
    return true;
  if (!node.is_structured())
    return false;
  for (const json &child : node)
    if (contains(child, target))
      return true;
  return false;
}

/// Whether any string in `node` is `name`: an identifier, attribute or
/// import of it, or a string naming it.
bool mentions(const json &node, const std::string &name)
{
  if (node.is_string())
    return node.get_ref<const std::string &>() == name;
  if (!node.is_structured())
    return false;
  for (const json &child : node)
    if (mentions(child, name))
      return true;
  return false;
}

/// How many loads of `name` there are under `node`, and how many of them
/// are the `field` of a node of kind `parent`.
std::pair<std::size_t, std::size_t> count_loads(
  const json &node,
  const std::string &name,
  const std::string &parent,
  const std::string &field)
{
  std::pair<std::size_t, std::size_t> n{0, 0};
  if (node.is_object())
  {
    if (is_name(node, name) && is_load(node))
      ++n.first;
    if (
      kind(node) == parent && node.contains(field) &&
      is_name(node[field], name) && is_load(node[field]) &&
      (!node.contains("ctx") || is_load(node)))
      ++n.second;
  }
  if (node.is_structured())
    for (const json &child : node)
    {
      const auto sub = count_loads(child, name, parent, field);
      n.first += sub.first;
      n.second += sub.second;
    }
  return n;
}

bool every_load_is(
  const json &node,
  const std::string &name,
  const std::string &parent,
  const std::string &field)
{
  const auto n = count_loads(node, name, parent, field);
  return n.first == n.second;
}

/// Whether a `global` or `nonlocal` statement under `node` names `name`.
bool declares(const json &node, const std::string &name)
{
  if (node.is_object())
  {
    const std::string k = kind(node);
    if (
      (k == "Global" || k == "Nonlocal") && node.contains("names") &&
      std::count(node["names"].begin(), node["names"].end(), name))
      return true;
  }
  if (node.is_structured())
    for (const json &child : node)
      if (declares(child, name))
        return true;
  return false;
}

std::string bound_name(const json &alias)
{
  if (alias.contains("asname") && alias["asname"].is_string())
    return alias["asname"];
  const std::string name = alias.value("name", "");
  return name.substr(0, name.find('.'));
}

/// The bindings of a name in one scope.
struct bindingst
{
  std::vector<const json *> values;
  std::vector<const json *> defs;
  /// (ImportFrom, alias) pairs.
  std::vector<std::pair<const json *, const json *>> imports;
  bool global = false;
  /// Bound some other way, or rebound from an inner scope.
  bool other = false;

  bool any() const
  {
    return !values.empty() || !defs.empty() || !imports.empty() || other;
  }
};

void scan(const json &node, const std::string &name, bindingst &b);

void scan_definition(const json &node, const std::string &name, bindingst &b)
{
  if (node.value("name", "") == name)
  {
    if (kind(node) == "FunctionDef")
      b.defs.push_back(&node);
    else
      b.other = true;
  }
  for (const char *key : {"decorator_list", "bases", "keywords"})
    if (node.contains(key))
      scan(node[key], name, b);
  if (node.contains("args"))
    for (const char *key : {"defaults", "kw_defaults"})
      if (node["args"].contains(key))
        scan(node["args"][key], name, b);
  b.other |= declares(node["body"], name);
}

void scan_assign(const json &node, const std::string &name, bindingst &b)
{
  const json &value = node["value"];
  const auto target = [&](const json &t) {
    if (!is_name(t, name))
      scan(t, name, b);
    else if (!value.is_null())
      b.values.push_back(&value);
  };
  if (kind(node) == "AnnAssign")
    target(node["target"]);
  else
    for (const json &t : node["targets"])
      target(t);
  scan(value, name, b);
}

void scan_object(const json &node, const std::string &name, bindingst &b)
{
  const std::string k = kind(node);
  if (k == "Name")
    b.other |= node.value("id", "") == name && !is_load(node);
  else if (k == "Global")
    b.global |= declares(node, name);
  else if (k == "Nonlocal")
    b.other |= declares(node, name);
  else if (k == "ImportFrom" || k == "Import")
  {
    for (const json &alias : node["names"])
      if (bound_name(alias) == name)
      {
        if (k == "ImportFrom")
          b.imports.emplace_back(&node, &alias);
        else
          b.other = true;
      }
  }
  else
  {
    b.other |= (k == "ExceptHandler" || k == "MatchAs" || k == "MatchStar") &&
               node.value("name", json()) == name;
    for (const json &child : node)
      scan(child, name, b);
  }
}

/// Every binding of `name` in the scope whose statements are `node`.
void scan(const json &node, const std::string &name, bindingst &b)
{
  if (node.is_array())
  {
    for (const json &child : node)
      scan(child, name, b);
    return;
  }
  if (!node.is_object())
    return;
  const std::string k = kind(node);
  if (k == "FunctionDef" || k == "AsyncFunctionDef" || k == "ClassDef")
    scan_definition(node, name, b);
  else if (k == "Lambda")
    scan(node["args"], name, b);
  else if (k == "Assign" || k == "AnnAssign")
    scan_assign(node, name, b);
  else
    scan_object(node, name, b);
}

bool block_binds(const json &block, const std::string &name);

/// Whether running `stmt` to its end always binds `name`.
bool definitely_binds(const json &stmt, const std::string &name)
{
  const std::string k = kind(stmt);
  if (k == "Assign")
    return std::any_of(
      stmt["targets"].begin(), stmt["targets"].end(), [&](const json &t) {
        return is_name(t, name);
      });
  if (k == "AnnAssign")
    return is_name(stmt["target"], name) && !stmt["value"].is_null();
  if (k == "FunctionDef")
    return stmt.value("name", "") == name;
  if (k == "ImportFrom")
    return std::any_of(
      stmt["names"].begin(), stmt["names"].end(), [&](const json &alias) {
        return bound_name(alias) == name;
      });
  if (k == "If")
    return block_binds(stmt["body"], name) && block_binds(stmt["orelse"], name);
  return false;
}

bool block_binds(const json &block, const std::string &name)
{
  return std::any_of(block.begin(), block.end(), [&](const json &stmt) {
    return definitely_binds(stmt, name);
  });
}

/// Whether `name` is bound whenever `read`, a node in `block`, runs.
bool bound_before(const json &block, const std::string &name, const json *read)
{
  for (const json &stmt : block)
  {
    if (definitely_binds(stmt, name) && !contains(stmt, read))
      return true;
    if (!contains(stmt, read))
      continue;
    for (const char *key : {"body", "orelse", "finalbody"})
      if (
        stmt.contains(key) && stmt[key].is_array() && contains(stmt[key], read))
        return bound_before(stmt[key], name, read);
    if (stmt.contains("handlers"))
      for (const json &handler : stmt["handlers"])
        if (contains(handler["body"], read))
          return bound_before(handler["body"], name, read);
    return false;
  }
  return false;
}

/// The argument `call` passes for the parameter at `index`, named `name`, of
/// a function with arguments `args`: positional, keyword or default. Null
/// when there is none or it cannot be told.
const json *passed_argument(
  const json &call,
  const json &args,
  std::size_t index,
  const std::string &name)
{
  const json &positional = call["args"];
  const json &keywords = call["keywords"];
  const auto unknown = [](const json &arg) {
    return kind(arg) == "Starred" ||
           (kind(arg) == "keyword" && !arg["arg"].is_string());
  };
  if (
    std::any_of(positional.begin(), positional.end(), unknown) ||
    std::any_of(keywords.begin(), keywords.end(), unknown))
    return nullptr;
  if (index < positional.size())
    return &positional[index];
  for (const json &keyword : keywords)
    if (keyword["arg"] == name)
      return &keyword["value"];
  const json &defaults = args["defaults"];
  const std::size_t first_default = args["args"].size() - defaults.size();
  return index >= first_default ? &defaults[index - first_default] : nullptr;
}

/// The undecorated top-level function `name` of `module`, or null.
const json *top_level_def(const json &module, const std::string &name)
{
  for (const json &stmt : module["body"])
    if (
      kind(stmt) == "FunctionDef" && stmt.value("name", "") == name &&
      stmt["decorator_list"].empty())
      return &stmt;
  return nullptr;
}

/// The parameter a positional argument `index` of `call` binds, when the
/// callee is a top-level function of `module` that only ever calls it and
/// never rebinds it, as (function, parameter); else null.
std::optional<std::pair<const json *, std::string>>
called_parameter(const json &module, const json &call, std::size_t index)
{
  const json *def = top_level_def(module, call["func"].value("id", ""));
  if (!def || index >= (*def)["args"]["args"].size())
    return std::nullopt;
  const std::string param = (*def)["args"]["args"][index].value("arg", "");
  if (
    !every_load_is((*def)["body"], param, "Call", "func") ||
    json_utils::count_name_bindings((*def)["body"], param) != 0)
    return std::nullopt;
  return std::make_pair(def, param);
}

/// The parameters function `name` is passed to and then only called through,
/// when every load of `name` is either a call or such an argument.
std::optional<std::vector<std::pair<const json *, std::string>>>
forwarded_parameters(const json &module, const std::string &name)
{
  std::vector<std::pair<const json *, std::string>> out;
  bool ok = true;
  std::vector<const json *> chain{&module};
  walk(module["body"], chain, [&](const json &call, const auto &) {
    if (kind(call) != "Call" || kind(call["func"]) != "Name")
      return;
    for (std::size_t i = 0; i < call["args"].size(); ++i)
      if (is_name(call["args"][i], name))
      {
        const auto param = called_parameter(module, call, i);
        ok &= param.has_value();
        if (param)
          out.push_back(*param);
      }
  });
  const auto loads = count_loads(module, name, "Call", "func");
  if (!ok || loads.first != loads.second + out.size())
    return std::nullopt;
  return out;
}

bool is_parameter(const json &def, const std::string &name)
{
  const json &args = def["args"];
  for (const char *key : {"posonlyargs", "args", "kwonlyargs"})
    for (const json &arg : args[key])
      if (arg.value("arg", "") == name)
        return true;
  for (const char *key : {"vararg", "kwarg"})
    if (args[key].is_object() && args[key].value("arg", "") == name)
      return true;
  return false;
}

/// Whether `block` ends in a `return` or `raise` on every path.
bool always_returns(const json &block)
{
  if (!block.is_array() || block.empty())
    return false;
  const json &last = block.back();
  const std::string k = kind(last);
  if (k == "Return" || k == "Raise")
    return true;
  return k == "If" && always_returns(last["body"]) &&
         always_returns(last["orelse"]);
}

/// Every value `node` returns, not looking into nested scopes. False when
/// some `return` has no value or the function is a generator or coroutine.
bool collect_returns(const json &node, std::vector<const json *> &values)
{
  if (node.is_array())
    return std::all_of(node.begin(), node.end(), [&](const json &child) {
      return collect_returns(child, values);
    });
  if (!node.is_object())
    return true;
  const std::string k = kind(node);
  if (is_scope(k) || is_comprehension(k))
    return true;
  if (k == "Yield" || k == "YieldFrom" || k == "Await")
    return false;
  if (k == "Return")
  {
    if (node["value"].is_null())
      return false;
    values.push_back(&node["value"]);
  }
  for (const json &child : node)
    if (!collect_returns(child, values))
      return false;
  return true;
}
} // namespace

function_values::function_values(
  const json &module,
  const std::vector<const json *> &others)
  : module_(module), others_(others)
{
}

std::optional<std::vector<function_values::candidate>>
function_values::of(const json &expr)
{
  const json *found = nullptr;
  chaint found_chain;
  bool ambiguous = false;
  chaint chain{&module_};
  walk(module_["body"], chain, [&](const json &node, const chaint &scopes) {
    if (!same_range(node, expr))
      return;
    ambiguous |= found != nullptr;
    found = &node;
    found_chain = scopes;
  });
  std::vector<candidate> out;
  if (!found || ambiguous || !resolve(*found, found_chain, out, 0))
    return std::nullopt;
  if (out.empty())
    return std::nullopt;
  return out;
}

bool function_values::resolve(
  const json &expr,
  const chaint &chain,
  std::vector<candidate> &out,
  unsigned depth)
{
  if (depth > max_depth)
    return false;
  const std::string k = kind(expr);
  if (k == "IfExp")
    return resolve(expr["body"], chain, out, depth + 1) &&
           resolve(expr["orelse"], chain, out, depth + 1);

  std::vector<valuet> values;
  bool ok = false;
  if (k == "Name")
    ok = name_values(expr, chain, values, out);
  else if (k == "Subscript")
    ok = element_values(expr, chain, values);
  else if (k == "Call")
  {
    std::vector<candidate> callees;
    ok = resolve(expr["func"], chain, callees, depth + 1) &&
         returned_values(callees, values);
  }
  if (!ok)
    return false;
  for (const valuet &value : values)
    if (!resolve(*value.first, value.second, out, depth + 1))
      return false;
  return true;
}

bool function_values::name_values(
  const json &name,
  const chaint &chain,
  std::vector<valuet> &values,
  std::vector<candidate> &out)
{
  const std::string id = name.value("id", "");
  const auto is_function = [](const json *scope) {
    return kind(*scope) == "FunctionDef";
  };
  if (
    !is_load(name) || !std::all_of(chain.begin() + 1, chain.end(), is_function))
    return false;

  for (std::size_t i = chain.size() - 1; i > 0; --i)
  {
    bindingst b;
    scan((*chain[i])["body"], id, b);
    if (b.global)
      break;
    // A read from an inner function cannot tell whether the name is bound.
    if (is_parameter(*chain[i], id) || b.any())
      return i + 1 == chain.size() && local_values(name, chain, values);
  }
  return module_name_values(name, chain, values, out);
}

bool function_values::local_values(
  const json &name,
  const chaint &chain,
  std::vector<valuet> &values)
{
  const std::string id = name.value("id", "");
  const json &scope = *chain.back();
  bindingst b;
  scan(scope["body"], id, b);
  if (b.other || !b.defs.empty() || !b.imports.empty())
    return false;
  if (
    is_parameter(scope, id) ? !parameter_values(scope, chain, id, values)
                            : !bound_before(scope["body"], id, &name))
    return false;
  for (const json *value : b.values)
    values.emplace_back(value, chain);
  return true;
}

bool function_values::module_name_values(
  const json &name,
  const chaint &chain,
  std::vector<valuet> &values,
  std::vector<candidate> &out)
{
  const std::string id = name.value("id", "");
  const json &module = *chain.front();
  bindingst b;
  scan(module["body"], id, b);
  if (
    b.other || !b.any() ||
    (!b.values.empty() &&
     (&module != &module_ || mentioned_elsewhere(module, id))))
    return false;
  // A function reads a module-level function or import whenever it is
  // called, but a variable only once it is assigned.
  if (
    chain.size() > 1 ? !b.values.empty()
                     : !bound_before(module["body"], id, &name))
    return false;
  for (const json *def : b.defs)
    out.push_back({def, &module});
  for (const auto &[import_from, alias] : b.imports)
    if (!imported_function(module, *import_from, *alias, out))
      return false;
  for (const json *value : b.values)
    values.emplace_back(value, chaint{&module});
  return true;
}

bool function_values::parameter_values(
  const json &def,
  const chaint &chain,
  const std::string &name,
  std::vector<valuet> &values)
{
  const json &args = def["args"];
  const json &params = args["args"];
  const auto param =
    std::find_if(params.begin(), params.end(), [&](const json &p) {
      return p.value("arg", "") == name;
    });
  const json &module = *chain.front();
  if (
    chain.size() != 2 || param == params.end() || !args["posonlyargs"].empty())
    return false;
  const auto callers = callers_of(def, module);
  if (!callers)
    return false;
  bool ok = true;
  for (const auto &[scope, callee] : *callers)
    ok &= bound_arguments(
      module, scope, callee, args, param - params.begin(), name, values);
  return ok;
}

std::optional<std::vector<std::pair<const json *, std::string>>>
function_values::callers_of(const json &def, const json &module) const
{
  // The module calls the function by its name, and a parameter it is passed
  // to and only called through calls it under that parameter's name.
  const std::string fname = def.value("name", "");
  std::vector<std::pair<const json *, std::string>> out{{&module, fname}};
  if (only_called(def, module))
    return out;
  const auto forwarded = forwarded_parameters(module, fname);
  if (
    &module != &module_ || !def["decorator_list"].empty() ||
    json_utils::count_name_bindings(module, fname) != 1 ||
    mentioned_elsewhere(module, fname) || !forwarded)
    return std::nullopt;
  out.insert(out.end(), forwarded->begin(), forwarded->end());
  return out;
}

bool function_values::bound_arguments(
  const json &module,
  const json *scope,
  const std::string &callee,
  const json &args,
  std::size_t index,
  const std::string &name,
  std::vector<valuet> &values) const
{
  bool ok = true;
  chaint scopes{&module};
  if (scope != &module)
    scopes.push_back(scope);
  walk(
    (*scope)["body"], scopes, [&](const json &call, const chaint &call_chain) {
      if (kind(call) != "Call" || !is_name(call["func"], callee))
        return;
      const json *arg = passed_argument(call, args, index, name);
      ok &= arg != nullptr;
      // A default is evaluated where the function is defined.
      if (arg)
        values.emplace_back(
          arg, contains(args, arg) ? chaint{&module} : call_chain);
    });
  return ok;
}

bool function_values::only_called(const json &def, const json &module) const
{
  const std::string name = def.value("name", "");
  return &module == &module_ && def["decorator_list"].empty() &&
         json_utils::count_name_bindings(module, name) == 1 &&
         every_load_is(module, name, "Call", "func") &&
         !mentioned_elsewhere(module, name);
}

bool function_values::element_values(
  const json &subscript,
  const chaint &chain,
  std::vector<valuet> &values)
{
  const json &list = subscript["value"];
  const std::string id = list.value("id", "");
  const json &module = *chain.front();
  // The list is only ever indexed, so nothing adds to it.
  if (
    kind(list) != "Name" || &module != &module_ ||
    !every_load_is(module, id, "Subscript", "value") ||
    mentioned_elsewhere(module, id))
    return false;

  std::vector<valuet> lists;
  std::vector<candidate> functions;
  if (!name_values(list, chain, lists, functions) || !functions.empty())
    return false;
  for (const valuet &literal : lists)
  {
    const std::string k = kind(*literal.first);
    if (k != "List" && k != "Tuple")
      return false;
    for (const json &element : (*literal.first)["elts"])
      values.emplace_back(&element, literal.second);
  }
  return true;
}

bool function_values::returned_values(
  const std::vector<candidate> &callees,
  std::vector<valuet> &values)
{
  for (const candidate &callee : callees)
  {
    const json &def = *callee.def;
    std::vector<const json *> returned;
    if (
      !def["decorator_list"].empty() || !always_returns(def["body"]) ||
      !collect_returns(def["body"], returned))
      return false;
    for (const json *value : returned)
      values.emplace_back(value, chaint{callee.module, callee.def});
  }
  return true;
}

bool function_values::imported_function(
  const json &module,
  const json &import_from,
  const json &alias,
  std::vector<candidate> &out)
{
  if (
    !import_from["module"].is_string() || import_from.value("level", 0) != 0 ||
    !module.contains("ast_output_dir"))
    return false;
  const std::string path = module["ast_output_dir"].get<std::string>() + "/" +
                           json_utils::dotted_to_path(import_from["module"]) +
                           ".json";
  auto loaded = loaded_.find(path);
  if (loaded == loaded_.end())
  {
    std::ifstream file(path);
    json imported = json::parse(file, nullptr, false);
    if (imported.is_discarded() || !imported.contains("body"))
      return false;
    if (!imported.contains("ast_output_dir"))
      imported["ast_output_dir"] = module["ast_output_dir"];
    loaded = loaded_.emplace(path, std::move(imported)).first;
  }

  const json &imported = loaded->second;
  bindingst b;
  scan(imported["body"], alias.value("name", ""), b);
  if (b.defs.size() != 1 || !b.values.empty() || !b.imports.empty() || b.other)
    return false;
  out.push_back({b.defs.front(), &imported});
  return true;
}

bool function_values::mentioned_elsewhere(
  const json &module,
  const std::string &name) const
{
  if (&module != &module_ && mentions(module_, name))
    return true;
  for (const json *other : others_)
    if (other != &module && mentions(*other, name))
      return true;
  for (const auto &[path, other] : loaded_)
    if (&other != &module && mentions(other, name))
      return true;
  return false;
}
