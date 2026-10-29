#include <python-frontend/converter/value_kinds.h>

#include <algorithm>
#include <functional>

namespace
{
constexpr unsigned max_depth = 12;

std::string node_type(const nlohmann::json &n)
{
  return n.is_object() ? n.value("_type", "") : "";
}

bool is_function(const nlohmann::json &n)
{
  return node_type(n) == "FunctionDef" || node_type(n) == "Lambda";
}

/// Visits every node under `root`, not entering a nested function, lambda or
/// class, whose body is another scope.
void walk_scope(
  const nlohmann::json &root,
  const std::function<void(const nlohmann::json &)> &visit)
{
  for (const auto &child : root)
  {
    if (!child.is_structured())
      continue;
    const std::string t = node_type(child);
    if (t == "FunctionDef" || t == "Lambda" || t == "ClassDef")
    {
      visit(child);
      continue;
    }
    visit(child);
    walk_scope(child, visit);
  }
}

/// The positional parameters of a function or lambda.
const nlohmann::json &params(const nlohmann::json &fn)
{
  return fn["args"]["args"];
}

const std::string &builtin_kind(const std::string &name)
{
  static const std::string none;
  static const std::string str = "str", num = "num";
  static const std::set<std::string> strs = {
    "str", "repr", "format", "chr", "hex", "oct", "bin"};
  static const std::set<std::string> nums = {
    "int", "float", "len", "abs", "round", "ord", "bool", "isinstance"};
  return strs.count(name) ? str : nums.count(name) ? num : none;
}
/// The kind of a literal, an f-string or a comparison, else null.
const char *literal_kind(const nlohmann::json &expr)
{
  const std::string t = node_type(expr);
  if (t == "JoinedStr")
    return "str";
  if (t == "Compare")
    return "num";
  if (t != "Constant")
    return nullptr;
  const nlohmann::json &v = expr["value"];
  if (v.is_string())
    return "str";
  if (v.is_null())
    return "none";
  return v.is_number() || v.is_boolean() ? "num" : nullptr;
}

/// The kind of `a op b` from its operands' kinds: str only for two strs, a
/// number for numbers, else unknown.
const char *binop_kind(
  const std::set<std::string> &left,
  const std::set<std::string> &right)
{
  if (left == std::set<std::string>{"str"} && right == left)
    return "str";
  if (left.count("str") || right.count("str"))
    return nullptr;
  return left.count("num") || right.count("num") ? "num" : nullptr;
}

/// The value statement `n` binds to the plain name `name`, else null.
const nlohmann::json *
assigned_value(const nlohmann::json &n, const std::string &name)
{
  const auto names = [&](const nlohmann::json &target) {
    return node_type(target) == "Name" && target.value("id", "") == name;
  };
  if (node_type(n) == "Assign")
    for (const auto &target : n["targets"])
      if (names(target))
        return &n["value"];
  if (
    node_type(n) == "AnnAssign" && names(n["target"]) && n.contains("value") &&
    !n["value"].is_null())
    return &n["value"];
  return nullptr;
}
} // namespace

value_kinds::value_kinds(const nlohmann::json &module) : module_(module)
{
  index(module_, &module_);
}

void value_kinds::index(const nlohmann::json &node, const nlohmann::json *scope)
{
  for (const auto &child : node)
  {
    if (!child.is_structured())
      continue;
    scope_[&child] = scope;
    if (node_type(child) == "Call")
      calls_.push_back(&child);
    if (node_type(child) == "FunctionDef")
      defs_.push_back(&child);
    index(child, is_function(child) ? &child : scope);
  }
}

bool value_kinds::may_be_non_number(
  const nlohmann::json &expr,
  const nlohmann::json &scope)
{
  std::set<std::string> kinds;
  kinds_of(expr, scope, kinds, 0);
  return kinds.size() > 1 || kinds.count("str") || kinds.count("none");
}

const nlohmann::json &
value_kinds::scope_of(const std::string &function_name) const
{
  const nlohmann::json *found = nullptr;
  for (const nlohmann::json *def : defs_)
    if (def->value("name", "") == function_name)
    {
      if (found)
        return module_;
      found = def;
    }
  return found ? *found : module_;
}

const nlohmann::json *value_kinds::top_level_def(const std::string &name) const
{
  for (const auto &stmt : module_["body"])
    if (node_type(stmt) == "FunctionDef" && stmt.value("name", "") == name)
      return &stmt;
  return nullptr;
}

const nlohmann::json &
value_kinds::enclosing_scope(const nlohmann::json &node) const
{
  auto it = scope_.find(&node);
  return it == scope_.end() ? module_ : *it->second;
}

void value_kinds::parameter_values(
  const nlohmann::json &def,
  const std::string &name,
  std::vector<scoped> &out,
  unsigned depth) const
{
  const nlohmann::json &ps = params(def);
  const auto it =
    std::find_if(ps.begin(), ps.end(), [&](const nlohmann::json &p) {
      return p.value("arg", "") == name;
    });
  if (it == ps.end())
    return;
  const size_t index = it - ps.begin();
  const nlohmann::json &defaults = def["args"]["defaults"];
  const size_t first_default = ps.size() - defaults.size();
  for (const nlohmann::json *call : callers(def, depth))
  {
    const nlohmann::json &n = *call;
    const nlohmann::json *scope = &enclosing_scope(n);
    if (index < n["args"].size())
    {
      if (node_type(n["args"][index]) != "Starred")
        out.emplace_back(&n["args"][index], scope);
      continue;
    }
    const auto kw =
      std::find_if(n["keywords"].begin(), n["keywords"].end(), [&](auto &k) {
        return k.value("arg", "") == name;
      });
    if (kw != n["keywords"].end())
      out.emplace_back(&(*kw)["value"], scope);
    else if (index >= first_default)
      out.emplace_back(&defaults[index - first_default], &module_);
  }
}

/// The calls that may reach `def`, by its name or through a value holding it.
/// While they are being found, `def` has none, so a cycle through it ends.
const std::vector<const nlohmann::json *> &
value_kinds::callers(const nlohmann::json &def, unsigned depth) const
{
  static const std::vector<const nlohmann::json *> none;
  if (auto it = callers_.find(&def); it != callers_.end())
    return it->second;
  if (!finding_callers_.insert(&def).second)
    return none;
  const std::string fname = def.value("name", "");
  std::vector<const nlohmann::json *> found;
  for (const nlohmann::json *call : calls_)
  {
    const nlohmann::json &func = (*call)["func"];
    bool reaches = node_type(func) == "Name" && func.value("id", "") == fname;
    if (!reaches)
    {
      const auto fns = callees(func, enclosing_scope(*call), depth + 1);
      reaches = std::any_of(fns.begin(), fns.end(), [&](const scoped &fn) {
        return fn.first == &def;
      });
    }
    if (reaches)
      found.push_back(call);
  }
  finding_callers_.erase(&def);
  return callers_[&def] = std::move(found);
}

std::vector<value_kinds::scoped> value_kinds::name_values(
  const std::string &name,
  const nlohmann::json &scope,
  unsigned depth) const
{
  std::vector<scoped> out;
  if (depth > max_depth)
    return out;
  if (is_function(scope))
  {
    const nlohmann::json &ps = params(scope);
    if (std::any_of(ps.begin(), ps.end(), [&](const nlohmann::json &p) {
          return p.value("arg", "") == name;
        }))
    {
      if (node_type(scope) == "FunctionDef")
        parameter_values(scope, name, out, depth);
      return out;
    }
  }

  // A name a function does not bind is read from the scope enclosing it.
  if (node_type(scope) == "Lambda")
    return name_values(name, enclosing_scope(scope), depth + 1);
  walk_scope(scope["body"], [&](const nlohmann::json &n) {
    if (node_type(n) == "FunctionDef" && n.value("name", "") == name)
      out.emplace_back(&n, &n);
    else if (const nlohmann::json *value = assigned_value(n, name))
      out.emplace_back(value, &scope);
  });
  if (out.empty() && &scope != &module_)
    return name_values(name, enclosing_scope(scope), depth + 1);
  return out;
}

std::vector<value_kinds::scoped> value_kinds::callees(
  const nlohmann::json &callee,
  const nlohmann::json &scope,
  unsigned depth) const
{
  std::vector<scoped> out;
  if (depth > max_depth)
    return out;
  if (is_function(callee))
  {
    out.emplace_back(&callee, &callee);
    return out;
  }
  if (node_type(callee) == "Call")
    return returned_functions(callee, scope, depth + 1);
  if (node_type(callee) != "Name")
    return out;
  for (const auto &[value, value_scope] :
       name_values(callee.value("id", ""), scope, depth + 1))
  {
    const auto more = callees(*value, *value_scope, depth + 1);
    out.insert(out.end(), more.begin(), more.end());
  }
  return out;
}

/// The functions `call` may return, read from what its callees return.
std::vector<value_kinds::scoped> value_kinds::returned_functions(
  const nlohmann::json &call,
  const nlohmann::json &scope,
  unsigned depth) const
{
  std::vector<scoped> out;
  for (const auto &[fn, fn_scope] : callees(call["func"], scope, depth + 1))
  {
    std::vector<const nlohmann::json *> values;
    if (node_type(*fn) == "Lambda")
      values.push_back(&(*fn)["body"]);
    else
      walk_scope((*fn)["body"], [&](const nlohmann::json &n) {
        if (
          node_type(n) == "Return" && n.contains("value") &&
          !n["value"].is_null())
          values.push_back(&n["value"]);
      });
    for (const nlohmann::json *value : values)
    {
      const auto more = callees(*value, *fn, depth + 1);
      out.insert(out.end(), more.begin(), more.end());
    }
  }
  return out;
}

void value_kinds::call_kinds(
  const nlohmann::json &call,
  const nlohmann::json &scope,
  std::set<std::string> &out,
  unsigned depth)
{
  const nlohmann::json &func = call["func"];
  if (node_type(func) == "Name" && !top_level_def(func.value("id", "")))
  {
    const std::string kind = builtin_kind(func.value("id", ""));
    if (!kind.empty())
    {
      out.insert(kind);
      return;
    }
  }

  for (const auto &[fn, fn_scope] : callees(func, scope, depth))
  {
    if (node_type(*fn) == "Lambda")
    {
      kinds_of((*fn)["body"], *fn, out, depth + 1);
      continue;
    }
    bool returns = false, yields = false;
    walk_scope((*fn)["body"], [&](const nlohmann::json &n) {
      const std::string t = node_type(n);
      yields = yields || t == "Yield" || t == "YieldFrom";
      if (t != "Return")
        return;
      returns = true;
      if (n.contains("value") && !n["value"].is_null())
        kinds_of(n["value"], *fn_scope, out, depth + 1);
      else
        out.insert("none");
    });
    if (!returns && !yields)
      out.insert("none");
  }
}

void value_kinds::kinds_of(
  const nlohmann::json &expr,
  const nlohmann::json &scope,
  std::set<std::string> &out,
  unsigned depth)
{
  if (depth > max_depth)
    return;
  const std::string t = node_type(expr);
  if (const char *kind = literal_kind(expr))
    out.insert(kind);
  else if (t == "Name")
  {
    for (const auto &[value, value_scope] :
         name_values(expr.value("id", ""), scope, depth + 1))
      if (!is_function(*value))
        kinds_of(*value, *value_scope, out, depth + 1);
  }
  else if (t == "IfExp")
  {
    kinds_of(expr["body"], scope, out, depth + 1);
    kinds_of(expr["orelse"], scope, out, depth + 1);
  }
  else if (t == "BinOp")
  {
    std::set<std::string> left, right;
    kinds_of(expr["left"], scope, left, depth + 1);
    kinds_of(expr["right"], scope, right, depth + 1);
    if (const char *kind = binop_kind(left, right))
      out.insert(kind);
  }
  else if (t == "UnaryOp")
    kinds_of(expr["operand"], scope, out, depth + 1);
  else if (t == "Call")
    call_kinds(expr, scope, out, depth);
}
