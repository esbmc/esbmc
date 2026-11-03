#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Pure-AST literal-kind divergence detection, shared by dynamic_type_handler
// and the annotation pre-pass (which has no symbol table to check against).
namespace dynamic_type_detail
{
inline bool
field_is(const nlohmann::json &node, const char *field, const std::string &name)
{
  return node.contains(field) && node[field].is_string() &&
         node[field].get<std::string>() == name;
}

inline std::string context_of(const nlohmann::json &node)
{
  if (!node.contains("ctx") || !node["ctx"].is_object())
    return "";
  return node["ctx"].value("_type", "");
}

inline bool lists_name(const nlohmann::json &node, const std::string &name)
{
  if (!node.contains("names") || !node["names"].is_array())
    return false;
  for (const auto &listed : node["names"])
    if (listed.is_string() && listed.get<std::string>() == name)
      return true;
  return false;
}

// `node` binds `name` by assignment, parameter, def, class, import, pattern,
// `global`, `nonlocal`, `except`, `del` or any other target.
inline bool binds_name_at(const nlohmann::json &node, const std::string &name)
{
  const std::string type = node.value("_type", "");
  if (type == "Name")
    return field_is(node, "id", name) && context_of(node) != "Load";
  if (type == "arg")
    return field_is(node, "arg", name);
  return field_is(node, "name", name) || field_is(node, "asname", name) ||
         field_is(node, "rest", name) || lists_name(node, name);
}

inline size_t
count_bindings(const nlohmann::json &node, const std::string &name)
{
  if (!node.is_structured())
    return 0;
  size_t count = node.is_object() && binds_name_at(node, name);
  for (const auto &child : node)
    count += count_bindings(child, name);
  return count;
}

// The value `node` binds to `name` when it is `name = value`.
inline const nlohmann::json *
assigned_value_at(const nlohmann::json &node, const std::string &name)
{
  const std::string type = node.value("_type", "");
  if (
    type == "Assign" && node["targets"].size() == 1 &&
    node["targets"][0].value("id", "") == name)
    return &node["value"];
  if (
    type == "AnnAssign" && node["target"].value("id", "") == name &&
    node.contains("value") && !node["value"].is_null())
    return &node["value"];
  return nullptr;
}

inline const nlohmann::json *
assigned_value_in(const nlohmann::json &node, const std::string &name)
{
  if (!node.is_structured())
    return nullptr;
  if (node.is_object())
    if (const nlohmann::json *value = assigned_value_at(node, name))
      return value;
  for (const auto &child : node)
    if (const nlohmann::json *found = assigned_value_in(child, name))
      return found;
  return nullptr;
}

// The value `name = value` binds, when that is the only binding of `name`
// anywhere in the module.
inline const nlohmann::json *single_assigned_value(
  const nlohmann::json &module_body,
  const std::string &name)
{
  return count_bindings(module_body, name) == 1
           ? assigned_value_in(module_body, name)
           : nullptr;
}

// The name a call's `func` spells, `m.f` for an attribute of a name, and
// its last component; both "" for any other shape.
inline std::pair<std::string, std::string>
callee_names(const nlohmann::json &func)
{
  if (
    func.value("_type", "") != "Attribute" || !func.contains("value") ||
    func["value"].value("_type", "") != "Name")
    return {func.value("id", ""), func.value("id", "")};
  const std::string attr = func.value("attr", "");
  return {func["value"].value("id", "") + "." + attr, attr};
}

// The kind of a scalar type named `type`, "" for any other name.
inline std::string scalar_name_kind(const std::string &type)
{
  if (type == "str")
    return "str";
  if (type == "int" || type == "bool")
    return "num";
  return type == "float" ? "float" : "";
}

// The kind a nondet_<type> function named `leaf` returns, "" for any other.
inline std::string nondet_leaf_kind(const std::string &leaf)
{
  return leaf.rfind("nondet_", 0) == 0 ? scalar_name_kind(leaf.substr(7)) : "";
}

// The kind of a literal constant's value.
inline std::string constant_kind(const nlohmann::json &lit)
{
  if (lit.is_string())
    return "str";
  if (lit.is_number_integer() || lit.is_boolean())
    return "num";
  return lit.is_number_float() ? "float" : "";
}

// Classifies a literal constant, a nondet_<type> call, plain or qualified
// by a module as in `_sv_verifier.nondet_str()`, or a call to a function
// with a scalar return annotation (e.g. `mk()` where `def mk() -> str`);
// "" otherwise. `module_body` holds the FunctionDefs a call may name, a
// qualified call `m.mk()` naming one called `m.mk`. It may be left empty
// where a caller has no access to the full module (the annotation
// pre-pass).
inline std::string classify_literal_value_kind(
  const nlohmann::json &value,
  const nlohmann::json &module_body = nlohmann::json(),
  unsigned depth = 0);

// The kind a builtin conversion or measure returns, "" for any other name.
inline std::string builtin_call_kind(const std::string &callee)
{
  static const std::unordered_map<std::string, std::string> kinds = {
    {"str", "str"}, {"int", "num"}, {"len", "num"}, {"float", "float"}};
  const auto it = kinds.find(callee);
  return it == kinds.end() ? "" : it->second;
}

// The kind of `left op right` for integer arithmetic; "" for any other
// operands, whose result depends on the values.
inline std::string binop_kind(
  const std::string &op,
  const std::string &left,
  const std::string &right)
{
  static const std::unordered_set<std::string> integer_ops = {
    "Add", "Sub", "Mult", "Mod", "FloorDiv"};
  return left == "num" && right == "num" && integer_ops.count(op) ? "num" : "";
}

// The kind a module-level def named `callee` returns, by its annotation.
inline std::string annotated_return_kind(
  const std::string &callee,
  const nlohmann::json &module_body)
{
  if (module_body.is_array())
  {
    for (const auto &stmt : module_body)
    {
      if (
        !stmt.is_object() || stmt.value("_type", "") != "FunctionDef" ||
        stmt.value("name", "") != callee || !stmt.contains("returns") ||
        stmt["returns"].is_null())
        continue;

      const std::string ret_kind =
        scalar_name_kind(stmt["returns"].value("id", ""));
      if (!ret_kind.empty())
        return ret_kind;
    }
  }

  return "";
}

// The kind `self.<name>()` returns: every class defining `name` must agree on
// one scalar return annotation, as `self` may be an instance of any of them.
inline std::string
method_return_kind(const std::string &name, const nlohmann::json &module_body)
{
  std::string agreed;
  if (!module_body.is_array())
    return agreed;
  for (const auto &cls : module_body)
  {
    if (!cls.is_object() || cls.value("_type", "") != "ClassDef")
      continue;
    for (const auto &stmt : cls["body"])
    {
      if (
        stmt.value("_type", "") != "FunctionDef" ||
        stmt.value("name", "") != name)
        continue;
      const std::string kind =
        stmt.contains("returns") && stmt["returns"].is_object()
          ? scalar_name_kind(stmt["returns"].value("id", ""))
          : "";
      if (kind.empty() || (!agreed.empty() && kind != agreed))
        return "";
      agreed = kind;
    }
  }
  return agreed;
}

// Whether some `global` statement declares `name`.
inline bool declares_global(const nlohmann::json &node, const std::string &name)
{
  if (!node.is_structured())
    return false;
  if (node.is_object() && node.value("_type", "") == "Global")
    for (const auto &declared : node["names"])
      if (declared == name)
        return true;
  for (const auto &child : node)
    if (declares_global(child, name))
      return true;
  return false;
}

// The value of the one module-level `name = value`, where no function can
// rebind `name`; a parameter or local of the same name elsewhere is another
// variable.
inline const nlohmann::json *module_instance_value(
  const nlohmann::json &module_body,
  const std::string &name)
{
  if (!module_body.is_array() || declares_global(module_body, name))
    return nullptr;
  const nlohmann::json *bound = nullptr;
  for (const auto &stmt : module_body)
  {
    if (!stmt.is_object())
      continue;
    const std::string type = stmt.value("_type", "");
    if (
      (type == "FunctionDef" || type == "ClassDef") &&
      stmt.value("name", "") == name)
      return nullptr;
    if (const nlohmann::json *value = assigned_value_at(stmt, name))
    {
      if (bound)
        return nullptr;
      bound = value;
    }
  }
  return bound;
}

// Whether `callee` spells `self.<leaf>` or `<name>.<leaf>` for a module-level
// `name = Class(...)` of a class defined in the module.
inline bool is_instance_call(
  const std::string &callee,
  const std::string &leaf,
  const nlohmann::json &module_body)
{
  if (callee.size() <= leaf.size() + 1)
    return false;
  const std::string receiver =
    callee.substr(0, callee.size() - leaf.size() - 1);
  if (receiver == "self")
    return true;
  const nlohmann::json *bound = module_instance_value(module_body, receiver);
  if (
    !bound || !bound->is_object() || bound->value("_type", "") != "Call" ||
    !bound->contains("func") || (*bound)["func"].value("_type", "") != "Name")
    return false;
  const std::string cls = (*bound)["func"].value("id", "");
  for (const auto &stmt : module_body)
    if (
      stmt.is_object() && stmt.value("_type", "") == "ClassDef" &&
      stmt.value("name", "") == cls)
      return true;
  return false;
}

// The kind of an unannotated module-level def whose body is one `return`.
inline std::string forwarded_return_kind(
  const std::string &callee,
  const nlohmann::json &module_body,
  unsigned depth)
{
  if (!module_body.is_array())
    return "";
  for (const auto &stmt : module_body)
  {
    if (
      !stmt.is_object() || stmt.value("_type", "") != "FunctionDef" ||
      stmt.value("name", "") != callee)
      continue;
    const auto body = stmt.find("body");
    if (body == stmt.end() || !body->is_array() || body->size() != 1)
      return "";
    const nlohmann::json &ret = body->front();
    if (
      !ret.is_object() || ret.value("_type", "") != "Return" ||
      !ret.contains("value") || !ret["value"].is_object())
      return "";
    return classify_literal_value_kind(ret["value"], module_body, depth + 1);
  }
  return "";
}

// The kind of a call: a nondet_<type> function, a builtin conversion, abs of
// a number, or a function with a scalar return annotation.
inline std::string classify_call_kind(
  const nlohmann::json &call,
  const nlohmann::json &module_body,
  unsigned depth)
{
  const auto [callee, leaf] = callee_names(call["func"]);
  const std::string nondet_kind = nondet_leaf_kind(leaf);
  if (!nondet_kind.empty())
    return nondet_kind;

  const std::string builtin_kind = builtin_call_kind(callee);
  if (!builtin_kind.empty())
    return builtin_kind;

  if (callee == "abs" && call.contains("args") && call["args"].size() == 1)
  {
    const std::string kind =
      classify_literal_value_kind(call["args"][0], module_body, depth + 1);
    return kind == "num" || kind == "float" ? kind : "";
  }

  const std::string kind = annotated_return_kind(callee, module_body);
  if (!kind.empty())
    return kind;
  const std::string method_kind = is_instance_call(callee, leaf, module_body)
                                    ? method_return_kind(leaf, module_body)
                                    : "";
  return method_kind.empty() ? forwarded_return_kind(callee, module_body, depth)
                             : method_kind;
}

inline std::string classify_literal_value_kind(
  const nlohmann::json &value,
  const nlohmann::json &module_body,
  unsigned depth)
{
  const std::string type = value.value("_type", "");
  if (type == "Constant" && value.contains("value"))
    return constant_kind(value["value"]);

  if (depth >= 8)
    return "";

  if (type == "Name" && module_body.is_array())
  {
    const nlohmann::json *bound =
      single_assigned_value(module_body, value.value("id", ""));
    return bound ? classify_literal_value_kind(*bound, module_body, depth + 1)
                 : "";
  }

  if (type == "BinOp" && value.contains("left") && value.contains("right"))
    return binop_kind(
      value["op"].value("_type", ""),
      classify_literal_value_kind(value["left"], module_body, depth + 1),
      classify_literal_value_kind(value["right"], module_body, depth + 1));

  return type == "Call" && value.contains("func")
           ? classify_call_kind(value, module_body, depth)
           : "";
}

// Whether a name joining these literal kinds has a type that depends on the
// path. An int and a float alone are a number of path dependent type, which
// is refused where it is read (#8973).
// Whether any name was found to join a float with another kind, so that a
// tag may hold a float, which the tagged arithmetic and comparison helpers
// do not model.
inline bool &float_tags_possible()
{
  static bool possible = false;
  return possible;
}

inline bool kinds_diverge(const std::unordered_set<std::string> &kinds)
{
  const bool int_float_only =
    kinds.size() == 2 && kinds.count("num") && kinds.count("float");
  const bool diverge = kinds.size() >= 2 && !int_float_only;
  if (diverge && kinds.count("float"))
    float_tags_possible() = true;
  return diverge;
}

// Resolves `x = n` via name's own assignment in scope_body.
inline std::string classify_name_kind(
  const std::string &name,
  const nlohmann::json &scope_body,
  const nlohmann::json &module_body = nlohmann::json())
{
  if (!scope_body.is_array())
    return "";

  std::string kind;
  for (const auto &stmt : scope_body)
  {
    if (!stmt.is_object())
      continue;

    const std::string stmt_type = stmt.value("_type", "");
    nlohmann::json target;
    if (stmt_type == "Assign")
    {
      if (!stmt.contains("targets") || stmt["targets"].size() != 1)
        continue;
      target = stmt["targets"][0];
    }
    else if (stmt_type == "AnnAssign")
    {
      if (!stmt.contains("target"))
        continue;
      target = stmt["target"];
    }
    else
      continue;

    if (target.value("_type", "") != "Name" || target.value("id", "") != name)
      continue;

    if (!stmt.contains("value") || stmt["value"].is_null())
      kind.clear();
    else
      kind = classify_literal_value_kind(stmt["value"], module_body);
  }
  return kind;
}

// `x` for a Name target, `C.v` for an attribute of a name, else "".
inline std::string assigned_name(const nlohmann::json &target)
{
  if (target.value("_type", "") == "Name")
    return target.value("id", "");
  if (
    target.value("_type", "") == "Attribute" && target.contains("value") &&
    target["value"].value("_type", "") == "Name")
    return target["value"].value("id", "") + "." + target.value("attr", "");
  return "";
}

// Classifies a branch's top-level assignments by literal type; a Name
// value (e.g. `x = n`) is resolved via scope_body.
inline std::unordered_map<std::string, std::string>
classify_branch_literal_assigns(
  const nlohmann::json &block,
  const nlohmann::json &scope_body,
  const nlohmann::json &module_body = nlohmann::json())
{
  std::unordered_map<std::string, std::string> types;
  if (!block.is_array())
    return types;

  for (const auto &stmt : block)
  {
    if (!stmt.is_object())
      continue;

    const std::string stmt_type = stmt.value("_type", "");
    nlohmann::json target;
    if (stmt_type == "Assign")
    {
      if (!stmt.contains("targets") || stmt["targets"].size() != 1)
        continue;
      target = stmt["targets"][0];
    }
    else if (stmt_type == "AnnAssign")
    {
      if (!stmt.contains("target"))
        continue;
      target = stmt["target"];
    }
    else
      continue;

    const std::string name = assigned_name(target);
    if (name.empty())
      continue;

    // A later reassignment invalidates an earlier recorded kind for `name`.
    if (!stmt.contains("value") || stmt["value"].is_null())
    {
      types.erase(name);
      continue;
    }
    const auto &value = stmt["value"];
    std::string kind = classify_literal_value_kind(value, module_body);
    if (
      kind.empty() && value.value("_type", "") == "Name" &&
      value.contains("id"))
      kind = classify_name_kind(
        value["id"].get<std::string>(), scope_body, module_body);

    if (kind.empty())
      types.erase(name);
    else
      types[name] = kind;
  }

  return types;
}

// Collects, per name, the literal kinds assigned to it across every leaf
// reachable from `block`, following elif/nested if-else chains. Returns
// false if any such chain is dangling (no final else).
inline bool collect_branch_literal_kinds(
  const nlohmann::json &block,
  std::unordered_map<std::string, std::unordered_set<std::string>> &kinds,
  std::unordered_map<std::string, int> &leaf_count,
  int &leaf_total,
  const nlohmann::json &scope_body,
  const nlohmann::json &module_body = nlohmann::json())
{
  if (
    block.is_array() && block.size() == 1 && block[0].is_object() &&
    block[0].value("_type", "") == "If")
  {
    const auto &nested = block[0];
    if (
      !nested.contains("body") || !nested.contains("orelse") ||
      nested["orelse"].empty())
      return false;

    return collect_branch_literal_kinds(
             nested["body"],
             kinds,
             leaf_count,
             leaf_total,
             scope_body,
             module_body) &&
           collect_branch_literal_kinds(
             nested["orelse"],
             kinds,
             leaf_count,
             leaf_total,
             scope_body,
             module_body);
  }

  leaf_total++;
  for (const auto &[name, kind] :
       classify_branch_literal_assigns(block, scope_body, module_body))
  {
    kinds[name].insert(kind);
    leaf_count[name]++;
  }
  return true;
}

// Same as above, but starting from an `If` node's body/orelse directly.
inline bool collect_if_node_literal_kinds(
  const nlohmann::json &if_node,
  std::unordered_map<std::string, std::unordered_set<std::string>> &kinds,
  std::unordered_map<std::string, int> &leaf_count,
  int &leaf_total,
  const nlohmann::json &scope_body,
  const nlohmann::json &module_body = nlohmann::json())
{
  if (!if_node.contains("body"))
    return false;
  if (!if_node.contains("orelse") || if_node["orelse"].empty())
    return false;

  return collect_branch_literal_kinds(
           if_node["body"],
           kinds,
           leaf_count,
           leaf_total,
           scope_body,
           module_body) &&
         collect_branch_literal_kinds(
           if_node["orelse"],
           kinds,
           leaf_count,
           leaf_total,
           scope_body,
           module_body);
}

// Whether `stmt` assigns to `name`.
inline bool assigns_name(const nlohmann::json &stmt, const std::string &name)
{
  const std::string kind = stmt.value("_type", "");
  const nlohmann::json *target = nullptr;
  if (kind == "AnnAssign")
    target = &stmt["target"];
  else if (kind == "Assign" && stmt["targets"].size() == 1)
    target = &stmt["targets"][0];
  return target && target->is_object() &&
         target->value("_type", "") == "Name" &&
         target->value("id", "") == name;
}

// Whether `value` is a conditional expression whose arms have incompatible
// literal kinds.
inline bool ifexp_arms_diverge(
  const nlohmann::json &value,
  const nlohmann::json &module_body)
{
  if (
    !value.is_object() || value.value("_type", "") != "IfExp" ||
    !value.contains("body") || !value.contains("orelse"))
    return false;
  const std::string then_kind =
    classify_literal_value_kind(value["body"], module_body);
  const std::string else_kind =
    classify_literal_value_kind(value["orelse"], module_body);
  return !then_kind.empty() && !else_kind.empty() &&
         kinds_diverge({then_kind, else_kind});
}

// True if some top-level `If` in `scope_body`, or a conditional expression
// bound to `name`, assigns `name` genuinely incompatible literal kinds across
// every branch.
inline bool scope_assigns_divergent_literal_types(
  const std::string &name,
  const nlohmann::json &scope_body,
  const nlohmann::json &module_body = nlohmann::json())
{
  if (!scope_body.is_array())
    return false;

  for (const auto &stmt : scope_body)
  {
    if (!stmt.is_object())
      continue;
    if (
      assigns_name(stmt, name) && stmt.contains("value") &&
      ifexp_arms_diverge(stmt["value"], module_body))
      return true;
    if (stmt.value("_type", "") != "If")
      continue;

    std::unordered_map<std::string, std::unordered_set<std::string>> kinds;
    std::unordered_map<std::string, int> leaf_count;
    int leaf_total = 0;
    if (!collect_if_node_literal_kinds(
          stmt, kinds, leaf_count, leaf_total, scope_body, module_body))
      continue;

    auto it = kinds.find(name);
    if (
      it != kinds.end() && leaf_count[name] == leaf_total &&
      kinds_diverge(it->second))
      return true;
  }
  return false;
}

// A binding of a name to a number: its kind, "int" (bool included) or
// "float", the value bound, and whether it sits inside a block.
struct numeric_bindingt
{
  std::string kind;
  nlohmann::json value;
  bool in_block;
};

// The kind of number a binding of `name` stores: "int" (bool included),
// "float", or "" for anything else or another name.
inline std::string
numeric_binding_kind(const nlohmann::json &stmt, const std::string &name)
{
  if (!assigns_name(stmt, name))
    return "";
  const nlohmann::json &annotation = stmt.value("annotation", nlohmann::json());
  const std::string type =
    annotation.is_object() ? annotation.value("id", "") : "";
  if (type == "int" || type == "bool")
    return "int";
  return type == "float" ? "float" : "";
}

// A line field of `node`, or -1 when the preprocessor left it unset.
inline int line_field(const nlohmann::json &node, const char *key)
{
  const auto it = node.find(key);
  return it != node.end() && it->is_number_integer() ? it->get<int>() : -1;
}

// Whether `stmt` starts after line `before`, so that, outside a loop that
// spans `before`, nothing it binds reaches that line.
inline bool starts_after(const nlohmann::json &stmt, int before)
{
  return line_field(stmt, "lineno") >= before;
}

// Whether `stmt` is a loop whose body runs again after line `before`. A loop
// the preprocessor built without an end line is taken to span it.
inline bool loop_spans(const nlohmann::json &stmt, int before)
{
  const std::string kind = stmt.value("_type", "");
  const int end = line_field(stmt, "end_lineno");
  return (kind == "While" || kind == "For" || kind == "AsyncFor") &&
         line_field(stmt, "lineno") <= before && (end < 0 || end >= before);
}

// Records the number `stmt` binds to `name`. A binding on the straight-line
// spine before line `before` replaces every binding ahead of it on all paths,
// so those are dropped.
inline void record_numeric_binding(
  const nlohmann::json &stmt,
  const std::string &name,
  bool nested,
  int before,
  std::vector<numeric_bindingt> &bindings)
{
  const int line = line_field(stmt, "lineno");
  if (!nested && line >= 0 && line < before && assigns_name(stmt, name))
    bindings.clear();
  const std::string number = numeric_binding_kind(stmt, name);
  if (!number.empty() && stmt.contains("value") && !stmt["value"].is_null())
    bindings.push_back({number, stmt["value"], nested});
}

// The number bindings of `name` in `body`, outside nested scopes, that are
// live at line `before`. With `include_later`, those starting after it count
// too, as they do inside a loop that spans it.
inline void collect_numeric_bindings(
  const nlohmann::json &body,
  const std::string &name,
  bool nested,
  int before,
  bool include_later,
  std::vector<numeric_bindingt> &bindings)
{
  for (const auto &stmt : body)
  {
    if (!stmt.is_object() || (!include_later && starts_after(stmt, before)))
      continue;
    const std::string kind = stmt.value("_type", "");
    if (kind == "FunctionDef" || kind == "ClassDef")
      continue;
    record_numeric_binding(stmt, name, nested, before, bindings);
    const bool later = include_later || loop_spans(stmt, before);
    for (const char *key : {"body", "orelse", "finalbody", "handlers", "cases"})
      if (stmt.contains(key) && stmt[key].is_array())
        collect_numeric_bindings(
          stmt[key], name, true, before, later, bindings);
  }
}

// Whether the bindings store an int on one path and a float on another: an
// int and a float, at least one of them inside a block. A name rebound on
// the straight-line spine has the same type on every path.
inline bool
is_path_dependent_number(const std::vector<numeric_bindingt> &bindings)
{
  bool has_int = false, has_float = false, in_block = false;
  for (const auto &binding : bindings)
  {
    has_int |= binding.kind == "int";
    has_float |= binding.kind == "float";
    in_block |= binding.in_block;
  }
  return has_int && has_float && in_block;
}
} // namespace dynamic_type_detail
