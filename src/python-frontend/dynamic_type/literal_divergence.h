#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>

// Pure-AST literal-kind divergence detection, shared by dynamic_type_handler
// and the annotation pre-pass (which has no symbol table to check against).
namespace dynamic_type_detail
{
// Classifies a literal constant or a nondet_<type> builtin call; "" otherwise.
inline std::string classify_literal_value_kind(const nlohmann::json &value)
{
  if (value.value("_type", "") == "Constant" && value.contains("value"))
  {
    const auto &lit = value["value"];
    if (lit.is_string())
      return "str";
    if (lit.is_number_integer() || lit.is_boolean())
      return "num";
    // A float is a number too: `1 if c else 2.5` is not a type divergence,
    // while `2.0` against a str is. Leaving floats unclassified made a
    // float/str join look like no divergence at all (github #8263).
    if (lit.is_number_float())
      return "num";
    return "";
  }

  if (
    value.value("_type", "") == "Call" && value.contains("func") &&
    value["func"].value("_type", "") == "Name")
  {
    const std::string &callee = value["func"].value("id", "");
    if (callee == "nondet_int" || callee == "nondet_bool")
      return "num";
    if (callee == "nondet_str")
      return "str";
  }

  return "";
}

// Resolves `x = n` via name's own assignment in scope_body.
inline std::string
classify_name_kind(const std::string &name, const nlohmann::json &scope_body)
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
      kind = classify_literal_value_kind(stmt["value"]);
  }
  return kind;
}

// Classifies a branch's top-level assignments by literal type; a Name
// value (e.g. `x = n`) is resolved via scope_body.
inline std::unordered_map<std::string, std::string>
classify_branch_literal_assigns(
  const nlohmann::json &block,
  const nlohmann::json &scope_body)
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

    if (target.value("_type", "") != "Name" || !target.contains("id"))
      continue;
    const std::string &name = target["id"].get<std::string>();

    // A later reassignment invalidates an earlier recorded kind for `name`.
    if (!stmt.contains("value") || stmt["value"].is_null())
    {
      types.erase(name);
      continue;
    }
    const auto &value = stmt["value"];
    std::string kind = classify_literal_value_kind(value);
    if (
      kind.empty() && value.value("_type", "") == "Name" &&
      value.contains("id"))
      kind = classify_name_kind(value["id"].get<std::string>(), scope_body);

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
  const nlohmann::json &scope_body)
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
             nested["body"], kinds, leaf_count, leaf_total, scope_body) &&
           collect_branch_literal_kinds(
             nested["orelse"], kinds, leaf_count, leaf_total, scope_body);
  }

  leaf_total++;
  for (const auto &[name, kind] :
       classify_branch_literal_assigns(block, scope_body))
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
  const nlohmann::json &scope_body)
{
  if (!if_node.contains("body"))
    return false;
  if (!if_node.contains("orelse") || if_node["orelse"].empty())
    return false;

  return collect_branch_literal_kinds(
           if_node["body"], kinds, leaf_count, leaf_total, scope_body) &&
         collect_branch_literal_kinds(
           if_node["orelse"], kinds, leaf_count, leaf_total, scope_body);
}

// Kind of an expression under the kinds currently in scope. Only the shapes a
// divergence can travel through are modelled: a literal, a name, and an
// arithmetic join of two of them. "" means "not known here", which is never
// reported as a divergence.
inline std::string expr_kind(
  const nlohmann::json &value,
  const std::unordered_map<std::string, std::string> &kinds)
{
  if (!value.is_object())
    return "";

  const std::string node = value.value("_type", "");
  if (node == "Name")
  {
    auto it = kinds.find(value.value("id", ""));
    return it == kinds.end() ? "" : it->second;
  }
  if (node == "BinOp" && value.contains("left") && value.contains("right"))
  {
    const std::string l = expr_kind(value["left"], kinds);
    const std::string r = expr_kind(value["right"], kinds);
    // `str + str` is a str and `num + num` a num; a mixed pair is already a
    // type error at that operator, not a divergence to carry forward.
    return (!l.empty() && l == r) ? l : "";
  }
  return classify_literal_value_kind(value);
}

// Walks `body` in order, threading each name's kind, and records every name a
// branch join leaves with two incompatible kinds. A one-armed `if` joins the
// body's kinds against the kinds that reach the `if`, which is what Python
// does.
inline void scan_scope_divergence(
  const nlohmann::json &body,
  std::unordered_map<std::string, std::string> &kinds,
  std::unordered_set<std::string> &divergent)
{
  if (!body.is_array())
    return;

  for (const auto &stmt : body)
  {
    if (!stmt.is_object())
      continue;

    const std::string node = stmt.value("_type", "");
    if (node == "Assign" || node == "AnnAssign")
    {
      nlohmann::json target;
      if (node == "Assign")
      {
        if (!stmt.contains("targets") || stmt["targets"].size() != 1)
          continue;
        target = stmt["targets"][0];
      }
      else
      {
        if (!stmt.contains("target"))
          continue;
        target = stmt["target"];
      }
      if (target.value("_type", "") != "Name")
        continue;
      const std::string name = target.value("id", "");
      const std::string kind =
        stmt.contains("value") && !stmt["value"].is_null()
          ? expr_kind(stmt["value"], kinds)
          : "";
      if (kind.empty())
        kinds.erase(name);
      else
        kinds[name] = kind;
      continue;
    }

    if (node == "If")
    {
      auto then_kinds = kinds;
      auto else_kinds = kinds;
      if (stmt.contains("body"))
        scan_scope_divergence(stmt["body"], then_kinds, divergent);
      if (stmt.contains("orelse") && !stmt["orelse"].empty())
        scan_scope_divergence(stmt["orelse"], else_kinds, divergent);

      std::unordered_map<std::string, std::string> joined;
      for (const auto &[name, kind] : then_kinds)
      {
        auto other = else_kinds.find(name);
        if (other == else_kinds.end() || other->second != kind)
        {
          if (other != else_kinds.end() && !other->second.empty())
            divergent.insert(name);
          continue;
        }
        joined.emplace(name, kind);
      }
      kinds = joined;
      continue;
    }

    // Any other statement may rebind a name in ways not modelled here; a
    // loop body in particular can run zero or many times. Drop what a
    // nested block assigns rather than carry a stale kind forward.
    if (stmt.contains("body"))
    {
      std::unordered_map<std::string, std::string> inner = kinds;
      scan_scope_divergence(stmt["body"], inner, divergent);
      for (const auto &[name, kind] : inner)
        if (!kinds.count(name) || kinds[name] != kind)
          kinds.erase(name);
    }
  }
}

// Names that a branch join in `scope_body` leaves holding two incompatible
// kinds.
inline std::unordered_set<std::string>
scope_divergent_names(const nlohmann::json &scope_body)
{
  std::unordered_map<std::string, std::string> kinds;
  std::unordered_set<std::string> divergent;
  scan_scope_divergence(scope_body, kinds, divergent);
  return divergent;
}

// Names assigned by a plain `x = ...` anywhere inside `node` (recursing into
// nested blocks), used to ask a scope-wide divergence set about one `if`.
inline void collect_assigned_names(
  const nlohmann::json &node,
  std::unordered_set<std::string> &names)
{
  if (node.is_array())
  {
    for (const auto &item : node)
      collect_assigned_names(item, names);
    return;
  }
  if (!node.is_object())
    return;
  const std::string type = node.value("_type", "");
  if (type == "Assign" && node.contains("targets"))
    for (const auto &t : node["targets"])
      if (t.value("_type", "") == "Name")
        names.insert(t.value("id", ""));
  if (
    type == "AnnAssign" && node.contains("target") &&
    node["target"].value("_type", "") == "Name")
    names.insert(node["target"].value("id", ""));
  for (const char *key : {"body", "orelse"})
    if (node.contains(key))
      collect_assigned_names(node[key], names);
}

// True if `name` is read anywhere inside `node`.
inline bool name_is_loaded(const nlohmann::json &node, const std::string &name)
{
  if (node.is_array())
  {
    for (const auto &item : node)
      if (name_is_loaded(item, name))
        return true;
    return false;
  }
  if (!node.is_object())
    return false;
  if (
    node.value("_type", "") == "Name" && node.value("id", "") == name &&
    node.contains("ctx") && node["ctx"].value("_type", "") == "Load")
    return true;
  for (const auto &[key, child] : node.items())
    if ((child.is_object() || child.is_array()) && name_is_loaded(child, name))
      return true;
  return false;
}

// True if `name` is read in `scope_body` after the statement `after`.
inline bool name_is_loaded_after(
  const nlohmann::json &scope_body,
  const nlohmann::json &after,
  const std::string &name)
{
  if (!scope_body.is_array())
    return false;
  bool passed = false;
  for (const auto &stmt : scope_body)
  {
    if (passed && name_is_loaded(stmt, name))
      return true;
    if (!passed && stmt == after)
      passed = true;
  }
  return false;
}

// True if some top-level `If` in `scope_body` assigns `name` genuinely
// incompatible literal kinds across every branch.
inline bool scope_assigns_divergent_literal_types(
  const std::string &name,
  const nlohmann::json &scope_body)
{
  if (!scope_body.is_array())
    return false;

  for (const auto &stmt : scope_body)
  {
    if (!stmt.is_object() || stmt.value("_type", "") != "If")
      continue;

    std::unordered_map<std::string, std::unordered_set<std::string>> kinds;
    std::unordered_map<std::string, int> leaf_count;
    int leaf_total = 0;
    if (!collect_if_node_literal_kinds(
          stmt, kinds, leaf_count, leaf_total, scope_body))
      continue;

    auto it = kinds.find(name);
    if (
      it != kinds.end() && leaf_count[name] == leaf_total &&
      it->second.size() >= 2)
      return true;
  }
  return false;
}
} // namespace dynamic_type_detail
