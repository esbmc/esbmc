#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>

// Pure-AST literal-kind divergence detection, shared by dynamic_type_handler
// and the annotation pre-pass (which has no symbol table to check against).
namespace dynamic_type_detail
{
// Classifies a branch's top-level literal assignments by literal type.
inline std::unordered_map<std::string, std::string>
classify_branch_literal_assigns(const nlohmann::json &block)
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
    if (value.value("_type", "") != "Constant" || !value.contains("value"))
    {
      types.erase(name);
      continue;
    }

    const auto &lit = value["value"];
    if (lit.is_string())
      types[name] = "str";
    else if (lit.is_number_integer() || lit.is_boolean())
      types[name] = "num";
    else
      types.erase(name);
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
  int &leaf_total)
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
             nested["body"], kinds, leaf_count, leaf_total) &&
           collect_branch_literal_kinds(
             nested["orelse"], kinds, leaf_count, leaf_total);
  }

  leaf_total++;
  for (const auto &[name, kind] : classify_branch_literal_assigns(block))
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
  int &leaf_total)
{
  if (!if_node.contains("body"))
    return false;
  if (!if_node.contains("orelse") || if_node["orelse"].empty())
    return false;

  return collect_branch_literal_kinds(
           if_node["body"], kinds, leaf_count, leaf_total) &&
         collect_branch_literal_kinds(
           if_node["orelse"], kinds, leaf_count, leaf_total);
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
    if (!collect_if_node_literal_kinds(stmt, kinds, leaf_count, leaf_total))
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
