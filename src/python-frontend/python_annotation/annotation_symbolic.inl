// Out-of-class member definitions for `python_annotation<Json>` that
// implement class-inheritance reasoning and per-class function
// specialisation:
//
//   - are_all_user_classes
//   - build_specialized_function_name
//   - resolve_name_assigned_class
//   - rewrite_specialized_calls
//   - apply_pending_specializations
//   - get_class_ancestors
//   - find_common_ancestor
//
// This file is `#include`d from python_annotation.h after the class
// definition. Bodies are verbatim from the previous in-class
// definitions; only the `template <class Json>` line and the
// `python_annotation<Json>::` qualifier are new. No semantic change.

#pragma once

template <class Json>
bool python_annotation<Json>::are_all_user_classes(
  const std::vector<std::string> &types) const
{
  if (types.empty())
    return false;

  for (const auto &type_name : types)
  {
    if (type_name.empty() || !json_utils::is_class(type_name, ast_))
      return false;
  }

  return true;
}

template <class Json>
std::string python_annotation<Json>::build_specialized_function_name(
  const std::string &base_name,
  const std::string &class_name) const
{
  return base_name + "__esbmc_poly_" + class_name;
}

template <class Json>
std::string python_annotation<Json>::resolve_name_assigned_class(
  const std::string &name,
  const Json &node)
{
  if (name.empty())
    return "";

  if (node.is_object())
  {
    if (
      node.contains("_type") && node["_type"] == "Assign" &&
      node.contains("targets") && node["targets"].is_array() &&
      !node["targets"].empty() && node["targets"][0].is_object() &&
      node["targets"][0].contains("_type") &&
      node["targets"][0]["_type"] == "Name" &&
      node["targets"][0].contains("id") && node["targets"][0]["id"] == name &&
      node.contains("value") && node["value"].is_object() &&
      node["value"].contains("_type") && node["value"]["_type"] == "Call" &&
      node["value"].contains("func") && node["value"]["func"].is_object() &&
      node["value"]["func"].contains("_type") &&
      node["value"]["func"]["_type"] == "Name" &&
      node["value"]["func"].contains("id"))
    {
      std::string class_name =
        node["value"]["func"]["id"].template get<std::string>();
      if (json_utils::is_class(class_name, ast_))
        return class_name;
    }

    for (auto it = node.begin(); it != node.end(); ++it)
    {
      std::string found = resolve_name_assigned_class(name, it.value());
      if (!found.empty())
        return found;
    }
  }
  else if (node.is_array())
  {
    for (const auto &element : node)
    {
      std::string found = resolve_name_assigned_class(name, element);
      if (!found.empty())
        return found;
    }
  }

  return "";
}

template <class Json>
void python_annotation<Json>::rewrite_specialized_calls(
  const std::string &original_name,
  size_t param_index,
  const std::unordered_map<std::string, std::string> &specialized_names,
  Json &node)
{
  if (node.is_object())
  {
    if (
      node.contains("_type") && node["_type"] == "Call" &&
      node.contains("func") && node["func"].is_object() &&
      node["func"].contains("_type") && node["func"]["_type"] == "Name" &&
      node["func"].contains("id") && node["func"]["id"] == original_name &&
      node.contains("args") && node["args"].is_array() &&
      param_index < node["args"].size())
    {
      std::string arg_type = get_argument_type(node["args"][param_index]);
      if (
        arg_type.empty() && node["args"][param_index].contains("_type") &&
        node["args"][param_index]["_type"] == "Name" &&
        node["args"][param_index].contains("id"))
      {
        const std::string arg_name =
          node["args"][param_index]["id"].template get<std::string>();
        arg_type = resolve_name_assigned_class(arg_name, ast_["body"]);
      }

      auto it = specialized_names.find(arg_type);
      if (it != specialized_names.end())
        node["func"]["id"] = it->second;
    }

    for (auto it = node.begin(); it != node.end(); ++it)
      rewrite_specialized_calls(
        original_name, param_index, specialized_names, it.value());
  }
  else if (node.is_array())
  {
    for (auto &element : node)
      rewrite_specialized_calls(
        original_name, param_index, specialized_names, element);
  }
}

template <class Json>
void python_annotation<Json>::apply_pending_specializations()
{
  for (const auto &spec : pending_specializations_)
  {
    const std::string &func_name = spec.function_name;
    size_t param_index = spec.param_index;

    Json original_function;
    bool found = false;
    size_t original_index = 0;
    size_t idx = 0;
    for (const auto &node : ast_["body"])
    {
      if (
        node.contains("_type") && node["_type"] == "FunctionDef" &&
        node.contains("name") && node["name"] == func_name)
      {
        original_function = node;
        found = true;
        original_index = idx;
        break;
      }
      ++idx;
    }

    if (!found)
      continue;

    if (!spec.class_types.empty())
    {
      for (auto &node : ast_["body"])
      {
        if (
          node.contains("_type") && node["_type"] == "FunctionDef" &&
          node.contains("name") && node["name"] == func_name &&
          node.contains("args") && node["args"].is_object() &&
          node["args"].contains("args") && node["args"]["args"].is_array() &&
          param_index < node["args"]["args"].size())
        {
          Json &param = node["args"]["args"][param_index];
          add_parameter_annotation(param, spec.class_types.front());
          break;
        }
      }
    }

    std::unordered_map<std::string, std::string> specialized_names;
    Json specialized_nodes = Json::array();
    for (const auto &class_name : spec.class_types)
    {
      Json specialized = original_function;
      const std::string specialized_name =
        build_specialized_function_name(func_name, class_name);
      specialized["name"] = specialized_name;

      if (
        specialized.contains("args") && specialized["args"].is_object() &&
        specialized["args"].contains("args") &&
        specialized["args"]["args"].is_array() &&
        param_index < specialized["args"]["args"].size())
      {
        Json &param = specialized["args"]["args"][param_index];
        add_parameter_annotation(param, class_name);
      }

      specialized_names[class_name] = specialized_name;
      specialized_nodes.push_back(std::move(specialized));
    }

    auto insert_it =
      ast_["body"].begin() + static_cast<ptrdiff_t>(original_index + 1);
    ast_["body"].insert(
      insert_it, specialized_nodes.begin(), specialized_nodes.end());

    rewrite_specialized_calls(
      func_name, param_index, specialized_names, ast_);
  }

  pending_specializations_.clear();
}

// Returns the ancestors of a class (including itself) in BFS order via
// "bases". Handles multiple inheritance and cycle-safe (visited guard).
// "object" is intentionally excluded — it has no ESBMC IR representation.
template <class Json>
std::vector<std::string>
python_annotation<Json>::get_class_ancestors(const std::string &class_name)
{
  std::vector<std::string> ancestors;
  std::unordered_set<std::string> visited;
  std::vector<std::string> worklist = {class_name};
  size_t idx = 0;
  while (idx < worklist.size())
  {
    std::string cur = worklist[idx++];
    if (cur.empty() || cur == "object" || visited.count(cur))
      continue;
    visited.insert(cur);
    ancestors.push_back(cur);
    Json cls = json_utils::find_class(ast_["body"], cur);
    if (cls.empty() || !cls.contains("bases"))
      continue;
    for (const auto &base : cls["bases"])
      if (base.contains("id"))
        worklist.push_back(base["id"].template get<std::string>());
  }
  return ancestors;
}

// Returns the lowest common ancestor of two user-defined class types, or "".
template <class Json>
std::string python_annotation<Json>::find_common_ancestor(
  const std::string &type_a,
  const std::string &type_b)
{
  std::vector<std::string> ancestors_a = get_class_ancestors(type_a);
  std::unordered_set<std::string> ancestors_b_set;
  for (const auto &anc : get_class_ancestors(type_b))
    ancestors_b_set.insert(anc);
  for (const auto &anc : ancestors_a)
    if (ancestors_b_set.count(anc))
      return anc;
  return "";
}
