// Out-of-class member definitions for `python_annotation<Json>` that
// construct annotation JSON nodes: Name / Subscript / Tuple-subscript
// annotations and the `update_assignment_node` /
// `add_parameter_annotation` / `update_end_col_offset` mutators.
//
// This file is `#include`d from python_annotation.h after the class
// definition. It must not be included elsewhere — the definitions
// depend on the class layout declared in python_annotation.h.
//
// Bodies are verbatim from the previous in-class definitions; only
// the `template <class Json>` line and the `python_annotation<Json>::`
// qualifier are new. No semantic change.

#pragma once

template <class Json>
Json python_annotation<Json>::create_name_annotation(
  const std::string &type_id,
  int lineno,
  int col_offset,
  int end_lineno,
  int end_col_offset)
{
  return {
    {"_type", "Name"},
    {"id", type_id},
    {"ctx", {{"_type", "Load"}}},
    {"lineno", lineno},
    {"col_offset", col_offset},
    {"end_lineno", end_lineno},
    {"end_col_offset", end_col_offset}};
}

template <class Json>
Json python_annotation<Json>::create_subscript_annotation(
  const std::string &base_type,
  const std::string &element_type,
  int lineno,
  int col_offset,
  int end_lineno)
{
  int base_end_col = col_offset + base_type.size();
  int slice_col = base_end_col + 1; // After '['
  int slice_end_col = slice_col + element_type.size();
  int total_end_col = col_offset + base_type.size() + 1 +
                      element_type.size() + 1; // type[element]

  // Check if this is a dict type with comma-separated types: dict[K, V]
  if (base_type == "dict" && element_type.find(',') != std::string::npos)
  {
    // Split element_type on comma
    size_t comma_pos = element_type.find(',');
    std::string key_type = element_type.substr(0, comma_pos);
    std::string value_type = element_type.substr(comma_pos + 1);

    // Trim whitespace
    key_type.erase(0, key_type.find_first_not_of(" \t"));
    key_type.erase(key_type.find_last_not_of(" \t") + 1);
    value_type.erase(0, value_type.find_first_not_of(" \t"));
    value_type.erase(value_type.find_last_not_of(" \t") + 1);

    // Create Tuple slice with two Name elements
    int key_col = slice_col;
    int key_end_col = key_col + key_type.size();
    int value_col = key_end_col + 2; // After ", "
    int value_end_col = value_col + value_type.size();

    Json tuple_slice = {
      {"_type", "Tuple"},
      {"elts",
       Json::array(
         {create_name_annotation(
            key_type, lineno, key_col, end_lineno, key_end_col),
          create_name_annotation(
            value_type, lineno, value_col, end_lineno, value_end_col)})},
      {"ctx", {{"_type", "Load"}}},
      {"lineno", lineno},
      {"col_offset", slice_col},
      {"end_lineno", end_lineno},
      {"end_col_offset", slice_end_col}};

    return {
      {"_type", "Subscript"},
      {"value",
       create_name_annotation(
         base_type, lineno, col_offset, end_lineno, base_end_col)},
      {"slice", tuple_slice},
      {"ctx", {{"_type", "Load"}}},
      {"lineno", lineno},
      {"col_offset", col_offset},
      {"end_lineno", end_lineno},
      {"end_col_offset", total_end_col}};
  }

  // Default: single element type (for list[T], set[T], etc.)
  return {
    {"_type", "Subscript"},
    {"value",
     create_name_annotation(
       base_type, lineno, col_offset, end_lineno, base_end_col)},
    {"slice",
     create_name_annotation(
       element_type, lineno, slice_col, end_lineno, slice_end_col)},
    {"ctx", {{"_type", "Load"}}},
    {"lineno", lineno},
    {"col_offset", col_offset},
    {"end_lineno", end_lineno},
    {"end_col_offset", total_end_col}};
}

// Build a `tuple[t0, t1, ...]` Subscript annotation node. Used by the
// parameter-inference pass to specialise bare `tuple` annotations once
// the element types have been recovered from call sites (GitHub #4515).
template <class Json>
Json python_annotation<Json>::create_tuple_subscript_annotation(
  const std::vector<std::string> &elem_types,
  int lineno,
  int col_offset,
  int end_lineno)
{
  static const std::string base_type = "tuple";
  int base_end_col = col_offset + base_type.size();
  int slice_col = base_end_col + 1;

  Json elts = Json::array();
  int cur = slice_col;
  for (const std::string &t : elem_types)
  {
    int end_col = cur + t.size();
    elts.push_back(
      create_name_annotation(t, lineno, cur, end_lineno, end_col));
    cur = end_col + 2; // ", "
  }
  int slice_end_col = cur - 2;
  int total_end_col = slice_end_col + 1;

  Json slice = {
    {"_type", "Tuple"},
    {"elts", elts},
    {"ctx", {{"_type", "Load"}}},
    {"lineno", lineno},
    {"col_offset", slice_col},
    {"end_lineno", end_lineno},
    {"end_col_offset", slice_end_col}};

  return {
    {"_type", "Subscript"},
    {"value",
     create_name_annotation(
       base_type, lineno, col_offset, end_lineno, base_end_col)},
    {"slice", slice},
    {"ctx", {{"_type", "Load"}}},
    {"lineno", lineno},
    {"col_offset", col_offset},
    {"end_lineno", end_lineno},
    {"end_col_offset", total_end_col}};
}

template <class Json>
Json python_annotation<Json>::create_annotation_from_type(
  const std::string &inferred_type,
  int lineno,
  int col_offset,
  int end_lineno)
{
  // Check if this is a generic type (e.g., list[dict])
  size_t bracket_pos = inferred_type.find('[');

  if (bracket_pos != std::string::npos)
  {
    // Generic type: extract base and element types
    std::string base_type = inferred_type.substr(0, bracket_pos);
    std::string element_type = inferred_type.substr(
      bracket_pos + 1, inferred_type.length() - bracket_pos - 2);

    return create_subscript_annotation(
      base_type, element_type, lineno, col_offset, end_lineno);
  }
  else
  {
    // Simple type
    int end_col_offset = col_offset + inferred_type.size();
    return create_name_annotation(
      inferred_type, lineno, col_offset, end_lineno, end_col_offset);
  }
}

template <class Json>
void python_annotation<Json>::update_assignment_node(
  Json &element,
  const std::string &inferred_type)
{
  auto target = element["targets"][0];
  std::string id;

  // Determine the ID based on the target type
  if (target.contains("id"))
    id = target["id"];
  // Get LHS from members access on assignments. e.g.: x.data = 10
  else if (target["_type"] == "Attribute")
  {
    // Only single-level obj.attr = ... is annotatable here. Nested writes
    // like obj.a.b = ... mutate an already-declared field and need no
    // inferred annotation; the inner value is an Attribute (no "id"),
    // not a Name, so reading "id" would throw json::type_error.
    if (!target["value"].is_object() || !target["value"].contains("id"))
      return;
    id = target["value"]["id"].template get<std::string>() + "." +
         target["attr"].template get<std::string>();
  }
  else if (target.contains("slice"))
    return; // No need to annotate subscript assignments (e.g. d["k"] = v).

  // Update type field only after confirming this is an annotatable target.
  element["_type"] = "AnnAssign";
  // Mark as inferred to distinguish from explicit
  // annotations like `x: Any = ...` during assignment type handling.
  element["_inferred_annotation"] = true;

  assert(!id.empty());

  // Calculate column offset with null safety
  int target_col_offset =
    target.contains("col_offset") && !target["col_offset"].is_null()
      ? target["col_offset"].template get<int>()
      : 0;
  int col_offset = target_col_offset + id.size() + 1;

  // Get line number with null safety
  int target_lineno = target.contains("lineno") && !target["lineno"].is_null()
                        ? target["lineno"].template get<int>()
                        : current_line_;

  int target_end_lineno =
    target.contains("end_lineno") && !target["end_lineno"].is_null()
      ? target["end_lineno"].template get<int>()
      : target_lineno;

  element["annotation"] = create_annotation_from_type(
    inferred_type, target_lineno, col_offset, target_end_lineno);

  // Update element properties with null safety
  int element_end_col_offset =
    element.contains("end_col_offset") && !element["end_col_offset"].is_null()
      ? element["end_col_offset"].template get<int>()
      : col_offset + inferred_type.size();

  element["end_col_offset"] =
    element_end_col_offset + inferred_type.size() + 1;
  element["end_lineno"] = target_lineno;
  element["simple"] = 1;

  // Replace "targets" array with "target" object
  element["target"] = std::move(target);
  element.erase("targets");

  // Remove unnecessary field
  element.erase("type_comment");

  // Update value fields with the correct offsets - with null safety
  auto update_offsets = [&inferred_type](Json &value) {
    if (value.contains("col_offset") && !value["col_offset"].is_null())
    {
      value["col_offset"] =
        value["col_offset"].template get<int>() + inferred_type.size() + 1;
    }
    if (
      value.contains("end_col_offset") && !value["end_col_offset"].is_null())
    {
      value["end_col_offset"] = value["end_col_offset"].template get<int>() +
                                inferred_type.size() + 1;
    }
  };

  update_offsets(element["value"]);

  // Adjust column offsets for function calls with arguments
  if (element["value"].contains("args"))
    for (auto &arg : element["value"]["args"])
      update_offsets(arg);

  // Adjust column offsets in function call node
  if (element["value"].contains("func"))
    update_offsets(element["value"]["func"]);
}

template <class Json>
void python_annotation<Json>::add_parameter_annotation(
  Json &param,
  const std::string &type)
{
  int col_offset = param["col_offset"].template get<int>() +
                   param["arg"].template get<std::string>().length() + 1;

  param["annotation"] = create_annotation_from_type(
    type,
    param["lineno"].template get<int>(),
    col_offset,
    param["lineno"].template get<int>());

  // Update the parameter's end_col_offset to account for the annotation
  param["end_col_offset"] =
    param["end_col_offset"].template get<int>() + type.size() + 1;
}

template <class Json>
void python_annotation<Json>::update_end_col_offset(Json &ast)
{
  int max_col_offset = ast["end_col_offset"];
  for (auto &elem : ast["body"])
  {
    if (elem["end_col_offset"] > max_col_offset)
      max_col_offset = elem["end_col_offset"];
  }
  ast["end_col_offset"] = max_col_offset;
}
