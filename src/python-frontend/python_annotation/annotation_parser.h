#pragma once

#include <nlohmann/json.hpp>

#include <string>

// AST-walking helpers extracted from python_annotation.h. These are
// pure functions over the JSON AST: they read the tree and return
// found nodes (or an empty Json). They never mutate the AST and do
// not depend on `python_annotation` class state.
namespace python_annotation_parser
{

// Find a top-level `Assign` whose target id is @p func_name and whose
// value is a `Lambda`. Returns the surrounding `Assign` node, or an
// empty Json when not found. Search is one level deep over @p body.
nlohmann::json find_lambda_in_body(
  const std::string &func_name,
  const nlohmann::json &body);

// Recursively walk @p body looking for a `FunctionDef` with the name
// @p func_name. Descends into nested function bodies and the `body`
// / `orelse` of any control-flow construct. Returns the matching
// node or an empty Json.
nlohmann::json find_function_recursive(
  const std::string &func_name,
  const nlohmann::json &body);

// Inside class @p cls (looked up in @p ast_body), find the
// `__init__` method and return the right-hand side of the first
// `self.@p attr = ...` assignment. Returns an empty Json when no
// such assignment exists. @p ast_body must be the top-level AST
// `body` array (i.e. `ast["body"]`) — `json_utils::find_class`
// iterates it expecting class-shaped nodes.
nlohmann::json find_self_attr_init_rhs(
  const std::string &cls,
  const std::string &attr,
  const nlohmann::json &ast_body);

} // namespace python_annotation_parser
