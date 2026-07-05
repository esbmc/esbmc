#pragma once

#include <nlohmann/json.hpp>
#include <util/type.h>

class type_handler;
class python_converter;
class namespacet;
class struct_typet;

// File-local helpers extracted from python_list.cpp.  These are pure
// type-inference / classification utilities with no python_list state; they are
// shared across the python-list translation units via this declaration.
namespace python_list_detail
{
bool is_excluded_struct_tag_for_object_ref(const struct_typet &st);

bool is_empty_user_class_object_type(const typet &type, const namespacet &ns);

int get_list_compare_depth();

// Extract element type from a variable/parameter annotation node.
typet get_elem_type_from_annotation(
  const nlohmann::json &node,
  const type_handler &type_handler_);

// Extract element type from a function's return annotation.
typet get_elem_type_from_return_annotation(
  const nlohmann::json &function_node,
  const type_handler &type_handler_);

// Infer the element type produced by a Call node from its callee's return
// annotation (free function or method).
typet infer_elem_type_from_call_return(
  const nlohmann::json &call_node,
  python_converter &converter);
} // namespace python_list_detail
