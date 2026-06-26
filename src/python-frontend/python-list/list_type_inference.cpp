#include "python_list_internal.h"

using namespace python_expr;

// Default depth for list comparison if option not set
static const int DEFAULT_LIST_COMPARE_DEPTH = 4;

namespace python_list_detail
{
bool is_excluded_struct_tag_for_object_ref(const struct_typet &st)
{
  // Internal Python model aggregates (tuple, dict, Optional) are not user class
  // instances; the kind is read from the attribute stamped at type-creation
  // time. See util/python_types.h.
  return is_python_internal_aggregate(st);
}

bool is_empty_user_class_object_type(const typet &type, const namespacet &ns)
{
  typet resolved = type;
  if (resolved.id() == "symbol")
    resolved = ns.follow(resolved);

  if (!resolved.is_struct())
    return false;

  const std::string tag = to_struct_type(resolved).tag().as_string();
  if (tag.empty())
    return false;

  if (
    tag.find("__ESBMC_") != std::string::npos ||
    tag.rfind("tag-struct __ESBMC_", 0) == 0)
    return false;

  if (is_excluded_struct_tag_for_object_ref(to_struct_type(resolved)))
    return false;

  // Empty user-defined classes (no data fields) should be stored as object
  // references. Non-empty classes keep value-copy semantics in list storage.
  return to_struct_type(resolved).components().empty();
}

int get_list_compare_depth()
{
  std::string opt_value =
    config.options.get_option("python-list-compare-depth");
  if (!opt_value.empty())
  {
    try
    {
      int depth = std::stoi(opt_value);
      if (depth > 0)
        return depth;
    }
    catch (...)
    {
    }
  }
  return DEFAULT_LIST_COMPARE_DEPTH;
}

// Extract element type from annotation
typet get_elem_type_from_annotation(
  const nlohmann::json &node,
  const type_handler &type_handler_)
{
  // Extract element type from a Subscript node such as list[T]
  auto extract_subscript_elem = [&](const nlohmann::json &ann) -> typet {
    if (!ann.contains("slice") || !ann["slice"].is_object())
      return typet();
    const auto &slice = ann["slice"];

    // Simple element: list[int], list[str], ...
    if (slice.contains("id") && slice["id"].is_string())
      return type_handler_.get_typet(slice["id"].get<std::string>());

    // Tuple element: list[Tuple[A, B]] / list[tuple[A, B]]. Build the concrete
    // tuple struct (not the opaque 0-member "Tuple") so subscript/unpack of a
    // W[i] read sees real components; see type_handler::get_typet(json) for why
    // the opaque form crashes the SMT cast-to-struct path.
    if (
      slice.contains("_type") && slice["_type"] == "Subscript" &&
      slice.contains("value") && slice["value"].is_object() &&
      slice["value"].contains("id") && slice["value"]["id"].is_string() &&
      (slice["value"]["id"] == "Tuple" || slice["value"]["id"] == "tuple"))
      return type_handler_.get_typet(slice);

    // Nested container element: list[list[T]], list[dict[K, V]], ...
    // Resolve to the container's own type (list[T] -> __ESBMC_PyListObj*),
    // so the caller treats W[i] as a list and re-routes W[i][j] through
    // __ESBMC_list_at.
    if (
      slice.contains("_type") && slice["_type"] == "Subscript" &&
      slice.contains("value") && slice["value"].is_object() &&
      slice["value"].contains("id") && slice["value"]["id"].is_string())
    {
      return type_handler_.get_typet(slice["value"]["id"].get<std::string>());
    }

    return typet();
  };

  if (!node.contains("annotation") || !node["annotation"].is_object())
    return typet();

  const auto &annotation = node["annotation"];

  // Case 1: Direct subscript annotation like list[str]
  if (annotation.is_object() && annotation.contains("slice"))
  {
    typet elem_type = extract_subscript_elem(annotation);
    if (elem_type != typet())
      return elem_type;
  }

  // Case 2: Union type annotation such as list[str] | None
  if (
    annotation.is_object() && annotation.contains("_type") &&
    annotation["_type"] == "BinOp")
  {
    // Try left side first (e.g., handles list[str] | None)
    if (
      annotation.contains("left") && annotation["left"].is_object() &&
      annotation["left"].contains("_type") &&
      annotation["left"]["_type"] == "Subscript")
    {
      typet elem_type = extract_subscript_elem(annotation["left"]);
      if (elem_type != typet())
        return elem_type;
    }

    // Try right side (e.g., handles None | list[str])
    if (
      annotation.contains("right") && annotation["right"].is_object() &&
      annotation["right"].contains("_type") &&
      annotation["right"]["_type"] == "Subscript")
    {
      typet elem_type = extract_subscript_elem(annotation["right"]);
      if (elem_type != typet())
        return elem_type;
    }
  }

  // Case 3: Direct type annotation such as str, int
  if (annotation.contains("id") && annotation["id"].is_string())
    return type_handler_.get_typet(annotation["id"].get<std::string>());

  // Return empty type if annotation structure is not recognized
  return typet();
}

typet get_elem_type_from_return_annotation(
  const nlohmann::json &function_node,
  const type_handler &type_handler_)
{
  if (
    !function_node.contains("returns") || !function_node["returns"].is_object())
    return typet();

  nlohmann::json annotation_node;
  annotation_node["annotation"] = function_node["returns"];
  return get_elem_type_from_annotation(annotation_node, type_handler_);
}

typet infer_elem_type_from_call_return(
  const nlohmann::json &call_node,
  python_converter &converter)
{
  if (
    !call_node.is_object() || call_node["_type"] != "Call" ||
    !call_node.contains("func") || !call_node["func"].is_object())
    return typet();

  const auto &func = call_node["func"];
  const auto &ast = converter.ast();
  const auto &type_handler_ = converter.get_type_handler();

  if (func["_type"] == "Name" && func.contains("id") && func["id"].is_string())
  {
    nlohmann::json function_node =
      json_utils::find_function(ast["body"], func["id"].get<std::string>());
    return get_elem_type_from_return_annotation(function_node, type_handler_);
  }

  if (
    func["_type"] == "Attribute" && func.contains("attr") &&
    func["attr"].is_string() && func.contains("value") &&
    func["value"].is_object() && func["value"]["_type"] == "Name" &&
    func["value"].contains("id") && func["value"]["id"].is_string())
  {
    const std::string recv_name = func["value"]["id"].get<std::string>();
    nlohmann::json recv_decl = json_utils::find_var_decl(
      recv_name, converter.current_function_name(), ast);

    if (
      recv_decl.contains("annotation") && recv_decl["annotation"].is_object() &&
      recv_decl["annotation"].contains("id") &&
      recv_decl["annotation"]["id"].is_string())
    {
      const std::string class_name =
        recv_decl["annotation"]["id"].get<std::string>();
      nlohmann::json class_node =
        json_utils::find_class(ast["body"], class_name);
      if (!class_node.is_null() && class_node.contains("body"))
      {
        nlohmann::json function_node = json_utils::find_function(
          class_node["body"], func["attr"].get<std::string>());
        return get_elem_type_from_return_annotation(
          function_node, type_handler_);
      }
    }
  }

  return typet();
}

} // namespace python_list_detail
