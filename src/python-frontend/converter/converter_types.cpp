#include <python-frontend/function_call/expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_utils.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/python_types.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <set>

exprt python_converter::unwrap_optional_if_needed(
  const exprt &expr,
  const nlohmann::json &element)
{
  if (!expr.type().is_struct())
    return expr;

  const struct_typet &struct_type = to_struct_type(expr.type());
  std::string tag = struct_type.tag().as_string();

  if (tag.starts_with("tag-Optional_"))
  {
    // A member can only be taken of an addressable value. When the optional is
    // produced by a call used directly in a comparison -- e.g. f(...) == 2 --
    // the operand is the call itself (a code_function_callt statement, or a
    // side-effect call expression), and member_exprt(<call>, "value") is
    // malformed: it aborts during goto migration (member2t requires a
    // struct/union/complex source). Materialise the call result into a
    // temporary first, mirroring the already-working assigned path
    // (r = f(...); r.value == 2). See #4807.
    exprt base = expr;
    if (base.is_code() && base.is_function_call())
      base = to_value_expr(base, name_space());

    if (base.id() == "sideeffect")
    {
      symbolt &tmp =
        create_tmp_symbol(element, "$optional_tmp$", base.type(), exprt());
      code_declt decl(symbol_expr(tmp));
      decl.location() = base.location();
      add_instruction(decl);

      code_assignt assign(symbol_expr(tmp), base);
      assign.location() = base.location();
      add_instruction(assign);
      base = symbol_expr(tmp);
    }

    // Extract the value field. V.3: IREP2 member access (round-trip).
    expr2tc b2;
    migrate_expr(base, b2);
    return migrate_expr_back(
      member2tc(migrate_type(struct_type.components()[1].type()), b2, "value"));
  }

  return expr;
}

exprt python_converter::wrap_in_optional(
  const exprt &value,
  const typet &optional_type)
{
  assert(optional_type.is_struct());
  const struct_typet &struct_type = to_struct_type(optional_type);

  // Create struct expression
  struct_exprt optional_value(struct_type);

  // Set is_none field based on whether value is None
  exprt is_none_value;
  if (value.type() == none_type())
  {
    is_none_value = gen_boolean(true);
    // Set value field to zero for None case
    optional_value.operands().push_back(is_none_value);
    optional_value.operands().push_back(
      gen_zero(struct_type.components()[1].type()));
  }
  else
  {
    is_none_value = gen_boolean(false);
    optional_value.operands().push_back(is_none_value);
    optional_value.operands().push_back(value);
  }

  return optional_value;
}

// Extract non-None type from union
std::string
python_converter::extract_non_none_type(const nlohmann::json &annotation_node)
{
  std::function<std::string(const nlohmann::json &)> extract_type =
    [&](const nlohmann::json &node) -> std::string {
    if (
      node.contains("_type") && node["_type"] == "Constant" &&
      node.contains("value") && node["value"].is_null() &&
      !node.contains("_bigint"))
      return ""; // This is None (a tagged bignum is not None — issue #4642)

    if (node.contains("id"))
      return node["id"].get<std::string>();

    // Handle Subscript nodes (such as Literal["bar"] or Sequence[str])
    if (node.contains("_type") && node["_type"] == "Subscript")
    {
      if (node.contains("value") && node["value"].is_object())
      {
        const auto &value_node = node["value"];
        // Handle Name nodes (e.g., List[int], Literal["bar"])
        if (value_node.contains("id"))
        {
          std::string subscript_type = value_node["id"].get<std::string>();
          if (subscript_type == "Literal")
            return "__LITERAL__"; // Special marker for Literal types
          // For Sequence[str], List[int], etc., return "list" as the concrete type
          if (subscript_type == "Sequence" || subscript_type == "List")
            return "list";
          // For other generic types, return the base type
          return subscript_type;
        }
        // Handle Attribute nodes (e.g., re.Match[str], typing.Optional[int])
        if (
          value_node.contains("_type") && value_node["_type"] == "Attribute" &&
          value_node.contains("attr"))
        {
          // Return special marker for external module types that should be
          // treated as opaque/any type (e.g., re.Match, typing.Pattern, etc.)
          return "__EXTERNAL_TYPE__";
        }
      }
      return ""; // Other subscript types
    }

    // Handle standalone Attribute nodes (e.g., module.Type without subscript)
    if (
      node.contains("_type") && node["_type"] == "Attribute" &&
      node.contains("attr"))
    {
      return "__EXTERNAL_TYPE__";
    }

    // Recursively handle nested BinOp (e.g., bool | str in bool | str | None)
    if (node.contains("_type") && node["_type"] == "BinOp")
    {
      if (node.contains("left"))
      {
        std::string left_type = extract_type(node["left"]);
        if (!left_type.empty())
          return left_type;
      }
      if (node.contains("right"))
        return extract_type(node["right"]);
    }

    return "";
  };

  // Guard: ensure annotation_node has left and right before accessing
  if (!annotation_node.contains("left") || !annotation_node.contains("right"))
    return "";

  const auto &left = annotation_node["left"];
  const auto &right = annotation_node["right"];

  // Extract the first non-None type
  std::string inner_type = extract_type(left);
  if (inner_type.empty())
    inner_type = extract_type(right);

  return inner_type;
}

typet python_converter::get_type_from_annotation(
  const nlohmann::json &annotation_node,
  const nlohmann::json &element)
{
  // Be defensive: not all annotation nodes are guaranteed to have the same
  // structure. In particular, forward references or tool-generated annotations
  // may appear as plain strings or objects without a "_type" field. On some
  // platforms (e.g., macOS debug builds) accessing a missing key via
  // operator[] triggers an assertion inside nlohmann::json, so we must guard
  // all such uses.
  if (!annotation_node.is_object())
  {
    // String-like forward reference, e.g. "CoordinateData | None"
    if (annotation_node.is_string())
    {
      std::string type_string = annotation_node.get<std::string>();
      type_string = type_utils::remove_quotes(type_string);
      return type_handler_.get_typet(type_string);
    }

    // Unknown/unsupported shape – fall back to empty type (no assertion
    // should be emitted for this annotation).
    return empty_typet();
  }

  if (!annotation_node.contains("_type"))
  {
    // Minimal object with direct "id" field, e.g. {"id": "int"}
    if (annotation_node.contains("id"))
    {
      std::string type_id = annotation_node["id"].get<std::string>();
      if (type_id == "NoneType")
        return any_type();

      if (type_id == "dict" || type_id == "Dict")
      {
        // User-defined class named "dict"/"Dict" takes precedence over built-in
        if (!json_utils::is_class(type_id, *ast_json))
          return dict_handler_->get_dict_struct_type();
      }
      if (type_id == "list" || type_id == "List")
      {
        // User-defined class named "list"/"List" takes precedence over built-in
        if (!json_utils::is_class(type_id, *ast_json))
          return type_handler_.get_list_type();
      }

      return type_handler_.get_typet(type_id);
    }

    // Nothing recognizable – treat as empty/unknown
    return empty_typet();
  }

  if (annotation_node["_type"] == "Subscript")
  {
    // Helper to safely get id from value node
    auto get_value_id = [&]() -> std::string {
      if (
        annotation_node.contains("value") &&
        annotation_node["value"].is_object() &&
        annotation_node["value"].contains("id"))
      {
        return annotation_node["value"]["id"].get<std::string>();
      }
      return "";
    };

    std::string value_id = get_value_id();

    if (value_id == "list" || value_id == "List")
      return type_handler_.get_list_type();

    if (value_id == "dict" || value_id == "Dict")
      return dict_handler_->get_dict_struct_type();

    if (value_id == "tuple" || value_id == "Tuple")
      return tuple_handler_->get_tuple_type_from_annotation(annotation_node);

    // Handle Literal[T]: extract the type from the literal value
    if (value_id == "Literal")
    {
      // Infer type from a literal constant value
      auto infer_literal_type = [](const nlohmann::json &value) -> typet {
        if (value.is_string())
          return gen_pointer_type(char_type());
        else if (value.is_number_integer())
          return long_long_int_type();
        else if (value.is_boolean())
          return bool_type();
        else if (value.is_number_float())
          return double_type();
        else if (value.is_null())
          return none_type();

        return empty_typet(); // Unsupported type
      };

      // Resolve a slice element to a constant value
      auto resolve_to_constant =
        [this](const nlohmann::json &elem) -> nlohmann::json {
        // Guard: ensure elem is an object with _type
        if (!elem.is_object() || !elem.contains("_type"))
          return nlohmann::json();

        // Direct constant
        if (elem["_type"] == "Constant" && elem.contains("value"))
          return elem["value"];
        // Variable reference: resolve it
        if (elem["_type"] == "Name" && elem.contains("id"))
        {
          std::string var_name = elem["id"].get<std::string>();
          nlohmann::json var_decl =
            json_utils::find_var_decl(var_name, "", *ast_json);
          if (
            !var_decl.empty() && var_decl.contains("value") &&
            var_decl["value"].is_object() &&
            var_decl["value"].contains("_type") &&
            var_decl["value"]["_type"] == "Constant" &&
            var_decl["value"].contains("value"))
          {
            return var_decl["value"]["value"];
          }
        }
        return nlohmann::json(); // Could not resolve
      };

      // Track type flags from a resolved type
      auto update_type_flags = [](
                                 const typet &type,
                                 TypeFlags &flags,
                                 bool &has_string,
                                 bool &has_none) {
        if (type == gen_pointer_type(char_type()))
          has_string = true;
        else if (type == double_type())
          flags.has_float = true;
        else if (type == long_long_int_type())
          flags.has_int = true;
        else if (type == bool_type())
          flags.has_bool = true;
        else if (type == none_type())
          has_none = true;
        else if (type == pointer_type())
        {
          // Mixed type: mark as having both string and numeric
          has_string = true;
          flags.has_int = true;
        }
      };

      if (annotation_node.contains("slice"))
      {
        const auto &slice = annotation_node["slice"];

        // Guard: ensure slice is an object with _type
        if (!slice.is_object() || !slice.contains("_type"))
          return empty_typet();

        // Helper to safely check if node is a Literal subscript
        auto is_literal_subscript_node =
          [](const nlohmann::json &node) -> bool {
          return node.is_object() && node.contains("_type") &&
                 node["_type"] == "Subscript" && node.contains("value") &&
                 node["value"].is_object() && node["value"].contains("id") &&
                 node["value"]["id"] == "Literal";
        };

        // Handle nested Literal (e.g., Literal[Literal["foo"]])
        if (is_literal_subscript_node(slice))
        {
          return get_type_from_annotation(slice, element);
        }
        // Handle Literal with single value (e.g., Literal["foo"] or Literal[NAME])
        if (slice["_type"] == "Constant" && slice.contains("value"))
        {
          // Bignum literal annotation: tagged Constants carry a null value
          // (issue #4642). The literal is still an int — do not let
          // infer_literal_type misclassify it as None.
          if (slice.contains("_bigint"))
            return long_long_int_type();
          typet result = infer_literal_type(slice["value"]);
          if (!result.is_empty())
            return result;
        }
        else if (slice["_type"] == "Name")
        {
          nlohmann::json resolved_value = resolve_to_constant(slice);
          if (!resolved_value.is_null())
          {
            typet result = infer_literal_type(resolved_value);
            if (!result.is_empty())
              return result;
          }
          if (slice.contains("id"))
          {
            throw std::runtime_error(
              "Literal annotation references variable '" +
              slice["id"].get<std::string>() +
              "' which could not be resolved to a constant value.");
          }
          throw std::runtime_error(
            "Literal annotation references variable which could not be "
            "resolved to a constant value.");
        }
        // Handle Literal with multiple values
        else if (slice["_type"] == "Tuple" && slice.contains("elts"))
        {
          const auto &elts = slice["elts"];
          if (elts.empty())
            throw std::runtime_error("Empty Literal tuple is not supported.");

          TypeFlags type_flags;
          bool has_string = false;
          bool has_none = false;

          for (size_t i = 0; i < elts.size(); ++i)
          {
            const auto &elem = elts[i];
            // Handle nested Literal in tuple
            if (is_literal_subscript_node(elem))
            {
              typet nested_type = get_type_from_annotation(elem, element);
              update_type_flags(nested_type, type_flags, has_string, has_none);
              continue;
            }
            // Try to resolve element to constant
            nlohmann::json resolved_value = resolve_to_constant(elem);
            if (resolved_value.is_null())
            {
              std::string error_msg =
                "Literal tuple element at index " + std::to_string(i);
              if (
                elem.is_object() && elem.contains("_type") &&
                elem["_type"] == "Name" && elem.contains("id"))
                error_msg +=
                  " references variable '" + elem["id"].get<std::string>() +
                  "' which could not be resolved to a constant value.";
              else
                error_msg += " is not a constant value.";
              throw std::runtime_error(error_msg);
            }
            typet elem_type = infer_literal_type(resolved_value);
            if (elem_type.is_empty())
            {
              throw std::runtime_error(
                "Unsupported literal type at index " + std::to_string(i) +
                " in Literal tuple.");
            }
            update_type_flags(elem_type, type_flags, has_string, has_none);
          }
          // Determine the widest type: string > float > int > bool > None
          if (has_string)
          {
            if (
              type_flags.has_float || type_flags.has_int || type_flags.has_bool)
              return pointer_type(); // Mixed string and numeric
            return gen_pointer_type(char_type());
          }
          if (type_flags.has_float)
            return double_type();
          if (type_flags.has_int)
            return long_long_int_type();
          if (type_flags.has_bool)
            return bool_type();
          if (has_none)
            return none_type();
          throw std::runtime_error(
            "Could not determine type for Literal tuple.");
        }
      }
      throw std::runtime_error(
        "Unsupported (or malformed) Literal type annotation. "
        "We currently support constant values (string, int, bool, float, or "
        "None).");
    }

    // Handle Optional[T] - extract the inner type T
    if (
      annotation_node.contains("value") &&
      annotation_node["value"].is_object() &&
      annotation_node["value"].contains("id") &&
      annotation_node["value"]["id"] == "Optional")
    {
      if (
        annotation_node.contains("slice") &&
        annotation_node["slice"].is_object())
      {
        const auto &slice = annotation_node["slice"];
        std::string inner_type;

        // Optional[List]: slice is a Name node
        if (slice.contains("id"))
          inner_type = slice["id"].get<std::string>();
        // Optional["List"]: forward reference string; slice is a Constant node
        else if (
          slice.contains("_type") && slice["_type"] == "Constant" &&
          slice.contains("value") && slice["value"].is_string())
          inner_type = slice["value"].get<std::string>();

        if (!inner_type.empty())
        {
          typet base_type;
          // If inner_type is a user-defined class, return its struct symbol type
          // rather than a built-in type (e.g., avoid mapping "List" to PyListObj).
          if (json_utils::is_class(inner_type, *ast_json))
            base_type = symbol_typet("tag-" + inner_type);
          else
            base_type = type_handler_.get_typet(inner_type);
          // Always use pointer type for Optional to properly represent None
          return gen_pointer_type(base_type);
        }
      }
    }

    // Handle external module types in Subscript (e.g., re.Match[str])
    // Treat as opaque/any type
    if (
      annotation_node.contains("value") &&
      annotation_node["value"].is_object() &&
      annotation_node["value"].contains("_type") &&
      annotation_node["value"]["_type"] == "Attribute")
    {
      return any_type();
    }

    return type_handler_.get_list_type(element);
  }
  else if (annotation_node["_type"] == "BinOp")
  {
    // Handle union types such as str | None (PEP 604 syntax)
    std::string inner_type = extract_non_none_type(annotation_node);

    // Special handling for Literal types in unions
    if (inner_type == "__LITERAL__")
    {
      // Find the Literal node and recursively process it
      const auto &left = annotation_node["left"];
      const auto &right = annotation_node["right"];

      // Helper to check if a node is a Literal subscript
      auto is_literal_subscript = [](const nlohmann::json &node) -> bool {
        return node.contains("_type") && node["_type"] == "Subscript" &&
               node.contains("value") && node["value"].is_object() &&
               node["value"].contains("id") && node["value"]["id"] == "Literal";
      };

      const auto &literal_node = is_literal_subscript(left) ? left : right;

      return get_type_from_annotation(literal_node, element);
    }

    // Special handling for external module types (e.g., re.Match[str] | None)
    // Treat them as opaque pointers (any_type)
    if (inner_type == "__EXTERNAL_TYPE__")
    {
      return any_type();
    }

    if (inner_type.empty())
    {
      // All types were None or couldn't be extracted - use any_type (void*)
      return any_type();
    }

    // Count the number of distinct type names in the union
    std::set<std::string> type_names;
    std::function<void(const nlohmann::json &)> collect_types;
    bool contains_none = false;
    collect_types = [&](const nlohmann::json &node) {
      // Guard: only process objects
      if (!node.is_object())
        return;

      if (
        node.contains("_type") && node["_type"] == "Constant" &&
        node.contains("value") && node["value"].is_null())
      {
        // This is None, skip it
        contains_none = true;
        return;
      }
      if (node.contains("id"))
        type_names.insert(node["id"].get<std::string>());
      // Handle Attribute nodes (e.g., re.Match in re.Match[str])
      if (
        node.contains("_type") && node["_type"] == "Attribute" &&
        node.contains("attr"))
        type_names.insert(node["attr"].get<std::string>());
      // Handle Subscript nodes (e.g., re.Match[str], List[int])
      if (node.contains("_type") && node["_type"] == "Subscript")
      {
        if (node.contains("value") && node["value"].is_object())
        {
          const auto &value_node = node["value"];
          if (value_node.contains("id"))
            type_names.insert(value_node["id"].get<std::string>());
          else if (
            value_node.contains("_type") &&
            value_node["_type"] == "Attribute" && value_node.contains("attr"))
            type_names.insert(value_node["attr"].get<std::string>());
        }
      }
      if (node.contains("_type") && node["_type"] == "BinOp")
      {
        if (node.contains("left"))
          collect_types(node["left"]);
        if (node.contains("right"))
          collect_types(node["right"]);
      }
    };
    collect_types(annotation_node);

    // If we have multiple types, treat as untyped pointer
    // This preserves the original behavior for type checking
    if (type_names.size() > 1 && contains_none)
      return gen_pointer_type(char_type());

    // Treat T | ... | None as Optional[T]
    typet base_type = type_handler_.get_typet(inner_type);

    // Single type + None: use Optional wrapper for primitives only
    if (
      base_type == long_long_int_type() || base_type == long_long_uint_type() ||
      base_type == double_type() || base_type == bool_type())
    {
      return type_handler_.build_optional_type(base_type);
    }

    // List types are already pointers
    if (base_type == type_handler_.get_list_type())
      return base_type;

    // For other types (e.g., classes, strings), use pointer type
    return gen_pointer_type(base_type);
  }
  else if (
    annotation_node["_type"] == "Constant" || annotation_node["_type"] == "Str")
  {
    // Handle None annotation: Constant with null value
    if (annotation_node["value"].is_null())
      return none_type();

    // Handle string annotations like "CoordinateData | None" (forward references)
    std::string type_string = annotation_node["value"].get<std::string>();
    type_string = type_utils::remove_quotes(type_string);
    // Support PEP 604 unions inside string annotations: "T | None"
    if (type_string.find('|') != std::string::npos)
    {
      // Split by '|' and trim whitespace
      auto trim_ws = [](std::string s) -> std::string {
        const auto not_space = [](unsigned char ch) {
          return !std::isspace(ch);
        };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
        return s;
      };
      std::vector<std::string> parts;
      std::string current;
      for (char c : type_string)
      {
        if (c == '|')
        {
          parts.push_back(trim_ws(current));
          current.clear();
        }
        else
        {
          current.push_back(c);
        }
      }
      parts.push_back(trim_ws(current));

      // Remove None/NoneType and keep track of non-none members
      std::string base_type_name;
      size_t non_none_count = 0;
      bool contains_none = false;
      for (const auto &p : parts)
      {
        if (p == "None" || p == "NoneType")
        {
          contains_none = true;
          continue;
        }
        if (base_type_name.empty())
          base_type_name = p;
        ++non_none_count;
      }

      if (base_type_name.empty())
        return any_type();

      // If there are multiple non-None members, fall back conservatively
      if (non_none_count > 1)
      {
        if (contains_none)
          return gen_pointer_type(char_type());
        return any_type();
      }

      typet base_type = type_handler_.get_typet(base_type_name);

      // Single type + None: use Optional wrapper for primitives only
      if (
        contains_none &&
        (base_type == long_long_int_type() ||
         base_type == long_long_uint_type() || base_type == double_type() ||
         base_type == bool_type()))
      {
        return type_handler_.build_optional_type(base_type);
      }

      if (base_type == type_handler_.get_list_type())
        return base_type;

      if (contains_none)
        return gen_pointer_type(base_type);

      return base_type;
    }

    return type_handler_.get_typet(type_string);
  }
  else if (
    annotation_node["_type"] == "Attribute" && annotation_node.contains("attr"))
    return type_handler_.get_typet(annotation_node["attr"].get<std::string>());
  else if (annotation_node.contains("id"))
  {
    std::string type_id = annotation_node["id"].get<std::string>();
    if (type_id == "NoneType")
      return any_type();

    // Special handling for dict type — but only if not shadowed by a user class
    if (
      (type_id == "dict" || type_id == "Dict") &&
      !json_utils::is_class(type_id, *ast_json))
      return dict_handler_->get_dict_struct_type();

    // Special handling for list type — but only if not shadowed by a user class
    if (
      (type_id == "list" || type_id == "List") &&
      !json_utils::is_class(type_id, *ast_json))
      return type_handler_.get_list_type();

    return type_handler_.get_typet(type_id);
  }
  else
  {
    throw std::runtime_error(
      "Unsupported annotation type: " +
      annotation_node["_type"].get<std::string>());
  }
}

typet python_converter::infer_tuple_struct_from_value(
  const nlohmann::json &value_node,
  const std::unordered_map<std::string, nlohmann::json> &param_annotations)
{
  if (!value_node.is_object() || !value_node.contains("_type"))
    return empty_typet();

  if (value_node["_type"] != "Tuple" || !value_node.contains("elts"))
    return empty_typet();

  std::vector<typet> elem_types;
  elem_types.reserve(value_node["elts"].size());

  for (const auto &elt : value_node["elts"])
  {
    typet elem_type;
    const std::string elt_kind = elt.value("_type", "");

    if (elt_kind == "Constant" && elt.contains("value"))
    {
      const auto &v = elt["value"];
      if (v.is_boolean())
        elem_type = bool_type();
      else if (v.is_number_integer())
        elem_type = long_long_int_type();
      else if (v.is_number_float())
        elem_type = double_type();
      else if (v.is_string())
        elem_type = gen_pointer_type(char_type());
    }
    else if (elt_kind == "Name" && elt.contains("id"))
    {
      const std::string name = elt["id"].get<std::string>();
      auto it = param_annotations.find(name);
      if (it != param_annotations.end())
        elem_type = get_type_from_annotation(it->second, elt);
    }

    if (elem_type.is_nil() || elem_type.id().as_string().empty())
      elem_type = any_type();

    elem_types.push_back(elem_type);
  }

  return tuple_handler_->create_tuple_struct_type(elem_types);
}

exprt python_converter::extract_type_from_boolean_op(const exprt &bool_op)
{
  // Only OR and AND are special
  if (!bool_op.is_and() && !bool_op.is_or())
    return gen_zero(bool_op.type());

  // Let's try to be smart and guess the type;
  // In the future this could be trivial with an Python Obj struct
  // 1. If there are no non-null constants, then guess any.
  // 2. If there is only one type of constant, then guess it.
  // 3. If there is more than one type of constant, then abort.

  typet found_type = empty_typet();
  assert(found_type.is_empty());

  for (exprt e : bool_op.operands())
  {
    // First, try to solve the underlying type...
    if (!e.is_constant() && !e.is_symbol())
      e = extract_type_from_boolean_op(e);

    typet operand_type = e.type();
    if (operand_type.is_empty() || operand_type.is_bool())
      continue;

    // Arrays are special, they have a length property which we don't care about right now
    if (operand_type.is_array())
      return gen_zero(any_type());

    if (found_type.is_empty())
      found_type = operand_type;
    else if (found_type != operand_type)
    {
      log_warning(
        "Boolean expression with more than one constant type; "
        "falling back to Any");
      return gen_zero(any_type());
    }
  }

  return found_type.is_empty() ? gen_zero(any_type()) : gen_zero(found_type);
}
