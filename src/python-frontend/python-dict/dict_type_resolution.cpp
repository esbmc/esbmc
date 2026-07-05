#include "python_dict_internal.h"

using namespace python_expr;

void python_dict_handler::resolve_dict_subscript_types(
  const nlohmann::json &left,
  const nlohmann::json &right,
  exprt &lhs,
  exprt &rhs)
{
  bool lhs_is_dict_subscript = type_utils::is_dict_subscript(left);
  bool rhs_is_dict_subscript = type_utils::is_dict_subscript(right);

  bool lhs_is_ptr = lhs.type().is_pointer();
  bool rhs_is_ptr = rhs.type().is_pointer();

  auto is_primitive_type = [](const typet &t) {
    return t.is_signedbv() || t.is_unsignedbv() || t.is_bool() ||
           t.is_floatbv();
  };

  bool lhs_is_primitive = is_primitive_type(lhs.type());
  bool rhs_is_primitive = is_primitive_type(rhs.type());

  // Case 1: LHS is dict subscript (returning pointer) and RHS is primitive
  if (lhs_is_dict_subscript && lhs_is_ptr && rhs_is_primitive)
  {
    exprt dict_expr = converter_.get_expr(left["value"]);
    if (dict_expr.type().is_struct() && is_dict_type(dict_expr.type()))
    {
      lhs = handle_dict_subscript(dict_expr, left["slice"], rhs.type());
      // Dereference the pointer to get the actual value
      if (lhs.type().is_pointer())
        lhs = build_dereference(lhs, lhs.type().subtype());
    }
  }

  // Case 2: RHS is dict subscript (returning pointer) and LHS is primitive
  if (rhs_is_dict_subscript && rhs_is_ptr && lhs_is_primitive)
  {
    exprt dict_expr = converter_.get_expr(right["value"]);
    if (dict_expr.type().is_struct() && is_dict_type(dict_expr.type()))
    {
      rhs = handle_dict_subscript(dict_expr, right["slice"], lhs.type());
      // Dereference the pointer to get the actual value
      if (rhs.type().is_pointer())
        rhs = build_dereference(rhs, rhs.type().subtype());
    }
  }

  // Case 3: Both sides are dict subscripts (returning pointers)
  // Default to long_int_type for dict-to-dict comparisons
  if (
    lhs_is_dict_subscript && rhs_is_dict_subscript && lhs_is_ptr && rhs_is_ptr)
  {
    typet default_type = long_int_type();

    exprt lhs_dict = converter_.get_expr(left["value"]);
    if (lhs_dict.type().is_struct() && is_dict_type(lhs_dict.type()))
    {
      lhs = handle_dict_subscript(lhs_dict, left["slice"], default_type);
      // Dereference the pointer to get the actual value
      if (lhs.type().is_pointer())
        lhs = build_dereference(lhs, lhs.type().subtype());
    }

    exprt rhs_dict = converter_.get_expr(right["value"]);
    if (rhs_dict.type().is_struct() && is_dict_type(rhs_dict.type()))
    {
      rhs = handle_dict_subscript(rhs_dict, right["slice"], default_type);
      // Dereference the pointer to get the actual value
      if (rhs.type().is_pointer())
      {
        rhs = build_dereference(rhs, rhs.type().subtype());
      }
    }
  }
}

typet python_dict_handler::get_dict_value_type_from_annotation(
  const nlohmann::json &annotation_node)
{
  // Get the slice which contains the key and value types
  if (!annotation_node.contains("slice"))
    return empty_typet();

  const auto &slice = annotation_node["slice"];

  // For dict[K, V], slice is a Tuple with two elements
  if (
    slice.contains("_type") && slice["_type"] == "Tuple" &&
    slice.contains("elts") && slice["elts"].size() >= 2)
  {
    // Return the value type (second element of the tuple)
    const auto &value_type_node = slice["elts"][1];
    return converter_.get_type_from_annotation(
      value_type_node, annotation_node);
  }

  return empty_typet();
}

typet python_dict_handler::resolve_expected_type_for_dict_subscript(
  const exprt &dict_expr)
{
  if (dict_expr.id() == "member")
  {
    const auto &member = to_member_expr(dict_expr);
    typet class_type = member.struct_op().type();
    namespacet ns(symbol_table_);

    if (class_type.is_pointer())
      class_type = ns.follow(class_type.subtype());
    else
      class_type = ns.follow(class_type);

    assert(class_type.is_struct());

    // For self.data[key], look up the dict value type from the class
    // annotation before falling back to the usual symbol-based path.
    const std::string class_name = converter_.extract_class_name_from_tag(
      to_struct_type(class_type).tag().as_string());
    const nlohmann::json class_node =
      json_utils::find_class(converter_.get_ast_json()["body"], class_name);

    if (!class_node.empty() && class_node.contains("body"))
    {
      for (const auto &class_elem : class_node["body"])
      {
        if (
          class_elem.contains("_type") &&
          class_elem["_type"] == "FunctionDef" && class_elem.contains("name") &&
          class_elem["name"] == "__init__" && class_elem.contains("body"))
        {
          for (const auto &stmt : class_elem["body"])
          {
            if (
              stmt.contains("_type") && stmt["_type"] == "AnnAssign" &&
              stmt.contains("target") && stmt["target"].is_object() &&
              stmt["target"].contains("_type") &&
              stmt["target"]["_type"] == "Attribute" &&
              stmt["target"].contains("value") &&
              stmt["target"]["value"].is_object() &&
              stmt["target"]["value"].contains("id") &&
              stmt["target"]["value"]["id"] == "self" &&
              stmt["target"].contains("attr") &&
              stmt["target"]["attr"].get<std::string>() ==
                member.get_component_name().as_string() &&
              stmt.contains("annotation"))
            {
              typet result =
                get_dict_value_type_from_annotation(stmt["annotation"]);
              if (!result.is_nil() && !result.is_empty())
                return result;
            }
          }
        }
      }
    }
  }

  // Only works if dict_expr is a symbol (variable reference)
  if (!dict_expr.is_symbol())
    return empty_typet();

  const symbolt *sym = symbol_table_.find_symbol(dict_expr.identifier());
  if (!sym)
    return empty_typet();

  // Look up the variable's declaration in the AST to get its annotation
  std::string var_name = sym->name.as_string();
  nlohmann::json var_decl = json_utils::find_var_decl(
    var_name, converter_.get_current_func_name(), converter_.get_ast_json());

  if (var_decl.empty())
    return empty_typet();

  // No annotation or bare `dict`:
  // peek at the literal's first value to infer the value type.
  const bool no_annotation =
    !var_decl.contains("annotation") || var_decl["annotation"].is_null();
  const bool bare_dict_annotation =
    !no_annotation && var_decl["annotation"].is_object() &&
    var_decl["annotation"].value("_type", std::string()) == "Name" &&
    var_decl["annotation"].value("id", std::string()) == "dict" &&
    !var_decl["annotation"].contains("slice");
  if (no_annotation || bare_dict_annotation)
  {
    if (
      var_decl.contains("value") && var_decl["value"].is_object() &&
      var_decl["value"].value("_type", std::string()) == "Dict" &&
      var_decl["value"].contains("values") &&
      var_decl["value"]["values"].is_array() &&
      !var_decl["value"]["values"].empty())
    {
      const std::string kind =
        var_decl["value"]["values"][0].value("_type", std::string());
      if (kind == "List")
        return type_handler_.get_list_type();
      if (kind == "Dict")
        return get_dict_struct_type();
    }

    // Dict comprehension `d = {k: v for ...}`: the variable carries no
    // annotation and its AST value is a DictComp, so the Dict-literal peek
    // above does not fire. Infer the subscript value type from the
    // comprehension's value expression. Without this the read falls through
    // to the char* default and returns the raw PyObject value pointer rather
    // than the stored scalar — a wrong verdict on `d[k]` reads, e.g. inside a
    // function body where no target type flows in (#5222).
    if (
      var_decl.contains("value") && var_decl["value"].is_object() &&
      var_decl["value"].value("_type", std::string()) == "DictComp" &&
      var_decl["value"].contains("value") &&
      var_decl["value"]["value"].is_object())
    {
      const auto &val_node = var_decl["value"]["value"];
      const std::string val_kind = val_node.value("_type", std::string());
      if (val_kind == "List")
        return type_handler_.get_list_type();
      if (val_kind == "Dict" || val_kind == "DictComp")
        return get_dict_struct_type();
      // Scalar literal value: classify exactly as the value is stored by the
      // comprehension (Python int -> int, float -> float, bool -> bool). Only
      // numeric/boolean literal constants are inferred; string, bytes and any
      // other value expression are left to the existing path, which already
      // resolves them to the char* default (unchanged behaviour).
      if (val_kind == "Constant" && val_node.contains("value"))
      {
        const auto &lit = val_node["value"];
        if (lit.is_boolean())
          return bool_type();
        if (lit.is_number_integer() || lit.is_number_unsigned())
          return type_handler_.get_typet("int", 0);
        if (lit.is_number_float())
          return type_handler_.get_typet("float", 0);
      }
    }
    // Empty literal `d = {}`:
    // scan the enclosing function's top-level statements and pick the most
    // recent `d[k] = Klass(...)` assignment as the subscript value type.
    // A later `d = {}` re-init resets the candidate so the next assignment
    // dominates — keeps the inferred type aligned with the live dict.
    // Without it, d[k] reads on an empty literal default to char*.
    if (
      var_decl.contains("value") && var_decl["value"].is_object() &&
      var_decl["value"].value("_type", std::string()) == "Dict" &&
      (!var_decl["value"].contains("values") ||
       !var_decl["value"]["values"].is_array() ||
       var_decl["value"]["values"].empty()))
    {
      std::vector<std::string> fn_path =
        json_utils::split_function_path(converter_.get_current_func_name());
      if (!fn_path.empty())
      {
        const auto &func_node =
          json_utils::find_function_by_path(converter_.get_ast_json(), fn_path);
        if (
          !func_node.empty() && func_node.contains("body") &&
          func_node["body"].is_array())
        {
          typet candidate;
          bool have_candidate = false;
          // Walk statements in document order, descending into For/While/If/
          // Try/With nested bodies — the d[k] = factory() assignments inserted
          // by the defaultdict preprocessor (see
          // preprocessor/loop_mixin.py:_lower_defaultdict_reads_in_expr) sit
          // inside the containing statement, which can be arbitrarily deep
          // inside nested loops (e.g. quixbugs/shortest_path_lengths).
          std::function<void(const nlohmann::json &)> visit_stmt;
          auto visit_body = [&](const nlohmann::json &body) {
            if (!body.is_array())
              return;
            for (const auto &s : body)
              visit_stmt(s);
          };
          visit_stmt = [&](const nlohmann::json &stmt) {
            if (!stmt.is_object() || !stmt.contains("_type"))
              return;
            const std::string &kind = stmt["_type"].get<std::string>();
            if (kind == "For" || kind == "AsyncFor" || kind == "While")
            {
              if (stmt.contains("body"))
                visit_body(stmt["body"]);
              if (stmt.contains("orelse"))
                visit_body(stmt["orelse"]);
              return;
            }
            if (kind == "If")
            {
              if (stmt.contains("body"))
                visit_body(stmt["body"]);
              if (stmt.contains("orelse"))
                visit_body(stmt["orelse"]);
              return;
            }
            if (kind == "With" || kind == "AsyncWith")
            {
              if (stmt.contains("body"))
                visit_body(stmt["body"]);
              return;
            }
            if (kind == "Try" || kind == "TryStar")
            {
              if (stmt.contains("body"))
                visit_body(stmt["body"]);
              if (stmt.contains("orelse"))
                visit_body(stmt["orelse"]);
              if (stmt.contains("finalbody"))
                visit_body(stmt["finalbody"]);
              if (stmt.contains("handlers") && stmt["handlers"].is_array())
                for (const auto &h : stmt["handlers"])
                  if (h.is_object() && h.contains("body"))
                    visit_body(h["body"]);
              return;
            }
            // Don't descend into nested FunctionDef/Lambda/ClassDef — their
            // assignments belong to a different scope.
            if (kind != "Assign")
              return;
            if (
              !stmt.contains("targets") || !stmt["targets"].is_array() ||
              stmt["targets"].empty())
              return;
            const auto &tgt = stmt["targets"][0];

            // `d = {}` re-init clears the candidate;
            // the next subscript assignment dominates.
            if (
              tgt.is_object() && tgt.value("_type", std::string()) == "Name" &&
              tgt.value("id", std::string()) == var_name &&
              stmt.contains("value") && stmt["value"].is_object() &&
              stmt["value"].value("_type", std::string()) == "Dict" &&
              (!stmt["value"].contains("values") ||
               !stmt["value"]["values"].is_array() ||
               stmt["value"]["values"].empty()))
            {
              have_candidate = false;
              return;
            }

            if (
              !tgt.is_object() ||
              tgt.value("_type", std::string()) != "Subscript" ||
              !tgt.contains("value") || !tgt["value"].is_object() ||
              tgt["value"].value("_type", std::string()) != "Name" ||
              tgt["value"].value("id", std::string()) != var_name ||
              !stmt.contains("value") || !stmt["value"].is_object())
              return;
            const auto &rhs = stmt["value"];

            // Literal RHS: pick the value type from the constant's kind.
            // Covers `d[k] = 5`, `d[k] = 0.0`, `d[k] = True`, `d[k] = "x"`.
            if (
              rhs.value("_type", std::string()) == "Constant" &&
              rhs.contains("value"))
            {
              const auto &lit = rhs["value"];
              if (lit.is_boolean())
              {
                candidate = bool_type();
                have_candidate = true;
              }
              else if (lit.is_number_integer() || lit.is_number_unsigned())
              {
                candidate = type_handler_.get_typet("int", 0);
                have_candidate = true;
              }
              else if (lit.is_number_float())
              {
                candidate = type_handler_.get_typet("float", 0);
                have_candidate = true;
              }
              else if (lit.is_string())
              {
                candidate = gen_pointer_type(char_type());
                have_candidate = true;
              }
              return;
            }

            if (
              rhs.value("_type", std::string()) == "Call" &&
              rhs.contains("func") && rhs["func"].is_object())
            {
              const auto &func = rhs["func"];

              // Lambda factory: `defaultdict(lambda: float('inf'))` is
              // lowered by the preprocessor to `d[k] = (<lambda>)()`. Inspect
              // the lambda body to determine the value type.
              if (
                func.value("_type", std::string()) == "Lambda" &&
                func.contains("body") && func["body"].is_object())
              {
                const auto &body = func["body"];
                if (
                  body.value("_type", std::string()) == "Constant" &&
                  body.contains("value"))
                {
                  const auto &lit = body["value"];
                  if (lit.is_boolean())
                  {
                    candidate = bool_type();
                    have_candidate = true;
                  }
                  else if (lit.is_number_integer() || lit.is_number_unsigned())
                  {
                    candidate = type_handler_.get_typet("int", 0);
                    have_candidate = true;
                  }
                  else if (lit.is_number_float())
                  {
                    candidate = type_handler_.get_typet("float", 0);
                    have_candidate = true;
                  }
                }
                else if (
                  body.value("_type", std::string()) == "Call" &&
                  body.contains("func") && body["func"].is_object() &&
                  body["func"].value("_type", std::string()) == "Name" &&
                  body["func"].contains("id"))
                {
                  const std::string body_callee =
                    body["func"]["id"].get<std::string>();
                  // `lambda: float('inf')`, `lambda: int()`, ...
                  if (body_callee == "float")
                  {
                    candidate = type_handler_.get_typet("float", 0);
                    have_candidate = true;
                  }
                  else if (body_callee == "int")
                  {
                    candidate = type_handler_.get_typet("int", 0);
                    have_candidate = true;
                  }
                  else if (body_callee == "bool")
                  {
                    candidate = bool_type();
                    have_candidate = true;
                  }
                  else if (body_callee == "str")
                  {
                    candidate = gen_pointer_type(char_type());
                    have_candidate = true;
                  }
                }
                return;
              }

              if (
                !func.contains("_type") || func["_type"] != "Name" ||
                !func.contains("id"))
                return;

              const std::string callee = func["id"].get<std::string>();

              // Built-in type constructors: `int()`, `float()`, `bool()`,
              // `str()` — emitted by the preprocessor when lowering
              // `defaultdict(<builtin>)` reads (see
              // preprocessor/loop_mixin.py:_make_defaultdict_missing_check).
              if (
                callee == "int" || callee == "float" || callee == "bool" ||
                callee == "str")
              {
                if (callee == "str")
                  candidate = gen_pointer_type(char_type());
                else
                  candidate = type_handler_.get_typet(callee, 0);
                have_candidate = true;
                return;
              }

              if (json_utils::is_class(callee, converter_.get_ast_json()))
              {
                candidate = symbol_typet("tag-" + callee);
                have_candidate = true;
              }
            }
          };
          visit_body(func_node["body"]);
          if (have_candidate)
            return candidate;
        }
      }
    }
    if (no_annotation)
      return empty_typet();
  }

  // Check if the annotation is just a simple type (e.g., "dict")
  // If so, try to get the full type from the RHS (function call)
  if (
    var_decl["annotation"]["_type"] == "Name" &&
    var_decl["annotation"]["id"] == "dict" && var_decl.contains("value") &&
    var_decl["value"]["_type"] == "Call")
  {
    const auto &call_node = var_decl["value"];
    if (call_node["func"]["_type"] == "Name")
    {
      std::string func_name = call_node["func"]["id"].get<std::string>();

      // Find the function definition to get its return type annotation
      nlohmann::json func_def =
        json_utils::find_function(converter_.get_ast_json()["body"], func_name);

      if (
        !func_def.empty() && func_def.contains("returns") &&
        !func_def["returns"].is_null())
      {
        // Extract the value type from the function's return annotation
        typet result = get_dict_value_type_from_annotation(func_def["returns"]);
        if (!result.is_nil())
          return result;
      }
    }
  }

  // Extract the value type from the dict annotation
  return get_dict_value_type_from_annotation(var_decl["annotation"]);
}

typet python_dict_handler::get_dict_key_type_from_annotation(
  const nlohmann::json &annotation_node)
{
  if (!annotation_node.contains("slice"))
    return empty_typet();

  const auto &slice = annotation_node["slice"];
  if (
    slice.contains("_type") && slice["_type"] == "Tuple" &&
    slice.contains("elts") && slice["elts"].size() >= 2)
  {
    return converter_.get_type_from_annotation(
      slice["elts"][0], annotation_node);
  }

  return empty_typet();
}
