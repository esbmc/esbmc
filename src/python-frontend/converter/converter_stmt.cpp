#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/convert_float_literal.h>
#include <python-frontend/function_call_expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_consteval.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_dict_handler.h>
#include <python-frontend/python_annotation.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/python_lambda.h>
#include <python-frontend/python_list.h>
#include <python-frontend/python_typechecking.h>
#include <python-frontend/python_math.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/string_handler.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/encoding.h>
#include <util/expr_util.h>
#include <util/irep.h>
#include <util/message.h>
#include <util/python_types.h>
#include <util/std_code.h>
#include <util/string_constant.h>
#include <util/symbolic_types.h>

#include <algorithm>
#include <stdexcept>

using namespace json_utils;

void python_converter::adjust_statement_types(exprt &lhs, exprt &rhs) const
{
  typet &lhs_type = lhs.type();
  typet &rhs_type = rhs.type();

  // Case 1: Promote RHS integer constant to float if LHS expects a float
  if (
    lhs_type.is_floatbv() && rhs.is_constant() &&
    type_utils::is_integer_type(rhs_type))
  {
    try
    {
      // Convert binary string value to integer
      BigInt value(
        binary2integer(rhs.value().as_string(), rhs_type.is_signedbv()));

      // Create a float literal string (e.g., "42.0")
      std::string rhs_float = std::to_string(value.to_int64()) + ".0";

      // Replace RHS with a float expression
      convert_float_literal(rhs_float, rhs);

      // Update the symbol table entry for RHS if needed
      update_symbol(rhs);
    }
    catch (const std::exception &e)
    {
      log_error(
        "adjust_statement_types: Failed to promote integer to float: {}",
        e.what());
    }
  }
  // Case 2: For Python assignments, if RHS is float but LHS is integer,
  // promote LHS to float to maintain Python's dynamic typing semantics
  else if (rhs_type.is_floatbv() && type_utils::is_integer_type(lhs_type))
  {
    // Update LHS variable type to match RHS float type
    lhs.type() = rhs_type;

    // Update symbol table if LHS is a symbol
    if (lhs.is_symbol())
      update_symbol(lhs);
  }
  // Case 3: Handles Python's / operator by promoting operands to floats
  // to ensure floating-point division, preventing division by zero, and
  // setting the result type to floatbv.
  else if (
    (rhs.id() == "/" || rhs.id() == "ieee_div") && rhs.operands().size() == 2)
  {
    auto &ops = rhs.operands();
    exprt &lhs_op = ops[0];
    exprt &rhs_op = ops[1];

    // Promote both operands to IEEE float (double precision) to match Python semantics
    const typet float_type =
      double_type(); // Python default float is double-precision

    // Handle constant operands
    if (lhs_op.is_constant() && type_utils::is_integer_type(lhs_op.type()))
      math_handler_.promote_int_to_float(lhs_op, float_type);
    // For non-constant operands, create explicit typecast
    else if (!lhs_op.type().is_floatbv())
      lhs_op = typecast_exprt(lhs_op, float_type);

    if (rhs_op.is_constant() && type_utils::is_integer_type(rhs_op.type()))
      math_handler_.promote_int_to_float(rhs_op, float_type);
    else if (!rhs_op.type().is_floatbv())
      rhs_op = typecast_exprt(rhs_op, float_type);

    // For in-place division (like x /= y), ensure LHS variable is promoted to float
    lhs.type() = float_type;
    if (lhs.is_symbol())
      update_symbol(lhs);

    // Update the division expression type and operator ID
    rhs.type() = float_type;
    rhs.id(python_frontend::map_operator("div", float_type));
  }
  // Case 4: Special case for IEEE division results - ensure LHS is float
  else if (rhs.id() == "ieee_div" && !lhs_type.is_floatbv())
  {
    // For any IEEE division result assigned to an integer variable,
    // promote the variable to float to avoid truncation
    const typet float_type = double_type();
    lhs.type() = float_type;

    if (lhs.is_symbol())
      update_symbol(lhs);

    // Ensure RHS type is also float
    if (!rhs_type.is_floatbv())
      rhs.type() = float_type;
  }
  // Case 5 (P19): Promote real RHS to complex when LHS is complex.
  // Must come BEFORE the width-alignment case: a complex struct is 128-bit
  // while a scalar float is 64-bit, so width alignment would otherwise fire
  // first and corrupt the float by assigning struct type to it.
  // Handles: z = 1.0, z = n, z = True where z is declared as complex.
  // Note: is_bool() must be explicit since is_integer_type() excludes bool.
  else if (
    is_complex_type(lhs_type) && !is_complex_type(rhs_type) &&
    (rhs_type.is_floatbv() || type_utils::is_integer_type(rhs_type) ||
     rhs_type.is_bool()))
  {
    rhs = promote_to_complex(rhs);
  }
  // Case 6: Align bit-widths between LHS and RHS if they differ
  else if (lhs_type.width() != rhs_type.width())
  {
    try
    {
      const int lhs_width = type_handler_.get_type_width(lhs_type);
      const int rhs_width = type_handler_.get_type_width(rhs_type);

      if (lhs_width > rhs_width)
      {
        // Promote RHS to LHS type
        rhs_type = lhs_type;
        if (rhs.is_symbol())
          update_symbol(rhs);
      }
      else
      {
        // Promote LHS to RHS type
        lhs_type = rhs_type;
        if (lhs.is_symbol())
          update_symbol(lhs);
      }
    }
    catch (const std::exception &e)
    {
      log_error(
        "adjust_statement_types: Failed to parse type widths: {}", e.what());
    }
  }
}
std::pair<std::string, typet>
python_converter::extract_type_info(const nlohmann::json &var_node)
{
  typet var_typet;
  std::string var_type_str("");

  if (var_node.contains("annotation") && !var_node["annotation"].is_null())
  {
    // Get type from annotation node
    size_t type_size = get_type_size(var_node);
    const auto &ann = var_node["annotation"];

    if (ann.contains("_type") && ann["_type"] == "Subscript")
    {
      if (ann.contains("value") && ann["value"].contains("id"))
        var_type_str = ann["value"]["id"];
      // Handle annotations written as ``typing.Tuple[...]`` (or any aliased
      // typing module): the Subscript base is an Attribute, not a Name.
      else if (
        ann.contains("value") && ann["value"].contains("_type") &&
        ann["value"]["_type"] == "Attribute" && ann["value"].contains("attr"))
        var_type_str = ann["value"]["attr"];

      // Preserve concrete tuple element types for Tuple[...] annotations
      // instead of resolving to the typing.Tuple class type.
      if (var_type_str == "Tuple" || var_type_str == "tuple")
      {
        var_typet = get_type_from_annotation(ann, var_node);
        return {var_type_str, var_typet};
      }
    }
    else if (
      ann.contains("_type") && ann["_type"] == "Attribute" &&
      ann.contains("attr"))
      var_type_str = ann["attr"];
    else if (ann.contains("id"))
      var_type_str = var_node["annotation"]["id"];
    else if (ann.contains("_type") && ann["_type"] == "BinOp")
    {
      // Handle union types (e.g., re.Match[str] | None)
      // Use get_type_from_annotation which has proper union handling
      var_typet = get_type_from_annotation(ann, var_node);
      return {var_type_str, var_typet};
    }

    if (var_type_str.empty())
      return {var_type_str, var_typet};

    // User-defined classes named "list"/"List" or "dict"/"Dict" take priority
    // over the built-in types when used as a plain Name annotation.
    if (
      (var_type_str == "dict" || var_type_str == "Dict") &&
      !json_utils::is_class(var_type_str, *ast_json))
      var_typet = dict_handler_->get_dict_struct_type();
    else if (
      (var_type_str == "list" || var_type_str == "List") &&
      !json_utils::is_class(var_type_str, *ast_json))
      var_typet = type_handler_.get_list_type();
    else
      var_typet = type_handler_.get_typet(var_type_str, type_size);
  }

  return {var_type_str, var_typet};
}

exprt python_converter::create_lhs_expression(
  const nlohmann::json &target,
  symbolt *lhs_symbol,
  const locationt &location)
{
  exprt lhs;
  const auto &target_type = target["_type"];

  if (target_type == "Attribute" || target_type == "Subscript")
  {
    is_converting_lhs = true;
    lhs = get_expr(target);
    is_converting_lhs = false;
  }
  else
    lhs = symbol_expr(*lhs_symbol);

  lhs.location() = location;
  return lhs;
}

void python_converter::handle_assignment_type_adjustments(
  symbolt *lhs_symbol,
  exprt &lhs,
  exprt &rhs,
  const std::string &lhs_type,
  const nlohmann::json &ast_node,
  bool is_ctor_call)
{
  const bool has_annotation =
    ast_node.contains("annotation") && !ast_node["annotation"].is_null();

  // For subscript targets (e.g. dp[i] = v).
  // The rhs writes an element, not the container.
  // Don't rewrite lhs_symbol's type.
  auto is_subscript_target = [](const nlohmann::json &t) {
    return t.is_object() && t.value("_type", "") == "Subscript";
  };
  const bool target_is_subscript =
    (ast_node.contains("targets") && ast_node["targets"].is_array() &&
     !ast_node["targets"].empty() &&
     is_subscript_target(ast_node["targets"][0])) ||
    (ast_node.contains("target") && is_subscript_target(ast_node["target"]));
  if (target_is_subscript)
    return;

  // Handle assignment of function to function pointer variable
  if (
    lhs.type().is_pointer() && lhs.type().subtype().is_code() &&
    rhs.type().is_code() && rhs.is_symbol())
  {
    rhs = address_of_exprt(rhs);
    if (lhs_symbol && !is_ctor_call)
      lhs_symbol->value = rhs;
    return;
  }

  // When a variable is assigned a function pointer returned from a
  // higher-order lambda call (e.g. `inner = outer(5)` or `inner:int = outer(5)`),
  // override any incorrect annotation (void*, int, …) with the concrete
  // function pointer type so the subsequent indirect call resolves correctly
  // instead of crashing in to_code_type.
  if (
    lhs_symbol && !is_ctor_call && rhs.type().is_pointer() &&
    rhs.type().subtype().is_code() &&
    !(lhs.type().is_pointer() && lhs.type().subtype().is_code()))
  {
    lhs_symbol->type = rhs.type();
    lhs.type() = rhs.type();
  }

  // Handle lambda assignments
  if (lambda_handler_->is_lambda_assignment(ast_node) && rhs.is_symbol())
  {
    lambda_handler_->handle_lambda_assignment(lhs_symbol, lhs, rhs);
    return;
  }
  // Handle tuple assignments with generic tuple annotation
  else if (
    lhs_symbol && lhs_symbol->type.id() == "empty" &&
    rhs.type().id() == "struct")
  {
    const struct_typet &rhs_struct = to_struct_type(rhs.type());

    // Check if RHS is a tuple (has tuple tag pattern)
    if (rhs_struct.tag().as_string().find("tag-tuple") == 0)
    {
      // Update symbol type from empty to concrete tuple type
      lhs_symbol->type = rhs.type();
      lhs.type() = rhs.type();
      lhs_symbol->value = rhs;
    }
  }
  else if (lhs_symbol)
  {
    // Handle explicit Any-typed annotation assignments
    // Only applies when the user explicitly wrote `from typing import Any`
    // and annotated `x: Any = value`.
    // Preprocessor-generated AnnAssign nodes
    // with Any annotation are excluded.
    if (
      ast_node.contains("_type") && ast_node["_type"] == "AnnAssign" &&
      !ast_node.value("_inferred_annotation", false) && has_annotation &&
      ast_node["annotation"].contains("id") &&
      ast_node["annotation"]["id"] == "Any" && lhs.type().is_pointer() &&
      [this]() {
        // Check if "from typing import Any" exists in the source file
        const auto &body = (*ast_json)["body"];
        for (const auto &stmt : body)
        {
          if (
            stmt.contains("_type") && stmt["_type"] == "ImportFrom" &&
            stmt.contains("module") && stmt["module"] == "typing" &&
            stmt.contains("names"))
          {
            for (const auto &name : stmt["names"])
            {
              if (name.contains("name") && name["name"] == "Any")
                return true;
            }
          }
        }
        return false;
      }())
    {
      if (rhs.type().is_array())
      {
        rhs = string_handler_.get_array_base_address(rhs);
        if (rhs.type() != lhs.type())
          rhs = typecast_exprt(rhs, lhs.type());
      }
      else if (!rhs.type().is_pointer() && !rhs.type().is_empty())
      {
        if (rhs.type().is_floatbv())
        {
          unsigned width =
            static_cast<const bv_typet &>(rhs.type()).get_width();
          exprt bitcast("bitcast", unsignedbv_typet(width));
          bitcast.copy_to_operands(rhs);
          rhs = bitcast;
        }
        rhs = typecast_exprt(rhs, lhs.type());
      }
      if (!rhs.type().is_empty() && !is_ctor_call)
        lhs_symbol->value = rhs;
      return;
    }
    // Handle string-to-string variable assignments
    if (lhs_type == "str" && rhs.is_symbol())
    {
      symbolt *rhs_symbol = symbol_table_.find_symbol(rhs.identifier());
      if (
        rhs_symbol && rhs_symbol->value.is_constant() &&
        rhs_symbol->value.type().is_array())
      {
        rhs = rhs_symbol->value;
        lhs_symbol->type = rhs.type();
        lhs.type() = rhs.type();
      }
    }
    // Array to pointer decay
    else if (lhs.type().id().empty() && rhs.type().is_array())
    {
      // TODO: This case is used to infer an unknown type.
      // Should we model it uniformly using char* ?
      const typet &element_type = to_array_type(rhs.type()).subtype();
      typet pointer_type = gen_pointer_type(element_type);
      lhs_symbol->type = pointer_type;
      lhs.type() = pointer_type;
      rhs = string_handler_.get_array_base_address(rhs);
    }
    else if (
      lhs.type().is_pointer() && rhs.type().is_array() &&
      lhs.type() != type_handler_.get_list_type())
    {
      // Array to pointer typecast
      // skip the list type until the list is moved to symex
      // TODO: remove list condition
      rhs = string_handler_.get_array_base_address(rhs);
    }
    // String and list type size adjustments
    else if (
      lhs_type == "str" || lhs_type == "chr" || lhs_type == "ord" ||
      lhs_type == "list" || rhs.type().is_array() ||
      rhs.type() == type_handler_.get_list_type())
    {
      if (!rhs.type().is_empty())
      {
        // Prevent type change from scalar (int/float/bool) to string/array
        // when a prior declaration exists with the scalar type, as this
        // creates a type inconsistency in the GOTO program.
        bool is_incompatible =
          rhs.type().is_array() && !lhs_symbol->type.is_array() &&
          !lhs_symbol->type.is_pointer() && !lhs_symbol->type.id().empty() &&
          !lhs_symbol->type.is_nil() &&
          lhs_symbol->type != type_handler_.get_list_type();
        if (!is_incompatible)
        {
          lhs_symbol->type = rhs.type();
          lhs.type() = rhs.type();
        }
      }
    }
    else if (rhs.type() == none_type())
    {
      // Adjust pointer_type() to pointer_typet(empty_typet())
      lhs_symbol->type = rhs.type();
      lhs.type() = rhs.type();
    }
    // No annotation or preprocessor-inferred Any: propagate rhs type to lhs.
    else if (
      (!has_annotation ||
       (ast_node.value("_inferred_annotation", false) &&
        ast_node["annotation"].value("id", std::string()) == "Any")) &&
      !rhs.type().is_empty() && lhs.type() != rhs.type() &&
      !rhs.type().is_code() &&
      !(rhs.type().is_pointer() && rhs.type().subtype().id() == "empty"))
    {
      lhs_symbol->type = rhs.type();
      lhs.type() = rhs.type();
    }

    if (!rhs.type().is_empty() && !is_ctor_call)
      lhs_symbol->value = rhs;
  }
}


void python_converter::handle_array_unpacking(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  exprt &rhs,
  codet &target_block)
{
  const auto &targets = target["elts"];

  for (size_t i = 0; i < targets.size(); i++)
  {
    if (targets[i]["_type"] != "Name")
    {
      throw std::runtime_error(
        "Array unpacking only supports simple names, not " +
        targets[i]["_type"].get<std::string>());
    }

    std::string var_name = targets[i]["id"].get<std::string>();
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(var_name);

    symbolt *var_symbol = find_symbol(var_sid.to_string());

    if (!var_symbol)
    {
      locationt loc = get_location_from_decl(targets[i]);
      typet elem_type = rhs.type().subtype();

      symbolt new_symbol = create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        elem_type);
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = symbol_table_.move_symbol_to_context(new_symbol);
    }

    // Create subscript: rhs[i]
    exprt index_expr = from_integer(i, size_type());
    index_exprt subscript(rhs, index_expr, rhs.type().subtype());

    code_assignt assign(symbol_expr(*var_symbol), subscript);
    assign.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }
}

void python_converter::handle_list_literal_unpacking(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  codet &target_block)
{
  const auto &value_node = ast_node["value"];
  const auto &elements = value_node["elts"];
  const auto &targets = target["elts"];

  // Find starred target (if any)
  int star_idx = -1;
  for (size_t i = 0; i < targets.size(); i++)
  {
    if (targets[i]["_type"] == "Starred")
    {
      star_idx = static_cast<int>(i);
      break;
    }
  }

  if (star_idx < 0)
  {
    // No starred target: strict size check
    if (elements.size() != targets.size())
    {
      throw std::runtime_error(
        "Cannot unpack list: expected " + std::to_string(targets.size()) +
        " values, got " + std::to_string(elements.size()));
    }
  }
  else
  {
    size_t non_star_count = targets.size() - 1;
    if (elements.size() < non_star_count)
    {
      throw std::runtime_error(
        "Cannot unpack list: not enough values (expected at least " +
        std::to_string(non_star_count) + ", got " +
        std::to_string(elements.size()) + ")");
    }
  }

  size_t before_star =
    (star_idx >= 0) ? static_cast<size_t>(star_idx) : targets.size();
  size_t after_star =
    (star_idx >= 0) ? targets.size() - static_cast<size_t>(star_idx) - 1 : 0;

  // Assign targets before the star
  for (size_t i = 0; i < before_star; i++)
  {
    if (targets[i]["_type"] != "Name")
    {
      throw std::runtime_error(
        "List unpacking only supports simple names, not " +
        targets[i]["_type"].get<std::string>());
    }

    std::string var_name = targets[i]["id"].get<std::string>();
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(var_name);

    symbolt *var_symbol = find_symbol(var_sid.to_string());

    is_converting_rhs = true;
    exprt elem_expr = get_expr(elements[i]);
    is_converting_rhs = false;

    if (!var_symbol)
    {
      locationt loc = get_location_from_decl(targets[i]);

      symbolt new_symbol = create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        elem_expr.type());
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = symbol_table_.move_symbol_to_context(new_symbol);
    }

    code_assignt assign(symbol_expr(*var_symbol), elem_expr);
    assign.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }

  // Handle starred target: collect remaining elements into a list
  if (star_idx >= 0)
  {
    const auto &starred_node = targets[static_cast<size_t>(star_idx)];
    const auto &star_value = starred_node["value"];

    if (star_value["_type"] != "Name")
    {
      throw std::runtime_error(
        "Starred unpacking only supports simple names, not " +
        star_value["_type"].get<std::string>());
    }

    // Build a synthetic list JSON node with the starred elements
    nlohmann::json star_list_node = value_node;
    star_list_node["_type"] = "List";
    star_list_node["elts"] = nlohmann::json::array();
    for (size_t j = before_star; j < elements.size() - after_star; j++)
      star_list_node["elts"].push_back(elements[j]);

    python_list star_list(*this, star_list_node);
    exprt list_expr = star_list.get();

    std::string var_name = star_value["id"].get<std::string>();
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(var_name);

    symbolt *var_symbol = find_symbol(var_sid.to_string());

    if (!var_symbol)
    {
      locationt loc = get_location_from_decl(star_value);

      symbolt new_symbol = create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        list_expr.type());
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = symbol_table_.move_symbol_to_context(new_symbol);
    }

    code_assignt assign(symbol_expr(*var_symbol), list_expr);
    assign.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }

  // Assign targets after the star (from the end)
  for (size_t i = 0; i < after_star; i++)
  {
    size_t target_idx = static_cast<size_t>(star_idx) + 1 + i;
    size_t elem_idx = elements.size() - after_star + i;

    if (targets[target_idx]["_type"] != "Name")
    {
      throw std::runtime_error(
        "List unpacking only supports simple names, not " +
        targets[target_idx]["_type"].get<std::string>());
    }

    std::string var_name = targets[target_idx]["id"].get<std::string>();
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(var_name);

    symbolt *var_symbol = find_symbol(var_sid.to_string());

    is_converting_rhs = true;
    exprt elem_expr = get_expr(elements[elem_idx]);
    is_converting_rhs = false;

    if (!var_symbol)
    {
      locationt loc = get_location_from_decl(targets[target_idx]);

      symbolt new_symbol = create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        elem_expr.type());
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = symbol_table_.move_symbol_to_context(new_symbol);
    }

    code_assignt assign(symbol_expr(*var_symbol), elem_expr);
    assign.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }
}

exprt python_converter::get_rhs_with_dict_resolution(
  const nlohmann::json &ast_node,
  const typet &target_type)
{
  if (!type_utils::is_dict_subscript(ast_node["value"]))
    return get_expr(ast_node["value"]);

  // Check if we need special dict subscript handling for typed variables
  // Including list type and dict type
  typet list_type = type_handler_.get_list_type();
  if (
    !target_type.is_signedbv() && !target_type.is_unsignedbv() &&
    !target_type.is_bool() && target_type != list_type &&
    !dict_handler_->is_dict_type(target_type))
  {
    return get_expr(ast_node["value"]);
  }

  exprt dict_expr = get_expr(ast_node["value"]["value"]);
  if (
    !dict_expr.type().is_struct() ||
    !dict_handler_->is_dict_type(dict_expr.type()))
    return get_expr(ast_node["value"]);

  return dict_handler_->handle_dict_subscript(
    dict_expr, ast_node["value"]["slice"], target_type);
}

std::string python_converter::infer_type_from_any_annotation(
  const nlohmann::json &ast_node,
  const std::string &lhs_type)
{
  if (ast_node["value"].is_null() || ast_node["value"]["_type"] != "Call")
    return lhs_type;

  const auto &func_node = ast_node["value"]["func"];
  std::string func_name;

  if (func_node["_type"] == "Name")
    func_name = func_node["id"].get<std::string>();
  else if (func_node["_type"] == "Attribute")
    func_name = func_node["attr"].get<std::string>();

  if (func_name.empty())
    return lhs_type;

  symbol_id func_sid(current_python_file, "", func_name);
  symbolt *func_symbol = symbol_table_.find_symbol(func_sid.to_string());

  // For method calls (e.g., b.f()), the method symbol is stored under the
  // class scope (py:file@C@ClassName@F@method), not the top-level scope.
  // Look up the object's class and retry the symbol lookup.
  if (
    !func_symbol && func_node["_type"] == "Attribute" &&
    func_node["value"].contains("id"))
  {
    const std::string obj_name = func_node["value"]["id"].get<std::string>();
    symbol_id obj_sid = create_symbol_id();
    obj_sid.set_object(obj_name);
    const symbolt *obj_sym = symbol_table_.find_symbol(obj_sid.to_string());
    if (obj_sym)
    {
      typet obj_type = ns.follow(obj_sym->type);
      std::string class_name;
      if (obj_type.is_struct())
        class_name = to_struct_type(obj_type).tag().as_string();
      if (class_name.rfind("tag-", 0) == 0)
        class_name = class_name.substr(4);
      if (!class_name.empty())
      {
        symbol_id method_sid(current_python_file, class_name, func_name);
        func_symbol = symbol_table_.find_symbol(method_sid.to_string());
      }
    }
  }

  if (func_symbol && func_symbol->type.is_code())
  {
    const code_typet &func_type = to_code_type(func_symbol->type);
    const typet &ret_type = func_type.return_type();

    if (lhs_type == "Any")
    {
      // For Any-annotated variables, always use the function's return type.
      current_element_type = ret_type;
      return ""; // Clear to avoid further "Any" processing
    }

    // Python type annotations are hints only and do not enforce runtime types.
    // When a function explicitly returns str (char*) but the variable is
    // annotated with a scalar type (e.g. y: int = f() where f() -> str),
    // use the actual return type so comparisons like y == "x" work correctly.
    bool ret_is_charptr =
      ret_type.is_pointer() && ret_type.subtype() == char_type();
    bool lhs_is_scalar =
      !current_element_type.is_pointer() && !current_element_type.is_array() &&
      !current_element_type.is_struct() && !current_element_type.id().empty();
    if (ret_is_charptr && lhs_is_scalar)
    {
      current_element_type = ret_type;
      return "";
    }
  }

  return lhs_type;
}

bool python_converter::handle_unpacking_assignment(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  codet &target_block)
{
  const auto &target_type = target["_type"];

  if (target_type != "Tuple" && target_type != "List")
    return false;

  // Get RHS
  is_converting_rhs = true;
  exprt rhs = get_expr(ast_node["value"]);
  is_converting_rhs = false;

  // Prepare RHS if it's a function call
  rhs = tuple_handler_->prepare_rhs_for_unpacking(ast_node, rhs, target_block);

  // Handle different unpacking types
  if (rhs.type().id() == "struct")
  {
    tuple_handler_->handle_tuple_unpacking(ast_node, target, rhs, target_block);
    return true;
  }
  else if (rhs.type().is_array())
  {
    handle_array_unpacking(ast_node, target, rhs, target_block);
    return true;
  }
  else if (rhs.type().is_pointer())
  {
    typet pointed_type = ns.follow(rhs.type().subtype());
    if (
      pointed_type.id() == "struct" &&
      tuple_handler_->is_tuple_type(pointed_type))
    {
      exprt tuple_value = dereference_exprt(rhs, pointed_type);
      tuple_value.location() = rhs.location();
      tuple_handler_->handle_tuple_unpacking(
        ast_node, target, tuple_value, target_block);
      return true;
    }

    const auto &value_node = ast_node["value"];
    if (value_node["_type"] == "List")
    {
      handle_list_literal_unpacking(ast_node, target, target_block);
      return true;
    }
    if (rhs.type() == get_type_handler().get_list_type())
    {
      python_list list(*this, ast_node["value"]);
      list.handle_list_var_unpacking(ast_node, target, rhs, target_block);
      return true;
    }
  }

  throw std::runtime_error(
    "Cannot unpack " + rhs.type().id_string() +
    " - only tuples and arrays can be unpacked");
}

symbolt *python_converter::create_symbol_for_unannotated_assign(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  const symbol_id &sid,
  bool is_global)
{
  if (is_global)
    return nullptr;

  if (!ast_node.contains("value") || !ast_node["value"].contains("_type"))
    return nullptr;

  const std::string &value_type = ast_node["value"]["_type"];
  locationt location = get_location_from_decl(target);
  std::string module_name = location.get_file().as_string();
  std::string name;

  if (target["_type"] == "Name")
    name = target["id"].get<std::string>();
  else if (target["_type"] == "Attribute")
    name = target["attr"].get<std::string>();

  typet inferred_type;

  if (value_type == "Lambda")
  {
    inferred_type = any_type();
  }
  else if (
    value_type == "Call" &&
    ast_node["value"]["func"].value("_type", "") == "Attribute" &&
    ast_node["value"]["func"]["value"].value("_type", "") == "Name")
  {
    // For dict method calls that emit instructions via converter_.add_instruction()
    // (pop, get, setdefault), calling get_expr() here for type inference would
    // execute the side effects a second time when the actual assignment is
    // processed.  Pop is especially harmful: the first evaluation removes the
    // key, so the second evaluation can't find it and throws KeyError.
    // Instead, infer the return type directly from the dict's value annotation.
    const std::string &method =
      ast_node["value"]["func"]["attr"].get<std::string>();
    const std::string &obj_name =
      ast_node["value"]["func"]["value"]["id"].get<std::string>();

    // Disambiguate by checking the actual symbol type, not just the annotation,
    // so that unannotated dict variables are also handled correctly.
    symbol_id obj_sid = create_symbol_id();
    obj_sid.set_object(obj_name);
    const symbolt *obj_sym = symbol_table_.find_symbol(obj_sid.to_string());

    // Value-returning dict methods emit IR instructions as a side-effect and
    // must not be called via get_expr() during type inference (double-eval).
    bool is_dict_method =
      python_dict_handler::is_value_returning_method(method) &&
      obj_sym != nullptr &&
      dict_handler_->is_dict_type(ns.follow(obj_sym->type));

    if (is_dict_method)
    {
      // obj_sym != nullptr is guaranteed by the is_dict_method check above
      if (method == "popitem")
      {
        // popitem() returns (key, value) tuple — infer the full tuple type
        inferred_type =
          dict_handler_->get_popitem_tuple_type(symbol_expr(*obj_sym));
      }
      else if (method == "copy")
      {
        // copy() returns a new dict, not a single element value.
        inferred_type = dict_handler_->get_dict_struct_type();
      }
      else
      {
        inferred_type = dict_handler_->resolve_expected_type_for_dict_subscript(
          symbol_expr(*obj_sym));
        if (inferred_type.is_nil() || inferred_type.is_empty())
        {
          // Untyped dict (e.g. `a = {}`): infer the return type from the default arg.
          // Any concrete literal (list, dict, int, float, str, bool, None)
          // is more precise than the `long_int` fallback applied just below.
          const std::string shape = python_annotation<nlohmann::json>::
            infer_type_from_default_arg_shape(ast_node["value"]["args"]);
          if (shape == "list")
            inferred_type = type_handler_.get_list_type();
          else if (shape == "dict")
            inferred_type = dict_handler_->get_dict_struct_type();
          else if (
            !shape.empty() && shape != "Any" &&
            type_utils::is_builtin_type(shape))
            inferred_type = type_handler_.get_typet(shape, 0);
          if (inferred_type.is_nil() || inferred_type.is_empty())
            inferred_type = long_int_type();
        }
      }
    }
    else
    {
      is_converting_rhs = true;
      exprt rhs_expr = get_expr(ast_node["value"]);
      is_converting_rhs = false;
      inferred_type = rhs_expr.type();
      if (inferred_type.is_empty())
        inferred_type = any_type();
      else if (inferred_type.is_code())
        inferred_type = gen_pointer_type(inferred_type);
    }
  }
  else
  {
    // Evaluate the RHS for any expression type (Call, BoolOp, Attribute,
    // Name, BinOp, Subscript, …) so that its type can be inferred.
    // If the expression is itself invalid — e.g. accessing a non-existent
    // attribute — get_expr will raise the correct, precise error at the
    // point of access rather than the misleading "Type undefined" later.
    is_converting_rhs = true;
    exprt rhs_expr = get_expr(ast_node["value"]);
    is_converting_rhs = false;

    inferred_type = rhs_expr.type();
    if (inferred_type.is_empty())
      inferred_type = any_type();
    // Function alias assignment (g = f): store as function pointer,
    // mirroring how lambda assignments are handled.
    else if (inferred_type.is_code())
      inferred_type = gen_pointer_type(inferred_type);
  }

  symbolt symbol =
    create_symbol(module_name, name, sid.to_string(), location, inferred_type);
  symbol.lvalue = true;
  symbol.file_local = true;
  symbol.is_extern = false;
  return symbol_table_.move_symbol_to_context(symbol);
}

void python_converter::handle_function_call_rhs(
  const nlohmann::json &ast_node,
  symbolt *lhs_symbol,
  exprt &lhs,
  exprt &rhs,
  const locationt &location,
  bool is_ctor_call,
  codet &target_block)
{
  if (is_ctor_call)
  {
    std::string func_name =
      ast_node["value"]["func"].contains("id")
        ? ast_node["value"]["func"]["id"].get<std::string>()
        : ast_node["value"]["func"]["attr"].get<std::string>();

    if (base_ctor_called)
    {
      auto class_node = json_utils::find_class((*ast_json)["body"], func_name);
      func_name = class_node["bases"][0]["id"].get<std::string>();
      base_ctor_called = false;
    }

    update_instance_from_self(func_name, func_name, lhs_symbol->id.as_string());
  }
  else
  {
    symbolt *func_symbol =
      symbol_table_.find_symbol(rhs.op1().identifier().c_str());
    assert(func_symbol);
    if (!static_cast<code_typet &>(func_symbol->type).return_type().is_empty())
    {
      if (auto ret = get_return_from_func(func_symbol->id.c_str());
          !ret.is_nil())
      {
        copy_instance_attributes(
          ret.op0().identifier().as_string(), lhs_symbol->id.as_string());
      }
    }
  }

  // Copy attributes from function arguments
  if (!is_ctor_call)
  {
    const code_function_callt &call =
      static_cast<const code_function_callt &>(rhs);
    for (const auto &arg : call.arguments())
    {
      const exprt *arg_ptr = &arg;
      if (arg.is_address_of())
        arg_ptr = &arg.op0();

      if (arg_ptr->is_symbol())
      {
        copy_instance_attributes(
          arg_ptr->identifier().as_string(), lhs_symbol->id.as_string());
      }
    }
  }

  // Set return destination
  if (rhs.type().is_pointer() && !is_ctor_call)
  {
    rhs.op0() = lhs;
  }
  else if (!rhs.type().is_pointer() && !rhs.type().is_empty() && !is_ctor_call)
    rhs.op0() = lhs;

  // Special handling for list return type
  if (rhs.type() == type_handler_.get_list_type())
  {
    if (auto ret = get_return_from_func(rhs.op1().identifier().c_str());
        !ret.is_nil())
    {
      python_list::copy_type_info(
        ret.op0().identifier().as_string(), lhs.identifier().as_string());
    }

    // If list_type_map is still empty for the LHS
    // e.g. the list was passed through as a parameter inside the function,
    // fall back to the called function's return-type annotation
    // to determine the element type.
    const std::string &lhs_id = lhs.identifier().as_string();
    if (python_list::get_list_type_map_size(lhs_id) == 0)
    {
      std::string func_name;
      if (
        ast_node.contains("value") && ast_node["value"].contains("func") &&
        ast_node["value"]["func"].is_object())
      {
        const auto &func_ref = ast_node["value"]["func"];
        if (func_ref.contains("id") && func_ref["id"].is_string())
          func_name = func_ref["id"].get<std::string>();
        else if (func_ref.contains("attr") && func_ref["attr"].is_string())
          func_name = func_ref["attr"].get<std::string>();
      }

      if (!func_name.empty())
      {
        const auto &func_def =
          json_utils::find_function((*ast_json)["body"], func_name);
        if (
          !func_def.empty() && func_def.contains("returns") &&
          !func_def["returns"].is_null())
        {
          const auto &returns = func_def["returns"];
          // Handle list[T] annotation
          // Subscript node with value.id == "list"
          if (
            returns.is_object() && returns.contains("_type") &&
            returns["_type"] == "Subscript" && returns.contains("value") &&
            returns["value"].is_object() && returns["value"].contains("id") &&
            returns["value"]["id"].is_string())
          {
            const std::string val_id =
              returns["value"]["id"].get<std::string>();
            if (val_id == "list" || val_id == "List")
            {
              // Extract element type from the slice, e.g. int in list[int]
              if (
                returns.contains("slice") && returns["slice"].is_object() &&
                returns["slice"].contains("id") &&
                returns["slice"]["id"].is_string())
              {
                typet elem_type = type_handler_.get_typet(
                  returns["slice"]["id"].get<std::string>());
                if (elem_type != typet())
                {
                  python_list::add_type_info_entry(
                    lhs_id, std::string(), elem_type);
                }
              }
            }
          }
        }
      }
    }

    typet l_type = type_handler_.get_list_type();
    symbolt &tmp_var_symbol =
      create_tmp_symbol(ast_node, "tmp_var", l_type, gen_zero(l_type));

    code_declt tmp_var_decl(symbol_expr(tmp_var_symbol));
    tmp_var_decl.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(tmp_var_decl);

    rhs.op0() = symbol_expr(tmp_var_symbol);
    target_block.copy_to_operands(rhs);

    code_assignt code_assign(lhs, symbol_expr(tmp_var_symbol));
    code_assign.location() = location;
    rhs = code_assign;
  }

  target_block.copy_to_operands(rhs);
}

exprt python_converter::handle_string_literal_rhs(
  const nlohmann::json &ast_node,
  const std::string &lhs_type,
  const exprt &rhs)
{
  if (lhs_type != "str" || !type_utils::is_integer_type(rhs.type()))
    return rhs;

  if (
    ast_node["value"]["_type"] != "Constant" ||
    !ast_node["value"]["value"].is_string())
    return rhs;

  std::string str_value = ast_node["value"]["value"].get<std::string>();

  typet string_type =
    type_handler_.build_array(char_type(), str_value.length() + 1);
  exprt str_array = gen_zero(string_type);

  for (size_t i = 0; i < str_value.length(); ++i)
  {
    BigInt char_val(static_cast<unsigned char>(str_value[i]));
    exprt char_expr = constant_exprt(
      integer2binary(char_val, 8), integer2string(char_val), char_type());
    str_array.operands().at(i) = char_expr;
  }

  return str_array;
}

bool python_converter::is_global_variable(const symbol_id &sid) const
{
  for (const std::string &s : global_declarations)
  {
    if (s == sid.global_to_string())
      return true;
  }
  return false;
}

std::string
python_converter::extract_target_name(const nlohmann::json &target) const
{
  const auto &target_type = target["_type"];

  if (target_type == "Name")
    return target["id"].get<std::string>();
  else if (target_type == "Attribute")
    return target["attr"].get<std::string>();
  else if (target_type == "Subscript")
    return target["value"]["id"].get<std::string>();

  throw std::runtime_error(
    "Unsupported assignment target type: " + target_type.get<std::string>());
}

void python_converter::preregister_global_variables(
  const nlohmann::json &ast_body)
{
  // Pre-register module-level annotated variable symbols so that class methods
  // can reference globals declared later in the source (Python LEGB rule).
  // Only annotated assignments (AnnAssign) carry enough type information for
  // symbol registration; plain Assign without annotation is skipped via the
  // nil-type guard below.
  for (const auto &element : ast_body)
  {
    if (element.value("_type", "") != "AnnAssign")
      continue;

    // Skip implicitly inferred annotations (plain Assign converted by the
    // annotator). Only preregister variables that the user explicitly annotated
    // (e.g., `x: SomeClass = ...`). Inferred globals like `l = [1, 2, 3]`
    // should not be visible inside functions that don't declare `global l`.
    if (element.value("_inferred_annotation", false))
      continue;

    // Skip union-type forward declarations (e.g., `x: str | datetime`).
    // These are bare declarations with no value and the union type cannot be
    // reliably resolved at this stage. The variable will be registered when
    // the actual assignment is processed (after imports are loaded).
    if (
      element.contains("annotation") && !element["annotation"].is_null() &&
      element["annotation"].value("_type", "") == "BinOp" &&
      element.contains("value") && element["value"].is_null())
      continue;

    if (!element.contains("target"))
      continue;

    const auto &target = element["target"];
    if (!target.contains("id"))
      continue;

    const std::string var_name = target["id"].get<std::string>();

    symbol_id sid(current_python_file, "", "");
    sid.set_object(var_name);

    if (symbol_table_.find_symbol(sid.to_string()))
      continue;

    typet var_type;
    try
    {
      var_type = extract_type_info(element).second;
    }
    catch (const std::exception &e)
    {
      // Type not yet resolvable (e.g., from an unprocessed import). Skip for
      // now; the variable will be registered when the assignment is processed
      // after imports are loaded.
      log_warning(
        "preregister_global_variables: skipping '{}' ({})",
        element["target"].value("id", "<unknown>"),
        e.what());
      continue;
    }
    if (var_type.is_nil() || var_type.is_empty())
      continue;

    locationt location = get_location_from_decl(element);
    std::string module_name =
      current_python_file.substr(0, current_python_file.find_last_of("."));

    symbolt symbol =
      create_symbol(module_name, var_name, sid.to_string(), location, var_type);
    symbol.lvalue = true;
    symbol.file_local = true;
    symbol.is_extern = false;

    symbol_table_.move_symbol_to_context(symbol);
  }
}

void python_converter::get_var_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  // Extract type information
  auto [lhs_type, element_type] = extract_type_info(ast_node);

  // Check if the RHS is a dictionary literal - set the element type
  if (
    ast_node.contains("value") && !ast_node["value"].is_null() &&
    dict_handler_->is_dict_literal(ast_node["value"]))
  {
    element_type = dict_handler_->get_dict_struct_type();
  }

  current_element_type = element_type;
  typet annotated_type = element_type;
  std::vector<typet> annotation_types;
  bool can_emit_annotation_check = false;
  locationt annotation_location;
  std::string annotated_name;
  std::vector<typet> annotation_candidates;

  exprt lhs;
  symbolt *lhs_symbol = nullptr;
  locationt location_begin;
  symbol_id sid = create_symbol_id();

  const auto &target = (ast_node.contains("targets")) ? ast_node["targets"][0]
                                                      : ast_node["target"];

  // Handle forward references
  if (
    ast_node.contains("value") && !ast_node["value"].is_null() &&
    ast_node["value"]["_type"] == "Call" &&
    type_handler_.is_constructor_call(ast_node["value"]))
  {
    process_forward_reference(ast_node["value"]["func"], target_block);
  }

  // Handle dict subscript assignment: dict[key] = value
  if (dict_handler_->handle_subscript_assignment_check(
        *this, ast_node, target, target_block))
    return;

  if (target.contains("_type") && target["_type"] == "Subscript")
  {
    exprt container_expr = get_expr(target["value"]);
    typet container_type = container_expr.type();

    // Tuple subscript assignment: tuples are immutable, raise TypeError
    if (tuple_handler_->is_tuple_type(container_type))
    {
      exprt raise = get_exception_handler().gen_exception_raise(
        "TypeError", "'tuple' object does not support item assignment");
      codet throw_code("expression");
      throw_code.operands().push_back(raise);
      target_block.copy_to_operands(throw_code);
      return;
    }

    // Handle object subscript assignment via __setitem__:
    //   obj[key] = value  ->  obj.__setitem__(key, value)
    if (
      target.contains("value") && target.contains("slice") &&
      ast_node.contains("value") && !ast_node["value"].is_null() &&
      has_dunder_method(target["value"], "__setitem__"))
    {
      nlohmann::json args = nlohmann::json::array();
      args.push_back(target["slice"]);
      args.push_back(ast_node["value"]);
      nlohmann::json call_node =
        build_dunder_call(target["value"], "__setitem__", args, ast_node);
      exprt setitem_call = get_function_call(call_node);
      target_block.copy_to_operands(convert_expression_to_code(setitem_call));
      return;
    }
  }

  if (ast_node["_type"] == "AnnAssign")
  {
    // Extract name and set in symbol ID
    std::string name = extract_target_name(target);
    sid.set_object(name);
    annotated_name = name;

    // Check if this is a forward declaration with union type and no value
    // e.g., dt: str | datetime (without assignment)
    // These should be skipped; wait for the actual assignment
    bool is_union_type = false;
    if (
      ast_node.contains("annotation") && !ast_node["annotation"].is_null() &&
      ast_node["annotation"].contains("_type") &&
      ast_node["annotation"]["_type"] == "BinOp")
    {
      is_union_type = true;
    }

    if (is_union_type && ast_node["value"].is_null())
    {
      // Skip this forward declaration; wait for the actual assignment
      // that will give us the type information
      return;
    }

    // Infer type from function return if annotation is "Any"
    lhs_type = infer_type_from_any_annotation(ast_node, lhs_type);

    // Process RHS before LHS if in function scope
    exprt rhs;
    if (
      sid.to_string().find("@F") != std::string::npos &&
      sid.to_string().find("@C") == std::string::npos)
    {
      is_right = true;
      if (!ast_node["value"].is_null())
      {
        // Skip getting expr for dict literals - handle specially later
        if (!dict_handler_->is_dict_literal(ast_node["value"]))
        {
          if (ast_node["_type"] != "Call")
          {
            rhs = get_rhs_with_dict_resolution(ast_node, current_element_type);
          }
        }
      }
      is_right = false;
    }

    // Location and symbol lookup
    location_begin = get_location_from_decl(target);
    annotation_location = location_begin;
    can_emit_annotation_check = true;
    lhs_symbol = symbol_table_.find_symbol(sid.to_string().c_str());

    bool is_global = is_global_variable(sid);
    if (is_global)
      lhs_symbol = symbol_table_.find_symbol(sid.global_to_string().c_str());

    // Symbol creation
    bool symbol_created = false;
    if (!lhs_symbol || !is_global)
    {
      std::string module_name = location_begin.get_file().as_string();

      symbolt symbol = create_symbol(
        module_name,
        name,
        sid.to_string(),
        location_begin,
        current_element_type);
      symbol.lvalue = true;
      symbol.file_local = true;
      symbol.is_extern = false;

      symbol_created = (lhs_symbol == nullptr);
      lhs_symbol = symbol_table_.move_symbol_to_context(symbol);

      // Add declaration statement ONLY for newly created local variables
      if (symbol_created && !current_func_name_.empty() && !is_global)
      {
        code_declt decl(symbol_expr(*lhs_symbol));
        decl.location() = location_begin;
        target_block.copy_to_operands(decl);
      }
    }

    if (lhs_symbol && ast_node.contains("annotation"))
      get_typechecker().cache_annotation_types(
        *lhs_symbol, ast_node["annotation"]);

    if (
      type_assertions_enabled() && lhs_symbol &&
      ast_node.contains("annotation"))
    {
      auto &tc = get_typechecker();
      annotation_types = tc.get_annotation_types(lhs_symbol->id.as_string());
      if (
        !annotation_types.empty() &&
        !tc.should_skip_type_assertion(annotated_type))
      {
        annotated_type = annotation_types.front();
        can_emit_annotation_check = true;
        annotation_location = location_begin;
        annotated_name = name;
        annotation_candidates = annotation_types;
      }
    }

    // Check for uninitialized usage
    for (std::string &s : local_loads)
    {
      if (lhs_symbol->id.as_string() == s)
      {
        throw std::runtime_error(
          "Variable " + sid.get_object() + " in function " +
          current_func_name_ + " is uninitialized.");
      }
    }

    // Create LHS expression
    lhs = create_lhs_expression(target, lhs_symbol, location_begin);

    // Handle dict literal assignment specially - after LHS is created
    if (dict_handler_->handle_literal_assignment_check(*this, ast_node, lhs))
    {
      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      return;
    }
  }
  else if (ast_node["_type"] == "Assign")
  {
    const auto &target = ast_node["targets"][0];
    location_begin = get_location_from_decl(target);

    // Handle tuple/list unpacking
    if (handle_unpacking_assignment(ast_node, target, target_block))
      return;

    // Normal assignment handling
    std::string name = extract_target_name(target);
    sid.set_object(name);
    lhs_symbol = symbol_table_.find_symbol(sid.to_string());

    bool is_global = is_global_variable(sid);

    // Handle unannotated dict literal assignment
    if (
      !lhs_symbol && dict_handler_->handle_unannotated_literal_check(
                       *this, ast_node, target, sid))
      return;

    // Create symbol for unannotated assignments with inferrable types.
    // If the annotator injected an "annotation" field and we already have a
    // valid type in current_element_type, use it directly so that user-defined
    // classes named "List"/"Dict" are not mis-resolved to built-in types.
    if (!lhs_symbol && !is_global)
    {
      if (
        ast_node.contains("annotation") && !ast_node["annotation"].is_null() &&
        !current_element_type.is_empty())
      {
        std::string module_name = location_begin.get_file().as_string();
        symbolt symbol = create_symbol(
          module_name,
          name,
          sid.to_string(),
          location_begin,
          current_element_type);
        symbol.lvalue = true;
        symbol.file_local = true;
        symbol.is_extern = false;
        lhs_symbol = symbol_table_.move_symbol_to_context(symbol);
      }
      else
      {
        lhs_symbol = create_symbol_for_unannotated_assign(
          ast_node, target, sid, is_global);
      }
    }

    if (!lhs_symbol && !is_global)
      throw std::runtime_error("Type undefined for \"" + name + "\"");

    lhs = create_lhs_expression(target, lhs_symbol, location_begin);

    if (lhs_symbol && ast_node.contains("annotation"))
      get_typechecker().cache_annotation_types(
        *lhs_symbol, ast_node["annotation"]);

    if (type_assertions_enabled() && lhs_symbol)
    {
      auto &tc = get_typechecker();
      annotation_types = tc.get_annotation_types(lhs_symbol->id.as_string());
      if (
        !annotation_types.empty() &&
        !tc.should_skip_type_assertion(lhs_symbol->type))
      {
        annotated_type = annotation_types.front();
        can_emit_annotation_check = true;
        annotation_location = location_begin;
        annotated_name = name;
        annotation_candidates = annotation_types;
      }
    }
  }

  if (
    type_assertions_enabled() && can_emit_annotation_check &&
    annotation_candidates.empty() &&
    !get_typechecker().should_skip_type_assertion(annotated_type))
    annotation_candidates.push_back(annotated_type);

  bool is_ctor_call = type_handler_.is_constructor_call(ast_node["value"]);
  current_lhs = &lhs;
  is_converting_lhs = false;

  // Get RHS
  exprt rhs;
  bool has_value = false;
  if (!ast_node["value"].is_null())
  {
    is_converting_rhs = true;

    if (lhs_symbol)
      rhs = get_rhs_with_dict_resolution(ast_node, lhs_symbol->type);
    else
      rhs = get_expr(ast_node["value"]);

    has_value = true;
    is_converting_rhs = false;

    // Handle string literal conversion
    rhs = handle_string_literal_rhs(ast_node, lhs_type, rhs);
  }

  if (has_value && rhs != exprt("_init_undefined"))
  {
    auto try_follow_symbol_type = [this](const typet &type) -> typet {
      if (type.id() != "symbol")
        return type;

      const irep_idt &symbol_id = type.identifier();
      if (symbol_id.empty())
        return type;

      if (symbol_table_.find_symbol(symbol_id) == nullptr)
        return type;

      return ns.follow(type);
    };

    auto is_list_model_type = [this,
                               &try_follow_symbol_type](const typet &in_type) {
      typet t = in_type;
      t = try_follow_symbol_type(t);
      if (t.is_pointer())
        t = t.subtype();
      t = try_follow_symbol_type(t);
      if (!t.is_struct())
        return false;
      return to_struct_type(t).tag().as_string().find("__ESBMC_PyListObj") !=
             std::string::npos;
    };

    auto resolve_runtime_type = [this,
                                 &try_follow_symbol_type](const exprt &expr) {
      typet t = expr.type();
      if (expr.is_symbol())
      {
        if (const symbolt *sym = symbol_table_.find_symbol(expr.identifier()))
          t = sym->type;
      }
      return try_follow_symbol_type(t);
    };

    const typet lhs_runtime_type = lhs_symbol ? lhs_symbol->type : lhs.type();
    const typet rhs_runtime_type = resolve_runtime_type(rhs);
    if (
      dict_handler_->is_dict_type(lhs_runtime_type) &&
      is_list_model_type(rhs_runtime_type))
    {
      throw std::runtime_error(
        "Unsupported reassignment from dict to list for variable '" +
        sid.get_object() + "'");
    }

    // Handle throw expression
    if (rhs.statement() == "cpp-throw")
    {
      rhs.location() = location_begin;
      codet code_expr("expression");
      code_expr.operands().push_back(rhs);
      code_declt decl(symbol_expr(*lhs_symbol));
      decl.location() = location_begin;

      target_block.copy_to_operands(code_expr);
      target_block.copy_to_operands(decl);
      current_lhs = nullptr;
      return;
    }

    // Python dynamic typing: if a variable already has a numeric type (e.g.
    // double from float()) and is being reassigned to a pointer/string type
    // (e.g. char* from chr()), the GOTO IR cannot represent this type change
    // safely — the old SSA constant and the new pointer type mismatch in both
    // the symex renamer and the SMT encoder. Skip the assignment so the prior
    // type and value are preserved. This is sound for verification as long as
    // the new value is not used in a subsequent assertion.
    if (
      lhs_symbol && !lhs.type().is_pointer() && rhs.type().is_pointer() &&
      rhs.type().subtype() ==
        char_type() && // only skip string (char*) reassignment, not None (void*/bool*)
      (lhs.type().is_floatbv() || lhs.type().is_signedbv() ||
       lhs.type().is_unsignedbv() || lhs.type().is_bool()))
    {
      // Still emit the RHS as a void call so exceptions/side-effects are
      // preserved (e.g. chr() out-of-range ValueError).
      if (
        rhs.id() == "sideeffect" &&
        rhs.statement() == irep_idt("function_call"))
      {
        const side_effect_expr_function_callt &se =
          to_side_effect_expr_function_call(rhs);
        code_function_callt void_call;
        void_call.function() = se.function();
        void_call.arguments() = se.arguments();
        void_call.location() = rhs.location();
        add_instruction(void_call);
      }
      current_lhs = nullptr;
      return;
    }

    // Handle type adjustments
    handle_assignment_type_adjustments(
      lhs_symbol, lhs, rhs, lhs_type, ast_node, is_ctor_call);

    // Propagate $input_str$ → $input_len$ companion mapping so that len()
    // on any alias of an input() string can use the symbolic length directly.
    // Must run before type-branching which may take early returns.
    if (lhs.is_symbol())
    {
      if (rhs.is_symbol())
      {
        const std::string rhs_id = rhs.identifier().as_string();
        auto it = input_str_to_len_sym_.find(rhs_id);
        if (it != input_str_to_len_sym_.end())
          input_str_to_len_sym_[lhs.identifier().as_string()] = it->second;
        else
          input_str_to_len_sym_.erase(lhs.identifier().as_string());
      }
      else
      {
        // RHS is not a symbol with a mapped input length: clear any stale mapping
        input_str_to_len_sym_.erase(lhs.identifier().as_string());
      }
    }

    // Function call handling
    if (rhs.is_function_call())
    {
      // Static constructor compatibility check for annotated variables:
      // if var is annotated with a class type (e.g., Animal) and the RHS
      // constructor is a different, non-derived class (e.g., Car), inject
      // an assertion failure.
      if (
        type_assertions_enabled() && can_emit_annotation_check &&
        is_ctor_call && ast_node.contains("annotation") &&
        ast_node["annotation"].contains("id"))
      {
        std::string expected_base =
          ast_node["annotation"]["id"].get<std::string>();
        std::string ctor_name =
          get_typechecker().get_constructor_name(ast_node["value"]["func"]);

        if (
          !expected_base.empty() && !ctor_name.empty() &&
          !get_typechecker().class_derives_from(ctor_name, expected_base))
        {
          code_assertt ctor_assert(gen_boolean(false));
          ctor_assert.location() = location_begin;
          ctor_assert.location().comment(
            "Constructor '" + ctor_name +
            "' is incompatible with annotated type '" + expected_base + "'");
          target_block.copy_to_operands(ctor_assert);
        }
      }

      handle_function_call_rhs(
        ast_node,
        lhs_symbol,
        lhs,
        rhs,
        location_begin,
        is_ctor_call,
        target_block);
      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      current_lhs = nullptr;
      return;
    }

    // Type-incompatible reassignment: scalar variable assigned a
    // string/array value. Must check BEFORE adjust_statement_types
    // which would coerce rhs type to match lhs, hiding the mismatch.
    // Only enforce when type assertions are enabled (--is-instance-check).
    if (
      type_assertions_enabled() && lhs.type() != rhs.type() &&
      !rhs.type().is_code() && !rhs.type().is_empty() &&
      rhs.type().is_array() && !lhs.type().is_array() &&
      !lhs.type().is_pointer())
    {
      code_assertt type_assert(gen_boolean(false));
      type_assert.location() = location_begin;
      type_assert.location().comment(
        "Type violation: incompatible types in assignment");
      target_block.copy_to_operands(type_assert);
      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      current_lhs = nullptr;
      return;
    }

    adjust_statement_types(lhs, rhs);

    // Handle list type info propagation
    if (lhs.type() == rhs.type() && lhs.type() == type_handler_.get_list_type())
    {
      const std::string &lhs_identifier = lhs.identifier().as_string();
      const std::string &rhs_identifier = rhs.identifier().as_string();
      python_list::copy_type_info(rhs_identifier, lhs_identifier);

      if (lhs_symbol)
      {
        const symbolt *rhs_symbol = nullptr;
        if (rhs.is_symbol())
          rhs_symbol = find_symbol(rhs.identifier().as_string());
        if (rhs_symbol && rhs_symbol->is_set)
          lhs_symbol->is_set = true;
      }
    }
    else if (
      rhs.type() != lhs.type() && lhs.type().is_array() &&
      !rhs.type().is_code())
    {
#ifndef NDEBUG
      const array_typet &thetype = lhs.type();
      thetype.size().is_constant();
      assert(thetype.size().is_nil());
#endif
      lhs_symbol->type = rhs.type();

      code_declt decl(symbol_expr(*lhs_symbol), rhs);
      decl.location() = location_begin;
      target_block.copy_to_operands(decl);
      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      current_lhs = nullptr;
      return;
    }
    code_assignt code_assign(lhs, rhs);
    code_assign.location() = location_begin;
    target_block.copy_to_operands(code_assign);
    if (type_assertions_enabled() && can_emit_annotation_check)
      get_typechecker().emit_type_annotation_assertion(
        lhs,
        annotated_type,
        annotation_types,
        annotated_name,
        annotation_location,
        target_block);
  }
  else
  {
    lhs_symbol->value = gen_zero(current_element_type, true);
    lhs_symbol->value.zero_initializer(true);

    code_declt decl(symbol_expr(*lhs_symbol));
    decl.location() = location_begin;
    target_block.copy_to_operands(decl);
  }

  current_lhs = nullptr;
}

typet python_converter::resolve_variable_type(
  const std::string &var_name,
  const locationt &loc)
{
  std::string function = loc.get_function().as_string();
  nlohmann::json decl_node = find_var_decl(var_name, function, *ast_json);

  if (!decl_node.empty())
  {
    if (decl_node.contains("annotation") && !decl_node["annotation"].is_null())
    {
      const auto &annotation = decl_node["annotation"];

      try
      {
        // Handle rich annotations such as Union, Optional, module attributes,
        // etc. via the unified helper.
        return get_type_from_annotation(annotation, decl_node);
      }
      catch (const std::exception &e)
      {
        log_warning(
          "Failed to resolve complex annotation for '{}': {}. Falling back to "
          "simple identifier lookup.",
          var_name,
          e.what());
      }

      if (annotation.contains("id"))
      {
        std::string type_annotation = annotation["id"].get<std::string>();
        return type_handler_.get_typet(type_annotation);
      }
    }
  }

  std::string filename = loc.get_file().as_string();
  std::string symbol_id = "py:" + filename + "@F@" + function + "@" + var_name;

  const symbolt *sym = symbol_table_.find_symbol(symbol_id);
  if (sym != nullptr)
    return sym->type;
  else
  {
    log_error(
      "Variable '{}' not found in symbol table; cannot determine type.",
      symbol_id);
    abort();
  }
}

void python_converter::get_compound_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  locationt loc = get_location_from_decl(ast_node);

  // Set flags for LHS processing
  is_converting_lhs = true;

  // Get the target expression first
  exprt lhs = get_expr(ast_node["target"]);

  // Reset LHS flag and set RHS flag
  is_converting_lhs = false;
  is_converting_rhs = true;

  std::string var_name;

  // Extract variable name based on target type
  if (ast_node["target"].contains("id"))
  {
    // Simple variable assignment: x += 1
    var_name = ast_node["target"]["id"].get<std::string>();
  }
  else if (ast_node["target"]["_type"] == "Attribute")
  {
    // Don't extract just the attribute name for type resolution
    // The type should come from the LHS expression we just created
    if (ast_node["target"].contains("attr"))
      var_name = ast_node["target"]["attr"].get<std::string>();
  }
  else if (ast_node["target"]["_type"] == "Subscript")
  {
    // Subscript assignment: arr[i] += 1
    throw std::runtime_error(
      "Subscript assignment not supported in compound assignment");
  }
  else
  {
    throw std::runtime_error(
      "Unsupported target type in compound assignment: " +
      ast_node["target"]["_type"].get<std::string>());
  }

  // For attribute assignments, use the type from the LHS expression
  // For other assignments, resolve the variable type
  if (!lhs.type().is_nil() && !lhs.type().id().empty())
    current_element_type = lhs.type();
  else
  {
    // Fallback to resolving the variable type from AST or symbol table
    current_element_type = resolve_variable_type(var_name, loc);
  }

  std::string op = ast_node["op"]["_type"].get<std::string>();

  // Check if this is a string concatenation based on variable annotation
  bool is_string_concat = false;
  if (op == "Add")
  {
    // Standard array-based string concatenation
    if (
      (lhs.type().is_array() && lhs.type().subtype() == char_type()) ||
      (current_element_type.is_array() &&
       current_element_type.subtype() == char_type()))
    {
      is_string_concat = true;
    }
    // Pointer-based string
    else if (
      (lhs.type().is_pointer() && lhs.type().subtype() == char_type()) ||
      (current_element_type.is_pointer() &&
       current_element_type.subtype() == char_type()))
    {
      is_string_concat = true;
    }
    // Check if variable is annotated as str but implemented as single char
    else if (
      type_utils::is_integer_type(lhs.type()) &&
      type_utils::is_integer_type(current_element_type))
    {
      // Check if the variable was declared with str annotation
      nlohmann::json decl_node = get_var_node(var_name, *ast_json);
      if (
        !decl_node.empty() && decl_node.contains("annotation") &&
        decl_node["annotation"].contains("id") &&
        decl_node["annotation"]["id"] == "str")
      {
        is_string_concat = true;
      }
    }
  }

  if (is_string_concat)
  {
    exprt rhs_expr = get_expr(ast_node["value"]);
    nlohmann::json left = ast_node["target"];
    nlohmann::json right = ast_node["value"];
    exprt concatenated =
      string_handler_.handle_string_concatenation(lhs, rhs_expr, left, right);

    // Update the variable's type to match the concatenated result
    // Handle both array and pointer results
    if (
      !var_name.empty() && (concatenated.type().is_array() ||
                            (concatenated.type().is_pointer() &&
                             concatenated.type().subtype() == char_type())))
    {
      symbol_id sid = create_symbol_id();
      sid.set_object(var_name);
      symbolt *symbol = symbol_table_.find_symbol(sid.to_string());
      if (symbol)
      {
        // Update the symbol's type to pointer if concatenated returns pointer
        symbol->type = concatenated.type();

        // Update LHS to be a symbol with the new type
        lhs = symbol_exprt(symbol->id, symbol->type);

        // For pointer results, don't update the value
        // (it will be assigned via the assignment statement)
        if (concatenated.type().is_array())
        {
          symbol->value = concatenated;
        }
      }
    }

    code_assignt code_assign(lhs, concatenated);
    code_assign.location() = loc;
    target_block.copy_to_operands(code_assign);

    // Reset RHS flag
    is_converting_rhs = false;
    return;
  }

  exprt rhs = get_binary_operator_expr(ast_node);

  // Reset RHS flag
  is_converting_rhs = false;

  // P27: Promote real RHS to complex when LHS is complex (AugAssign path).
  // adjust_statement_types() is NOT called on this path, so without this
  // check, `z += 1.0` / `z *= 2` produce a struct/scalar type mismatch in IR.
  if (
    is_complex_type(lhs.type()) && !is_complex_type(rhs.type()) &&
    (rhs.type().is_floatbv() || type_utils::is_integer_type(rhs.type()) ||
     rhs.type().is_bool()))
  {
    rhs = promote_to_complex(rhs);
  }

  code_assignt code_assign(lhs, rhs);
  code_assign.location() = loc;
  target_block.copy_to_operands(code_assign);
}

typet resolve_ternary_type(
  const typet &then_type,
  const typet &else_type,
  const typet &default_type)
{
  if (then_type == else_type)
    return then_type;

  // Enhanced numeric promotion: int < float
  if (type_utils::is_integer_type(then_type) && else_type.is_floatbv())
    return else_type;
  if (type_utils::is_integer_type(else_type) && then_type.is_floatbv())
    return then_type;

  // String handling: use pointer type for consistency
  // Handles: array+array, array+pointer, pointer+array
  bool then_is_string =
    (then_type.is_array() && then_type.subtype() == char_type()) ||
    (then_type.is_pointer() && then_type.subtype() == char_type());
  bool else_is_string =
    (else_type.is_array() && else_type.subtype() == char_type()) ||
    (else_type.is_pointer() && else_type.subtype() == char_type());

  if (then_is_string && else_is_string)
    return gen_pointer_type(char_type());

  // Both arrays (non-strings)
  if (then_type.is_array() && else_type.is_array())
    return then_type;

  // Mixed signed/unsigned integers - prefer signed for safety
  if (then_type.is_signedbv() && else_type.is_unsignedbv())
    return then_type;
  if (then_type.is_unsignedbv() && else_type.is_signedbv())
    return else_type;

  // Incompatible types
  log_debug(
    "python-frontend",
    "[resolve_ternary_type] Ternary branches have incompatible types: {} vs "
    "{}, using default {}",
    then_type.id_string(),
    else_type.id_string(),
    default_type.id_string());

  return default_type;
}

exprt python_converter::get_conditional_stm(const nlohmann::json &ast_node)
{
  // Copy current type
  typet t = current_element_type;
  // Change to boolean before extracting condition
  current_element_type = bool_type();

  // Check if we need to materialize function calls in the condition
  // This handles cases like: if not math.isnan(x): or if isinstance(x, type):
  auto test_type = ast_node["test"]["_type"].get<std::string>();

  bool has_nested_call = false;
  nlohmann::json call_node;
  bool is_wrapped_in_unary = false;

  // Check for function call wrapped in UnaryOp (e.g., "not func()")
  if (test_type == "UnaryOp" && ast_node["test"].contains("operand"))
  {
    auto operand_type = ast_node["test"]["operand"]["_type"].get<std::string>();
    if (operand_type == "Call")
    {
      has_nested_call = true;
      is_wrapped_in_unary = true;
      call_node = ast_node["test"]["operand"];
    }
  }
  // Check for direct function call
  else if (test_type == "Call")
  {
    has_nested_call = true;
    call_node = ast_node["test"];
  }

  auto type = ast_node["_type"];
  if (type == "While" && has_nested_call)
  {
    locationt location = get_location_from_decl(ast_node);
    locationt call_location = get_location_from_decl(call_node);

    code_blockt transformed;

    // Reuse a single condition temporary to avoid redeclaring symbols
    // at each iteration of the lowered loop.
    symbolt cond_symbol =
      create_return_temp_variable(bool_type(), call_location, "while_cond");
    symbol_table_.add(cond_symbol);
    exprt cond_tmp = symbol_expr(cond_symbol);

    code_declt cond_decl(cond_tmp);
    cond_decl.location() = call_location;
    transformed.copy_to_operands(cond_decl);

    code_blockt loop_body;

    code_blockt *saved_block = current_block;
    current_block = &loop_body;
    exprt *saved_lhs = current_lhs;
    current_lhs = nullptr;
    exprt func_call = get_expr(call_node);
    current_lhs = saved_lhs;
    current_block = saved_block;

    if (func_call.is_function_call())
    {
      if (!func_call.type().is_empty())
        func_call.op0() = cond_tmp;
      loop_body.copy_to_operands(func_call);
    }
    else
    {
      code_assignt cond_assign(cond_tmp, func_call);
      cond_assign.location() = call_location;
      loop_body.copy_to_operands(cond_assign);
    }

    exprt overall_cond = cond_tmp;
    if (is_wrapped_in_unary)
    {
      overall_cond = exprt("not", bool_type());
      overall_cond.copy_to_operands(cond_tmp);
    }

    exprt break_cond("not", bool_type());
    break_cond.copy_to_operands(overall_cond);

    code_breakt break_stmt;
    break_stmt.location() = location;
    code_ifthenelset break_if;
    break_if.cond() = break_cond;
    break_if.then_case() = break_stmt;
    break_if.location() = location;
    loop_body.copy_to_operands(break_if);

    exprt body_expr;
    if (ast_node["body"].is_array())
      body_expr = get_block(ast_node["body"]);
    else
      body_expr = get_expr(ast_node["body"]);
    body_expr.location() = location;
    loop_body.copy_to_operands(body_expr);

    codet while_code;
    while_code.set_statement("while");
    while_code.location() = location;
    while_code.copy_to_operands(gen_boolean(true), loop_body);

    transformed.copy_to_operands(while_code);
    current_element_type = t;
    return transformed;
  }

  // Extract condition from AST
  exprt cond;

  // Keep `and` and `or` in conditions short-circuited.
  const bool coverage_mode = is_coverage_mode();
  const bool pytest_generation_mode = is_pytest_generation_mode();
  const bool model_mode = is_model_file(ast_node["test"]);
  auto to_bool_condition =
    [&](const exprt &value_expr, const nlohmann::json &value_node) -> exprt {
    if (value_expr.type().is_bool())
      return value_expr;

    typet list_type = type_handler_.get_list_type();
    if (
      value_expr.type() == list_type ||
      (value_expr.type().is_pointer() &&
       value_expr.type().subtype() == list_type))
    {
      const symbolt *size_func =
        symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
      if (!size_func)
        throw std::runtime_error(
          "__ESBMC_list_size not found for list condition check");

      side_effect_expr_function_callt size_call;
      size_call.function() = symbol_expr(*size_func);
      size_call.type() = size_type();
      size_call.location() = get_location_from_decl(value_node);
      if (value_expr.type().is_pointer())
        size_call.arguments().push_back(value_expr);
      else
        size_call.arguments().push_back(address_of_exprt(value_expr));

      exprt cond("notequal", bool_type());
      cond.copy_to_operands(size_call, gen_zero(size_type()));
      cond.location() = get_location_from_decl(value_node);
      return cond;
    }

    exprt bool_expr = typecast_exprt(value_expr, bool_type());
    bool_expr.location() = get_location_from_decl(value_node);
    return bool_expr;
  };

  if (
    test_type == "BoolOp" && current_block && type != "While" &&
    !coverage_mode && !pytest_generation_mode && !model_mode)
  {
    const auto &test_node = ast_node["test"];
    const auto &operands = test_node["values"];
    if (!operands.empty())
    {
      exprt *saved_lhs = current_lhs;
      current_lhs = nullptr;
      // Start from the leftmost operand and carry the running result forward.
      cond = to_bool_condition(get_expr(operands[0]), operands[0]);
      current_lhs = saved_lhs;

      symbolt result_symbol = create_return_temp_variable(
        bool_type(), get_location_from_decl(test_node), "boolop_cond");
      symbol_table_.add(result_symbol);
      exprt result_expr = symbol_expr(result_symbol);

      code_declt result_decl(result_expr);
      result_decl.location() = get_location_from_decl(test_node);
      current_block->copy_to_operands(result_decl);

      code_assignt result_init(result_expr, cond);
      result_init.location() = get_location_from_decl(test_node);
      current_block->copy_to_operands(result_init);

      const bool is_and = test_node["op"]["_type"] == "And";
      for (size_t i = 1; i < operands.size(); ++i)
      {
        code_blockt next_operand_block;
        code_blockt *saved_block = current_block;
        current_block = &next_operand_block;
        saved_lhs = current_lhs;
        current_lhs = nullptr;
        // Build the next operand only in the branch where it is still needed.
        exprt next_operand =
          to_bool_condition(get_expr(operands[i]), operands[i]);
        current_lhs = saved_lhs;
        current_block = saved_block;

        code_assignt result_update(result_expr, next_operand);
        result_update.location() = get_location_from_decl(operands[i]);
        next_operand_block.copy_to_operands(result_update);

        code_ifthenelset short_circuit_if;
        short_circuit_if.location() = get_location_from_decl(operands[i]);
        short_circuit_if.location().property("skipped");
        // `and` keeps going while the running result is true; `or` keeps
        // going while it is false.
        if (is_and)
          short_circuit_if.cond() = result_expr;
        else
        {
          exprt not_result("not", bool_type());
          not_result.copy_to_operands(result_expr);
          short_circuit_if.cond() = not_result;
        }
        short_circuit_if.then_case() = next_operand_block;
        current_block->copy_to_operands(short_circuit_if);
      }

      cond = result_expr;
    }
  }
  else if (test_type == "BoolOp" && !model_mode)
  {
    exprt boolop_expr(
      python_frontend::map_operator(ast_node["test"]["op"]["_type"], bool_type()), bool_type());
    for (const auto &operand : ast_node["test"]["values"])
      boolop_expr.copy_to_operands(
        to_bool_condition(get_expr(operand), operand));
    cond = boolop_expr;
  }
  else if (has_nested_call)
  {
    locationt location = get_location_from_decl(call_node);

    auto apply_wrapped_unary = [&](const exprt &base_expr) -> exprt {
      if (!is_wrapped_in_unary)
        return base_expr;

      auto op = ast_node["test"]["op"]["_type"].get<std::string>();
      if (op == "Not")
      {
        exprt unary_expr("not", bool_type());
        unary_expr.copy_to_operands(base_expr);
        return unary_expr;
      }
      return base_expr;
    };

    // Get the function call expression with special handling
    // Temporarily disable the conditional processing to avoid recursion
    exprt *saved_lhs = current_lhs;
    current_lhs = nullptr;
    exprt func_call = get_expr(call_node);
    current_lhs = saved_lhs;

    if (func_call.is_function_call())
    {
      // Create temporary variable for function call result
      symbolt temp_symbol =
        create_return_temp_variable(func_call.type(), location, "cond");
      symbol_table_.add(temp_symbol);
      exprt temp_var_expr = symbol_expr(temp_symbol);

      // Create declaration for temporary
      code_declt temp_decl(temp_var_expr);
      temp_decl.location() = location;

      // Set the LHS of the function call
      if (!func_call.type().is_empty())
        func_call.op0() = temp_var_expr;

      // Add both declaration and function call to current_block
      if (current_block)
      {
        current_block->copy_to_operands(temp_decl);
        current_block->copy_to_operands(func_call);
      }

      cond = apply_wrapped_unary(temp_var_expr);
    }
    else
    {
      cond = apply_wrapped_unary(func_call);
    }
  }
  else
  {
    // Normal path: no function call to materialize
    cond = get_expr(ast_node["test"]);
  }

  if (!(test_type == "BoolOp" && current_block && type != "While" &&
        !coverage_mode && !pytest_generation_mode && !model_mode))
  {
    cond.location() = get_location_from_decl(ast_node["test"]);

    if (!cond.type().is_bool())
    {
      const locationt location = get_location_from_decl(ast_node["test"]);
      typet value_type = ns.follow(cond.type());
      if (value_type.is_pointer())
        value_type = ns.follow(value_type.subtype());

      // Objects in conditions are converted with __bool__() when available.
      if (value_type.is_struct())
      {
        if (const std::string class_name = extract_class_name_from_tag(
              to_struct_type(value_type).tag().as_string());
            !class_name.empty())
        {
          if (symbolt *bool_method = find_dunder_method(class_name, "__bool__"))
          {
            exprt bool_object = cond;
            // __bool__ expects self by address, so the condition must be an object.
            if (!bool_object.is_symbol())
              bool_object =
                store_call_result(bool_object, location, "cond_obj");
            const code_typet &method_type = to_code_type(bool_method->type);
            side_effect_expr_function_callt bool_call;
            bool_call.function() = symbol_expr(*bool_method);
            bool_call.type() = method_type.return_type();
            bool_call.location() = location;
            bool_call.arguments().push_back(gen_address_of(bool_object));
            cond = store_call_result(bool_call, location, "cond_bool");
            cond.location() = location;
          }
        }
      }

      typet list_type = type_handler_.get_list_type();
      // Python treats lists in conditions by their size, for example:
      // `1 if xs else 0`.
      if (
        current_block &&
        (cond.type() == list_type ||
         (cond.type().is_pointer() && cond.type().subtype() == list_type)))
      {
        const symbolt *size_func =
          symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
        if (!size_func)
          throw std::runtime_error(
            "__ESBMC_list_size not found for list condition check");

        // Keep the size query inside the condition expression so constructs
        // like `while heap:` re-evaluate the current list size every iteration.
        side_effect_expr_function_callt size_call;
        size_call.function() = symbol_expr(*size_func);
        if (cond.type().is_pointer())
          size_call.arguments().push_back(cond);
        else
          size_call.arguments().push_back(address_of_exprt(cond));
        size_call.type() = size_type();
        size_call.location() = location;

        cond = exprt("notequal", bool_type());
        cond.copy_to_operands(size_call, gen_zero(size_type()));
        cond.location() = location;
      }

      // Python treats strings in conditions by their length: "" is falsy.
      if (type_utils::is_string_type(cond.type()))
      {
        const symbolt *strlen_sym = symbol_table_.find_symbol("c:@F@strlen");
        if (!strlen_sym)
          throw std::runtime_error(
            "strlen not found for string truthiness check");

        side_effect_expr_function_callt strlen_call;
        strlen_call.function() = symbol_expr(*strlen_sym);
        strlen_call.arguments().push_back(
          string_handler_.get_array_base_address(cond));
        strlen_call.type() = size_type();
        strlen_call.location() = location;

        cond = exprt("notequal", bool_type());
        cond.copy_to_operands(strlen_call, gen_zero(size_type()));
        cond.location() = location;
      }
    }
  }

  // P12: Python truthiness for complex in conditional contexts:
  // bool(z) == (z.real != 0.0 or z.imag != 0.0).
  // Delegates to the single canonical implementation in type_handler.h.
  if (is_complex_type(cond.type()))
  {
    locationt loc = get_location_from_decl(ast_node["test"]);
    cond = complex_to_bool_expr(cond);
    cond.location() = loc;
  }

  // Recover type
  current_element_type = t;

  // Extract 'then' block from AST
  exprt then;

  // Skip the 'then' block when the condition evaluates to false.
  if (cond.is_constant() && cond.value() == "false" && type != "IfExp")
  {
    then = code_blockt();
  }
  else
  {
    if (ast_node["body"].is_array())
      then = get_block(ast_node["body"]);
    else
      then = get_expr(ast_node["body"]);
  }

  locationt location = get_location_from_decl(ast_node);
  then.location() = location;

  // Extract 'else' block from AST
  exprt else_expr;
  if (ast_node.contains("orelse") && !ast_node["orelse"].empty())
  {
    // Append 'else' block to the statement
    if (ast_node["orelse"].is_array())
      else_expr = get_block(ast_node["orelse"]);
    else
      else_expr = get_expr(ast_node["orelse"]);
  }

  // ternary operator
  if (type == "IfExp")
  {
    // Normalize branches: code_function_callt must become side_effect_expr so
    // that migration to irep2 preserves the correct return type in if2t.
    then = to_value_expr(then, ns);
    else_expr = to_value_expr(else_expr, ns);

    bool then_is_none = (then.type() == none_type());
    bool else_is_none = (else_expr.type() == none_type());

    typet result_type;
    if (then_is_none != else_is_none)
    {
      // One branch is None, the other is T → Optional[T] models Python's T | None
      typet concrete_type = then_is_none ? else_expr.type() : then.type();
      result_type = type_handler_.build_optional_type(concrete_type);
      then = wrap_in_optional(then, result_type);
      else_expr = wrap_in_optional(else_expr, result_type);
    }
    else
    {
      // Resolve result type based on branch types
      result_type = resolve_ternary_type(
        then.type(), else_expr.type(), current_element_type);

      // Handle array-to-pointer conversion for ternary expressions
      // When assigning to a pointer (e.g., str field), convert array branches to pointers
      if (
        then.type().is_array() && else_expr.type().is_array() && current_lhs &&
        current_lhs->type().is_pointer())
      {
        then = string_handler_.get_array_base_address(then);
        else_expr = string_handler_.get_array_base_address(else_expr);
        result_type = then.type(); // Use pointer type as result
      }
    }

    // Create fully symbolic if expression
    exprt if_expr("if", result_type);
    if_expr.copy_to_operands(cond, then, else_expr);
    return if_expr;
  }

  // Create if or while code
  codet code;
  if (type == "If")
    code.set_statement("ifthenelse");
  else if (type == "While")
    code.set_statement("while");

  // Set location for the conditional statement
  code.location() = get_location_from_decl(ast_node);

  // Append "then" block
  code.copy_to_operands(cond, then);
  if (!else_expr.id_string().empty())
    code.copy_to_operands(else_expr);

  return code;
}
void python_converter::get_return_statements(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  if (ast_node["value"].is_null())
  {
    // Handle bare return statement (return with no value)
    locationt location = get_location_from_decl(ast_node);
    code_returnt return_code;
    return_code.location() = location;

    // If the function returns Optional, wrap None in Optional struct
    if (current_func_return_type_.is_struct())
    {
      const struct_typet &st = to_struct_type(current_func_return_type_);
      if (st.tag().as_string().starts_with("tag-Optional_"))
      {
        constant_exprt none_expr(none_type());
        return_code.return_value() =
          wrap_in_optional(none_expr, current_func_return_type_);
      }
    }

    target_block.copy_to_operands(return_code);
    return;
  }

  exprt return_value = get_expr(ast_node["value"]);
  locationt location = get_location_from_decl(ast_node);

  // Check if return value is a function call
  // get_function_call() returns code_function_callt (code statement), not side_effect_expr_function_callt
  bool is_func_call =
    return_value.is_code() && return_value.get("statement") == "function_call";

  if (is_func_call)
  {
    // Extract function name for temporary variable naming.
    // get_expr() also returns a function-call expression when a Subscript
    // dispatches to a user-defined __getitem__ (see GitHub #4541); in that
    // case the AST node type is "Subscript" rather than "Call", but the
    // returned code is still a function_call that needs the same temp-LHS
    // materialisation, otherwise the call expression ends up embedded
    // directly in the GOTO RETURN and trips value-set's make_member
    // assertion at value_set.cpp:1543.
    const std::string ast_type = ast_node["value"]["_type"].get<std::string>();
    std::string func_name = "func";
    if (ast_type == "Call")
    {
      if (ast_node["value"]["func"]["_type"] == "Name")
        func_name = ast_node["value"]["func"]["id"].get<std::string>();
      else if (ast_node["value"]["func"]["_type"] == "Attribute")
        func_name = ast_node["value"]["func"]["attr"].get<std::string>();
    }
    else if (ast_type == "Subscript")
      func_name = "__getitem__";

    // Determine return type: check if it's empty (forward reference)
    typet return_type = return_value.type();

    if (return_type.is_empty() || return_type.id() == typet::t_empty)
    {
      // Forward reference: function not yet processed
      // Look up return type from AST
      const auto &func_node =
        json_utils::find_function((*ast_json)["body"], func_name);

      if (
        !func_node.empty() && func_node.contains("returns") &&
        !func_node["returns"].is_null())
        return_type = get_type_from_annotation(func_node["returns"], func_node);
      else
      {
        // Default to void* if we can't determine the type
        return_type = any_type();
      }
    }

    // Create temporary variable to store function call result
    symbolt temp_symbol =
      create_return_temp_variable(return_type, location, func_name);
    symbol_table_.add(temp_symbol);
    exprt temp_var_expr = symbol_expr(temp_symbol);

    // Create declaration for temporary variable
    code_declt temp_decl(temp_var_expr);
    temp_decl.location() = location;
    target_block.copy_to_operands(temp_decl);

    // If a constructor is being invoked, the temporary variable is passed as 'self'
    // For constructors, we don't set LHS because they modify the object through
    // the first parameter (self), not through LHS
    bool is_constructor = type_handler_.is_constructor_call(ast_node["value"]);

    // Set the LHS of the function call to our temporary variable (only for non-constructors)
    if (!return_type.is_empty() && !is_constructor)
      return_value.op0() = temp_var_expr;

    if (is_constructor)
    {
      code_function_callt &call =
        static_cast<code_function_callt &>(return_value);

      // Strip any temporary $ctor_self$ parameters and add correct self
      exprt::operandst filtered_args =
        function_call_expr::strip_ctor_self_parameters(call.arguments());
      exprt::operandst new_args;
      new_args.push_back(gen_address_of(temp_var_expr));
      for (const auto &arg : filtered_args)
        new_args.push_back(arg);
      call.arguments() = new_args;
      update_instance_from_self(
        func_name, func_name, temp_var_expr.identifier().as_string());
    }

    // Add the function call statement to the block
    target_block.copy_to_operands(return_value);

    // Wrap in Optional if the function returns Optional
    exprt ret_expr = temp_var_expr;
    if (current_func_return_type_.is_struct())
    {
      const struct_typet &st = to_struct_type(current_func_return_type_);
      if (st.tag().as_string().starts_with("tag-Optional_"))
        ret_expr = wrap_in_optional(ret_expr, current_func_return_type_);
    }

    // Return the temporary variable
    code_returnt return_code;
    return_code.return_value() = ret_expr;
    return_code.location() = location;
    target_block.copy_to_operands(return_code);
  }
  else
  {
    // If we're returning an array but the function expects a pointer,
    // convert the array to a pointer (for string literals).
    const typet &expected_return_type = current_func_return_type_;

    if (expected_return_type.is_pointer() && return_value.type().is_array())
    {
      // For constant array literals (string literals), convert to string_constantt
      if (return_value.is_constant())
      {
        // Extract the string content from the constant array
        std::string str_content;
        for (const auto &operand : return_value.operands())
        {
          if (operand.is_constant())
          {
            BigInt char_val = binary2integer(
              operand.value().as_string(), operand.type().is_signedbv());
            if (char_val == 0)
              break; // Stop at null terminator
            str_content += static_cast<char>(char_val.to_int64());
          }
        }

        // Create a string_constantt with proper type
        typet string_type = return_value.type();
        return_value = string_constantt(
          str_content, string_type, string_constantt::k_default);

        // Get its address (converts array to pointer)
        return_value = address_of_exprt(return_value);
      }
      else
      {
        // For non-constant arrays (variables), convert to pointer
        return_value = string_handler_.get_array_base_address(return_value);
      }
    }

    // When returning a class-typed parameter (internally A*), dereference it
    // so the return type matches the annotation (A).  This is needed because
    // user-defined class parameters are modelled as pointers internally for
    // Python object reference semantics, but callers expect a value return.
    if (return_value.type().is_pointer())
    {
      typet ret_sub = return_value.type().subtype();
      typet expected = current_func_return_type_;
      if (ret_sub.id() == "symbol")
        ret_sub = ns.follow(ret_sub);
      if (expected.id() == "symbol")
        expected = ns.follow(expected);
      if (ret_sub.is_struct() && expected.is_struct())
      {
        const struct_typet &rs = to_struct_type(ret_sub);
        const struct_typet &es = to_struct_type(expected);
        if (rs.tag() == es.tag())
        {
          exprt deref("dereference");
          deref.type() = return_value.type().subtype();
          deref.copy_to_operands(return_value);
          return_value = deref;
        }
      }
    }

    // Wrap return value in Optional if the function returns Optional
    if (current_func_return_type_.is_struct())
    {
      const struct_typet &st = to_struct_type(current_func_return_type_);
      if (st.tag().as_string().starts_with("tag-Optional_"))
        return_value =
          wrap_in_optional(return_value, current_func_return_type_);
    }

    code_returnt return_code;
    return_code.return_value() = return_value;
    return_code.location() = location;
    target_block.copy_to_operands(return_code);
  }
}

exprt python_converter::get_block(const nlohmann::json &ast_block)
{
  code_blockt block, *old_block = current_block;
  current_block = &block;

  // Iterate over block statements
  for (auto &element : ast_block)
  {
    StatementType type = python_frontend::get_statement_type(element);

    switch (type)
    {
    case StatementType::VARIABLE_ASSIGN:
    {
      // Add an assignment to the block
      get_var_assign(element, block);
      break;
    }
    case StatementType::IF_STATEMENT:
    case StatementType::WHILE_STATEMENT:
    {
      exprt cond = get_conditional_stm(element);
      block.copy_to_operands(cond);
      break;
    }
    case StatementType::FOR_STATEMENT:
    {
      // For loops are transformed to while loops by the preprocessor
      // This case should not be reached in normal operation
      throw std::runtime_error(
        "For loops should be preprocessed before reaching converter");
    }
    case StatementType::COMPOUND_ASSIGN:
    {
      get_compound_assign(element, block);
      break;
    }
    case StatementType::FUNC_DEFINITION:
    {
      get_function_definition(element);
      global_declarations.clear();
      local_loads.clear();
      break;
    }
    case StatementType::RETURN:
    {
      get_return_statements(element, block);
      break;
    }
    case StatementType::ASSERT:
    {
      current_element_type = bool_type();
      exprt test = get_expr(element["test"]);
      if (test.statement() == "cpp-throw")
      {
        test.location() = get_location_from_decl(element);
        codet code_expr("expression");
        code_expr.operands().push_back(test);
        block.move_to_operands(code_expr);
        break;
      }

      // Convert dictionary to boolean (truthiness check)
      if (dict_handler_->is_dict_type(test.type()))
      {
        locationt location = get_location_from_decl(element);
        typet list_type = type_handler_.get_list_type();

        // Get dict.keys member
        member_exprt keys_member(test, "keys", list_type);

        // Find __ESBMC_list_size function
        const symbolt *size_func =
          symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
        if (!size_func)
          throw std::runtime_error(
            "__ESBMC_list_size not found for dict truthiness check");

        // Create temporary variable to store the size result
        symbolt &size_result = create_tmp_symbol(
          element, "$dict_size$", size_type(), gen_zero(size_type()));
        code_declt size_decl(symbol_expr(size_result));
        size_decl.location() = location;
        block.copy_to_operands(size_decl);

        // Call __ESBMC_list_size(dict.keys)
        code_function_callt size_call;
        size_call.function() = symbol_expr(*size_func);
        size_call.lhs() = symbol_expr(size_result);
        size_call.arguments().push_back(keys_member);
        size_call.type() = size_type();
        size_call.location() = location;
        block.copy_to_operands(size_call);

        // Replace test with: size != 0 (non-empty dict is truthy)
        exprt is_not_empty("notequal", bool_type());
        is_not_empty.copy_to_operands(
          symbol_expr(size_result), gen_zero(size_type()));
        is_not_empty.location() = location;
        test = is_not_empty;
      }

      // Attach assertion message if present
      auto attach_assert_message = [&element](code_assertt &assert_code) {
        if (element.contains("msg") && !element["msg"].is_null())
        {
          std::string msg;
          if (
            element["msg"]["_type"] == "Constant" &&
            element["msg"]["value"].is_string())
          {
            msg = element["msg"]["value"].get<std::string>();
          }
          else if (element["msg"]["_type"] == "JoinedStr")
          {
            // For f-strings, this is just a placeholder
            // TODO: Full f-string evaluation would require more complex handling
            msg = "<formatted string message>";
          }

          if (!msg.empty())
            assert_code.location().comment(msg);
        }
      };

      // Handle list assertions
      if (
        test.type() == type_handler_.get_list_type() ||
        (test.type().is_pointer() &&
         test.type().subtype() == type_handler_.get_list_type()))
      {
        exception_handler_->handle_list_assertion(
          element, test, block, attach_assert_message);
        break;
      }

      // Check for function call assertions
      const exprt *func_call_expr = nullptr;
      bool is_negated = false;

      // Case 1: Direct function call - assert func()
      if (test.id() == "code" && test.get("statement") == "function_call")
      {
        func_call_expr = &test;
        is_negated = false;
      }
      // Case 2: Negated function call - assert not func()
      else if (
        test.id() == "not" && test.operands().size() == 1 &&
        test.operands()[0].id() == "code" &&
        test.operands()[0].get("statement") == "function_call")
      {
        func_call_expr = &test.operands()[0];
        is_negated = true;
      }

      if (func_call_expr != nullptr)
      {
        exception_handler_->handle_function_call_assertion(
          element, *func_call_expr, is_negated, block, attach_assert_message);
      }
      else
      {
        // Direct assertion
        if (!test.type().is_bool())
          test.make_typecast(current_element_type);

        code_assertt assert_code;
        assert_code.assertion() = test;
        assert_code.location() = get_location_from_decl(element);
        attach_assert_message(assert_code);
        block.move_to_operands(assert_code);
      }
      break;
    }
    case StatementType::EXPR:
    {
      // Skip yield expressions: the preprocessor inlines them into assignments.
      // Reject yield from: the preprocessor does not expand it, so reaching
      // here means the generator was not fully lowered and verification would
      // silently produce wrong results.
      if (element.contains("value") && element["value"].contains("_type"))
      {
        const auto &inner_type = element["value"]["_type"];
        if (inner_type == "Yield")
          break;
        if (inner_type == "YieldFrom")
          throw std::runtime_error(
            "'yield from' is not supported in ESBMC's Python frontend");
      }

      // Function calls are handled here
      exprt empty;
      exprt expr = get_expr(element["value"]);
      if (expr != empty)
      {
        codet code_stmt = convert_expression_to_code(expr);
        block.move_to_operands(code_stmt);
      }

      break;
    }
    case StatementType::CLASS_DEFINITION:
    {
      get_class_definition(element, block);
      break;
    }
    case StatementType::BREAK:
    {
      code_breakt break_expr;
      block.move_to_operands(break_expr);
      break;
    }
    case StatementType::CONTINUE:
    {
      code_continuet continue_expr;
      block.move_to_operands(continue_expr);
      break;
    }
    case StatementType::GLOBAL:
    {
      symbol_id sid = create_symbol_id();
      for (const auto &item : element["names"])
      {
        sid.set_object(item);
        global_declarations.push_back(sid.global_to_string());
      }
      break;
    }
    case StatementType::TRY:
    {
      exception_handler_->get_try_statement(element, block);
      break;
    }
    case StatementType::EXCEPTHANDLER:
    {
      exception_handler_->get_except_handler_statement(element, block);
      break;
    }
    case StatementType::RAISE:
    {
      exception_handler_->get_raise_statement(element, block);
      break;
    }
    case StatementType::DELETE_STATEMENT:
    {
      get_delete_statement(element, block);
      break;
    }
    /* "https://docs.python.org/3/tutorial/controlflow.html:
     * "The pass statement does nothing. It can be used when a statement
     *  is required syntactically but the program requires no action." */
    case StatementType::PASS:
    // Imports are handled by parser.py so we can just ignore here.
    case StatementType::IMPORT:
      // TODO: Raises are ignored for now. Handling case to avoid calling abort() on default.
      break;
    case StatementType::UNKNOWN:
    default:
      throw std::runtime_error(
        element["_type"].get<std::string>() + " statements are not supported");
    }
  }

  current_block = old_block;

  return block;
}

exprt python_converter::get_static_array(
  const nlohmann::json &arr,
  const typet &shape)
{
  exprt zero = gen_zero(size_type());
  exprt list = gen_zero(shape);

  unsigned int i = 0;
  for (auto &e : arr["elts"])
  {
    exprt element_expr = get_expr(e);
    list.operands().at(i++) = element_expr;
  }

  symbolt &cl = create_tmp_symbol(arr, "$compound-literal$", shape, list);

  exprt expr = symbol_expr(cl);
  code_declt decl(expr);
  decl.operands().push_back(list);
  assert(current_block);
  current_block->copy_to_operands(decl);

  return expr;
}
void python_converter::get_delete_statement(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  if (!ast_node.contains("targets") || !ast_node["targets"].is_array())
  {
    throw std::runtime_error("Delete statement missing targets");
  }

  for (const auto &target : ast_node["targets"])
  {
    if (target["_type"] == "Subscript")
    {
      exprt dict_expr = get_expr(target["value"]);
      const nlohmann::json &slice = target["slice"];

      typet dict_type = dict_expr.type();
      if (dict_expr.is_symbol())
      {
        const symbolt *sym = symbol_table_.find_symbol(dict_expr.identifier());
        if (sym)
          dict_type = sym->type;
      }

      if (dict_type.id() == "symbol")
        dict_type = ns.follow(dict_type);

      if (!dict_type.is_struct())
      {
        throw std::runtime_error(
          "del on subscript requires a dictionary (struct) type");
      }

      // Delegate to dict_handler which handles both constant and variable keys
      dict_handler_->handle_dict_delete(dict_expr, slice, target_block);
    }
    else if (target["_type"] == "Attribute")
    {
      // del obj.attr — Python semantics: remove the instance attribute so that
      // subsequent reads fall back to the class-level attribute.
      // We model this by resetting the struct member to the class default and
      // removing the instance-attribute registration.
      if (target["value"]["_type"] != "Name")
      {
        throw std::runtime_error(
          "del on nested attribute chains is not supported");
      }

      const std::string var_name = target["value"]["id"].get<std::string>();
      const std::string attr_name = target["attr"].get<std::string>();

      // Find the instance symbol (with fallback to global scope).
      symbol_id inst_sid = create_symbol_id();
      inst_sid.set_object(var_name);
      symbolt *inst_sym = find_symbol(inst_sid.to_string());
      if (!inst_sym)
      {
        inst_sid.set_function("");
        inst_sym = find_symbol(inst_sid.to_string());
      }
      if (!inst_sym)
      {
        throw std::runtime_error(
          "del attribute: instance variable '" + var_name + "' not found");
      }

      // Determine the class struct type from the instance symbol type.
      const typet &sym_type =
        inst_sym->type.is_pointer() ? inst_sym->type.subtype() : inst_sym->type;
      typet resolved = sym_type;
      if (resolved.id() == "symbol")
        resolved = ns.follow(resolved);
      if (resolved.id() != "struct")
      {
        throw std::runtime_error(
          "del attribute: '" + var_name + "' is not a struct instance");
      }

      const struct_typet &struct_type = to_struct_type(resolved);
      const std::string class_tag = struct_type.tag().as_string();
      const std::string class_name = extract_class_name_from_tag(class_tag);

      // Look up the authoritative class-type symbol so we see any dynamically
      // added components (e.g. added during a.x = 2 processing).
      const std::string class_tag_id = "tag-" + class_tag;
      const symbolt *class_type_sym = symbol_table_.find_symbol(class_tag_id);
      const struct_typet &class_struct =
        class_type_sym ? to_struct_type(class_type_sym->type) : struct_type;

      // Find the class-level attribute symbol (the default value to restore).
      symbol_id class_sid = create_symbol_id();
      class_sid.set_function("");
      class_sid.set_class(class_name);
      class_sid.set_object(attr_name);
      symbolt *class_attr_sym =
        symbol_table_.find_symbol(class_sid.to_string());
      if (!class_attr_sym)
      {
        throw std::runtime_error(
          "del attribute: class '" + class_name +
          "' has no class-level attribute '" + attr_name + "'");
      }

      // Emit: obj.attr = ClassName::attr  (restore class default)
      if (class_struct.has_component(attr_name))
      {
        const typet &attr_type = class_struct.get_component(attr_name).type();
        exprt lhs = create_member_expression(*inst_sym, attr_name, attr_type);
        exprt rhs = symbol_expr(*class_attr_sym);
        if (rhs.type() != lhs.type())
          rhs = typecast_exprt(rhs, lhs.type());
        code_assignt assign(lhs, rhs);
        target_block.copy_to_operands(assign);
      }

      // Unregister the instance attribute so future reads fall back to the
      // class-level symbol instead of the (now-reset) struct member.
      auto map_it = instance_attr_map.find(inst_sym->id.as_string());
      if (map_it != instance_attr_map.end())
        map_it->second.erase(attr_name);
    }
    else if (target["_type"] == "Name")
    {
      log_warning("del on simple variables is not fully supported");
    }
    else
    {
      throw std::runtime_error(
        "Delete statement target type not supported: " +
        target["_type"].get<std::string>());
    }
  }
}

