#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_utils.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/python_types.h>

exprt python_converter::get_unary_operator_expr(const nlohmann::json &element)
{
  typet type = current_element_type;
  if (
    element["operand"].contains("value") &&
    element["operand"]["_type"] == "Constant")
  {
    type = type_handler_.get_typet(element["operand"]["value"]);
  }
  else if (element["operand"]["_type"] == "Name")
  {
    const std::string var_type =
      type_handler_.get_var_type(element["operand"]["id"].get<std::string>());
    type = type_handler_.get_typet(var_type);
  }

  // Get the operand expression
  exprt unary_sub = get_expr(element["operand"]);

  // Use operand's exact type to preserve metadata
  if (!unary_sub.type().is_nil() && !unary_sub.type().is_empty())
  {
    std::string op = element["op"]["_type"].get<std::string>();
    if (op == "USub" || op == "UAdd") // Unary minus/plus
      if (
        unary_sub.type().is_floatbv() ||
        type_utils::is_integer_type(unary_sub.type()))
        type = unary_sub.type();
  }

  // Handle 'not' operator on dictionary types: convert to emptiness check
  std::string op = element["op"]["_type"].get<std::string>();
  if (op == "Not" && dict_handler_->is_dict_type(unary_sub.type()))
  {
    if (!current_block)
      throw std::runtime_error(
        "Dictionary truthiness check requires a statement context");

    locationt location = get_location_from_decl(element);
    typet list_type = type_handler_.get_list_type();

    // Get dict.keys member
    member_exprt keys_member(unary_sub, "keys", list_type);

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
    current_block->copy_to_operands(size_decl);

    // Call __ESBMC_list_size(dict.keys)
    code_function_callt size_call;
    size_call.function() = symbol_expr(*size_func);
    size_call.lhs() = symbol_expr(size_result);
    size_call.arguments().push_back(keys_member);
    size_call.type() = size_type();
    size_call.location() = location;
    current_block->copy_to_operands(size_call);

    // Return comparison: size == 0 (empty dict is truthy for 'not')
    exprt is_empty("=", bool_type());
    is_empty.copy_to_operands(symbol_expr(size_result), gen_zero(size_type()));
    is_empty.location() = location;

    return is_empty;
  }

  // Handle 'not' operator on string types: convert to strlen(a) == 0.
  // In Python, a non-empty string is truthy; only "" is falsy.
  if (op == "Not" && type_utils::is_string_type(unary_sub.type()))
  {
    locationt location = get_location_from_decl(element);
    const symbolt *strlen_sym = symbol_table_.find_symbol("c:@F@strlen");
    if (!strlen_sym)
      throw std::runtime_error("strlen not found for string truthiness check");

    side_effect_expr_function_callt strlen_call;
    strlen_call.function() = symbol_expr(*strlen_sym);
    strlen_call.arguments().push_back(
      string_handler_.get_array_base_address(unary_sub));
    strlen_call.type() = size_type();
    strlen_call.location() = location;

    return equality_exprt(strlen_call, gen_zero(size_type()));
  }

  // Handle 'not' operator on list types: convert to emptiness check
  typet list_type = type_handler_.get_list_type();
  if (
    op == "Not" && (unary_sub.type() == list_type ||
                    (unary_sub.type().is_pointer() &&
                     unary_sub.type().subtype() == list_type)))
  {
    if (!current_block)
      throw std::runtime_error(
        "List truthiness check requires a statement context");

    locationt location = get_location_from_decl(element);

    // Find __ESBMC_list_size function
    const symbolt *size_func =
      symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
    if (!size_func)
      throw std::runtime_error(
        "__ESBMC_list_size not found for list truthiness check");

    // Create temporary variable to store the size result
    symbolt &size_result = create_tmp_symbol(
      element, "$list_size$", size_type(), gen_zero(size_type()));
    code_declt size_decl(symbol_expr(size_result));
    size_decl.location() = location;
    current_block->copy_to_operands(size_decl);

    // Call __ESBMC_list_size(list)
    code_function_callt size_call;
    size_call.function() = symbol_expr(*size_func);
    size_call.lhs() = symbol_expr(size_result);
    // Pass address if not already a pointer
    if (unary_sub.type().is_pointer())
      size_call.arguments().push_back(unary_sub);
    else
      size_call.arguments().push_back(address_of_exprt(unary_sub));
    size_call.type() = size_type();
    size_call.location() = location;
    current_block->copy_to_operands(size_call);

    // Return comparison: size == 0 (empty list is falsy, so 'not list' is true when empty)
    exprt is_empty =
      equality_exprt(symbol_expr(size_result), gen_zero(size_type()));
    is_empty.location() = location;

    return is_empty;
  }

  {
    exprt dunder_result = dispatch_unary_dunder_operator(
      op, unary_sub, get_location_from_decl(element));
    if (!dunder_result.is_nil())
      return dunder_result;
  }

  // Built-in complex arithmetic: unary + and - operate component-wise.
  if (is_complex_type(unary_sub.type()) && (op == "USub" || op == "UAdd"))
    return complex_handler_.handle_unary_op(op, unary_sub);

  exprt unary_expr(python_frontend::map_operator(op, type), type);
  unary_expr.operands().push_back(unary_sub);

  return unary_expr;
}
