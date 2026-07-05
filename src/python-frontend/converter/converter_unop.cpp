#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_utils.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <irep2/irep2_utils.h>
#include <util/migrate.h>
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

    // Get dict.keys member. V.3: IREP2 member access (exact round-trip of
    // member_exprt); `unary_sub` is dict-typed (is_dict_type ⇒ struct), so the
    // member2t source precondition holds.
    expr2tc dict2;
    migrate_expr(unary_sub, dict2);
    exprt keys_member =
      migrate_expr_back(member2tc(migrate_type(list_type), dict2, "keys"));

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

    // Return comparison: size == 0 (empty dict is truthy for 'not'). Phase 4.4:
    // build the returned comparison in IREP2 and lower once at the return. The
    // size_decl/size_call statements above stay legacy — their operands feed
    // legacy code_*t shells, so moving them in isolation buys nothing (P1).
    expr2tc is_empty2 = equality2tc(
      symbol_expr2tc(size_result), gen_zero(migrate_type(size_type())));
    exprt is_empty = migrate_expr_back(is_empty2);
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

    // Phase 4.4 (first wiring): build the `strlen(base) == 0` result in IREP2
    // internally via the Phase 4.2 helpers, lowering to legacy `exprt` at a
    // single seam (the `return`). The call argument comes back from the string
    // handler as a legacy `exprt`, so migrate it forward; the whole subtree is
    // then IREP2 until the one back-migration. This is statement-free, so no
    // legacy `codet` shell is involved (P1 untouched).
    const type2tc size_t2 = migrate_type(size_type());
    expr2tc base2;
    migrate_expr(string_handler_.get_array_base_address(unary_sub), base2);

    expr2tc strlen_call2 = side_effect_function_call2tc(
      size_t2, symbol_expr2tc(*strlen_sym), {base2});

    expr2tc is_empty2 = equality2tc(strlen_call2, gen_zero(size_t2));

    exprt is_empty = migrate_expr_back(is_empty2);
    is_empty.location() = location;
    return is_empty;
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

    // Return comparison: size == 0 (empty list is falsy, so 'not list' is true
    // when empty). Phase 4.4: build in IREP2, lower once at the return; the
    // size_decl/size_call statements above stay legacy (P1).
    expr2tc is_empty2 = equality2tc(
      symbol_expr2tc(size_result), gen_zero(migrate_type(size_type())));
    exprt is_empty = migrate_expr_back(is_empty2);
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

  // Python's `not` always yields a bool, regardless of its operand's type.
  // Type the node accordingly (matching every other `not` site in the
  // frontend) so it is well-formed IR: migrate_expr asserts that `not`/`and`/
  // `or` nodes are bool-typed, and consumers that migrate eagerly (e.g.
  // build_typecast) would otherwise abort before the C adjuster normalises it.
  const typet result_type = (op == "Not") ? bool_type() : type;
  const std::string op_id = python_frontend::map_operator(op, result_type);

  // V.3: build the generic unary node directly in IREP2, back-migrating once.
  // The Python AST has exactly four unary operators and map_operator covers
  // each: Not→"not", USub→"unary-", Invert→"bitnot", UAdd→"unary+". The
  // factories below reproduce migrate_expr's lowering of those ids verbatim
  // (not2t fixes a bool result; neg2t/bitnot2t take the node type; unary plus
  // is the identity, dropping the wrapper exactly as migrate_expr does at
  // util/migrate.cpp:1676), so this is behaviour-preserving under the
  // mandatory --irep2-bodies round-trip.
  expr2tc operand2;
  migrate_expr(unary_sub, operand2);

  expr2tc unary2;
  if (op_id == "not")
    unary2 = not2tc(operand2);
  else if (op_id == "unary-")
    unary2 = neg2tc(migrate_type(result_type), operand2);
  else if (op_id == "bitnot")
    unary2 = bitnot2tc(migrate_type(result_type), operand2);
  else if (op_id == "unary+")
    unary2 = operand2; // unary plus is the identity
  else
  {
    // Unknown operator: preserve the legacy (malformed-node + warning) path.
    exprt unary_expr(op_id, result_type);
    unary_expr.operands().push_back(unary_sub);
    return unary_expr;
  }

  return migrate_expr_back(unary2);
}
