#include <python-frontend/python_set.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_list.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/std_code.h>
#include <functional>

symbolt &python_set::create_set_list()
{
  locationt location = converter_.get_location_from_decl(set_value_);
  const type_handler &type_handler = converter_.get_type_handler();

  // Create list symbol for set representation
  const typet list_type = type_handler.get_list_type();
  symbolt &list_symbol =
    converter_.create_tmp_symbol(set_value_, "$py_set$", list_type, exprt());

  // Declare list
  code_declt list_decl(symbol_expr(list_symbol));
  list_decl.location() = location;
  converter_.add_instruction(list_decl);

  // Initialize list with storage array
  const symbolt *create_func_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_create");
  assert(create_func_sym);

  // Add list_create call to the block
  code_function_callt list_create_func_call;
  list_create_func_call.function() = symbol_expr(*create_func_sym);
  list_create_func_call.lhs() = symbol_expr(list_symbol);
  list_create_func_call.type() = list_type;
  list_create_func_call.location() = location;
  converter_.add_instruction(list_create_func_call);

  // Mark as set
  list_symbol.is_set = true;

  return list_symbol;
}

exprt python_set::get()
{
  symbolt &set_symbol = create_set_list();
  const std::string &set_id = set_symbol.id.as_string();

  // Create a python_list helper to reuse push functionality
  python_list list_helper(converter_, set_value_);

  // Track unique elements (sets don't allow duplicates)
  std::set<exprt> elements;

  for (auto &e : set_value_["elts"])
  {
    exprt elem = converter_.get_expr(e);

    // Skip duplicates
    if (elements.count(elem))
      continue;

    elements.insert(elem);

    // Use list push functionality
    exprt set_push_func_call =
      list_helper.build_push_list_call(set_symbol, set_value_, elem);
    converter_.add_instruction(set_push_func_call);

    // Track type information
    list_helper.add_type_info(
      set_id, elem.identifier().as_string(), elem.type());
  }

  return symbol_expr(set_symbol);
}

exprt python_set::get_empty_set()
{
  // Create an empty list structure for the set
  symbolt &set_symbol = create_set_list();

  // No elements to add for empty set
  // Type information will be determined when elements are added
  return symbol_expr(set_symbol);
}

exprt python_set::build_set_difference_call(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &element)
{
  symbolt &result_set = create_set_list();
  locationt loc = converter_.get_location_from_decl(element);

  const symbolt *size_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  const symbolt *at_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_at");
  const symbolt *contains_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_contains");
  const symbolt *push_obj_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push_object");

  assert(size_func && at_func && contains_func && push_obj_func);

  // Get size of LHS set
  symbolt &n_sym = converter_.create_tmp_symbol(
    element, "$lhs_size$", size_type(), gen_zero(size_type()));
  code_declt n_decl(symbol_expr(n_sym));
  converter_.add_instruction(n_decl);

  code_function_callt get_size;
  get_size.function() = symbol_expr(*size_func);
  get_size.arguments().push_back(
    lhs.type().is_pointer() ? lhs : address_of_exprt(lhs));
  get_size.lhs() = symbol_expr(n_sym);
  get_size.type() = size_type();
  get_size.location() = loc;
  converter_.add_instruction(get_size);

  // Loop counter
  symbolt &i_sym = converter_.create_tmp_symbol(
    element, "$i$", size_type(), gen_zero(size_type()));
  code_declt i_decl(symbol_expr(i_sym));
  converter_.add_instruction(i_decl);

  code_assignt i_init(symbol_expr(i_sym), gen_zero(size_type()));
  converter_.add_instruction(i_init);

  // Loop condition: i < n
  exprt cond("<", bool_type());
  cond.copy_to_operands(symbol_expr(i_sym), symbol_expr(n_sym));

  code_blockt body;

  // Get element at index i from LHS
  side_effect_expr_function_callt at_call;
  at_call.function() = symbol_expr(*at_func);
  at_call.arguments().push_back(
    lhs.type().is_pointer() ? lhs : address_of_exprt(lhs));
  at_call.arguments().push_back(symbol_expr(i_sym));
  at_call.type() =
    pointer_typet(converter_.get_type_handler().get_list_element_type());
  at_call.location() = loc;

  symbolt &elem_sym = converter_.create_tmp_symbol(
    element,
    "$elem$",
    pointer_typet(converter_.get_type_handler().get_list_element_type()),
    exprt());
  code_declt elem_decl(symbol_expr(elem_sym));
  elem_decl.copy_to_operands(at_call);
  body.copy_to_operands(elem_decl);

  // Check if element is in RHS
  symbolt &contains_result = converter_.create_tmp_symbol(
    element, "$contains$", bool_type(), gen_boolean(false));
  code_declt contains_decl(symbol_expr(contains_result));
  body.copy_to_operands(contains_decl);

  code_function_callt contains_call;
  contains_call.function() = symbol_expr(*contains_func);
  contains_call.lhs() = symbol_expr(contains_result);
  contains_call.arguments().push_back(
    rhs.type().is_pointer() ? rhs : address_of_exprt(rhs));

  // Extract element value
  member_exprt elem_value(
    symbol_expr(elem_sym), "value", pointer_typet(empty_typet()));
  {
    exprt &base = elem_value.struct_op();
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
  contains_call.arguments().push_back(elem_value);

  // Extract element type_id
  member_exprt elem_type_id(symbol_expr(elem_sym), "type_id", size_type());
  {
    exprt &base = elem_type_id.struct_op();
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
  contains_call.arguments().push_back(elem_type_id);

  // Extract element size
  member_exprt elem_size(symbol_expr(elem_sym), "size", size_type());
  {
    exprt &base = elem_size.struct_op();
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
  contains_call.arguments().push_back(elem_size);

  const std::string string_type_name = "__python_str";
  constant_exprt string_type_id(size_type());
  string_type_id.set_value(integer2binary(
    std::hash<std::string>{}(string_type_name), config.ansi_c.address_width));
  contains_call.arguments().push_back(string_type_id);

  contains_call.type() = bool_type();
  contains_call.location() = loc;
  body.copy_to_operands(contains_call);

  // If element NOT in RHS, add to result
  code_blockt then_block;

  side_effect_expr_function_callt push_call;
  push_call.function() = symbol_expr(*push_obj_func);
  push_call.arguments().push_back(symbol_expr(result_set));
  push_call.arguments().push_back(symbol_expr(elem_sym));
  push_call.type() = bool_type();
  push_call.location() = loc;
  then_block.copy_to_operands(converter_.convert_expression_to_code(push_call));

  exprt not_contains("not", bool_type());
  not_contains.copy_to_operands(symbol_expr(contains_result));

  codet if_stmt;
  if_stmt.set_statement("ifthenelse");
  if_stmt.copy_to_operands(not_contains, then_block);
  body.copy_to_operands(if_stmt);

  // Increment counter
  plus_exprt i_inc(symbol_expr(i_sym), gen_one(size_type()));
  code_assignt i_step(symbol_expr(i_sym), i_inc);
  body.copy_to_operands(i_step);

  // Create loop
  codet loop;
  loop.set_statement("while");
  loop.copy_to_operands(cond, body);
  converter_.add_instruction(loop);

  return symbol_expr(result_set);
}

exprt python_set::build_set_intersection_call(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &element)
{
  symbolt &result_set = create_set_list();
  locationt loc = converter_.get_location_from_decl(element);

  const symbolt *size_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  const symbolt *at_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_at");
  const symbolt *contains_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_contains");
  const symbolt *push_obj_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push_object");

  assert(size_func && at_func && contains_func && push_obj_func);

  // Get size of LHS set
  symbolt &n_sym = converter_.create_tmp_symbol(
    element, "$lhs_size$", size_type(), gen_zero(size_type()));
  code_declt n_decl(symbol_expr(n_sym));
  converter_.add_instruction(n_decl);

  code_function_callt get_size;
  get_size.function() = symbol_expr(*size_func);
  get_size.arguments().push_back(
    lhs.type().is_pointer() ? lhs : address_of_exprt(lhs));
  get_size.lhs() = symbol_expr(n_sym);
  get_size.type() = size_type();
  get_size.location() = loc;
  converter_.add_instruction(get_size);

  // Loop counter
  symbolt &i_sym = converter_.create_tmp_symbol(
    element, "$i$", size_type(), gen_zero(size_type()));
  code_declt i_decl(symbol_expr(i_sym));
  converter_.add_instruction(i_decl);

  code_assignt i_init(symbol_expr(i_sym), gen_zero(size_type()));
  converter_.add_instruction(i_init);

  // Loop condition: i < n
  exprt cond("<", bool_type());
  cond.copy_to_operands(symbol_expr(i_sym), symbol_expr(n_sym));

  code_blockt body;

  // Get element at index i from LHS
  side_effect_expr_function_callt at_call;
  at_call.function() = symbol_expr(*at_func);
  at_call.arguments().push_back(
    lhs.type().is_pointer() ? lhs : address_of_exprt(lhs));
  at_call.arguments().push_back(symbol_expr(i_sym));
  at_call.type() =
    pointer_typet(converter_.get_type_handler().get_list_element_type());
  at_call.location() = loc;

  symbolt &elem_sym = converter_.create_tmp_symbol(
    element,
    "$elem$",
    pointer_typet(converter_.get_type_handler().get_list_element_type()),
    exprt());
  code_declt elem_decl(symbol_expr(elem_sym));
  elem_decl.copy_to_operands(at_call);
  body.copy_to_operands(elem_decl);

  // Check if element is in RHS
  symbolt &contains_result = converter_.create_tmp_symbol(
    element, "$contains$", bool_type(), gen_boolean(false));
  code_declt contains_decl(symbol_expr(contains_result));
  body.copy_to_operands(contains_decl);

  code_function_callt contains_call;
  contains_call.function() = symbol_expr(*contains_func);
  contains_call.lhs() = symbol_expr(contains_result);
  contains_call.arguments().push_back(
    rhs.type().is_pointer() ? rhs : address_of_exprt(rhs));

  // Extract element value
  member_exprt elem_value(
    symbol_expr(elem_sym), "value", pointer_typet(empty_typet()));
  {
    exprt &base = elem_value.struct_op();
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
  contains_call.arguments().push_back(elem_value);

  // Extract element type_id
  member_exprt elem_type_id(symbol_expr(elem_sym), "type_id", size_type());
  {
    exprt &base = elem_type_id.struct_op();
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
  contains_call.arguments().push_back(elem_type_id);

  // Extract element size
  member_exprt elem_size(symbol_expr(elem_sym), "size", size_type());
  {
    exprt &base = elem_size.struct_op();
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
  contains_call.arguments().push_back(elem_size);

  const std::string string_type_name = "__python_str";
  constant_exprt string_type_id(size_type());
  string_type_id.set_value(integer2binary(
    std::hash<std::string>{}(string_type_name), config.ansi_c.address_width));
  contains_call.arguments().push_back(string_type_id);

  contains_call.type() = bool_type();
  contains_call.location() = loc;
  body.copy_to_operands(contains_call);

  // If element IS in RHS, add to result
  code_blockt then_block;

  side_effect_expr_function_callt push_call;
  push_call.function() = symbol_expr(*push_obj_func);
  push_call.arguments().push_back(symbol_expr(result_set));
  push_call.arguments().push_back(symbol_expr(elem_sym));
  push_call.type() = bool_type();
  push_call.location() = loc;
  then_block.copy_to_operands(converter_.convert_expression_to_code(push_call));

  codet if_stmt;
  if_stmt.set_statement("ifthenelse");
  if_stmt.copy_to_operands(symbol_expr(contains_result), then_block);
  body.copy_to_operands(if_stmt);

  // Increment counter
  plus_exprt i_inc(symbol_expr(i_sym), gen_one(size_type()));
  code_assignt i_step(symbol_expr(i_sym), i_inc);
  body.copy_to_operands(i_step);

  // Create loop
  codet loop;
  loop.set_statement("while");
  loop.copy_to_operands(cond, body);
  converter_.add_instruction(loop);

  return symbol_expr(result_set);
}

exprt python_set::build_set_union_call(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &element)
{
  symbolt &result_set = create_set_list();
  locationt loc = converter_.get_location_from_decl(element);

  // First, copy all elements from LHS
  const symbolt *extend_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_extend");
  assert(extend_func);

  code_function_callt extend_call;
  extend_call.function() = symbol_expr(*extend_func);
  extend_call.arguments().push_back(symbol_expr(result_set));
  extend_call.arguments().push_back(
    lhs.type().is_pointer() ? lhs : address_of_exprt(lhs));
  extend_call.type() = empty_typet();
  extend_call.location() = loc;
  converter_.add_instruction(extend_call);

  // Now iterate through RHS and add elements not already in result
  const symbolt *size_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  const symbolt *at_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_at");
  const symbolt *contains_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_contains");
  const symbolt *push_obj_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push_object");

  assert(size_func && at_func && contains_func && push_obj_func);

  // Get size of RHS set
  symbolt &n_sym = converter_.create_tmp_symbol(
    element, "$rhs_size$", size_type(), gen_zero(size_type()));
  code_declt n_decl(symbol_expr(n_sym));
  converter_.add_instruction(n_decl);

  code_function_callt get_size;
  get_size.function() = symbol_expr(*size_func);
  get_size.arguments().push_back(
    rhs.type().is_pointer() ? rhs : address_of_exprt(rhs));
  get_size.lhs() = symbol_expr(n_sym);
  get_size.type() = size_type();
  get_size.location() = loc;
  converter_.add_instruction(get_size);

  // Loop counter
  symbolt &i_sym = converter_.create_tmp_symbol(
    element, "$i$", size_type(), gen_zero(size_type()));
  code_declt i_decl(symbol_expr(i_sym));
  converter_.add_instruction(i_decl);

  code_assignt i_init(symbol_expr(i_sym), gen_zero(size_type()));
  converter_.add_instruction(i_init);

  // Loop condition: i < n
  exprt cond("<", bool_type());
  cond.copy_to_operands(symbol_expr(i_sym), symbol_expr(n_sym));

  code_blockt body;

  // Get element at index i from RHS
  side_effect_expr_function_callt at_call;
  at_call.function() = symbol_expr(*at_func);
  at_call.arguments().push_back(
    rhs.type().is_pointer() ? rhs : address_of_exprt(rhs));
  at_call.arguments().push_back(symbol_expr(i_sym));
  at_call.type() =
    pointer_typet(converter_.get_type_handler().get_list_element_type());
  at_call.location() = loc;

  symbolt &elem_sym = converter_.create_tmp_symbol(
    element,
    "$elem$",
    pointer_typet(converter_.get_type_handler().get_list_element_type()),
    exprt());
  code_declt elem_decl(symbol_expr(elem_sym));
  elem_decl.copy_to_operands(at_call);
  body.copy_to_operands(elem_decl);

  // Check if element is already in result
  symbolt &contains_result = converter_.create_tmp_symbol(
    element, "$contains$", bool_type(), gen_boolean(false));
  code_declt contains_decl(symbol_expr(contains_result));
  body.copy_to_operands(contains_decl);

  code_function_callt contains_call;
  contains_call.function() = symbol_expr(*contains_func);
  contains_call.lhs() = symbol_expr(contains_result);
  contains_call.arguments().push_back(symbol_expr(result_set));

  // Extract element value
  member_exprt elem_value(
    symbol_expr(elem_sym), "value", pointer_typet(empty_typet()));
  {
    exprt &base = elem_value.struct_op();
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
  contains_call.arguments().push_back(elem_value);

  // Extract element type_id
  member_exprt elem_type_id(symbol_expr(elem_sym), "type_id", size_type());
  {
    exprt &base = elem_type_id.struct_op();
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
  contains_call.arguments().push_back(elem_type_id);

  // Extract element size
  member_exprt elem_size(symbol_expr(elem_sym), "size", size_type());
  {
    exprt &base = elem_size.struct_op();
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
  contains_call.arguments().push_back(elem_size);

  const std::string string_type_name = "__python_str";
  constant_exprt string_type_id(size_type());
  string_type_id.set_value(integer2binary(
    std::hash<std::string>{}(string_type_name), config.ansi_c.address_width));
  contains_call.arguments().push_back(string_type_id);

  contains_call.type() = bool_type();
  contains_call.location() = loc;
  body.copy_to_operands(contains_call);

  // If element NOT in result, add it
  code_blockt then_block;

  side_effect_expr_function_callt push_call;
  push_call.function() = symbol_expr(*push_obj_func);
  push_call.arguments().push_back(symbol_expr(result_set));
  push_call.arguments().push_back(symbol_expr(elem_sym));
  push_call.type() = bool_type();
  push_call.location() = loc;
  then_block.copy_to_operands(converter_.convert_expression_to_code(push_call));

  exprt not_contains("not", bool_type());
  not_contains.copy_to_operands(symbol_expr(contains_result));

  codet if_stmt;
  if_stmt.set_statement("ifthenelse");
  if_stmt.copy_to_operands(not_contains, then_block);
  body.copy_to_operands(if_stmt);

  // Increment counter
  plus_exprt i_inc(symbol_expr(i_sym), gen_one(size_type()));
  code_assignt i_step(symbol_expr(i_sym), i_inc);
  body.copy_to_operands(i_step);

  // Create loop
  codet loop;
  loop.set_statement("while");
  loop.copy_to_operands(cond, body);
  converter_.add_instruction(loop);

  return symbol_expr(result_set);
}

exprt python_set::handle_operations(
  python_converter &converter,
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element)
{
  const type_handler &type_handler = converter.get_type_handler();
  typet list_type = type_handler.get_list_type();

  // Ensure both operands are lists (sets are represented as lists)
  if (lhs.type() != list_type || rhs.type() != list_type)
    return nil_exprt();

  // Resolve function calls to temporary variables
  auto resolve_list_call = [&](exprt &expr) -> bool {
    if (
      expr.id().as_string() != "sideeffect" ||
      expr.get("statement") != "function_call" || expr.type() != list_type)
      return false;

    locationt location = converter.get_location_from_decl(element);

    // Create temporary variable for the list
    symbolt &tmp_var_symbol = converter.create_tmp_symbol(
      element, "tmp_set_op", list_type, gen_zero(list_type));

    code_declt tmp_var_decl(symbol_expr(tmp_var_symbol));
    tmp_var_decl.location() = location;
    converter.add_instruction(tmp_var_decl);

    side_effect_expr_function_callt &side_effect =
      to_side_effect_expr_function_call(expr);

    code_function_callt call;
    call.function() = side_effect.function();
    call.arguments() = side_effect.arguments();
    call.lhs() = symbol_expr(tmp_var_symbol);
    call.type() = list_type;
    call.location() = location;

    converter.add_instruction(call);
    expr = symbol_expr(tmp_var_symbol);
    return true;
  };

  resolve_list_call(lhs);
  resolve_list_call(rhs);

  python_set set_handler(converter, element);

  // Map Python set operations to internal functions
  if (op == "Sub") // Set difference: a - b
    return set_handler.build_set_difference_call(lhs, rhs, element);
  else if (op == "BitAnd") // Set intersection: a & b
    return set_handler.build_set_intersection_call(lhs, rhs, element);
  else if (op == "BitOr") // Set union: a | b
    return set_handler.build_set_union_call(lhs, rhs, element);

  return nil_exprt();
}
