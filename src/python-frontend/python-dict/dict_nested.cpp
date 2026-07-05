#include "python_dict_internal.h"

using namespace python_expr;

size_t
python_dict_handler::generate_nested_dict_type_hash(const typet &dict_type)
{
  // Use the type's pretty name which includes full type information
  // e.g., "struct tag-dict_int_dict_int_int"
  std::string type_identifier = dict_type.pretty_name().as_string();

  // Fallback to ID string if pretty_name is empty
  if (type_identifier.empty())
    type_identifier = dict_type.id_string();

  // Add a prefix to avoid collisions with other type hashes
  type_identifier = "nested_dict:" + type_identifier;

  return std::hash<std::string>{}(type_identifier);
}

exprt python_dict_handler::safe_cast_to_dict_pointer(
  const nlohmann::json &node,
  const exprt &obj_value,
  const typet &target_ptr_type,
  const locationt &location)
{
  // Step 1: Cast void* to pointer_type* and dereference to get pointer_type value
  exprt as_ptr_type_ptr =
    build_typecast(obj_value, pointer_typet(pointer_type()));
  exprt ptr_as_ptr_type = build_dereference(as_ptr_type_ptr, pointer_type());

  // Step 2: Store pointer_type value in temporary to ensure proper evaluation order
  symbolt &ptr_type_var = converter_.create_tmp_symbol(
    node, "$dict_ptr_as_int$", pointer_type(), exprt());
  code_declt ptr_type_decl(build_symbol(ptr_type_var));
  ptr_type_decl.location() = location;
  converter_.add_instruction(ptr_type_decl);

  code_assignt ptr_type_assign(build_symbol(ptr_type_var), ptr_as_ptr_type);
  ptr_type_assign.location() = location;
  converter_.add_instruction(ptr_type_assign);

  // Step 3: Cast pointer_type value to target pointer type
  exprt dict_ptr = build_typecast(build_symbol(ptr_type_var), target_ptr_type);

  // Step 4: Store the typed pointer
  symbolt &dict_ptr_var = converter_.create_tmp_symbol(
    node, "$dict_ptr_typed$", target_ptr_type, exprt());
  code_declt ptr_decl(build_symbol(dict_ptr_var));
  ptr_decl.location() = location;
  converter_.add_instruction(ptr_decl);

  code_assignt ptr_assign(build_symbol(dict_ptr_var), dict_ptr);
  ptr_assign.location() = location;
  converter_.add_instruction(ptr_assign);

  return build_symbol(dict_ptr_var);
}

void python_dict_handler::store_nested_dict_value(
  const nlohmann::json &element,
  const symbolt &values_list,
  const exprt &value_expr,
  const locationt &location)
{
  // Get __ESBMC_list_push_dict_ptr which stores dict* directly (no byte copy)
  const symbolt *push_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_push_dict_ptr");

  if (!push_func)
  {
    log_error("__ESBMC_list_push_dict_ptr not found in symbol table");
    throw std::runtime_error("Required list operation function not available");
  }

  // Create pointer to the dict value
  typet ptr_type = pointer_typet(value_expr.type());
  symbolt &ptr_var = converter_.create_tmp_symbol(
    element, "$nested_dict_ptr$", ptr_type, exprt());
  code_declt ptr_decl(build_symbol(ptr_var));
  ptr_decl.location() = location;
  converter_.add_instruction(ptr_decl);

  // Store the address of the dict
  code_assignt ptr_assign(build_symbol(ptr_var), build_address_of(value_expr));
  ptr_assign.location() = location;
  converter_.add_instruction(ptr_assign);

  // Generate a proper type hash based on actual type information
  size_t type_hash_value = generate_nested_dict_type_hash(value_expr.type());
  constant_exprt type_hash(size_type());
  type_hash.set_value(
    integer2binary(type_hash_value, config.ansi_c.address_width));

  // Call __ESBMC_list_push_dict_ptr(list, ptr_var, type_hash)
  // ptr_var (dict*) is stored directly in item->value — no byte extraction needed
  code_function_callt push_call;
  push_call.function() = build_symbol(*push_func);
  push_call.arguments().push_back(build_symbol(values_list));
  push_call.arguments().push_back(build_symbol(ptr_var));
  push_call.arguments().push_back(type_hash);
  push_call.type() = bool_type();
  push_call.location() = location;

  converter_.add_instruction(push_call);
}

exprt python_dict_handler::retrieve_nested_dict_value(
  const nlohmann::json &slice_node,
  const exprt &obj_value,
  const typet &expected_type,
  const locationt &location)
{
  // Validate expected_type is actually a dict
  if (expected_type.is_nil())
  {
    throw std::runtime_error(
      "retrieve_nested_dict_value: expected_type is nil");
  }

  // obj_value = item->value = the dict* stored directly by __ESBMC_list_push_dict_ptr
  // Cast void* to dict* directly (no byte extraction needed)
  pointer_typet dict_ptr_type(expected_type);
  exprt dict_ptr = build_typecast(obj_value, dict_ptr_type);

  // Dereference to get the actual dict struct
  exprt dict_struct = build_dereference(dict_ptr, expected_type);

  // Store in final temporary for return
  symbolt &result_dict = converter_.create_tmp_symbol(
    slice_node, "$dict_retrieved$", expected_type, exprt());
  code_declt temp_decl(build_symbol(result_dict));
  temp_decl.location() = location;
  converter_.add_instruction(temp_decl);

  code_assignt result_assign(build_symbol(result_dict), dict_struct);
  result_assign.location() = location;
  converter_.add_instruction(result_assign);

  return build_symbol(result_dict);
}
