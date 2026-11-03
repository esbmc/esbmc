#include "python_dict_internal.h"
#include <util/lang/c_sizeof.h>

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
  // Step 1: Cast void* to pointer_type* and dereference to get pointer_type
  // value
  exprt as_ptr_type_ptr =
    build_typecast(obj_value, pointer_typet(pointer_type()));
  exprt ptr_as_ptr_type = build_dereference(as_ptr_type_ptr, pointer_type());

  // Step 2: Store pointer_type value in temporary to ensure proper evaluation
  // order
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
  const symbolt *push_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_push_dict_ptr");
  if (!push_func)
    throw std::runtime_error("__ESBMC_list_push_dict_ptr not found");

  // The value may be a local of a function that returns the outer dict, so
  // store a pointer to a heap copy rather than its address (#9306).
  code_blockt block;
  const symbolt &ptr_var = heap_copy_dict(element, value_expr, location, block);

  code_function_callt push_call;
  push_call.function() = build_symbol(*push_func);
  push_call.arguments() = {
    build_symbol(values_list),
    build_symbol(ptr_var),
    nested_dict_type_hash(value_expr.type())};
  push_call.type() = bool_type();
  push_call.location() = location;
  block.copy_to_operands(push_call);
  converter_.add_instruction(block);
}

constant_exprt
python_dict_handler::nested_dict_type_hash(const typet &dict_type)
{
  constant_exprt type_hash(size_type());
  type_hash.set_value(integer2binary(
    generate_nested_dict_type_hash(dict_type), config.ansi_c.address_width));
  return type_hash;
}

const symbolt &python_dict_handler::heap_copy_dict(
  const nlohmann::json &node,
  const exprt &value,
  const locationt &location,
  codet &target_block)
{
  const symbolt *alloca_func = symbol_table_.find_symbol("c:@F@__ESBMC_alloca");
  if (!alloca_func)
    throw std::runtime_error("__ESBMC_alloca not found");

  const typet &dict_type = value.type();
  pointer_typet ptr_type(dict_type);
  symbolt &ptr_var =
    converter_.create_tmp_symbol(node, "$dict_value_ptr$", ptr_type, exprt());
  code_declt ptr_decl(build_symbol(ptr_var));
  ptr_decl.location() = location;
  target_block.copy_to_operands(ptr_decl);

  // sizeof(T) lets goto_convert type the allocated object as the dict.
  exprt size("sizeof", size_type());
  size.copy_to_operands(
    type_exprt(dict_type), c_sizeof(dict_type, converter_.name_space()));
  code_function_callt alloc_call;
  alloc_call.function() = build_symbol(*alloca_func);
  alloc_call.lhs() = build_symbol(ptr_var);
  alloc_call.arguments().push_back(size);
  alloc_call.type() = ptr_type;
  alloc_call.location() = location;
  target_block.copy_to_operands(alloc_call);

  code_assignt copy_assign(
    build_dereference(build_symbol(ptr_var), dict_type), value);
  copy_assign.location() = location;
  target_block.copy_to_operands(copy_assign);
  return ptr_var;
}

void python_dict_handler::store_dict_value_by_pointer(
  const nlohmann::json &node,
  const exprt &value,
  const exprt &values_member,
  const exprt &index,
  const locationt &location,
  codet &target_block,
  code_function_callt &set_call,
  code_function_callt &push_call)
{
  const symbolt *set_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_set_dict_ptr");
  const symbolt *push_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_push_dict_ptr");
  if (!set_func || !push_func)
    throw std::runtime_error("dict value store model functions not found");

  // The scalar byte copy would move the dict's list pointers through
  // uint64_t, losing their provenance, and the address of `value` dies with
  // its scope, so store a pointer to a heap copy.
  const symbolt &ptr_var = heap_copy_dict(node, value, location, target_block);
  const constant_exprt type_hash = nested_dict_type_hash(value.type());

  set_call = code_function_callt();
  set_call.function() = build_symbol(*set_func);
  set_call.arguments() = {
    values_member, index, build_symbol(ptr_var), type_hash};
  set_call.type() = bool_type();
  set_call.location() = location;

  push_call = code_function_callt();
  push_call.function() = build_symbol(*push_func);
  push_call.arguments() = {values_member, build_symbol(ptr_var), type_hash};
  push_call.type() = bool_type();
  push_call.location() = location;
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

  // obj_value = item->value = the dict* stored directly by
  // __ESBMC_list_push_dict_ptr Cast void* to dict* directly (no byte extraction
  // needed)
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
