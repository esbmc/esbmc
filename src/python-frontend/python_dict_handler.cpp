#include <python-frontend/python_dict_handler.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/type_handler.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/context.h>
#include <util/std_code.h>
#include <functional>

int python_dict_handler::dict_counter_ = 0;

python_dict_handler::python_dict_handler(
  python_converter &converter,
  contextt &symbol_table,
  type_handler &type_handler)
  : converter_(converter),
    symbol_table_(symbol_table),
    type_handler_(type_handler)
{
}

bool python_dict_handler::is_dict_literal(const nlohmann::json &element) const
{
  return element.contains("_type") && element["_type"] == "Dict";
}

bool python_dict_handler::is_dict_type(const typet &type) const
{
  if (!type.is_struct())
    return false;

  const struct_typet &struct_type = to_struct_type(type);
  std::string tag = struct_type.tag().as_string();
  return tag == "__python_dict__";
}

struct_typet python_dict_handler::get_dict_struct_type()
{
  const std::string dict_type_name = "tag-__python_dict__";
  symbolt *existing = symbol_table_.find_symbol(dict_type_name);
  if (existing)
    return to_struct_type(existing->type);

  struct_typet dict_struct;
  dict_struct.tag("__python_dict__");

  typet list_type = type_handler_.get_list_type();

  struct_typet::componentt keys_comp("keys", "keys", list_type);
  keys_comp.set_access("public");
  dict_struct.components().push_back(keys_comp);

  struct_typet::componentt values_comp("values", "values", list_type);
  values_comp.set_access("public");
  dict_struct.components().push_back(values_comp);

  symbolt type_symbol;
  type_symbol.id = dict_type_name;
  type_symbol.name = dict_type_name;
  type_symbol.type = dict_struct;
  type_symbol.mode = "Python";
  type_symbol.is_type = true;
  symbol_table_.add(type_symbol);

  return dict_struct;
}

exprt python_dict_handler::get_dict_literal(const nlohmann::json &element)
{
  if (!is_dict_literal(element))
    throw std::runtime_error("Expected Dict literal");

  // For nested dictionaries, we need to create a temporary variable
  // because the dict needs to exist as a concrete symbol to be used as a value
  locationt location = converter_.get_location_from_decl(element);
  std::string dict_name = "$py_dict$" + std::to_string(dict_counter_++);

  struct_typet dict_type = get_dict_struct_type();

  // Create a temporary symbol for this dict literal
  symbolt &dict_sym =
    converter_.create_tmp_symbol(element, dict_name, dict_type, exprt());

  code_declt dict_decl(symbol_expr(dict_sym));
  dict_decl.location() = location;
  converter_.add_instruction(dict_decl);

  // Initialize the dictionary with its literal values
  create_dict_from_literal(element, symbol_expr(dict_sym));

  // Return the symbol expression pointing to the initialized dictionary
  return symbol_expr(dict_sym);
}

exprt python_dict_handler::create_dict_from_literal(
  const nlohmann::json &element,
  const exprt &target_symbol)
{
  locationt location = converter_.get_location_from_decl(element);
  std::string dict_name = "$py_dict$" + std::to_string(dict_counter_++);

  struct_typet dict_type = get_dict_struct_type();
  typet list_type = type_handler_.get_list_type();

  // Create keys list
  symbolt &keys_list = converter_.create_tmp_symbol(
    element, dict_name + "_keys", list_type, exprt());

  code_declt keys_decl(symbol_expr(keys_list));
  keys_decl.location() = location;
  converter_.add_instruction(keys_decl);

  const symbolt *create_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_create");
  if (!create_func)
    throw std::runtime_error("__ESBMC_list_create not found");

  code_function_callt keys_create;
  keys_create.function() = symbol_expr(*create_func);
  keys_create.lhs() = symbol_expr(keys_list);
  keys_create.type() = list_type;
  keys_create.location() = location;
  converter_.add_instruction(keys_create);

  // Create values list
  symbolt &values_list = converter_.create_tmp_symbol(
    element, dict_name + "_values", list_type, exprt());

  code_declt values_decl(symbol_expr(values_list));
  values_decl.location() = location;
  converter_.add_instruction(values_decl);

  code_function_callt values_create;
  values_create.function() = symbol_expr(*create_func);
  values_create.lhs() = symbol_expr(values_list);
  values_create.type() = list_type;
  values_create.location() = location;
  converter_.add_instruction(values_create);

  // Push initial key-value pairs
  const auto &keys = element["keys"];
  const auto &values = element["values"];

  python_list list_handler(converter_, element);

  for (size_t i = 0; i < keys.size(); ++i)
  {
    exprt key_expr = converter_.get_expr(keys[i]);
    exprt push_key =
      list_handler.build_push_list_call(keys_list, element, key_expr);
    converter_.add_instruction(push_key);

    exprt value_expr = converter_.get_expr(values[i]);

    // Special handling for nested dict values
    if (value_expr.type().is_struct() && is_dict_type(value_expr.type()))
    {
      // For nested dicts, store a pointer to the dict struct (8 bytes)
      // Get the list_push function
      const symbolt *push_func =
        symbol_table_.find_symbol("c:@F@__ESBMC_list_push");
      if (!push_func)
        throw std::runtime_error("__ESBMC_list_push not found");

      // Create a pointer variable to hold the address
      typet ptr_type = pointer_typet(value_expr.type());
      symbolt &ptr_var = converter_.create_tmp_symbol(
        element, "$nested_dict_ptr$", ptr_type, exprt());

      code_declt ptr_decl(symbol_expr(ptr_var));
      ptr_decl.location() = location;
      converter_.add_instruction(ptr_decl);

      // Assign the address
      code_assignt ptr_assign(
        symbol_expr(ptr_var), address_of_exprt(value_expr));
      ptr_assign.location() = location;
      converter_.add_instruction(ptr_assign);

      // Manually create type hash: use simple "dict_ptr" string
      const std::string ptr_type_name = "dict_ptr";
      constant_exprt type_hash(size_type());
      type_hash.set_value(integer2binary(
        std::hash<std::string>{}(ptr_type_name), config.ansi_c.address_width));

      // Use from_integer for pointer size
      exprt ptr_size = from_integer(8, size_type());

      // Call list_push directly
      code_function_callt push_call;
      push_call.function() = symbol_expr(*push_func);
      push_call.arguments().push_back(symbol_expr(values_list));
      push_call.arguments().push_back(address_of_exprt(symbol_expr(ptr_var)));
      push_call.arguments().push_back(
        type_hash);
      push_call.arguments().push_back(
        ptr_size);
      push_call.type() = bool_type();
      push_call.location() = location;
      converter_.add_instruction(push_call);
    }
    else
    {
      exprt push_value =
        list_handler.build_push_list_call(values_list, element, value_expr);
      converter_.add_instruction(push_value);
    }
  }

  // Assign keys and values to target dict struct members
  member_exprt keys_member(target_symbol, "keys", list_type);
  code_assignt keys_assign(keys_member, symbol_expr(keys_list));
  keys_assign.location() = location;
  converter_.add_instruction(keys_assign);

  member_exprt values_member(target_symbol, "values", list_type);
  code_assignt values_assign(values_member, symbol_expr(values_list));
  values_assign.location() = location;
  converter_.add_instruction(values_assign);

  return target_symbol;
}

exprt python_dict_handler::get_key_expr(const nlohmann::json &slice_node)
{
  return converter_.get_expr(slice_node);
}

exprt python_dict_handler::handle_dict_subscript(
  const exprt &dict_expr,
  const nlohmann::json &slice_node,
  const typet &expected_type)
{
  locationt location = converter_.get_location_from_decl(slice_node);
  typet list_type = type_handler_.get_list_type();

  exprt key_expr = get_key_expr(slice_node);

  // Get dict.keys and dict.values
  member_exprt keys_member(dict_expr, "keys", list_type);
  member_exprt values_member(dict_expr, "values", list_type);

  // Find __ESBMC_list_find_index function
  const symbolt *find_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_find_index");
  if (!find_func)
    throw std::runtime_error(
      "__ESBMC_list_find_index not found - add it to list.c model");

  // Create temp for index result
  symbolt &index_var = converter_.create_tmp_symbol(
    slice_node, "$dict_idx$", size_type(), gen_zero(size_type()));

  code_declt index_decl(symbol_expr(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  // Get element info for the key
  python_list list_handler(converter_, slice_node);
  list_elem_info key_info =
    list_handler.get_list_element_info(slice_node, key_expr);

  // Call find_index(keys, key, type_hash, size)
  code_function_callt find_call;
  find_call.function() = symbol_expr(*find_func);
  find_call.lhs() = symbol_expr(index_var);
  find_call.arguments().push_back(keys_member);

  exprt key_arg;
  if (
    key_info.elem_symbol->type.is_pointer() &&
    key_info.elem_symbol->type.subtype() == char_type())
    key_arg = symbol_expr(*key_info.elem_symbol);
  else
    key_arg = address_of_exprt(symbol_expr(*key_info.elem_symbol));

  find_call.arguments().push_back(key_arg);
  find_call.arguments().push_back(symbol_expr(*key_info.elem_type_sym));
  find_call.arguments().push_back(key_info.elem_size);
  find_call.type() = size_type();
  find_call.location() = location;
  converter_.add_instruction(find_call);

  // Get values[index] using list_at
  const symbolt *at_func = symbol_table_.find_symbol("c:@F@__ESBMC_list_at");
  if (!at_func)
    throw std::runtime_error("__ESBMC_list_at not found");

  // Create temp for the PyObject* result
  typet obj_ptr_type = pointer_typet(type_handler_.get_list_element_type());
  symbolt &obj_var = converter_.create_tmp_symbol(
    slice_node, "$dict_val_obj$", obj_ptr_type, exprt());

  code_declt obj_decl(symbol_expr(obj_var));
  obj_decl.location() = location;
  converter_.add_instruction(obj_decl);

  // Call list_at(values, index)
  code_function_callt at_call;
  at_call.function() = symbol_expr(*at_func);
  at_call.lhs() = symbol_expr(obj_var);
  at_call.arguments().push_back(values_member);
  at_call.arguments().push_back(symbol_expr(index_var));
  at_call.type() = obj_ptr_type;
  at_call.location() = location;
  converter_.add_instruction(at_call);

  // Extract obj->value (void* pointing to actual data)
  member_exprt obj_value(
    dereference_exprt(
      symbol_expr(obj_var), type_handler_.get_list_element_type()),
    "value",
    pointer_typet(empty_typet()));

  // Handle dict types 
  // Check expected_type first, then fall back to checking obj_value type
  if (!expected_type.is_nil() && is_dict_type(expected_type))
  {
    // obj_value is void* pointing to 8-byte storage that contains the dict pointer
    typet uint64_t = unsignedbv_typet(64);

    // Cast obj->value from void* to uint64_t* and dereference
    typecast_exprt as_uint64_ptr(obj_value, pointer_typet(uint64_t));
    dereference_exprt ptr_as_uint64(as_uint64_ptr, uint64_t);

    // Store the uint64 value in a temporary
    symbolt &uint64_var = converter_.create_tmp_symbol(
      slice_node, "$dict_ptr_uint64$", uint64_t, exprt());

    code_declt uint64_decl(symbol_expr(uint64_var));
    uint64_decl.location() = location;
    converter_.add_instruction(uint64_decl);

    code_assignt uint64_assign(symbol_expr(uint64_var), ptr_as_uint64);
    uint64_assign.location() = location;
    converter_.add_instruction(uint64_assign);

    // Cast the uint64 value to dict*
    typecast_exprt dict_ptr(
      symbol_expr(uint64_var), pointer_typet(expected_type));

    // Store dict pointer
    symbolt &dict_ptr_var = converter_.create_tmp_symbol(
      slice_node, "$dict_ptr$", pointer_typet(expected_type), exprt());

    code_declt ptr_decl(symbol_expr(dict_ptr_var));
    ptr_decl.location() = location;
    converter_.add_instruction(ptr_decl);

    code_assignt ptr_assign(symbol_expr(dict_ptr_var), dict_ptr);
    ptr_assign.location() = location;
    converter_.add_instruction(ptr_assign);

    // Dereference the dict pointer to get the dict struct
    dereference_exprt dict_struct(symbol_expr(dict_ptr_var), expected_type);
    dict_struct.type() = expected_type;

    // Store the dict struct in the final variable
    symbolt &temp_dict = converter_.create_tmp_symbol(
      slice_node, "$dict_retrieved$", expected_type, exprt());

    code_declt temp_decl(symbol_expr(temp_dict));
    temp_decl.location() = location;
    converter_.add_instruction(temp_decl);

    code_assignt temp_assign(symbol_expr(temp_dict), dict_struct);
    temp_assign.location() = location;
    converter_.add_instruction(temp_assign);

    return symbol_expr(temp_dict);
  }

  // Handle list types
  if (expected_type == list_type)
  {
    // obj_value is void* pointing to a (PyListObj*)
    // We need to:
    // 1. Cast void* to (PyListObj**)  - pointer to pointer to list
    // 2. Dereference to get PyListObj*
    typecast_exprt value_as_list_ptr_ptr(obj_value, pointer_typet(list_type));
    dereference_exprt list_ptr(value_as_list_ptr_ptr, list_type);
    list_ptr.type() = list_type;
    return list_ptr;
  }

  // Handle float types: cast void* to double*, then dereference
  if (expected_type.is_floatbv())
  {
    typecast_exprt value_as_float_ptr(obj_value, pointer_typet(expected_type));
    dereference_exprt result(value_as_float_ptr, expected_type);
    result.type() = expected_type;
    return result;
  }

  // Handle integer types: cast void* to int*, then dereference
  if (expected_type.is_signedbv() || expected_type.is_unsignedbv())
  {
    typecast_exprt value_as_int_ptr(obj_value, pointer_typet(expected_type));
    dereference_exprt result(value_as_int_ptr, expected_type);
    result.type() = expected_type;
    return result;
  }

  // Handle boolean types: cast void* to bool*, then dereference
  if (expected_type.is_bool())
  {
    typecast_exprt value_as_bool_ptr(obj_value, pointer_typet(bool_type()));
    dereference_exprt result(value_as_bool_ptr, bool_type());
    result.type() = bool_type();
    return result;
  }

  // Default: cast void* to char* for string values
  typecast_exprt value_as_string(obj_value, gen_pointer_type(char_type()));
  return value_as_string;
}

void python_dict_handler::handle_dict_subscript_assign(
  const exprt &dict_expr,
  const nlohmann::json &slice_node,
  const exprt &value,
  codet &target_block)
{
  locationt location = converter_.get_location_from_decl(slice_node);
  typet list_type = type_handler_.get_list_type();

  exprt key_expr = get_key_expr(slice_node);

  member_exprt keys_member(dict_expr, "keys", list_type);
  member_exprt values_member(dict_expr, "values", list_type);

  // Check if key exists using membership test
  nlohmann::json dummy_json;
  python_list list_handler_check(converter_, dummy_json);
  exprt key_exists = list_handler_check.contains(key_expr, keys_member);

  // Create the "key exists" branch: update existing value
  code_blockt update_block;

  // Find __ESBMC_list_find_index function
  const symbolt *find_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_find_index");
  if (!find_func)
    throw std::runtime_error(
      "__ESBMC_list_find_index not found - add it to list.c model");

  // Find __ESBMC_list_set_at function
  const symbolt *set_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_set_at");
  if (!set_func)
    throw std::runtime_error(
      "__ESBMC_list_set_at not found - add it to list.c model");

  python_list list_handler(converter_, slice_node);

  // Create temp for index result
  symbolt &index_var = converter_.create_tmp_symbol(
    slice_node, "$dict_update_idx$", size_type(), gen_zero(size_type()));

  code_declt index_decl(symbol_expr(index_var));
  index_decl.location() = location;
  update_block.copy_to_operands(index_decl);

  // Get element info for the key
  list_elem_info key_info =
    list_handler.get_list_element_info(slice_node, key_expr);

  // Call find_index(keys, key, type_hash, size) to get the index
  code_function_callt find_call;
  find_call.function() = symbol_expr(*find_func);
  find_call.lhs() = symbol_expr(index_var);
  find_call.arguments().push_back(keys_member);

  exprt key_arg;
  if (
    key_info.elem_symbol->type.is_pointer() &&
    key_info.elem_symbol->type.subtype() == char_type())
    key_arg = symbol_expr(*key_info.elem_symbol);
  else
    key_arg = address_of_exprt(symbol_expr(*key_info.elem_symbol));

  find_call.arguments().push_back(key_arg);
  find_call.arguments().push_back(symbol_expr(*key_info.elem_type_sym));
  find_call.arguments().push_back(key_info.elem_size);
  find_call.type() = size_type();
  find_call.location() = location;
  update_block.copy_to_operands(find_call);

  // Update value at index using list_set_at
  list_elem_info value_info =
    list_handler.get_list_element_info(slice_node, value);

  code_function_callt set_value_call;
  set_value_call.function() = symbol_expr(*set_func);
  set_value_call.arguments().push_back(values_member);
  set_value_call.arguments().push_back(symbol_expr(index_var));

  exprt value_arg;
  if (
    value_info.elem_symbol->type.is_pointer() &&
    value_info.elem_symbol->type.subtype() == char_type())
    value_arg = symbol_expr(*value_info.elem_symbol);
  else
    value_arg = address_of_exprt(symbol_expr(*value_info.elem_symbol));

  set_value_call.arguments().push_back(value_arg);
  set_value_call.arguments().push_back(symbol_expr(*value_info.elem_type_sym));
  set_value_call.arguments().push_back(value_info.elem_size);
  set_value_call.type() = bool_type();
  set_value_call.location() = location;
  update_block.copy_to_operands(set_value_call);

  // Create the "key doesn't exist" branch: push new key-value pair
  code_blockt insert_block;

  const symbolt *push_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_push");
  if (!push_func)
    throw std::runtime_error("__ESBMC_list_push not found");

  // Push key
  code_function_callt push_key_call;
  push_key_call.function() = symbol_expr(*push_func);
  push_key_call.arguments().push_back(keys_member);
  push_key_call.arguments().push_back(key_arg);
  push_key_call.arguments().push_back(symbol_expr(*key_info.elem_type_sym));
  push_key_call.arguments().push_back(key_info.elem_size);
  push_key_call.type() = bool_type();
  push_key_call.location() = location;
  insert_block.copy_to_operands(push_key_call);

  // Push value
  code_function_callt push_value_call;
  push_value_call.function() = symbol_expr(*push_func);
  push_value_call.arguments().push_back(values_member);
  push_value_call.arguments().push_back(value_arg);
  push_value_call.arguments().push_back(symbol_expr(*value_info.elem_type_sym));
  push_value_call.arguments().push_back(value_info.elem_size);
  push_value_call.type() = bool_type();
  push_value_call.location() = location;
  insert_block.copy_to_operands(push_value_call);

  // Create if-then-else: if (key_exists) { update } else { insert }
  code_ifthenelset if_stmt;
  if_stmt.cond() = key_exists;
  if_stmt.then_case() = update_block;
  if_stmt.else_case() = insert_block;
  if_stmt.location() = location;

  target_block.copy_to_operands(if_stmt);
}

exprt python_dict_handler::handle_dict_membership(
  const exprt &key_expr,
  const exprt &dict_expr,
  bool negated)
{
  typet list_type = type_handler_.get_list_type();
  member_exprt keys_member(dict_expr, "keys", list_type);

  nlohmann::json dummy_json;
  python_list list_handler(converter_, dummy_json);

  exprt contains_result = list_handler.contains(key_expr, keys_member);

  if (negated)
    return not_exprt(contains_result);

  return contains_result;
}

void python_dict_handler::handle_dict_delete(
  const exprt &dict_expr,
  const nlohmann::json &slice_node,
  codet &target_block)
{
  locationt location = converter_.get_location_from_decl(slice_node);
  typet list_type = type_handler_.get_list_type();

  exprt key_expr = get_key_expr(slice_node);

  // Get dict.keys and dict.values
  member_exprt keys_member(dict_expr, "keys", list_type);
  member_exprt values_member(dict_expr, "values", list_type);

  // First, check if key exists using membership test
  // This avoids calling __ESBMC_list_find_index on empty dict or missing key
  nlohmann::json dummy_json;
  python_list list_handler_check(converter_, dummy_json);
  exprt key_exists = list_handler_check.contains(key_expr, keys_member);

  // Create KeyError exception for when key doesn't exist
  std::string exc_type_str = "KeyError";
  typet keyerror_type = type_handler_.get_typet(exc_type_str);
  std::string error_msg = "KeyError: key not found in dictionary";

  // Build the error message as a string constant
  exprt msg_size = constant_exprt(
    integer2binary(error_msg.size(), bv_width(size_type())),
    integer2string(error_msg.size()),
    size_type());
  typet str_type = array_typet(char_type(), msg_size);

  // Create a temporary variable to hold the error message string
  symbolt &error_msg_var = converter_.create_tmp_symbol(
    slice_node, "$keyerror_msg$", str_type, exprt());

  code_declt error_msg_decl(symbol_expr(error_msg_var));
  error_msg_decl.location() = location;
  target_block.copy_to_operands(error_msg_decl);

  // Assign the string literal to the temp variable
  exprt error_string =
    converter_.get_string_builder().build_string_literal(error_msg);
  code_assignt error_msg_assign(symbol_expr(error_msg_var), error_string);
  error_msg_assign.location() = location;
  target_block.copy_to_operands(error_msg_assign);

  // Construct exception struct with address of the temp variable
  exprt exception_struct("struct", keyerror_type);
  exception_struct.copy_to_operands(
    address_of_exprt(symbol_expr(error_msg_var)));

  // Create the throw expression
  exprt raise_keyerror = side_effect_exprt("cpp-throw", keyerror_type);
  raise_keyerror.move_to_operands(exception_struct);
  raise_keyerror.location() = location;

  // Create the "then" branch: perform the actual deletion
  code_blockt delete_block;

  // Find __ESBMC_list_find_index function
  const symbolt *find_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_find_index");
  if (!find_func)
    throw std::runtime_error(
      "__ESBMC_list_find_index not found - add it to list.c model");

  // Find __ESBMC_list_remove_at function
  const symbolt *remove_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_remove_at");
  if (!remove_func)
    throw std::runtime_error(
      "__ESBMC_list_remove_at not found - add it to list.c model");

  // Create temp for index result
  symbolt &index_var = converter_.create_tmp_symbol(
    slice_node, "$dict_del_idx$", size_type(), gen_zero(size_type()));

  code_declt index_decl(symbol_expr(index_var));
  index_decl.location() = location;
  delete_block.copy_to_operands(index_decl);

  // Get element info for the key
  python_list list_handler(converter_, slice_node);
  list_elem_info key_info =
    list_handler.get_list_element_info(slice_node, key_expr);

  // Call find_index(keys, key, type_hash, size) to get the index of the key
  code_function_callt find_call;
  find_call.function() = symbol_expr(*find_func);
  find_call.lhs() = symbol_expr(index_var);
  find_call.arguments().push_back(keys_member);

  exprt key_arg;
  if (
    key_info.elem_symbol->type.is_pointer() &&
    key_info.elem_symbol->type.subtype() == char_type())
    key_arg = symbol_expr(*key_info.elem_symbol);
  else
    key_arg = address_of_exprt(symbol_expr(*key_info.elem_symbol));

  find_call.arguments().push_back(key_arg);
  find_call.arguments().push_back(symbol_expr(*key_info.elem_type_sym));
  find_call.arguments().push_back(key_info.elem_size);
  find_call.type() = size_type();
  find_call.location() = location;
  delete_block.copy_to_operands(find_call);

  // Call list_remove_at(keys, index) to remove the key
  code_function_callt remove_key_call;
  remove_key_call.function() = symbol_expr(*remove_func);
  remove_key_call.arguments().push_back(keys_member);
  remove_key_call.arguments().push_back(symbol_expr(index_var));
  remove_key_call.type() = bool_type();
  remove_key_call.location() = location;
  delete_block.copy_to_operands(remove_key_call);

  // Call list_remove_at(values, index) to remove the corresponding value
  code_function_callt remove_value_call;
  remove_value_call.function() = symbol_expr(*remove_func);
  remove_value_call.arguments().push_back(values_member);
  remove_value_call.arguments().push_back(symbol_expr(index_var));
  remove_value_call.type() = bool_type();
  remove_value_call.location() = location;
  delete_block.copy_to_operands(remove_value_call);

  // Create the "else" branch: raise KeyError
  code_expressiont raise_code(raise_keyerror);
  raise_code.location() = location;

  // Create if-then-else: if (key_exists) { delete } else { raise KeyError }
  code_ifthenelset if_stmt;
  if_stmt.cond() = key_exists;
  if_stmt.then_case() = delete_block;
  if_stmt.else_case() = raise_code;
  if_stmt.location() = location;

  target_block.copy_to_operands(if_stmt);
}