#include <python-frontend/json_utils.h>
#include <python-frontend/python_dict_handler.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_handler.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/context.h>
#include <util/python_types.h>
#include <util/std_code.h>

#include <algorithm>
#include <functional>
#include <sstream>

python_dict_handler::python_dict_handler(
  python_converter &converter,
  contextt &symbol_table,
  type_handler &type_handler)
  : converter_(converter),
    symbol_table_(symbol_table),
    type_handler_(type_handler)
{
}

std::string python_dict_handler::generate_unique_dict_name(
  const nlohmann::json &element,
  const locationt &location) const
{
  std::ostringstream name;
  name << "$py_dict$";

  // Try to use location information for deterministic naming
  if (!location.get_file().empty() && !location.get_line().empty())
  {
    // Use file name (without path) + line + column
    std::string file = location.get_file().as_string();

    // Extract just the filename without path
    size_t last_slash = file.find_last_of("/\\");
    if (last_slash != std::string::npos)
      file = file.substr(last_slash + 1);

    // Replace dots and special chars with underscores for valid identifiers
    std::replace(file.begin(), file.end(), '.', '_');
    std::replace(file.begin(), file.end(), '-', '_');

    name << file << "$" << location.get_line().as_string() << "$"
         << location.get_column().as_string();
  }
  else
  {
    // Fallback: use hash of the JSON element for uniqueness
    // This handles cases where location info is missing
    std::hash<std::string> hasher;
    size_t hash = hasher(element.dump());
    name << "noloc$" << std::hex << hash;
  }

  // Add a disambiguator based on element content hash to handle
  // multiple dicts at the same location (e.g., in list comprehensions)
  std::hash<std::string> hasher;
  size_t content_hash = hasher(element.dump());
  name << "$" << std::hex << (content_hash & 0xFFFF); // Use last 4 hex digits

  return name.str();
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
  typecast_exprt as_ptr_type_ptr(obj_value, pointer_typet(pointer_type()));
  dereference_exprt ptr_as_ptr_type(as_ptr_type_ptr, pointer_type());

  // Step 2: Store pointer_type value in temporary to ensure proper evaluation order
  symbolt &ptr_type_var = converter_.create_tmp_symbol(
    node, "$dict_ptr_as_int$", pointer_type(), exprt());
  code_declt ptr_type_decl(symbol_expr(ptr_type_var));
  ptr_type_decl.location() = location;
  converter_.add_instruction(ptr_type_decl);

  code_assignt ptr_type_assign(symbol_expr(ptr_type_var), ptr_as_ptr_type);
  ptr_type_assign.location() = location;
  converter_.add_instruction(ptr_type_assign);

  // Step 3: Cast pointer_type value to target pointer type
  typecast_exprt dict_ptr(symbol_expr(ptr_type_var), target_ptr_type);

  // Step 4: Store the typed pointer
  symbolt &dict_ptr_var = converter_.create_tmp_symbol(
    node, "$dict_ptr_typed$", target_ptr_type, exprt());
  code_declt ptr_decl(symbol_expr(dict_ptr_var));
  ptr_decl.location() = location;
  converter_.add_instruction(ptr_decl);

  code_assignt ptr_assign(symbol_expr(dict_ptr_var), dict_ptr);
  ptr_assign.location() = location;
  converter_.add_instruction(ptr_assign);

  return symbol_expr(dict_ptr_var);
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
  code_declt ptr_decl(symbol_expr(ptr_var));
  ptr_decl.location() = location;
  converter_.add_instruction(ptr_decl);

  // Store the address of the dict
  code_assignt ptr_assign(symbol_expr(ptr_var), address_of_exprt(value_expr));
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
  push_call.function() = symbol_expr(*push_func);
  push_call.arguments().push_back(symbol_expr(values_list));
  push_call.arguments().push_back(symbol_expr(ptr_var));
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
  typecast_exprt dict_ptr(obj_value, dict_ptr_type);

  // Dereference to get the actual dict struct
  dereference_exprt dict_struct(dict_ptr, expected_type);

  // Store in final temporary for return
  symbolt &result_dict = converter_.create_tmp_symbol(
    slice_node, "$dict_retrieved$", expected_type, exprt());
  code_declt temp_decl(symbol_expr(result_dict));
  temp_decl.location() = location;
  converter_.add_instruction(temp_decl);

  code_assignt result_assign(symbol_expr(result_dict), dict_struct);
  result_assign.location() = location;
  converter_.add_instruction(result_assign);

  return symbol_expr(result_dict);
}

exprt python_dict_handler::get_dict_literal(const nlohmann::json &element)
{
  if (!is_dict_literal(element))
    throw std::runtime_error("Expected Dict literal");

  // For nested dictionaries, we need to create a temporary variable
  // because the dict needs to exist as a concrete symbol to be used as a value
  locationt location = converter_.get_location_from_decl(element);

  // Generate unique name based on location
  std::string dict_name = generate_unique_dict_name(element, location);

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

exprt python_dict_handler::get_dict_comprehension(const nlohmann::json &element)
{
  if (!element.contains("generators") || element["generators"].empty())
    throw std::runtime_error("DictComp missing generators clause");

  if (element["generators"].size() != 1)
    throw std::runtime_error("Only single-generator DictComp is supported");

  const auto &generator = element["generators"][0];
  const auto &target = generator["target"];
  const auto &iter = generator["iter"];

  locationt location = converter_.get_location_from_decl(element);
  std::string dict_name = generate_unique_dict_name(element, location);
  struct_typet dict_type = get_dict_struct_type();
  typet list_type = type_handler_.get_list_type();

  symbolt &dict_sym =
    converter_.create_tmp_symbol(element, dict_name, dict_type, exprt());
  code_declt dict_decl(symbol_expr(dict_sym));
  dict_decl.location() = location;
  converter_.add_instruction(dict_decl);

  // Start with an empty dict and fill it inside the comprehension loop
  nlohmann::json empty_dict = element;
  empty_dict["_type"] = "Dict";
  empty_dict["keys"] = nlohmann::json::array();
  empty_dict["values"] = nlohmann::json::array();
  create_dict_from_literal(empty_dict, symbol_expr(dict_sym));

  exprt iterable_expr = converter_.get_expr(iter);
  // If the iterable comes from a call such as range(...), store it first so
  // the loop can reuse the same list on each iteration
  if (
    iterable_expr.is_code() &&
    iterable_expr.get("statement") == "function_call" &&
    to_code_function_call(to_code(iterable_expr)).type() == list_type)
  {
    const code_function_callt &call =
      to_code_function_call(to_code(iterable_expr));
    symbolt &tmp_var_symbol = converter_.create_tmp_symbol(
      element, "$dictcomp_iter$", list_type, gen_zero(list_type));

    code_declt tmp_var_decl(symbol_expr(tmp_var_symbol));
    tmp_var_decl.location() = location;
    converter_.add_instruction(tmp_var_decl);

    code_function_callt new_call;
    new_call.function() = call.function();
    new_call.arguments() = call.arguments();
    new_call.lhs() = symbol_expr(tmp_var_symbol);
    new_call.type() = list_type;
    new_call.location() = location;
    converter_.add_instruction(new_call);

    iterable_expr = symbol_expr(tmp_var_symbol);
  }

  if (target["_type"] != "Name")
    throw std::runtime_error("Only simple targets are supported in DictComp");

  std::string loop_var_name = target["id"].get<std::string>();
  symbol_id loop_var_sid = converter_.create_symbol_id();
  loop_var_sid.set_object(loop_var_name);

  typet loop_var_type = any_type();
  // range(...) produces integers, so keep the loop variable in the
  // same type used later by dict lookups
  if (
    iter.contains("_type") && iter["_type"] == "Call" &&
    iter.contains("func") && iter["func"].contains("_type") &&
    iter["func"]["_type"] == "Name" && iter["func"].contains("id") &&
    iter["func"]["id"] == "range")
  {
    loop_var_type = long_long_int_type();
  }

  symbolt loop_var_symbol = converter_.create_symbol(
    location.get_file().as_string(),
    loop_var_name,
    loop_var_sid.to_string(),
    location,
    loop_var_type);
  loop_var_symbol.lvalue = true;
  loop_var_symbol.file_local = true;
  loop_var_symbol.is_extern = false;
  symbolt *loop_var =
    converter_.symbol_table().move_symbol_to_context(loop_var_symbol);

  symbolt &index_var = converter_.create_tmp_symbol(
    element, "$dictcomp_i$", size_type(), gen_zero(size_type()));
  code_declt index_decl(symbol_expr(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  code_assignt index_init(symbol_expr(index_var), gen_zero(size_type()));
  index_init.location() = location;
  converter_.add_instruction(index_init);

  exprt length_expr;
  if (iterable_expr.type() == list_type)
  {
    const symbolt *size_func =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
    if (!size_func)
      throw std::runtime_error("__ESBMC_list_size not found in symbol table");

    symbolt &length_var = converter_.create_tmp_symbol(
      element, "$dictcomp_len$", size_type(), gen_zero(size_type()));
    code_declt length_decl(symbol_expr(length_var));
    length_decl.location() = location;
    converter_.add_instruction(length_decl);

    code_function_callt size_call;
    size_call.function() = symbol_expr(*size_func);
    size_call.arguments().push_back(
      iterable_expr.type().is_pointer() ? iterable_expr
                                        : address_of_exprt(iterable_expr));
    size_call.lhs() = symbol_expr(length_var);
    size_call.type() = size_type();
    size_call.location() = location;
    converter_.add_instruction(size_call);

    length_expr = symbol_expr(length_var);
  }
  else
  {
    throw std::runtime_error(
      "Unsupported iterable type in DictComp: " +
      iterable_expr.type().id_string());
  }

  python_list list_handler(converter_, element);
  code_blockt loop_body;

  // Read the current element from the list and assign it to the
  // comprehension target before evaluating key and value
  exprt current_element = list_handler.build_list_at_call(
    iterable_expr, symbol_expr(index_var), element);
  current_element =
    list_handler.extract_pyobject_value(current_element, loop_var_type);

  code_assignt loop_var_assign(symbol_expr(*loop_var), current_element);
  loop_var_assign.location() = location;
  loop_body.copy_to_operands(loop_var_assign);

  // Keep current_block redirected to loop_body across the whole pair build:
  // get_expr, contains(), get_list_element_info() and other helpers used by
  // handle_dict_subscript_assign emit DECL/ASSIGN side effects via
  // add_instruction(), which targets current_block. Restoring too early would
  // hoist the contains check and the key/value temporaries out of the loop,
  // freezing them at their pre-loop values and leaving the dict empty.
  code_blockt *saved_block = converter_.current_block;
  converter_.current_block = &loop_body;
  exprt key_expr = converter_.get_expr(element["key"]);
  exprt value_expr = converter_.get_expr(element["value"]);

  // Reuse the normal dict assignment path for each generated pair
  code_blockt pair_block;
  handle_dict_subscript_assign(
    symbol_expr(dict_sym), key_expr, value_expr, location, element, pair_block);
  converter_.current_block = saved_block;

  if (generator.contains("ifs") && !generator["ifs"].empty())
  {
    // Dict comprehensions may include filters such as
    // {k: v for x in xs if cond1 if cond2}
    exprt combined_condition = gen_boolean(true);
    for (const auto &if_clause : generator["ifs"])
    {
      exprt if_expr = converter_.get_expr(if_clause);
      if (combined_condition.is_true())
        combined_condition = if_expr;
      else
      {
        exprt and_expr("and", bool_type());
        and_expr.copy_to_operands(combined_condition, if_expr);
        combined_condition = and_expr;
      }
    }

    codet if_stmt;
    if_stmt.set_statement("ifthenelse");
    if_stmt.copy_to_operands(combined_condition, pair_block);
    if_stmt.location() = location;
    loop_body.copy_to_operands(if_stmt);
  }
  else
  {
    loop_body.copy_to_operands(pair_block);
  }

  exprt increment("+", size_type());
  increment.copy_to_operands(symbol_expr(index_var), gen_one(size_type()));
  code_assignt index_increment(symbol_expr(index_var), increment);
  index_increment.location() = location;
  loop_body.copy_to_operands(index_increment);

  exprt loop_condition("<", bool_type());
  loop_condition.copy_to_operands(symbol_expr(index_var), length_expr);

  codet while_stmt;
  while_stmt.set_statement("while");
  while_stmt.copy_to_operands(loop_condition, loop_body);
  while_stmt.location() = location;
  converter_.add_instruction(while_stmt);

  return symbol_expr(dict_sym);
}

exprt python_dict_handler::create_dict_from_literal(
  const nlohmann::json &element,
  const exprt &target_symbol)
{
  locationt location = converter_.get_location_from_decl(element);

  // Generate unique name based on location
  std::string dict_name = generate_unique_dict_name(element, location);

  struct_typet dict_type = get_dict_struct_type();
  typet list_type = type_handler_.get_list_type();

  // Freeze the target as dict-typed lvalue so previously emitted member
  // accesses remain valid even if the frontend later mutates the symbol type.
  exprt dict_target = typecast_exprt(target_symbol, dict_type);

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

    // Convert lambda/function symbol to function pointer for dict storage
    if (value_expr.type().is_code() && value_expr.is_symbol())
      value_expr = address_of_exprt(value_expr);

    // Check if this is a nested dict that needs special pointer storage
    if (value_expr.type().is_struct() && is_dict_type(value_expr.type()))
    {
      // Nested dict: store pointer to dict (reference semantics)
      store_nested_dict_value(element, values_list, value_expr, location);
    }
    else
    {
      // Regular value: store value directly (value semantics).
      // Disable float path so dict comparisons via *(void**)item->value use
      // the integer bit-pattern copy instead of the float_buf pointer.
      exprt push_value = list_handler.build_push_list_call(
        values_list, element, value_expr, false);
      converter_.add_instruction(push_value);
    }
  }

  // Assign keys and values to target dict struct members
  member_exprt keys_member(dict_target, "keys", list_type);
  code_assignt keys_assign(keys_member, symbol_expr(keys_list));
  keys_assign.location() = location;
  converter_.add_instruction(keys_assign);

  member_exprt values_member(dict_target, "values", list_type);
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

  // If expected_type is not provided, try to infer it from the dict's annotation
  typet resolved_type = expected_type;
  if (resolved_type.is_nil() || resolved_type.is_empty())
    resolved_type = resolve_expected_type_for_dict_subscript(dict_expr);

  exprt key_expr = get_key_expr(slice_node);

  // Get dict.keys and dict.values
  member_exprt keys_member(dict_expr, "keys", list_type);
  member_exprt values_member(dict_expr, "values", list_type);

  // Use try_find_index so a missing key returns SIZE_MAX instead of asserting.
  // We then emit a cpp-throw KeyError for the not-found case, which allows
  // try/except KeyError blocks to catch it (Python semantics).
  const symbolt *find_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_try_find_index");
  if (!find_func)
    throw std::runtime_error(
      "__ESBMC_list_try_find_index not found - add it to list.c model");

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

  // Call try_find_index(keys, key, type_hash, size) — returns SIZE_MAX if absent
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

  // If index == SIZE_MAX the key was not found: throw KeyError so that
  // try/except KeyError handlers can catch it (instead of failing the property).
  {
    const BigInt size_max_val = power(2, bv_width(size_type())) - 1;
    constant_exprt size_max(size_max_val, size_type());
    exprt key_not_found = equality_exprt(symbol_expr(index_var), size_max);

    std::string keyerror_msg = "KeyError: key not found in dictionary";
    std::string keyerror_type_str = "KeyError";
    typet keyerror_type = type_handler_.get_typet(keyerror_type_str);

    exprt msg_size_expr = constant_exprt(
      integer2binary(keyerror_msg.size() + 1, bv_width(size_type())),
      integer2string(keyerror_msg.size() + 1),
      size_type());
    typet str_arr_type = array_typet(char_type(), msg_size_expr);

    symbolt &err_msg_var = converter_.create_tmp_symbol(
      slice_node, "$keyerror_msg$", str_arr_type, exprt());
    code_declt err_msg_decl(symbol_expr(err_msg_var));
    err_msg_decl.location() = location;

    exprt err_str =
      converter_.get_string_builder().build_string_literal(keyerror_msg);
    code_assignt err_msg_assign(symbol_expr(err_msg_var), err_str);
    err_msg_assign.location() = location;

    exprt exc_struct("struct", keyerror_type);
    exc_struct.copy_to_operands(address_of_exprt(symbol_expr(err_msg_var)));

    exprt raise_keyerror = side_effect_exprt("cpp-throw", keyerror_type);
    raise_keyerror.move_to_operands(exc_struct);
    raise_keyerror.location() = location;

    code_blockt throw_block;
    throw_block.copy_to_operands(err_msg_decl);
    throw_block.copy_to_operands(err_msg_assign);
    code_expressiont raise_code(raise_keyerror);
    raise_code.location() = location;
    throw_block.copy_to_operands(raise_code);

    code_ifthenelset key_check;
    key_check.cond() = key_not_found;
    key_check.then_case() = throw_block;
    key_check.location() = location;
    converter_.add_instruction(key_check);
  }

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
  // Resolve symbol type to actual struct type before dereferencing
  typet element_type = type_handler_.get_list_element_type();
  if (element_type.is_symbol())
  {
    const symbol_typet &sym_type = to_symbol_type(element_type);
    const symbolt *elem_sym =
      symbol_table_.find_symbol(sym_type.get_identifier());
    if (elem_sym)
      element_type = elem_sym->type;
  }

  // Create dereference and explicitly set its type
  dereference_exprt deref_obj(symbol_expr(obj_var), element_type);
  deref_obj.type() = element_type;
  member_exprt obj_value(deref_obj, "value", pointer_typet(empty_typet()));

  // Handle dict types
  if (!resolved_type.is_nil() && is_dict_type(resolved_type))
  {
    return retrieve_nested_dict_value(
      slice_node, obj_value, resolved_type, location);
  }

  // Handle list types
  if (resolved_type == list_type)
  {
    typecast_exprt value_as_list_ptr_ptr(obj_value, pointer_typet(list_type));
    dereference_exprt list_ptr(value_as_list_ptr_ptr, list_type);
    list_ptr.type() = list_type;

    // Create a temporary symbol for this list to store in the type map
    symbolt &list_result = converter_.create_tmp_symbol(
      slice_node, "$dict_list_result$", list_type, exprt());

    code_declt list_decl(symbol_expr(list_result));
    list_decl.location() = location;
    converter_.add_instruction(list_decl);

    code_assignt list_assign(symbol_expr(list_result), list_ptr);
    list_assign.location() = location;
    converter_.add_instruction(list_assign);

    // Extract element type and populate list_type_map for correct iteration
    if (dict_expr.is_symbol())
    {
      const symbolt *sym = symbol_table_.find_symbol(dict_expr.identifier());
      if (!sym)
      {
        log_warning(
          "Could not find symbol '{}' in symbol table for dict subscript type "
          "resolution",
          dict_expr.identifier());
      }
      else
      {
        std::string var_name = sym->name.as_string();
        nlohmann::json var_decl = json_utils::find_var_decl(
          var_name,
          converter_.get_current_func_name(),
          converter_.get_ast_json());

        if (
          !var_decl.empty() && var_decl.contains("value") &&
          var_decl["value"]["_type"] == "Call" &&
          var_decl["value"]["func"]["_type"] == "Name")
        {
          std::string func_name =
            var_decl["value"]["func"]["id"].get<std::string>();
          nlohmann::json func_def = json_utils::find_function(
            converter_.get_ast_json()["body"], func_name);

          if (
            !func_def.empty() && func_def.contains("returns") &&
            !func_def["returns"].is_null())
          {
            const auto &returns = func_def["returns"];
            if (
              returns.contains("slice") &&
              returns["slice"]["_type"] == "Tuple" &&
              returns["slice"]["elts"].size() >= 2)
            {
              const auto &value_type = returns["slice"]["elts"][1];
              if (
                value_type["_type"] == "Subscript" &&
                value_type.contains("slice") &&
                value_type["slice"].contains("id"))
              {
                std::string elem_type_str =
                  value_type["slice"]["id"].get<std::string>();
                typet elem_type = type_handler_.get_typet(elem_type_str);

                const std::string &list_id = list_result.id.as_string();
                python_list::list_type_map[list_id].push_back(
                  std::make_pair("", elem_type));
              }
            }
          }
        }
      }
    }

    return symbol_expr(list_result);
  }

  // Handle float types
  if (resolved_type.is_floatbv())
  {
    typecast_exprt value_as_float_ptr(obj_value, pointer_typet(resolved_type));
    dereference_exprt result(value_as_float_ptr, resolved_type);
    result.type() = resolved_type;
    return result;
  }

  // Handle integer types
  if (resolved_type.is_signedbv() || resolved_type.is_unsignedbv())
  {
    typecast_exprt value_as_int_ptr(obj_value, pointer_typet(resolved_type));
    dereference_exprt result(value_as_int_ptr, resolved_type);
    result.type() = resolved_type;
    return result;
  }

  // Handle boolean types
  if (resolved_type.is_bool())
  {
    typecast_exprt value_as_bool_ptr(obj_value, pointer_typet(bool_type()));
    dereference_exprt result(value_as_bool_ptr, bool_type());
    result.type() = bool_type();
    return result;
  }

  // Handle non-char pointer types (e.g., Optional[T] stored as T*).
  // The value was stored by-reference (value_arg = address_of(T* var)), so
  // item->value points to a buffer holding the T* bytes.
  // Dereference as T** to recover the stored T* value.
  if (
    resolved_type.is_pointer() && resolved_type.subtype() != char_type() &&
    !resolved_type.is_nil())
  {
    typecast_exprt value_as_ptr_ptr(obj_value, pointer_typet(resolved_type));
    dereference_exprt result(value_as_ptr_ptr, resolved_type);
    result.type() = resolved_type;
    return result;
  }

  // Class-struct value: cast stored void* to the struct type and dereference.
  {
    namespacet ns(symbol_table_);
    typet underlying = resolved_type;
    if (underlying.id() == "symbol")
      underlying = ns.follow(underlying);
    if (underlying.is_struct() && !is_dict_type(underlying))
    {
      typecast_exprt value_as_struct_ptr(obj_value, pointer_typet(underlying));
      dereference_exprt result(value_as_struct_ptr, underlying);
      result.type() = underlying;
      return result;
    }
  }

  // Default: cast void* to char* for string values
  typecast_exprt value_as_string(obj_value, gen_pointer_type(char_type()));
  return value_as_string;
}

void python_dict_handler::handle_dict_subscript_assign(
  const exprt &dict_expr,
  const exprt &key_expr,
  const exprt &value,
  const locationt &location,
  const nlohmann::json &node,
  codet &target_block)
{
  typet list_type = type_handler_.get_list_type();

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

  python_list list_handler(converter_, node);

  symbolt &index_var = converter_.create_tmp_symbol(
    node, "$dict_update_idx$", size_type(), gen_zero(size_type()));

  code_declt index_decl(symbol_expr(index_var));
  index_decl.location() = location;
  update_block.copy_to_operands(index_decl);

  // Get element info for the key
  list_elem_info key_info = list_handler.get_list_element_info(node, key_expr);

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
  list_elem_info value_info = list_handler.get_list_element_info(node, value);

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
  set_value_call.arguments().push_back(from_integer(BigInt(0), size_type()));
  set_value_call.arguments().push_back(from_integer(
    BigInt(converter_.get_type_handler().is_pointer_free(
      value_info.elem_symbol->type)),
    int_type()));
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
  push_key_call.arguments().push_back(from_integer(BigInt(0), size_type()));
  push_key_call.arguments().push_back(from_integer(
    BigInt(converter_.get_type_handler().is_pointer_free(
      key_info.elem_symbol->type)),
    int_type()));
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
  push_value_call.arguments().push_back(from_integer(BigInt(0), size_type()));
  push_value_call.arguments().push_back(from_integer(
    BigInt(converter_.get_type_handler().is_pointer_free(
      value_info.elem_symbol->type)),
    int_type()));
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

  // Build the error message as a string constant (+1 for null terminator,
  // matching the char[N+1] type produced by build_string_literal).
  exprt msg_size = constant_exprt(
    integer2binary(error_msg.size() + 1, bv_width(size_type())),
    integer2string(error_msg.size() + 1),
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
        lhs = dereference_exprt(lhs, lhs.type().subtype());
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
        rhs = dereference_exprt(rhs, rhs.type().subtype());
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
        lhs = dereference_exprt(lhs, lhs.type().subtype());
    }

    exprt rhs_dict = converter_.get_expr(right["value"]);
    if (rhs_dict.type().is_struct() && is_dict_type(rhs_dict.type()))
    {
      rhs = handle_dict_subscript(rhs_dict, right["slice"], default_type);
      // Dereference the pointer to get the actual value
      if (rhs.type().is_pointer())
      {
        rhs = dereference_exprt(rhs, rhs.type().subtype());
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

bool python_dict_handler::handle_subscript_assignment_check(
  python_converter &converter,
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  codet &target_block)
{
  if (target["_type"] != "Subscript")
    return false;

  exprt container_expr = converter.get_expr(target["value"]);
  typet container_type = container_expr.type();

  if (container_expr.is_symbol())
  {
    const symbolt *sym = symbol_table_.find_symbol(container_expr.identifier());
    if (sym)
      container_type = sym->type;
  }

  // Use the namespace from converter, not a method that doesn't exist
  namespacet ns(symbol_table_);
  if (container_type.id() == "symbol")
    container_type = ns.follow(container_type);

  if (!is_dict_type(container_type))
    return false;

  // Handle dict[key] = value assignment
  converter.set_converting_rhs(true);
  exprt rhs = converter.get_expr(ast_node["value"]);
  converter.set_converting_rhs(false);

  handle_dict_subscript_assign(
    container_expr,
    get_key_expr(target["slice"]),
    rhs,
    converter.get_location_from_decl(target["slice"]),
    target["slice"],
    target_block);
  return true;
}

bool python_dict_handler::handle_literal_assignment_check(
  python_converter &converter,
  const nlohmann::json &ast_node,
  const exprt &lhs)
{
  if (!ast_node.contains("value") || ast_node["value"].is_null())
    return false;

  if (!is_dict_literal(ast_node["value"]))
    return false;

  create_dict_from_literal(ast_node["value"], lhs);
  converter.set_current_lhs(nullptr);
  return true;
}

bool python_dict_handler::handle_unannotated_literal_check(
  python_converter &converter,
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  const symbol_id &sid)
{
  if (!ast_node.contains("value") || !ast_node["value"].contains("_type"))
    return false;

  if (!is_dict_literal(ast_node["value"]))
    return false;

  locationt location = converter.get_location_from_decl(target);
  std::string module_name = location.get_file().as_string();
  std::string name;

  if (target["_type"] == "Name")
    name = target["id"].get<std::string>();
  else if (target["_type"] == "Attribute")
    name = target["attr"].get<std::string>();

  symbolt symbol = converter.create_symbol(
    module_name, name, sid.to_string(), location, get_dict_struct_type());
  symbol.lvalue = true;
  symbol.file_local = true;
  symbol.is_extern = false;
  symbolt *lhs_symbol = converter.add_symbol_and_get_ptr(symbol);

  exprt lhs = converter.create_lhs_expression(target, lhs_symbol, location);
  create_dict_from_literal(ast_node["value"], lhs);
  converter.set_current_lhs(nullptr);
  return true;
}

exprt python_dict_handler::handle_dict_get(
  const exprt &dict_expr,
  const nlohmann::json &call_node)
{
  locationt location = converter_.get_location_from_decl(call_node);
  typet list_type = type_handler_.get_list_type();

  const auto &args = call_node["args"];

  if (args.empty())
    throw std::runtime_error("get() missing required argument: 'key'");

  exprt key_expr = converter_.get_expr(args[0]);

  // Determine the default value
  exprt default_value;
  if (args.size() >= 2)
  {
    default_value = converter_.get_expr(args[1]);
    // Unwrap Optional if the default comes from a dict.get() without explicit
    // default (which now returns Optional[T]). We want the raw value type so
    // that result_var and default_value have compatible types.
    default_value = converter_.unwrap_optional_if_needed(default_value);
  }
  else
    default_value = gen_zero(none_type());

  // Infer result type
  typet result_type = resolve_expected_type_for_dict_subscript(dict_expr);

  // If we can't infer from dict annotation, use sensible defaults
  if (result_type.is_nil() || result_type.is_empty())
  {
    if (default_value.type() != none_type())
      result_type = default_value.type(); // Use explicit default's type
    else
      result_type =
        long_int_type(); // Default to int when returning None or unknown
  }

  // When no explicit default is given, dict.get() returns Optional[T]:
  // either the value (key found) or None (key not found). Use an Optional
  // struct so that `result is None` correctly checks the is_none field.
  // Exception: for string result types (char array or char*), use a null char*
  // to represent None instead. Storing a char* inside an Optional struct field
  // causes the ESBMC SMT encoder's dereference layer to fail (is_struct_type
  // assertion in dereference.cpp). A null char* is correctly handled by the
  // isnone evaluator's pointer path: `result is None` => `result == NULL`.
  // Also normalize the result type to char* (not char[0]) for consistency with
  // handle_dict_subscript which always returns char* for string values.
  const bool no_explicit_default = (args.size() < 2);
  const bool is_string_result =
    (result_type.is_array() && result_type.subtype() == char_type()) ||
    (result_type.is_pointer() && result_type.subtype() == char_type());
  const bool use_optional = no_explicit_default && !is_string_result;
  // Normalize string types to char* so the result variable holds a proper
  // pointer, not a zero-length char array.
  typet normalized_result_type =
    is_string_result ? gen_pointer_type(char_type()) : result_type;
  typet effective_result_type =
    use_optional ? type_handler_.build_optional_type(normalized_result_type)
                 : normalized_result_type;

  // Get dict members
  member_exprt keys_member(dict_expr, "keys", list_type);
  member_exprt values_member(dict_expr, "values", list_type);

  const symbolt *try_find_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_try_find_index");
  if (!try_find_func)
    throw std::runtime_error("__ESBMC_list_try_find_index not found");

  // Create temp for index result
  symbolt &index_var = converter_.create_tmp_symbol(
    call_node, "$dict_get_idx$", size_type(), exprt());
  code_declt index_decl(symbol_expr(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  // Get element info for the key
  python_list list_handler(converter_, call_node);
  list_elem_info key_info =
    list_handler.get_list_element_info(call_node, key_expr);

  // Call try_find_index (returns SIZE_MAX if not found)
  code_function_callt try_find_call;
  try_find_call.function() = symbol_expr(*try_find_func);
  try_find_call.lhs() = symbol_expr(index_var);
  try_find_call.arguments().push_back(keys_member);

  exprt key_arg;
  if (
    key_info.elem_symbol->type.is_pointer() &&
    key_info.elem_symbol->type.subtype() == char_type())
    key_arg = symbol_expr(*key_info.elem_symbol);
  else
    key_arg = address_of_exprt(symbol_expr(*key_info.elem_symbol));

  try_find_call.arguments().push_back(key_arg);
  try_find_call.arguments().push_back(symbol_expr(*key_info.elem_type_sym));
  try_find_call.arguments().push_back(key_info.elem_size);
  try_find_call.type() = size_type();
  try_find_call.location() = location;
  converter_.add_instruction(try_find_call);

  // Check if key was found (index != SIZE_MAX)
  const BigInt size_max_val = power(2, bv_width(size_type())) - 1;
  constant_exprt size_max(size_max_val, size_type());
  exprt key_found = not_exprt(equality_exprt(symbol_expr(index_var), size_max));

  // Create result variable
  symbolt &result_var = converter_.create_tmp_symbol(
    call_node, "$dict_get_result$", effective_result_type, exprt());
  code_declt result_decl(symbol_expr(result_var));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  // Then branch: key found, retrieve value
  code_blockt then_block;

  const symbolt *at_func = symbol_table_.find_symbol("c:@F@__ESBMC_list_at");
  if (!at_func)
    throw std::runtime_error("__ESBMC_list_at not found");

  typet obj_ptr_type = pointer_typet(type_handler_.get_list_element_type());
  symbolt &obj_var = converter_.create_tmp_symbol(
    call_node, "$dict_get_obj$", obj_ptr_type, exprt());
  code_declt obj_decl(symbol_expr(obj_var));
  obj_decl.location() = location;
  then_block.copy_to_operands(obj_decl);

  code_function_callt at_call;
  at_call.function() = symbol_expr(*at_func);
  at_call.lhs() = symbol_expr(obj_var);
  at_call.arguments().push_back(values_member);
  at_call.arguments().push_back(symbol_expr(index_var));
  at_call.type() = obj_ptr_type;
  at_call.location() = location;
  then_block.copy_to_operands(at_call);

  member_exprt obj_value(
    dereference_exprt(
      symbol_expr(obj_var), type_handler_.get_list_element_type()),
    "value",
    pointer_typet(empty_typet()));

  // Cast and assign the retrieved value to result_type
  exprt retrieved_value;
  if (result_type.is_floatbv())
  {
    typecast_exprt value_as_float_ptr(obj_value, pointer_typet(result_type));
    retrieved_value = dereference_exprt(value_as_float_ptr, result_type);
  }
  else if (result_type.is_signedbv() || result_type.is_unsignedbv())
  {
    typecast_exprt value_as_int_ptr(obj_value, pointer_typet(result_type));
    retrieved_value = dereference_exprt(value_as_int_ptr, result_type);
  }
  else if (result_type.is_bool())
  {
    typecast_exprt value_as_bool_ptr(obj_value, pointer_typet(bool_type()));
    retrieved_value = dereference_exprt(value_as_bool_ptr, bool_type());
  }
  else if (result_type == none_type())
  {
    // For none_type, just cast the void* directly
    typecast_exprt value_as_none(obj_value, result_type);
    retrieved_value = value_as_none;
  }
  else if (is_string_result)
  {
    // For string types: cast void* directly to char* (same as
    // handle_dict_subscript). Avoid casting to char[0] which is unusable.
    typecast_exprt value_as_string(obj_value, gen_pointer_type(char_type()));
    retrieved_value = value_as_string;
  }
  else
  {
    typecast_exprt value_as_typed(obj_value, result_type);
    retrieved_value = value_as_typed;
  }

  exprt then_value =
    use_optional
      ? converter_.wrap_in_optional(retrieved_value, effective_result_type)
      : retrieved_value;
  code_assignt value_assign(symbol_expr(result_var), then_value);
  value_assign.location() = location;
  then_block.copy_to_operands(value_assign);

  // Else branch: key not found, use default
  code_blockt else_block;

  if (use_optional)
  {
    // No default given: return Optional(is_none=true) so `result is None` holds.
    constant_exprt none_expr(none_type());
    none_expr.set_value("NULL");
    exprt optional_none =
      converter_.wrap_in_optional(none_expr, effective_result_type);
    code_assignt default_assign(symbol_expr(result_var), optional_none);
    default_assign.location() = location;
    else_block.copy_to_operands(default_assign);
  }
  else if (no_explicit_default)
  {
    // String/pointer type with no explicit default: assign NULL char* to
    // represent None. The isnone evaluator's pointer path handles
    // `result is None` by checking pointer == NULL.
    code_assignt default_assign(
      symbol_expr(result_var), gen_zero(effective_result_type));
    default_assign.location() = location;
    else_block.copy_to_operands(default_assign);
  }
  else if (default_value.type() == none_type() && result_type != none_type())
  {
    // Explicit None default: cast to result_type (represents None as zero)
    typecast_exprt casted_default(default_value, result_type);
    code_assignt default_assign(symbol_expr(result_var), casted_default);
    default_assign.location() = location;
    else_block.copy_to_operands(default_assign);
  }
  else
  {
    code_assignt default_assign(symbol_expr(result_var), default_value);
    default_assign.location() = location;
    else_block.copy_to_operands(default_assign);
  }

  // Create if-then-else
  code_ifthenelset if_stmt;
  if_stmt.cond() = key_found;
  if_stmt.then_case() = then_block;
  if_stmt.else_case() = else_block;
  if_stmt.location() = location;
  converter_.add_instruction(if_stmt);

  return symbol_expr(result_var);
}

exprt python_dict_handler::handle_dict_setdefault(
  const exprt &dict_expr,
  const nlohmann::json &call_node)
{
  locationt location = converter_.get_location_from_decl(call_node);
  typet list_type = type_handler_.get_list_type();

  const auto &args = call_node["args"];

  if (args.empty())
    throw std::runtime_error("setdefault() missing required argument: 'key'");

  exprt key_expr = converter_.get_expr(args[0]);

  // Determine the default value
  bool has_explicit_default = args.size() >= 2;
  exprt default_value;
  if (has_explicit_default)
    default_value = converter_.get_expr(args[1]);
  else
    default_value = gen_zero(none_type());

  // Infer result type
  typet result_type = resolve_expected_type_for_dict_subscript(dict_expr);

  // If we can't infer from dict annotation, use sensible defaults
  if (result_type.is_nil() || result_type.is_empty())
  {
    if (has_explicit_default && default_value.type() != none_type())
      result_type = default_value.type();
    else
      result_type = long_int_type();
  }

  // Strings are stored as char arrays; list_at returns a void* to the first character.
  bool is_string_result =
    (result_type.is_pointer() && result_type.subtype() == char_type()) ||
    (result_type.is_array() && result_type.subtype() == char_type());
  if (is_string_result)
    result_type = gen_pointer_type(char_type());

  if (is_dict_type(result_type))
    throw std::runtime_error("setdefault(): dict value type is not supported");

  // List values are stored by pointer so that
  // `a.setdefault(k, []).append(x)` mutates the stored list.
  const bool is_list_result = (result_type == list_type);

  // Get dict members
  member_exprt keys_member(dict_expr, "keys", list_type);
  member_exprt values_member(dict_expr, "values", list_type);

  const symbolt *try_find_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_try_find_index");
  if (!try_find_func)
    throw std::runtime_error("__ESBMC_list_try_find_index not found");

  // Create temp for index result
  symbolt &index_var = converter_.create_tmp_symbol(
    call_node, "$dict_setdefault_idx$", size_type(), exprt());
  code_declt index_decl(symbol_expr(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  // Get element info for the key
  python_list list_handler(converter_, call_node);
  list_elem_info key_info =
    list_handler.get_list_element_info(call_node, key_expr);

  // Build key arg
  exprt key_arg;
  if (
    key_info.elem_symbol->type.is_pointer() &&
    key_info.elem_symbol->type.subtype() == char_type())
    key_arg = symbol_expr(*key_info.elem_symbol);
  else
    key_arg = address_of_exprt(symbol_expr(*key_info.elem_symbol));

  // Call try_find_index (returns SIZE_MAX if not found)
  code_function_callt try_find_call;
  try_find_call.function() = symbol_expr(*try_find_func);
  try_find_call.lhs() = symbol_expr(index_var);
  try_find_call.arguments().push_back(keys_member);
  try_find_call.arguments().push_back(key_arg);
  try_find_call.arguments().push_back(symbol_expr(*key_info.elem_type_sym));
  try_find_call.arguments().push_back(key_info.elem_size);
  try_find_call.type() = size_type();
  try_find_call.location() = location;
  converter_.add_instruction(try_find_call);

  // Check if key was found (index != SIZE_MAX)
  const BigInt size_max_val = power(2, bv_width(size_type())) - 1;
  constant_exprt size_max(size_max_val, size_type());
  exprt key_found = not_exprt(equality_exprt(symbol_expr(index_var), size_max));

  // Create result variable
  symbolt &result_var = converter_.create_tmp_symbol(
    call_node, "$dict_setdefault_result$", result_type, exprt());
  code_declt result_decl(symbol_expr(result_var));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  // Then branch: key found, retrieve value (no dict mutation)
  code_blockt then_block;

  const symbolt *at_func = symbol_table_.find_symbol("c:@F@__ESBMC_list_at");
  if (!at_func)
    throw std::runtime_error("__ESBMC_list_at not found");

  typet obj_ptr_type = pointer_typet(type_handler_.get_list_element_type());
  symbolt &obj_var = converter_.create_tmp_symbol(
    call_node, "$dict_setdefault_obj$", obj_ptr_type, exprt());
  code_declt obj_decl(symbol_expr(obj_var));
  obj_decl.location() = location;
  then_block.copy_to_operands(obj_decl);

  code_function_callt at_call;
  at_call.function() = symbol_expr(*at_func);
  at_call.lhs() = symbol_expr(obj_var);
  at_call.arguments().push_back(values_member);
  at_call.arguments().push_back(symbol_expr(index_var));
  at_call.type() = obj_ptr_type;
  at_call.location() = location;
  then_block.copy_to_operands(at_call);

  member_exprt obj_value(
    dereference_exprt(
      symbol_expr(obj_var), type_handler_.get_list_element_type()),
    "value",
    pointer_typet(empty_typet()));

  // Cast and assign the retrieved value to result_type
  exprt retrieved_value;
  if (result_type.is_floatbv())
  {
    typecast_exprt value_as_float_ptr(obj_value, pointer_typet(result_type));
    retrieved_value = dereference_exprt(value_as_float_ptr, result_type);
  }
  else if (result_type.is_signedbv() || result_type.is_unsignedbv())
  {
    typecast_exprt value_as_int_ptr(obj_value, pointer_typet(result_type));
    retrieved_value = dereference_exprt(value_as_int_ptr, result_type);
  }
  else if (result_type.is_bool())
  {
    typecast_exprt value_as_bool_ptr(obj_value, pointer_typet(bool_type()));
    retrieved_value = dereference_exprt(value_as_bool_ptr, bool_type());
  }
  else if (is_list_result)
  {
    // List values are stored as raw PyListObject*, so cast the void*
    // straight back, no extra dereference.
    retrieved_value = typecast_exprt(obj_value, result_type);
  }
  else if (
    result_type.is_pointer() && result_type.subtype() != char_type() &&
    !result_type.is_nil())
  {
    // Non-char pointer (e.g., Optional[T] stored as T*): stored by-reference,
    // so dereference as T** to recover the stored T* value.
    typecast_exprt value_as_ptr_ptr(obj_value, pointer_typet(result_type));
    retrieved_value = dereference_exprt(value_as_ptr_ptr, result_type);
  }
  else if (result_type == none_type())
  {
    typecast_exprt value_as_none(obj_value, result_type);
    retrieved_value = value_as_none;
  }
  else
  {
    typecast_exprt value_as_typed(obj_value, result_type);
    retrieved_value = value_as_typed;
  }

  code_assignt value_assign(symbol_expr(result_var), retrieved_value);
  value_assign.location() = location;
  then_block.copy_to_operands(value_assign);

  // Else branch: key not found — insert (key, default) and return default
  code_blockt else_block;

  // Compute a single effective default used for both insertion and return.
  // When the caller omits the default or passes None, approximate None as
  // zero of the result type so that both the stored value and the returned
  // value are identical.
  exprt effective_default =
    (has_explicit_default && default_value.type() != none_type())
      ? default_value
      : gen_zero(result_type);

  // value_arg is the proper char* / typed pointer to the value to insert.
  // Hoisted so the default assignment below can reuse it for string results.
  exprt value_arg;

  {
    const symbolt *push_func =
      symbol_table_.find_symbol("c:@F@__ESBMC_list_push");
    if (!push_func)
      throw std::runtime_error("__ESBMC_list_push not found");

    // Push key into keys list
    code_function_callt push_key_call;
    push_key_call.function() = symbol_expr(*push_func);
    push_key_call.arguments().push_back(keys_member);
    push_key_call.arguments().push_back(key_arg);
    push_key_call.arguments().push_back(symbol_expr(*key_info.elem_type_sym));
    push_key_call.arguments().push_back(key_info.elem_size);
    push_key_call.arguments().push_back(from_integer(BigInt(0), size_type()));
    push_key_call.arguments().push_back(from_integer(
      BigInt(converter_.get_type_handler().is_pointer_free(
        key_info.elem_symbol->type)),
      int_type()));
    push_key_call.type() = bool_type();
    push_key_call.location() = location;
    else_block.copy_to_operands(push_key_call);

    // Push value into values list.
    // List values reuse __ESBMC_list_push_dict_ptr to store the raw pointer.
    if (is_list_result)
    {
      const symbolt *push_ptr_func =
        symbol_table_.find_symbol("c:@F@__ESBMC_list_push_dict_ptr");
      if (!push_ptr_func)
        throw std::runtime_error("__ESBMC_list_push_dict_ptr not found");

      constant_exprt list_type_hash(size_type());
      list_type_hash.set_value(integer2binary(
        generate_nested_dict_type_hash(list_type),
        config.ansi_c.address_width));

      code_function_callt push_list_call;
      push_list_call.function() = symbol_expr(*push_ptr_func);
      push_list_call.arguments().push_back(values_member);
      push_list_call.arguments().push_back(effective_default);
      push_list_call.arguments().push_back(list_type_hash);
      push_list_call.type() = bool_type();
      push_list_call.location() = location;
      else_block.copy_to_operands(push_list_call);
    }
    else
    {
      list_elem_info value_info =
        list_handler.get_list_element_info(call_node, effective_default);

      if (
        value_info.elem_symbol->type.is_pointer() &&
        value_info.elem_symbol->type.subtype() == char_type())
        value_arg = symbol_expr(*value_info.elem_symbol);
      else
        value_arg = address_of_exprt(symbol_expr(*value_info.elem_symbol));

      code_function_callt push_value_call;
      push_value_call.function() = symbol_expr(*push_func);
      push_value_call.arguments().push_back(values_member);
      push_value_call.arguments().push_back(value_arg);
      push_value_call.arguments().push_back(
        symbol_expr(*value_info.elem_type_sym));
      push_value_call.arguments().push_back(value_info.elem_size);
      push_value_call.arguments().push_back(
        from_integer(BigInt(0), size_type()));
      push_value_call.arguments().push_back(from_integer(
        BigInt(converter_.get_type_handler().is_pointer_free(
          value_info.elem_symbol->type)),
        int_type()));
      push_value_call.type() = bool_type();
      push_value_call.location() = location;
      else_block.copy_to_operands(push_value_call);
    }
  }

  // Assign effective_default to result.
  // For strings use value_arg (the char* pointer already staged for insertion)
  // so that the returned pointer is identical to what was pushed into the list.
  // For all other types, effective_default already has the correct type.
  {
    exprt result_expr = is_string_result ? value_arg : effective_default;
    code_assignt default_assign(symbol_expr(result_var), result_expr);
    default_assign.location() = location;
    else_block.copy_to_operands(default_assign);
  }

  // Create if-then-else
  code_ifthenelset if_stmt;
  if_stmt.cond() = key_found;
  if_stmt.then_case() = then_block;
  if_stmt.else_case() = else_block;
  if_stmt.location() = location;
  converter_.add_instruction(if_stmt);

  return symbol_expr(result_var);
}

bool python_dict_handler::is_value_returning_method(
  const std::string &method_name)
{
  return method_name == "pop" || method_name == "get" ||
         method_name == "setdefault" || method_name == "popitem";
}

// Retrieve a typed value from a PyObj's void* value field.
// Used by both handle_dict_pop and handle_dict_popitem.
static exprt
retrieve_list_value(const exprt &obj_value, const typet &result_type)
{
  if (result_type.is_pointer() && result_type.subtype() == char_type())
    return typecast_exprt(obj_value, gen_pointer_type(char_type()));
  if (result_type.is_floatbv())
    return dereference_exprt(
      typecast_exprt(obj_value, pointer_typet(result_type)), result_type);
  if (result_type.is_signedbv() || result_type.is_unsignedbv())
    return dereference_exprt(
      typecast_exprt(obj_value, pointer_typet(result_type)), result_type);
  if (result_type.is_bool())
    return dereference_exprt(
      typecast_exprt(obj_value, pointer_typet(bool_type())), bool_type());
  return typecast_exprt(obj_value, result_type);
}

exprt python_dict_handler::handle_dict_pop(
  const exprt &dict_expr,
  const nlohmann::json &call_node)
{
  locationt location = converter_.get_location_from_decl(call_node);
  typet list_type = type_handler_.get_list_type();

  const auto &args = call_node["args"];
  if (args.empty())
    throw std::runtime_error("pop() missing required argument: 'key'");

  exprt key_expr = converter_.get_expr(args[0]);
  const bool has_default = (args.size() >= 2);

  exprt default_value;
  if (has_default)
  {
    default_value = converter_.get_expr(args[1]);
    default_value = converter_.unwrap_optional_if_needed(default_value);
  }

  // Infer result type
  typet result_type = resolve_expected_type_for_dict_subscript(dict_expr);
  if (result_type.is_nil() || result_type.is_empty())
  {
    if (has_default && default_value.type() != none_type())
      result_type = default_value.type();
    else
      result_type = long_int_type();
  }

  // Normalize string types to char*
  const bool is_string_result =
    (result_type.is_array() && result_type.subtype() == char_type()) ||
    (result_type.is_pointer() && result_type.subtype() == char_type());
  if (is_string_result)
    result_type = gen_pointer_type(char_type());

  member_exprt keys_member(dict_expr, "keys", list_type);
  member_exprt values_member(dict_expr, "values", list_type);

  const symbolt *try_find_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_try_find_index");
  if (!try_find_func)
    throw std::runtime_error("__ESBMC_list_try_find_index not found");

  const symbolt *at_func = symbol_table_.find_symbol("c:@F@__ESBMC_list_at");
  if (!at_func)
    throw std::runtime_error("__ESBMC_list_at not found");

  const symbolt *remove_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_remove_at");
  if (!remove_func)
    throw std::runtime_error("__ESBMC_list_remove_at not found");

  // Create temp for index result
  symbolt &index_var = converter_.create_tmp_symbol(
    call_node, "$dict_pop_idx$", size_type(), exprt());
  code_declt index_decl(symbol_expr(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  // Get element info for the key
  python_list list_handler(converter_, call_node);
  list_elem_info key_info =
    list_handler.get_list_element_info(call_node, key_expr);

  // Call try_find_index (returns SIZE_MAX if not found)
  code_function_callt try_find_call;
  try_find_call.function() = symbol_expr(*try_find_func);
  try_find_call.lhs() = symbol_expr(index_var);
  try_find_call.arguments().push_back(keys_member);

  exprt key_arg;
  if (
    key_info.elem_symbol->type.is_pointer() &&
    key_info.elem_symbol->type.subtype() == char_type())
    key_arg = symbol_expr(*key_info.elem_symbol);
  else
    key_arg = address_of_exprt(symbol_expr(*key_info.elem_symbol));

  try_find_call.arguments().push_back(key_arg);
  try_find_call.arguments().push_back(symbol_expr(*key_info.elem_type_sym));
  try_find_call.arguments().push_back(key_info.elem_size);
  try_find_call.type() = size_type();
  try_find_call.location() = location;
  converter_.add_instruction(try_find_call);

  const BigInt size_max_val = power(2, bv_width(size_type())) - 1;
  constant_exprt size_max(size_max_val, size_type());
  exprt key_found = not_exprt(equality_exprt(symbol_expr(index_var), size_max));

  // Create result variable
  symbolt &result_var = converter_.create_tmp_symbol(
    call_node, "$dict_pop_result$", result_type, exprt());
  code_declt result_decl(symbol_expr(result_var));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  // Then-branch: key found — retrieve value, remove from both lists
  code_blockt then_block;

  typet obj_ptr_type = pointer_typet(type_handler_.get_list_element_type());
  symbolt &obj_var = converter_.create_tmp_symbol(
    call_node, "$dict_pop_obj$", obj_ptr_type, exprt());
  code_declt obj_decl(symbol_expr(obj_var));
  obj_decl.location() = location;
  then_block.copy_to_operands(obj_decl);

  code_function_callt at_call;
  at_call.function() = symbol_expr(*at_func);
  at_call.lhs() = symbol_expr(obj_var);
  at_call.arguments().push_back(values_member);
  at_call.arguments().push_back(symbol_expr(index_var));
  at_call.type() = obj_ptr_type;
  at_call.location() = location;
  then_block.copy_to_operands(at_call);

  member_exprt obj_value(
    dereference_exprt(
      symbol_expr(obj_var), type_handler_.get_list_element_type()),
    "value",
    pointer_typet(empty_typet()));

  exprt retrieved_value = retrieve_list_value(obj_value, result_type);

  code_assignt value_assign(symbol_expr(result_var), retrieved_value);
  value_assign.location() = location;
  then_block.copy_to_operands(value_assign);

  // Remove key and value at found index from both lists
  code_function_callt remove_key_call;
  remove_key_call.function() = symbol_expr(*remove_func);
  remove_key_call.arguments().push_back(keys_member);
  remove_key_call.arguments().push_back(symbol_expr(index_var));
  remove_key_call.type() = bool_type();
  remove_key_call.location() = location;
  then_block.copy_to_operands(remove_key_call);

  code_function_callt remove_value_call;
  remove_value_call.function() = symbol_expr(*remove_func);
  remove_value_call.arguments().push_back(values_member);
  remove_value_call.arguments().push_back(symbol_expr(index_var));
  remove_value_call.type() = bool_type();
  remove_value_call.location() = location;
  then_block.copy_to_operands(remove_value_call);

  // Else-branch: key not found
  code_blockt else_block;

  if (has_default)
  {
    exprt coerced_default = default_value;
    if (default_value.type() != result_type)
      coerced_default = typecast_exprt(default_value, result_type);
    code_assignt default_assign(symbol_expr(result_var), coerced_default);
    default_assign.location() = location;
    else_block.copy_to_operands(default_assign);
  }
  else
  {
    // Raise KeyError
    std::string error_msg = "KeyError: key not found in dictionary";
    exprt msg_size = constant_exprt(
      integer2binary(error_msg.size() + 1, bv_width(size_type())),
      integer2string(error_msg.size() + 1),
      size_type());
    typet str_type = array_typet(char_type(), msg_size);

    symbolt &error_msg_var = converter_.create_tmp_symbol(
      call_node, "$keyerror_msg$", str_type, exprt());
    code_declt error_msg_decl(symbol_expr(error_msg_var));
    error_msg_decl.location() = location;
    else_block.copy_to_operands(error_msg_decl);

    exprt error_string =
      converter_.get_string_builder().build_string_literal(error_msg);
    code_assignt error_msg_assign(symbol_expr(error_msg_var), error_string);
    error_msg_assign.location() = location;
    else_block.copy_to_operands(error_msg_assign);

    std::string keyerror_type_str = "KeyError";
    typet keyerror_type = type_handler_.get_typet(keyerror_type_str);
    exprt exception_struct("struct", keyerror_type);
    exception_struct.copy_to_operands(
      address_of_exprt(symbol_expr(error_msg_var)));

    exprt raise_keyerror = side_effect_exprt("cpp-throw", keyerror_type);
    raise_keyerror.move_to_operands(exception_struct);
    raise_keyerror.location() = location;

    code_expressiont raise_code(raise_keyerror);
    raise_code.location() = location;
    else_block.copy_to_operands(raise_code);
  }

  code_ifthenelset if_stmt;
  if_stmt.cond() = key_found;
  if_stmt.then_case() = then_block;
  if_stmt.else_case() = else_block;
  if_stmt.location() = location;
  converter_.add_instruction(if_stmt);

  return symbol_expr(result_var);
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

typet python_dict_handler::get_popitem_tuple_type(const exprt &dict_expr)
{
  // Get key type from annotation (default: char* / str).
  // Mirrors the value-type lookup in resolve_expected_type_for_dict_subscript.
  typet key_type = empty_typet();
  if (dict_expr.is_symbol())
  {
    const symbolt *sym = symbol_table_.find_symbol(dict_expr.identifier());
    if (sym)
    {
      std::string var_name = sym->name.as_string();
      nlohmann::json var_decl = json_utils::find_var_decl(
        var_name,
        converter_.get_current_func_name(),
        converter_.get_ast_json());
      if (!var_decl.empty() && var_decl.contains("annotation"))
      {
        // For update(other_dict), try dict[K, V] first before using the
        // default dict fallback.
        key_type = get_dict_key_type_from_annotation(var_decl["annotation"]);

        // If annotation is a bare "dict" with a function-call initializer,
        // look up the key type from the function's return annotation.
        if (
          (key_type.is_nil() || key_type.is_empty()) &&
          var_decl["annotation"]["_type"] == "Name" &&
          var_decl["annotation"]["id"] == "dict" &&
          var_decl.contains("value") && var_decl["value"]["_type"] == "Call" &&
          var_decl["value"]["func"]["_type"] == "Name")
        {
          std::string func_name =
            var_decl["value"]["func"]["id"].get<std::string>();
          nlohmann::json func_def = json_utils::find_function(
            converter_.get_ast_json()["body"], func_name);
          if (
            !func_def.empty() && func_def.contains("returns") &&
            !func_def["returns"].is_null())
          {
            key_type = get_dict_key_type_from_annotation(func_def["returns"]);
          }
        }
      }
    }
  }
  if (key_type.is_nil() || key_type.is_empty())
    key_type = gen_pointer_type(char_type());
  if (key_type.is_array() && key_type.subtype() == char_type())
    key_type = gen_pointer_type(char_type());

  // Get value type from annotation (default: long int)
  typet val_type = resolve_expected_type_for_dict_subscript(dict_expr);
  if (val_type.is_nil() || val_type.is_empty())
    val_type = long_int_type();
  if (val_type.is_array() && val_type.subtype() == char_type())
    val_type = gen_pointer_type(char_type());

  return converter_.get_tuple_handler().create_tuple_struct_type(
    {key_type, val_type});
}

exprt python_dict_handler::handle_dict_popitem(
  const exprt &dict_expr,
  const nlohmann::json &call_node)
{
  if (!call_node["args"].empty())
    throw std::runtime_error("popitem() takes no arguments");

  locationt location = converter_.get_location_from_decl(call_node);
  typet list_type = type_handler_.get_list_type();

  member_exprt keys_member(dict_expr, "keys", list_type);
  member_exprt values_member(dict_expr, "values", list_type);

  typet tuple_type = get_popitem_tuple_type(dict_expr);
  const struct_typet &tuple_struct = to_struct_type(tuple_type);
  typet key_type = tuple_struct.components()[0].type();
  typet val_type = tuple_struct.components()[1].type();

  const symbolt *size_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
  if (!size_func)
    throw std::runtime_error("__ESBMC_list_size not found");

  const symbolt *at_func = symbol_table_.find_symbol("c:@F@__ESBMC_list_at");
  if (!at_func)
    throw std::runtime_error("__ESBMC_list_at not found");

  const symbolt *remove_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_remove_at");
  if (!remove_func)
    throw std::runtime_error("__ESBMC_list_remove_at not found");

  // Create result variable (tuple)
  symbolt &result_var = converter_.create_tmp_symbol(
    call_node, "$dict_popitem_result$", tuple_type, exprt());
  code_declt result_decl(symbol_expr(result_var));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  // Get size of the keys list
  symbolt &size_var = converter_.create_tmp_symbol(
    call_node, "$dict_popitem_size$", size_type(), exprt());
  code_declt size_decl(symbol_expr(size_var));
  size_decl.location() = location;
  converter_.add_instruction(size_decl);

  code_function_callt size_call;
  size_call.function() = symbol_expr(*size_func);
  size_call.lhs() = symbol_expr(size_var);
  // list_type is already a pointer (PyListObject*), pass directly
  size_call.arguments().push_back(keys_member);
  size_call.type() = size_type();
  size_call.location() = location;
  converter_.add_instruction(size_call);

  // Empty dict → raise KeyError
  exprt is_empty = equality_exprt(symbol_expr(size_var), gen_zero(size_type()));

  code_blockt empty_block;
  {
    std::string error_msg = "popitem(): dictionary is empty";
    exprt msg_size_expr = constant_exprt(
      integer2binary(error_msg.size() + 1, bv_width(size_type())),
      integer2string(error_msg.size() + 1),
      size_type());
    typet str_type = array_typet(char_type(), msg_size_expr);

    symbolt &err_var = converter_.create_tmp_symbol(
      call_node, "$popitem_err$", str_type, exprt());
    code_declt err_decl(symbol_expr(err_var));
    err_decl.location() = location;
    empty_block.copy_to_operands(err_decl);

    exprt err_str =
      converter_.get_string_builder().build_string_literal(error_msg);
    code_assignt err_assign(symbol_expr(err_var), err_str);
    err_assign.location() = location;
    empty_block.copy_to_operands(err_assign);

    std::string keyerror_type_str = "KeyError";
    typet keyerror_type = type_handler_.get_typet(keyerror_type_str);
    exprt exc_struct("struct", keyerror_type);
    exc_struct.copy_to_operands(address_of_exprt(symbol_expr(err_var)));

    exprt raise = side_effect_exprt("cpp-throw", keyerror_type);
    raise.move_to_operands(exc_struct);
    raise.location() = location;

    code_expressiont raise_code(raise);
    raise_code.location() = location;
    empty_block.copy_to_operands(raise_code);
  }

  // Non-empty: get last (key, value), remove, build tuple
  code_blockt nonempty_block;
  {
    // last_idx = size - 1
    symbolt &last_idx_var = converter_.create_tmp_symbol(
      call_node, "$dict_popitem_last_idx$", size_type(), exprt());
    code_declt last_idx_decl(symbol_expr(last_idx_var));
    last_idx_decl.location() = location;
    nonempty_block.copy_to_operands(last_idx_decl);

    exprt last_idx_expr("-", size_type());
    last_idx_expr.copy_to_operands(symbol_expr(size_var), gen_one(size_type()));
    code_assignt last_idx_assign(symbol_expr(last_idx_var), last_idx_expr);
    last_idx_assign.location() = location;
    nonempty_block.copy_to_operands(last_idx_assign);

    typet obj_ptr_type = pointer_typet(type_handler_.get_list_element_type());

    // Retrieve key
    symbolt &key_obj_var = converter_.create_tmp_symbol(
      call_node, "$dict_popitem_key_obj$", obj_ptr_type, exprt());
    code_declt key_obj_decl(symbol_expr(key_obj_var));
    key_obj_decl.location() = location;
    nonempty_block.copy_to_operands(key_obj_decl);

    code_function_callt key_at_call;
    key_at_call.function() = symbol_expr(*at_func);
    key_at_call.lhs() = symbol_expr(key_obj_var);
    key_at_call.arguments().push_back(keys_member);
    key_at_call.arguments().push_back(symbol_expr(last_idx_var));
    key_at_call.type() = obj_ptr_type;
    key_at_call.location() = location;
    nonempty_block.copy_to_operands(key_at_call);

    // Assign key into tuple before removing from list
    member_exprt key_obj_value(
      dereference_exprt(
        symbol_expr(key_obj_var), type_handler_.get_list_element_type()),
      "value",
      pointer_typet(empty_typet()));
    member_exprt key_field(symbol_expr(result_var), "element_0", key_type);
    code_assignt key_assign(
      key_field, retrieve_list_value(key_obj_value, key_type));
    key_assign.location() = location;
    nonempty_block.copy_to_operands(key_assign);

    // Retrieve value
    symbolt &val_obj_var = converter_.create_tmp_symbol(
      call_node, "$dict_popitem_val_obj$", obj_ptr_type, exprt());
    code_declt val_obj_decl(symbol_expr(val_obj_var));
    val_obj_decl.location() = location;
    nonempty_block.copy_to_operands(val_obj_decl);

    code_function_callt val_at_call;
    val_at_call.function() = symbol_expr(*at_func);
    val_at_call.lhs() = symbol_expr(val_obj_var);
    val_at_call.arguments().push_back(values_member);
    val_at_call.arguments().push_back(symbol_expr(last_idx_var));
    val_at_call.type() = obj_ptr_type;
    val_at_call.location() = location;
    nonempty_block.copy_to_operands(val_at_call);

    // Assign value into tuple before removing from list.
    member_exprt val_obj_value(
      dereference_exprt(
        symbol_expr(val_obj_var), type_handler_.get_list_element_type()),
      "value",
      pointer_typet(empty_typet()));
    member_exprt val_field(symbol_expr(result_var), "element_1", val_type);
    code_assignt val_assign(
      val_field, retrieve_list_value(val_obj_value, val_type));
    val_assign.location() = location;
    nonempty_block.copy_to_operands(val_assign);

    // Now safe to remove: tuple fields are already populated.
    code_function_callt remove_key;
    remove_key.function() = symbol_expr(*remove_func);
    remove_key.arguments().push_back(keys_member);
    remove_key.arguments().push_back(symbol_expr(last_idx_var));
    remove_key.type() = bool_type();
    remove_key.location() = location;
    nonempty_block.copy_to_operands(remove_key);

    code_function_callt remove_val;
    remove_val.function() = symbol_expr(*remove_func);
    remove_val.arguments().push_back(values_member);
    remove_val.arguments().push_back(symbol_expr(last_idx_var));
    remove_val.type() = bool_type();
    remove_val.location() = location;
    nonempty_block.copy_to_operands(remove_val);
  }

  code_ifthenelset if_stmt;
  if_stmt.cond() = is_empty;
  if_stmt.then_case() = empty_block;
  if_stmt.else_case() = nonempty_block;
  if_stmt.location() = location;
  converter_.add_instruction(if_stmt);

  return symbol_expr(result_var);
}

exprt python_dict_handler::handle_dict_fromkeys(const nlohmann::json &call_node)
{
  const auto &args = call_node["args"];

  if (args.empty() || args.size() > 2)
    throw std::runtime_error(
      "fromkeys() takes 1 or 2 arguments (got " + std::to_string(args.size()) +
      ")");

  const nlohmann::json &iterable = args[0];
  if (iterable["_type"] != "List")
    throw std::runtime_error(
      "fromkeys() currently supports a list literal as iterable");

  nlohmann::json value_template;
  if (args.size() == 2)
  {
    value_template = args[1];
  }
  else
  {
    // No explicit default: Python's dict.fromkeys uses None.
    // Emit Constant(None) so the synthetic Dict below stays well-formed.
    value_template = call_node;
    value_template.erase("args");
    value_template.erase("func");
    value_template["_type"] = "Constant";
    value_template["value"] = nullptr;
  }

  // Dedup keys: fromkeys([1, 1, 2], v) == {1: v, 2: v} in Python
  // and collapsing here avoids redundant IR inserts.
  // Constants compare by value, Names by id;
  // other expressions are left distinct.
  auto same_key = [](const nlohmann::json &a, const nlohmann::json &b) -> bool {
    if (a.value("_type", "") != b.value("_type", ""))
      return false;
    const std::string type = a["_type"];
    if (type == "Constant")
      return a.value("value", nlohmann::json(nullptr)) ==
             b.value("value", nlohmann::json(nullptr));
    if (type == "Name")
      return a.value("id", "") == b.value("id", "");
    return false;
  };

  nlohmann::json unique_keys = nlohmann::json::array();
  for (const auto &elt : iterable["elts"])
  {
    bool duplicate = false;
    for (const auto &existing : unique_keys)
      if (same_key(elt, existing))
      {
        duplicate = true;
        break;
      }
    if (!duplicate)
      unique_keys.push_back(elt);
  }

  // Synthesize {k: default for k in unique_keys} and hand it off as if
  // the user had written a Dict literal directly.
  nlohmann::json synthetic_dict = call_node;
  synthetic_dict.erase("args");
  synthetic_dict.erase("func");
  synthetic_dict["_type"] = "Dict";
  synthetic_dict["keys"] = unique_keys;
  synthetic_dict["values"] = nlohmann::json::array();
  for (size_t i = 0; i < unique_keys.size(); ++i)
    synthetic_dict["values"].push_back(value_template);

  locationt location = converter_.get_location_from_decl(call_node);
  std::string dict_name =
    generate_unique_dict_name(synthetic_dict, location) + "_fromkeys";
  struct_typet dict_type = get_dict_struct_type();

  symbolt &dict_sym =
    converter_.create_tmp_symbol(call_node, dict_name, dict_type, exprt());
  code_declt dict_decl(symbol_expr(dict_sym));
  dict_decl.location() = location;
  converter_.add_instruction(dict_decl);

  create_dict_from_literal(synthetic_dict, symbol_expr(dict_sym));
  return symbol_expr(dict_sym);
}

exprt python_dict_handler::handle_dict_update(
  const exprt &dict_expr,
  const nlohmann::json &call_node)
{
  const auto &args = call_node["args"];

  if (args.size() != 1)
    throw std::runtime_error("update() takes exactly one argument");

  const nlohmann::json &arg = args[0];

  if (is_dict_literal(arg))
  {
    // Keep the literal fast path unchanged.
    const auto &keys = arg["keys"];
    const auto &values = arg["values"];

    for (size_t i = 0; i < keys.size(); ++i)
    {
      exprt value_expr = converter_.get_expr(values[i]);
      code_blockt pair_block;
      handle_dict_subscript_assign(
        dict_expr,
        get_key_expr(keys[i]),
        value_expr,
        converter_.get_location_from_decl(keys[i]),
        keys[i],
        pair_block);
      converter_.add_instruction(pair_block);
    }

    return nil_exprt();
  }

  // For update(other_dict), iterate over the source keys/values lists and
  // reinsert each pair into the destination dict.
  exprt other_dict = converter_.get_expr(arg);
  if (!is_dict_type(other_dict.type()))
    throw std::runtime_error("update() argument must be a dict in ESBMC model");

  // Read entries from the source dict through its internal keys/values lists.
  locationt location = converter_.get_location_from_decl(call_node);
  typet list_type = type_handler_.get_list_type();
  member_exprt keys_member(other_dict, "keys", list_type);
  member_exprt values_member(other_dict, "values", list_type);

  // Get the helper used to determine how many entries need to be copied.
  const symbolt *size_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
  if (!size_func)
    throw std::runtime_error("__ESBMC_list_size not found");

  // Recover the key and value types so each entry can be reconstructed.
  python_list list_handler(converter_, call_node);
  typet tuple_type = get_popitem_tuple_type(other_dict);
  const struct_typet &tuple_struct = to_struct_type(tuple_type);
  typet key_type = tuple_struct.components()[0].type();
  typet val_type = tuple_struct.components()[1].type();

  // Compute the number of entries in the source dict
  symbolt &size_var = converter_.create_tmp_symbol(
    call_node, "$dict_update_size$", size_type(), exprt());
  code_declt size_decl(symbol_expr(size_var));
  size_decl.location() = location;
  converter_.add_instruction(size_decl);

  code_function_callt size_call;
  size_call.function() = symbol_expr(*size_func);
  size_call.lhs() = symbol_expr(size_var);
  size_call.arguments().push_back(keys_member);
  size_call.type() = size_type();
  size_call.location() = location;
  converter_.add_instruction(size_call);

  // Create the loop index used to walk over the source entries.
  symbolt &index_var = converter_.create_tmp_symbol(
    call_node, "$dict_update_iter$", size_type(), gen_zero(size_type()));
  code_declt index_decl(symbol_expr(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  code_assignt index_init(symbol_expr(index_var), gen_zero(size_type()));
  index_init.location() = location;
  converter_.add_instruction(index_init);

  exprt loop_cond("<", bool_type());
  loop_cond.copy_to_operands(symbol_expr(index_var), symbol_expr(size_var));

  // Rebuild the current key/value pair from the source lists.
  code_blockt loop_body;
  exprt key_obj = list_handler.build_list_at_call(
    keys_member, symbol_expr(index_var), call_node);
  exprt key_expr = list_handler.extract_pyobject_value(key_obj, key_type);
  exprt value_obj = list_handler.build_list_at_call(
    values_member, symbol_expr(index_var), call_node);
  exprt value_expr = list_handler.extract_pyobject_value(value_obj, val_type);

  // Reuse dict[key] = value handling for each copied entry.
  code_blockt pair_block;
  handle_dict_subscript_assign(
    dict_expr, key_expr, value_expr, location, call_node, pair_block);
  loop_body.copy_to_operands(pair_block);

  // Advance to the next source entry.
  exprt next_index = plus_exprt(symbol_expr(index_var), gen_one(size_type()));
  code_assignt index_update(symbol_expr(index_var), next_index);
  index_update.location() = location;
  loop_body.copy_to_operands(index_update);

  // Copy all entries from the source dict into the destination dict.
  codet while_loop("while");
  while_loop.copy_to_operands(loop_cond, loop_body);
  while_loop.location() = location;
  converter_.add_instruction(while_loop);

  return nil_exprt();
}

exprt python_dict_handler::compare(
  const exprt &lhs,
  const exprt &rhs,
  const std::string &op)
{
  locationt location = lhs.location();
  if (location.is_nil())
    location = rhs.location();

  typet list_type = type_handler_.get_list_type();

  // Get keys and values from both dicts
  member_exprt lhs_keys(lhs, "keys", list_type);
  member_exprt lhs_values(lhs, "values", list_type);
  member_exprt rhs_keys(rhs, "keys", list_type);
  member_exprt rhs_values(rhs, "values", list_type);

  // Find __ESBMC_dict_eq function
  const symbolt *dict_eq_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_dict_eq");

  if (!dict_eq_func)
    throw std::runtime_error("__ESBMC_dict_eq not found in symbol table");

  // Create temp for result
  symbolt &result_var = converter_.create_tmp_symbol(
    nlohmann::json(), "$dict_eq_result$", bool_type(), exprt());
  code_declt result_decl(symbol_expr(result_var));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  // Call __ESBMC_dict_eq(lhs_keys, lhs_values, rhs_keys, rhs_values)
  code_function_callt dict_eq_call;
  dict_eq_call.function() = symbol_expr(*dict_eq_func);
  dict_eq_call.lhs() = symbol_expr(result_var);
  dict_eq_call.arguments().push_back(lhs_keys);
  dict_eq_call.arguments().push_back(lhs_values);
  dict_eq_call.arguments().push_back(rhs_keys);
  dict_eq_call.arguments().push_back(rhs_values);
  dict_eq_call.type() = bool_type();
  dict_eq_call.location() = location;
  converter_.add_instruction(dict_eq_call);

  // Return result
  exprt result = symbol_expr(result_var);
  result.location() = location;

  if (op == "NotEq")
  {
    exprt negated("not", bool_type());
    negated.move_to_operands(result);
    negated.location() = location;
    return negated;
  }

  return result;
}
