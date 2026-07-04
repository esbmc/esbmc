#include "python_dict_internal.h"

using namespace python_expr;

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
  exprt keys_member = build_member(dict_expr, "keys", list_type);
  exprt values_member = build_member(dict_expr, "values", list_type);

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

  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  // Get element info for the key
  python_list list_handler(converter_, slice_node);
  list_elem_info key_info =
    list_handler.get_list_element_info(slice_node, key_expr);

  // Call try_find_index(keys, key, type_hash, size) — returns SIZE_MAX if absent
  code_function_callt find_call;
  find_call.function() = build_symbol(*find_func);
  find_call.lhs() = build_symbol(index_var);
  find_call.arguments().push_back(keys_member);

  exprt key_arg;
  if (
    key_info.elem_symbol->get_type().is_pointer() &&
    key_info.elem_symbol->get_type().subtype() == char_type())
    key_arg = build_symbol(*key_info.elem_symbol);
  else
    key_arg = build_address_of(build_symbol(*key_info.elem_symbol));

  find_call.arguments().push_back(key_arg);
  find_call.arguments().push_back(build_symbol(*key_info.elem_type_sym));
  find_call.arguments().push_back(key_info.elem_size);
  find_call.type() = size_type();
  find_call.location() = location;
  converter_.add_instruction(find_call);

  // If index == SIZE_MAX the key was not found: throw KeyError so that
  // try/except KeyError handlers can catch it (instead of failing the property).
  {
    // V.3: build the not-found check (index == SIZE_MAX) in IREP2.
    const BigInt size_max_val = power(2, bv_width(size_type())) - 1;
    const constant_exprt size_max(size_max_val, size_type());
    expr2tc idx2, max2;
    migrate_expr(build_symbol(index_var), idx2);
    migrate_expr(size_max, max2);
    exprt key_not_found = migrate_expr_back(equality2tc(idx2, max2));

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
    code_declt err_msg_decl(build_symbol(err_msg_var));
    err_msg_decl.location() = location;

    exprt err_str =
      converter_.get_string_builder().build_string_literal(keyerror_msg);
    code_assignt err_msg_assign(build_symbol(err_msg_var), err_str);
    err_msg_assign.location() = location;

    exprt exc_struct("struct", keyerror_type);
    exc_struct.copy_to_operands(build_address_of(build_symbol(err_msg_var)));

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

  code_declt obj_decl(build_symbol(obj_var));
  obj_decl.location() = location;
  converter_.add_instruction(obj_decl);

  // Call list_at(values, index)
  code_function_callt at_call;
  at_call.function() = build_symbol(*at_func);
  at_call.lhs() = build_symbol(obj_var);
  at_call.arguments().push_back(values_member);
  at_call.arguments().push_back(build_symbol(index_var));
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
      element_type = elem_sym->get_type();
  }

  // Create dereference and explicitly set its type
  exprt deref_obj = build_dereference(build_symbol(obj_var), element_type);
  deref_obj.type() = element_type;
  exprt obj_value =
    build_member(deref_obj, "value", pointer_typet(empty_typet()));

  // Handle dict types
  if (!resolved_type.is_nil() && is_dict_type(resolved_type))
  {
    return retrieve_nested_dict_value(
      slice_node, obj_value, resolved_type, location);
  }

  // Handle list types
  if (resolved_type == list_type)
  {
    exprt value_as_list_ptr_ptr =
      build_typecast(obj_value, pointer_typet(list_type));
    exprt list_ptr = build_dereference(value_as_list_ptr_ptr, list_type);
    list_ptr.type() = list_type;

    // Create a temporary symbol for this list to store in the type map
    symbolt &list_result = converter_.create_tmp_symbol(
      slice_node, "$dict_list_result$", list_type, exprt());

    code_declt list_decl(build_symbol(list_result));
    list_decl.location() = location;
    converter_.add_instruction(list_decl);

    code_assignt list_assign(build_symbol(list_result), list_ptr);
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

    return build_symbol(list_result);
  }

  // A heterogeneous int/float dict stores each value inline with its own
  // type_id and native bit pattern (the float path is disabled for mixed
  // dicts — see dict_construction.cpp — so there is no float_buf indirection).
  // A single static resolved_type misreads whichever value carries the other
  // type: {0: 1, 1: 2.5} resolves to int and reads 2.5's bits through the int
  // path, and symmetrically {0: 2.5, 1: 1} misreads the int. Read by the
  // value's runtime type_id instead, yielding a double either way (int values
  // widen; value equality still holds), mirroring how .values()/dict_eq read a
  // mixed dict. Only literal-built dicts have a values-list in the map; a dict
  // parameter returns an empty id and falls through to the static paths below.
  if (
    resolved_type.is_signedbv() || resolved_type.is_unsignedbv() ||
    resolved_type.is_floatbv())
  {
    const std::string dict_id = dict_expr.is_symbol()
                                  ? dict_expr.identifier().as_string()
                                  : std::string();
    const std::string vals_id =
      dict_id.empty() ? std::string() : get_internal_list_id(dict_id, false);
    if (!vals_id.empty() && python_list::has_mixed_numeric_types(vals_id))
    {
      const size_t float_type_id =
        std::hash<std::string>{}(type_handler_.type_to_string(
          type_handler_.get_typet(std::string("float"))));
      // Python int values are stored as long_long (64-bit) and read as int64_t
      // by the model (list.c), so recover them at that width — not
      // config.ansi_c.long_int_width, which is 32 under LLP64/ILP32. (Bignum
      // --int-encoding stores ints wider than 8 bytes and is out of scope here,
      // matching list.c's own size==8-gated numeric path.)
      const typet long_t = long_long_int_type();

      exprt tid = build_member(deref_obj, "type_id", size_type());
      exprt is_float =
        equality_exprt(tid, from_integer(float_type_id, size_type()));

      exprt as_double = build_dereference(
        build_typecast(obj_value, pointer_typet(double_type())), double_type());
      as_double.type() = double_type();

      exprt as_int = build_dereference(
        build_typecast(obj_value, pointer_typet(long_t)), long_t);
      as_int.type() = long_t;

      exprt result =
        if_exprt(is_float, as_double, build_typecast(as_int, double_type()));
      result.type() = double_type();
      return result;
    }
  }

  // Handle float types
  if (resolved_type.is_floatbv())
  {
    exprt value_as_float_ptr =
      build_typecast(obj_value, pointer_typet(resolved_type));
    exprt result = build_dereference(value_as_float_ptr, resolved_type);
    result.type() = resolved_type;
    return result;
  }

  // Handle integer types
  if (resolved_type.is_signedbv() || resolved_type.is_unsignedbv())
  {
    exprt value_as_int_ptr =
      build_typecast(obj_value, pointer_typet(resolved_type));
    exprt result = build_dereference(value_as_int_ptr, resolved_type);
    result.type() = resolved_type;
    return result;
  }

  // Handle boolean types
  if (resolved_type.is_bool())
  {
    exprt value_as_bool_ptr =
      build_typecast(obj_value, pointer_typet(bool_type()));
    exprt result = build_dereference(value_as_bool_ptr, bool_type());
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
    exprt value_as_ptr_ptr =
      build_typecast(obj_value, pointer_typet(resolved_type));
    exprt result = build_dereference(value_as_ptr_ptr, resolved_type);
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
      exprt value_as_struct_ptr =
        build_typecast(obj_value, pointer_typet(underlying));
      exprt result = build_dereference(value_as_struct_ptr, underlying);
      result.type() = underlying;
      return result;
    }
  }

  // Default: cast void* to char* for string values
  exprt value_as_string =
    build_typecast(obj_value, gen_pointer_type(char_type()));
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

  exprt keys_member = build_member(dict_expr, "keys", list_type);
  exprt values_member = build_member(dict_expr, "values", list_type);

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

  // Record the assigned value's type so a later subscript read can detect a
  // heterogeneous int/float dict: a value assigned via d[k]=v reaches the same
  // misread path as a mixed literal (see handle_dict_subscript). Only
  // literal-built dicts (including `{}` then assigned) have a values-list id.
  if (dict_expr.is_symbol())
  {
    const std::string vals_id =
      get_internal_list_id(dict_expr.identifier().as_string(), false);
    if (!vals_id.empty())
      list_handler.add_type_info(vals_id, std::string(), value.type());
  }

  symbolt &index_var = converter_.create_tmp_symbol(
    node, "$dict_update_idx$", size_type(), gen_zero(size_type()));

  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  update_block.copy_to_operands(index_decl);

  // Get element info for the key
  list_elem_info key_info = list_handler.get_list_element_info(node, key_expr);

  // Call find_index(keys, key, type_hash, size) to get the index
  code_function_callt find_call;
  find_call.function() = build_symbol(*find_func);
  find_call.lhs() = build_symbol(index_var);
  find_call.arguments().push_back(keys_member);

  exprt key_arg;
  if (
    key_info.elem_symbol->get_type().is_pointer() &&
    key_info.elem_symbol->get_type().subtype() == char_type())
    key_arg = build_symbol(*key_info.elem_symbol);
  else
    key_arg = build_address_of(build_symbol(*key_info.elem_symbol));

  find_call.arguments().push_back(key_arg);
  find_call.arguments().push_back(build_symbol(*key_info.elem_type_sym));
  find_call.arguments().push_back(key_info.elem_size);
  find_call.type() = size_type();
  find_call.location() = location;
  update_block.copy_to_operands(find_call);

  // Update value at index using list_set_at
  list_elem_info value_info = list_handler.get_list_element_info(node, value);

  code_function_callt set_value_call;
  set_value_call.function() = build_symbol(*set_func);
  set_value_call.arguments().push_back(values_member);
  set_value_call.arguments().push_back(build_symbol(index_var));

  exprt value_arg;
  if (
    value_info.elem_symbol->get_type().is_pointer() &&
    value_info.elem_symbol->get_type().subtype() == char_type())
    value_arg = build_symbol(*value_info.elem_symbol);
  else
    value_arg = build_address_of(build_symbol(*value_info.elem_symbol));

  // float_type_id: route a float value into the real-sorted __ESBMC_float_buf
  // (and record its float_idx) so .values()/.items() reads it back correctly
  // rather than as float_buf[0] (#5501). Mirrors the dict-literal value push.
  exprt value_float_type_id =
    value_info.elem_symbol->get_type().is_floatbv()
      ? static_cast<exprt>(build_symbol(*value_info.elem_type_sym))
      : static_cast<exprt>(from_integer(BigInt(0), size_type()));

  set_value_call.arguments().push_back(value_arg);
  set_value_call.arguments().push_back(build_symbol(*value_info.elem_type_sym));
  set_value_call.arguments().push_back(value_info.elem_size);
  set_value_call.arguments().push_back(value_float_type_id);
  set_value_call.arguments().push_back(from_integer(
    BigInt(converter_.get_type_handler().is_pointer_free(
      value_info.elem_symbol->get_type())),
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
  push_key_call.function() = build_symbol(*push_func);
  push_key_call.arguments().push_back(keys_member);
  push_key_call.arguments().push_back(key_arg);
  push_key_call.arguments().push_back(build_symbol(*key_info.elem_type_sym));
  push_key_call.arguments().push_back(key_info.elem_size);
  push_key_call.arguments().push_back(from_integer(BigInt(0), size_type()));
  push_key_call.arguments().push_back(from_integer(
    BigInt(converter_.get_type_handler().is_pointer_free(
      key_info.elem_symbol->get_type())),
    int_type()));
  push_key_call.type() = bool_type();
  push_key_call.location() = location;
  insert_block.copy_to_operands(push_key_call);

  // Push value
  code_function_callt push_value_call;
  push_value_call.function() = build_symbol(*push_func);
  push_value_call.arguments().push_back(values_member);
  push_value_call.arguments().push_back(value_arg);
  push_value_call.arguments().push_back(
    build_symbol(*value_info.elem_type_sym));
  push_value_call.arguments().push_back(value_info.elem_size);
  push_value_call.arguments().push_back(value_float_type_id);
  push_value_call.arguments().push_back(from_integer(
    BigInt(converter_.get_type_handler().is_pointer_free(
      value_info.elem_symbol->get_type())),
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
  exprt keys_member = build_member(dict_expr, "keys", list_type);

  nlohmann::json dummy_json;
  python_list list_handler(converter_, dummy_json);

  exprt contains_result = list_handler.contains(key_expr, keys_member);

  if (negated)
  {
    // V.3: build the negated-membership result in IREP2.
    expr2tc cr2;
    migrate_expr(contains_result, cr2);
    return migrate_expr_back(not2tc(cr2));
  }

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
  exprt keys_member = build_member(dict_expr, "keys", list_type);
  exprt values_member = build_member(dict_expr, "values", list_type);

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

  code_declt error_msg_decl(build_symbol(error_msg_var));
  error_msg_decl.location() = location;
  target_block.copy_to_operands(error_msg_decl);

  // Assign the string literal to the temp variable
  exprt error_string =
    converter_.get_string_builder().build_string_literal(error_msg);
  code_assignt error_msg_assign(build_symbol(error_msg_var), error_string);
  error_msg_assign.location() = location;
  target_block.copy_to_operands(error_msg_assign);

  // Construct exception struct with address of the temp variable
  exprt exception_struct("struct", keyerror_type);
  exception_struct.copy_to_operands(
    build_address_of(build_symbol(error_msg_var)));

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

  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  delete_block.copy_to_operands(index_decl);

  // Get element info for the key
  python_list list_handler(converter_, slice_node);
  list_elem_info key_info =
    list_handler.get_list_element_info(slice_node, key_expr);

  // Call find_index(keys, key, type_hash, size) to get the index of the key
  code_function_callt find_call;
  find_call.function() = build_symbol(*find_func);
  find_call.lhs() = build_symbol(index_var);
  find_call.arguments().push_back(keys_member);

  exprt key_arg;
  if (
    key_info.elem_symbol->get_type().is_pointer() &&
    key_info.elem_symbol->get_type().subtype() == char_type())
    key_arg = build_symbol(*key_info.elem_symbol);
  else
    key_arg = build_address_of(build_symbol(*key_info.elem_symbol));

  find_call.arguments().push_back(key_arg);
  find_call.arguments().push_back(build_symbol(*key_info.elem_type_sym));
  find_call.arguments().push_back(key_info.elem_size);
  find_call.type() = size_type();
  find_call.location() = location;
  delete_block.copy_to_operands(find_call);

  // Call list_remove_at(keys, index) to remove the key
  code_function_callt remove_key_call;
  remove_key_call.function() = build_symbol(*remove_func);
  remove_key_call.arguments().push_back(keys_member);
  remove_key_call.arguments().push_back(build_symbol(index_var));
  remove_key_call.type() = bool_type();
  remove_key_call.location() = location;
  delete_block.copy_to_operands(remove_key_call);

  // Call list_remove_at(values, index) to remove the corresponding value
  code_function_callt remove_value_call;
  remove_value_call.function() = build_symbol(*remove_func);
  remove_value_call.arguments().push_back(values_member);
  remove_value_call.arguments().push_back(build_symbol(index_var));
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
