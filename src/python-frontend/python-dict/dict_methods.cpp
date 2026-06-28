#include "python_dict_internal.h"

using namespace python_expr;

namespace
{
// Build "key found" = !(index_var == SIZE_MAX) in IREP2, back-migrated once
// (V.3). SIZE_MAX is the dict lookup's not-found sentinel.
exprt build_key_found(const symbolt &index_var)
{
  const BigInt size_max_val = power(2, bv_width(size_type())) - 1;
  const constant_exprt size_max(size_max_val, size_type());
  expr2tc idx2, max2;
  migrate_expr(build_symbol(index_var), idx2);
  migrate_expr(size_max, max2);
  return migrate_expr_back(not2tc(equality2tc(idx2, max2)));
}
} // namespace

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
  exprt keys_member = build_member(dict_expr, "keys", list_type);
  exprt values_member = build_member(dict_expr, "values", list_type);

  const symbolt *try_find_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_try_find_index");
  if (!try_find_func)
    throw std::runtime_error("__ESBMC_list_try_find_index not found");

  // Create temp for index result
  symbolt &index_var = converter_.create_tmp_symbol(
    call_node, "$dict_get_idx$", size_type(), exprt());
  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  // Get element info for the key
  python_list list_handler(converter_, call_node);
  list_elem_info key_info =
    list_handler.get_list_element_info(call_node, key_expr);

  // Call try_find_index (returns SIZE_MAX if not found)
  code_function_callt try_find_call;
  try_find_call.function() = build_symbol(*try_find_func);
  try_find_call.lhs() = build_symbol(index_var);
  try_find_call.arguments().push_back(keys_member);

  exprt key_arg;
  if (
    key_info.elem_symbol->get_type().is_pointer() &&
    key_info.elem_symbol->get_type().subtype() == char_type())
    key_arg = build_symbol(*key_info.elem_symbol);
  else
    key_arg = build_address_of(build_symbol(*key_info.elem_symbol));

  try_find_call.arguments().push_back(key_arg);
  try_find_call.arguments().push_back(build_symbol(*key_info.elem_type_sym));
  try_find_call.arguments().push_back(key_info.elem_size);
  try_find_call.type() = size_type();
  try_find_call.location() = location;
  converter_.add_instruction(try_find_call);

  // Check if key was found (index != SIZE_MAX)
  exprt key_found = build_key_found(index_var); // V.3

  // Create result variable
  symbolt &result_var = converter_.create_tmp_symbol(
    call_node, "$dict_get_result$", effective_result_type, exprt());
  code_declt result_decl(build_symbol(result_var));
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
  code_declt obj_decl(build_symbol(obj_var));
  obj_decl.location() = location;
  then_block.copy_to_operands(obj_decl);

  code_function_callt at_call;
  at_call.function() = build_symbol(*at_func);
  at_call.lhs() = build_symbol(obj_var);
  at_call.arguments().push_back(values_member);
  at_call.arguments().push_back(build_symbol(index_var));
  at_call.type() = obj_ptr_type;
  at_call.location() = location;
  then_block.copy_to_operands(at_call);

  exprt obj_value = build_member(
    build_dereference(
      build_symbol(obj_var), type_handler_.get_list_element_type()),
    "value",
    pointer_typet(empty_typet()));

  // Cast and assign the retrieved value to result_type
  exprt retrieved_value;
  if (result_type.is_floatbv())
  {
    exprt value_as_float_ptr =
      build_typecast(obj_value, pointer_typet(result_type));
    retrieved_value = build_dereference(value_as_float_ptr, result_type);
  }
  else if (result_type.is_signedbv() || result_type.is_unsignedbv())
  {
    exprt value_as_int_ptr =
      build_typecast(obj_value, pointer_typet(result_type));
    retrieved_value = build_dereference(value_as_int_ptr, result_type);
  }
  else if (result_type.is_bool())
  {
    exprt value_as_bool_ptr =
      build_typecast(obj_value, pointer_typet(bool_type()));
    retrieved_value = build_dereference(value_as_bool_ptr, bool_type());
  }
  else if (result_type == none_type())
  {
    // For none_type, just cast the void* directly
    exprt value_as_none = build_typecast(obj_value, result_type);
    retrieved_value = value_as_none;
  }
  else if (is_string_result)
  {
    // For string types: cast void* directly to char* (same as
    // handle_dict_subscript). Avoid casting to char[0] which is unusable.
    exprt value_as_string =
      build_typecast(obj_value, gen_pointer_type(char_type()));
    retrieved_value = value_as_string;
  }
  else
  {
    exprt value_as_typed = build_typecast(obj_value, result_type);
    retrieved_value = value_as_typed;
  }

  exprt then_value =
    use_optional
      ? converter_.wrap_in_optional(retrieved_value, effective_result_type)
      : retrieved_value;
  code_assignt value_assign(build_symbol(result_var), then_value);
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
    code_assignt default_assign(build_symbol(result_var), optional_none);
    default_assign.location() = location;
    else_block.copy_to_operands(default_assign);
  }
  else if (no_explicit_default)
  {
    // String/pointer type with no explicit default: assign NULL char* to
    // represent None. The isnone evaluator's pointer path handles
    // `result is None` by checking pointer == NULL.
    code_assignt default_assign(
      build_symbol(result_var), gen_zero(effective_result_type));
    default_assign.location() = location;
    else_block.copy_to_operands(default_assign);
  }
  else if (default_value.type() == none_type() && result_type != none_type())
  {
    // Explicit None default: cast to result_type (represents None as zero)
    exprt casted_default = build_typecast(default_value, result_type);
    code_assignt default_assign(build_symbol(result_var), casted_default);
    default_assign.location() = location;
    else_block.copy_to_operands(default_assign);
  }
  else
  {
    code_assignt default_assign(build_symbol(result_var), default_value);
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

  return build_symbol(result_var);
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
  exprt keys_member = build_member(dict_expr, "keys", list_type);
  exprt values_member = build_member(dict_expr, "values", list_type);

  const symbolt *try_find_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_try_find_index");
  if (!try_find_func)
    throw std::runtime_error("__ESBMC_list_try_find_index not found");

  // Create temp for index result
  symbolt &index_var = converter_.create_tmp_symbol(
    call_node, "$dict_setdefault_idx$", size_type(), exprt());
  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  // Get element info for the key
  python_list list_handler(converter_, call_node);
  list_elem_info key_info =
    list_handler.get_list_element_info(call_node, key_expr);

  // Build key arg
  exprt key_arg;
  if (
    key_info.elem_symbol->get_type().is_pointer() &&
    key_info.elem_symbol->get_type().subtype() == char_type())
    key_arg = build_symbol(*key_info.elem_symbol);
  else
    key_arg = build_address_of(build_symbol(*key_info.elem_symbol));

  // Call try_find_index (returns SIZE_MAX if not found)
  code_function_callt try_find_call;
  try_find_call.function() = build_symbol(*try_find_func);
  try_find_call.lhs() = build_symbol(index_var);
  try_find_call.arguments().push_back(keys_member);
  try_find_call.arguments().push_back(key_arg);
  try_find_call.arguments().push_back(build_symbol(*key_info.elem_type_sym));
  try_find_call.arguments().push_back(key_info.elem_size);
  try_find_call.type() = size_type();
  try_find_call.location() = location;
  converter_.add_instruction(try_find_call);

  // Check if key was found (index != SIZE_MAX)
  exprt key_found = build_key_found(index_var); // V.3

  // Create result variable
  symbolt &result_var = converter_.create_tmp_symbol(
    call_node, "$dict_setdefault_result$", result_type, exprt());
  code_declt result_decl(build_symbol(result_var));
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
  code_declt obj_decl(build_symbol(obj_var));
  obj_decl.location() = location;
  then_block.copy_to_operands(obj_decl);

  code_function_callt at_call;
  at_call.function() = build_symbol(*at_func);
  at_call.lhs() = build_symbol(obj_var);
  at_call.arguments().push_back(values_member);
  at_call.arguments().push_back(build_symbol(index_var));
  at_call.type() = obj_ptr_type;
  at_call.location() = location;
  then_block.copy_to_operands(at_call);

  exprt obj_value = build_member(
    build_dereference(
      build_symbol(obj_var), type_handler_.get_list_element_type()),
    "value",
    pointer_typet(empty_typet()));

  // Cast and assign the retrieved value to result_type
  exprt retrieved_value;
  if (result_type.is_floatbv())
  {
    exprt value_as_float_ptr =
      build_typecast(obj_value, pointer_typet(result_type));
    retrieved_value = build_dereference(value_as_float_ptr, result_type);
  }
  else if (result_type.is_signedbv() || result_type.is_unsignedbv())
  {
    exprt value_as_int_ptr =
      build_typecast(obj_value, pointer_typet(result_type));
    retrieved_value = build_dereference(value_as_int_ptr, result_type);
  }
  else if (result_type.is_bool())
  {
    exprt value_as_bool_ptr =
      build_typecast(obj_value, pointer_typet(bool_type()));
    retrieved_value = build_dereference(value_as_bool_ptr, bool_type());
  }
  else if (is_list_result)
  {
    // List values are stored as raw PyListObject*, so cast the void*
    // straight back, no extra dereference.
    retrieved_value = build_typecast(obj_value, result_type);
  }
  else if (
    result_type.is_pointer() && result_type.subtype() != char_type() &&
    !result_type.is_nil())
  {
    // Non-char pointer (e.g., Optional[T] stored as T*): stored by-reference,
    // so dereference as T** to recover the stored T* value.
    exprt value_as_ptr_ptr =
      build_typecast(obj_value, pointer_typet(result_type));
    retrieved_value = build_dereference(value_as_ptr_ptr, result_type);
  }
  else if (result_type == none_type())
  {
    exprt value_as_none = build_typecast(obj_value, result_type);
    retrieved_value = value_as_none;
  }
  else
  {
    exprt value_as_typed = build_typecast(obj_value, result_type);
    retrieved_value = value_as_typed;
  }

  code_assignt value_assign(build_symbol(result_var), retrieved_value);
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
      push_list_call.function() = build_symbol(*push_ptr_func);
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
        value_info.elem_symbol->get_type().is_pointer() &&
        value_info.elem_symbol->get_type().subtype() == char_type())
        value_arg = build_symbol(*value_info.elem_symbol);
      else
        value_arg = build_address_of(build_symbol(*value_info.elem_symbol));

      code_function_callt push_value_call;
      push_value_call.function() = build_symbol(*push_func);
      push_value_call.arguments().push_back(values_member);
      push_value_call.arguments().push_back(value_arg);
      push_value_call.arguments().push_back(
        build_symbol(*value_info.elem_type_sym));
      push_value_call.arguments().push_back(value_info.elem_size);
      push_value_call.arguments().push_back(
        from_integer(BigInt(0), size_type()));
      push_value_call.arguments().push_back(from_integer(
        BigInt(converter_.get_type_handler().is_pointer_free(
          value_info.elem_symbol->get_type())),
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
    code_assignt default_assign(build_symbol(result_var), result_expr);
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

  return build_symbol(result_var);
}

exprt python_dict_handler::handle_dict_copy(
  const exprt &dict_expr,
  const nlohmann::json &call_node)
{
  if (!call_node["args"].empty())
    throw std::runtime_error("dict.copy() takes no arguments");

  locationt location = converter_.get_location_from_decl(call_node);
  struct_typet dict_type = get_dict_struct_type();
  typet list_type = type_handler_.get_list_type();

  // Resolve __ESBMC_list_copy: returns a new PyListObject* with the same
  // elements as its argument. Used here on both keys and values lists.
  const symbolt *list_copy_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_copy");
  if (!list_copy_func)
    throw std::runtime_error("__ESBMC_list_copy not found");

  // Allocate destination dict.
  symbolt &new_dict_sym =
    converter_.create_tmp_symbol(call_node, "$dict_copy$", dict_type, exprt());
  code_declt new_dict_decl(build_symbol(new_dict_sym));
  new_dict_decl.location() = location;
  converter_.add_instruction(new_dict_decl);

  // Copy each list independently so mutating the copy leaves the source
  // untouched.
  auto copy_list_member = [&](const irep_idt &name) {
    exprt src = build_member(dict_expr, name, list_type);
    exprt dst = build_member(build_symbol(new_dict_sym), name, list_type);
    code_function_callt copy_call;
    copy_call.function() = build_symbol(*list_copy_func);
    copy_call.arguments().push_back(src);
    copy_call.lhs() = dst;
    copy_call.type() = list_type;
    copy_call.location() = location;
    converter_.add_instruction(copy_call);
  };

  copy_list_member("keys");
  copy_list_member("values");

  return build_symbol(new_dict_sym);
}

exprt python_dict_handler::handle_dict_clear(
  const exprt &dict_expr,
  const nlohmann::json &call_node)
{
  if (!call_node["args"].empty())
    throw std::runtime_error("dict.clear() takes no arguments");

  locationt location = converter_.get_location_from_decl(call_node);
  typet list_type = type_handler_.get_list_type();

  // dict.clear() empties the dict in place by clearing both backing lists.
  // The keys/values members are PyListObject* (get_list_type() is a pointer),
  // so they pass directly to __ESBMC_list_clear, which sets the list size to 0.
  const symbolt *clear_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_clear");
  if (!clear_func)
    throw std::runtime_error("__ESBMC_list_clear not found");

  auto clear_list_member = [&](const irep_idt &name) {
    code_function_callt clear_call;
    clear_call.function() = build_symbol(*clear_func);
    clear_call.arguments().push_back(build_member(dict_expr, name, list_type));
    clear_call.type() = empty_typet();
    clear_call.location() = location;
    converter_.add_instruction(clear_call);
  };

  clear_list_member("keys");
  clear_list_member("values");

  return nil_exprt();
}

bool python_dict_handler::is_value_returning_method(
  const std::string &method_name)
{
  return method_name == "pop" || method_name == "get" ||
         method_name == "setdefault" || method_name == "popitem" ||
         method_name == "copy";
}

// Retrieve a typed value from a PyObj's void* value field.
// Used by both handle_dict_pop and handle_dict_popitem.
static exprt
retrieve_list_value(const exprt &obj_value, const typet &result_type)
{
  if (result_type.is_pointer() && result_type.subtype() == char_type())
    return build_typecast(obj_value, gen_pointer_type(char_type()));
  if (result_type.is_floatbv())
    return build_dereference(
      build_typecast(obj_value, pointer_typet(result_type)), result_type);
  if (result_type.is_signedbv() || result_type.is_unsignedbv())
    return build_dereference(
      build_typecast(obj_value, pointer_typet(result_type)), result_type);
  if (result_type.is_bool())
    return build_dereference(
      build_typecast(obj_value, pointer_typet(bool_type())), bool_type());
  return build_typecast(obj_value, result_type);
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

  exprt keys_member = build_member(dict_expr, "keys", list_type);
  exprt values_member = build_member(dict_expr, "values", list_type);

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
  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  // Get element info for the key
  python_list list_handler(converter_, call_node);
  list_elem_info key_info =
    list_handler.get_list_element_info(call_node, key_expr);

  // Call try_find_index (returns SIZE_MAX if not found)
  code_function_callt try_find_call;
  try_find_call.function() = build_symbol(*try_find_func);
  try_find_call.lhs() = build_symbol(index_var);
  try_find_call.arguments().push_back(keys_member);

  exprt key_arg;
  if (
    key_info.elem_symbol->get_type().is_pointer() &&
    key_info.elem_symbol->get_type().subtype() == char_type())
    key_arg = build_symbol(*key_info.elem_symbol);
  else
    key_arg = build_address_of(build_symbol(*key_info.elem_symbol));

  try_find_call.arguments().push_back(key_arg);
  try_find_call.arguments().push_back(build_symbol(*key_info.elem_type_sym));
  try_find_call.arguments().push_back(key_info.elem_size);
  try_find_call.type() = size_type();
  try_find_call.location() = location;
  converter_.add_instruction(try_find_call);

  exprt key_found = build_key_found(index_var); // V.3

  // Create result variable
  symbolt &result_var = converter_.create_tmp_symbol(
    call_node, "$dict_pop_result$", result_type, exprt());
  code_declt result_decl(build_symbol(result_var));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  // Then-branch: key found — retrieve value, remove from both lists
  code_blockt then_block;

  typet obj_ptr_type = pointer_typet(type_handler_.get_list_element_type());
  symbolt &obj_var = converter_.create_tmp_symbol(
    call_node, "$dict_pop_obj$", obj_ptr_type, exprt());
  code_declt obj_decl(build_symbol(obj_var));
  obj_decl.location() = location;
  then_block.copy_to_operands(obj_decl);

  code_function_callt at_call;
  at_call.function() = build_symbol(*at_func);
  at_call.lhs() = build_symbol(obj_var);
  at_call.arguments().push_back(values_member);
  at_call.arguments().push_back(build_symbol(index_var));
  at_call.type() = obj_ptr_type;
  at_call.location() = location;
  then_block.copy_to_operands(at_call);

  exprt obj_value = build_member(
    build_dereference(
      build_symbol(obj_var), type_handler_.get_list_element_type()),
    "value",
    pointer_typet(empty_typet()));

  exprt retrieved_value = retrieve_list_value(obj_value, result_type);

  code_assignt value_assign(build_symbol(result_var), retrieved_value);
  value_assign.location() = location;
  then_block.copy_to_operands(value_assign);

  // Remove key and value at found index from both lists
  code_function_callt remove_key_call;
  remove_key_call.function() = build_symbol(*remove_func);
  remove_key_call.arguments().push_back(keys_member);
  remove_key_call.arguments().push_back(build_symbol(index_var));
  remove_key_call.type() = bool_type();
  remove_key_call.location() = location;
  then_block.copy_to_operands(remove_key_call);

  code_function_callt remove_value_call;
  remove_value_call.function() = build_symbol(*remove_func);
  remove_value_call.arguments().push_back(values_member);
  remove_value_call.arguments().push_back(build_symbol(index_var));
  remove_value_call.type() = bool_type();
  remove_value_call.location() = location;
  then_block.copy_to_operands(remove_value_call);

  // Else-branch: key not found
  code_blockt else_block;

  if (has_default)
  {
    exprt coerced_default = default_value;
    if (default_value.type() != result_type)
      coerced_default = build_typecast(default_value, result_type);
    code_assignt default_assign(build_symbol(result_var), coerced_default);
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
    code_declt error_msg_decl(build_symbol(error_msg_var));
    error_msg_decl.location() = location;
    else_block.copy_to_operands(error_msg_decl);

    exprt error_string =
      converter_.get_string_builder().build_string_literal(error_msg);
    code_assignt error_msg_assign(build_symbol(error_msg_var), error_string);
    error_msg_assign.location() = location;
    else_block.copy_to_operands(error_msg_assign);

    std::string keyerror_type_str = "KeyError";
    typet keyerror_type = type_handler_.get_typet(keyerror_type_str);
    exprt exception_struct("struct", keyerror_type);
    exception_struct.copy_to_operands(
      build_address_of(build_symbol(error_msg_var)));

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

  return build_symbol(result_var);
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

  exprt keys_member = build_member(dict_expr, "keys", list_type);
  exprt values_member = build_member(dict_expr, "values", list_type);

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
  code_declt result_decl(build_symbol(result_var));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  // Get size of the keys list
  symbolt &size_var = converter_.create_tmp_symbol(
    call_node, "$dict_popitem_size$", size_type(), exprt());
  code_declt size_decl(build_symbol(size_var));
  size_decl.location() = location;
  converter_.add_instruction(size_decl);

  code_function_callt size_call;
  size_call.function() = build_symbol(*size_func);
  size_call.lhs() = build_symbol(size_var);
  // list_type is already a pointer (PyListObject*), pass directly
  size_call.arguments().push_back(keys_member);
  size_call.type() = size_type();
  size_call.location() = location;
  converter_.add_instruction(size_call);

  // Empty dict → raise KeyError (V.3: build the check in IREP2).
  expr2tc size2;
  migrate_expr(build_symbol(size_var), size2);
  exprt is_empty = migrate_expr_back(equality2tc(size2, gen_zero(size2->type)));

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
    code_declt err_decl(build_symbol(err_var));
    err_decl.location() = location;
    empty_block.copy_to_operands(err_decl);

    exprt err_str =
      converter_.get_string_builder().build_string_literal(error_msg);
    code_assignt err_assign(build_symbol(err_var), err_str);
    err_assign.location() = location;
    empty_block.copy_to_operands(err_assign);

    std::string keyerror_type_str = "KeyError";
    typet keyerror_type = type_handler_.get_typet(keyerror_type_str);
    exprt exc_struct("struct", keyerror_type);
    exc_struct.copy_to_operands(build_address_of(build_symbol(err_var)));

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
    code_declt last_idx_decl(build_symbol(last_idx_var));
    last_idx_decl.location() = location;
    nonempty_block.copy_to_operands(last_idx_decl);

    // size_var and the 1 literal are both size_type — same width.
    exprt last_idx_expr =
      build_sub(build_symbol(size_var), gen_one(size_type()), size_type());
    code_assignt last_idx_assign(build_symbol(last_idx_var), last_idx_expr);
    last_idx_assign.location() = location;
    nonempty_block.copy_to_operands(last_idx_assign);

    typet obj_ptr_type = pointer_typet(type_handler_.get_list_element_type());

    // Retrieve key
    symbolt &key_obj_var = converter_.create_tmp_symbol(
      call_node, "$dict_popitem_key_obj$", obj_ptr_type, exprt());
    code_declt key_obj_decl(build_symbol(key_obj_var));
    key_obj_decl.location() = location;
    nonempty_block.copy_to_operands(key_obj_decl);

    code_function_callt key_at_call;
    key_at_call.function() = build_symbol(*at_func);
    key_at_call.lhs() = build_symbol(key_obj_var);
    key_at_call.arguments().push_back(keys_member);
    key_at_call.arguments().push_back(build_symbol(last_idx_var));
    key_at_call.type() = obj_ptr_type;
    key_at_call.location() = location;
    nonempty_block.copy_to_operands(key_at_call);

    // Assign key into tuple before removing from list
    exprt key_obj_value = build_member(
      build_dereference(
        build_symbol(key_obj_var), type_handler_.get_list_element_type()),
      "value",
      pointer_typet(empty_typet()));
    exprt key_field =
      build_member(build_symbol(result_var), "element_0", key_type);
    code_assignt key_assign(
      key_field, retrieve_list_value(key_obj_value, key_type));
    key_assign.location() = location;
    nonempty_block.copy_to_operands(key_assign);

    // Retrieve value
    symbolt &val_obj_var = converter_.create_tmp_symbol(
      call_node, "$dict_popitem_val_obj$", obj_ptr_type, exprt());
    code_declt val_obj_decl(build_symbol(val_obj_var));
    val_obj_decl.location() = location;
    nonempty_block.copy_to_operands(val_obj_decl);

    code_function_callt val_at_call;
    val_at_call.function() = build_symbol(*at_func);
    val_at_call.lhs() = build_symbol(val_obj_var);
    val_at_call.arguments().push_back(values_member);
    val_at_call.arguments().push_back(build_symbol(last_idx_var));
    val_at_call.type() = obj_ptr_type;
    val_at_call.location() = location;
    nonempty_block.copy_to_operands(val_at_call);

    // Assign value into tuple before removing from list.
    exprt val_obj_value = build_member(
      build_dereference(
        build_symbol(val_obj_var), type_handler_.get_list_element_type()),
      "value",
      pointer_typet(empty_typet()));
    exprt val_field =
      build_member(build_symbol(result_var), "element_1", val_type);
    code_assignt val_assign(
      val_field, retrieve_list_value(val_obj_value, val_type));
    val_assign.location() = location;
    nonempty_block.copy_to_operands(val_assign);

    // Now safe to remove: tuple fields are already populated.
    code_function_callt remove_key;
    remove_key.function() = build_symbol(*remove_func);
    remove_key.arguments().push_back(keys_member);
    remove_key.arguments().push_back(build_symbol(last_idx_var));
    remove_key.type() = bool_type();
    remove_key.location() = location;
    nonempty_block.copy_to_operands(remove_key);

    code_function_callt remove_val;
    remove_val.function() = build_symbol(*remove_func);
    remove_val.arguments().push_back(values_member);
    remove_val.arguments().push_back(build_symbol(last_idx_var));
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

  return build_symbol(result_var);
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
  code_declt dict_decl(build_symbol(dict_sym));
  dict_decl.location() = location;
  converter_.add_instruction(dict_decl);

  create_dict_from_literal(synthetic_dict, build_symbol(dict_sym));
  return build_symbol(dict_sym);
}

exprt python_dict_handler::handle_dict_constructor(
  const nlohmann::json &call_node)
{
  if (!call_node.contains("args") || !call_node["args"].is_array())
    return nil_exprt();
  const auto &args = call_node["args"];

  // dict() / dict(a=1, b=2, ...): no positional iterable, build the dict from
  // the keyword arguments (an empty dict when there are none). Each keyword
  // name becomes a string-constant key. `dict(**other)` carries a null keyword
  // arg and is not modelled — bail so it is not silently mis-lowered.
  if (args.empty())
  {
    nlohmann::json keys = nlohmann::json::array();
    nlohmann::json values = nlohmann::json::array();
    if (call_node.contains("keywords") && call_node["keywords"].is_array())
    {
      for (const auto &kw : call_node["keywords"])
      {
        if (!kw.contains("arg") || !kw["arg"].is_string())
          return nil_exprt();
        nlohmann::json key_node;
        key_node["_type"] = "Constant";
        key_node["value"] = kw["arg"];
        for (const char *f :
             {"lineno", "col_offset", "end_lineno", "end_col_offset"})
          if (call_node.contains(f))
            key_node[f] = call_node[f];
        keys.push_back(std::move(key_node));
        values.push_back(kw["value"]);
      }
    }
    nlohmann::json synthetic_dict = call_node;
    synthetic_dict.erase("args");
    synthetic_dict.erase("func");
    synthetic_dict.erase("keywords");
    synthetic_dict["_type"] = "Dict";
    synthetic_dict["keys"] = std::move(keys);
    synthetic_dict["values"] = std::move(values);
    return get_dict_literal(synthetic_dict);
  }

  if (args.size() != 1)
    return nil_exprt();

  // Peel any nesting of set/list/tuple/frozenset wrappers around the
  // iterable. dict(set([(k,v),...])) and dict(list([])) are common, and
  // dict(list(set(...))) appears occasionally — stop at the first inner
  // value that is not a recognised wrapper Call.
  const nlohmann::json *arg = &args[0];
  while (arg->value("_type", "") == "Call" && arg->contains("func") &&
         (*arg)["func"].value("_type", "") == "Name")
  {
    const std::string id = (*arg)["func"].value("id", "");
    if (
      (id != "set" && id != "list" && id != "tuple" && id != "frozenset") ||
      !arg->contains("args") || !(*arg)["args"].is_array() ||
      (*arg)["args"].size() != 1)
      break;
    arg = &(*arg)["args"][0];
  }

  const std::string &arg_type = arg->value("_type", "");
  if (arg_type != "List" && arg_type != "Tuple" && arg_type != "Set")
    return nil_exprt();
  if (!arg->contains("elts") || !(*arg)["elts"].is_array())
    return nil_exprt();

  nlohmann::json keys = nlohmann::json::array();
  nlohmann::json values = nlohmann::json::array();
  for (const auto &elt : (*arg)["elts"])
  {
    // (k, v) tuple element.
    if (
      elt.value("_type", "") == "Tuple" && elt.contains("elts") &&
      elt["elts"].is_array() && elt["elts"].size() == 2)
    {
      keys.push_back(elt["elts"][0]);
      values.push_back(elt["elts"][1]);
      continue;
    }
    // 2-char string Constant: dict(["ab"]) → {'a': 'b'}.
    if (
      elt.value("_type", "") == "Constant" && elt.contains("value") &&
      elt["value"].is_string() && elt["value"].get<std::string>().size() == 2)
    {
      const std::string s = elt["value"].get<std::string>();
      nlohmann::json k = elt;
      nlohmann::json v = elt;
      k["value"] = std::string(1, s[0]);
      v["value"] = std::string(1, s[1]);
      keys.push_back(std::move(k));
      values.push_back(std::move(v));
      continue;
    }
    // Anything else.
    return nil_exprt();
  }

  // Hand off to the Dict-literal lowering used for {k:v, ...}.
  nlohmann::json synthetic_dict = call_node;
  synthetic_dict.erase("args");
  synthetic_dict.erase("func");
  synthetic_dict["_type"] = "Dict";
  synthetic_dict["keys"] = std::move(keys);
  synthetic_dict["values"] = std::move(values);
  return get_dict_literal(synthetic_dict);
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
  exprt keys_member = build_member(other_dict, "keys", list_type);
  exprt values_member = build_member(other_dict, "values", list_type);

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
  code_declt size_decl(build_symbol(size_var));
  size_decl.location() = location;
  converter_.add_instruction(size_decl);

  code_function_callt size_call;
  size_call.function() = build_symbol(*size_func);
  size_call.lhs() = build_symbol(size_var);
  size_call.arguments().push_back(keys_member);
  size_call.type() = size_type();
  size_call.location() = location;
  converter_.add_instruction(size_call);

  // Create the loop index used to walk over the source entries.
  symbolt &index_var = converter_.create_tmp_symbol(
    call_node, "$dict_update_iter$", size_type(), gen_zero(size_type()));
  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  code_assignt index_init(build_symbol(index_var), gen_zero(size_type()));
  index_init.location() = location;
  converter_.add_instruction(index_init);

  // (both $dict_update_iter$ and $dict_update_size$ are size_type — same width)
  exprt loop_cond =
    build_less_than(build_symbol(index_var), build_symbol(size_var));

  // Rebuild the current key/value pair from the source lists.
  code_blockt loop_body;
  exprt key_obj = list_handler.build_list_at_call(
    keys_member, build_symbol(index_var), call_node);
  exprt key_expr = list_handler.extract_pyobject_value(key_obj, key_type);
  exprt value_obj = list_handler.build_list_at_call(
    values_member, build_symbol(index_var), call_node);
  exprt value_expr = list_handler.extract_pyobject_value(value_obj, val_type);

  // Reuse dict[key] = value handling for each copied entry.
  code_blockt pair_block;
  handle_dict_subscript_assign(
    dict_expr, key_expr, value_expr, location, call_node, pair_block);
  loop_body.copy_to_operands(pair_block);

  // Advance to the next source entry (V.3: build index + 1 in IREP2).
  exprt next_index =
    build_add(build_symbol(index_var), gen_one(size_type()), size_type());
  code_assignt index_update(build_symbol(index_var), next_index);
  index_update.location() = location;
  loop_body.copy_to_operands(index_update);

  // Copy all entries from the source dict into the destination dict.
  codet while_loop("while");
  while_loop.copy_to_operands(loop_cond, loop_body);
  while_loop.location() = location;
  converter_.add_instruction(while_loop);

  return nil_exprt();
}
