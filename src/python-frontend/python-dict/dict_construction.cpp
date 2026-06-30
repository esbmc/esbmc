#include "python_dict_internal.h"

using namespace python_expr;

namespace
{
// True when `node` is a call to the built-in range(): range(stop),
// range(start, stop), or range(start, stop, step).
bool is_range_call(const nlohmann::json &node)
{
  return node.contains("_type") && node["_type"] == "Call" &&
         node.contains("func") && node["func"].contains("_type") &&
         node["func"]["_type"] == "Name" && node["func"].contains("id") &&
         node["func"]["id"] == "range" && node.contains("args");
}
} // namespace

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

  code_declt dict_decl(build_symbol(dict_sym));
  dict_decl.location() = location;
  converter_.add_instruction(dict_decl);

  // Initialize the dictionary with its literal values
  create_dict_from_literal(element, build_symbol(dict_sym));

  // Return the symbol expression pointing to the initialized dictionary
  return build_symbol(dict_sym);
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
  code_declt dict_decl(build_symbol(dict_sym));
  dict_decl.location() = location;
  converter_.add_instruction(dict_decl);

  // Start with an empty dict and fill it inside the comprehension loop
  nlohmann::json empty_dict = element;
  empty_dict["_type"] = "Dict";
  empty_dict["keys"] = nlohmann::json::array();
  empty_dict["values"] = nlohmann::json::array();
  create_dict_from_literal(empty_dict, build_symbol(dict_sym));

  // A dict comprehension over range(...) with a non-constant bound, e.g.
  // {k: v for i in range(n)}. Materialising range(n) into a backing list only
  // sets the list size and leaves its elements nondeterministic (see #5222 /
  // python_list::handle_symbolic_range), so reading them back as the
  // comprehension target produced wrong keys. Iterate via a counter whose
  // value IS the range element instead, mirroring the for-loop-over-range
  // counter lowering that list comprehensions already use. Constant ranges keep
  // the existing concrete-materialisation path (which works), and ranges with
  // an explicit step fall through too (the step direction would change the
  // loop condition).
  if (target["_type"] == "Name" && is_range_call(iter))
  {
    const auto &range_args = iter["args"];
    auto is_const_int_arg = [](const nlohmann::json &arg) {
      if (arg.contains("_type") && arg["_type"] == "Constant")
        return true;
      return arg.contains("_type") && arg["_type"] == "UnaryOp" &&
             arg.contains("operand") && arg["operand"].contains("_type") &&
             arg["operand"]["_type"] == "Constant";
    };
    const bool all_const =
      std::all_of(range_args.begin(), range_args.end(), is_const_int_arg);
    if ((range_args.size() == 1 || range_args.size() == 2) && !all_const)
      return build_range_dict_comprehension(element, generator, dict_sym);
  }

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

    code_declt tmp_var_decl(build_symbol(tmp_var_symbol));
    tmp_var_decl.location() = location;
    converter_.add_instruction(tmp_var_decl);

    code_function_callt new_call;
    new_call.function() = call.function();
    new_call.arguments() = call.arguments();
    new_call.lhs() = build_symbol(tmp_var_symbol);
    new_call.type() = list_type;
    new_call.location() = location;
    converter_.add_instruction(new_call);

    iterable_expr = build_symbol(tmp_var_symbol);
  }

  // A tuple target (`{k: v for a, b in pairs}`) iterates a list of tuples and
  // unpacks each element into the component names. Hold the element in a temp
  // of the tuple's struct type and let tuple_handler::handle_tuple_unpacking
  // emit the per-component assignments (the same path `a, b = t` uses), so the
  // key/value expressions see the unpacked names.
  const bool is_tuple_target = (target["_type"] == "Tuple");
  typet tuple_elem_type;
  if (is_tuple_target)
  {
    if (!iterable_expr.is_symbol())
      throw std::runtime_error(
        "DictComp tuple target requires a named list of tuples");
    tuple_elem_type =
      converter_.name_space().follow(python_list::get_list_element_type(
        iterable_expr.identifier().as_string(), 0));
    if (!converter_.get_tuple_handler().is_tuple_type(tuple_elem_type))
      throw std::runtime_error(
        "DictComp tuple target requires iterating a list of tuples");
  }
  else if (target["_type"] != "Name")
    throw std::runtime_error("Only simple targets are supported in DictComp");

  std::string loop_var_name =
    is_tuple_target ? "$dictcomp_tuple$" : target["id"].get<std::string>();
  symbol_id loop_var_sid = converter_.create_symbol_id();
  loop_var_sid.set_object(loop_var_name);

  // Infer the comprehension target's element type. Defaulting to any_type()
  // boxes the loop variable's value through the raw dynamic backing store,
  // whose alignment ESBMC cannot prove when a later dict lookup reads the key
  // as a 64-bit word in __ESBMC_values_equal (dereference-alignment failure).
  // Mirror the subscript read used by the desugared for-loop: take the element
  // type from the static list_type_map and only fall back to any_type() when it
  // cannot be determined, so no currently-working case regresses.
  typet loop_var_type = any_type();
  bool mixed_numeric = false;
  if (is_tuple_target)
  {
    // The temp holds the element tuple; component reads come from unpacking.
    loop_var_type = tuple_elem_type;
  }
  else if (
    iter.contains("_type") && iter["_type"] == "Call" &&
    iter.contains("func") && iter["func"].contains("_type") &&
    iter["func"]["_type"] == "Name" && iter["func"].contains("id") &&
    iter["func"]["id"] == "range")
  {
    // range(...) produces integers, so keep the loop variable in the
    // same type used later by dict lookups
    loop_var_type = long_long_int_type();
  }
  else if (iterable_expr.is_symbol())
  {
    // numeric_element_type() is non-throwing: it returns the common numeric
    // element type (double for an int/float mix), or an empty typet() when the
    // list is unknown, empty, or contains any non-numeric / mixed-width element.
    // Restricting specialisation to all-numeric lists keeps the read sound and
    // leaves every other case on the previous any_type() path (no regression).
    const std::string list_id = iterable_expr.identifier().as_string();
    typet num = python_list::numeric_element_type(list_id);
    if (num != typet())
    {
      loop_var_type = num;
      mixed_numeric = python_list::has_mixed_numeric_types(list_id);
    }
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
  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  code_assignt index_init(build_symbol(index_var), gen_zero(size_type()));
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
    code_declt length_decl(build_symbol(length_var));
    length_decl.location() = location;
    converter_.add_instruction(length_decl);

    code_function_callt size_call;
    size_call.function() = build_symbol(*size_func);
    size_call.arguments().push_back(
      iterable_expr.type().is_pointer() ? iterable_expr
                                        : build_address_of(iterable_expr));
    size_call.lhs() = build_symbol(length_var);
    size_call.type() = size_type();
    size_call.location() = location;
    converter_.add_instruction(size_call);

    length_expr = build_symbol(length_var);
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
    iterable_expr, build_symbol(index_var), element);
  current_element = list_handler.extract_pyobject_value(
    current_element, loop_var_type, mixed_numeric);

  code_assignt loop_var_assign(build_symbol(*loop_var), current_element);
  loop_var_assign.location() = location;
  loop_body.copy_to_operands(loop_var_assign);

  // For a tuple target, unpack the element tuple held in the temp into the
  // component names (a, b) so the key/value expressions can reference them.
  if (is_tuple_target)
  {
    exprt tuple_value = build_symbol(*loop_var);
    converter_.get_tuple_handler().handle_tuple_unpacking(
      element, target, tuple_value, loop_body);
  }

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
    build_symbol(dict_sym),
    key_expr,
    value_expr,
    location,
    element,
    pair_block);
  converter_.current_block = saved_block;

  if (generator.contains("ifs") && !generator["ifs"].empty())
  {
    // Dict comprehensions may include filters such as
    // {k: v for x in xs if cond1 if cond2}
    // V.3: build the comprehension filter and-fold in IREP2. The outer guard
    // guarantees at least one clause, so no `true` sentinel is needed.
    expr2tc combined2;
    bool first = true;
    for (const auto &if_clause : generator["ifs"])
    {
      expr2tc c2;
      migrate_expr(converter_.get_expr(if_clause), c2);
      combined2 = first ? c2 : and2tc(combined2, c2);
      first = false;
    }
    exprt combined_condition = migrate_expr_back(combined2);

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

  // V.3: build index + 1 and index < length in IREP2.
  exprt increment =
    build_add(build_symbol(index_var), gen_one(size_type()), size_type());
  code_assignt index_increment(build_symbol(index_var), increment);
  index_increment.location() = location;
  loop_body.copy_to_operands(index_increment);

  exprt loop_condition = build_less_than(build_symbol(index_var), length_expr);

  codet while_stmt;
  while_stmt.set_statement("while");
  while_stmt.copy_to_operands(loop_condition, loop_body);
  while_stmt.location() = location;
  converter_.add_instruction(while_stmt);

  return build_symbol(dict_sym);
}

exprt python_dict_handler::build_range_dict_comprehension(
  const nlohmann::json &element,
  const nlohmann::json &generator,
  symbolt &dict_sym)
{
  locationt location = converter_.get_location_from_decl(element);
  const auto &target = generator["target"];
  const auto &iter = generator["iter"];
  const auto &range_args = iter["args"];
  const typet idx_type = long_long_int_type();

  // The comprehension target is the loop counter: its value IS the range
  // element, so the key/value expressions read it directly.
  std::string loop_var_name = target["id"].get<std::string>();
  symbol_id loop_var_sid = converter_.create_symbol_id();
  loop_var_sid.set_object(loop_var_name);
  symbolt loop_var_symbol = converter_.create_symbol(
    location.get_file().as_string(),
    loop_var_name,
    loop_var_sid.to_string(),
    location,
    idx_type);
  loop_var_symbol.lvalue = true;
  loop_var_symbol.file_local = true;
  loop_var_symbol.is_extern = false;
  symbolt *loop_var =
    converter_.symbol_table().move_symbol_to_context(loop_var_symbol);

  // range(stop) -> start 0; range(start, stop) -> explicit start. Step is 1
  // here (callers route explicit-step ranges to the materialisation path).
  exprt start_expr;
  exprt stop_arg;
  if (range_args.size() == 1)
  {
    start_expr = gen_zero(idx_type);
    stop_arg = converter_.get_expr(range_args[0]);
  }
  else
  {
    start_expr = converter_.get_expr(range_args[0]);
    stop_arg = converter_.get_expr(range_args[1]);
  }
  start_expr = build_typecast(start_expr, idx_type);
  stop_arg = build_typecast(stop_arg, idx_type);

  // Freeze the loop-invariant stop bound in a temporary.
  symbolt &stop_var = converter_.create_tmp_symbol(
    element, "$dictcomp_stop$", idx_type, gen_zero(idx_type));
  code_declt stop_decl(build_symbol(stop_var));
  stop_decl.location() = location;
  converter_.add_instruction(stop_decl);
  code_assignt stop_init(build_symbol(stop_var), stop_arg);
  stop_init.location() = location;
  converter_.add_instruction(stop_init);

  // loop_var = start
  code_declt loop_decl(build_symbol(*loop_var));
  loop_decl.location() = location;
  converter_.add_instruction(loop_decl);
  code_assignt loop_init(build_symbol(*loop_var), start_expr);
  loop_init.location() = location;
  converter_.add_instruction(loop_init);

  // Build the loop body with current_block redirected to it (see the matching
  // comment in get_dict_comprehension): key/value temporaries and the
  // subscript-assign side effects must land inside the loop.
  code_blockt loop_body;
  code_blockt *saved_block = converter_.current_block;
  converter_.current_block = &loop_body;
  exprt key_expr = converter_.get_expr(element["key"]);
  exprt value_expr = converter_.get_expr(element["value"]);

  code_blockt pair_block;
  handle_dict_subscript_assign(
    build_symbol(dict_sym),
    key_expr,
    value_expr,
    location,
    element,
    pair_block);
  converter_.current_block = saved_block;

  if (generator.contains("ifs") && !generator["ifs"].empty())
  {
    // V.3: build the comprehension filter and-fold in IREP2. The outer guard
    // guarantees at least one clause, so no `true` sentinel is needed.
    expr2tc combined2;
    bool first = true;
    for (const auto &if_clause : generator["ifs"])
    {
      expr2tc c2;
      migrate_expr(converter_.get_expr(if_clause), c2);
      combined2 = first ? c2 : and2tc(combined2, c2);
      first = false;
    }
    exprt combined_condition = migrate_expr_back(combined2);

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

  // loop_var = loop_var + 1
  // V.3: build loop_var + 1 and loop_var < stop in IREP2.
  exprt increment =
    build_add(build_symbol(*loop_var), gen_one(idx_type), idx_type);
  code_assignt loop_inc(build_symbol(*loop_var), increment);
  loop_inc.location() = location;
  loop_body.copy_to_operands(loop_inc);

  // while (loop_var < stop)
  exprt loop_condition =
    build_less_than(build_symbol(*loop_var), build_symbol(stop_var));

  codet while_stmt;
  while_stmt.set_statement("while");
  while_stmt.copy_to_operands(loop_condition, loop_body);
  while_stmt.location() = location;
  converter_.add_instruction(while_stmt);

  return build_symbol(dict_sym);
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
  exprt dict_target = build_typecast(target_symbol, dict_type);

  // Create keys list
  symbolt &keys_list = converter_.create_tmp_symbol(
    element, dict_name + "_keys", list_type, exprt());

  code_declt keys_decl(build_symbol(keys_list));
  keys_decl.location() = location;
  converter_.add_instruction(keys_decl);

  const symbolt *create_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_list_create");
  if (!create_func)
    throw std::runtime_error("__ESBMC_list_create not found");

  code_function_callt keys_create;
  keys_create.function() = build_symbol(*create_func);
  keys_create.lhs() = build_symbol(keys_list);
  keys_create.type() = list_type;
  keys_create.location() = location;
  converter_.add_instruction(keys_create);

  // Create values list
  symbolt &values_list = converter_.create_tmp_symbol(
    element, dict_name + "_values", list_type, exprt());

  code_declt values_decl(build_symbol(values_list));
  values_decl.location() = location;
  converter_.add_instruction(values_decl);

  code_function_callt values_create;
  values_create.function() = build_symbol(*create_func);
  values_create.lhs() = build_symbol(values_list);
  values_create.type() = list_type;
  values_create.location() = location;
  converter_.add_instruction(values_create);

  // Push initial key-value pairs
  const auto &keys = element["keys"];
  const auto &values = element["values"];

  // A `**` unpack inside a dict literal (e.g. {**a, "b": 2}) serialises the
  // unpacked source with a null key. Dict unpacking is not modelled, so reject
  // it with a clean diagnostic rather than crashing on get_expr(null) below
  // (mirrors the dict-union '|' guard in converter_binop.cpp).
  for (const auto &key_node : keys)
    if (key_node.is_null())
      throw std::runtime_error(
        "dict unpacking ({**d}) is not supported");

  python_list list_handler(converter_, element);

  // Enable the float storage path only when *every* value is statically float
  // (a homogeneous-float dict). Then the read site also knows the element type
  // is float and lowers .values()/.items() reads/comparisons as floats, so the
  // real-sorted __ESBMC_float_buf payload is read back correctly (#5501).
  // For a heterogeneous dict (e.g. {"a": int, "b": float}) the value type is
  // erased to void*, so reads compare item->value as a raw pointer; routing the
  // float into float_buf would then make v a float_buf pointer that no longer
  // equals the integer bit-pattern, breaking the comparison (github_3719_4).
  // In that case fall back to the integer bit-pattern copy.
  bool all_values_float = !values.empty();
  for (const auto &value_node : values)
  {
    // get_typet throws "Invalid type" for value shapes it cannot statically
    // classify (lambdas/function values, user-function calls, conditional
    // expressions, f-strings, comprehensions). Treat any such case — and any
    // non-float result — as "not all float": the integer/void* path is the
    // safe default, so declining the float optimization here is sound.
    // Catch (...) rather than (const std::exception &): on macOS/clang a
    // std::runtime_error thrown by get_typet escaped a base-class catch across
    // the translation-unit boundary, crashing convert() with "ERROR: Invalid
    // type" (a homogeneous-key dict with a lambda value, e.g. github_3690).
    // The catch-all matches via the unwinder without consulting type_info, so
    // it always intercepts the throw and falls back to the integer path.
    try
    {
      if (!type_handler_.get_typet(value_node).is_floatbv())
      {
        all_values_float = false;
        break;
      }
    }
    catch (...)
    {
      all_values_float = false;
      break;
    }
  }

  for (size_t i = 0; i < keys.size(); ++i)
  {
    exprt key_expr = converter_.get_expr(keys[i]);
    exprt push_key =
      list_handler.build_push_list_call(keys_list, element, key_expr);
    converter_.add_instruction(push_key);
    // Track the key's element type so that handle_index_access can resolve
    // the concrete struct type for tuple keys instead of relying solely on
    // the "tuple" annotation (which produces empty_typet as a sentinel).
    list_handler.add_type_info(
      keys_list.id.as_string(), std::string(), key_expr.type());

    exprt value_expr = converter_.get_expr(values[i]);

    // Convert lambda/function symbol to function pointer for dict storage
    if (value_expr.type().is_code() && value_expr.is_symbol())
      value_expr = build_address_of(value_expr);

    // Check if this is a nested dict that needs special pointer storage
    if (value_expr.type().is_struct() && is_dict_type(value_expr.type()))
    {
      // Nested dict: store pointer to dict (reference semantics)
      store_nested_dict_value(element, values_list, value_expr, location);
    }
    else
    {
      // Regular value: store value directly (value semantics). For a
      // homogeneous-float dict, keep the float path enabled (as for keys) so a
      // float value is copied into the real-sorted __ESBMC_float_buf and its
      // float_idx is recorded; reading it back through .values()/.items() then
      // yields the correct value instead of float_buf[0] (#5501). For a
      // heterogeneous dict, disable it so reads use the integer bit-pattern
      // copy instead of the float_buf pointer (github_3719_4).
      exprt push_value = list_handler.build_push_list_call(
        values_list, element, value_expr, all_values_float);
      converter_.add_instruction(push_value);
    }
  }

  // Assign keys and values to target dict struct members
  exprt keys_member = build_member(dict_target, "keys", list_type);
  code_assignt keys_assign(keys_member, build_symbol(keys_list));
  keys_assign.location() = location;
  converter_.add_instruction(keys_assign);

  exprt values_member = build_member(dict_target, "values", list_type);
  code_assignt values_assign(values_member, build_symbol(values_list));
  values_assign.location() = location;
  converter_.add_instruction(values_assign);

  // Record the dict symbol → internal list symbol mapping so that
  // converter_stmt can propagate list_type_map entries when it sees
  // ESBMC_keys_N = dict_sym.keys (a member expression, not a symbol copy).
  if (target_symbol.is_symbol())
  {
    const std::string &dict_id = target_symbol.identifier().as_string();
    dict_keys_list_id_[dict_id] = keys_list.id.as_string();
    dict_vals_list_id_[dict_id] = values_list.id.as_string();
  }

  return target_symbol;
}
