#include "python_list_internal.h"

using namespace python_expr;
using namespace python_list_detail;

exprt python_list::handle_comprehension(const nlohmann::json &element)
{
  if (!element.contains("generators") || element["generators"].empty())
  {
    throw std::runtime_error(
      "Comprehension expression missing generators clause");
  }

  const auto &generator = element["generators"][0];
  const auto &elt = element["elt"];
  const auto &target = generator["target"];
  const auto &iter = generator["iter"];

  locationt location = converter_.get_location_from_decl(element);
  typet list_type = converter_.get_type_handler().get_list_type();

  // 1. Create result list
  symbolt &result_list = create_list();
  const std::string &result_list_id = result_list.id.as_string();

  // 2. Get iterable expression
  exprt iterable_expr = converter_.get_expr(iter);

  // 2a. Materialize function calls that return lists
  if (
    iterable_expr.is_code() &&
    iterable_expr.get("statement") == "function_call")
  {
    const code_function_callt &call =
      to_code_function_call(to_code(iterable_expr));

    if (call.type() == list_type)
    {
      // Create temporary variable for the list
      symbolt &tmp_var_symbol = converter_.create_tmp_symbol(
        element, "$iter_temp$", list_type, gen_zero(list_type));

      // Declare the temporary
      code_declt tmp_var_decl(build_symbol(tmp_var_symbol));
      tmp_var_decl.location() = location;
      converter_.add_instruction(tmp_var_decl);

      // Create function call with temp as LHS
      code_function_callt new_call;
      new_call.function() = call.function();
      new_call.arguments() = call.arguments();
      new_call.lhs() = build_symbol(tmp_var_symbol);
      new_call.type() = list_type;
      new_call.location() = location;
      converter_.add_instruction(new_call);

      // Use the temp variable as the iterable
      iterable_expr = build_symbol(tmp_var_symbol);
    }
  }
  // Check for empty list early
  else if (iterable_expr.type() == list_type && iterable_expr.is_symbol())
  {
    const std::string &list_id = iterable_expr.identifier().as_string();
    auto type_map_it = list_type_map.find(list_id);
    if (type_map_it == list_type_map.end() || type_map_it->second.empty())
      return build_symbol(result_list);
  }

  // 3. Create loop variable
  std::string loop_var_name = target["id"].get<std::string>();
  symbol_id loop_var_sid = converter_.create_symbol_id();
  loop_var_sid.set_object(loop_var_name);

  // Infer loop variable type from iterable
  typet loop_var_type;
  if (iterable_expr.type() == list_type)
  {
    // For list iteration, we need to determine the element type from type_map
    loop_var_type = iterable_expr.type(); // default

    if (iterable_expr.is_symbol())
    {
      const std::string &list_id = iterable_expr.identifier().as_string();
      auto type_map_it = list_type_map.find(list_id);
      if (type_map_it != list_type_map.end() && !type_map_it->second.empty())
      {
        // Use the actual element type from type_map
        loop_var_type = type_map_it->second[0].second;
      }
    }
  }
  else if (iterable_expr.type().is_array())
    loop_var_type = iterable_expr.type().subtype();
  else if (iterable_expr.type().is_pointer())
    loop_var_type = iterable_expr.type().subtype();
  else
    loop_var_type = any_type();

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

  // 4. Create index variable
  symbolt &index_var = converter_.create_tmp_symbol(
    element, "comp_i", size_type(), gen_zero(size_type()));

  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  // Initialize index to 0
  code_assignt index_init(build_symbol(index_var), gen_zero(size_type()));
  index_init.location() = location;
  converter_.add_instruction(index_init);

  // 5. Get length of iterable
  exprt length_expr;
  if (iterable_expr.type() == list_type)
  {
    const symbolt *size_func =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
    if (!size_func)
      throw std::runtime_error("__ESBMC_list_size not found in symbol table");

    symbolt &length_var = converter_.create_tmp_symbol(
      element, "comp_len", size_type(), gen_zero(size_type()));

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
  else if (iterable_expr.type().is_array())
  {
    const array_typet &arr_type = to_array_type(iterable_expr.type());
    length_expr = arr_type.size();
  }
  else if (iterable_expr.type().is_pointer())
  {
    const symbolt *strlen_func =
      converter_.symbol_table().find_symbol("c:@F@strlen");
    if (strlen_func)
    {
      symbolt &length_var = converter_.create_tmp_symbol(
        element, "comp_len", size_type(), gen_zero(size_type()));

      code_declt length_decl(build_symbol(length_var));
      length_decl.location() = location;
      converter_.add_instruction(length_decl);

      code_function_callt strlen_call;
      strlen_call.function() = build_symbol(*strlen_func);
      strlen_call.arguments().push_back(iterable_expr);
      strlen_call.lhs() = build_symbol(length_var);
      strlen_call.type() = size_type();
      strlen_call.location() = location;
      converter_.add_instruction(strlen_call);

      length_expr = build_symbol(length_var);
    }
    else
    {
      throw std::runtime_error("strlen not found for string iteration");
    }
  }
  else
  {
    throw std::runtime_error(
      "Unsupported iterable type in comprehension: " +
      iterable_expr.type().id_string());
  }

  // 6. Build while loop body
  code_blockt loop_body;

  // Get current element: loop_var = iterable[i]
  exprt current_element;
  if (iterable_expr.type() == list_type)
  {
    // For lists, determine the actual element type from type map
    typet actual_elem_type = loop_var_type;

    // Try to get the actual element type from the list's type map
    if (iterable_expr.is_symbol())
    {
      const std::string &list_id = iterable_expr.identifier().as_string();
      auto type_map_it = list_type_map.find(list_id);
      if (type_map_it != list_type_map.end() && !type_map_it->second.empty())
      {
        // Get the element type from the first entry
        actual_elem_type = type_map_it->second[0].second;
      }
    }

    // Use list_at and extract_pyobject_value for consistent handling
    current_element =
      build_list_at_call(iterable_expr, build_symbol(index_var), element);
    current_element = extract_pyobject_value(current_element, actual_elem_type);
  }
  else if (iterable_expr.type().is_array() || iterable_expr.type().is_pointer())
  {
    // For arrays/strings, use direct indexing
    exprt array_index =
      build_index(iterable_expr, build_symbol(index_var), loop_var_type);
    current_element = array_index;
  }
  else
  {
    throw std::runtime_error(
      "Cannot index into type: " + iterable_expr.type().id_string());
  }

  code_assignt loop_var_assign(build_symbol(*loop_var), current_element);
  loop_var_assign.location() = location;
  loop_body.copy_to_operands(loop_var_assign);

  // 7. Handle filter conditions (if present)
  code_blockt conditional_block;

  // 8. Evaluate element expression and append to result
  // Switch context to loop body for all operations
  code_blockt *saved_block = converter_.current_block;
  converter_.current_block = &loop_body;

  // Evaluate element expression - temporaries go to loop_body
  exprt element_expr = converter_.get_expr(elt);

  // Build push call - temporaries also go to loop_body
  exprt push_call = build_push_list_call(result_list, element, element_expr);
  loop_body.copy_to_operands(push_call);

  // Restore context
  converter_.current_block = saved_block;

  // Update type map
  list_type_map[result_list_id].push_back(
    std::make_pair(element_expr.identifier().as_string(), element_expr.type()));

  // If we had filter conditions, wrap append in if statement
  if (generator.contains("ifs") && !generator["ifs"].empty())
  {
    // V.3: build the comprehension filter and-fold in IREP2. The outer guard
    // guarantees at least one clause, so no `true` sentinel is needed.
    expr2tc filt_combined;
    bool filt_first = true;
    for (const auto &if_clause : generator["ifs"])
    {
      expr2tc filt_c;
      migrate_expr(converter_.get_expr(if_clause), filt_c);
      filt_combined = filt_first ? filt_c : and2tc(filt_combined, filt_c);
      filt_first = false;
    }
    exprt combined_condition = migrate_expr_back(filt_combined);

    codet if_stmt;
    if_stmt.set_statement("ifthenelse");
    if_stmt.copy_to_operands(combined_condition, conditional_block);
    if_stmt.location() = location;
    loop_body.copy_to_operands(if_stmt);
  }

  // 9. Increment index: i = i + 1 (V.3: built in IREP2).
  exprt increment =
    build_add(build_symbol(index_var), gen_one(size_type()), size_type());
  code_assignt index_increment(build_symbol(index_var), increment);
  index_increment.location() = location;
  loop_body.copy_to_operands(index_increment);

  // 10. Create while loop: while (i < length)
  exprt loop_condition("<", bool_type());
  loop_condition.copy_to_operands(build_symbol(index_var), length_expr);

  codet while_stmt;
  while_stmt.set_statement("while");
  while_stmt.copy_to_operands(loop_condition, loop_body);
  while_stmt.location() = location;
  converter_.add_instruction(while_stmt);

  return build_symbol(result_list);
}

void python_list::handle_list_var_unpacking(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  const exprt &list_expr,
  codet &target_block)
{
  const auto &targets = target["elts"];
  const locationt loc = converter_.get_location_from_decl(ast_node);

  // Find starred target index (-1 if none)
  int star_idx = -1;
  for (size_t i = 0; i < targets.size(); i++)
  {
    if (targets[i]["_type"] == "Starred")
    {
      star_idx = static_cast<int>(i);
      break;
    }
  }

  const size_t before_star =
    (star_idx >= 0) ? static_cast<size_t>(star_idx) : targets.size();
  const size_t after_star =
    (star_idx >= 0) ? targets.size() - static_cast<size_t>(star_idx) - 1 : 0;

  // Get element type from list_type_map or from the variable's annotation
  typet elem_type;
  if (list_expr.is_symbol())
  {
    const std::string &list_id = list_expr.identifier().as_string();
    auto it = list_type_map.find(list_id);
    if (it != list_type_map.end() && !it->second.empty())
      elem_type = it->second[0].second;
  }
  if (elem_type == typet() && ast_node["value"].contains("id"))
  {
    const std::string &var_name = ast_node["value"]["id"].get<std::string>();
    nlohmann::json decl = json_utils::find_var_decl(
      var_name, converter_.current_function_name(), converter_.ast());
    elem_type =
      get_elem_type_from_annotation(decl, converter_.get_type_handler());
  }

  // Subscript-chain fallback: e.g. `w, v = items[i-1]` where the RHS is a
  // Subscript whose base Name carries a list[list[T]]-style annotation.
  // Walk down the chain and peel one extra annotation layer (the unpacked
  // tuple/list is the element of the innermost subscript).
  if (
    elem_type == typet() && ast_node["value"].is_object() &&
    ast_node["value"].contains("_type") &&
    ast_node["value"]["_type"] == "Subscript")
  {
    size_t subscript_depth = 0;
    const nlohmann::json *cur = &ast_node["value"];
    while (cur->is_object() && cur->contains("value") &&
           (*cur)["value"].is_object() && (*cur)["value"].contains("_type") &&
           (*cur)["value"]["_type"] == "Subscript")
    {
      ++subscript_depth;
      cur = &(*cur)["value"];
    }
    if (
      cur->is_object() && cur->contains("value") &&
      (*cur)["value"].is_object() && (*cur)["value"].contains("id") &&
      (*cur)["value"]["id"].is_string())
    {
      const std::string base_name = (*cur)["value"]["id"].get<std::string>();
      nlohmann::json base_decl = json_utils::find_var_decl(
        base_name, converter_.current_function_name(), converter_.ast());
      if (!base_decl.is_null() && base_decl.contains("annotation"))
      {
        nlohmann::json drilled = base_decl["annotation"];
        // Peel one layer per Subscript node plus one more for the unpacked
        // element itself.
        for (size_t k = 0; k <= subscript_depth; ++k)
        {
          if (
            !drilled.is_object() || !drilled.contains("_type") ||
            drilled["_type"] != "Subscript" || !drilled.contains("slice"))
          {
            drilled = nlohmann::json();
            break;
          }
          drilled = drilled["slice"];
        }
        if (!drilled.is_null())
        {
          nlohmann::json synth;
          synth["annotation"] = drilled;
          elem_type =
            get_elem_type_from_annotation(synth, converter_.get_type_handler());
        }
      }
    }
  }

  // Final fallback: treat elements as Any rather than aborting the conversion.
  // Mirrors the subscript-read path which falls back to any_type() when the
  // element type cannot be inferred (see `python_list::get_expr`).
  if (elem_type == typet())
    elem_type = any_type();

  // Helper: find or create a variable symbol and assign an expression to it
  auto assign_to_target = [&](
                            const nlohmann::json &tgt_node, const exprt &val) {
    if (tgt_node["_type"] != "Name")
      throw std::runtime_error(
        "List unpacking only supports simple names, not " +
        tgt_node["_type"].get<std::string>());

    const std::string var_name = tgt_node["id"].get<std::string>();
    symbol_id var_sid = converter_.create_symbol_id();
    var_sid.set_object(var_name);
    symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());
    if (!var_symbol)
    {
      symbolt new_symbol = converter_.create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        val.type());
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = converter_.symbol_table().move_symbol_to_context(new_symbol);
    }
    code_assignt assign(build_symbol(*var_symbol), val);
    assign.location() = loc;
    target_block.copy_to_operands(assign);
  };

  // Assign targets before the star using concrete indices
  for (size_t i = 0; i < before_star; i++)
  {
    exprt idx = from_integer(i, size_type());
    exprt list_at = build_list_at_call(list_expr, idx, list_value_);
    exprt val = extract_pyobject_value(list_at, elem_type);
    assign_to_target(targets[i], val);
  }

  // Handle starred target: collect remaining elements into a new list
  if (star_idx >= 0)
  {
    const auto &star_value = targets[static_cast<size_t>(star_idx)]["value"];
    if (star_value["_type"] != "Name")
      throw std::runtime_error(
        "Starred unpacking only supports simple names, not " +
        star_value["_type"].get<std::string>());

    // Create new list for the starred variable
    symbolt &star_list = create_list();

    // Compute source list size once
    const symbolt *size_func =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
    assert(size_func);

    symbolt &size_var = converter_.create_tmp_symbol(
      list_value_, "$unpack_size$", size_type(), gen_zero(size_type()));
    code_declt size_decl(build_symbol(size_var));
    target_block.copy_to_operands(size_decl);

    code_function_callt size_call;
    size_call.function() = build_symbol(*size_func);
    size_call.arguments().push_back(
      list_expr.type().is_pointer() ? list_expr : build_address_of(list_expr));
    size_call.lhs() = build_symbol(size_var);
    size_call.type() = size_type();
    size_call.location() = loc;
    target_block.copy_to_operands(size_call);

    // upper = size - after_star. size_var and the literal are both size_type
    // (synthetic), so build the subtraction in IREP2 (V.3).
    exprt upper_expr;
    if (after_star > 0)
      upper_expr = build_sub(
        build_symbol(size_var),
        from_integer(after_star, size_type()),
        size_type());
    else
      upper_expr = build_symbol(size_var);

    // Loop: for loop_idx = before_star; loop_idx < upper; loop_idx++
    symbolt &loop_idx = converter_.create_tmp_symbol(
      list_value_, "$i$", size_type(), gen_zero(size_type()));
    code_assignt idx_init(
      build_symbol(loop_idx), from_integer(before_star, size_type()));
    target_block.copy_to_operands(idx_init);

    // loop_idx is size_type; upper_expr is a size_type symbol/literal, so both
    // operands share width.
    exprt loop_cond = build_less_than(build_symbol(loop_idx), upper_expr);

    code_blockt loop_body;

    // tmp_at = __ESBMC_list_at(list_expr, loop_idx)
    const exprt at_call =
      build_list_at_call(list_expr, build_symbol(loop_idx), list_value_);
    symbolt &tmp_at = converter_.create_tmp_symbol(
      list_value_,
      "tmp_unpack_at",
      pointer_typet(converter_.get_type_handler().get_list_element_type()),
      exprt());
    code_declt tmp_at_decl(build_symbol(tmp_at));
    tmp_at_decl.copy_to_operands(at_call);
    loop_body.copy_to_operands(tmp_at_decl);

    // __ESBMC_list_push_shallow(star_list, tmp_at): preserve element value
    // pointers so nested lists survive the unpack copy uncorrupted (#5102).
    const symbolt *push_obj_func =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push_shallow");
    assert(push_obj_func);

    // Nested-list elements keep their inner pointer; scalars are byte-copied,
    // so the unpack copy does not corrupt nested lists (#5102).
    constant_exprt star_list_type_id(size_type());
    star_list_type_id.set_value(integer2binary(
      std::hash<std::string>{}(converter_.get_type_handler().type_to_string(
        converter_.get_type_handler().get_list_type())),
      config.ansi_c.address_width));
    exprt push_call = build_call_expr(
      *push_obj_func,
      bool_type(),
      {build_symbol(star_list), build_symbol(tmp_at), star_list_type_id});
    push_call.location() = loc;
    loop_body.copy_to_operands(
      converter_.convert_expression_to_code(push_call));

    // loop_idx++
    exprt inc =
      build_add(build_symbol(loop_idx), gen_one(size_type()), size_type());
    code_assignt inc_assign(build_symbol(loop_idx), inc);
    loop_body.copy_to_operands(inc_assign);

    codet while_loop;
    while_loop.set_statement("while");
    while_loop.copy_to_operands(loop_cond, loop_body);
    target_block.copy_to_operands(while_loop);

    // Record element type for the starred list
    python_list::add_type_info_entry(star_list.id.as_string(), "", elem_type);

    // Assign the new list to the starred variable and register its type info
    assign_to_target(star_value, build_symbol(star_list));
    // Also register the starred variable's own symbol id so subsequent list
    // method calls (e.g., rest.append(x)) can look up the element type.
    {
      const std::string var_name = star_value["id"].get<std::string>();
      symbol_id var_sid = converter_.create_symbol_id();
      var_sid.set_object(var_name);
      python_list::add_type_info_entry(var_sid.to_string(), "", elem_type);
    }

    // Assign targets after the star using size_var
    for (size_t j = 0; j < after_star; j++)
    {
      size_t target_idx = static_cast<size_t>(star_idx) + 1 + j;
      // index = size - (after_star - j). size_var and the literal are both
      // size_type (synthetic), so build the subtraction in IREP2 (V.3).
      exprt after_idx = build_sub(
        build_symbol(size_var),
        from_integer(after_star - j, size_type()),
        size_type());
      exprt list_at = build_list_at_call(list_expr, after_idx, list_value_);
      exprt val = extract_pyobject_value(list_at, elem_type);
      assign_to_target(targets[target_idx], val);
    }
  }
}
