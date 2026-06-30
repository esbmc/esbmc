#include "python_list_internal.h"

using namespace python_expr;

symbolt &python_list::create_list()
{
  locationt location = converter_.get_location_from_decl(list_value_);
  const type_handler &type_handler = converter_.get_type_handler();

  // Create list symbol
  const typet list_type = type_handler.get_list_type();
  symbolt &list_symbol =
    converter_.create_tmp_symbol(list_value_, "$py_list$", list_type, exprt());

  // Declare list
  code_declt list_decl(build_symbol(list_symbol));
  list_decl.location() = location;
  converter_.add_instruction(list_decl);

  // Initialize list with storage array
  const symbolt *create_func_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_create");
  assert(create_func_sym);

  // Add list_create call to the block
  code_function_callt list_create_func_call;
  list_create_func_call.function() = build_symbol(*create_func_sym);
  list_create_func_call.lhs() = build_symbol(list_symbol);
  list_create_func_call.type() = list_type;
  list_create_func_call.location() = location;
  converter_.add_instruction(list_create_func_call);

  return list_symbol;
}

exprt python_list::build_symbolic_fill_list(
  const exprt &size,
  const exprt &fill_value,
  const typet &elem_type)
{
  using namespace python_list_detail;
  locationt location = converter_.get_location_from_decl(list_value_);

  symbolt &result = create_list();
  const std::string result_id = result.id.as_string();

  // i = 0
  symbolt &index_var = converter_.create_tmp_symbol(
    list_value_, "$sfill_i$", size_type(), gen_zero(size_type()));
  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  code_assignt index_init(build_symbol(index_var), gen_zero(size_type()));
  index_init.location() = location;
  converter_.add_instruction(index_init);

  // Materialise symbolic size once to avoid re-evaluating a side-effecting
  // expression on every loop iteration.
  symbolt &size_sym = converter_.create_tmp_symbol(
    list_value_, "$sfill_n$", size_type(), gen_zero(size_type()));
  code_declt size_decl(build_symbol(size_sym));
  size_decl.location() = location;
  converter_.add_instruction(size_decl);

  exprt size_as_unsigned =
    size.type().is_unsignedbv() ? size : build_typecast(size, size_type());
  code_assignt size_assign(build_symbol(size_sym), size_as_unsigned);
  size_assign.location() = location;
  converter_.add_instruction(size_assign);

  // Build loop body in current_block-redirect pattern so that
  // build_push_list_call's internal declarations land inside the loop.
  code_blockt loop_body;
  code_blockt *saved_block = converter_.current_block;
  converter_.current_block = &loop_body;

  exprt push_call = build_push_list_call(result, list_value_, fill_value);

  converter_.current_block = saved_block;
  loop_body.copy_to_operands(push_call);

  exprt increment =
    build_add(build_symbol(index_var), gen_one(size_type()), size_type());
  code_assignt index_increment(build_symbol(index_var), increment);
  index_increment.location() = location;
  loop_body.copy_to_operands(index_increment);

  exprt loop_cond =
    build_less_than(build_symbol(index_var), build_symbol(size_sym));

  codet while_stmt;
  while_stmt.set_statement("while");
  while_stmt.copy_to_operands(loop_cond, loop_body);
  while_stmt.location() = location;
  converter_.add_instruction(while_stmt);

  add_type_info_entry(result_id, "", elem_type);

  return build_symbol(result);
}

exprt python_list::get()
{
  symbolt &list_symbol = create_list();

  const std::string &list_id = list_symbol.id.as_string();
  locationt location = converter_.get_location_from_decl(list_value_);

  auto materialize_list_elem = [&](const exprt &elem) -> exprt {
    if (elem.is_symbol())
      return elem;

    // A list element that is itself a function call (e.g. ``[nd(), nd()]``)
    // comes back from get_expr as a code_function_callt (a code statement).
    // Assigning that directly to the temp produces a malformed assignment
    // whose RHS is a call statement; it leaks into the SSA and crashes the
    // SMT backend on a null operand (issue #4699). Normalise it to a
    // side-effect call expression so create_tmp_symbol/code_assignt lower it
    // to a proper function call, exactly as a statement-level ``x = nd()``.
    const exprt value_elem = to_value_expr(elem, converter_.name_space());

    symbolt &tmp = converter_.create_tmp_symbol(
      list_value_, "$list_elem$", value_elem.type(), value_elem);
    code_declt decl(build_symbol(tmp));
    decl.location() = location;
    converter_.add_instruction(decl);

    code_assignt assign(build_symbol(tmp), value_elem);
    assign.location() = location;
    converter_.add_instruction(assign);
    return build_symbol(tmp);
  };

  // Convert every element once, recording whether the literal mixes integer and
  // floating-point values. Python promotes such a list to a homogeneous float
  // list (e.g. [4.0, 3] is [4.0, 3.0]); storing the ints unconverted would leave
  // their float_buf slot unset and read back as a stale float (issue #5156).
  std::vector<exprt> elems;
  elems.reserve(list_value_["elts"].size());
  bool has_int = false, has_float = false;
  for (auto &e : list_value_["elts"])
  {
    // Clear current_lhs so that constructor calls inside list elements
    // (e.g. [A(), B()]) create their own self temp variable instead of
    // inheriting the outer assignment target as self.
    exprt *saved_lhs = converter_.current_lhs;
    converter_.current_lhs = nullptr;
    elems.push_back(converter_.get_expr(e));
    converter_.current_lhs = saved_lhs;

    const typet &t = elems.back().type();
    if (t.is_floatbv())
      has_float = true;
    else if (t.is_signedbv() || t.is_unsignedbv() || t.is_bool())
      has_int = true; // bool is int-like in Python (True == 1)
  }

  const bool promote_ints = has_int && has_float;

  for (exprt &elem : elems)
  {
    // Promote integer (and bool) elements to double in a mixed int/float
    // literal, matching the list[float] annotation Python infers for the mix.
    const typet &t = elem.type();
    if (promote_ints && (t.is_signedbv() || t.is_unsignedbv() || t.is_bool()))
      elem = build_typecast(elem, double_type());

    exprt map_elem = materialize_list_elem(elem);

    exprt list_push_func_call =
      build_push_list_call(list_symbol, list_value_, map_elem);
    converter_.add_instruction(list_push_func_call);
    list_type_map[list_id].push_back(
      std::make_pair(map_elem.identifier().as_string(), map_elem.type()));
  }

  return build_symbol(list_symbol);
}

exprt python_list::build_list_from_exprs(const std::vector<exprt> &elems)
{
  symbolt &list_symbol = create_list();
  const std::string &list_id = list_symbol.id.as_string();

  for (const exprt &elem : elems)
  {
    // build_push_list_call materializes the value into a temp symbol, derives
    // the element type-id from its type, and copies it into the list storage.
    exprt push_call = build_push_list_call(list_symbol, list_value_, elem);
    converter_.add_instruction(push_call);
    list_type_map[list_id].push_back(
      std::make_pair(std::string(), elem.type()));
  }

  return build_symbol(list_symbol);
}

exprt python_list::create_vla(
  const nlohmann::json &element,
  const exprt &count,
  const std::vector<exprt> &list_elems)
{
  locationt location = converter_.get_location_from_decl(element);

  // Fresh result list: a literal source (e.g. [0]) already holds its initial
  // element, so pushing onto it would yield count + 1 elements.
  symbolt &result = create_list();

  // Materialise the count (which may be a compound expression such as m + 1)
  // into an int symbol to use as the loop bound.
  exprt bound_value =
    (count.type() == int_type()) ? count : build_typecast(count, int_type());
  symbolt &bound = converter_.create_tmp_symbol(
    element, "$list_rep_count$", int_type(), exprt());
  code_declt bound_decl(build_symbol(bound));
  bound_decl.location() = location;
  converter_.add_instruction(bound_decl);
  code_assignt bound_assign(build_symbol(bound), bound_value);
  bound_assign.location() = location;
  converter_.add_instruction(bound_assign);

  // counter = 0
  symbolt &counter = converter_.create_tmp_symbol(
    element, "counter", int_type(), gen_zero(int_type()));
  code_assignt counter_code(build_symbol(counter), gen_zero(int_type()));
  counter_code.location() = location;
  converter_.add_instruction(counter_code);

  // while (counter < bound) { push each elem in order; counter += 1; }
  // (counter and bound are both int_type — same width)
  exprt cond = build_less_than(build_symbol(counter), build_symbol(bound));

  code_blockt then;
  for (const auto &list_elem : list_elems)
    then.copy_to_operands(build_push_list_call(result, element, list_elem));

  exprt incr =
    build_add(build_symbol(counter), gen_one(int_type()), int_type());
  code_assignt update(build_symbol(counter), incr);
  then.copy_to_operands(update);

  codet while_cod;
  while_cod.set_statement("while");
  while_cod.copy_to_operands(cond, then);
  converter_.add_instruction(while_cod);

  // Record one type-map entry per source element for index-based type lookups.
  auto &result_types = list_type_map[result.id.as_string()];
  for (const auto &list_elem : list_elems)
    result_types.push_back(std::make_pair(std::string(), list_elem.type()));

  return build_symbol(result);
}

exprt python_list::build_list_from_range(
  python_converter &converter,
  const nlohmann::json &range_args,
  const nlohmann::json &element)
{
  // Validate argument count
  if (range_args.empty() || range_args.size() > 3)
    throw std::runtime_error("range() takes 1 to 3 arguments");

  // Extract constant integer, handling UnaryOp for negative numbers
  auto extract_constant =
    [&](const nlohmann::json &arg) -> std::optional<long long> {
    exprt expr = converter.get_expr(arg);

    if (expr.is_constant())
      return binary2integer(expr.value().as_string(), expr.type().is_signedbv())
        .to_int64();

    // Handle UnaryOp (e.g., -1)
    if (expr.id() == "unary-" && expr.operands().size() == 1)
    {
      const exprt &operand = expr.operands()[0];
      if (operand.is_constant())
      {
        long long val =
          binary2integer(
            operand.value().as_string(), operand.type().is_signedbv())
            .to_int64();
        return -val;
      }
    }

    return std::nullopt;
  };

  // Extract all arguments
  std::optional<long long> arg0 = extract_constant(range_args[0]);
  std::optional<long long> arg1;
  std::optional<long long> arg2;

  if (range_args.size() > 1)
    arg1 = extract_constant(range_args[1]);
  if (range_args.size() > 2)
    arg2 = extract_constant(range_args[2]);

  // Check if all required arguments are constant
  const bool all_constant = arg0.has_value() &&
                            (range_args.size() <= 1 || arg1.has_value()) &&
                            (range_args.size() <= 2 || arg2.has_value());

  // Handle symbolic (non-constant) case
  if (!all_constant)
  {
    return handle_symbolic_range(converter, range_args, element);
  }

  // All arguments are constant
  return build_concrete_range(converter, range_args, element, arg0, arg1, arg2);
}

exprt python_list::build_list_from_tuple(
  python_converter &converter,
  const exprt &tuple_expr,
  const nlohmann::json &element)
{
  python_list helper(converter, element);
  const typet &tuple_type = converter.name_space().follow(tuple_expr.type());

  std::vector<exprt> components;
  for (const auto &comp : to_struct_type(tuple_type).components())
    components.push_back(
      build_member(tuple_expr, comp.get_name(), comp.type()));

  return helper.build_list_from_exprs(components);
}

exprt python_list::handle_symbolic_range(
  python_converter &converter,
  const nlohmann::json &range_args,
  const nlohmann::json &element)
{
  if (range_args.size() == 1)
  {
    // range(n) case: create list with symbolic size n
    exprt n_expr = converter.get_expr(range_args[0]);

    // Create an empty list using existing create_list infrastructure
    nlohmann::json list_node;
    list_node["_type"] = "List";
    list_node["elts"] = nlohmann::json::array();
    converter.copy_location_fields_from_decl(element, list_node);

    python_list temp_list(converter, list_node);
    exprt list_expr = temp_list.get();

    // Set symbolic size using helper method
    set_list_symbolic_size(converter, list_expr, n_expr, element);

    return list_expr;
  }

  // For multi-argument symbolic ranges, return empty list
  nlohmann::json empty_list_node;
  empty_list_node["_type"] = "List";
  empty_list_node["elts"] = nlohmann::json::array();
  converter.copy_location_fields_from_decl(element, empty_list_node);
  python_list list(converter, empty_list_node);
  return list.get();
}

void python_list::set_list_symbolic_size(
  python_converter &converter,
  exprt &list_expr,
  const exprt &size_expr,
  const nlohmann::json &element)
{
  if (!list_expr.type().is_pointer())
    return;

  typet pointee_type = list_expr.type().subtype();

  // Follow symbol types to get actual struct type
  if (pointee_type.is_symbol())
    pointee_type = converter.ns.follow(pointee_type);

  if (!pointee_type.is_struct())
    return;

  const struct_typet &struct_type = to_struct_type(pointee_type);

  // Find and update the size member
  for (const auto &comp : struct_type.components())
  {
    if (comp.get_name() == "size")
    {
      // Create assignment: list->size = n. pointee_type is a resolved struct
      // (followed and checked above), so build *(list).size in IREP2 (V.3).
      exprt deref = build_dereference(list_expr, pointee_type);
      exprt size_member = build_member(deref, comp.get_name(), comp.type());
      exprt size_value = build_typecast(size_expr, comp.type());
      code_assignt size_assignment(size_member, size_value);

      size_assignment.location() = element.contains("lineno")
                                     ? converter.get_location_from_decl(element)
                                     : locationt();

      converter.current_block->operands().push_back(size_assignment);
      break;
    }
  }
}

exprt python_list::build_concrete_range(
  python_converter &converter,
  const nlohmann::json &range_args,
  const nlohmann::json &element,
  const std::optional<long long> &arg0,
  const std::optional<long long> &arg1,
  const std::optional<long long> &arg2)
{
  // Determine start, stop, step based on argument count
  long long start, stop, step;

  switch (range_args.size())
  {
  case 1:
    start = 0;
    stop = arg0.value();
    step = 1;
    break;

  case 2:
    start = arg0.value();
    stop = arg1.value();
    step = 1;
    break;

  case 3:
    start = arg0.value();
    stop = arg1.value();
    step = arg2.value();
    break;

  default:
    throw std::runtime_error("Invalid range argument count");
  }

  // Validate step
  if (step == 0)
    throw std::runtime_error("range() step argument must not be zero");

  // Calculate and validate range size
  long long range_size;
  if (step > 0)
    range_size = std::max(0LL, (stop - start + step - 1) / step);
  else
    range_size = std::max(0LL, (stop - start + step + 1) / step);

  if (range_size > kMaxSequenceExpansion)
  {
    throw std::runtime_error(
      "range() size too large for expansion: " + std::to_string(range_size) +
      " elements (max: " + std::to_string(kMaxSequenceExpansion) + ")");
  }

  // Build the list of elements
  nlohmann::json list_node;
  list_node["_type"] = "List";
  list_node["elts"] = nlohmann::json::array();

  // Generate list elements
  for (long long i = start; (step > 0) ? (i < stop) : (i > stop); i += step)
  {
    nlohmann::json elem;
    elem["_type"] = "Constant";
    elem["value"] = i;
    elem["kind"] = nullptr;
    converter.copy_location_fields_from_decl(element, elem);
    list_node["elts"].push_back(elem);
  }

  converter.copy_location_fields_from_decl(element, list_node);

  python_list list(converter, list_node);
  return list.get();
}
