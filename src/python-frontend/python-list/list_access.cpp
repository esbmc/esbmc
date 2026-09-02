#include "python_list_internal.h"
#include <python-frontend/python-dict/python_dict_handler.h>
#include <python-frontend/numpy/ndarray_descriptor.h>
#include <algorithm>
#include <optional>

using namespace python_expr;
using namespace python_list_detail;

namespace
{
// Structural equality of two AST JSON nodes, ignoring source-location keys
// (lineno/col_offset/...). Two textually distinct occurrences of the same
// expression — e.g. the `l[i+1:]` on each side of `l[i+1:] = reversed(l[i+1:])`
// — differ only in their location fields, so a raw `==` would wrongly report
// them as different. Used to prove the read-slice and write-slice are the same
// before collapsing the reverse-in-place idiom.
bool ast_equal_ignoring_location(
  const nlohmann::json &a,
  const nlohmann::json &b)
{
  static constexpr const char *loc_keys[] = {
    "lineno", "col_offset", "end_lineno", "end_col_offset"};
  auto is_loc_key = [&](const std::string &k) {
    for (const char *lk : loc_keys)
      if (k == lk)
        return true;
    return false;
  };

  if (a.type() != b.type())
    return false;

  if (a.is_object())
  {
    // Compare the non-location keys of both objects symmetrically.
    for (auto it = a.begin(); it != a.end(); ++it)
    {
      if (is_loc_key(it.key()))
        continue;
      if (
        !b.contains(it.key()) ||
        !ast_equal_ignoring_location(it.value(), b[it.key()]))
        return false;
    }
    for (auto it = b.begin(); it != b.end(); ++it)
    {
      if (is_loc_key(it.key()))
        continue;
      if (!a.contains(it.key()))
        return false;
    }
    return true;
  }

  if (a.is_array())
  {
    if (a.size() != b.size())
      return false;
    for (size_t i = 0; i < a.size(); ++i)
      if (!ast_equal_ignoring_location(a[i], b[i]))
        return false;
    return true;
  }

  return a == b;
}
} // namespace

exprt python_list::build_list_at_call(
  const exprt &list,
  const exprt &index,
  const nlohmann::json &element)
{
  pointer_typet obj_type(converter_.get_type_handler().get_list_element_type());

  const symbolt *list_at_func_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_at");
  assert(list_at_func_sym);

  locationt location = converter_.get_location_from_decl(element);

  // An unsigned index (size_type, used throughout for loop counters: list
  // comprehensions, starred unpacking, boolean-mask filtering, ...) can never
  // be negative, so the negative-index normalization below is provably dead
  // for it. Skipping it avoids emitting a redundant __ESBMC_list_size call
  // and an unreachable IndexError guard on every loop iteration that indexes
  // a list this way -- each one otherwise becomes its own extra symbolic-
  // execution step purely to prove what the index's type already guarantees.
  const bool index_may_be_negative = !index.type().is_unsignedbv();

  // A discarded type probe only reads this call's type, and the real RHS
  // build re-emits the whole check; normalizing here would emit a second
  // __ESBMC_list_size call and IndexError guard per subscript assignment.
  if (!index_may_be_negative || converter_.in_rhs_type_probe_)
  {
    exprt index_as_size = build_typecast(index, size_type());
    exprt list_at_call = build_call_expr(
      *list_at_func_sym,
      obj_type,
      {list.type().is_pointer() ? list : build_address_of(list),
       index_as_size});
    list_at_call.location() = location;
    return list_at_call;
  }

  // Get list size
  const symbolt *size_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  assert(size_func && "list_size function not found");

  symbolt &size_var = converter_.create_tmp_symbol(
    element, "$list_size$", size_type(), gen_zero(size_type()));
  code_declt size_decl(build_symbol(size_var));
  size_decl.location() = location;
  converter_.add_instruction(size_decl);

  code_function_callt size_call;
  size_call.function() = build_symbol(*size_func);
  size_call.arguments().push_back(
    list.type().is_pointer() ? list : build_address_of(list));
  size_call.lhs() = build_symbol(size_var);
  size_call.type() = size_type();
  size_call.location() = location;
  converter_.add_instruction(size_call);

  // Convert index to size_t for comparison and arithmetic
  exprt index_as_size = build_typecast(index, size_type());

  // Create: actual_index = (index < 0) ? (size + index) : index
  // (V.3: build the negative-index normalization in IREP2.)
  const type2tc size_t2 = migrate_type(size_type());
  expr2tc index2, index_as_size2, size_var2;
  migrate_expr(index, index2);
  migrate_expr(index_as_size, index_as_size2);
  migrate_expr(build_symbol(size_var), size_var2);

  expr2tc is_negative = lessthan2tc(index2, gen_zero(index2->type));
  // For negative: size + index (since index is negative, this is
  // size - abs(index)).
  expr2tc positive_index = add2tc(size_t2, size_var2, index_as_size2);
  // Choose between positive conversion or original.
  expr2tc converted_index2 =
    if2tc(size_t2, is_negative, positive_index, index_as_size2);
  exprt converted_index = migrate_expr_back(converted_index2);

  if (!config.options.get_bool_option("no-bounds-check"))
  {
    // Runtime guard for any out-of-bounds normalized index -- both an
    // underflowed negative index (e.g. `[][-1]`) and a too-large non-negative
    // index. Raising a catchable IndexError here (rather than letting the read
    // trip __ESBMC_list_at's hard assert) lets `try/except IndexError` observe
    // the error, matching CPython and the negative-index path. This path is
    // only reached for signed indices; provably-non-negative unsigned indices
    // (loop counters) early-return above without a bounds call, so the hot path
    // is unaffected. oob = converted_index >= size (V.3: IREP2).
    expr2tc oob = greaterthanequal2tc(converted_index2, size_var2);

    exprt raise = converter_.get_exception_handler().gen_exception_raise(
      "IndexError", "list index out of range");
    codet throw_code("expression");
    throw_code.operands().push_back(raise);

    code_ifthenelset guard;
    guard.cond() = migrate_expr_back(oob);
    guard.then_case() = throw_code;
    guard.location() = location;
    guard.location().property("skipped");
    converter_.add_instruction(guard);
  }

  // Use the converted expression directly in the call
  exprt list_at_call = build_call_expr(
    *list_at_func_sym,
    obj_type,
    {list.type().is_pointer() ? list : build_address_of(list),
     converted_index});
  list_at_call.location() = location;

  return list_at_call;
}

exprt python_list::index(const exprt &array, const nlohmann::json &slice_node)
{
  if (slice_node["_type"] == "Slice") // arr[lower:upper]
  {
    return handle_range_slice(array, slice_node);
  }
  else
  {
    return handle_index_access(array, slice_node);
  }
  return exprt();
}

exprt python_list::build_bool_mask_index(
  const exprt &array,
  const exprt &mask,
  const nlohmann::json &element)
{
  const namespacet ns(converter_.symbol_table());
  const array_typet &array_type =
    static_cast<const array_typet &>(ns.follow(array.type()));
  const array_typet &mask_type =
    static_cast<const array_typet &>(ns.follow(mask.type()));

  const BigInt array_len =
    binary2integer(array_type.size().value().c_str(), false);
  const BigInt mask_len =
    binary2integer(mask_type.size().value().c_str(), false);

  if (array_len != mask_len)
  {
    std::ostringstream msg;
    msg << "IndexError: boolean index did not match indexed array; mask "
           "length "
        << mask_len << " does not match array length " << array_len;
    const locationt loc = converter_.get_location_from_decl(element);
    if (!loc.is_nil())
      msg << " at " << loc.get_file() << ":" << loc.get_line();
    throw std::runtime_error(msg.str());
  }

  const locationt location = converter_.get_location_from_decl(element);
  const typet elem_type = array_type.subtype();

  // Whole-row selection (mask over the outer axis of an n-D array) cannot
  // reuse the runtime-list path below: pushing an array-typed element into
  // the PyListObject model produces a bit-vector/array sort mismatch at the
  // solver backend (confirmed empirically: "Sorts (_ BitVec N) and (Array
  // ...) are incompatible"), matching the encoding gap already flagged for
  // list comprehensions over n-D arrays. Route it through
  // build_bool_mask_row_select instead, which selects rows into a
  // fixed-size array result the same way build_column_select does.
  if (elem_type.is_array())
    return build_bool_mask_row_select(array, mask, element);

  symbolt &result_list = create_list();
  const std::string result_list_id = result_list.id.as_string();

  symbolt &index_var = converter_.create_tmp_symbol(
    element, "$mask_i$", size_type(), gen_zero(size_type()));
  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  code_assignt index_init(build_symbol(index_var), gen_zero(size_type()));
  index_init.location() = location;
  converter_.add_instruction(index_init);

  code_blockt loop_body;

  // build_push_list_call emits its element-staging declarations via
  // converter_.add_instruction(), which targets whatever block is
  // current_block. Redirect it to loop_body so those declarations land
  // inside the loop (and so the staged element is refreshed every
  // iteration) instead of being hoisted once before the loop runs.
  code_blockt *saved_block = converter_.current_block;
  converter_.current_block = &loop_body;

  exprt mask_elem = build_index(mask, build_symbol(index_var), bool_type());
  exprt array_elem = build_index(array, build_symbol(index_var), elem_type);

  exprt push_call = build_push_list_call(result_list, element, array_elem);

  converter_.current_block = saved_block;

  codet if_stmt;
  if_stmt.set_statement("ifthenelse");
  code_blockt then_block;
  then_block.copy_to_operands(push_call);
  if_stmt.copy_to_operands(mask_elem, then_block);
  if_stmt.location() = location;
  loop_body.copy_to_operands(if_stmt);

  exprt increment =
    build_add(build_symbol(index_var), gen_one(size_type()), size_type());
  code_assignt index_increment(build_symbol(index_var), increment);
  index_increment.location() = location;
  loop_body.copy_to_operands(index_increment);

  exprt loop_condition =
    build_less_than(build_symbol(index_var), array_type.size());

  codet while_stmt;
  while_stmt.set_statement("while");
  while_stmt.copy_to_operands(loop_condition, loop_body);
  while_stmt.location() = location;
  converter_.add_instruction(while_stmt);

  elem_types().record(result_list_id, "", elem_type);

  return build_symbol(result_list);
}

exprt python_list::build_bool_mask_row_select(
  const exprt &array,
  const exprt &mask,
  const nlohmann::json &element)
{
  const namespacet ns(converter_.symbol_table());
  const array_typet &outer_type =
    static_cast<const array_typet &>(ns.follow(array.type()));
  const typet row_type = outer_type.subtype();
  const BigInt num_rows =
    binary2integer(outer_type.size().value().c_str(), false);
  const locationt location = converter_.get_location_from_decl(element);

  const array_typet &mask_type =
    static_cast<const array_typet &>(ns.follow(mask.type()));
  const BigInt mask_len =
    binary2integer(mask_type.size().value().c_str(), false);
  if (mask_len != num_rows)
  {
    std::ostringstream msg;
    msg << "IndexError: boolean index did not match indexed array; mask "
           "length "
        << mask_len << " does not match array length " << num_rows;
    if (!location.is_nil())
      msg << " at " << location.get_file() << ":" << location.get_line();
    throw std::runtime_error(msg.str());
  }

  auto reject = [&](const std::string &reason) -> exprt {
    std::ostringstream msg;
    msg << "TypeError: boolean-mask row selection (a[mask]) " << reason;
    if (!location.is_nil())
      msg << " at " << location.get_file() << ":" << location.get_line();
    throw std::runtime_error(msg.str());
  };

  if (
    !element.contains("slice") ||
    element["slice"].value("_type", "") != "Name" ||
    !element["slice"].contains("id"))
    return reject(
      "requires a mask variable whose value is a concrete boolean literal");

  const std::string mask_name = element["slice"]["id"].get<std::string>();

  // find_var_decl returns the first textual assignment to mask_name in
  // scope, not the one that actually reaches this use site. A reassigned
  // mask can't be trusted via that first declaration for the literal
  // fast path below, but build_bool_mask_row_select_symbolic reads the
  // mask's live runtime value directly (not its AST declaration), so
  // reassignment is sound there.
  if (json_utils::has_multiple_assignments_in_scope(
        mask_name, converter_.current_function_name(), converter_.ast()))
    return build_bool_mask_row_select_symbolic(array, mask, element);

  const nlohmann::json mask_decl = json_utils::find_var_decl(
    mask_name, converter_.current_function_name(), converter_.ast());

  // No local declaration is found for a mask that is itself a function
  // parameter: there is no AST assignment to resolve a literal from, so
  // read its runtime value instead, same as the
  // reassigned-mask case above.
  if (
    mask_decl.is_null() || !mask_decl.contains("value") ||
    mask_decl["value"].value("_type", "") != "Call" ||
    !mask_decl["value"].contains("args") ||
    mask_decl["value"]["args"].empty() ||
    mask_decl["value"]["args"][0].value("_type", "") != "List")
    return build_bool_mask_row_select_symbolic(array, mask, element);

  const nlohmann::json &mask_elts = mask_decl["value"]["args"][0]["elts"];

  // Every element a literal bool takes the compile-time-sized fast path
  // below; any other element (nondet, a computed value, ...) means the
  // mask is symbolic, which build_bool_mask_row_select_symbolic handles by
  // reading the mask's runtime value in a bounded loop instead.
  std::vector<bool> mask_values;
  mask_values.reserve(mask_elts.size());
  for (const auto &elt : mask_elts)
  {
    if (
      elt.value("_type", "") != "Constant" || !elt.contains("value") ||
      !elt["value"].is_boolean())
      return build_bool_mask_row_select_symbolic(array, mask, element);
    mask_values.push_back(elt["value"].get<bool>());
  }

  if (BigInt(mask_values.size()) != num_rows)
    return reject(
      "requires the mask literal's element count to match the mask's "
      "declared array length");

  BigInt selected_count = 0;
  for (bool v : mask_values)
    if (v)
      ++selected_count;

  const array_typet result_type(
    row_type, from_integer(selected_count, size_type()));
  symbolt &result = converter_.create_tmp_symbol(
    element, "$bool_mask_rows$", result_type, exprt());
  code_declt result_decl(build_symbol(result));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  // Whole-array assignment (`dst[k] = src[i]` where both sides are
  // themselves arrays) is not valid C and is unsupported by the backend --
  // confirmed empirically, ESBMC's C frontend itself rejects `b[0]=a[0]` as
  // "array type ... is not assignable" -- so each row is copied column by
  // column instead, mirroring build_column_select's element-wise copy.
  const array_typet &row_array_type = to_array_type(ns.follow(row_type));
  const typet elem_type = ns.follow(row_array_type.subtype());
  if (elem_type.is_array())
    return reject(
      "currently supports only 2-D arrays; 3-D+ rows are not modelled");
  const BigInt num_cols =
    binary2integer(row_array_type.size().value().c_str(), false);

  BigInt dst_row = 0;
  for (BigInt src_row = 0; src_row < num_rows; ++src_row)
  {
    if (!mask_values[src_row.to_uint64()])
      continue;

    exprt src_row_expr =
      build_index(array, from_integer(src_row, size_type()), row_type);
    exprt dst_row_expr = build_index(
      build_symbol(result), from_integer(dst_row, size_type()), row_type);

    for (BigInt col = 0; col < num_cols; ++col)
    {
      exprt src_elem =
        build_index(src_row_expr, from_integer(col, size_type()), elem_type);
      exprt dst_elem =
        build_index(dst_row_expr, from_integer(col, size_type()), elem_type);
      code_assignt assign(dst_elem, src_elem);
      assign.location() = location;
      converter_.add_instruction(assign);
    }
    ++dst_row;
  }

  return build_symbol(result);
}

bool python_list::is_bool_mask_rows_type(const typet &type)
{
  if (!type.is_struct())
    return false;
  return to_struct_type(type).tag().as_string().find(
           "tag-numpy_bool_mask_rows") == 0;
}

exprt python_list::build_bool_mask_row_select_symbolic(
  const exprt &array,
  const exprt &mask,
  const nlohmann::json &element)
{
  const namespacet ns(converter_.symbol_table());
  const array_typet &outer_type =
    static_cast<const array_typet &>(ns.follow(array.type()));
  const typet row_type = outer_type.subtype();
  const BigInt num_rows =
    binary2integer(outer_type.size().value().c_str(), false);
  const locationt location = converter_.get_location_from_decl(element);

  const array_typet &row_array_type = to_array_type(ns.follow(row_type));
  const typet elem_type = ns.follow(row_array_type.subtype());
  if (elem_type.is_array())
  {
    std::ostringstream msg;
    msg << "TypeError: boolean-mask row selection (a[mask]) currently "
           "supports only 2-D arrays; 3-D+ rows are not modelled";
    if (!location.is_nil())
      msg << " at " << location.get_file() << ":" << location.get_line();
    throw std::runtime_error(msg.str());
  }
  const BigInt num_cols =
    binary2integer(row_array_type.size().value().c_str(), false);

  // Canonical bounded descriptor result: a `rows` buffer sized
  // at the worst case (every row selected) plus a `count` member holding
  // the number of rows actually selected, so the logical row count is part
  // of the modelled value instead of a detached counter.
  const array_typet rows_buffer_type(
    row_type, from_integer(num_rows, size_type()));
  struct_typet result_type;
  result_type.components().push_back(
    struct_typet::componentt("rows", "rows", rows_buffer_type));
  result_type.components().push_back(
    struct_typet::componentt("count", "count", size_type()));
  result_type.tag(
    "tag-numpy_bool_mask_rows_" + row_type.to_string() + "_" +
    integer2string(num_rows));

  symbolt &result = converter_.create_tmp_symbol(
    element, "$bool_mask_rows$", result_type, exprt());
  code_declt result_decl(build_symbol(result));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  exprt rows_member =
    build_member(build_symbol(result), "rows", rows_buffer_type);
  exprt count_member = build_member(build_symbol(result), "count", size_type());

  code_assignt count_init(count_member, gen_zero(size_type()));
  count_init.location() = location;
  converter_.add_instruction(count_init);

  symbolt &index_var = converter_.create_tmp_symbol(
    element, "$mask_row_i$", size_type(), gen_zero(size_type()));
  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);
  code_assignt index_init(build_symbol(index_var), gen_zero(size_type()));
  index_init.location() = location;
  converter_.add_instruction(index_init);

  code_blockt loop_body;

  // Redirect current_block so every statement built below (the per-column
  // copy and the count increment) lands inside the loop body instead of
  // being hoisted once before the loop runs (mirrors build_bool_mask_index).
  code_blockt *saved_block = converter_.current_block;
  converter_.current_block = &loop_body;

  exprt mask_elem = build_index(mask, build_symbol(index_var), bool_type());
  exprt src_row_expr = build_index(array, build_symbol(index_var), row_type);
  exprt dst_row_expr = build_index(rows_member, count_member, row_type);

  code_blockt then_block;
  for (BigInt col = 0; col < num_cols; ++col)
  {
    exprt src_elem =
      build_index(src_row_expr, from_integer(col, size_type()), elem_type);
    exprt dst_elem =
      build_index(dst_row_expr, from_integer(col, size_type()), elem_type);
    code_assignt assign(dst_elem, src_elem);
    assign.location() = location;
    then_block.copy_to_operands(assign);
  }
  exprt count_increment =
    build_add(count_member, gen_one(size_type()), size_type());
  code_assignt count_inc_stmt(count_member, count_increment);
  count_inc_stmt.location() = location;
  then_block.copy_to_operands(count_inc_stmt);

  codet if_stmt;
  if_stmt.set_statement("ifthenelse");
  if_stmt.copy_to_operands(mask_elem, then_block);
  if_stmt.location() = location;
  loop_body.copy_to_operands(if_stmt);

  exprt index_increment =
    build_add(build_symbol(index_var), gen_one(size_type()), size_type());
  code_assignt index_inc_stmt(build_symbol(index_var), index_increment);
  index_inc_stmt.location() = location;
  loop_body.copy_to_operands(index_inc_stmt);

  converter_.current_block = saved_block;

  exprt loop_condition =
    build_less_than(build_symbol(index_var), outer_type.size());

  codet while_stmt;
  while_stmt.set_statement("while");
  while_stmt.copy_to_operands(loop_condition, loop_body);
  while_stmt.location() = location;
  converter_.add_instruction(while_stmt);

  return build_symbol(result);
}

exprt python_list::index_bool_mask_rows(
  const exprt &array,
  const nlohmann::json &slice_node,
  const nlohmann::json &element)
{
  const locationt location = converter_.get_location_from_decl(element);

  if (slice_node.value("_type", "") == "Slice")
  {
    std::ostringstream msg;
    msg << "TypeError: slicing a boolean-mask row-selection result is not "
           "supported yet";
    if (!location.is_nil())
      msg << " at " << location.get_file() << ":" << location.get_line();
    throw std::runtime_error(msg.str());
  }

  const namespacet ns(converter_.symbol_table());
  const struct_typet &result_type = to_struct_type(ns.follow(array.type()));
  const array_typet &rows_array_type =
    to_array_type(ns.follow(result_type.components()[0].type()));
  const typet row_type = rows_array_type.subtype();

  exprt index = converter_.get_expr(slice_node);
  index = converter_.unwrap_optional_if_needed(index);

  exprt rows_member = build_member(array, "rows", rows_array_type);
  exprt count_member = build_member(array, "count", size_type());
  exprt index_as_size = build_typecast(index, size_type());

  // actual_index = (index < 0) ? (count + index) : index, normalized
  // against the logical row count rather than the buffer's physical
  // capacity (mirrors build_list_at_call's negative-index handling).
  const type2tc size_t2 = migrate_type(size_type());
  expr2tc index2, index_as_size2, count2;
  migrate_expr(index, index2);
  migrate_expr(index_as_size, index_as_size2);
  migrate_expr(count_member, count2);

  expr2tc is_negative = lessthan2tc(index2, gen_zero(index2->type));
  expr2tc positive_index = add2tc(size_t2, count2, index_as_size2);
  expr2tc converted_index2 =
    if2tc(size_t2, is_negative, positive_index, index_as_size2);
  exprt converted_index = migrate_expr_back(converted_index2);

  if (!config.options.get_bool_option("no-bounds-check"))
  {
    expr2tc oob = greaterthanequal2tc(converted_index2, count2);

    exprt raise = converter_.get_exception_handler().gen_exception_raise(
      "IndexError", "index out of range for boolean-mask row selection");
    codet throw_code("expression");
    throw_code.operands().push_back(raise);

    code_ifthenelset guard;
    guard.cond() = migrate_expr_back(oob);
    guard.then_case() = throw_code;
    guard.location() = location;
    guard.location().property("skipped");
    converter_.add_instruction(guard);
  }

  return build_index(rows_member, converted_index, row_type);
}

namespace
{
// Recognizes a literal integer index node: a plain Constant, or a negated
// Constant (UnaryOp USub), and extracts its signed value. Used to resolve
// 2-D column/row indices and fancy-index entries entirely at conversion
// time, without needing a runtime representation for the index list.
bool try_get_literal_int(const nlohmann::json &node, BigInt &out)
{
  // Booleans are deliberately excluded: NumPy treats an all-bool literal
  // list (`a[[True, False]]`) as a boolean mask, not positional indices, so
  // accepting them here would silently compute the wrong result instead of
  // rejecting it. Let callers fail with a clear TypeError instead.
  if (
    node.contains("_type") && node["_type"] == "Constant" &&
    node.contains("value") && node["value"].is_number_integer())
  {
    out = BigInt(node["value"].get<long long>());
    return true;
  }
  if (
    node.contains("_type") && node["_type"] == "UnaryOp" &&
    node.contains("op") && node["op"]["_type"] == "USub" &&
    node.contains("operand") && node["operand"]["_type"] == "Constant" &&
    node["operand"].contains("value") &&
    node["operand"]["value"].is_number_integer())
  {
    out = -BigInt(node["operand"]["value"].get<long long>());
    return true;
  }
  return false;
}

// Resolves literal lower/upper bounds (or their step-dependent defaults)
// into raw, negative-index-adjusted start/stop values -- not yet clamped to
// the source length, that is literal_slice_length's job below. nullopt when
// either given bound is present but not a compile-time literal.
std::optional<std::pair<long long, long long>> resolve_literal_slice_bounds(
  const nlohmann::json &slice_node,
  long long source_len,
  long long step)
{
  auto bound = [&](const char *name) -> std::optional<long long> {
    if (!slice_node.contains(name) || slice_node[name].is_null())
      return std::nullopt;

    BigInt value;
    if (!try_get_literal_int(slice_node[name], value))
      return std::nullopt;
    return value.to_int64();
  };

  std::optional<long long> lower = bound("lower");
  std::optional<long long> upper = bound("upper");
  const bool has_lower =
    slice_node.contains("lower") && !slice_node["lower"].is_null();
  const bool has_upper =
    slice_node.contains("upper") && !slice_node["upper"].is_null();
  if ((has_lower && !lower) || (has_upper && !upper))
    return std::nullopt;

  long long start = lower.value_or(step < 0 ? source_len - 1 : 0);
  long long stop = upper.value_or(step < 0 ? -1 : source_len);

  if (has_lower && start < 0)
    start += source_len;
  if (has_upper && stop < 0)
    stop += source_len;

  return std::make_pair(start, stop);
}

// out_start, when non-null, receives the normalized (negative-index-adjusted,
// clamped) literal start offset -- the same value used to compute the
// length below, exposed for callers that need the actual byte/element
// offset into the source (e.g. building a pointer view), not just the
// resulting slice length.
std::optional<long long> literal_slice_length(
  const nlohmann::json &slice_node,
  long long source_len,
  long long step,
  long long *out_start = nullptr)
{
  std::optional<std::pair<long long, long long>> bounds =
    resolve_literal_slice_bounds(slice_node, source_len, step);
  if (!bounds)
    return std::nullopt;
  long long start = bounds->first;
  long long stop = bounds->second;

  if (step < 0)
  {
    start = std::min(std::max(start, -1LL), source_len - 1);
    stop = std::min(std::max(stop, -1LL), source_len - 1);
    if (out_start)
      *out_start = start;
    if (start <= stop)
      return 0;
    return ((start - stop - 1) / -step) + 1;
  }

  start = std::min(std::max(start, 0LL), source_len);
  stop = std::min(std::max(stop, 0LL), source_len);
  if (out_start)
    *out_start = start;
  if (start >= stop)
    return 0;
  return ((stop - start - 1) / step) + 1;
}

// Real identity of a numpy array's source symbol (ADR-NP-003's canonical
// buffer_id), 0 for a non-symbol source. See docs/roadmap/
// numpy-support-assessment.md, "Definitive view descriptor model", for why
// nothing consults this yet.
std::size_t numpy_symbol_buffer_id(const exprt &array)
{
  return array.is_symbol()
           ? std::hash<std::string>{}(array.identifier().as_string())
           : 0;
}

void append_array_shape(const typet &type, std::vector<long long> &shape)
{
  if (!type.is_array())
    return;

  const array_typet &array_type = to_array_type(type);
  if (array_type.size().is_nil() || !array_type.size().is_constant())
    return;
  shape.push_back(
    binary2integer(array_type.size().value().c_str(), false).to_int64());
  append_array_shape(array_type.subtype(), shape);
}

struct row_pointer_view_info
{
  typet elem_type;
  long long row_index;
  long long col_count;
};

struct slice_step_info
{
  long long value;
  bool literal_zero;
  bool literal;
};

slice_step_info get_slice_step_info(const nlohmann::json &slice_node)
{
  slice_step_info info{1, false, true};
  if (!slice_node.contains("step") || slice_node["step"].is_null())
    return info;

  const auto &step_node = slice_node["step"];
  BigInt literal_value;
  if (!try_get_literal_int(step_node, literal_value))
  {
    info.literal = false;
    return info;
  }

  info.value = literal_value.to_int64();
  if (info.value == 0)
  {
    info.literal_zero = true;
    info.value = 1;
  }
  return info;
}

bool can_build_scalar_pointer_view(
  const exprt &array,
  exprt *current_lhs,
  bool is_numpy_array)
{
  return array.is_symbol() && current_lhs && current_lhs->is_symbol() &&
         is_numpy_array;
}

struct fixed_2d_shape_info
{
  typet elem_type;
  long long row_count;
  long long col_count;
  typet row_type;
};

// Resolves array's compile-time-known 2-D shape (row_count x col_count) and
// scalar element type, or nullopt when array is not a fixed-shape 2-D
// array (including 3-D+, where elem_type itself would still be an array).
// Shared by every producer that addresses a row or column of a 2-D numpy
// array (try_build_row_pointer_view, try_build_column_pointer_view).
std::optional<fixed_2d_shape_info>
get_fixed_2d_shape_info(const exprt &array, const contextt &symbol_table)
{
  const namespacet ns(symbol_table);
  const typet array_type = ns.follow(array.type());
  if (!array_type.is_array())
    return std::nullopt;

  const array_typet &outer_array = to_array_type(array_type);
  if (outer_array.size().is_nil() || !outer_array.size().is_constant())
    return std::nullopt;

  const typet row_type = ns.follow(outer_array.subtype());
  if (!row_type.is_array())
    return std::nullopt;

  const array_typet &row_array = to_array_type(row_type);
  if (row_array.size().is_nil() || !row_array.size().is_constant())
    return std::nullopt;

  const typet elem_type = ns.follow(row_array.subtype());
  if (elem_type.is_array())
    return std::nullopt;

  const long long row_count =
    binary2integer(outer_array.size().value().c_str(), false).to_int64();
  const long long col_count =
    binary2integer(row_array.size().value().c_str(), false).to_int64();
  if (row_count < 0 || col_count < 0)
    return std::nullopt;

  return fixed_2d_shape_info{elem_type, row_count, col_count, row_type};
}

std::optional<row_pointer_view_info> get_row_pointer_view_info(
  const exprt &array,
  const nlohmann::json &slice_node,
  const contextt &symbol_table,
  exprt *current_lhs,
  bool is_numpy_array)
{
  if (!can_build_scalar_pointer_view(array, current_lhs, is_numpy_array))
    return std::nullopt;

  BigInt literal_index;
  if (!try_get_literal_int(slice_node, literal_index))
    return std::nullopt;

  std::optional<fixed_2d_shape_info> shape =
    get_fixed_2d_shape_info(array, symbol_table);
  if (!shape)
    return std::nullopt;

  row_pointer_view_info info{
    shape->elem_type, literal_index.to_int64(), shape->col_count};
  if (info.row_index < 0)
    info.row_index += shape->row_count;
  if (info.row_index < 0 || info.row_index >= shape->row_count)
    return std::nullopt;

  return info;
}

struct column_pointer_view_info
{
  typet elem_type;
  long long col_index;
  long long num_cols;
  long long num_rows;
};

std::optional<column_pointer_view_info> get_column_pointer_view_info(
  const exprt &array,
  const nlohmann::json &col_index_node,
  const contextt &symbol_table,
  exprt *current_lhs,
  bool is_numpy_array)
{
  if (!can_build_scalar_pointer_view(array, current_lhs, is_numpy_array))
    return std::nullopt;

  BigInt literal_index;
  if (!try_get_literal_int(col_index_node, literal_index))
    return std::nullopt;

  std::optional<fixed_2d_shape_info> shape =
    get_fixed_2d_shape_info(array, symbol_table);
  if (!shape)
    return std::nullopt;

  column_pointer_view_info info{
    shape->elem_type,
    literal_index.to_int64(),
    shape->col_count,
    shape->row_count};
  if (info.col_index < 0)
    info.col_index += shape->col_count;
  if (info.col_index < 0 || info.col_index >= shape->col_count)
    return std::nullopt;

  return info;
}

struct diagonal_pointer_view_info
{
  typet elem_type;
  long long offset;
  long long length;
  long long stride;
};

// np.diagonal(a[, offset=k]): the main diagonal (k=0) or a parallel one.
// k >= 0 starts at a[0][k] (flat offset k); k < 0 starts at a[-k][0] (flat
// offset (-k)*num_cols); either way the diagonal steps num_cols+1 flat
// elements per logical index, and its length is clamped to whichever axis
// runs out first (0 when the offset itself is already past every row or
// column, e.g. k >= num_cols). Structural eligibility only (fixed 2-D
// shape) -- deliberately independent of current_lhs, unlike
// get_row_pointer_view_info/get_column_pointer_view_info, because a Call
// RHS (unlike a Subscript) gets a genuine second, current_lhs-correct
// conversion pass with no cached-reuse trap to work around; see
// try_build_diagonal_pointer_view for why current_lhs is checked there
// instead.
std::optional<diagonal_pointer_view_info> get_diagonal_pointer_view_info(
  const exprt &array,
  long long k,
  const contextt &symbol_table)
{
  std::optional<fixed_2d_shape_info> shape =
    get_fixed_2d_shape_info(array, symbol_table);
  if (!shape)
    return std::nullopt;

  long long offset;
  long long length;
  if (k >= 0)
  {
    offset = k;
    length = std::min(shape->row_count, shape->col_count - k);
  }
  else
  {
    offset = (-k) * shape->col_count;
    length = std::min(shape->row_count + k, shape->col_count);
  }
  if (length <= 0)
  {
    // An out-of-range offset (e.g. k <= -row_count or k >= col_count) still
    // computes a flat offset above, but with no elements to index through
    // it, that offset can run arbitrarily far past the buffer (k=-100 on a
    // 3x3 array gives offset=300). Forming that pointer at all -- even
    // though never dereferenced, since length is 0 -- is undefined behaviour
    // past one-past-the-end (C 6.5.6p8), so clamp it to the buffer's own
    // one-past-the-end instead.
    length = 0;
    offset = shape->row_count * shape->col_count;
  }

  return diagonal_pointer_view_info{
    shape->elem_type, offset, length, shape->col_count + 1};
}

struct flat_array_shape_info
{
  typet elem_type;
  long long total_length;
};

// Resolves array's compile-time-known total element count for a fixed-shape
// 1-D or 2-D array, or nullopt for anything else (including 3-D+, which
// np.ravel()'s pointer-view aliasing does not cover -- it keeps using the
// existing copy path there).
std::optional<flat_array_shape_info>
get_flat_1d_or_2d_shape_info(const exprt &array, const contextt &symbol_table)
{
  const namespacet ns(symbol_table);
  const typet outer_type = ns.follow(array.type());
  if (!outer_type.is_array())
    return std::nullopt;

  const array_typet &outer_array = to_array_type(outer_type);
  if (outer_array.size().is_nil() || !outer_array.size().is_constant())
    return std::nullopt;
  const long long outer_count =
    binary2integer(outer_array.size().value().c_str(), false).to_int64();
  if (outer_count < 0)
    return std::nullopt;

  const typet inner_type = ns.follow(outer_array.subtype());
  if (!inner_type.is_array())
    return flat_array_shape_info{inner_type, outer_count};

  const array_typet &inner_array = to_array_type(inner_type);
  if (inner_array.size().is_nil() || !inner_array.size().is_constant())
    return std::nullopt;
  const long long inner_count =
    binary2integer(inner_array.size().value().c_str(), false).to_int64();
  if (inner_count < 0)
    return std::nullopt;

  const typet elem_type = ns.follow(inner_array.subtype());
  if (elem_type.is_array())
    return std::nullopt;

  return flat_array_shape_info{elem_type, outer_count * inner_count};
}
} // namespace

exprt python_list::resolve_fixed_axis_index(
  const nlohmann::json &idx_node,
  const BigInt &axis_len,
  unsigned axis,
  const nlohmann::json &element)
{
  const locationt location = converter_.get_location_from_decl(element);

  BigInt literal_value;
  if (try_get_literal_int(idx_node, literal_value))
  {
    BigInt normalized = literal_value;
    if (normalized < 0)
      normalized += axis_len;

    if (normalized < 0 || normalized >= axis_len)
    {
      std::ostringstream msg;
      msg << "IndexError: index " << literal_value
          << " is out of bounds for axis " << axis << " with size " << axis_len;
      if (!location.is_nil())
        msg << " at " << location.get_file() << ":" << location.get_line();
      throw std::runtime_error(msg.str());
    }

    return from_integer(normalized, size_type());
  }

  // Runtime (non-constant) index: normalize negative values against the
  // compile-time-known axis length and guard with an in-model IndexError,
  // mirroring build_list_at_call's negative-index normalization.
  exprt idx_expr = converter_.get_expr(idx_node);
  const typet signed_t = signed_size_type();
  exprt idx_signed = build_typecast(idx_expr, signed_t);
  exprt len_signed = from_integer(axis_len, signed_t);

  const type2tc signed_t2 = migrate_type(signed_t);
  expr2tc idx2, len2;
  migrate_expr(idx_signed, idx2);
  migrate_expr(len_signed, len2);

  expr2tc is_negative = lessthan2tc(idx2, gen_zero(signed_t2));
  expr2tc positive_index = add2tc(signed_t2, len2, idx2);
  expr2tc normalized2 = if2tc(signed_t2, is_negative, positive_index, idx2);
  exprt normalized = migrate_expr_back(normalized2);

  if (!config.options.get_bool_option("no-bounds-check"))
  {
    expr2tc norm2;
    migrate_expr(normalized, norm2);
    expr2tc oob = or2tc(
      lessthan2tc(norm2, gen_zero(signed_t2)),
      greaterthanequal2tc(norm2, len2));

    exprt raise = converter_.get_exception_handler().gen_exception_raise(
      "IndexError", "index is out of bounds for axis " + std::to_string(axis));
    codet throw_code("expression");
    throw_code.operands().push_back(raise);

    code_ifthenelset guard;
    guard.cond() = migrate_expr_back(oob);
    guard.then_case() = throw_code;
    guard.location() = location;
    guard.location().property("skipped");
    converter_.add_instruction(guard);
  }

  return build_typecast(normalized, size_type());
}

exprt python_list::build_column_select(
  const exprt &array,
  const nlohmann::json &col_index_node,
  const nlohmann::json &element)
{
  // ADR-NP-003 etapa 2: a literal, in-range column of a fixed 2-D numpy
  // array assigned directly to a bare name aliases the source's buffer via
  // a strided pointer instead of the element-by-element copy below.
  // Declines for every case this function's own validation below still
  // needs to run (non-2-D, 3-D+, out-of-range column, symbolic column,
  // non-numpy source, ...), so those diagnostics are unaffected.
  if (
    std::optional<exprt> view =
      try_build_column_pointer_view(array, col_index_node))
    return *view;

  const namespacet ns(converter_.symbol_table());
  const typet resolved_array_type = ns.follow(array.type());
  const locationt location = converter_.get_location_from_decl(element);

  if (!resolved_array_type.is_array())
  {
    std::ostringstream msg;
    msg << "TypeError: 2-D column slicing (a[:, j]) requires a fixed-shape "
           "2-D array";
    if (!location.is_nil())
      msg << " at " << location.get_file() << ":" << location.get_line();
    throw std::runtime_error(msg.str());
  }

  const array_typet &outer_type = to_array_type(resolved_array_type);
  const typet row_type = ns.follow(outer_type.subtype());
  if (!row_type.is_array())
  {
    std::ostringstream msg;
    msg << "TypeError: multi-dimensional indexing (a[i, j, ...]) is not "
           "supported; numpy arrays are modelled as 1D lists";
    if (!location.is_nil())
      msg << " at " << location.get_file() << ":" << location.get_line();
    throw std::runtime_error(msg.str());
  }

  const array_typet &row_array_type = to_array_type(row_type);
  const typet elem_type = ns.follow(row_array_type.subtype());
  if (elem_type.is_array())
  {
    std::ostringstream msg;
    msg << "TypeError: 2-D column slicing (a[:, j]) currently supports only "
           "2-D arrays; 3-D+ arrays are not modelled";
    if (!location.is_nil())
      msg << " at " << location.get_file() << ":" << location.get_line();
    throw std::runtime_error(msg.str());
  }
  const BigInt num_rows =
    binary2integer(outer_type.size().value().c_str(), false);
  const BigInt num_cols =
    binary2integer(row_array_type.size().value().c_str(), false);

  exprt col_idx =
    resolve_fixed_axis_index(col_index_node, num_cols, 1, element);

  symbolt &result = converter_.create_tmp_symbol(
    element, "$col_slice$", array_typet(elem_type, outer_type.size()), exprt());
  code_declt result_decl(build_symbol(result));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  for (BigInt row = 0; row < num_rows; ++row)
  {
    exprt row_expr =
      build_index(array, from_integer(row, size_type()), row_type);
    exprt src_elem = build_index(row_expr, col_idx, elem_type);
    exprt dst_elem = build_index(
      build_symbol(result), from_integer(row, size_type()), elem_type);
    code_assignt assign(dst_elem, src_elem);
    assign.location() = location;
    converter_.add_instruction(assign);
  }

  return build_symbol(result);
}

exprt python_list::build_strided_column_select(
  const exprt &array,
  const nlohmann::json &col_slice_node,
  const nlohmann::json &element)
{
  const namespacet ns(converter_.symbol_table());
  const typet resolved_array_type = ns.follow(array.type());
  const locationt location = converter_.get_location_from_decl(element);

  auto reject = [&](const std::string &reason) -> exprt {
    std::ostringstream msg;
    msg << "TypeError: strided column slicing (a[:, ::step]) " << reason;
    if (!location.is_nil())
      msg << " at " << location.get_file() << ":" << location.get_line();
    throw std::runtime_error(msg.str());
  };

  if (!resolved_array_type.is_array())
    return reject("requires a fixed-shape 2-D array");
  const array_typet &outer_type = to_array_type(resolved_array_type);
  const typet row_type = ns.follow(outer_type.subtype());
  if (!row_type.is_array())
    return reject("requires a fixed-shape 2-D array");
  const array_typet &row_array_type = to_array_type(row_type);
  const typet elem_type = ns.follow(row_array_type.subtype());
  if (elem_type.is_array())
    return reject(
      "currently supports only 2-D arrays; 3-D+ arrays are not modelled");

  const BigInt num_rows =
    binary2integer(outer_type.size().value().c_str(), false);
  const BigInt num_cols =
    binary2integer(row_array_type.size().value().c_str(), false);

  BigInt step_val = 1;
  if (col_slice_node.contains("step") && !col_slice_node["step"].is_null())
  {
    if (!try_get_literal_int(col_slice_node["step"], step_val))
      return reject("requires a literal step");
    if (step_val == 0)
      return reject("step cannot be zero");
  }

  const bool has_lower =
    col_slice_node.contains("lower") && !col_slice_node["lower"].is_null();
  const bool has_upper =
    col_slice_node.contains("upper") && !col_slice_node["upper"].is_null();

  if (!has_lower && !has_upper && step_val < 0 && step_val != -1)
    return reject("currently supports only step=-1 for negative steps");

  auto resolve_bound = [&](const std::string &name, bool is_lower) -> BigInt {
    const bool present =
      col_slice_node.contains(name) && !col_slice_node[name].is_null();
    if (!present)
    {
      if (step_val > 0)
        return is_lower ? BigInt(0) : num_cols;
      return is_lower ? num_cols - 1 : BigInt(-1);
    }

    BigInt value;
    if (!try_get_literal_int(col_slice_node[name], value))
      return reject("requires literal bounds"), BigInt(0);

    if (value < 0)
      value += num_cols;

    if (step_val > 0)
    {
      if (value < 0)
        return BigInt(0);
      if (value > num_cols)
        return num_cols;
      return value;
    }

    if (value < 0)
      return BigInt(-1);
    if (value >= num_cols)
      return num_cols - 1;
    return value;
  };

  const BigInt start = resolve_bound("lower", true);
  const BigInt stop = resolve_bound("upper", false);
  std::vector<BigInt> selected_cols;
  if (step_val > 0)
  {
    for (BigInt col = start; col < stop; col += step_val)
      selected_cols.push_back(col);
  }
  else
  {
    for (BigInt col = start; col > stop; col += step_val)
      selected_cols.push_back(col);
  }

  const BigInt result_cols = BigInt(selected_cols.size());

  const array_typet new_row_type(
    elem_type, from_integer(result_cols, size_type()));
  const array_typet result_type(
    new_row_type, from_integer(num_rows, size_type()));
  symbolt &result = converter_.create_tmp_symbol(
    element, "$strided_col_slice$", result_type, exprt());
  code_declt result_decl(build_symbol(result));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  for (BigInt row = 0; row < num_rows; ++row)
  {
    exprt row_expr =
      build_index(array, from_integer(row, size_type()), row_type);
    exprt dst_row = build_index(
      build_symbol(result), from_integer(row, size_type()), new_row_type);

    for (std::size_t col = 0; col < selected_cols.size(); ++col)
    {
      exprt src_elem = build_index(
        row_expr, from_integer(selected_cols[col], size_type()), elem_type);
      exprt dst_elem =
        build_index(dst_row, from_integer(BigInt(col), size_type()), elem_type);
      code_assignt assign(dst_elem, src_elem);
      assign.location() = location;
      converter_.add_instruction(assign);
    }
  }

  return build_symbol(result);
}

exprt python_list::build_mixed_slice_tuple_select(
  const exprt &array,
  const std::vector<nlohmann::json> &idx_nodes,
  const nlohmann::json &element)
{
  const namespacet ns(converter_.symbol_table());
  const locationt location = converter_.get_location_from_decl(element);

  auto reject = [&](const std::string &reason) -> void {
    std::ostringstream msg;
    msg << "TypeError: mixed slice/index tuple indexing " << reason;
    if (!location.is_nil())
      msg << " at " << location.get_file() << ":" << location.get_line();
    throw std::runtime_error(msg.str());
  };

  // Walk one nested array dimension per idx_nodes entry, collecting each
  // axis's extent and the type reached after indexing it. A dimension count
  // shorter than idx_nodes means the tuple has more entries than the array
  // has axes.
  std::vector<BigInt> axis_sizes(idx_nodes.size());
  std::vector<typet> types_after(idx_nodes.size());
  typet current_type = ns.follow(array.type());
  for (std::size_t axis = 0; axis < idx_nodes.size(); ++axis)
  {
    if (!current_type.is_array())
    {
      std::ostringstream msg;
      msg << "IndexError: too many indices for array: array has fewer "
             "dimensions than the "
          << idx_nodes.size() << " indices given";
      if (!location.is_nil())
        msg << " at " << location.get_file() << ":" << location.get_line();
      throw std::runtime_error(msg.str());
    }

    const array_typet &at = to_array_type(current_type);
    axis_sizes[axis] = binary2integer(at.size().value().c_str(), false);
    current_type = ns.follow(at.subtype());
    types_after[axis] = current_type;
  }

  auto resolve_slice_indices = [&](
                                 const nlohmann::json &slice_node,
                                 std::size_t axis) -> std::vector<BigInt> {
    BigInt step_val = 1;
    if (slice_node.contains("step") && !slice_node["step"].is_null())
    {
      if (!try_get_literal_int(slice_node["step"], step_val))
        reject("requires literal slice bounds");
      if (step_val == 0)
        reject("slice step cannot be zero");
    }

    const BigInt axis_len = axis_sizes[axis];
    auto resolve_bound = [&](const std::string &name, bool is_lower) -> BigInt {
      const bool present =
        slice_node.contains(name) && !slice_node[name].is_null();
      if (!present)
      {
        if (step_val > 0)
          return is_lower ? BigInt(0) : axis_len;
        return is_lower ? axis_len - 1 : BigInt(-1);
      }

      BigInt value;
      if (!try_get_literal_int(slice_node[name], value))
        reject("requires literal slice bounds");

      if (value < 0)
        value += axis_len;

      if (step_val > 0)
      {
        if (value < 0)
          return BigInt(0);
        if (value > axis_len)
          return axis_len;
        return value;
      }

      if (value < 0)
        return BigInt(-1);
      if (value >= axis_len)
        return axis_len - 1;
      return value;
    };

    const BigInt start = resolve_bound("lower", true);
    const BigInt stop = resolve_bound("upper", false);
    std::vector<BigInt> indices;
    if (step_val > 0)
    {
      for (BigInt idx = start; idx < stop; idx += step_val)
        indices.push_back(idx);
    }
    else
    {
      for (BigInt idx = start; idx > stop; idx += step_val)
        indices.push_back(idx);
    }
    return indices;
  };

  std::vector<std::vector<BigInt>> slice_indices(idx_nodes.size());
  std::vector<exprt> fixed_idx(idx_nodes.size());
  std::vector<std::size_t> slice_axes;
  for (std::size_t axis = 0; axis < idx_nodes.size(); ++axis)
  {
    if (idx_nodes[axis].value("_type", "") == "Slice")
    {
      slice_axes.push_back(axis);
      slice_indices[axis] = resolve_slice_indices(idx_nodes[axis], axis);
    }
    else
    {
      fixed_idx[axis] = resolve_fixed_axis_index(
        idx_nodes[axis],
        axis_sizes[axis],
        static_cast<unsigned>(axis),
        element);
    }
  }

  if (slice_axes.empty())
    reject("requires at least one slice axis");

  const typet result_elem_type = types_after.back();
  if (result_elem_type.is_array())
    reject("currently supports only fully-indexed trailing axes");

  typet result_type = result_elem_type;
  for (auto it = slice_axes.rbegin(); it != slice_axes.rend(); ++it)
  {
    const BigInt len = BigInt(slice_indices[*it].size());
    result_type = array_typet(result_type, from_integer(len, size_type()));
  }
  symbolt &result = converter_.create_tmp_symbol(
    element, "$tuple_mixed_slice$", result_type, exprt());
  code_declt result_decl(build_symbol(result));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  std::function<void(std::size_t, const exprt &, const exprt &)> copy_axis =
    [&](std::size_t axis, const exprt &src, const exprt &dst) {
      if (axis == idx_nodes.size())
      {
        code_assignt assign(dst, src);
        assign.location() = location;
        converter_.add_instruction(assign);
        return;
      }

      if (idx_nodes[axis].value("_type", "") != "Slice")
      {
        exprt src_next = build_index(src, fixed_idx[axis], types_after[axis]);
        copy_axis(axis + 1, src_next, dst);
        return;
      }

      for (std::size_t pos = 0; pos < slice_indices[axis].size(); ++pos)
      {
        exprt src_next = build_index(
          src,
          from_integer(slice_indices[axis][pos], size_type()),
          types_after[axis]);
        const array_typet &dst_array_type =
          to_array_type(ns.follow(dst.type()));
        const typet dst_elem_type = ns.follow(dst_array_type.subtype());
        exprt dst_next = build_index(
          dst, from_integer(BigInt(pos), size_type()), dst_elem_type);
        copy_axis(axis + 1, src_next, dst_next);
      }
    };

  copy_axis(0, array, build_symbol(result));

  return build_symbol(result);
}

exprt python_list::build_fancy_index(
  const exprt &array,
  const std::vector<nlohmann::json> &indices,
  const nlohmann::json &element)
{
  const namespacet ns(converter_.symbol_table());
  const typet resolved_array_type = ns.follow(array.type());
  const locationt location = converter_.get_location_from_decl(element);

  if (!resolved_array_type.is_array())
  {
    std::ostringstream msg;
    msg << "TypeError: fancy indexing (a[[i, j, ...]]) requires a "
           "fixed-shape array";
    if (!location.is_nil())
      msg << " at " << location.get_file() << ":" << location.get_line();
    throw std::runtime_error(msg.str());
  }

  const array_typet &array_type = to_array_type(resolved_array_type);
  const typet elem_type = ns.follow(array_type.subtype());

  const BigInt array_len =
    binary2integer(array_type.size().value().c_str(), false);

  // Every requested index must be a concrete literal integer: fancy indexing
  // is restricted here to compile-time-known selections, so each index is
  // resolved and bounds-checked up front, before any IR for the result.
  std::vector<exprt> resolved_indices;
  resolved_indices.reserve(indices.size());
  for (const auto &idx_node : indices)
  {
    BigInt literal_value;
    if (!try_get_literal_int(idx_node, literal_value))
    {
      std::ostringstream msg;
      msg << "TypeError: fancy indexing only supports concrete integer "
             "indices";
      if (!location.is_nil())
        msg << " at " << location.get_file() << ":" << location.get_line();
      throw std::runtime_error(msg.str());
    }
    resolved_indices.push_back(
      resolve_fixed_axis_index(idx_node, array_len, 0, element));
  }

  const array_typet result_type(
    array_type.subtype(),
    from_integer(BigInt(resolved_indices.size()), size_type()));
  symbolt &result = converter_.create_tmp_symbol(
    element, "$fancy_index$", result_type, exprt());
  code_declt result_decl(build_symbol(result));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  // Whole-array assignment (`dst[k] = src[i]` where both sides are arrays)
  // is not valid C and is unsupported by the backend -- confirmed
  // empirically, ESBMC's C frontend itself rejects `b[0]=a[0]` as "array
  // type ... is not assignable" -- so a row selection (2-D fancy indexing)
  // is copied column by column instead of via a single assignment.
  if (elem_type.is_array())
  {
    const array_typet &row_array_type = to_array_type(elem_type);
    const typet row_elem_type = ns.follow(row_array_type.subtype());
    if (row_elem_type.is_array())
    {
      std::ostringstream msg;
      msg << "TypeError: fancy indexing currently supports only up to 2-D "
             "arrays; 3-D+ arrays are not modelled";
      if (!location.is_nil())
        msg << " at " << location.get_file() << ":" << location.get_line();
      throw std::runtime_error(msg.str());
    }
    const BigInt num_cols =
      binary2integer(row_array_type.size().value().c_str(), false);

    for (std::size_t k = 0; k < resolved_indices.size(); ++k)
    {
      exprt src_row = build_index(array, resolved_indices[k], elem_type);
      exprt dst_row = build_index(
        build_symbol(result), from_integer(BigInt(k), size_type()), elem_type);

      for (BigInt col = 0; col < num_cols; ++col)
      {
        exprt src_elem =
          build_index(src_row, from_integer(col, size_type()), row_elem_type);
        exprt dst_elem =
          build_index(dst_row, from_integer(col, size_type()), row_elem_type);
        code_assignt assign(dst_elem, src_elem);
        assign.location() = location;
        converter_.add_instruction(assign);
      }
    }

    return build_symbol(result);
  }

  for (std::size_t k = 0; k < resolved_indices.size(); ++k)
  {
    exprt src_elem =
      build_index(array, resolved_indices[k], array_type.subtype());
    exprt dst_elem = build_index(
      build_symbol(result),
      from_integer(BigInt(k), size_type()),
      array_type.subtype());
    code_assignt assign(dst_elem, src_elem);
    assign.location() = location;
    converter_.add_instruction(assign);
  }

  return build_symbol(result);
}

exprt python_list::remove_function_calls_recursive(
  exprt &e,
  const nlohmann::json &node)
{
  // Bounds might generate intermediate calls, we need to add lhs to all of
  // them.
  const auto add_lhs_var_bound = [&](exprt &foo) -> exprt {
    if (!foo.is_function_call())
      return foo;
    code_function_callt &call = static_cast<code_function_callt &>(foo);
    symbolt &lhs = converter_.create_tmp_symbol(
      node, "__python_function_call_lhs$", size_type(), exprt());
    call.lhs() = build_symbol(lhs);
    converter_.add_instruction(call);
    return build_symbol(lhs);
  };

  auto res = add_lhs_var_bound(e);
  for (auto &ee : res.operands())
  {
    ee = add_lhs_var_bound(ee);
    remove_function_calls_recursive(ee, node);
  }

  return res;
}

const symbolt &python_list::get_str_slice_sym()
{
  const std::string id = "c:@F@__python_str_slice";
  symbolt *sym = converter_.symbol_table().find_symbol(id);
  if (!sym)
  {
    symbolt new_symbol;
    new_symbol.name = "__python_str_slice";
    new_symbol.id = id;
    new_symbol.mode = "C";
    new_symbol.is_extern = true;
    // char* __python_str_slice(const char*, long long, long long, long long)
    code_typet slice_type;
    typet char_ptr = gen_pointer_type(char_type());
    typet ll_type = signedbv_typet(64);
    slice_type.return_type() = char_ptr;
    slice_type.arguments().push_back(code_typet::argumentt(char_ptr));
    slice_type.arguments().push_back(code_typet::argumentt(ll_type));
    slice_type.arguments().push_back(code_typet::argumentt(ll_type));
    slice_type.arguments().push_back(code_typet::argumentt(ll_type));
    new_symbol.set_type(slice_type);
    converter_.symbol_table().add(new_symbol);
    sym = converter_.symbol_table().find_symbol(id);
  }
  return *sym;
}

// Shared core of every ADR-NP-003 scalar pointer view producer (1-D slice,
// row, column, diagonal, ravel/.flat): builds a pointer_typet(elem_type)
// into array's own flat buffer at element offset `offset`, retypes
// current_lhs to it (its DECL has not been emitted yet at this point --
// see try_build_1d_pointer_view's comment below), and tracks
// {length, stride, readonly} for later consultation (len()/.shape/.ndim,
// indexed reads/writes, detach_numpy_pointer_views_of on source rebind).
// Callers have already validated eligibility; this function only
// assembles the result.
//
// Declines (returns std::nullopt, falling back to whichever copy path the
// caller's own caller uses) if current_lhs's symbol id is *already*
// registered: the map is a single, frontend-only table with no per-branch
// state, so `if c: b = np.ravel(a)` else `b = np.ravel(x)` would otherwise
// have the textually-later branch's {length, stride} silently overwrite
// the other's and apply to both control-flow paths regardless of which one
// actually ran. This is conservative even for a plain, unconditional
// re-assignment of the same name to a second pointer-view-eligible RHS
// (which is unambiguous at runtime and could in principle re-register
// safely) -- there is no cheap way here to distinguish that case from the
// unsafe branch-merge one, so both fall back to an independent copy.
std::optional<exprt> python_list::build_scalar_pointer_view(
  const exprt &array,
  const typet &elem_type,
  long long offset,
  std::size_t length,
  long long stride,
  bool readonly)
{
  const std::string lhs_id = converter_.current_lhs->identifier().as_string();
  if (converter_.numpy_pointer_view_info_.count(lhs_id) != 0)
    return std::nullopt;

  const typet view_ptr_type = pointer_typet(elem_type);
  // Pointer arithmetic, not build_index(array, offset, elem_type): an
  // empty view whose offset lands exactly at the source's element count
  // computes a one-past-the-end address, legal to form but not to
  // dereference (C 6.5.6p8). index2tc would route it through the array
  // bounds checker as an out-of-bounds subscript; add2tc over the decayed
  // pointer is checked under pointer, not array-index, rules.
  exprt base_ptr = build_typecast(build_address_of(array), view_ptr_type);
  exprt view_ptr =
    build_add(base_ptr, from_integer(offset, size_type()), view_ptr_type);
  converter_.current_lhs->type() = view_ptr_type;
  converter_.update_symbol(*converter_.current_lhs);
  converter_.numpy_pointer_view_info_[lhs_id] = {length, stride, readonly};
  return view_ptr;
}

// ADR-NP-003 etapa 2: a literal-bound, literal-step 1-D slice assigned
// directly to a bare name (current_lhs set) becomes a strided pointer into
// the base array's own storage instead of an independent copy -- unit
// stride (v = a[1:3]), a larger step (v = a[::2]), or a negative step
// (v = a[::-1]), in which case literal_start (from literal_slice_length)
// is already the first element the slice visits, so stride is simply
// step_val, positive or negative. Reads/writes through either the view or
// the base then observe each other automatically via ordinary pointer
// semantics -- no manual synchronization needed. Declines (returns
// std::nullopt) for: step == 0 (a ValueError, not a view), char/string
// slices (own null-terminator semantics), a source element type that is
// itself an array (row/column views), a non-symbol base, any slice not
// the direct RHS of a plain assignment (current_lhs unset), and a source
// that is not a tracked numpy array -- `handle_range_slice` is shared by
// numpy arrays, plain Python lists, and bytes/str, and this ADR is
// numpy-scoped: a `bytes`/list slice must keep the existing copy
// semantics (github PR review, bytes_slice_len regression) -- the caller
// falls through to the existing copy for all of these.
std::optional<exprt> python_list::try_build_1d_pointer_view(
  const exprt &array,
  const typet &elem_type,
  long long step_val,
  bool needs_null_term,
  long long literal_start,
  long long static_slice_len)
{
  const namespacet ns(converter_.symbol_table());
  if (
    step_val == 0 || needs_null_term || ns.follow(elem_type).is_array() ||
    !array.is_symbol() || !converter_.current_lhs ||
    !converter_.current_lhs->is_symbol() ||
    converter_.numpy_array_symbols_.count(array.identifier().as_string()) ==
      0 ||
    // literal_start < 0 only happens for an out-of-range negative-step
    // slice that resolves to zero elements (e.g. a[-10:1:-1]):
    // literal_slice_length clamps start to -1 as its "no elements" sentinel
    // for that case. Unlike one-past-the-end (legal to form, C 6.5.6p8),
    // &array[-1] forms a pointer *before* the object, which has no such
    // exemption -- always undefined behaviour, dereferenced or not. Fall
    // through to the existing copy path for this rather than form it.
    literal_start < 0)
    return std::nullopt;

  return build_scalar_pointer_view(
    array,
    elem_type,
    literal_start,
    static_cast<std::size_t>(static_slice_len),
    step_val,
    false);
}

std::optional<exprt> python_list::try_build_row_pointer_view(
  const exprt &array,
  const nlohmann::json &slice_node)
{
  std::optional<row_pointer_view_info> info = get_row_pointer_view_info(
    array,
    slice_node,
    converter_.symbol_table(),
    converter_.current_lhs,
    converter_.numpy_array_symbols_.count(array.identifier().as_string()) != 0);
  if (!info)
    return std::nullopt;

  return build_scalar_pointer_view(
    array,
    info->elem_type,
    info->row_index * info->col_count,
    static_cast<std::size_t>(info->col_count),
    1,
    false);
}

// `col = a[:, j]` (single literal column of a fixed 2-D array): a pointer
// into a's own buffer starting at element j, with stride num_cols so
// col[i] lands on a[i][j]. Declines (returns std::nullopt, falls through
// to build_column_select's existing element-by-element copy) for a
// non-literal or out-of-range column index, a non-symbol/non-numpy-tracked
// base, or when current_lhs is unset -- build_column_select's own
// resolve_fixed_axis_index call still raises the correct IndexError for
// the out-of-range case, so this function does not duplicate that check.
std::optional<exprt> python_list::try_build_column_pointer_view(
  const exprt &array,
  const nlohmann::json &col_index_node)
{
  std::optional<column_pointer_view_info> info = get_column_pointer_view_info(
    array,
    col_index_node,
    converter_.symbol_table(),
    converter_.current_lhs,
    converter_.numpy_array_symbols_.count(array.identifier().as_string()) != 0);
  if (!info)
    return std::nullopt;

  return build_scalar_pointer_view(
    array,
    info->elem_type,
    info->col_index,
    static_cast<std::size_t>(info->num_rows),
    info->num_cols,
    false);
}

// np.diagonal(a[, offset=k])/a.diagonal([k]): a read-only strided pointer
// into a's own buffer (see get_diagonal_pointer_view_info for the
// offset/length/stride math). Read-only because NumPy's own diagonal() is
// -- callers needing a mutable diagonal are expected to use
// np.fill_diagonal() instead, which mutates the source array directly.
std::optional<exprt>
python_list::try_build_diagonal_pointer_view(const exprt &array, long long k)
{
  if (
    !array.is_symbol() ||
    converter_.numpy_array_symbols_.count(array.identifier().as_string()) == 0)
    return std::nullopt;

  std::optional<diagonal_pointer_view_info> info =
    get_diagonal_pointer_view_info(array, k, converter_.symbol_table());
  if (!info)
    return std::nullopt;

  if (converter_.in_rhs_type_probe_)
  {
    // create_symbol_for_unannotated_assign's type-inference pre-pass
    // converts a fresh `d = np.diagonal(a)`'s RHS once, before the real
    // `d` symbol (and current_lhs) exist, purely to learn its type -- the
    // exprt itself is discarded right after (only .type() is read), and a
    // real, current_lhs-correct second pass follows unconditionally for a
    // Call RHS (no cached-reuse trap the way a Subscript RHS has). Return
    // a same-typed placeholder rather than the actual view so that pass
    // succeeds instead of throwing "not assigned to a variable" for a
    // case that no longer applies by the second pass. Gated on the probe
    // flag specifically (not just current_lhs being unset), because a
    // genuinely nested use such as `np.diagonal(a)[i]` also has current_lhs
    // unset (Subscript nulls it while converting its base) but is never
    // re-converted with a valid target -- there the placeholder's raw,
    // unoffset pointer would silently stand in as the final value instead
    // of the correct strided one, so that case must fall through and
    // decline below instead.
    return build_typecast(
      build_address_of(array), pointer_typet(info->elem_type));
  }

  if (!converter_.current_lhs || !converter_.current_lhs->is_symbol())
    return std::nullopt;

  return build_scalar_pointer_view(
    array,
    info->elem_type,
    info->offset,
    static_cast<std::size_t>(info->length),
    info->stride,
    /*readonly=*/true);
}

// np.trace(a, offset=k): sum of the same elements np.diagonal(a, k) would
// view, without building a view -- a scalar reduction, not aliasing.
std::optional<exprt>
python_list::build_trace_reduction(const exprt &array, long long k)
{
  if (
    !array.is_symbol() ||
    converter_.numpy_array_symbols_.count(array.identifier().as_string()) == 0)
    return std::nullopt;

  std::optional<fixed_2d_shape_info> shape =
    get_fixed_2d_shape_info(array, converter_.symbol_table());
  if (!shape)
    return std::nullopt;

  std::optional<diagonal_pointer_view_info> info =
    get_diagonal_pointer_view_info(array, k, converter_.symbol_table());
  if (!info)
    return std::nullopt;

  // NumPy promotes a bool or narrower-than-default-int element type to the
  // default int accumulator for trace (confirmed against NumPy 2.5.2:
  // trace() of a bool or int8 matrix returns int64, summing without
  // wrapping/truncating), while a float or already-default-width-or-wider
  // int element type keeps its own type.
  typet accumulator_type = shape->elem_type;
  if (
    accumulator_type.is_bool() ||
    ((accumulator_type.is_signedbv() || accumulator_type.is_unsignedbv()) &&
     bv_width(accumulator_type) < bv_width(long_long_int_type())))
    accumulator_type = long_long_int_type();

  if (info->length <= 0)
    return from_integer(0, accumulator_type);

  long long row = k >= 0 ? 0 : -k;
  long long col = k >= 0 ? k : 0;

  exprt sum;
  for (long long i = 0; i < info->length; ++i, ++row, ++col)
  {
    exprt row_expr =
      build_index(array, from_integer(row, size_type()), shape->row_type);
    exprt elem_expr =
      build_index(row_expr, from_integer(col, size_type()), shape->elem_type);
    if (elem_expr.type() != accumulator_type)
      elem_expr = build_typecast(elem_expr, accumulator_type);
    sum = i == 0 ? elem_expr : build_add(sum, elem_expr, accumulator_type);
  }
  return sum;
}

namespace
{
// fill_diagonal accepts a scalar or a 1-D literal of scalars; an array/
// pointer-typed value (e.g. another ndarray passed by name) would otherwise
// be silently reinterpreted as an integer by code_assignt.
void reject_non_scalar_fill_diagonal_value(const typet &value_type)
{
  if (value_type.is_array() || value_type.is_pointer())
    throw std::runtime_error(
      "TypeError: numpy fill_diagonal value must be a scalar or 1-D "
      "literal");
}
} // namespace

// Evaluates value_node once and snapshots it into a fresh temporary, rather
// than letting the caller re-embed the same expression at multiple write
// sites: if it reads the array being mutated (e.g. a[0][0] + a[1][1]),
// re-evaluating it after earlier positions are already overwritten would
// observe partially-mutated state instead of the single, pre-mutation value
// NumPy computes once. Throws if the value isn't scalar-typed.
exprt python_list::snapshot_scalar_value(const nlohmann::json &value_node)
{
  exprt raw_value = converter_.get_expr(value_node);
  reject_non_scalar_fill_diagonal_value(raw_value.type());

  symbolt &tmp = converter_.create_tmp_symbol(
    value_node, "$fill_diagonal_value$", raw_value.type(), exprt());
  locationt location = converter_.get_location_from_decl(value_node);
  code_declt decl(build_symbol(tmp));
  decl.location() = location;
  converter_.add_instruction(decl);
  code_assignt init(build_symbol(tmp), raw_value);
  init.location() = location;
  converter_.add_instruction(init);
  return build_symbol(tmp);
}

// np.fill_diagonal(a, value): mutates a's main diagonal in place, using the
// same offset/stride math as np.diagonal(a)/np.trace(a) with k=0. `value` is
// either a single scalar (snapshotted once and reused for every position) or
// a List literal, repeated (shorter than the diagonal) or truncated (longer)
// to fill it -- confirmed against NumPy 2.5.2: fill_diagonal(a, [10, 20]) on
// a 3-long diagonal writes [10, 20, 10], never raising for a length
// mismatch. Every literal element is still evaluated exactly once (Python
// itself evaluates the whole list literal up front to construct it, whether
// or not every element ends up used) and snapshotted before any position is
// written, both to honour that once-each evaluation and, for elements that
// do repeat, to avoid re-reading an element expression that depends on `a`
// against already-mutated state.
bool python_list::try_build_fill_diagonal_mutation(
  const exprt &array,
  const nlohmann::json &value_node)
{
  if (
    !array.is_symbol() ||
    converter_.numpy_array_symbols_.count(array.identifier().as_string()) == 0)
    return false;

  std::optional<fixed_2d_shape_info> shape =
    get_fixed_2d_shape_info(array, converter_.symbol_table());
  if (!shape)
    return false;

  std::optional<diagonal_pointer_view_info> info =
    get_diagonal_pointer_view_info(array, /*k=*/0, converter_.symbol_table());
  if (!info || info->length <= 0)
    return true;

  const bool value_is_list_literal = value_node.value("_type", "") == "List";
  const nlohmann::json *elts =
    value_is_list_literal && value_node.contains("elts") ? &value_node["elts"]
                                                         : nullptr;
  if (value_is_list_literal && (!elts || elts->empty()))
    return true;

  std::vector<exprt> element_values;
  if (value_is_list_literal)
  {
    element_values.reserve(elts->size());
    for (const auto &elt_node : *elts)
      element_values.push_back(snapshot_scalar_value(elt_node));
  }
  else
    element_values.push_back(snapshot_scalar_value(value_node));

  for (long long i = 0; i < info->length; ++i)
  {
    exprt row_expr =
      build_index(array, from_integer(i, size_type()), shape->row_type);
    exprt elem_lhs =
      build_index(row_expr, from_integer(i, size_type()), shape->elem_type);
    const exprt &value_expr =
      element_values[static_cast<std::size_t>(i) % element_values.size()];
    converter_.add_instruction(code_assignt(elem_lhs, value_expr));
  }
  return true;
}

// np.ravel(a)/a.ravel(): a writable, contiguous pointer view into a's own
// buffer ({offset=0, length=total elements, stride=1}) for a fixed-shape
// 1-D or 2-D array. 3-D+ declines (returns std::nullopt) and falls back to
// the pre-existing copy path (see the shared ravel/flatten/nditer handling
// in numpy_call_expr.cpp, tried only after this declines) -- flatten()
// itself is a distinct dispatch and always stays a copy.
std::optional<exprt>
python_list::try_build_ravel_pointer_view(const exprt &array)
{
  if (
    !array.is_symbol() ||
    converter_.numpy_array_symbols_.count(array.identifier().as_string()) == 0)
    return std::nullopt;

  std::optional<flat_array_shape_info> shape =
    get_flat_1d_or_2d_shape_info(array, converter_.symbol_table());
  if (!shape)
    return std::nullopt;

  if (converter_.in_rhs_type_probe_)
    return build_typecast(
      build_address_of(array), pointer_typet(shape->elem_type));

  if (!converter_.current_lhs || !converter_.current_lhs->is_symbol())
    return std::nullopt;

  return build_scalar_pointer_view(
    array,
    shape->elem_type,
    /*offset=*/0,
    static_cast<std::size_t>(shape->total_length),
    /*stride=*/1,
    /*readonly=*/false);
}

std::optional<exprt> python_list::try_copy_numpy_pointer_view_slice(
  const exprt &array,
  const typet &elem_type,
  const nlohmann::json &slice_node,
  long long step_val,
  bool literal_step)
{
  if (!literal_step || !array.is_symbol())
    return std::nullopt;

  auto info_it =
    converter_.numpy_pointer_view_info_.find(array.identifier().as_string());
  if (info_it == converter_.numpy_pointer_view_info_.end())
    return std::nullopt;
  // Slicing a non-unit-stride view (column, stepped, diagonal) is outside
  // this PR's scope: src_index below assumes the source's own logical
  // indices map 1:1 onto pointer offsets, true only for stride 1. Declining
  // here (instead of rejecting) would fall through to the generic
  // array/list slicing path below, which does not recognise a pointer-typed
  // operand and mishandles it as a runtime list, so reject explicitly.
  if (info_it->second.stride != 1)
    throw std::runtime_error(
      "TypeError: slicing a strided numpy view is not supported");

  long long literal_start = 0;
  std::optional<long long> static_slice_len = literal_slice_length(
    slice_node,
    static_cast<long long>(info_it->second.length),
    step_val,
    &literal_start);
  if (!static_slice_len)
    return std::nullopt;

  array_typet result_type(
    elem_type, from_integer(*static_slice_len, size_type()));
  symbolt &result = converter_.create_tmp_symbol(
    slice_node, "$array_slice$", result_type, exprt());

  code_declt result_decl(build_symbol(result));
  result_decl.location() = converter_.get_location_from_decl(slice_node);
  converter_.add_instruction(result_decl);

  for (long long idx = 0; idx < *static_slice_len; ++idx)
  {
    const long long src_index = literal_start + (idx * step_val);
    exprt src =
      build_index(array, from_integer(src_index, size_type()), elem_type);
    exprt dst = build_index(
      build_symbol(result), from_integer(idx, size_type()), elem_type);
    converter_.add_instruction(code_assignt(dst, src));
  }

  return build_symbol(result);
}

void python_list::emit_slice_zero_step_raise(
  const nlohmann::json &slice_node,
  bool literal_zero_step)
{
  if (!literal_zero_step)
    return;

  exprt raise = converter_.get_exception_handler().gen_exception_raise(
    "ValueError", "slice step cannot be zero");
  codet throw_code("expression");
  throw_code.operands().push_back(raise);
  throw_code.location() = converter_.get_location_from_decl(slice_node);
  converter_.add_instruction(throw_code);
}

// Normalizes a logical (possibly negative) numpy-view index against
// `length`, raises IndexError if still out of bounds after normalizing, and
// scales the result by `stride` (a view's indices are logical (0..length),
// but a non-unit-stride view's pointer offsets into the *source's* flat
// buffer). Shared by guard_numpy_pointer_view_index (a registered view
// symbol's own length/stride) and try_build_flat_index_assignment_target
// (a.flat[i]'s length/stride computed on the fly, with no registered view
// symbol involved).
exprt python_list::normalize_and_scale_index(
  const exprt &index,
  long long length,
  long long stride,
  const nlohmann::json &slice_node)
{
  const typet ll_type = signedbv_typet(64);
  const locationt loc = converter_.get_location_from_decl(slice_node);
  symbolt &idx_sym = converter_.create_tmp_symbol(
    slice_node, "$numpy_view_idx$", ll_type, gen_zero(ll_type));
  code_declt idx_decl(build_symbol(idx_sym));
  idx_decl.location() = loc;
  converter_.add_instruction(idx_decl);

  code_assignt idx_init(build_symbol(idx_sym), build_typecast(index, ll_type));
  idx_init.location() = loc;
  converter_.add_instruction(idx_init);

  exprt view_len = from_integer(length, ll_type);
  exprt idx_lt_zero =
    build_less_than(build_symbol(idx_sym), from_integer(0, ll_type));
  code_assignt normalize(
    build_symbol(idx_sym), build_add(build_symbol(idx_sym), view_len, ll_type));
  normalize.location() = loc;

  code_ifthenelset normalize_guard;
  normalize_guard.cond() = idx_lt_zero;
  normalize_guard.then_case() = normalize;
  normalize_guard.location() = loc;
  normalize_guard.location().property("skipped");
  converter_.add_instruction(normalize_guard);

  const type2tc ll_type2 = migrate_type(ll_type);
  expr2tc idx2, len2;
  migrate_expr(build_symbol(idx_sym), idx2);
  migrate_expr(view_len, len2);
  expr2tc still_negative = lessthan2tc(idx2, gen_zero(ll_type2));
  expr2tc past_end = greaterthanequal2tc(idx2, len2);

  exprt raise = converter_.get_exception_handler().gen_exception_raise(
    "IndexError", "index is out of bounds for numpy view");
  codet throw_code("expression");
  throw_code.operands().push_back(raise);
  throw_code.location() = loc;

  code_ifthenelset oob_guard;
  oob_guard.cond() = migrate_expr_back(or2tc(still_negative, past_end));
  oob_guard.then_case() = throw_code;
  oob_guard.location() = loc;
  oob_guard.location().property("skipped");
  converter_.add_instruction(oob_guard);

  exprt scaled =
    stride == 1
      ? build_symbol(idx_sym)
      : build_mul(
          build_symbol(idx_sym), from_integer(stride, ll_type), ll_type);
  return build_typecast(scaled, size_type());
}

exprt python_list::guard_numpy_pointer_view_index(
  const exprt &array,
  const exprt &index,
  const nlohmann::json &slice_node)
{
  if (!array.is_symbol() || !array.type().is_pointer())
    return index;

  const std::string view_id = array.identifier().as_string();
  auto info_it = converter_.numpy_pointer_view_info_.find(view_id);
  if (info_it == converter_.numpy_pointer_view_info_.end())
    return index;

  return normalize_and_scale_index(
    index, info_it->second.length, info_it->second.stride, slice_node);
}

std::optional<exprt> python_list::try_build_flat_index_assignment_target(
  const exprt &array,
  const nlohmann::json &index_node)
{
  if (
    !array.is_symbol() ||
    converter_.numpy_array_symbols_.count(array.identifier().as_string()) == 0)
    return std::nullopt;

  std::optional<flat_array_shape_info> shape =
    get_flat_1d_or_2d_shape_info(array, converter_.symbol_table());
  if (!shape)
    return std::nullopt;

  exprt index = converter_.get_expr(index_node);
  exprt normalized_idx = normalize_and_scale_index(
    index, shape->total_length, /*stride=*/1, index_node);

  const typet ptr_type = pointer_typet(shape->elem_type);
  exprt base_ptr = build_typecast(build_address_of(array), ptr_type);
  exprt elem_ptr = build_add(base_ptr, normalized_idx, ptr_type);
  return build_dereference(elem_ptr, shape->elem_type);
}

exprt python_list::normalize_negative_slice_bound(
  const nlohmann::json &operand_node,
  const exprt &logical_len,
  bool negative_step)
{
  exprt abs_value = converter_.get_expr(operand_node);
  if (abs_value.type() != size_type())
    abs_value = build_typecast(abs_value, size_type());
  // Called only for "upper" when !negative_step (see handle_range_slice's
  // call sites), so a negative_step here always means this is the "lower"
  // bound of a reverse slice. When abs_value exceeds logical_len, e.g.
  // a[-10:1:-1] on a 3-element array, the clamp must match
  // literal_slice_length's own -1 "before the start" sentinel (there is
  // nothing left to reverse-iterate from), not 0 (the first element) -- 0
  // would be wrong here since it makes the length computation below
  // (lower_expr + 1) count one spurious element instead of zero.
  // Represented as the unsigned size_type()'s two's-complement -1 (its max
  // value) specifically so that + 1 wraps to exactly 0, matching a
  // genuinely empty slice, the same way pointer-offset arithmetic
  // elsewhere in this file relies on wraparound being well-defined for
  // unsigned types. (V.3: built in IREP2.)
  const type2tc size_t2 = migrate_type(size_type());
  expr2tc abs2, len2, converted2;
  migrate_expr(abs_value, abs2);
  migrate_expr(logical_len, len2);
  migrate_expr(build_sub(logical_len, abs_value, size_type()), converted2);
  exprt clamped =
    negative_step ? from_integer(-1, size_type()) : gen_zero(size_type());
  expr2tc clamped2;
  migrate_expr(clamped, clamped2);
  return migrate_expr_back(
    if2tc(size_t2, greaterthan2tc(abs2, len2), clamped2, converted2));
}

/// A dict view (d.keys()[:] / d.values()[:]) is a member expression, so
/// handle_range_slice's element-by-element copy never runs and its id-keyed
/// fallbacks cannot reach it: the entries live under the dict's internal list
/// id. Without this the slice result is untyped and a tuple element reads back
/// as a bare pointer, which unpacking then rejects.
void python_list::copy_dict_view_elem_types(
  const exprt &array,
  const std::string &sliced_id)
{
  if (elem_types().size(sliced_id) != 0 || array.id() != exprt::member)
    return;

  const exprt &dict_sym = array.op0();
  const std::string component =
    to_member_expr(array).get_component_name().as_string();
  if (!dict_sym.is_symbol() || (component != "keys" && component != "values"))
    return;

  const std::string &src = python_dict_handler::get_internal_list_id(
    dict_sym.identifier().as_string(), component == "keys");
  if (!src.empty())
    elem_types().assign_from(src, sliced_id);

  // Tuple values live in the dict_value_types slot rather than the
  // values-list's elements slot, so the copy above is a no-op for them.
  if (component == "values" && elem_types().size(sliced_id) == 0)
  {
    const typet tuple_t =
      converter_.get_dict_handler()->recorded_tuple_value_type(dict_sym);
    if (!tuple_t.is_nil() && !tuple_t.is_empty())
      elem_types().record(sliced_id, std::string(), tuple_t);
  }
}

exprt python_list::handle_range_slice(
  const exprt &array,
  const nlohmann::json &slice_node)
{
  const namespacet ns(converter_.symbol_table());
  const typet list_type = converter_.get_type_handler().get_list_type();
  const typet resolved_array_type = ns.follow(array.type());
  const typet resolved_list_type = ns.follow(list_type);

  // Handle regular array/string slicing (not list slicing)
  // String parameters come as pointer-to-char, so handle both arrays and char
  // pointers
  bool is_string_slice = (resolved_array_type != resolved_list_type &&
                          resolved_array_type.is_array()) ||
                         (resolved_array_type.is_pointer() &&
                          resolved_array_type.subtype() == char_type());

  // Determine step value (default 1).
  const slice_step_info step_info = get_slice_step_info(slice_node);
  const long long step_val = step_info.value;
  // Python raises ValueError on step==0. Raise it from the frontend (a
  // cpp-throw) so try/except ValueError can catch it — a code_assert would be
  // an uncatchable property violation. The raise diverts control, so the rest
  // of the slice IR (kept consistent with step_val==1) is a dead path.
  emit_slice_zero_step_raise(slice_node, step_info.literal_zero);
  bool negative_step = (step_val < 0);

  if (
    std::optional<exprt> slice_copy = try_copy_numpy_pointer_view_slice(
      array,
      resolved_array_type.subtype(),
      slice_node,
      step_val,
      step_info.literal))
    return *slice_copy;

  if (is_string_slice)
  {
    locationt location = converter_.get_location_from_decl(slice_node);

    // For pointer types (function parameters), delegate to __python_str_slice
    // which uses __ESBMC_alloca and survives function returns.
    if (
      resolved_array_type.is_pointer() &&
      resolved_array_type.subtype() == char_type())
    {
      const symbolt &slice_func_ref = get_str_slice_sym();

      // Extract start/end bounds from slice node
      auto get_bound_expr =
        [&](const std::string &name, long long default_val) -> exprt {
        if (!slice_node.contains(name) || slice_node[name].is_null())
          return from_integer(default_val, signedbv_typet(64));

        const auto &bound = slice_node[name];
        if (bound["_type"] == "UnaryOp" && bound["op"]["_type"] == "USub")
        {
          exprt abs_value = converter_.get_expr(bound["operand"]);
          // 0 - (int64)abs_value: both operands are signedbv64 (same width).
          return build_sub(
            from_integer(0, signedbv_typet(64)),
            build_typecast(abs_value, signedbv_typet(64)),
            signedbv_typet(64));
        }
        exprt e = converter_.get_expr(bound);
        e = remove_function_calls_recursive(e, slice_node);
        return build_typecast(e, signedbv_typet(64));
      };

      // Defaults: for positive step start=0,end=MAX; for negative step
      // start=MAX,end=MIN We use large sentinel values; __python_str_slice
      // clamps them
      long long start_default = negative_step ? 999999 : 0;
      long long end_default = negative_step ? -999999 : 999999;

      exprt start_expr = get_bound_expr("lower", start_default);
      exprt end_expr = get_bound_expr("upper", end_default);
      exprt step_expr = from_integer(step_val, signedbv_typet(64));

      // Call __python_str_slice(s, start, end, step) as side-effect expression
      exprt slice_call = build_call_expr(
        slice_func_ref,
        pointer_typet(char_type()),
        {array, start_expr, end_expr, step_expr});
      slice_call.location() = location;

      return slice_call;
    }

    // For array types (local string literals), generate inline loop
    auto to_size_expr = [&](const exprt &expr) -> exprt {
      if (expr.type() == size_type())
        return expr;
      return build_typecast(expr, size_type());
    };
    // V.1k keystone: slice-bound size_type arithmetic built in IREP2. Every
    // call site feeds size_type operands (process_bound, to_size_expr,
    // array_len and logical_len all yield size_type), so operand and result
    // widths match the size_type result — no reconciliation needed.
    auto size_add = [](const exprt &lhs, const exprt &rhs) -> exprt {
      return build_add(lhs, rhs, size_type());
    };
    auto size_sub = [](const exprt &lhs, const exprt &rhs) -> exprt {
      return build_sub(lhs, rhs, size_type());
    };
    auto size_mul = [](const exprt &lhs, const exprt &rhs) -> exprt {
      return build_mul(lhs, rhs, size_type());
    };
    auto size_div = [](const exprt &lhs, const exprt &rhs) -> exprt {
      expr2tc l, r;
      migrate_expr(lhs, l);
      migrate_expr(rhs, r);
      return migrate_expr_back(div2tc(migrate_type(size_type()), l, r));
    };

    // Determine element type and logical length
    typet elem_type;
    exprt array_len;
    exprt logical_len;

    {
      const array_typet &src_type = to_array_type(resolved_array_type);
      elem_type = src_type.subtype();
      array_len = to_size_expr(src_type.size());
      // For char arrays (strings), exclude null terminator from logical length
      logical_len = (elem_type == char_type())
                      ? size_sub(array_len, gen_one(size_type()))
                      : array_len;
    }

    if (!step_info.literal && elem_type != char_type())
      throw std::runtime_error(
        "TypeError: numpy view slicing requires a literal stride");

    // Process slice bounds (handles null, negative indices)
    auto process_bound =
      [&](const std::string &bound_name, const exprt &default_value) -> exprt {
      if (!slice_node.contains(bound_name) || slice_node[bound_name].is_null())
        return to_size_expr(default_value);

      const auto &bound = slice_node[bound_name];

      // Check if negative index
      if (bound["_type"] == "UnaryOp" && bound["op"]["_type"] == "USub")
        return normalize_negative_slice_bound(
          bound["operand"], logical_len, negative_step);

      exprt e = converter_.get_expr(bound);
      return to_size_expr(remove_function_calls_recursive(e, slice_node));
    };

    // Process bounds: defaults depend on step direction
    exprt lower_expr, upper_expr;
    if (negative_step)
    {
      // For negative step: default lower = len-1, default upper = -1 (before 0)
      lower_expr =
        process_bound("lower", size_sub(logical_len, gen_one(size_type())));
    }
    else
    {
      lower_expr = process_bound("lower", gen_zero(size_type()));
    }

    // Upper bound
    if (!negative_step)
      upper_expr = process_bound("upper", logical_len);

    // Clamp bounds to [0, logical_len] to match Python semantics.
    if (!negative_step)
    {
      // bound = (bound >= logical_len) ? logical_len : bound
      // (V.3: built in IREP2.)
      const type2tc size_t2 = migrate_type(size_type());
      expr2tc len2;
      migrate_expr(logical_len, len2);
      auto clamp_to_len = [&](exprt &bound) {
        expr2tc b2;
        migrate_expr(bound, b2);
        bound = migrate_expr_back(
          if2tc(size_t2, greaterthanequal2tc(b2, len2), len2, b2));
      };
      clamp_to_len(lower_expr);
      clamp_to_len(upper_expr);
    }

    // Calculate slice length
    exprt slice_len;
    if (negative_step)
    {
      // For [::-1]: length = lower + 1 (e.g., lower=len-1 → length=len)
      if (
        (!slice_node.contains("lower") || slice_node["lower"].is_null()) &&
        (!slice_node.contains("upper") || slice_node["upper"].is_null()))
      {
        slice_len = logical_len;
      }
      else
      {
        slice_len = size_add(lower_expr, gen_one(size_type()));
      }
    }
    else if (step_val != 1)
    {
      // For step > 1: length = ceil((upper - lower) / step)
      exprt range =
        size_sub(to_size_expr(upper_expr), to_size_expr(lower_expr));
      exprt step_const = from_integer(step_val, size_type());
      exprt step_minus_one = from_integer(step_val - 1, size_type());
      slice_len = size_div(size_add(range, step_minus_one), step_const);
    }
    else
    {
      slice_len = size_sub(to_size_expr(upper_expr), to_size_expr(lower_expr));
    }

    // Char-array slices (strings) keep a trailing null terminator so the
    // strlen-based length and C-string consumers stay correct. Non-char element
    // arrays (e.g. bytes, modelled as wide-int arrays) must NOT carry a phantom
    // null element: len() counts elements there, so an extra slot would
    // over-count. Size the result accordingly.
    const bool needs_null_term = (elem_type == char_type());
    exprt result_size = slice_len;
    if (step_info.literal)
    {
      const array_typet &src_type = to_array_type(resolved_array_type);
      if (!src_type.size().is_nil() && src_type.size().is_constant())
      {
        const long long source_len =
          binary2integer(src_type.size().value().c_str(), false).to_int64() -
          (needs_null_term ? 1 : 0);
        long long literal_start = 0;
        if (
          std::optional<long long> static_slice_len = literal_slice_length(
            slice_node, source_len, step_val, &literal_start))
        {
          result_size = from_integer(*static_slice_len, size_type());
          std::vector<long long> view_shape;
          view_shape.push_back(*static_slice_len);
          append_array_shape(ns.follow(elem_type), view_shape);
          ndarray_descriptor descriptor(
            view_shape, "", numpy_symbol_buffer_id(array));
          descriptor.validate();

          if (
            std::optional<exprt> view_ptr = try_build_1d_pointer_view(
              array,
              elem_type,
              step_val,
              needs_null_term,
              literal_start,
              *static_slice_len))
            return *view_ptr;
        }
      }
    }
    if (needs_null_term)
    {
      // slice_len and the literal are both size_type (built above), so this is
      // a clean IREP2 add (V.1k keystone).
      result_size = build_add(slice_len, gen_one(size_type()), size_type());
    }
    array_typet result_type(elem_type, result_size);

    // Create temporary for sliced array
    symbolt &result = converter_.create_tmp_symbol(
      slice_node, "$array_slice$", result_type, exprt());

    code_declt result_decl(build_symbol(result));
    result_decl.location() = location;
    converter_.add_instruction(result_decl);

    // Create loop: for i = 0; i < slice_len; i++
    symbolt &idx = converter_.create_tmp_symbol(
      slice_node, "$i$", size_type(), gen_zero(size_type()));
    code_assignt idx_init(build_symbol(idx), gen_zero(size_type()));
    converter_.add_instruction(idx_init);

    // idx is size_type; slice_len is built via size_add/sub/div (size_type),
    // so both operands share width.
    exprt cond = build_less_than(build_symbol(idx), slice_len);

    code_blockt body;

    // Compute source index based on step direction
    exprt src_idx;
    if (negative_step)
    {
      // result[i] = array[lower - i] (for step=-1)
      // For other negative steps: result[i] = array[lower - i * |step|]
      if (step_val == -1)
        src_idx = size_sub(to_size_expr(lower_expr), build_symbol(idx));
      else
      {
        exprt abs_step = from_integer(-step_val, size_type());
        src_idx = size_sub(
          to_size_expr(lower_expr), size_mul(build_symbol(idx), abs_step));
      }
    }
    else if (step_val != 1)
    {
      // result[i] = array[lower + i * step]
      exprt step_const = from_integer(step_val, size_type());
      src_idx = size_add(
        to_size_expr(lower_expr), size_mul(build_symbol(idx), step_const));
    }
    else
    {
      // result[i] = array[lower + i]
      src_idx = size_add(to_size_expr(lower_expr), build_symbol(idx));
    }

    exprt src = build_index(array, src_idx, elem_type);
    exprt dst = build_index(build_symbol(result), build_symbol(idx), elem_type);

    // Whole-row assignment (`dst[k] = src[i]` where both sides are
    // themselves arrays, i.e. this is the outer axis of a 2-D array being
    // sliced with a step) is not valid C and is unsupported by the backend,
    // same as build_bool_mask_row_select/build_column_select - copy each
    // row column by column instead. The column count is fixed at
    // conversion time even though the row position is only known at
    // runtime, so the inner copy can be unrolled.
    const typet resolved_elem_type = ns.follow(elem_type);
    if (resolved_elem_type.is_array())
    {
      const array_typet &row_type = to_array_type(resolved_elem_type);
      const typet col_elem_type = ns.follow(row_type.subtype());
      if (col_elem_type.is_array())
      {
        std::ostringstream msg;
        msg << "TypeError: strided slicing currently supports only 2-D "
               "arrays; 3-D+ arrays are not modelled";
        if (!location.is_nil())
          msg << " at " << location.get_file() << ":" << location.get_line();
        throw std::runtime_error(msg.str());
      }
      const BigInt num_cols =
        binary2integer(row_type.size().value().c_str(), false);
      for (BigInt col = 0; col < num_cols; ++col)
      {
        exprt src_elem =
          build_index(src, from_integer(col, size_type()), col_elem_type);
        exprt dst_elem =
          build_index(dst, from_integer(col, size_type()), col_elem_type);
        code_assignt col_assign(dst_elem, src_elem);
        body.copy_to_operands(col_assign);
      }
    }
    else
    {
      code_assignt assign(dst, src);
      body.copy_to_operands(assign);
    }

    // i++ (V.3: synthetic size_type increment, built in IREP2)
    exprt incr =
      build_add(build_symbol(idx), gen_one(size_type()), size_type());
    code_assignt update(build_symbol(idx), incr);
    body.copy_to_operands(update);

    codet loop;
    loop.set_statement("while");
    loop.copy_to_operands(cond, body);
    converter_.add_instruction(loop);

    // Add null terminator at result[slice_len] (char-array slices only).
    if (needs_null_term)
    {
      exprt null_pos = build_index(build_symbol(result), slice_len, elem_type);
      code_assignt add_null(null_pos, gen_zero(elem_type));
      add_null.location() = location;
      converter_.add_instruction(add_null);
    }

    return build_symbol(result);
  }

  // Handle list slicing
  symbolt &sliced_list = create_list();
  const locationt location = converter_.get_location_from_decl(list_value_);

  // Compute list size symbol once. Both bound resolution and (for negative
  // step) the size-1 default reuse this temporary instead of re-emitting
  // __ESBMC_list_size calls.
  const symbolt *size_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  if (!size_func)
    throw std::runtime_error("__ESBMC_list_size not found in symbol table");

  exprt size_call = build_call_expr(
    *size_func,
    size_type(),
    {array.type().is_pointer() ? array : build_address_of(array)});
  size_call.location() = location;

  symbolt &size_sym = converter_.create_tmp_symbol(
    list_value_, "$slice_size$", size_type(), exprt());
  code_declt size_decl(build_symbol(size_sym));
  size_decl.copy_to_operands(size_call);
  converter_.add_instruction(size_decl);

  // Counter type: signed for negative step (must reach -1 sentinel without
  // unsigned underflow); unsigned otherwise to match the existing IR shape
  // for the common step==1 case.
  const typet signed_t = signed_size_type();
  const typet counter_type = negative_step ? signed_t : size_type();
  // V.3: build the slice-bound arithmetic natively in IREP2. Every call site
  // passes operands uniformly at signed_t (size_signed/val_signed via
  // build_typecast, one_s via from_integer), so add2t/sub2t with an explicit
  // signed_t2 result is the exact round-trip of the legacy plus/minus whose
  // result type was overridden to signed_t.
  const type2tc signed_t2 = migrate_type(signed_t);
  auto signed_add = [&](const exprt &lhs, const exprt &rhs) -> exprt {
    expr2tc lhs2, rhs2;
    migrate_expr(lhs, lhs2);
    migrate_expr(rhs, rhs2);
    return migrate_expr_back(add2tc(signed_t2, lhs2, rhs2));
  };
  auto signed_sub = [&](const exprt &lhs, const exprt &rhs) -> exprt {
    expr2tc lhs2, rhs2;
    migrate_expr(lhs, lhs2);
    migrate_expr(rhs, rhs2);
    return migrate_expr_back(sub2tc(signed_t2, lhs2, rhs2));
  };

  // Resolve a slice bound following CPython's slice.indices(len) semantics.
  // Returns an expression of `counter_type`. We always work through signed
  // arithmetic so negative bounds (literal or runtime), the -1 stop sentinel,
  // and the clamp tree stay representable.
  auto resolve_bound = [&](const std::string &name, bool is_upper) -> exprt {
    const exprt size_signed = build_typecast(build_symbol(size_sym), signed_t);
    const exprt zero_s = from_integer(0, signed_t);
    const exprt one_s = from_integer(1, signed_t);

    // Step 1: produce a signed `resolved` value.
    //   - missing bound → step-direction default
    //   - present bound → val (signed); if val < 0 add size at runtime so the
    //     fix also covers non-literal negative bounds like xs[5:b:-1].
    exprt resolved;
    if (!slice_node.contains(name) || slice_node[name].is_null())
    {
      if (negative_step)
        resolved = is_upper ? from_integer(-1, signed_t)
                            : signed_sub(size_signed, one_s);
      else
        resolved = is_upper ? size_signed : zero_s;
    }
    else
    {
      exprt val_signed =
        build_typecast(converter_.get_expr(slice_node[name]), signed_t);
      exprt size_plus_val = signed_add(size_signed, val_signed);
      // val < 0 ? size + val : val  (V.3: built in IREP2).
      expr2tc val2, spv2;
      migrate_expr(val_signed, val2);
      migrate_expr(size_plus_val, spv2);
      resolved = migrate_expr_back(
        if2tc(signed_t2, lessthan2tc(val2, gen_zero(signed_t2)), spv2, val2));
    }

    // Step 2: clamp to the CPython-defined window for this step direction.
    //   pos step: [0, size]                  (start and stop)
    //   neg step start: [-1, size-1]
    //   neg step stop:  [-1, size-1] (stop>=size is harmless — the loop
    //                                terminates immediately — but we clamp
    //                                to keep the expression in a known range)
    const exprt under = negative_step ? from_integer(-1, signed_t) : zero_s;
    const exprt over =
      negative_step ? signed_sub(size_signed, one_s) : size_signed;

    // c1 = max(resolved, under); c2 = min(c1, over)  (V.3: built in IREP2).
    expr2tc resolved2, under2, over2;
    migrate_expr(resolved, resolved2);
    migrate_expr(under, under2);
    migrate_expr(over, over2);
    expr2tc c1 =
      if2tc(signed_t2, lessthan2tc(resolved2, under2), under2, resolved2);
    expr2tc c2 = if2tc(signed_t2, greaterthan2tc(c1, over2), over2, c1);
    exprt c2_legacy = migrate_expr_back(c2);

    return negative_step ? c2_legacy : build_typecast(c2_legacy, size_type());
  };

  exprt lower_expr = resolve_bound("lower", false);
  exprt upper_expr = resolve_bound("upper", true);

  // Initialize counter at lower.
  symbolt &counter = converter_.create_tmp_symbol(
    list_value_, "counter", counter_type, lower_expr);
  code_assignt counter_init(build_symbol(counter), lower_expr);
  converter_.add_instruction(counter_init);

  // Loop condition: counter < upper (positive step) or counter > upper
  // (negative step). Negative step uses signed comparison thanks to the
  // signed counter/upper type.
  exprt loop_condition(negative_step ? ">" : "<", bool_type());
  loop_condition.copy_to_operands(build_symbol(counter), upper_expr);

  // Loop body
  code_blockt loop_body;

  exprt index_expr = build_symbol(counter);
  if (negative_step)
    index_expr = build_typecast(index_expr, size_type());

  const exprt list_at_call = build_list_at_call(array, index_expr, list_value_);
  const symbolt &at_result = converter_.create_tmp_symbol(
    list_value_,
    "tmp_list_at",
    pointer_typet(converter_.get_type_handler().get_list_element_type()),
    exprt());

  code_declt at_decl(build_symbol(at_result));
  at_decl.copy_to_operands(list_at_call);
  loop_body.copy_to_operands(at_decl);

  // Shallow append: preserve element value pointers so nested lists survive the
  // slice copy uncorrupted (esbmc/esbmc#5102).
  const symbolt *push_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push_shallow");
  if (!push_func)
    throw std::runtime_error("Push function symbol not found");

  // Nested-list elements keep their inner pointer; scalars are byte-copied, so
  // the slice copy does not corrupt nested lists (#5102).
  constant_exprt slice_list_type_id(size_type());
  slice_list_type_id.set_value(integer2binary(
    std::hash<std::string>{}(converter_.get_type_handler().type_to_string(
      converter_.get_type_handler().get_list_type())),
    config.ansi_c.address_width));
  // The source list's element width, when every element shares one. Without it
  // the per-element copy length is the symbolic o->size and each copy unwinds
  // memcpy's byte loop to --unwind, which is what made slicing the most
  // expensive list operation (docs/roadmap/symex-dead-work-cost-plan.md W3).
  BigInt slice_elem_size = uniform_scalar_elem_size(array);

  exprt push_call = build_call_expr(
    *push_func,
    bool_type(),
    {build_symbol(sliced_list),
     build_symbol(at_result),
     slice_list_type_id,
     from_integer(slice_elem_size, size_type())});
  push_call.location() = location;
  loop_body.copy_to_operands(converter_.convert_expression_to_code(push_call));

  // counter += step_val. step_val is signed; for positive step it's >0 and
  // converts to size_type cleanly via from_integer.
  exprt step_const = from_integer(step_val, counter_type);
  exprt increment = build_add(build_symbol(counter), step_const, counter_type);
  code_assignt counter_update(build_symbol(counter), increment);
  loop_body.copy_to_operands(counter_update);

  codet while_loop;
  while_loop.set_statement("while");
  while_loop.copy_to_operands(loop_condition, loop_body);
  converter_.add_instruction(while_loop);

  // Update type map for sliced elements. Element types are taken positionally
  // from the source list literal. This is valid when step == 1 and each bound
  // is either absent (defaulting to the full range) or a non-negative constant
  // literal. Absent bounds cover the idiomatic full copy src[:] and half-open
  // slices like src[1:] / src[:2], whose nested-list element types would
  // otherwise be lost — corrupting reads such as sl[0][0] (esbmc/esbmc#5102).
  // For step != 1 the index mapping is no longer lower + i, and for negative or
  // non-constant bounds the static range is unknown; both keep the generic
  // fallback below.
  bool bounds_usable = true;
  bool has_lower = false;
  bool has_upper = false;
  size_t lower_bound = 0;
  size_t upper_bound = 0;
  for (int which = 0; which < 2; ++which)
  {
    const char *key = which == 0 ? "lower" : "upper";
    bool &present = which == 0 ? has_lower : has_upper;

    present = slice_node.contains(key) && !slice_node[key].is_null();
    if (!present)
      continue; // absent / None: defaults to the list bound below
    const nlohmann::json &b = slice_node[key];
    if (
      b.is_object() && b.contains("_type") && b["_type"] == "Constant" &&
      b.contains("value") && b["value"].is_number_integer() &&
      b["value"].get<long long>() >= 0)
    {
      size_t &out = which == 0 ? lower_bound : upper_bound;
      out = b["value"].get<size_t>();
    }
    else
      bounds_usable = false; // negative or non-constant: not known statically
  }

  if (step_val == 1 && bounds_usable)
  {
    nlohmann::json list_node;
    if (
      list_value_.contains("value") && list_value_["value"].is_object() &&
      list_value_["value"].contains("id") &&
      list_value_["value"]["id"].is_string())
    {
      list_node = json_utils::get_var_value(
        list_value_["value"]["id"],
        converter_.current_function_name(),
        converter_.ast());
    }

    // Only update type map for actual lists (not strings or other types)
    if (
      !list_node.is_null() && list_node.contains("value") &&
      list_node["value"].contains("elts") &&
      list_node["value"]["elts"].is_array())
    {
      const size_t elts_size = list_node["value"]["elts"].size();
      const size_t begin = has_lower ? std::min(lower_bound, elts_size) : 0;
      const size_t end =
        has_upper ? std::min(upper_bound, elts_size) : elts_size;

      for (size_t i = begin; i < end; ++i)
      {
        exprt element = converter_.get_expr(list_node["value"]["elts"][i]);
        if (!element.is_symbol())
        {
          symbolt &tmp = converter_.create_tmp_symbol(
            list_value_, "$slice_elem$", element.type(), element);
          code_declt decl(build_symbol(tmp));
          decl.location() = location;
          converter_.add_instruction(decl);

          code_assignt assign(build_symbol(tmp), element);
          assign.location() = location;
          converter_.add_instruction(assign);
          element = build_symbol(tmp);
        }

        elem_types().record(
          sliced_list.id.as_string(),
          element.identifier().as_string(),
          element.type());
      }
    }
  }

  // This handles cases where one or both bounds are null or negative (e.g.
  // numbers[:-1]), or where the source is a function parameter rather than a
  // locally constructed list, so the registry has no entries for it.
  const std::string &sliced_id = sliced_list.id.as_string();

  copy_dict_view_elem_types(array, sliced_id);

  if (elem_types().size(sliced_id) == 0)
  {
    if (list_value_.contains("value") && list_value_["value"].contains("id"))
    {
      const std::string &param_name =
        list_value_["value"]["id"].get<std::string>();

      nlohmann::json param_node = json_utils::find_var_decl(
        param_name, converter_.current_function_name(), converter_.ast());

      // Only use annotation fallback for function parameters (arg nodes),
      // not for local variable declarations whose element type should come
      // from the type map populated during list construction.
      if (!param_node.is_null() && param_node["_type"] == "arg")
      {
        const typet elem_type = get_elem_type_from_annotation(
          param_node, converter_.get_type_handler());

        if (elem_type != typet())
          elem_types().record(sliced_id, std::string(), elem_type);
      }
    }
  }

  return build_symbol(sliced_list);
}

void python_list::handle_slice_assignment(
  const exprt &list_expr,
  const nlohmann::json &slice_node,
  const nlohmann::json &value_node)
{
  // The step is restricted to a constant integer literal (or absent) for
  // now; the model receives it as a runtime value and branches on it at
  // solve time, so this could be lifted, but a symbolic step is untested.
  // Mirrors the literal-only step extraction in handle_range_slice, but
  // rejects a non-constant step instead of silently slicing with step 1.
  long long step_val = 1;
  if (slice_node.contains("step") && !slice_node["step"].is_null())
  {
    const auto &step_node = slice_node["step"];
    if (
      step_node["_type"] == "UnaryOp" && step_node["op"]["_type"] == "USub" &&
      step_node["operand"]["_type"] == "Constant" &&
      step_node["operand"]["value"].is_number_integer())
      step_val = -(long long)step_node["operand"]["value"].get<std::int64_t>();
    else if (
      step_node["_type"] == "Constant" &&
      step_node["value"].is_number_integer())
      step_val = step_node["value"].get<std::int64_t>();
    else
      throw std::runtime_error(
        "List slice assignment requires a constant integer step");
  }

  const locationt location = converter_.get_location_from_decl(slice_node);

  // Reverse-in-place idiom: `l[a:b] = reversed(l[a:b])`. The literal form emits
  // three heap passes (slice-read copy, reversed() rebuild, slice-assign
  // resize+copy); collapse them into a single in-place swap over the sub-range.
  // Fire ONLY when it is provably the same list variable and the same slice on
  // both sides, with step 1 — then reversing the read slice and storing it back
  // is exactly an in-place reverse of [a:b]. Any mismatch falls through to the
  // correct (slower) generic path, so a missed match is harmless and a wrong
  // match is impossible.
  //
  // `reversed` is matched by name, consistent with the rest of the frontend
  // (annotation_intrinsics.cpp, loop_mixin.py, expr.cpp all key on the builtin
  // name); a user-shadowed `reversed` is not honoured anywhere, so this adds no
  // new assumption. Bounds are evaluated once here vs twice in the three-step
  // form — immaterial for the pure index expressions this fires on.
  if (
    step_val == 1 && value_node.is_object() &&
    value_node.value("_type", "") == "Call" && value_node.contains("func") &&
    value_node["func"].value("_type", "") == "Name" &&
    value_node["func"].value("id", "") == "reversed" &&
    value_node.contains("args") && value_node["args"].is_array() &&
    value_node["args"].size() == 1)
  {
    const nlohmann::json &arg = value_node["args"][0];
    const bool arg_is_same_slice =
      arg.is_object() && arg.value("_type", "") == "Subscript" &&
      arg.contains("slice") && arg["slice"].is_object() &&
      arg["slice"].value("_type", "") == "Slice" &&
      // Same list variable (Name) on both sides.
      arg.contains("value") && arg["value"].is_object() &&
      arg["value"].value("_type", "") == "Name" &&
      list_value_.contains("value") && list_value_["value"].is_object() &&
      list_value_["value"].value("_type", "") == "Name" &&
      arg["value"].value("id", "") == list_value_["value"].value("id", "") &&
      // Same slice bounds (ignoring source locations).
      ast_equal_ignoring_location(arg["slice"], slice_node);

    if (arg_is_same_slice)
    {
      const symbolt *rev_fn = converter_.symbol_table().find_symbol(
        "c:@F@__ESBMC_list_reverse_range");
      if (rev_fn)
      {
        const typet i64r = signedbv_typet(64);
        auto rev_bound = [&](const char *name, bool &present) -> exprt {
          present = slice_node.contains(name) && !slice_node[name].is_null();
          if (!present)
            return from_integer(0, i64r);
          exprt e = converter_.get_expr(slice_node[name]);
          e = remove_function_calls_recursive(e, slice_node);
          return build_typecast(e, i64r);
        };
        bool rlo = false, rup = false;
        exprt rlower = rev_bound("lower", rlo);
        exprt rupper = rev_bound("upper", rup);

        exprt rev_call = build_call_expr(
          *rev_fn,
          empty_typet(),
          {list_expr.type().is_pointer() ? list_expr
                                         : build_address_of(list_expr),
           rlower,
           from_integer(rlo ? 1 : 0, int_type()),
           rupper,
           from_integer(rup ? 1 : 0, int_type())});
        rev_call.location() = location;
        converter_.add_instruction(
          converter_.convert_expression_to_code(rev_call));
        return;
      }
    }
  }

  // Evaluate the right-hand side; materialize a call (e.g. sorted(...)) into
  // a temporary so its result can be passed to the model by address. get_expr
  // returns calls as code_function_callt, which must be emitted as an
  // instruction with the temporary as its lhs — embedding it as a decl
  // operand would leak a side effect into the GOTO program.
  exprt rhs = converter_.get_expr(value_node);
  if (rhs.is_function_call())
  {
    code_function_callt &fcall = static_cast<code_function_callt &>(rhs);
    if (!fcall.function().type().is_code())
      throw std::runtime_error(
        "Unsupported callable on list slice assignment right-hand side");
    const typet ret_type = to_code_type(fcall.function().type()).return_type();
    symbolt &tmp = converter_.create_tmp_symbol(
      value_node, "$slice_assign_rhs$", ret_type, exprt());
    code_declt decl(build_symbol(tmp));
    decl.location() = location;
    converter_.add_instruction(decl);
    fcall.lhs() = build_symbol(tmp);
    converter_.add_instruction(fcall);
    rhs = build_symbol(tmp);
  }
  else if (rhs.id() == "sideeffect")
  {
    symbolt &tmp = converter_.create_tmp_symbol(
      value_node, "$slice_assign_rhs$", rhs.type(), exprt());
    code_declt decl(build_symbol(tmp));
    decl.copy_to_operands(rhs);
    decl.location() = location;
    converter_.add_instruction(decl);
    rhs = build_symbol(tmp);
  }

  const namespacet ns(converter_.symbol_table());
  const typet list_type = converter_.get_type_handler().get_list_type();
  const typet resolved_list_type = ns.follow(list_type);
  const typet rhs_type = ns.follow(rhs.type());
  const bool rhs_is_list =
    rhs_type == resolved_list_type ||
    (rhs_type.is_pointer() &&
     ns.follow(rhs_type.subtype()) == resolved_list_type);
  if (!rhs_is_list)
    throw std::runtime_error(
      "List slice assignment requires a list right-hand side");

  const typet i64 = signedbv_typet(64);
  auto bound_expr = [&](const char *name, bool &present) -> exprt {
    present = slice_node.contains(name) && !slice_node[name].is_null();
    if (!present)
      return from_integer(0, i64);
    exprt e = converter_.get_expr(slice_node[name]);
    e = remove_function_calls_recursive(e, slice_node);
    return build_typecast(e, i64);
  };
  bool has_lower = false, has_upper = false;
  exprt lower = bound_expr("lower", has_lower);
  exprt upper = bound_expr("upper", has_upper);

  const symbolt *fn =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_slice_assign");
  if (!fn)
    throw std::runtime_error(
      "__ESBMC_list_slice_assign not found in symbol table");

  // Same nested-list type id as the slice-read path: lets the model keep
  // nested-list records shared instead of byte-copying them (#5102).
  constant_exprt list_type_id(size_type());
  list_type_id.set_value(integer2binary(
    std::hash<std::string>{}(
      converter_.get_type_handler().type_to_string(list_type)),
    config.ansi_c.address_width));

  // Statically-known element byte size, passed to the model so its per-element
  // copy uses a compile-time-constant size and takes __ESBMC_copy_value's
  // branch-free fast path, instead of reading the symbolic-index field
  // src->items[k].size (which forces memcpy's per-byte loop to unwind per
  // element — the dominant cost on large slice assignments). Emitted ONLY for
  // an unambiguous fixed-width scalar element type; 0 otherwise, which makes
  // the model fall back to the per-element o->size read (exact prior behaviour,
  // so a missing/ambiguous type can never produce a wrong copy length).
  // Derive the element byte size from the RHS source list's elements so the
  // model copies with a compile-time-constant size (fast path) instead of the
  // symbolic-index field read src->items[k].size. We read the *converted* RHS
  // element type, which is the actual data being stored, so the size is always
  // correct. Only a plain fixed-width scalar (bitvector) is emitted; anything
  // else (pointers, structs, strings, nested lists, non-literal/heterogeneous
  // RHS) keeps elem_size 0, making the model fall back to the per-element
  // o->size read — exact prior behaviour, so a wrong length is impossible.
  auto scalar_width = [](const typet &t) -> size_t {
    if (
      (t.id() == "signedbv" || t.id() == "unsignedbv" || t.id() == "floatbv" ||
       t.id() == "fixedbv") &&
      !t.width().empty())
      return std::stoull(t.width().as_string(), nullptr, 10) / 8;
    return 0;
  };
  size_t elem_size_bytes = 0;
  if (
    value_node.contains("_type") && value_node["_type"] == "List" &&
    value_node.contains("elts") && value_node["elts"].is_array() &&
    !value_node["elts"].empty())
  {
    // RHS is a list literal: size from its (uniform) element types.
    bool uniform = true;
    size_t w = 0;
    for (const auto &elt : value_node["elts"])
    {
      size_t ew = 0;
      try
      {
        ew = scalar_width(converter_.get_expr(elt).type());
      }
      catch (...)
      {
        uniform = false;
        break;
      }
      if (ew == 0 || (w != 0 && ew != w))
      {
        uniform = false;
        break;
      }
      w = ew;
    }
    if (uniform)
      elem_size_bytes = w;
  }
  else if (
    value_node.contains("_type") && value_node["_type"] == "Name" &&
    value_node.contains("id"))
  {
    // RHS is a list variable (includes self-assignment l[...] = l): size from
    // its declared element type.
    const typet et =
      elem_types().element_type(value_node["id"].get<std::string>());
    if (!et.is_nil())
      elem_size_bytes = scalar_width(et);
  }
  constant_exprt elem_size(size_type());
  elem_size.set_value(
    integer2binary(BigInt(elem_size_bytes), config.ansi_c.address_width));

  exprt call = build_call_expr(
    *fn,
    bool_type(),
    {list_expr.type().is_pointer() ? list_expr : build_address_of(list_expr),
     lower,
     from_integer(has_lower ? 1 : 0, int_type()),
     upper,
     from_integer(has_upper ? 1 : 0, int_type()),
     from_integer(step_val, i64),
     rhs.type().is_pointer() ? rhs : build_address_of(rhs),
     list_type_id,
     elem_size});
  call.location() = location;
  converter_.add_instruction(converter_.convert_expression_to_code(call));
}

/// A bare `list` annotation names no element type, so the read would carry
/// none and neither arithmetic nor equality on it would behave. Recover what
/// the call sites told us about this parameter, seeded when the function was
/// converted (#7187). Returns `annotated` unchanged when this is not a
/// bare-`list` parameter, or nothing was inferred for it.
typet python_list::bare_list_param_elem_type(
  const nlohmann::json &param_node,
  const std::string &param_id,
  const typet &annotated)
{
  if (
    !param_node.contains("annotation") || !param_node["annotation"].is_object())
    return annotated;

  const nlohmann::json &annotation = param_node["annotation"];
  const std::string ann_id = annotation.value("id", "");
  if (
    annotation.value("_type", "") != "Name" ||
    (ann_id != "list" && ann_id != "List"))
    return annotated;

  const typet inferred = elem_types().element_type(param_id, 0);
  return inferred.is_empty() ? annotated : inferred;
}

std::optional<exprt> python_list::resolve_nested_list_element(
  const exprt &array,
  const exprt &pos_expr,
  size_t index,
  typet &elem_type)
{
  // Check for nested list access
  if (array.type() == converter_.get_type_handler().get_list_type())
  {
    const auto &key = array.identifier().as_string();
    if (const auto *recorded = elem_types().find(key))
    {
      // Homogeneous lists (e.g. list comprehensions) record a single
      // element-type entry; ESBMC models lists as homogeneous, so reuse
      // that entry for any in-structure index.  Without this, a constant
      // outer index >= 1 into a comprehension-built nested list skips the
      // nested-element handling below, the inner element type (float) is
      // lost, and the value reaches the SMT FP encoder as a non-FP sort
      // (get_exponent_width abort / Z3 "rm and fp sorts", #5129).  Literal
      // lists record one entry per element, so for an in-bounds index
      // eff_index == index and behaviour is unchanged.  The runtime value
      // is still read at pos_expr below, so only the static type is taken
      // from the homogeneous entry.
      const size_t eff_index =
        index < recorded->size() ? index : recorded->size() - 1;
      const std::string &elem_id = recorded->at(eff_index).first;
      elem_type = recorded->at(eff_index).second;

      if (elem_type == converter_.get_type_handler().get_list_type())
      {
        // Nested-list element.  The recorded elem_id names the inner-list
        // symbol, but assign_from copies that id verbatim across a
        // function-return boundary (Q = build()), where it is a *callee*
        // frame local — returning build_symbol(elem_id) would reference a
        // symbol that is never assigned in the caller's symex frame, so its
        // value is nondet (float_buf OOB / wrong value, #5103/#5102).
        //
        // Instead, read the inner list pointer at runtime from `array`
        // (valid in every scope: list_at returns the stored pointer, so
        // aliasing/mutation semantics are preserved), bind it to a fresh
        // caller-scope symbol, and copy the inner element-type map onto
        // that symbol so deeper subscripts (Q[i][j]) still resolve
        // int/float.  For parameter-annotation-only entries (elem_id
        // empty) fall through to the dynamic __ESBMC_list_at path below.
        if (!elem_id.empty())
        {
          exprt list_at_call = build_list_at_call(array, pos_expr, list_value_);
          exprt inner_ptr = extract_pyobject_value(list_at_call, elem_type);

          // As a store target the subscript must stay an lvalue into the
          // slot: binding it to the temporary below sends `xs[0] = xs[1]`
          // into a dead local and leaves xs untouched, so len(xs[0]) keeps
          // answering the overwritten element's length (#7360). Reads
          // nested inside a target still take the temp path.
          if (converter_.is_store_target(list_value_))
            return inner_ptr;

          const locationt loc = converter_.get_location_from_decl(list_value_);

          symbolt &inner_sym = converter_.create_tmp_symbol(
            list_value_, "$nested_list$", elem_type, exprt());
          code_declt inner_decl(build_symbol(inner_sym));
          inner_decl.location() = loc;
          converter_.add_instruction(inner_decl);

          code_assignt inner_assign(build_symbol(inner_sym), inner_ptr);
          inner_assign.location() = loc;
          converter_.add_instruction(inner_assign);

          elem_types().assign_from(elem_id, inner_sym.id.as_string());
          return build_symbol(inner_sym);
        }
      }
    }
  }
  return std::nullopt;
}

exprt python_list::handle_index_access(
  const exprt &array,
  const nlohmann::json &slice_node)
{
  const namespacet ns(converter_.symbol_table());
  const typet resolved_array_type = ns.follow(array.type());

  // Find list node for type information
  nlohmann::json list_node;
  if (
    list_value_.contains("value") && list_value_["value"].is_object() &&
    list_value_["value"].contains("id"))
  {
    list_node = json_utils::find_var_decl(
      list_value_["value"]["id"],
      converter_.current_function_name(),
      converter_.ast());
  }

  exprt pos_expr = converter_.get_expr(slice_node);
  pos_expr = converter_.unwrap_optional_if_needed(pos_expr);
  size_t index = 0;

  // Validate index type
  if (pos_expr.type().is_array())
  {
    locationt l = converter_.get_location_from_decl(list_value_);
    throw std::runtime_error(
      "TypeError at " + l.get_file().as_string() + " " +
      l.get_line().as_string() +
      ": list indices must be integers or slices, not str");
  }

  // Handle negative indices
  if (slice_node.contains("op") && slice_node["op"]["_type"] == "USub")
  {
    // Both compile-time branches below assume the negated operand is a
    // constant literal (a[-1]). For a non-constant operand (a[-i]) the value
    // is only known at runtime, so leave pos_expr (= -i) and index untouched:
    // build_list_at_call normalizes the negative index at runtime via
    // __ESBMC_list_size, and the element-type lookup falls back to element 0,
    // which is correct for the homogeneous lists ESBMC models (#4926).
    const bool operand_is_constant =
      slice_node.contains("operand") &&
      slice_node["operand"]["_type"] == "Constant" &&
      slice_node["operand"].contains("value");

    if (!operand_is_constant)
    {
      // Nothing to do: runtime normalization handles a[-i].
    }
    // For char* (string parameters), skip compile-time normalization: the size
    // is not known statically, so normalization happens at runtime in the
    // char* indexing block below.
    else if (
      !array.type().is_pointer() &&
      (list_node.is_null() || list_node["value"]["_type"] != "List"))
    {
      BigInt v = binary2integer(pos_expr.op0().value().c_str(), true);
      v *= -1;

      const array_typet &t = static_cast<const array_typet &>(array.type());
      BigInt s = binary2integer(t.size().value().c_str(), true);

      // For char arrays (strings), exclude null terminator from logical length
      if (t.subtype() == char_type())
        s -= 1;

      v += s;
      pos_expr = from_integer(v, pos_expr.type());
    }
    else
    {
      // Compute index for compile-time type lookup only.
      // Do NOT overwrite pos_expr: the list may have been mutated
      // (append/insert/extend), so we must resolve the negative index
      // at runtime via build_list_at_call using __ESBMC_list_size.
      index = slice_node["operand"]["value"].get<size_t>();
      index = list_node["value"]["elts"].size() - index;
    }
  }
  else if (slice_node["_type"] == "Constant")
  {
    index = slice_node["value"].get<size_t>();
  }

  // Handle different array types
  const bool is_char_array = resolved_array_type.is_array() &&
                             resolved_array_type.subtype() == char_type();
  const bool is_char_ptr = resolved_array_type.is_pointer() &&
                           resolved_array_type.subtype() == char_type();

  if (
    (array.type().is_symbol() || array.type().subtype().is_symbol()) &&
    !is_char_array && !is_char_ptr)
  {
    // Handle list types (symbol-based)
    typet elem_type;

    if (
      std::optional<exprt> nested =
        resolve_nested_list_element(array, pos_expr, index, elem_type))
      return *nested;

    // Determine element type
    if (list_node.contains("_type") && list_node["_type"] == "arg")
    {
      elem_type =
        get_elem_type_from_annotation(list_node, converter_.get_type_handler());

      elem_type = bare_list_param_elem_type(
        list_node, array.identifier().as_string(), elem_type);
    }
    else if (
      slice_node["_type"] == "Constant" || slice_node["_type"] == "BinOp" ||
      (slice_node["_type"] == "UnaryOp" &&
       slice_node["operand"]["_type"] == "Constant"))
    {
      const std::string &list_name = array.identifier().as_string();

      const auto *recorded = elem_types().find(list_name);
      if (!recorded)
      {
        /* Fall back to annotation for function parameters */
        if (
          list_value_.contains("value") && list_value_["value"].is_object() &&
          list_value_["value"].contains("id") &&
          list_value_["value"]["id"].is_string())
        {
          const nlohmann::json list_value_node = json_utils::get_var_value(
            list_value_["value"]["id"],
            converter_.current_function_name(),
            converter_.ast());

          elem_type = get_elem_type_from_annotation(
            list_value_node, converter_.get_type_handler());
        }
      }
      else
      {
        size_t type_index =
          (!list_node.is_null() && list_node.contains("value") &&
           list_node["value"].is_object() &&
           list_node["value"].contains("_type") &&
           list_node["value"]["_type"] == "BinOp")
            ? 0
            : index;

        if (type_index < recorded->size())
          elem_type = (*recorded)[type_index].second;
        else
        {
          // Do not emit a frontend conversion error for a constant OOB index.
          // The runtime list access model can raise IndexError, which is
          // observable by Python try/except code. Use the known element type
          // for homogeneous dynamic lists.
          elem_type = recorded->back().second;

          if (elem_type == typet())
          {
            const locationt l = converter_.get_location_from_decl(list_value_);
            throw std::runtime_error(
              "List out of bounds at " + l.get_file().as_string() +
              " line: " + l.get_line().as_string());
          }
        }
      }
    }
    else if (slice_node["_type"] == "Name")
    {
      // First try to get element type from list_node if it's an AnnAssign
      if (
        !list_node.is_null() && list_node["_type"] == "AnnAssign" &&
        list_node.contains("annotation") && elem_type == typet())
      {
        elem_type = get_elem_type_from_annotation(
          list_node, converter_.get_type_handler());
      }

      // If still no elem_type, try to get it from the array variable's type
      // annotation
      if (array.is_symbol() && elem_type == typet())
      {
        // Extract variable name from the symbol identifier
        std::string list_var_name = json_utils::extract_var_name_from_symbol_id(
          array.identifier().as_string());

        // Find the variable's declaration to check for type annotation
        nlohmann::json list_var_decl = json_utils::find_var_decl(
          list_var_name, converter_.current_function_name(), converter_.ast());

        // If the variable has a type annotation such as list[str], extract
        // element type
        if (!list_var_decl.is_null() && list_var_decl.contains("annotation"))
        {
          elem_type = get_elem_type_from_annotation(
            list_var_decl, converter_.get_type_handler());
        }
      }

      // Handle variable-based indexing
      if (
        elem_type == typet() && !list_node.is_null() &&
        list_node.contains("value") && list_node["value"].is_object() &&
        list_node["value"].contains("_type") &&
        list_node["value"]["_type"] == "Call")
      {
        elem_type =
          infer_elem_type_from_call_return(list_node["value"], converter_);
        if (elem_type != typet() && array.is_symbol())
        {
          const std::string &list_name = array.identifier().as_string();
          if (elem_types().size(list_name) == 0)
            elem_types().record(list_name, "", elem_type);
        }
      }

      if (!list_node.is_null() && list_node["_type"] == "arg")
      {
        elem_type = get_elem_type_from_annotation(
          list_node, converter_.get_type_handler());
      }
      else
      {
        // Handle case where we need to find the variable declaration
        while (!list_node.is_null() && (!list_node.contains("value") ||
                                        !list_node["value"].contains("elts") ||
                                        !list_node["value"]["elts"].is_array()))
        {
          if (list_node.contains("value") && list_node["value"].contains("id"))
            list_node = json_utils::find_var_decl(
              list_node["value"]["id"],
              converter_.current_function_name(),
              converter_.ast());
          else
          {
            break;
          }
        }

        if (!list_node.is_null() && list_node["_type"] == "arg")
        {
          elem_type = get_elem_type_from_annotation(
            list_node, converter_.get_type_handler());
        }
        else if (!list_node.is_null() && list_node.contains("value"))
        {
          // Check if the value is a Subscript (such as d['a'])
          if (
            list_node["value"].contains("_type") &&
            list_node["value"]["_type"] == "Subscript")
          {
            // For ESBMC_iter_0 = d['a'], get element type from dict's actual
            // value
            if (
              list_node["value"].contains("value") &&
              list_node["value"]["value"].is_object() &&
              list_node["value"]["value"].contains("_type") &&
              list_node["value"]["value"]["_type"] == "Name")
            {
              std::string dict_var_name =
                list_node["value"]["value"]["id"].get<std::string>();

              // Find the dict's declaration
              nlohmann::json dict_node = json_utils::find_var_decl(
                dict_var_name,
                converter_.current_function_name(),
                converter_.ast());

              if (!dict_node.is_null() && dict_node.contains("value"))
              {
                const auto &dict_value = dict_node["value"];

                // Get the key being accessed (e.g., 'a' in d['a'])
                if (list_node["value"].contains("slice"))
                {
                  const auto &key_node = list_node["value"]["slice"];

                  // Handle a constant key of any JSON scalar type
                  // (string, int, bool); comparing the JSON values directly
                  // avoids get<std::string>() throwing on an int key (d[1]).
                  if (
                    key_node["_type"] == "Constant" &&
                    key_node.contains("value"))
                  {
                    const auto &key_val = key_node["value"];

                    // For dict literals, get the corresponding value
                    if (
                      dict_value["_type"] == "Dict" &&
                      dict_value.contains("keys") &&
                      dict_value.contains("values"))
                    {
                      const auto &keys = dict_value["keys"];
                      const auto &values = dict_value["values"];

                      // Find the matching key
                      for (size_t i = 0; i < keys.size(); i++)
                      {
                        if (
                          keys[i]["_type"] == "Constant" &&
                          keys[i].contains("value") &&
                          keys[i]["value"] == key_val)
                        {
                          // Found the value: now get its element type
                          // (promotion-aware, see infer_literal_element_type).
                          elem_type = infer_literal_element_type(values[i]);
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          else if (
            list_node["value"].contains("elts") &&
            list_node["value"]["elts"].is_array() &&
            !list_node["value"]["elts"].empty())
          {
            // Infer element type from the literal, accounting for the
            // int->float promotion applied to mixed numeric literals at
            // construction.
            elem_type = infer_literal_element_type(list_node["value"]);
          }
        }
      }
    }

    // For variable indices, prefer compile-time list element type information
    // when available.  Also fire when elem_type is empty_typet(): that is the
    // sentinel returned by get_typet("tuple"), meaning "don't use the
    // annotation as-is; resolve the concrete struct type via the registry."
    if ((elem_type == typet() || elem_type.is_empty()) && array.is_symbol())
    {
      const std::string &list_name = array.identifier().as_string();
      if (const auto *recorded = elem_types().find(list_name))
        elem_type = recorded->back().second;
    }

    // Nested subscript chains like W[i][j] where the base is a Name with a
    // nested list annotation (e.g. list[list[T]]).  list_node is null in this
    // case because list_value_["value"] is itself a Subscript rather than a
    // Name, so the standard paths above can't see W's annotation.  Walk the
    // Subscript chain down to the base Name and peel that many layers off the
    // annotation, then hand the result to the existing extractor.
    if (elem_type == typet())
    {
      size_t subscript_depth = 0;
      const nlohmann::json *cur = &list_value_;
      while (cur->is_object() && cur->contains("value") &&
             (*cur)["value"].is_object() && (*cur)["value"].contains("_type") &&
             (*cur)["value"]["_type"] == "Subscript")
      {
        ++subscript_depth;
        cur = &(*cur)["value"];
      }
      if (
        subscript_depth > 0 && cur->is_object() && cur->contains("value") &&
        (*cur)["value"].is_object() && (*cur)["value"].contains("id") &&
        (*cur)["value"]["id"].is_string())
      {
        nlohmann::json base_decl = json_utils::find_var_decl(
          (*cur)["value"]["id"].get<std::string>(),
          converter_.current_function_name(),
          converter_.ast());
        if (!base_decl.is_null() && base_decl.contains("annotation"))
        {
          nlohmann::json drilled = base_decl["annotation"];
          for (size_t k = 0; k < subscript_depth; ++k)
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
            elem_type = get_elem_type_from_annotation(
              synth, converter_.get_type_handler());
          }
        }
      }
    }

    // Python allows indexing lists with unknown element type in dynamic code.
    // If we resolved the index but not the element type, treat elements as Any
    // instead of raising a frontend conversion error.
    if (pos_expr != exprt() && elem_type == typet())
      elem_type = any_type();

    if (pos_expr == exprt() || elem_type == typet())
    {
      const bool has_const_index =
        (slice_node["_type"] == "Constant" && slice_node.contains("value")) ||
        (slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
         slice_node["op"]["_type"] == "USub" &&
         slice_node.contains("operand") &&
         slice_node["operand"]["_type"] == "Constant");
      if (has_const_index)
      {
        // If we can prove out-of-bounds from a concrete list literal, raise
        // IndexError in-model so Python try/except(IndexError) can catch it.
        if (
          array.is_symbol() && !list_node.is_null() &&
          list_node.contains("value") && list_node["value"].is_object() &&
          list_node["value"].contains("elts") &&
          list_node["value"]["elts"].is_array())
        {
          bool negative_index = false;
          size_t index_abs = 0;

          if (slice_node["_type"] == "Constant" && slice_node.contains("value"))
          {
            index_abs = slice_node["value"].get<size_t>();
          }
          else
          {
            negative_index = true;
            index_abs = slice_node["operand"]["value"].get<size_t>();
          }

          const size_t list_size = list_node["value"]["elts"].size();
          const bool out_of_bounds =
            (!negative_index && index_abs >= list_size) ||
            (negative_index && index_abs > list_size);

          if (out_of_bounds)
          {
            exprt raise =
              converter_.get_exception_handler().gen_exception_raise(
                "IndexError", "list index out of range");
            codet throw_code("expression");
            throw_code.operands().push_back(raise);
            converter_.add_instruction(throw_code);
            // Unreachable after raise, but return a placeholder expression.
            return gen_zero(size_type());
          }
        }

        // If array is a constant placeholder (e.g., from a chained OOB access),
        // we're in dead code after a prior IndexError. Emit IndexError and
        // return a placeholder rather than crashing the frontend.
        if (!array.is_symbol() && array.is_constant())
        {
          exprt raise = converter_.get_exception_handler().gen_exception_raise(
            "IndexError", "list index out of range");
          codet throw_code("expression");
          throw_code.operands().push_back(raise);
          converter_.add_instruction(throw_code);
          return gen_zero(size_type());
        }

        const locationt l = converter_.get_location_from_decl(list_value_);
        throw std::runtime_error(
          "List out of bounds at " + l.get_file().as_string() +
          " line: " + l.get_line().as_string());
      }

      // Keep historical frontend diagnostic for literal lists with
      // compile-time constant OOB indexing (including nested accesses).
      if (
        array.is_symbol() && !list_node.is_null() &&
        list_node.contains("value") && list_node["value"].is_object() &&
        list_node["value"].contains("elts") &&
        list_node["value"]["elts"].is_array())
      {
        bool has_const_index = false;
        bool negative_index = false;
        size_t index_abs = 0;

        if (slice_node["_type"] == "Constant" && slice_node.contains("value"))
        {
          has_const_index = true;
          index_abs = slice_node["value"].get<size_t>();
        }
        else if (
          slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
          slice_node["op"]["_type"] == "USub" &&
          slice_node.contains("operand") &&
          slice_node["operand"]["_type"] == "Constant")
        {
          has_const_index = true;
          negative_index = true;
          index_abs = slice_node["operand"]["value"].get<size_t>();
        }

        if (has_const_index)
        {
          const size_t list_size = list_node["value"]["elts"].size();
          const bool out_of_bounds =
            (!negative_index && index_abs >= list_size) ||
            (negative_index && index_abs > list_size);
          if (out_of_bounds)
          {
            const locationt l = converter_.get_location_from_decl(list_value_);
            throw std::runtime_error(
              "List out of bounds at " + l.get_file().as_string() +
              " line: " + l.get_line().as_string());
          }
        }
      }

      throw std::runtime_error(
        "Invalid list access: could not resolve position or element type");
    }

    // Preserve historical frontend OOB diagnostics only when we can prove we
    // are indexing a stable list literal (not a reassigned/mutated/derived
    // list). Using the type map alone is too broad and causes false positives.
    // Emit an IndexError raise instead of a C++ exception so Python
    // try/except(IndexError) can observe the error.
    if (array.is_symbol())
    {
      const std::string &list_name = array.identifier().as_string();
      if (const auto *recorded = elem_types().find(list_name))
      {
        bool has_const_index = false;
        bool negative_index = false;
        size_t index_abs = 0;

        if (slice_node["_type"] == "Constant" && slice_node.contains("value"))
        {
          has_const_index = true;
          index_abs = slice_node["value"].get<size_t>();
        }
        else if (
          slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
          slice_node["op"]["_type"] == "USub" &&
          slice_node.contains("operand") &&
          slice_node["operand"]["_type"] == "Constant")
        {
          has_const_index = true;
          negative_index = true;
          index_abs = slice_node["operand"]["value"].get<size_t>();
        }

        const bool is_slice_derived_var =
          !list_node.is_null() && list_node.contains("value") &&
          list_node["value"].contains("_type") &&
          list_node["value"]["_type"] == "Subscript";

        bool is_stable_list_literal = false;
        if (
          !list_node.is_null() && list_node.contains("value") &&
          list_node["value"].contains("_type") &&
          list_node["value"]["_type"] == "List" &&
          list_node["value"].contains("elts") &&
          list_node["value"]["elts"].is_array())
        {
          // If sizes diverge, the symbol was likely reassigned/mutated.
          is_stable_list_literal =
            recorded->size() == list_node["value"]["elts"].size();
        }

        // A list that escaped into a function/method call may have been grown
        // by the callee, which this static length does not reflect; skip the
        // convert-time bounds check so the runtime __ESBMC_list_at path handles
        // it (GitHub #5991).
        if (
          has_const_index && !is_slice_derived_var && is_stable_list_literal &&
          !converter_.is_list_call_escaped(list_name))
        {
          const size_t known_size = recorded->size();
          const bool oob = negative_index ? (index_abs > known_size)
                                          : (index_abs >= known_size);
          if (oob)
          {
            exprt raise =
              converter_.get_exception_handler().gen_exception_raise(
                "IndexError", "list index out of range");
            codet throw_code("expression");
            throw_code.operands().push_back(raise);
            converter_.add_instruction(throw_code);
            // Short-circuit: the raise makes the rest unreachable.
            return gen_zero(elem_type);
          }
        }
      }
    }

    // A non-constant subscript into a heterogeneous int/float list cannot
    // resolve a single static element type from the index, so the code above
    // defaults to element 0's type. That misreads any element of the other
    // numeric kind (e.g. a float stored at index 2 read as an int, #5160).
    // Python promotes int to float in numeric expressions, so treat the read
    // as float and dispatch on the stored type_id at runtime (see
    // extract_pyobject_value), which yields the correct value for both kinds.
    const bool constant_index =
      slice_node["_type"] == "Constant" ||
      (slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
       slice_node["op"]["_type"] == "USub" && slice_node.contains("operand") &&
       slice_node["operand"]["_type"] == "Constant");
    const bool mixed_numeric =
      !constant_index && array.is_symbol() &&
      elem_types().has_mixed_numeric(array.identifier().as_string());
    if (mixed_numeric)
      elem_type = double_type();

    // A float-typed element read must dispatch on the stored type_id even for a
    // constant index into a statically "pure-float" list: a list[float]
    // parameter can receive a list whose elements are actually int (Python does
    // not enforce annotations), and reading __ESBMC_float_buf unconditionally
    // would reinterpret those int payloads as float garbage (math.dist with
    // integer coordinates returned ~0). The type_id dispatch promotes int
    // payloads correctly, so widen the dispatch to every float read.
    const bool dispatch_numeric = mixed_numeric || elem_type.is_floatbv();

    // A heap-migrated user-class instance is stored in a list as a `Class*`
    // pointer: the push path records a pointer-sized element (type_size 8), not
    // the full struct. If the element type resolved to a by-value user-class
    // struct, read it back as the pointer it actually is so
    // extract_pyobject_value dereferences a single `Class*` instead of copying
    // sizeof(struct) bytes off an 8-byte pointer slot, which overruns it
    // (#4805). ESBMC-internal model helper classes (reserved `__ESBMC_` prefix,
    // e.g. the dataclasses `__ESBMC_DataclassField`) are stored by value by
    // their hand-written models and must be left as structs.
    if (converter_.is_user_class_struct_type(elem_type))
    {
      const std::string tag =
        elem_type.id() == "symbol"
          ? to_symbol_type(elem_type).get_identifier().as_string()
          : to_struct_type(elem_type).tag().as_string();
      const std::string cls = converter_.extract_class_name_from_tag(tag);
      if (cls.rfind("__ESBMC", 0) != 0)
        elem_type = gen_pointer_type(elem_type);
    }

    // Build list access and cast result
    exprt list_at_call = build_list_at_call(array, pos_expr, list_value_);

    // The mixed-numeric read dereferences the element three times (type_id,
    // float_idx, value), so bind the __ESBMC_list_at result to a temp first and
    // evaluate the access once instead of three times on the hot path.
    if (dispatch_numeric)
    {
      const locationt loc = converter_.get_location_from_decl(list_value_);
      symbolt &elem_obj = converter_.create_tmp_symbol(
        list_value_,
        "$list_elem_obj$",
        pointer_typet(converter_.get_type_handler().get_list_element_type()),
        exprt());
      code_declt obj_decl(build_symbol(elem_obj));
      obj_decl.location() = loc;
      converter_.add_instruction(obj_decl);
      code_assignt obj_assign(build_symbol(elem_obj), list_at_call);
      obj_assign.location() = loc;
      converter_.add_instruction(obj_assign);
      list_at_call = build_symbol(elem_obj);
    }

    // Extract and dereference PyObject value. string_safe: a subscript read is
    // value-context (assigned to the target), so an any_type string element may
    // be returned by pointer without overrunning its short char[] storage.
    return extract_pyobject_value(
      list_at_call, elem_type, dispatch_numeric, /*string_safe=*/true);
  }

  // Handle static string indexing with IndexError on out-of-bounds
  if (is_char_array)
  {
    exprt idx = pos_expr;
    if (idx.type() != size_type())
      idx = build_typecast(idx, size_type());

    // Logical string length excludes the null terminator
    exprt array_size = to_array_type(resolved_array_type).size();
    if (array_size.type() != size_type())
      array_size = build_typecast(array_size, size_type());
    // str_len = array_size - 1. array_size is typecast to size_type above, so
    // both operands share width; build the subtraction in IREP2 (V.3).
    exprt str_len =
      build_sub(array_size, from_integer(1, size_type()), size_type());

    // Emit: if (idx >= str_len) throw IndexError("string index out of range")
    // (idx is typecast to size_type above; str_len is a size_type subtraction —
    // both operands share width.)
    exprt oob_cond = build_greater_equal(idx, str_len);

    exprt raise = converter_.get_exception_handler().gen_exception_raise(
      "IndexError", "string index out of range");
    codet throw_code("expression");
    throw_code.operands().push_back(raise);

    code_ifthenelset guard;
    guard.cond() = oob_cond;
    guard.then_case() = throw_code;
    converter_.add_instruction(guard);

    // Tag with #cpp_type==char so downstream consumers (notably
    // python_converter::get_python_type_category) can distinguish a 1-char
    // string element from an arbitrary 8-bit int (e.g. dtype=np.int8) without
    // resorting to a fragile width-only heuristic.
    typet char_t = char_type();
    type_utils::set_cpp_type(char_t, "char");
    return build_index(array, idx, char_t);
  }

  // For char* (string function parameter), implement Python single-index
  // semantics: normalize negative indices, raise IndexError on out-of-bounds,
  // then return a fresh single-char null-terminated string via
  // __python_str_slice so the result is never a dangling pointer.
  if (is_char_ptr)
  {
    typet ll_type = signedbv_typet(64);
    locationt loc = converter_.get_location_from_decl(list_value_);

    // --- 1. strlen(array) → len_sym ---
    const symbolt *strlen_sym =
      converter_.symbol_table().find_symbol("c:@F@strlen");
    if (!strlen_sym)
      throw std::runtime_error("strlen not found in symbol table");

    symbolt &len_sym = converter_.create_tmp_symbol(
      list_value_, "$str_len$", size_type(), gen_zero(size_type()));
    code_declt len_decl(build_symbol(len_sym));
    len_decl.location() = loc;
    converter_.add_instruction(len_decl);

    code_function_callt strlen_call;
    strlen_call.function() = build_symbol(*strlen_sym);
    strlen_call.lhs() = build_symbol(len_sym);
    strlen_call.arguments().push_back(array);
    strlen_call.type() = size_type();
    strlen_call.location() = loc;
    converter_.add_instruction(strlen_call);

    // --- 2. idx_sym = (long long)pos_expr ---
    symbolt &idx_sym = converter_.create_tmp_symbol(
      list_value_, "$str_idx$", ll_type, gen_zero(ll_type));
    code_declt idx_decl(build_symbol(idx_sym));
    idx_decl.location() = loc;
    converter_.add_instruction(idx_decl);

    code_assignt idx_init(
      build_symbol(idx_sym), build_typecast(pos_expr, ll_type));
    idx_init.location() = loc;
    converter_.add_instruction(idx_init);

    // --- 3. Normalize negative index: if (idx < 0) idx += (ll)len ---
    // ($str_idx$ and the 0 literal are both ll_type — same width)
    exprt idx_lt_zero =
      build_less_than(build_symbol(idx_sym), from_integer(0, ll_type));

    exprt idx_plus_len = build_add(
      build_symbol(idx_sym),
      build_typecast(build_symbol(len_sym), ll_type),
      ll_type);

    code_assignt normalize(build_symbol(idx_sym), idx_plus_len);
    normalize.location() = loc;

    code_ifthenelset norm_guard;
    norm_guard.cond() = idx_lt_zero;
    norm_guard.then_case() = normalize;
    norm_guard.location() = loc;
    norm_guard.location().property("skipped");
    converter_.add_instruction(norm_guard);

    // --- 4. OOB check: if (idx < 0 || idx >= (ll)len) raise IndexError ---
    // V.3: built in IREP2, back-migrated for the legacy code_ifthenelset.
    const type2tc oob_llt = migrate_type(ll_type);
    expr2tc oob_idx, oob_len;
    migrate_expr(build_symbol(idx_sym), oob_idx);
    migrate_expr(build_typecast(build_symbol(len_sym), ll_type), oob_len);
    expr2tc still_neg = lessthan2tc(oob_idx, gen_zero(oob_llt));
    expr2tc idx_ge_len = greaterthanequal2tc(oob_idx, oob_len);

    exprt raise = converter_.get_exception_handler().gen_exception_raise(
      "IndexError", "string index out of range");
    codet throw_code("expression");
    throw_code.operands().push_back(raise);

    code_ifthenelset oob_guard;
    oob_guard.cond() = migrate_expr_back(or2tc(still_neg, idx_ge_len));
    oob_guard.then_case() = throw_code;
    oob_guard.location() = loc;
    oob_guard.location().property("skipped");
    converter_.add_instruction(oob_guard);

    // --- 5. __python_str_slice(array, idx, idx+1, 1) ---
    // idx is now a normalized, in-bounds positive index; the slice helper
    // produces a fresh alloca'd single-char null-terminated string.
    exprt end_expr =
      build_add(build_symbol(idx_sym), from_integer(1, ll_type), ll_type);

    exprt slice_call = build_call_expr(
      get_str_slice_sym(),
      gen_pointer_type(char_type()),
      {array, build_symbol(idx_sym), end_expr, from_integer(1, ll_type)});
    slice_call.location() = loc;
    return slice_call;
  }

  // Handle static arrays
  exprt guarded_pos =
    guard_numpy_pointer_view_index(array, pos_expr, slice_node);
  return try_build_row_pointer_view(array, slice_node)
    .value_or(build_index(array, guarded_pos, array.type().subtype()));
}

exprt python_list::extract_pyobject_value(
  const exprt &pyobject_expr,
  const typet &elem_type,
  bool mixed_numeric,
  bool string_safe)
{
  // For float types, read __ESBMC_float_buf[item->float_idx].
  // This avoids the void*→integer truncation in --ir mode: float_idx is a
  // size_t (no sort mismatch in BV mode), and float_buf is a typed global
  // double array (real-sorted in --ir mode), so the array read gives the
  // correct real value.
  if (elem_type.is_floatbv())
  {
    // Helper: build (*pyobject_expr).field for a given field/type.
    auto member_of = [&](const char *field, const typet &ftype) -> exprt {
      return build_deref_member(pyobject_expr, field, ftype);
    };

    // Look up __ESBMC_float_buf global (static in list.c, but still in symbol
    // table)
    const symbolt *fbuf_sym =
      converter_.symbol_table().find_symbol("c:list.c@__ESBMC_float_buf");
    assert(fbuf_sym && "could not find __ESBMC_float_buf symbol");

    // Build __ESBMC_float_buf[item->float_idx]
    exprt float_idx_member = member_of("float_idx", size_type());
    exprt float_val =
      build_index(build_symbol(*fbuf_sym), float_idx_member, elem_type);

    if (!mixed_numeric)
      return float_val;

    // Dynamic index into a heterogeneous int/float list: the element may be an
    // int whose value lives behind item->value (not in float_buf), so reading
    // float_buf unconditionally would misread it (#5160). Dispatch on the
    // stored type_id: float elements come from float_buf, int elements are
    // promoted to double from their 8-byte payload. The float type_id is the
    // same hash the push path stamps onto float elements (float_type_id).
    const size_t float_type_id = std::hash<std::string>{}(
      converter_.get_type_handler().type_to_string(elem_type));

    // (double)*(long long *)item->value
    exprt value_member = member_of("value", pointer_typet(empty_typet()));
    exprt as_int_ptr =
      build_typecast(value_member, pointer_typet(long_long_int_type()));
    exprt int_val = build_dereference(as_int_ptr, long_long_int_type());
    exprt int_as_float = build_typecast(int_val, elem_type);

    // item->type_id == float_type_id ? float_buf[float_idx] : (double)int
    // V.3: built in IREP2 (both branches are elem_type, so the if2t types
    // agree), back-migrated at the return.
    const type2tc et2 = migrate_type(elem_type);
    expr2tc tid2, fv2, iaf2;
    migrate_expr(member_of("type_id", size_type()), tid2);
    migrate_expr(float_val, fv2);
    migrate_expr(int_as_float, iaf2);
    const expr2tc is_float =
      equality2tc(tid2, from_integer(float_type_id, migrate_type(size_type())));
    return migrate_expr_back(if2tc(et2, is_float, fv2, iaf2));
  }

  // Extract value from PyObject: (*pyobject_expr).value
  exprt obj_value =
    build_deref_member(pyobject_expr, "value", pointer_typet(empty_typet()));

  // Element whose static type was erased to any_type() (void*): this happens
  // when a dict/list crosses into an *unannotated* parameter, where keys()/
  // values() reads have no compile-time element type (#5444). The generic
  // "cast void* to T* and dereference" path below overruns a string element: a
  // str is stored as a short char[] at item->value (e.g. a 2-byte key), so an
  // 8-byte void* dereference reads past the array (array-bounds violation). The
  // element's *runtime* type is recorded in item->type_id (stamped at push
  // time), so when it marks a string, keep the stored pointer (== char*, the
  // same value the annotated str path returns) with no dereference. Every other
  // type keeps the proven dereference read, so numeric elements are unaffected.
  // The dispatch key is the caller-stamped type, not a usage heuristic, so this
  // stays sound. Gated on string_safe because the result is an if-expression
  // (rvalue): callers that take its address (membership/find on heaps, queues,
  // sets) need the plain dereference lvalue, so only value-context reads
  // (subscript) opt in.
  if (
    string_safe && elem_type.is_pointer() &&
    elem_type.subtype().id() == "empty")
  {
    const size_t str_type_id = std::hash<std::string>{}(
      converter_.get_type_handler().type_to_string(pointer_typet(char_type())));
    exprt type_id_member =
      build_deref_member(pyobject_expr, "type_id", size_type());
    // Default: cast void* to void** and dereference (the read that already
    // works for numeric / pointer-by-value elements).
    exprt as_default = build_dereference(
      build_typecast(obj_value, pointer_typet(elem_type)), elem_type);
    // item->type_id == str_type_id ? obj_value : *(T*)obj_value
    // V.3: built in IREP2 (both branches are elem_type/void*, so the if2t
    // types agree), back-migrated at the return. Mirrors the float-dispatch
    // if2t above.
    const type2tc et2 = migrate_type(elem_type);
    expr2tc tid2, ov2, def2;
    migrate_expr(type_id_member, tid2);
    migrate_expr(obj_value, ov2);
    migrate_expr(as_default, def2);
    const expr2tc is_str =
      equality2tc(tid2, from_integer(str_type_id, migrate_type(size_type())));
    return migrate_expr_back(if2tc(et2, is_str, ov2, def2));
  }

  // For array types, return pointer to element type instead of pointer to array
  // The dereference system doesn't support array types as target types
  if (elem_type.is_array())
  {
    const array_typet &arr_type = to_array_type(elem_type);
    // Cast to pointer to element type (e.g., char* instead of char[2]*)
    return build_typecast(obj_value, pointer_typet(arr_type.subtype()));
  }

  // For char* strings and None (_Bool*), the void* already contains the pointer
  // value. For all other types, the void* contains a pointer to the value.
  if (
    elem_type.is_pointer() &&
    (elem_type.subtype() == char_type() || elem_type.subtype() == bool_type()))
  {
    // String and None case: cast void* directly to the pointer type (no
    // dereference needed)
    return build_typecast(obj_value, elem_type);
  }
  else
  {
    // All other types: cast void* to pointer-to-type, then dereference
    exprt tc = build_typecast(obj_value, pointer_typet(elem_type));
    return build_dereference(tc, elem_type);
  }
}
