#include <cassert>
#include <complex>
#include <functional>
#include <goto-symex/execution_state.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/printf_formatter.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <irep2/irep2.h>
#include <util/message.h>
#include <util/message/format.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/std_types.h>
#include <vector>
#include <algorithm>
#include <util/array2string.h>

expr2tc goto_symext::symex_malloc(
  const expr2tc &lhs,
  const sideeffect2t &code,
  const guardt &guard)
{
  return symex_mem(true, lhs, code, guard);
}

expr2tc goto_symext::symex_alloca(
  const expr2tc &lhs,
  const sideeffect2t &code,
  const guardt &guard)
{
  return symex_mem(false, lhs, code, guard);
}

expr2tc goto_symext::create_dynamic_memory_symbol(
  const type2tc &elem_type,
  const expr2tc &size_expr,
  const std::string &name_prefix)
{
  unsigned int &dynamic_counter = get_dynamic_counter();
  dynamic_counter++;

  symbolt symbol;
  symbol.name = name_prefix + "_" + i2string(dynamic_counter) + "_array";
  symbol.id = std::string("symex_dynamic::") + id2string(symbol.name);
  symbol.lvalue = true;
  symbol.mode = "C";

  typet renamedtype = ns.follow(migrate_type_back(elem_type));
  symbol.type = typet(typet::t_array);
  symbol.type.subtype() = renamedtype;
  symbol.type.size(migrate_expr_back(size_expr));
  symbol.type.dynamic(true);
  symbol.type.set(
    "alignment", constant_exprt(config.ansi_c.max_alignment(), size_type()));

  new_context.add(symbol);
  type2tc new_type = migrate_type(symbol.type);
  return symbol2tc(new_type, symbol.id);
}

void goto_symext::copy_memory_content(
  const expr2tc &old_base_array,
  const expr2tc &new_array,
  const expr2tc &old_elem_count,
  const expr2tc &new_elem_count,
  const type2tc &elem_type,
  bool old_is_array,
  const guardt &guard)
{
  if (
    is_nil_expr(old_base_array) || is_nil_expr(old_elem_count) ||
    is_nil_expr(new_elem_count))
    return;

  type2tc new_elem_type = to_array_type(new_array->type).subtype;

  expr2tc copy_count = if2tc(
    size_type2(),
    lessthan2tc(old_elem_count, new_elem_count),
    old_elem_count,
    new_elem_count);

  // default value
  uint64_t max_symbolic_copy = 128;
  std::string option_value = options.get_option("max-symbolic-realloc-copy");
  if (!option_value.empty())
    max_symbolic_copy = std::stoull(option_value);

  if (is_constant_int2t(copy_count))
  {
    uint64_t const_copy_count = to_constant_int2t(copy_count).value.to_uint64();
    uint64_t actual_copy_count = std::min(const_copy_count, max_symbolic_copy);

    for (uint64_t i = 0; i < actual_copy_count; i++)
    {
      expr2tc idx = constant_int2tc(size_type2(), BigInt(i));
      copy_single_element(
        old_base_array,
        new_array,
        idx,
        elem_type,
        new_elem_type,
        old_is_array,
        guard);
    }
  }
  else
  {
    for (uint64_t i = 0; i < max_symbolic_copy; i++)
    {
      expr2tc idx = constant_int2tc(size_type2(), BigInt(i));
      expr2tc should_copy = lessthan2tc(idx, copy_count);
      guardt copy_guard = guard;
      copy_guard.add(should_copy);

      if (!copy_guard.is_false())
        copy_single_element(
          old_base_array,
          new_array,
          idx,
          elem_type,
          new_elem_type,
          old_is_array,
          copy_guard);
    }
  }
}

void goto_symext::copy_single_element(
  const expr2tc &old_base_array,
  const expr2tc &new_array,
  const expr2tc &idx,
  const type2tc &elem_type,
  const type2tc &new_elem_type,
  bool old_is_array,
  const guardt &guard)
{
  expr2tc old_elem =
    old_is_array ? index2tc(elem_type, old_base_array, idx) : old_base_array;
  expr2tc new_elem = index2tc(new_elem_type, new_array, idx);

  cur_state->rename(old_elem);
  symex_assign(code_assign2tc(new_elem, old_elem), false, guard);
}

void goto_symext::symex_realloc(
  const expr2tc &lhs,
  const sideeffect2t &code,
  const guardt &guard)
{
  expr2tc src_ptr = code.operand;
  expr2tc realloc_size = code.size; // This is in bytes
  cur_state->rename(realloc_size);

  // ===== handle reallocC(ptr, 0) - free and return NULL =====
  if (handle_realloc_zero_size(lhs, code, guard, realloc_size))
    return;

  // ===== determine element type and old object info =====
  type2tc elem_type;
  expr2tc old_base_array;
  bool old_is_array = false;
  expr2tc old_elem_count;

  if (!analyze_old_object(
        src_ptr, elem_type, old_base_array, old_is_array, old_elem_count))
  {
    // Fallback element type determination
    elem_type = determine_fallback_element_type(code, lhs);
  }

  // calculate new element count
  expr2tc elem_size = type_byte_size_expr(elem_type);
  cur_state->rename(elem_size);
  do_simplify(elem_size);

  expr2tc new_elem_count = calculate_element_count(realloc_size, elem_size);

  // allocate new memory
  expr2tc new_array =
    create_dynamic_memory_symbol(elem_type, realloc_size, "realloc");

  // copy data
  copy_memory_content(
    old_base_array,
    new_array,
    old_elem_count,
    new_elem_count,
    elem_type,
    old_is_array,
    guard);

  // create result and handle failure modelling
  expr2tc result = create_result_pointer(new_array, lhs->type);
  result = model_allocation_failure(result, code.operand, guard);

  // finalize assignment and tracking
  finalize_realloc_result(lhs, result, new_array, guard, realloc_size);
}

bool goto_symext::handle_realloc_zero_size(
  const expr2tc &lhs,
  const sideeffect2t &code,
  const guardt &guard,
  const expr2tc &realloc_size)
{
  expr2tc zero_size = gen_zero(realloc_size->type);
  expr2tc is_zero_size = equality2tc(realloc_size, zero_size);
  do_simplify(is_zero_size);

  if (is_true(is_zero_size))
  {
    symex_free(code_free2tc(code.operand));
    expr2tc null_ptr = gen_zero(lhs->type);
    symex_assign(code_assign2tc(lhs, null_ptr), true, guard);
    return true;
  }
  return false;
}

bool goto_symext::analyze_old_object(
  const expr2tc &src_ptr,
  type2tc &elem_type,
  expr2tc &old_base_array,
  bool &old_is_array,
  expr2tc &old_elem_count)
{
  internal_deref_items.clear();
  expr2tc deref = dereference2tc(get_uint8_type(), src_ptr);
  dereference(deref, dereferencet::INTERNAL);

  if (internal_deref_items.empty())
    return false;

  expr2tc old_obj = internal_deref_items.front().object;

  // Determine element type and base array from old object
  if (is_index2t(old_obj))
  {
    old_base_array = to_index2t(old_obj).source_value;
    old_is_array = is_array_type(old_base_array->type);
    elem_type = old_is_array ? to_array_type(old_base_array->type).subtype
                             : old_base_array->type;
  }
  else if (is_array_type(old_obj->type))
  {
    old_base_array = old_obj;
    old_is_array = true;
    elem_type = to_array_type(old_obj->type).subtype;
  }
  else
  {
    old_base_array = old_obj;
    old_is_array = false;
    elem_type = old_obj->type;
  }

  // Calculate old element count
  old_elem_count =
    calculate_old_element_count(old_base_array, elem_type, old_is_array);

  return true;
}

type2tc goto_symext::determine_fallback_element_type(
  const sideeffect2t &code,
  const expr2tc &lhs)
{
  if (!is_nil_type(code.alloctype) && !is_empty_type(code.alloctype))
    return code.alloctype;
  else if (is_pointer_type(lhs->type))
  {
    type2tc subtype = to_pointer_type(lhs->type).subtype;
    if (is_empty_type(subtype))
      return get_uint8_type();
    return subtype;
  }
  else
    return get_uint8_type();
}

expr2tc goto_symext::calculate_element_count(
  const expr2tc &size_bytes,
  const expr2tc &elem_size)
{
  if (
    is_constant_int2t(elem_size) &&
    to_constant_int2t(elem_size).value.to_uint64() > 0)
  {
    expr2tc count = div2tc(size_type2(), size_bytes, elem_size);
    cur_state->rename(count);
    do_simplify(count);
    return count;
  }
  return expr2tc(); // nil expr for invalid cases
}

expr2tc goto_symext::calculate_old_element_count(
  const expr2tc &old_base_array,
  const type2tc &elem_type,
  bool old_is_array)
{
  if (old_is_array && is_array_type(old_base_array->type))
  {
    const array_type2t &arr_type = to_array_type(old_base_array->type);
    if (!is_nil_expr(arr_type.array_size))
    {
      expr2tc size_bytes = arr_type.array_size;
      cur_state->rename(size_bytes);
      do_simplify(size_bytes);

      expr2tc elem_size = type_byte_size_expr(elem_type);
      cur_state->rename(elem_size);
      do_simplify(elem_size);

      return calculate_element_count(size_bytes, elem_size);
    }
  }
  else if (!old_is_array)
  {
    return constant_int2tc(size_type2(), BigInt(1));
  }

  return expr2tc(); // nil expr for unhandled cases
}

expr2tc goto_symext::create_result_pointer(
  const expr2tc &new_array,
  const type2tc &lhs_type)
{
  type2tc new_elem_type = to_array_type(new_array->type).subtype;
  expr2tc idx_val = gen_long(size_type2(), 0L);
  expr2tc idx = index2tc(new_elem_type, new_array, idx_val);
  expr2tc result = address_of2tc(new_elem_type, idx);

  if (result->type != lhs_type)
    result = typecast2tc(lhs_type, result);

  cur_state->rename(result);
  return result;
}

expr2tc goto_symext::model_allocation_failure(
  const expr2tc &result,
  const expr2tc &old_ptr,
  const guardt &guard)
{
  if (!options.get_bool_option("force-realloc-success"))
  {
    expr2tc alloc_fail = sideeffect2tc(
      get_bool_type(),
      expr2tc(),
      expr2tc(),
      std::vector<expr2tc>(),
      type2tc(),
      sideeffect2t::nondet);
    replace_nondet(alloc_fail);

    expr2tc null_ptr = symbol2tc(result->type, "NULL");
    expr2tc conditional_result =
      if2tc(result->type, alloc_fail, null_ptr, result);

    // Update validity array conditionally
    update_pointer_validity(old_ptr, alloc_fail, guard);

    return conditional_result;
  }
  else
  {
    // Always free old pointer when forced success
    symex_free(code_free2tc(old_ptr));
  }

  return result;
}

void goto_symext::update_pointer_validity(
  const expr2tc &old_ptr,
  const expr2tc &alloc_fail,
  const guardt &guard)
{
  expr2tc old_ptr_obj = pointer_object2tc(pointer_type2(), old_ptr);
  dereference(old_ptr_obj, dereferencet::READ);

  type2tc sym_type = array_type2tc(get_bool_type(), expr2tc(), true);
  expr2tc valid_sym = symbol2tc(sym_type, valid_ptr_arr_name);
  expr2tc valid_index_expr = index2tc(get_bool_type(), valid_sym, old_ptr_obj);

  // If realloc fails (alloc_fail=true), keep old pointer valid (true)
  // If realloc succeeds (alloc_fail=false), invalidate old pointer (false)
  expr2tc new_validity =
    if2tc(get_bool_type(), alloc_fail, gen_true_expr(), gen_false_expr());
  symex_assign(code_assign2tc(valid_index_expr, new_validity), true, guard);
}

void goto_symext::finalize_realloc_result(
  const expr2tc &lhs,
  const expr2tc &result,
  const expr2tc &new_array,
  const guardt &guard,
  const expr2tc &realloc_size)
{
  expr2tc result_copy(result);

  // Assign result to lhs
  symex_assign(code_assign2tc(lhs, result), true, guard);

  // Track the new pointer
  expr2tc ptr_obj = pointer_object2tc(pointer_type2(), result);
  track_new_pointer(ptr_obj, new_array->type, guard, realloc_size);

  // Add to dynamic memory tracking
  guardt alloc_guard = cur_state->guard;
  alloc_guard.append(guard);

  unsigned int dynamic_counter = get_dynamic_counter();
  std::string symbol_name = "dynamic_" + i2string(dynamic_counter) + "_array";
  dynamic_memory.emplace_back(result_copy, alloc_guard, false, symbol_name);
}

expr2tc goto_symext::symex_mem_inf(
  const expr2tc &lhs,
  const type2tc &base_type,
  const guardt &guard)
{
  if (is_nil_expr(lhs))
    return expr2tc(); // ignore

  // size
  type2tc type = base_type;

  assert(!is_nil_type(base_type));
  unsigned int &dynamic_counter = get_dynamic_counter();
  dynamic_counter++;

  // value
  symbolt symbol;

  symbol.name = "dynamic_" + i2string(dynamic_counter) + "_inf_array";

  symbol.id = std::string("symex_dynamic::") + id2string(symbol.name);
  symbol.lvalue = true;

  typet renamedtype = ns.follow(migrate_type_back(type));

  symbol.type = array_typet(renamedtype, exprt("infinity", size_type()));
  symbol.type.dynamic(true);
  symbol.type.set(
    "alignment", constant_exprt(config.ansi_c.max_alignment(), size_type()));
  symbol.mode = "C";
  new_context.add(symbol);

  type2tc new_type = migrate_type(symbol.type);

  type2tc rhs_type;
  expr2tc rhs_ptr_obj;

  type2tc subtype = migrate_type(symbol.type.subtype());
  expr2tc sym = symbol2tc(new_type, symbol.id);
  expr2tc idx_val = gen_long(size_type2(), 0L);
  expr2tc idx = index2tc(subtype, sym, idx_val);
  do_simplify(idx);
  rhs_type = migrate_type(symbol.type.subtype());
  rhs_ptr_obj = idx;

  expr2tc rhs_addrof = address_of2tc(rhs_type, rhs_ptr_obj);
  do_simplify(rhs_addrof);
  expr2tc rhs = rhs_addrof;
  expr2tc ptr_rhs = rhs;
  guardt alloc_guard = cur_state->guard;

  if (rhs->type != lhs->type)
    rhs = typecast2tc(lhs->type, rhs);

  cur_state->rename(rhs);
  expr2tc rhs_copy(rhs);

  symex_assign(code_assign2tc(lhs, rhs), true, guard);

  expr2tc ptr_obj = pointer_object2tc(pointer_type2(), ptr_rhs);

  track_new_pointer(ptr_obj, new_type, guard, gen_one(size_type2()));

  alloc_guard.append(guard);
  dynamic_memory.emplace_back(
    rhs_copy, alloc_guard, true, symbol.name.as_string());

  return to_address_of2t(rhs_addrof).ptr_obj;
}

expr2tc goto_symext::symex_mem(
  const bool is_malloc,
  const expr2tc &lhs,
  const sideeffect2t &code,
  const guardt &guard)
{
  if (is_nil_expr(lhs))
    return expr2tc(); // ignore

  // size
  type2tc type = code.alloctype;
  expr2tc size = code.size;
  bool size_is_one = false;

  if (is_nil_type(type))
    type = char_type2();

  if (is_nil_expr(size))
    size_is_one = true;
  else
  {
    cur_state->rename(size);

    // Detect malloc(-N) before do_simplify folds typecast(size_t, -N) into
    // a large positive constant and erases the sign. The simplifier's
    // behaviour varies between the normal and --no-slice paths, so we
    // capture the inner operand's sign up front and also re-check the
    // post-simplify value below.
    bool is_negative_size = false;
    if (is_malloc && is_typecast2t(size))
    {
      expr2tc inner = to_typecast2t(size).from;
      do_simplify(inner);
      is_negative_size = is_constant_int2t(inner) &&
                         to_constant_int2t(inner).value.is_negative();
    }

    do_simplify(size);
    if (is_constant_int2t(size))
    {
      const BigInt &val = to_constant_int2t(size).value;
      // Check negativity before inspecting the magnitude: to_uint64()
      // discards the sign, so malloc(-1) would otherwise be mistaken for
      // a 1-byte allocation.
      if (is_malloc && (is_negative_size || val.is_negative()))
      {
        // Negative size cast to size_t: return NULL even under
        // --force-malloc-success, matching real OS behaviour.
        expr2tc null_sym = symbol2tc(pointer_type2tc(type), "NULL");
        if (null_sym->type != lhs->type)
          null_sym = typecast2tc(lhs->type, null_sym);
        symex_assign(code_assign2tc(lhs, null_sym), true, guard);
        return null_sym;
      }
      uint64_t v = val.to_uint64();
      if (v == 1)
        size_is_one = true;
      else if (v == 0 && options.get_bool_option("malloc-zero-is-null"))
        return symbol2tc(pointer_type2tc(type), "NULL");
    }
  }

  unsigned int &dynamic_counter = get_dynamic_counter();
  dynamic_counter++;

  // value
  symbolt symbol;

  symbol.name = "dynamic_" + i2string(dynamic_counter) +
                (size_is_one ? "_value" : "_array");

  symbol.id = std::string("symex_dynamic::") + (!is_malloc ? "alloca::" : "") +
              id2string(symbol.name);
  symbol.lvalue = true;

  typet renamedtype = ns.follow(migrate_type_back(type));
  if (size_is_one)
    symbol.type = renamedtype;
  else
  {
    symbol.type = typet(typet::t_array);
    symbol.type.subtype() = renamedtype;
    symbol.type.size(migrate_expr_back(size));
  }

  symbol.type.dynamic(true);

  symbol.type.set(
    "alignment", constant_exprt(config.ansi_c.max_alignment(), size_type()));

  symbol.mode = "C";

  new_context.add(symbol);

  type2tc new_type = migrate_type(symbol.type);

  type2tc rhs_type;
  expr2tc rhs_ptr_obj;

  if (size_is_one)
  {
    rhs_type = migrate_type(symbol.type);
    rhs_ptr_obj = symbol2tc(new_type, symbol.id);
  }
  else
  {
    type2tc subtype = migrate_type(symbol.type.subtype());
    expr2tc sym = symbol2tc(new_type, symbol.id);
    expr2tc idx_val = gen_long(size->type, 0L);
    expr2tc idx = index2tc(subtype, sym, idx_val);
    do_simplify(idx);
    rhs_type = migrate_type(symbol.type.subtype());
    rhs_ptr_obj = idx;
  }

  expr2tc rhs_addrof = address_of2tc(rhs_type, rhs_ptr_obj);
  do_simplify(rhs_addrof);

  expr2tc rhs = rhs_addrof;
  expr2tc ptr_rhs = rhs;
  guardt alloc_guard = cur_state->guard;

  if (options.get_bool_option("malloc-zero-is-null"))
  {
    expr2tc null_sym = symbol2tc(rhs->type, "NULL");
    expr2tc choice = greaterthan2tc(size, gen_long(size->type, 0));
    alloc_guard.add(choice);
    rhs = if2tc(rhs->type, choice, rhs, null_sym);
  }

  if (!options.get_bool_option("force-malloc-success") && is_malloc)
  {
    expr2tc null_sym = symbol2tc(rhs->type, "NULL");
    expr2tc choice = sideeffect2tc(
      get_bool_type(),
      expr2tc(),
      expr2tc(),
      std::vector<expr2tc>(),
      type2tc(),
      sideeffect2t::nondet);
    replace_nondet(choice);

    rhs = if2tc(rhs->type, choice, rhs, null_sym);
    alloc_guard.add(choice);

    ptr_rhs = rhs;
  }

  if (rhs->type != lhs->type)
    rhs = typecast2tc(lhs->type, rhs);

  cur_state->rename(rhs);
  expr2tc rhs_copy(rhs);

  symex_assign(code_assign2tc(lhs, rhs), true, guard);

  expr2tc ptr_obj = pointer_object2tc(pointer_type2(), ptr_rhs);

  if (size_is_one)
    track_new_pointer(ptr_obj, new_type, guard);
  else
    track_new_pointer(ptr_obj, new_type, guard, size);

  alloc_guard.append(guard);
  dynamic_memory.emplace_back(
    rhs_copy, alloc_guard, !is_malloc, symbol.name.as_string());

  return to_address_of2t(rhs_addrof).ptr_obj;
}

void goto_symext::track_new_pointer(
  const expr2tc &ptr_obj,
  const type2tc &new_type,
  const guardt &guard,
  const expr2tc &size)
{
  // Simplify ptr_obj before using it in any expressions
  expr2tc simplified_ptr_obj = ptr_obj;
  do_simplify(simplified_ptr_obj);

  // Also update all the accounting data.

  // Mark that object as being dynamic, in the __ESBMC_is_dynamic array
  type2tc sym_type = array_type2tc(get_bool_type(), expr2tc(), true);
  expr2tc sym = symbol2tc(sym_type, dyn_info_arr_name);

  expr2tc idx = index2tc(get_bool_type(), sym, ptr_obj);
  expr2tc truth = gen_true_expr();
  symex_assign(code_assign2tc(idx, truth), true, guard);

  expr2tc valid_sym = symbol2tc(sym_type, valid_ptr_arr_name);
  expr2tc valid_index_expr = index2tc(get_bool_type(), valid_sym, ptr_obj);
  truth = gen_true_expr();
  symex_assign(code_assign2tc(valid_index_expr, truth), true, guard);

  type2tc sz_sym_type = array_type2tc(size_type2(), expr2tc(), true);
  expr2tc sz_sym = symbol2tc(sz_sym_type, alloc_size_arr_name);
  expr2tc sz_index_expr = index2tc(size_type2(), sz_sym, ptr_obj);

  expr2tc object_size_exp =
    is_nil_expr(size) ? type_byte_size_expr(new_type) : size;

  symex_assign(code_assign2tc(sz_index_expr, object_size_exp), true, guard);
}

void goto_symext::symex_free(const expr2tc &expr)
{
  const auto &code = static_cast<const code_expression_data &>(*expr);

  // Trigger 'free'-mode dereference of this pointer. Should generate various
  // dereference failure callbacks.
  expr2tc tmp = code.operand;
  dereference(tmp, dereferencet::FREE);

  // Don't rely on the output of dereference in free mode; instead fetch all
  // the internal dereference state for pointed at objects, and creates claims
  // that if pointed at, their offset is zero.
  internal_deref_items.clear();
  tmp = code.operand;

  // Create temporary, dummy, dereference
  tmp = dereference2tc(get_uint8_type(), tmp);
  dereference(tmp, dereferencet::INTERNAL);

  // Only add assertions to check pointer offset if pointer check is enabled
  if (!options.get_bool_option("no-pointer-check"))
  {
    // Get all dynamic objects allocated using alloca
    std::vector<allocated_obj> allocad;
    for (auto const &item : dynamic_memory)
      if (item.auto_deallocd)
        allocad.push_back(item);

    for (auto const &item : internal_deref_items)
    {
      guardt g = cur_state->guard;
      g.add(item.guard);

      // Check if the offset of the object being freed is zero
      expr2tc offset = item.offset;
      expr2tc eq = equality2tc(offset, gen_ulong(0));
      g.guard_expr(eq);
      if (options.get_bool_option("conv-assert-to-assume"))
        assume(eq);
      else
        claim(eq, "Operand of free must have zero pointer offset");

      // Check if we are not freeing an dynamic object allocated using alloca
      for (auto const &a : allocad)
      {
        expr2tc alloc_obj = get_base_object(a.obj);
        while (is_if2t(alloc_obj))
        {
          const if2t &the_if = to_if2t(alloc_obj);
          assert(is_symbol2t(the_if.false_value));
          assert(to_symbol2t(the_if.false_value).thename == "NULL");
          alloc_obj = get_base_object(the_if.true_value);
        }
        assert(is_symbol2t(alloc_obj));
        const irep_idt &id_alloc_obj = to_symbol2t(alloc_obj).thename;
        const irep_idt &id_item_obj = to_symbol2t(item.object).thename;
        // Check if the object allocated with alloca is the same
        // as given in the free function
        if (id_alloc_obj == id_item_obj)
        {
          expr2tc noteq = notequal2tc(alloc_obj, item.object);
          g.guard_expr(noteq);
          if (options.get_bool_option("conv-assert-to-assume"))
            assume(noteq);
          else
            claim(noteq, "dereference failure: invalid pointer freed");
        }
      }
    }
  }

  // Clear the alloc bit.
  type2tc sym_type = array_type2tc(get_bool_type(), expr2tc(), true);
  expr2tc ptr_obj = pointer_object2tc(pointer_type2(), code.operand);
  dereference(ptr_obj, dereferencet::READ);

  expr2tc valid_sym = symbol2tc(sym_type, valid_ptr_arr_name);
  expr2tc valid_index_expr = index2tc(get_bool_type(), valid_sym, ptr_obj);
  expr2tc falsity = gen_false_expr();
  symex_assign(code_assign2tc(valid_index_expr, falsity), true);
}
