#include <cassert>
#include <complex>
#include <functional>
#include <goto-symex/execution_state.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/reachability_tree.h>
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

expr2tc goto_symext::symex_malloc(const expr2tc &lhs, const sideeffect2t &code)
{
  return symex_mem(true, lhs, code);
}

expr2tc goto_symext::symex_alloca(const expr2tc &lhs, const sideeffect2t &code)
{
  return symex_mem(false, lhs, code);
}

void goto_symext::symex_realloc(const expr2tc &lhs, const sideeffect2t &code)
{
  expr2tc src_ptr = code.operand;
  expr2tc realloc_size = code.size;

  internal_deref_items.clear();
  dereference2tc deref(get_empty_type(), src_ptr);
  dereference(deref, dereferencet::INTERNAL);
  // src_ptr is now invalidated.

  // Free the given pointer. This just uses the pointer object from the pointer
  // variable that's the argument to realloc. It also leads to pointer validity
  // checking, and checks that the offset is zero.
  code_free2tc fr(code.operand);
  symex_free(fr);

  // We now have a list of things to work on. Recurse into them, build a result,
  // and then switch between those results afterwards.
  // Result list is the address of the reallocated piece of data, and the guard.
  std::list<std::pair<expr2tc, expr2tc>> result_list;
  for(auto &item : internal_deref_items)
  {
    expr2tc guard = item.guard;
    cur_state->rename_address(item.object);
    cur_state->guard.guard_expr(guard);
    target->renumber(guard, item.object, realloc_size, cur_state->source);
    type2tc new_ptr = type2tc(new pointer_type2t(item.object->type));
    address_of2tc addrof(new_ptr, item.object);
    result_list.emplace_back(addrof, item.guard);

    // Bump the realloc-numbering of the object. This ensures that, after
    // renaming, the address_of we just generated compares differently to
    // previous address_of's before the realloc.
    unsigned int cur_num = 0;
    if(cur_state->realloc_map.find(item.object) != cur_state->realloc_map.end())
    {
      cur_num = cur_state->realloc_map[item.object];
    }

    cur_num++;
    std::map<expr2tc, unsigned>::value_type v(item.object, cur_num);
    cur_state->realloc_map.insert(v);
  }

  // Rebuild a gigantic if-then-else chain from the result list.
  expr2tc result;
  if(result_list.size() == 0)
  {
    // Nothing happened; there was nothing, or only null, to point at.
    // In this case, just return right now and leave the pointer free. The
    // symex_free that occurred above should trigger a dereference failure.
    return;
  }

  result = expr2tc();
  for(auto const &it : result_list)
  {
    if(is_nil_expr(result))
      result = it.first;
    else
      result = if2tc(result->type, it.second, it.first, result);
  }

  // Install pointer modelling data into the relevant arrays.
  pointer_object2tc ptr_obj(pointer_type2(), result);
  track_new_pointer(ptr_obj, type2tc(), realloc_size);

  symex_assign(code_assign2tc(lhs, result), true);
}

expr2tc goto_symext::symex_mem(
  const bool is_malloc,
  const expr2tc &lhs,
  const sideeffect2t &code)
{
  if(is_nil_expr(lhs))
    return expr2tc(); // ignore

  // size
  type2tc type = code.alloctype;
  expr2tc size = code.size;
  bool size_is_one = false;

  if(is_nil_expr(size))
    size_is_one = true;
  else
  {
    cur_state->rename(size);
    BigInt i;
    if(is_constant_int2t(size) && to_constant_int2t(size).as_ulong() == 1)
      size_is_one = true;
  }

  if(is_nil_type(type))
    type = char_type2();

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
  if(size_is_one)
    symbol.type = renamedtype;
  else
  {
    symbol.type = typet(typet::t_array);
    symbol.type.subtype() = renamedtype;
    symbol.type.size(migrate_expr_back(size));
  }

  symbol.type.dynamic(true);

  symbol.mode = "C";

  new_context.add(symbol);

  type2tc new_type = migrate_type(symbol.type);

  address_of2tc rhs_addrof(get_empty_type(), expr2tc());

  if(size_is_one)
  {
    rhs_addrof->type = migrate_type(pointer_typet(symbol.type));
    rhs_addrof->ptr_obj = symbol2tc(new_type, symbol.id);
  }
  else
  {
    type2tc subtype = migrate_type(symbol.type.subtype());
    expr2tc sym = symbol2tc(new_type, symbol.id);
    expr2tc idx_val = gen_ulong(0);
    expr2tc idx = index2tc(subtype, sym, idx_val);
    rhs_addrof->type = migrate_type(pointer_typet(symbol.type.subtype()));
    rhs_addrof->ptr_obj = idx;
  }

  expr2tc rhs = rhs_addrof;
  expr2tc ptr_rhs = rhs;
  guardt alloc_guard = cur_state->guard;

  if(!options.get_bool_option("force-malloc-success") && is_malloc)
  {
    symbol2tc null_sym(rhs->type, "NULL");
    sideeffect2tc choice(
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

  if(rhs->type != lhs->type)
    rhs = typecast2tc(lhs->type, rhs);

  cur_state->rename(rhs);
  expr2tc rhs_copy(rhs);

  symex_assign(code_assign2tc(lhs, rhs), true);

  pointer_object2tc ptr_obj(pointer_type2(), ptr_rhs);
  track_new_pointer(ptr_obj, new_type);

  dynamic_memory.emplace_back(
    rhs_copy, alloc_guard, !is_malloc, symbol.name.as_string());

  return rhs_addrof->ptr_obj;
}

void goto_symext::track_new_pointer(
  const expr2tc &ptr_obj,
  const type2tc &new_type,
  const expr2tc &size)
{
  // Also update all the accounting data.

  // Mark that object as being dynamic, in the __ESBMC_is_dynamic array
  type2tc sym_type =
    type2tc(new array_type2t(get_bool_type(), expr2tc(), true));
  symbol2tc sym(sym_type, dyn_info_arr_name);

  index2tc idx(get_bool_type(), sym, ptr_obj);
  expr2tc truth = gen_true_expr();
  symex_assign(code_assign2tc(idx, truth), true);

  symbol2tc valid_sym(sym_type, valid_ptr_arr_name);
  index2tc valid_index_expr(get_bool_type(), valid_sym, ptr_obj);
  truth = gen_true_expr();
  symex_assign(code_assign2tc(valid_index_expr, truth), true);

  type2tc sz_sym_type =
    type2tc(new array_type2t(uint_type2(), expr2tc(), true));
  symbol2tc sz_sym(sz_sym_type, alloc_size_arr_name);
  index2tc sz_index_expr(get_bool_type(), sz_sym, ptr_obj);

  expr2tc object_size_exp;
  if(is_nil_expr(size))
  {
    try
    {
      BigInt object_size = type_byte_size(new_type);
      object_size_exp = constant_int2tc(uint_type2(), object_size.to_uint64());
    }
    catch(const array_type2t::dyn_sized_array_excp &e)
    {
      object_size_exp = typecast2tc(uint_type2(), e.size);
    }
  }
  else
  {
    object_size_exp = size;
  }

  symex_assign(code_assign2tc(sz_index_expr, object_size_exp), true);
}

void goto_symext::symex_free(const expr2tc &expr)
{
  const code_free2t &code = to_code_free2t(expr);

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
  if(!options.get_bool_option("no-pointer-check"))
  {
    // Get all dynamic objects allocated using alloca
    std::vector<allocated_obj> allocad;
    for(auto const &item : dynamic_memory)
      if(item.auto_deallocd)
        allocad.push_back(item);

    for(auto const &item : internal_deref_items)
    {
      guardt g = cur_state->guard;
      g.add(item.guard);

      // Check if the offset of the object being freed is zero
      expr2tc offset = item.offset;
      expr2tc eq = equality2tc(offset, gen_ulong(0));
      g.guard_expr(eq);
      claim(eq, "Operand of free must have zero pointer offset");

      // Check if we are not freeing an dynamic object allocated using alloca
      for(auto const &a : allocad)
      {
        expr2tc alloc_obj = get_base_object(a.obj);
        const irep_idt &id_alloc_obj = to_symbol2t(alloc_obj).thename;
        const irep_idt &id_item_obj = to_symbol2t(item.object).thename;
        // Check if the object allocated with alloca is the same
        // as given in the free function
        if(id_alloc_obj == id_item_obj)
        {
          expr2tc noteq = notequal2tc(alloc_obj, item.object);
          g.guard_expr(noteq);
          claim(noteq, "dereference failure: invalid pointer freed");
        }
      }
    }
  }

  // Clear the alloc bit.
  type2tc sym_type =
    type2tc(new array_type2t(get_bool_type(), expr2tc(), true));
  expr2tc ptr_obj = pointer_object2tc(pointer_type2(), code.operand);
  dereference(ptr_obj, dereferencet::READ);

  symbol2tc valid_sym(sym_type, valid_ptr_arr_name);
  index2tc valid_index_expr(get_bool_type(), valid_sym, ptr_obj);
  expr2tc falsity = gen_false_expr();
  symex_assign(code_assign2tc(valid_index_expr, falsity), true);
}

void goto_symext::symex_printf(const expr2tc &, const expr2tc &rhs)
{
  assert(is_code_printf2t(rhs));

  code_printf2tc new_rhs(to_code_printf2t(rhs));
  cur_state->rename(new_rhs);

  // The expr2tc in position 0 is the string format
  const irep_idt fmt = get_string_argument(new_rhs->operands[0]);

  // Now we pop the format
  new_rhs->operands.erase(new_rhs->operands.begin());

  std::list<expr2tc> args;
  new_rhs->foreach_operand([this, &args](const expr2tc &e) {
    expr2tc tmp = e;
    do_simplify(tmp);
    args.push_back(tmp);
  });

  target->output(
    cur_state->guard.as_expr(), cur_state->source, fmt.as_string(), args);
}

void goto_symext::symex_cpp_new(const expr2tc &lhs, const sideeffect2t &code)
{
  bool do_array;

  do_array = (code.kind == sideeffect2t::cpp_new_arr);

  unsigned int &dynamic_counter = get_dynamic_counter();
  dynamic_counter++;

  const std::string count_string(i2string(dynamic_counter));

  // value
  symbolt symbol;
  symbol.name = do_array ? "dynamic_" + count_string + "_array"
                         : "dynamic_" + count_string + "_value";
  symbol.id = "symex_dynamic::" + id2string(symbol.name);
  symbol.lvalue = true;
  symbol.mode = "C++";

  const pointer_type2t &ptr_ref = to_pointer_type(code.type);
  type2tc renamedtype2 =
    migrate_type(ns.follow(migrate_type_back(ptr_ref.subtype)));

  type2tc newtype =
    do_array ? type2tc(new array_type2t(renamedtype2, code.size, false))
             : renamedtype2;

  symbol.type = migrate_type_back(newtype);

  symbol.type.dynamic(true);

  new_context.add(symbol);

  // make symbol expression

  address_of2tc rhs(renamedtype2, expr2tc());

  if(do_array)
  {
    symbol2tc sym(newtype, symbol.id);
    index2tc idx(renamedtype2, sym, gen_ulong(0));
    rhs->ptr_obj = idx;
  }
  else
    rhs->ptr_obj = symbol2tc(newtype, symbol.id);

  cur_state->rename(rhs);
  expr2tc rhs_copy(rhs);

  symex_assign(code_assign2tc(lhs, rhs), true);

  // Mark that object as being dynamic, in the __ESBMC_is_dynamic array
  type2tc sym_type =
    type2tc(new array_type2t(get_bool_type(), expr2tc(), true));
  symbol2tc sym(sym_type, "__ESBMC_is_dynamic");

  pointer_object2tc ptr_obj(pointer_type2(), lhs);
  index2tc idx(get_bool_type(), sym, ptr_obj);
  expr2tc truth = gen_true_expr();

  symex_assign(code_assign2tc(idx, truth), true);

  dynamic_memory.emplace_back(
    rhs_copy, cur_state->guard, false, symbol.name.as_string());
}

// XXX - implement as a call to free?
void goto_symext::symex_cpp_delete(const expr2tc &)
{
  //bool do_array=code.statement()=="delete[]";
}

void goto_symext::intrinsic_yield(reachability_treet &art)
{
  // Don't context switch if the guard is false.
  if(!cur_state->guard.is_false())
    art.get_cur_state().force_cswitch();
}

void goto_symext::intrinsic_switch_to(
  const code_function_call2t &call,
  reachability_treet &art)
{
  // Switch to other thread.
  const expr2tc &num = call.operands[0];
  if(!is_constant_int2t(num))
  {
    log_error("Can't switch to non-constant thread id no\n{}", *num);
    abort();
  }

  const constant_int2t &thread_num = to_constant_int2t(num);

  unsigned int tid = thread_num.value.to_uint64();
  if(tid != art.get_cur_state().get_active_state_number())
    art.get_cur_state().switch_to_thread(tid);
}

void goto_symext::intrinsic_switch_from(reachability_treet &art)
{
  // Mark switching back to this thread as already having been explored
  art.get_cur_state()
    .DFS_traversed[art.get_cur_state().get_active_state_number()] = true;

  // And force a context switch.
  art.get_cur_state().force_cswitch();
}

void goto_symext::intrinsic_get_thread_id(
  const code_function_call2t &call,
  reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();

  unsigned int thread_id = art.get_cur_state().get_active_state_number();
  constant_int2tc tid(call.ret->type, BigInt(thread_id));

  state.value_set.assign(call.ret, tid);

  symex_assign(code_assign2tc(call.ret, tid), true);
}

void goto_symext::intrinsic_set_thread_data(
  const code_function_call2t &call,
  reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  expr2tc threadid = call.operands[0];
  expr2tc startdata = call.operands[1];

  state.global_guard.add(cur_state->guard.as_expr());
  state.rename(threadid);
  state.rename(startdata);

  while(is_typecast2t(threadid))
    threadid = to_typecast2t(threadid).from;

  if(!is_constant_int2t(threadid))
  {
    log_error("__ESBMC_set_start_data received nonconstant thread id");
    abort();
  }
  unsigned int tid = to_constant_int2t(threadid).value.to_uint64();
  art.get_cur_state().set_thread_start_data(tid, startdata);
}

void goto_symext::intrinsic_get_thread_data(
  const code_function_call2t &call,
  reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  expr2tc threadid = call.operands[0];

  state.level2.rename(threadid);

  while(is_typecast2t(threadid))
    threadid = to_typecast2t(threadid).from;

  if(!is_constant_int2t(threadid))
  {
    log_error("__ESBMC_get_start_data received nonconstant thread id");
    abort();
  }

  unsigned int tid = to_constant_int2t(threadid).value.to_uint64();
  const expr2tc &startdata = art.get_cur_state().get_thread_start_data(tid);

  assert(base_type_eq(call.ret->type, startdata->type, ns));

  state.value_set.assign(call.ret, startdata);
  symex_assign(code_assign2tc(call.ret, startdata), true);
}

void goto_symext::intrinsic_spawn_thread(
  const code_function_call2t &call,
  reachability_treet &art)
{
  if(
    (k_induction || inductive_step) &&
    !options.get_bool_option("disable-inductive-step"))
  {
    log_warning(
      "WARNING: k-induction does not support concurrency yet. "
      "Disabling inductive step");

    // Disable inductive step on multi threaded code
    options.set_option("disable-inductive-step", true);
  }

  // As an argument, we expect the address of a symbol.
  expr2tc addr = call.operands[0];
  simplify(addr); /* simplification is not needed for clang-11, but clang-13
                   * inserts a no-op typecast here. */
  assert(is_address_of2t(addr));
  const address_of2t &addrof = to_address_of2t(addr);
  assert(is_symbol2t(addrof.ptr_obj));
  const irep_idt &symname = to_symbol2t(addrof.ptr_obj).thename;

  goto_functionst::function_mapt::const_iterator it =
    art.goto_functions.function_map.find(symname);
  if(it == art.goto_functions.function_map.end())
  {
    log_error("Spawning thread \"{}\": symbol not found", symname);
    abort();
  }

  if(!it->second.body_available)
  {
    log_error("Spawning thread \"{}\": no body", symname);
    abort();
  }

  const goto_programt &prog = it->second.body;

  // Invalidates current state reference!
  unsigned int thread_id = art.get_cur_state().add_thread(&prog);

  statet &state = art.get_cur_state().get_active_state();

  constant_int2tc thread_id_exp(call.ret->type, BigInt(thread_id));

  state.value_set.assign(call.ret, thread_id_exp);

  symex_assign(code_assign2tc(call.ret, thread_id_exp), true);

  // Force a context switch point. If the caller is in an atomic block, it'll be
  // blocked, but a context switch will be forced when we exit the atomic block.
  // Otherwise, this will cause the required context switch.
  art.get_cur_state().force_cswitch();
}

void goto_symext::intrinsic_terminate_thread(reachability_treet &art)
{
  art.get_cur_state().end_thread();
  // No need to force a context switch; an ended thread will cause the run to
  // end and the switcher to be invoked.
}

void goto_symext::intrinsic_get_thread_state(
  const code_function_call2t &call,
  reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  expr2tc threadid = call.operands[0];
  state.level2.rename(threadid);

  while(is_typecast2t(threadid))
    threadid = to_typecast2t(threadid).from;

  if(!is_constant_int2t(threadid))
  {
    log_error("__ESBMC_get_thread_state received nonconstant thread id");
    abort();
  }

  unsigned int tid = to_constant_int2t(threadid).value.to_uint64();
  // Possibly we should handle this error; but meh.
  assert(art.get_cur_state().threads_state.size() >= tid);

  // Thread state is simply whether the thread is ended or not.
  unsigned int flags =
    (art.get_cur_state().threads_state[tid].thread_ended) ? 1 : 0;

  // Reuse threadid
  constant_int2tc flag_expr(get_uint_type(config.ansi_c.int_width), flags);
  symex_assign(code_assign2tc(call.ret, flag_expr), true);
}

void goto_symext::intrinsic_really_atomic_begin(reachability_treet &art)
{
  art.get_cur_state().increment_active_atomic_number();
}

void goto_symext::intrinsic_really_atomic_end(reachability_treet &art)
{
  art.get_cur_state().decrement_active_atomic_number();
}

void goto_symext::intrinsic_switch_to_monitor(reachability_treet &art)
{
  execution_statet &ex_state = art.get_cur_state();

  // Don't do this if we're in the initialization function.
  if(cur_state->source.pc->function == "__ESBMC_main")
    return;

  ex_state.switch_to_monitor();
}

void goto_symext::intrinsic_switch_from_monitor(reachability_treet &art)
{
  execution_statet &ex_state = art.get_cur_state();
  ex_state.switch_away_from_monitor();
}

void goto_symext::intrinsic_register_monitor(
  const code_function_call2t &call,
  reachability_treet &art)
{
  execution_statet &ex_state = art.get_cur_state();

  if(ex_state.tid_is_set)
    assert(
      0 && "Monitor thread ID was already set (__ESBMC_register_monitor)\n");

  statet &state = art.get_cur_state().get_active_state();
  expr2tc threadid = call.operands[0];
  state.level2.rename(threadid);

  while(is_typecast2t(threadid))
    threadid = to_typecast2t(threadid).from;

  if(!is_constant_int2t(threadid))
  {
    log_error("__ESBMC_register_monitor received nonconstant thread id");
    abort();
  }

  unsigned int tid = to_constant_int2t(threadid).value.to_uint64();
  assert(art.get_cur_state().threads_state.size() >= tid);
  ex_state.monitor_tid = tid;
  ex_state.tid_is_set = true;
}

void goto_symext::intrinsic_kill_monitor(reachability_treet &art)
{
  execution_statet &ex_state = art.get_cur_state();
  ex_state.kill_monitor_thread();
}

void goto_symext::symex_va_arg(const expr2tc &lhs, const sideeffect2t &code)
{
  // Get symbol
  expr2tc symbol = code.operand;
  assert(is_symbol2t(symbol));

  // to allow constant propagation
  cur_state->rename(symbol);
  do_simplify(symbol);

  expr2tc next_symbol = symbol;
  if(is_typecast2t(next_symbol))
    next_symbol = to_typecast2t(symbol).from;

  if(is_address_of2t(next_symbol))
    next_symbol = to_address_of2t(next_symbol).ptr_obj;

  assert(is_symbol2t(next_symbol));
  irep_idt id = to_symbol2t(next_symbol).thename;
  std::string base =
    id2string(cur_state->top().function_identifier) + "::va_arg";

  id = base + std::to_string(cur_state->top().va_index++);

  expr2tc va_rhs;

  const symbolt *s = new_context.find_symbol(id);
  if(s != nullptr)
  {
    type2tc symbol_type = migrate_type(s->type);

    va_rhs = symbol2tc(symbol_type, s->id);
    cur_state->top().level1.get_ident_name(va_rhs);

    va_rhs = address_of2tc(symbol_type, va_rhs);
    va_rhs = typecast2tc(lhs->type, va_rhs);
  }
  else
  {
    va_rhs = gen_zero(lhs->type);
  }

  symex_assign(code_assign2tc(lhs, va_rhs), true);
}

// Computes the equivalent object value when considering a memset operation on it
expr2tc gen_byte_expression_byte_update(
  const type2tc &type,
  const expr2tc &src,
  const expr2tc &value,
  const size_t num_of_bytes,
  const size_t offset)
{
  // Sadly, our simplifier can not typecast from value operations
  // safely. We can however :)
  auto new_src = src;
  auto new_type = type;

  auto found_constant = false;
  auto optimized = src->simplify();
  if(optimized)
  {
    found_constant = is_typecast2t(optimized) &&
                     is_constant_int2t(to_typecast2t(optimized).from);
    if(found_constant)
    {
      new_src = to_typecast2t(optimized).from;
      new_type = get_int64_type();
    }
  }

  expr2tc result = new_src;
  auto value_downcast = typecast2tc(get_uint8_type(), value);

  constant_int2tc off(get_int32_type(), BigInt(offset));
  for(size_t counter = 0; counter < num_of_bytes; counter++)
  {
    constant_int2tc increment(get_int32_type(), BigInt(counter));
    result = byte_update2tc(
      new_type,
      result,
      add2tc(off->type, off, increment),
      value_downcast,
      false);
  }

  if(found_constant)
    result = typecast2tc(type, result);

  simplify(result);

  return result;
}

// Computes the equivalent object value when considering a memset operation on it
expr2tc gen_byte_expression(
  const type2tc &type,
  const expr2tc &src,
  const expr2tc &value,
  const size_t num_of_bytes,
  const size_t offset)
{
  /**
   * The idea of this expression is to compute the object value
   * in the case where every byte `value` was set set up until num_of_bytes
   *
   * @warning this function does not add any pointer/memory/bounds check!
   *          they should be added before calling this function!
   *
   * In summary, there are two main computations here:
   *
   * A. Generate the byte representation, this is mostly through
   *    the `result` expression. The expression is initialized with zero
   *    and then, until the num_of_bytes is reached it will do a full byte
   *    left-shift followed by an bitor operation with the byte value:
   *
   *    Example, for a integer(4 bytes) with memset using 3 bytes and value 0xF1
   *
   *    step 1: 0x00000000 -- left-shift 8 -- 0x00000000 -- bitor -- 0x000000F1
   *    step 2: 0x000000F1 -- left-shift 8 -- 0x0000F100 -- bitor -- 0x0000F1F1
   *    step 3: 0x0000F1F1 -- left-shift 8 -- 0x00F1F100 -- bitor -- 0x00F1F1F1
   *
   *    Since we only want 3 bytes, the initialized object value would be 0x00F1F1F1
   *
   * B. Generate a mask of the bits that were not set, this is done because skipped bits
   *    need to be returned back. The computation of this is simple, we initialize every
   *    bit that was changed by the byte-representation computation with a 1. Which is then
   *    negated to be applied with an bitand in the original value:
   *
   *    Back to the example in A, we had the byte-representation of  0x00F1F1F1. If the
   *    original value was 0xA2A2A2A2, then we would have the following mask:
   *
   *    step 1: 0x00000000 -- set-bits -- 0x000000FF
   *    step 2: 0x000000FF -- set-bits -- 0x0000FFFF
   *    step 3: 0x0000FFFF -- set-bits -- 0x00FFFFFF
   *
   *   So, 0x00FFFFFF is the mask for all bits changed. We can negate it to: 0xFF000000
   *
   *   Then, we can apply it to the original source value with bitand
   *
   *   0xA2A2A2A2 AND 0xFF000000 --> 0xA2000000
   *
   * Finally, we get the result from A and B and unify them through a bitor
   *
   *  0xA2000000 OR 0x00F1F1F1 --> 0xA2F1F1F1
   *
   * Note about offsets: To handle them, we apply left shifts to the remaining offset after
   * the computation of the object-value and initial mask representation
   *
   */

  if(is_pointer_type(type))
    return gen_byte_expression_byte_update(
      type, src, value, num_of_bytes, offset);
  expr2tc result = gen_zero(type);
  auto value_downcast = typecast2tc(get_uint8_type(), value);
  auto value_upcast = typecast2tc(
    type,
    value_downcast); // so smt_conv will complain about the width of the type

  expr2tc mask = gen_zero(type);

  auto eight = constant_int2tc(int_type2(), BigInt(8));
  auto one = constant_int2tc(int_type2(), BigInt(1));
  for(unsigned i = 0; i < num_of_bytes; i++)
  {
    result = shl2tc(type, result, eight);
    result = bitor2tc(type, result, value_upcast);

    for(int m = 0; m < 8; m++)
    {
      mask = shl2tc(type, mask, one);
      mask = bitor2tc(type, mask, one);
    }
  }

  // Do the rest of the offset!
  for(unsigned i = 0; i < offset; i++)
  {
    result = shl2tc(type, result, eight);
    mask = shl2tc(type, mask, eight);
  }

  mask = bitnot2tc(type, mask);
  mask = bitand2tc(type, src, mask);
  result = bitor2tc(type, result, mask);

  simplify(result);
  return result;
}

inline expr2tc gen_value_by_byte(
  const type2tc &type,
  const expr2tc &src,
  const expr2tc &value,
  const size_t num_of_bytes,
  const size_t offset)
{
  /**
   * @brief Construct a new object, initializing it with the memset equivalent
   *
   * There are a few corner cases here:
   *
   * 1 - Primitives: these are simple: just generate the byte_expression directly
   * 2 - Arrays: these are ok: just keep generating byte_expression for each member
   *        until a limit has arrived. Dynamic memory is dealt here.
   * 3 - Structs/Union: these are the hardest as we have to take the alignment into
   *        account when dealing with it. Hopefully the clang-frontend already give it
   *        to us.
   *
   */

  // I am not sure if bitwise operations are valid for floats
  if(is_floatbv_type(type) || is_fixedbv_type(type))
    return expr2tc();

  if(is_array_type(type))
  {
    /*
     * Very straighforward, get the total number_of_bytes and keep subtracting until
     * the end
     */

    constant_array2tc result = gen_zero(type);

    auto base_size = type_byte_size(to_array_type(type).subtype).to_uint64();

    auto bytes_left = num_of_bytes;
    auto offset_left = offset;

    for(unsigned i = 0; i < result->datatype_members.size(); i++)
    {
      BigInt position(i);
      index2tc local_member(
        to_array_type(type).subtype,
        src,
        constant_int2tc(get_uint32_type(), position));
      // Skip offsets
      if(offset_left >= base_size)
      {
        result->datatype_members[i] = local_member;
        offset_left -= base_size;
      }
      else
      {
        assert(offset_left < base_size);
        auto bytes_to_write = bytes_left < base_size ? bytes_left : base_size;
        result->datatype_members[i] = gen_value_by_byte(
          to_array_type(type).subtype,
          local_member,
          value,
          bytes_to_write,
          offset_left);
        bytes_left =
          bytes_left <= base_size ? 0 : bytes_left - (base_size - offset_left);
        offset_left = offset_left <= base_size ? 0 : offset_left - base_size;
        assert(offset_left == 0);
      }
    }

    return result;
  }

  if(is_struct_type(type))
  {
    /** Similar to array, however get the size of
     * each component
     */
    constant_struct2tc result = gen_zero(type);

    auto bytes_left = num_of_bytes;
    auto offset_left = offset;

    for(unsigned i = 0; i < result->datatype_members.size(); i++)
    {
      auto name = to_struct_type(type).member_names[i];
      member2tc local_member(to_struct_type(type).members[i], src, name);

      // Since it is a symbol, lets start from the old value
      if(is_pointer_type(to_struct_type(type).members[i]))
        result->datatype_members[i] = local_member;

      auto current_member_type = result->datatype_members[i]->type;

      auto current_member_size =
        type_byte_size(current_member_type).to_uint64();

      // Skip offsets
      if(offset_left >= current_member_size)
      {
        result->datatype_members[i] = local_member;
        offset_left -= current_member_size;
      }
      else
      {
        assert(offset_left < current_member_size);
        auto bytes_to_write = std::min(bytes_left, current_member_size);
        result->datatype_members[i] = gen_value_by_byte(
          current_member_type,
          local_member,
          value,
          bytes_to_write,
          offset_left);

        if(!result->datatype_members[i])
          return expr2tc();

        bytes_left = bytes_left < current_member_size
                       ? 0
                       : bytes_left - (current_member_size - offset_left);
        offset_left = offset_left <= current_member_size
                        ? 0
                        : offset_left - current_member_size;
        assert(offset_left == 0);
      }
    }
    return result;
  }

  if(is_union_type(type))
  {
    /**
     * Unions are not nice, let's go through every member
     * and get the biggest one! And then use it directly
     *
     * @warning there is a semantic difference on this when
     * compared to c:@F@__memset_impl. While this function
     * will yield the same result as `clang` would, ESBMC
     * will handle the dereference (in the __memset_impl)
     * using the first member, which can lead to overflows.
     * See GitHub Issue #639
     *
     */
    constant_union2tc result = gen_zero(type);

    auto union_total_size = type_byte_size(type).to_uint64();
    // Let's find a member with the biggest size
    int selected_member_index;

    for(unsigned i = 0; i < to_union_type(type).members.size(); i++)
    {
      if(
        type_byte_size(to_union_type(type).members[i]).to_uint64() ==
        union_total_size)
      {
        selected_member_index = i;
        break;
      }
    }

    auto name = to_union_type(type).member_names[selected_member_index];
    auto member_type = to_union_type(type).members[selected_member_index];
    member2tc member(member_type, src, name);

    result->init_field = name;
    result->datatype_members[0] =
      gen_value_by_byte(member_type, member, value, num_of_bytes, offset);
    return result;
  }

  // Found a primitive! Just apply the function
  return gen_byte_expression(type, src, value, num_of_bytes, offset);
}

void goto_symext::intrinsic_memset(
  reachability_treet &art,
  const code_function_call2t &func_call)
{
  /**
     * @brief This function will try to initialize the object pointed by
     * the address in a smarter way, minimizing the number of assignments.
     * This is intend to optimize the behaviour of a memset operation:
     *
     * memset(void* ptr, int value, size_t num_of_bytes)
     *
     * - ptr can point to anything. We have to add checks!
     * - value is interpreted as a uchar.
     * - num_of_bytes must be known. If it is nondet, we will bump the call
     *
     * In plain C, the objective of a call such as:
     *
     * int a;
     * memset(&a, value, num)
     *
     * Would generate something as:
     *
     * int temp = 0;
     * for(int i = 0; i < num; i++) temp = byte | (temp << 8);
     * a = temp;
     *
     * This is just a simplification for understanding though. During the
     * instrumentation size checks will be added, and also, the original
     * bytes from `a` that were not overwritten must be mantained!
     * Arrays will need to be added up to an nth element.
     *
     * In ESBMC though, we have 2 main methods of dealing with memory objects:
     *
     * A. Heap objects, which are valid/invalid. They are the easiest to deal
     *    with, as the dereference will actually return a big array of char to us.
     *    For this case, we can just overwrite the members directly with the value
     *
     * B. Stack objects, which are typed. It will be hard, this will require operations
     *    which depends on the base type and also on padding.
     *
     */

  // 1. Check for the functions parameters and do the deref and processing!

  assert(func_call.operands.size() == 3 && "Wrong memset signature");
  auto &ex_state = art.get_cur_state();
  if(ex_state.cur_state->guard.is_false())
    return;

  // Define a local function for translating to calling the unwinding C
  // implementation of memset
  auto bump_call = [this, &func_call]() -> void {
    // We're going to execute a function call, and that's going to mess with
    // the program counter. Set it back *onto* pointing at this intrinsic, so
    // symex_function_call calculates the right return address. Misery.
    cur_state->source.pc--;

    expr2tc newcall = func_call.clone();
    code_function_call2t &mutable_funccall = to_code_function_call2t(newcall);
    mutable_funccall.function =
      symbol2tc(get_empty_type(), "c:@F@__memset_impl");
    // Execute call
    symex_function_call(newcall);
    return;
  };

  /* Get the arguments
     * arg0: ptr to object
     * arg1: int for the new byte value
     * arg2: number of bytes to be set */
  expr2tc arg0 = func_call.operands[0];
  expr2tc arg1 = func_call.operands[1];
  expr2tc arg2 = func_call.operands[2];

  msg.debug("[memset] started call");

  // Checks where arg0 points to
  internal_deref_items.clear();
  dereference2tc deref(get_empty_type(), arg0);
  dereference(deref, dereferencet::INTERNAL);

  /* Preconditions for the optimization:
     * A: It should point to someplace
     * B: byte itself should be renamed properly
     * C: Number of bytes cannot be symbolic
     * D: This is a simplification. So don't run with --no-simplify */
  cur_state->rename(arg1);
  cur_state->rename(arg2);
  if(
    !internal_deref_items.size() || !arg1 || !arg2 || is_symbol2t(arg2) ||
    options.get_bool_option("no-simplify"))
  {
    /* Not sure what to do here, let's rely
       * on the default implementation then */
    msg.debug("[memset] Couldn't optimize memset due to precondition");
    bump_call();
    return;
  }

  simplify(arg2);
  if(!is_constant_int2t(arg2))
  {
    msg.debug("[memset] TODO: simplifier issues :/");
    bump_call();
    return;
  }

  auto number_of_bytes = to_constant_int2t(arg2).as_ulong();

  // If no byte was changed... we are finished

  // Where are we pointing to?
  for(auto &item : internal_deref_items)
  {
    auto guard = ex_state.cur_state->guard;
    auto item_object = item.object;
    auto item_offset = item.offset;
    guard.add(item.guard);

    cur_state->rename(item_object);
    cur_state->rename(item_offset);

    /* Pre-requisites locally:
       * item_object must be something!
       * item_offset must be something! */
    if(!item_object || !item_offset)
    {
      msg.debug("[memset] Couldn't get item_object/item_offset");
      bump_call();
      return;
    }

    simplify(item_offset);
    // We can't optimize symbolic offsets :/
    if(is_symbol2t(item_offset))
    {
      msg.debug(fmt::format(
        "[memset] Item offset is symbolic: {}",
        to_symbol2t(item_offset).get_symbol_name()));
      bump_call();
      return;
    }

    // TODO: Why (X*Y)/Y is not X?
    if(is_div2t(item_offset))
    {
      auto as_div = to_div2t(item_offset);
      if(is_mul2t(as_div.side_1) && is_constant_int2t(as_div.side_2))
      {
        auto as_mul = to_mul2t(as_div.side_1);
        if(
          is_constant_int2t(as_mul.side_2) &&
          (to_constant_int2t(as_mul.side_2).as_ulong() ==
           to_constant_int2t(as_div.side_2).as_ulong()))
        {
          // if side_1 of mult is a pointer_offset, then it is just zero
          if(is_pointer_offset2t(as_mul.side_1))
          {
            msg.debug("[memset] TODO: some simplifications are missing");
            item_offset = constant_int2tc(get_uint64_type(), BigInt(0));
          }
        }
      }
    }

    if(!is_constant_int2t(item_offset))
    {
      /* If we reached here, item_offset is not symbolic
       * and we don't know what the actual value of it is...
       *
       * For now bump_call, later we should expand our simplifier
       */
      msg.debug(
        "[memset] TODO: some simplifications are missing, bumping call");
      bump_call();
      return;
    }

    auto number_of_offset = to_constant_int2t(item_offset).value.to_uint64();
    auto type_size = type_byte_size(item_object->type).to_uint64();

    if(is_code_type(item_object->type))
    {
      auto error_msg =
        fmt::format("dereference failure: trying to deref a ptr code");

      auto false_expr = gen_false_expr();
      guard.guard_expr(false_expr);
      claim(false_expr, error_msg);
      continue;
    }

    auto is_out_bounds = ((type_size - number_of_offset) < number_of_bytes) ||
                         (number_of_offset > type_size);
    if(
      is_out_bounds && !options.get_bool_option("no-pointer-check") &&
      !options.get_bool_option("no-bounds-check"))
    {
      auto error_msg = fmt::format(
        "dereference failure: memset of memory segment of size {} with {} "
        "bytes",
        type_size - number_of_offset,
        number_of_bytes);

      guard.add(gen_false_expr());
      claim(gen_false_expr(), error_msg);
      continue;
    }

    auto new_object = gen_value_by_byte(
      item_object->type, item_object, arg1, number_of_bytes, number_of_offset);

    // Where we able to optimize it? If not... bump call
    if(!new_object)
    {
      msg.debug("[memset] gen_value_by_byte failed");
      bump_call();
      return;
    }
    // 4. Assign the new object
    symex_assign(code_assign2tc(item.object, new_object), false, guard);
  }
  // Lastly, let's add a NULL ptr check
  if(!options.get_bool_option("no-pointer-check"))
  {
    symbol2tc null_sym(arg0->type, "NULL");
    same_object2tc obj(arg0, null_sym);
    not2tc null_check(same_object2tc(arg0, null_sym));
    ex_state.cur_state->guard.guard_expr(null_check);
    claim(null_check, " dereference failure: NULL pointer");
  }

  expr2tc ret_ref = func_call.ret;
  dereference(ret_ref, dereferencet::READ);
  symex_assign(code_assign2tc(ret_ref, arg0), false, cur_state->guard);
}

void goto_symext::intrinsic_get_object_size(
  const code_function_call2t &func_call,
  reachability_treet &)
{
  assert(func_call.operands.size() == 1 && "Wrong get_object_size signature");
  auto ptr = func_call.operands[0];

  // Work out what the ptr points at.
  internal_deref_items.clear();
  dereference2tc deref(get_empty_type(), ptr);
  dereference(deref, dereferencet::INTERNAL);

  assert(is_array_type(internal_deref_items.front().object->type));
  auto obj_size =
    to_array_type(internal_deref_items.front().object->type).array_size;

  expr2tc ret_ref = func_call.ret;
  dereference(ret_ref, dereferencet::READ);
  symex_assign(code_assign2tc(ret_ref, obj_size), false, cur_state->guard);
}
