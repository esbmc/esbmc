/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <complex>
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
#include <util/dcutil.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/std_types.h>
#include <vector>

expr2tc
goto_symext::symex_malloc(
  const expr2tc &lhs,
  const sideeffect2t &code)
{
  return symex_mem(true, lhs, code);
}

expr2tc
goto_symext::symex_alloca(
  const expr2tc &lhs,
  const sideeffect2t &code)
{
  return symex_mem(false, lhs, code);
}

void
goto_symext::symex_realloc(const expr2tc &lhs, const sideeffect2t &code)
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
  std::list<std::pair<expr2tc,expr2tc> > result_list;
  for (auto &item : internal_deref_items) {
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
    if (cur_state->realloc_map.find(item.object) !=
        cur_state->realloc_map.end()) {
      cur_num = cur_state->realloc_map[item.object];
    }

    cur_num++;
    std::map<expr2tc, unsigned>::value_type v(item.object, cur_num);
    cur_state->realloc_map.insert(v);
  }

  // Rebuild a gigantic if-then-else chain from the result list.
  expr2tc result;
  if (result_list.size() == 0) {
    // Nothing happened; there was nothing, or only null, to point at.
    // In this case, just return right now and leave the pointer free. The
    // symex_free that occurred above should trigger a dereference failure.
    return;
  } else {
    result = expr2tc();
    for (auto const &it : result_list)
    {
      if (is_nil_expr(result))
        result = it.first;
      else
        result = if2tc(result->type, it.second, it.first, result);
    }
  }

  // Install pointer modelling data into the relevant arrays.
  pointer_object2tc ptr_obj(pointer_type2(), result);
  track_new_pointer(ptr_obj, type2tc(), realloc_size);

  guardt guard;
  symex_assign_rec(lhs, result, guard, symex_targett::STATE);
}

expr2tc
goto_symext::symex_mem(
  const bool is_malloc,
  const expr2tc &lhs,
  const sideeffect2t &code)
{
  if (is_nil_expr(lhs))
    return expr2tc(); // ignore

  // size
  type2tc type = code.alloctype;
  expr2tc size = code.size;
  bool size_is_one = false;

  if (is_nil_expr(size))
    size_is_one=true;
  else
  {
    cur_state->rename(size);
    mp_integer i;
    if (is_constant_int2t(size) && to_constant_int2t(size).as_ulong() == 1)
      size_is_one = true;
  }

  if (is_nil_type(type))
    type = char_type2();
  else if (is_union_type(type)) {
    // Filter out creation of instantiated unions. They're now all byte arrays.
    size_is_one = false;
    type = char_type2();
  }

  unsigned int &dynamic_counter = get_dynamic_counter();
  dynamic_counter++;

  // value
  symbolt symbol;

  symbol.base_name = "dynamic_" + i2string(dynamic_counter) +
                     (size_is_one ? "_value" : "_array");

  symbol.name = "symex_dynamic::" + id2string(symbol.base_name);
  symbol.lvalue = true;

  typet renamedtype = ns.follow(migrate_type_back(type));
  if(size_is_one)
    symbol.type=renamedtype;
  else
  {
    symbol.type=typet(typet::t_array);
    symbol.type.subtype()=renamedtype;
    symbol.type.size(migrate_expr_back(size));
  }

  symbol.type.dynamic(true);

  symbol.mode="C";

  new_context.add(symbol);

  type2tc new_type;
  migrate_type(symbol.type, new_type);

  address_of2tc rhs_addrof(get_empty_type(), expr2tc());

  if(size_is_one)
  {
    rhs_addrof.get()->type = get_pointer_type(pointer_typet(symbol.type));
    rhs_addrof.get()->ptr_obj = symbol2tc(new_type, symbol.name);
  }
  else
  {
    type2tc subtype;
    migrate_type(symbol.type.subtype(), subtype);
    expr2tc sym = symbol2tc(new_type, symbol.name);
    expr2tc idx_val = gen_ulong(0);
    expr2tc idx = index2tc(subtype, sym, idx_val);
    rhs_addrof.get()->type =
      get_pointer_type(pointer_typet(symbol.type.subtype()));
    rhs_addrof.get()->ptr_obj = idx;
  }

  expr2tc rhs = rhs_addrof;
  expr2tc ptr_rhs = rhs;
  guardt alloc_guard = cur_state->guard;

  if (!options.get_bool_option("force-malloc-success")) {
    symbol2tc null_sym(rhs->type, "NULL");
    sideeffect2tc choice(get_bool_type(), expr2tc(), expr2tc(), std::vector<expr2tc>(), type2tc(), sideeffect2t::nondet);
    replace_nondet(choice);

    rhs = if2tc(rhs->type, choice, rhs, null_sym);
    alloc_guard.add(choice);

    ptr_rhs = rhs;
  }

  if (rhs->type != lhs->type)
    rhs = typecast2tc(lhs->type, rhs);

  cur_state->rename(rhs);
  expr2tc rhs_copy(rhs);

  guardt guard;
  symex_assign_rec(lhs, rhs, guard, symex_targett::STATE);

  pointer_object2tc ptr_obj(pointer_type2(), ptr_rhs);
  track_new_pointer(ptr_obj, new_type);

  dynamic_memory.emplace_back(rhs_copy, alloc_guard, !is_malloc);

  return rhs_addrof->ptr_obj;
}

void
goto_symext::track_new_pointer(const expr2tc &ptr_obj, const type2tc &new_type,
                               const expr2tc& size)
{
  guardt guard;

  // Also update all the accounting data.

  // Mark that object as being dynamic, in the __ESBMC_is_dynamic array
  type2tc sym_type = type2tc(new array_type2t(get_bool_type(),
                                              expr2tc(), true));
  symbol2tc sym(sym_type, dyn_info_arr_name);

  index2tc idx(get_bool_type(), sym, ptr_obj);
  expr2tc truth = gen_true_expr();
  symex_assign_rec(idx, truth, guard, symex_targett::STATE);

  symbol2tc valid_sym(sym_type, valid_ptr_arr_name);
  index2tc valid_index_expr(get_bool_type(), valid_sym, ptr_obj);
  truth = gen_true_expr();
  symex_assign_rec(valid_index_expr, truth, guard, symex_targett::STATE);

  symbol2tc dealloc_sym(sym_type, deallocd_arr_name);
  index2tc dealloc_index_expr(get_bool_type(), dealloc_sym, ptr_obj);
  expr2tc falseity = gen_false_expr();
  symex_assign_rec(dealloc_index_expr, falseity, guard, symex_targett::STATE);

  type2tc sz_sym_type =
    type2tc(new array_type2t(pointer_type2(), expr2tc(),true));
  symbol2tc sz_sym(sz_sym_type, alloc_size_arr_name);
  index2tc sz_index_expr(get_bool_type(), sz_sym, ptr_obj);

  expr2tc object_size_exp;
  if (is_nil_expr(size)) {
    try {
      mp_integer object_size = type_byte_size(new_type);
      object_size_exp =
        constant_int2tc(pointer_type2(), object_size.to_ulong());
    } catch (array_type2t::dyn_sized_array_excp *e) {
      object_size_exp = typecast2tc(pointer_type2(), e->size);
    }
  } else {
    object_size_exp = size;
  }

  symex_assign_rec(sz_index_expr, object_size_exp, guard, symex_targett::STATE);
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

  for (auto const &item : internal_deref_items) {
    guardt g = cur_state->guard;
    g.add(item.guard);
    expr2tc offset = item.offset;
    expr2tc eq = equality2tc(offset, gen_ulong(0));
    g.guard_expr(eq);
    claim(eq, "Operand of free must have zero pointer offset");
  }

  // Clear the alloc bit, and set the deallocated bit.
  guardt guard;
  type2tc sym_type = type2tc(new array_type2t(get_bool_type(),
                                              expr2tc(), true));
  expr2tc ptr_obj = pointer_object2tc(pointer_type2(), code.operand);
  dereference(ptr_obj, dereferencet::READ);

  symbol2tc dealloc_sym(sym_type, deallocd_arr_name);
  index2tc dealloc_index_expr(get_bool_type(), dealloc_sym, ptr_obj);
  expr2tc truth = gen_true_expr();
  symex_assign_rec(dealloc_index_expr, truth, guard, symex_targett::STATE);

  symbol2tc valid_sym(sym_type, valid_ptr_arr_name);
  index2tc valid_index_expr(get_bool_type(), valid_sym, ptr_obj);
  expr2tc falsity = gen_false_expr();
  symex_assign_rec(valid_index_expr, falsity, guard, symex_targett::STATE);
}

void goto_symext::symex_printf(
  const expr2tc &lhs __attribute__((unused)),
  const expr2tc &rhs)
{

  assert(is_code_printf2t(rhs));
  expr2tc new_rhs = rhs;
  cur_state->rename(new_rhs);

  const expr2tc &format = *new_rhs->get_sub_expr(0);

  if (is_address_of2t(format))
  {
    const address_of2t &addrof = to_address_of2t(format);
    if (is_index2t(addrof.ptr_obj))
    {
      const index2t &idx = to_index2t(addrof.ptr_obj);
      if(is_constant_string2t(idx.source_value)
         && is_constant_int2t(idx.index)
         && to_constant_int2t(idx.index).as_ulong() == 0)
      {
        const std::string &fmt =
          to_constant_string2t(idx.source_value).value.as_string();

        std::list<expr2tc> args;
        new_rhs->foreach_operand([this, &args] (const expr2tc &e)
          {
            expr2tc tmp = e;
            do_simplify(tmp);
            args.push_back(tmp);
          });

        target->output(cur_state->guard.as_expr(), cur_state->source, fmt, args);
      }
    }
  }
}

void goto_symext::symex_cpp_new(
  const expr2tc &lhs,
  const sideeffect2t &code)
{
  bool do_array;

  do_array = (code.kind == sideeffect2t::cpp_new_arr);

  unsigned int &dynamic_counter = get_dynamic_counter();
  dynamic_counter++;

  const std::string count_string(i2string(dynamic_counter));

  // value
  symbolt symbol;
  symbol.base_name=
    do_array?"dynamic_"+count_string+"_array":
             "dynamic_"+count_string+"_value";
  symbol.name="symex_dynamic::"+id2string(symbol.base_name);
  symbol.lvalue=true;
  symbol.mode="C++";

  const pointer_type2t &ptr_ref = to_pointer_type(code.type);
  typet renamedtype = ns.follow(migrate_type_back(ptr_ref.subtype));
  type2tc newtype, renamedtype2;
  migrate_type(renamedtype, renamedtype2);

  if(do_array)
  {
    newtype = type2tc(new array_type2t(renamedtype2, code.size, false));
  }
  else
    newtype = renamedtype2;

  symbol.type = migrate_type_back(newtype);

  symbol.type.dynamic(true);

  new_context.add(symbol);

  // make symbol expression

  address_of2tc rhs(renamedtype2, expr2tc());

  if(do_array)
  {
    symbol2tc sym(newtype, symbol.name);
    index2tc idx(renamedtype2, sym, gen_ulong(0));
    rhs.get()->ptr_obj = idx;
  }
  else
    rhs.get()->ptr_obj = symbol2tc(newtype, symbol.name);

  cur_state->rename(rhs);
  expr2tc rhs_copy(rhs);

  guardt guard;
  symex_assign_rec(lhs, rhs, guard, symex_targett::STATE);

  // Mark that object as being dynamic, in the __ESBMC_is_dynamic array
  type2tc sym_type = type2tc(new array_type2t(get_bool_type(),
                                              expr2tc(), true));
  symbol2tc sym(sym_type, "__ESBMC_is_dynamic");

  pointer_object2tc ptr_obj(pointer_type2(), lhs);
  index2tc idx(get_bool_type(), sym, ptr_obj);
  expr2tc truth = gen_true_expr();

  symex_assign_rec(idx, truth, guard, symex_targett::STATE);

  dynamic_memory.emplace_back(rhs_copy, cur_state->guard, false);
}

// XXX - implement as a call to free?
void goto_symext::symex_cpp_delete(const expr2tc &code __attribute__((unused)))
{
  //bool do_array=code.statement()=="delete[]";
}

void
goto_symext::intrinsic_yield(reachability_treet &art)
{

  art.get_cur_state().force_cswitch();
}


void
goto_symext::intrinsic_switch_to(const code_function_call2t &call,
                                 reachability_treet &art)
{

  // Switch to other thread.
  const expr2tc &num = call.operands[0];
  if (!is_constant_int2t(num)) {
    std::cerr << "Can't switch to non-constant thread id no";
    abort();
  }

  const constant_int2t &thread_num = to_constant_int2t(num);

  unsigned int tid = thread_num.value.to_long();
  if (tid != art.get_cur_state().get_active_state_number())
    art.get_cur_state().switch_to_thread(tid);
}

void
goto_symext::intrinsic_switch_from(reachability_treet &art)
{

  // Mark switching back to this thread as already having been explored
  art.get_cur_state().DFS_traversed[art.get_cur_state().get_active_state_number()] = true;

  // And force a context switch.
  art.get_cur_state().force_cswitch();
}


void
goto_symext::intrinsic_get_thread_id(const code_function_call2t &call,
                                     reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();

  unsigned int thread_id = art.get_cur_state().get_active_state_number();
  constant_int2tc tid(call.ret->type, BigInt(thread_id));

  state.value_set.assign(call.ret, tid);

  code_assign2tc assign(call.ret, tid);
  symex_assign(assign);
}

void
goto_symext::intrinsic_set_thread_data(const code_function_call2t &call,
                                       reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  expr2tc threadid = call.operands[0];
  expr2tc startdata = call.operands[1];

  state.rename(threadid);
  state.rename(startdata);

  while (is_typecast2t(threadid))
    threadid = to_typecast2t(threadid).from;

  if (!is_constant_int2t(threadid)) {
    std::cerr << "__ESBMC_set_start_data received nonconstant thread id";
    std::cerr << std::endl;
    abort();
  }

  unsigned int tid = to_constant_int2t(threadid).value.to_ulong();
  art.get_cur_state().set_thread_start_data(tid, startdata);
}

void
goto_symext::intrinsic_get_thread_data(const code_function_call2t &call,
                                       reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  expr2tc threadid = call.operands[0];

  state.level2.rename(threadid);

  while (is_typecast2t(threadid))
    threadid = to_typecast2t(threadid).from;

  if (!is_constant_int2t(threadid)) {
    std::cerr << "__ESBMC_set_start_data received nonconstant thread id";
    std::cerr << std::endl;
    abort();
  }

  unsigned int tid = to_constant_int2t(threadid).value.to_ulong();
  const expr2tc &startdata = art.get_cur_state().get_thread_start_data(tid);

  code_assign2tc assign(call.ret, startdata);
  assert(base_type_eq(call.ret->type, startdata->type, ns));

  state.value_set.assign(call.ret, startdata);
  symex_assign(assign);
}

void
goto_symext::intrinsic_spawn_thread(const code_function_call2t &call,
                                    reachability_treet &art)
{
  // As an argument, we expect the address of a symbol.
  const expr2tc &addr = call.operands[0];
  assert(is_address_of2t(addr));
  const address_of2t &addrof = to_address_of2t(addr);
  assert(is_symbol2t(addrof.ptr_obj));
  const irep_idt &symname = to_symbol2t(addrof.ptr_obj).thename;

  goto_functionst::function_mapt::const_iterator it =
    art.goto_functions.function_map.find(symname);
  if (it == art.goto_functions.function_map.end()) {
    std::cerr << "Spawning thread \"" << symname << "\": symbol not found";
    std::cerr << std::endl;
    abort();
  }

  if (!it->second.body_available) {
    std::cerr << "Spawning thread \"" << symname << "\": no body" << std::endl;
    abort();
  }

  const goto_programt &prog = it->second.body;

  // Invalidates current state reference!
  unsigned int thread_id = art.get_cur_state().add_thread(&prog);

  statet &state = art.get_cur_state().get_active_state();

  constant_int2tc thread_id_exp(call.ret->type, BigInt(thread_id));

  state.value_set.assign(call.ret, thread_id_exp);

  code_assign2tc assign(call.ret, thread_id_exp);
  symex_assign(assign);

  // Force a context switch point. If the caller is in an atomic block, it'll be
  // blocked, but a context switch will be forced when we exit the atomic block.
  // Otherwise, this will cause the required context switch.
  art.get_cur_state().force_cswitch();
}

void
goto_symext::intrinsic_terminate_thread(reachability_treet &art)
{
  art.get_cur_state().end_thread();
  // No need to force a context switch; an ended thread will cause the run to
  // end and the switcher to be invoked.
}

void
goto_symext::intrinsic_get_thread_state(const code_function_call2t &call, reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  expr2tc threadid = call.operands[0];
  state.level2.rename(threadid);

  while (is_typecast2t(threadid))
    threadid = to_typecast2t(threadid).from;

  if (!is_constant_int2t(threadid)) {
    std::cerr << "__ESBMC_get_thread_state received nonconstant thread id";
    std::cerr << std::endl;
    abort();
  }

  unsigned int tid = to_constant_int2t(threadid).value.to_ulong();
  // Possibly we should handle this error; but meh.
  assert(art.get_cur_state().threads_state.size() >= tid);

  // Thread state is simply whether the thread is ended or not.
  unsigned int flags = (art.get_cur_state().threads_state[tid].thread_ended)
                       ? 1 : 0;

  // Reuse threadid
  constant_int2tc flag_expr(get_uint_type(config.ansi_c.int_width), flags);
  code_assign2tc assign(call.ret, flag_expr);
  symex_assign(assign);
}

void
goto_symext::intrinsic_really_atomic_begin(reachability_treet &art)
{

  art.get_cur_state().increment_active_atomic_number();
}

void
goto_symext::intrinsic_really_atomic_end(reachability_treet &art)
{

  art.get_cur_state().decrement_active_atomic_number();
}

void
goto_symext::intrinsic_switch_to_monitor(reachability_treet &art)
{
  execution_statet &ex_state = art.get_cur_state();

  // Don't do this if we're in the initialization function.
  if (cur_state->source.pc->function == "__ESBMC_main")
    return;

  ex_state.switch_to_monitor();
}

void
goto_symext::intrinsic_switch_from_monitor(reachability_treet &art)
{
  execution_statet &ex_state = art.get_cur_state();
  ex_state.switch_away_from_monitor();
}

void
goto_symext::intrinsic_register_monitor(const code_function_call2t &call, reachability_treet &art)
{
  execution_statet &ex_state = art.get_cur_state();

  if (ex_state.tid_is_set)
    assert(0 && "Monitor thread ID was already set (__ESBMC_register_monitor)\n");

  statet &state = art.get_cur_state().get_active_state();
  expr2tc threadid = call.operands[0];
  state.level2.rename(threadid);

  while (is_typecast2t(threadid))
    threadid = to_typecast2t(threadid).from;

  if (!is_constant_int2t(threadid)) {
    std::cerr << "__ESBMC_register_monitor received nonconstant thread id";
    std::cerr << std::endl;
    abort();
  }

  unsigned int tid = to_constant_int2t(threadid).value.to_ulong();
  assert(art.get_cur_state().threads_state.size() >= tid);
  ex_state.monitor_tid = tid;
  ex_state.tid_is_set = true;
}

void
goto_symext::intrinsic_kill_monitor(reachability_treet &art)
{
  execution_statet &ex_state = art.get_cur_state();
  ex_state.kill_monitor_thread();
}

void goto_symext::symex_va_arg(const expr2tc& lhs, const sideeffect2t &code)
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
  std::string base = id2string(cur_state->top().function_identifier) + "::va_arg";

  id = base + std::to_string(cur_state->top().va_index++);

  expr2tc va_rhs;

  const symbolt *s = new_context.find_symbol(id);
  if(s != nullptr)
  {
    type2tc symbol_type;
    migrate_type(s->type, symbol_type);

    va_rhs = symbol2tc(
      symbol_type, s->name, symbol2t::level1, 0, 0,
      cur_state->top().level1.thread_id, 0);

    va_rhs = address_of2tc(symbol_type, va_rhs);
    va_rhs = typecast2tc(lhs->type, va_rhs);
  }
  else
  {
    va_rhs = gen_zero(lhs->type);
  }

  symex_assign(code_assign2tc(lhs, va_rhs));
}
