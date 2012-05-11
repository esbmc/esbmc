/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>

#include <assert.h>

#include <expr_util.h>
#include <i2string.h>
#include <arith_tools.h>
#include <cprover_prefix.h>
#include <std_types.h>

#include <ansi-c/c_types.h>

#include "goto_symex.h"
#include "execution_state.h"
#include "reachability_tree.h"

void goto_symext::symex_malloc(
  const expr2tc &lhs,
  const sideeffect2t &code)
{
    
  if (is_nil_expr(lhs))
    return; // ignore

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
  
  expr2tc rhs = expr2tc(new address_of2t(type_pool.get_empty(), expr2tc()));
  address_of2t &rhs_ref = to_address_of2t(rhs);

  if(size_is_one)
  {
    rhs_ref.type = type_pool.get_pointer(pointer_typet(symbol.type));
    rhs_ref.ptr_obj = expr2tc(new symbol2t(new_type, symbol.name));
  }
  else
  {
    type2tc subtype;
    migrate_type(symbol.type.subtype(), subtype);
    expr2tc sym = expr2tc(new symbol2t(new_type, symbol.name));
    expr2tc idx_val = expr2tc(new constant_int2t(int_type2(), BigInt(0)));
    expr2tc idx = expr2tc(new index2t(subtype, sym, idx_val));
    rhs_ref.type = type_pool.get_pointer(pointer_typet(symbol.type.subtype()));
    rhs_ref.ptr_obj = idx;
  }
  
  if (rhs_ref.type != lhs->type)
    rhs = expr2tc(new typecast2t(lhs->type, rhs));

  // Pas this point, rhs_ref may be an invalid reference.

  cur_state->rename(rhs);
  
  guardt guard;
  exprt blah_rhs = migrate_expr_back(rhs);
  symex_assign_rec(migrate_expr_back(lhs), blah_rhs, guard);
  migrate_expr(blah_rhs, rhs); // fetch symex_assign_rec's modifications

  // Mark that object as being dynamic, in the __ESBMC_is_dynamic array
  type2tc sym_type = type2tc(new array_type2t(type_pool.get_bool(),
                                              expr2tc(), true));
  expr2tc sym = expr2tc(new symbol2t(sym_type, "c::__ESBMC_is_dynamic"));

  expr2tc ptr_obj = expr2tc(new pointer_object2t(int_type2(), lhs));

  expr2tc idx = expr2tc(new index2t(type_pool.get_bool(), sym, ptr_obj));

  expr2tc truth = expr2tc(new constant_bool2t(true));

  exprt idx_back = migrate_expr_back(idx);
  exprt truth_back = migrate_expr_back(truth);
  symex_assign_rec(idx_back, truth_back, guard);
}

void goto_symext::symex_printf(
  const expr2tc &lhs __attribute__((unused)),
  const expr2tc &rhs)
{

  assert(is_code_printf2t(rhs));
  expr2tc new_rhs = rhs;
  cur_state->rename(new_rhs);

  std::vector<const expr2tc *> operands;
  new_rhs->list_operands(operands);

  const expr2tc &format = *operands[0];
  
  if (is_address_of2t(format)) {
    const address_of2t &addrof = to_address_of2t(format);
    if (is_index2t(addrof.ptr_obj)) {
      const index2t &idx = to_index2t(addrof.ptr_obj);
      if (is_constant_string2t(idx.source_value) &&
          is_constant_int2t(idx.index) &&
          to_constant_int2t(idx.index).as_ulong() == 0) {
        const std::string &fmt =
          to_constant_string2t(idx.source_value).value.as_string();

        std::list<expr2tc> args; 
        for (std::vector<const expr2tc *>::const_iterator it = operands.begin();
             it != operands.end(); it++)
          args.push_back(**it);

        expr2tc guard;
        migrate_expr(cur_state->guard.as_expr(), guard);
        target->output(guard, cur_state->source, fmt, args);
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

  expr2tc rhs = expr2tc(new address_of2t(
                                     type2tc(new pointer_type2t(renamedtype2)),
                                     expr2tc()));
  address_of2t &addrof = to_address_of2t(rhs);

  if(do_array)
  {
    expr2tc sym = expr2tc(new symbol2t(type2tc(new pointer_type2t(newtype)),
                                       symbol.name));
    expr2tc zero = expr2tc(new constant_int2t(int_type2(), BigInt(0)));
    expr2tc idx = expr2tc(new index2t(renamedtype2, sym, zero));
    addrof.ptr_obj = idx;
  }
  else
    addrof.ptr_obj = expr2tc(new symbol2t(type2tc(new pointer_type2t(newtype)),
                                          symbol.name));
  
  cur_state->rename(rhs);

  guardt guard;
  exprt lhs_back = migrate_expr_back(lhs);
  exprt rhs_back = migrate_expr_back(rhs);
  symex_assign_rec(lhs_back, rhs_back, guard);
}

// XXX - implement as a call to free?
void goto_symext::symex_cpp_delete(const codet &code __attribute__((unused)))
{
  //bool do_array=code.statement()=="delete[]";
}

void
goto_symext::intrinsic_yield(reachability_treet &art)
{

  art.force_cswitch_point();
  return;
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

  unsigned int tid = thread_num.constant_value.to_long();
  if (tid != art.get_cur_state().get_active_state_number())
    art.get_cur_state().switch_to_thread(tid);

  return;
}

void
goto_symext::intrinsic_switch_from(reachability_treet &art)
{

  // Mark switching back to this thread as already having been explored
  art.get_cur_state().DFS_traversed[art.get_cur_state().get_active_state_number()] = true;

  // And force a context switch.
  art.force_cswitch_point();
  return;
}


void
goto_symext::intrinsic_get_thread_id(const code_function_call2t &call,
                                     reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  unsigned int thread_id;

  thread_id = art.get_cur_state().get_active_state_number();
  expr2tc tid = expr2tc(new constant_int2t(uint_type2(), BigInt(thread_id)));

  state.value_set.assign(call.ret, tid, ns);

  expr2tc assign = expr2tc(new code_assign2t(call.ret, tid));
  assert(call.ret->type == tid->type);
  exprt tmp = migrate_expr_back(assign);
  symex_assign(static_cast<codet&>(static_cast<irept&>(tmp)));
  return;
}

void
goto_symext::intrinsic_set_thread_data(code_function_callt &call,
                                       reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  exprt threadid = call.arguments()[0];
  exprt startdata = call.arguments()[1];

  expr2tc new_threadid, new_startdata;
  migrate_expr(threadid, new_threadid);
  migrate_expr(startdata, new_startdata);
  state.rename(new_threadid);
  state.rename(new_startdata);
  threadid = migrate_expr_back(new_threadid);
  startdata = migrate_expr_back(new_startdata);

  if (threadid.id() != "constant") {
    std::cerr << "__ESBMC_set_start_data received nonconstant thread id";
    std::cerr << std::endl;
    abort();
  }

  unsigned int tid = binary2integer(threadid.value().as_string(), false).to_long();
  art.get_cur_state().set_thread_start_data(tid, startdata);
}

void
goto_symext::intrinsic_get_thread_data(code_function_callt &call,
                                       reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  exprt &threadid = call.arguments()[0];

  expr2tc new_threadid;
  migrate_expr(threadid, new_threadid);
  state.level2.rename(new_threadid);
  threadid = migrate_expr_back(new_threadid);

  if (threadid.id() != "constant") {
    std::cerr << "__ESBMC_set_start_data received nonconstant thread id";
    std::cerr << std::endl;
    abort();
  }

  unsigned int tid = binary2integer(threadid.value().as_string(), false).to_long();
  const exprt &startdata = art.get_cur_state().get_thread_start_data(tid);

  code_assignt assign(call.lhs(), startdata);
  assert(call.lhs().type() == startdata.type());
  expr2tc tmp_expr, tmp_startdata;
  migrate_expr(call.lhs(), tmp_expr);
  migrate_expr(startdata, tmp_startdata);
  state.value_set.assign(tmp_expr, tmp_startdata, ns);
  symex_assign(assign);
  return;
}

void
goto_symext::intrinsic_spawn_thread(code_function_callt &call, reachability_treet &art)
{

  // As an argument, we expect the address of a symbol.
  const exprt &args = call.operands()[2];
  assert(args.id() == "arguments");
  const exprt &addrof = args.operands()[0];
  assert(addrof.id() == "address_of");
  const exprt &symexpr = addrof.operands()[0];
  assert(symexpr.id() == "symbol");
  irep_idt symname = symexpr.get("identifier");

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

  constant_exprt thread_id_expr(unsignedbv_typet(config.ansi_c.int_width));
  thread_id_expr.set_value(integer2binary(thread_id, config.ansi_c.int_width));
  code_assignt assign(call.lhs(), thread_id_expr);
  expr2tc tmp_expr, tmp_thread_id;
  migrate_expr(call.lhs(), tmp_expr);
  migrate_expr(thread_id_expr, tmp_thread_id);
  state.value_set.assign(tmp_expr, tmp_thread_id, ns);
  symex_assign(assign);

  // Force a context switch point. If the caller is in an atomic block, it'll be
  // blocked, but a context switch will be forced when we exit the atomic block.
  // Otherwise, this will cause the required context switch.
  art.force_cswitch_point();

  return;
}

void
goto_symext::intrinsic_terminate_thread(reachability_treet &art)
{

  art.get_cur_state().end_thread();
  // No need to force a context switch; an ended thread will cause the run to
  // end and the switcher to be invoked.
  return;
}

void
goto_symext::intrinsic_get_thread_state(code_function_callt &call, reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  exprt threadid = call.arguments()[0];

  expr2tc new_threadid;
  migrate_expr(threadid, new_threadid);
  state.level2.rename(new_threadid);
  threadid = migrate_expr_back(new_threadid);

  if (threadid.id() != "constant") {
    std::cerr << "__ESBMC_get_thread_state received nonconstant thread id";
    std::cerr << std::endl;
    abort();
  }

  unsigned int tid = binary2integer(threadid.value().as_string(), false).to_long();
  // Possibly we should handle this error; but meh.
  assert(art.get_cur_state().threads_state.size() >= tid);

  // Thread state is simply whether the thread is ended or not.
  unsigned int flags = (art.get_cur_state().threads_state[tid].thread_ended)
                       ? 1 : 0;

  // Reuse threadid
  constant_exprt flag_expr(unsignedbv_typet(config.ansi_c.int_width));
  flag_expr.set_value(integer2binary(flags, config.ansi_c.int_width));
  code_assignt assign(call.lhs(), flag_expr);
  symex_assign(assign);
  return;
}
