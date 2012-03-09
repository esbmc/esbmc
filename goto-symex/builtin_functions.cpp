/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

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
  const exprt &lhs,
  const side_effect_exprt &code)
{
  if(code.operands().size()!=1)
    throw "malloc expected to have one operand";
    
  if(lhs.is_nil())
    return; // ignore

  // size
  typet type=static_cast<const typet &>(code.cmt_type());
  exprt size=static_cast<const exprt &>(code.cmt_size());
  bool size_is_one;

  if(size.is_nil())
    size_is_one=true;
  else
  {
    cur_state->rename(size);
    mp_integer i;
    size_is_one=(!to_integer(size, i) && i==1);
  }
  
  if(type.is_nil())
    type=char_type();

  unsigned int &dynamic_counter = get_dynamic_counter();
  dynamic_counter++;

  // value
  symbolt symbol;

  symbol.base_name="dynamic_"+
    i2string(dynamic_counter)+
    (size_is_one?"_value":"_array");

  symbol.name="symex_dynamic::"+id2string(symbol.base_name);
  symbol.lvalue=true;
  
  if(size_is_one)
    symbol.type=type;
  else
  {
    symbol.type=typet(typet::t_array);
    symbol.type.subtype()=type;
    symbol.type.size(size);
  }

  symbol.type.dynamic(true);

  symbol.mode="C";

  new_context.add(symbol);
  
  exprt rhs(exprt::addrof, typet(typet::t_pointer));
  
  if(size_is_one)
  {
    rhs.type().subtype()=symbol.type;
    rhs.copy_to_operands(symbol_expr(symbol));
  }
  else
  {
    exprt index_expr(exprt::index, symbol.type.subtype());
    index_expr.copy_to_operands(symbol_expr(symbol), gen_zero(int_type()));
    rhs.type().subtype()=symbol.type.subtype();
    rhs.move_to_operands(index_expr);
  }
  
  if(rhs.type()!=lhs.type())
    rhs.make_typecast(lhs.type());

  cur_state->rename(rhs);
  
  guardt guard;
  symex_assign_rec(lhs, rhs, guard);

  // Mark that object as being dynamic, in the __ESBMC_is_dynamic array
  exprt sym("symbol", array_typet());
  sym.type().subtype() = bool_typet();
  sym.set("identifier", "c::__ESBMC_is_dynamic");
  exprt pointerobj("pointer_object", signedbv_typet());
  exprt ptrsrc = lhs;
  pointerobj.move_to_operands(ptrsrc);
  exprt index("index", bool_typet());
  index.move_to_operands(sym, pointerobj);
  exprt truth("constant", bool_typet());
  truth.set("value", "true");
  symex_assign_rec(index, truth, guard);
}

void goto_symext::symex_printf(
  const exprt &lhs __attribute__((unused)),
  const exprt &rhs)
{
  if(rhs.operands().empty())
    throw "printf expected to have at least one operand";

  exprt tmp_rhs=rhs;
  cur_state->rename(tmp_rhs);

  const exprt::operandst &operands=tmp_rhs.operands();
  std::list<exprt> args;

  for(unsigned i=1; i<operands.size(); i++)
    args.push_back(operands[i]);

  const exprt &format=operands[0];
  
  if(format.id()==exprt::addrof &&
     format.operands().size()==1 &&
     format.op0().id()==exprt::index &&
     format.op0().operands().size()==2 &&
     format.op0().op0().id()=="string-constant" &&
     format.op0().op1().is_zero())
  {
    const exprt &fmt_str=format.op0().op0();
    const std::string &fmt=fmt_str.value().as_string();

    target->output(cur_state->guard, cur_state->source, fmt, args);
  }
}

void goto_symext::symex_cpp_new(
  const exprt &lhs,
  const side_effect_exprt &code)
{
  bool do_array;

  if(code.type().id()!=typet::t_pointer)
    throw "new expected to return pointer";

  do_array=(code.statement()=="cpp_new[]");
      
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
  
  if(do_array)
  {
    symbol.type=array_typet();
    symbol.type.subtype()=code.type().subtype();
    symbol.type.size(code.size_irep());
  }
  else
    symbol.type=code.type().subtype();

  //symbol.type.active(symbol_expr(active_symbol));
  symbol.type.dynamic(true);
  
  new_context.add(symbol);

  // make symbol expression

  exprt rhs(exprt::addrof, typet(typet::t_pointer));
  rhs.type().subtype()=code.type().subtype();
  
  if(do_array)
  {
    exprt index_expr(exprt::index, code.type().subtype());
    index_expr.copy_to_operands(symbol_expr(symbol), gen_zero(int_type()));
    rhs.move_to_operands(index_expr);
  }
  else
    rhs.copy_to_operands(symbol_expr(symbol));
  
  cur_state->rename(rhs);

  guardt guard;
  symex_assign_rec(lhs, rhs, guard);
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
goto_symext::intrinsic_switch_to(code_function_callt &call,
                                 reachability_treet &art)
{

  assert(call.arguments().size() == 1);

  // Switch to other thread.
  exprt &num = call.arguments()[0];
  if (num.id() != "constant") {
    std::cerr << "Can't switch to non-constant thread id no";
    abort();
  }

  unsigned int tid = binary2integer(num.value().as_string(), false).to_long();
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
goto_symext::intrinsic_get_thread_id(code_function_callt &call,
                                     reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  unsigned int thread_id;

  thread_id = art.get_cur_state().get_active_state_number();
  constant_exprt tid(unsignedbv_typet(config.ansi_c.int_width));
  tid.set_value(integer2binary(thread_id, config.ansi_c.int_width));

  code_assignt assign(call.lhs(), tid);
  assert(call.lhs().type() == tid.type());
  state.value_set.assign(call.lhs(), tid, ns);
  symex_assign(assign);
  return;
}

void
goto_symext::intrinsic_set_thread_data(code_function_callt &call,
                                       reachability_treet &art)
{
  statet &state = art.get_cur_state().get_active_state();
  exprt threadid = call.arguments()[0];
  exprt startdata = call.arguments()[1];

  state.rename(threadid);
  state.rename(startdata);

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

  state.level2.rename(threadid);

  if (threadid.id() != "constant") {
    std::cerr << "__ESBMC_set_start_data received nonconstant thread id";
    std::cerr << std::endl;
    abort();
  }

  unsigned int tid = binary2integer(threadid.value().as_string(), false).to_long();
  const exprt &startdata = art.get_cur_state().get_thread_start_data(tid);

  code_assignt assign(call.lhs(), startdata);
  assert(call.lhs().type() == startdata.type());
  state.value_set.assign(call.lhs(), startdata, ns);
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
  state.value_set.assign(call.lhs(), thread_id_expr, ns);
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
  state.level2.rename(threadid);

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

void
goto_symext::intrinsic_really_atomic_begin(reachability_treet &art)
{

  art.get_cur_state().increment_active_atomic_number();
  return;
}

void
goto_symext::intrinsic_really_atomic_end(reachability_treet &art)
{

  art.get_cur_state().decrement_active_atomic_number();
  return;
}

void
goto_symext::intrinsic_switch_to_monitor(reachability_treet &art)
{
  execution_statet &ex_state = art.get_cur_state();
  ex_state.switch_to_monitor();
  return;
}

void
goto_symext::intrinsic_switch_from_monitor(reachability_treet &art)
{
  execution_statet &ex_state = art.get_cur_state();
  ex_state.switch_away_from_monitor();
}

void
goto_symext::intrinsic_register_monitor(code_function_callt &call, reachability_treet &art)
{
  execution_statet &ex_state = art.get_cur_state();

  if (ex_state.tid_is_set)
    assert(0 && "Monitor thread ID was already set (__ESBMC_register_monitor)\n");

  statet &state = art.get_cur_state().get_active_state();
  exprt threadid = call.arguments()[0];
  state.level2.rename(threadid);

  if (threadid.id() != "constant") {
    std::cerr << "__ESBMC_register_monitor received nonconstant thread id";
    std::cerr << std::endl;
    abort();
  }

  unsigned int tid = binary2integer(threadid.value().as_string(), false).to_long();
  assert(art.get_cur_state().threads_state.size() >= tid);
  ex_state.monitor_tid = tid;
  ex_state.tid_is_set = true;
}
