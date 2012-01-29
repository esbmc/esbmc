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

#include "basic_symex.h"
#include "goto_symex.h"

/*******************************************************************\

Function: basic_symext::symex_malloc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_malloc(
  statet &state,
  const exprt &lhs,
  const side_effect_exprt &code,
  execution_statet &ex_state,
        unsigned node_id)
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
    state.rename(size, ns, node_id);
    mp_integer i;
    size_is_one=(!to_integer(size, i) && i==1);
  }
  
  if(type.is_nil())
    type=char_type();

  ex_state.dynamic_counter++;

  // value
  symbolt symbol;

  symbol.base_name="dynamic_"+
    i2string(ex_state.dynamic_counter)+
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

  state.rename(rhs, ns,node_id);
  
  guardt guard;
  symex_assign_rec(state, ex_state, lhs, rhs, guard,node_id);

  // Mark that object as being dynamic, in the __ESBMC_is_dynamic array
  exprt sym("symbol", array_typet());
  sym.type().subtype() = bool_typet();
  sym.set("identifier", "__ESBMC_is_dynamic");
  exprt pointerobj("pointer_object", signedbv_typet());
  exprt ptrsrc = lhs;
  pointerobj.move_to_operands(ptrsrc);
  exprt index("index", bool_typet());
  index.move_to_operands(sym, pointerobj);
  exprt truth("constant", bool_typet());
  truth.set("value", "true");
  symex_assign_rec(state, ex_state, index, truth, guard,node_id);
}

/*******************************************************************\

Function: basic_symext::symex_printf

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_printf(
  statet &state,
  const exprt &lhs,
  const exprt &rhs,
        unsigned node_id)
{
  if(rhs.operands().empty())
    throw "printf expected to have at least one operand";

  exprt tmp_rhs=rhs;
  state.rename(tmp_rhs, ns, node_id);

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

    target->output(state.guard, state.source, fmt, args);
  }
}

/*******************************************************************\

Function: basic_symext::symex_cpp_new

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_cpp_new(
  statet &state,
  const exprt &lhs,
  const side_effect_exprt &code,
  execution_statet &ex_state,
        unsigned node_id)
{
  bool do_array;

  if(code.type().id()!=typet::t_pointer)
    throw "new expected to return pointer";

  do_array=(code.statement()=="cpp_new[]");
      
  ex_state.dynamic_counter++;

  const std::string count_string(i2string(ex_state.dynamic_counter));

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
  
  state.rename(rhs, ns,node_id);

  guardt guard;
  symex_assign_rec(state, ex_state, lhs, rhs, guard,node_id);
}

/*******************************************************************\

Function: basic_symext::symex_cpp_delete

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_cpp_delete(
  statet &state,
  const codet &code)
{
  //bool do_array=code.statement()=="delete[]";
}

/*******************************************************************\

Function: basic_symext::symex_trace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_trace(
  statet &state,
  const code_function_callt &code,
        unsigned node_id)
{
  if(code.arguments().size()<2)
    throw "CBMC_trace expects at least two arguments";

  int debug_thresh=atol(options.get_option("debug-level").c_str());
  
  mp_integer debug_lvl;

  if(to_integer(code.arguments()[0], debug_lvl))
    throw "CBMC_trace expects constant as first argument";
    
  if(code.arguments()[1].id()!="implicit_address_of" ||
     code.arguments()[1].operands().size()!=1 ||
     code.arguments()[1].op0().id()!="string-constant")
    throw "CBMC_trace expects string constant as second argument";
  
  if(mp_integer(debug_thresh)>=debug_lvl)
  {
    std::list<exprt> vars;
    
    exprt trace_event("trave_event");
    trace_event.event(code.arguments()[1].op0().value());
    
    vars.push_back(trace_event);

    for(unsigned j=2; j<code.arguments().size(); j++)
    {
      exprt var(code.arguments()[j]);
      state.rename(var, ns,node_id);
      vars.push_back(var);
    }

    target->output(state.guard, state.source, "", vars);
  }
}

/*******************************************************************\

Function: basic_symext::symex_fkt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_fkt(
  statet &state,
  const code_function_callt &code)
{
  #if 0
  exprt new_fc("function", fc.type());

  new_fc.reserve_operands(fc.operands().size()-1);

  bool first=true;

  Forall_operands(it, fc)
    if(first) first=false; else new_fc.move_to_operands(*it);

  new_fc.identifier(fc.op0().identifier());

  fc.swap(new_fc);
  #endif
}

/*******************************************************************\

Function: basic_symext::symex_macro

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_macro(
  statet &state,
  const code_function_callt &code)
{
  const irep_idt &identifier=code.op0().identifier();

  if(identifier==CPROVER_MACRO_PREFIX "waitfor")
  {
    #if 0
    exprt new_fc("waitfor", fc.type());

    if(fc.operands().size()!=4)
      throw "waitfor expected to have four operands";

    exprt &cycle_var=fc.op1();
    exprt &bound=fc.op2();
    exprt &predicate=fc.op3();

    if(cycle_var.id()!=exprt::symbol)
      throw "waitfor expects symbol as first operand but got "+
            cycle_var.id();

    exprt new_cycle_var(cycle_var);
    new_cycle_var.id("waitfor_symbol");
    new_cycle_var.copy_to_operands(bound);

    replace_expr(cycle_var, new_cycle_var, predicate);

    new_fc.operands().resize(4);
    new_fc.op0().swap(cycle_var);
    new_fc.op1().swap(new_cycle_var);
    new_fc.op2().swap(bound);
    new_fc.op3().swap(predicate);

    fc.swap(new_fc);
    #endif
  }
  else
    throw "unknown macro: "+id2string(identifier);
}

void
goto_symext::intrinsic_yield(reachability_treet &art)
{

  art.generate_states();
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
  art.get_cur_state().set_active_state(tid);
  return;
}

void
goto_symext::intrinsic_set_start_arg(code_function_callt &call,
                                     reachability_treet &art)
{

  art.get_cur_state().set_next_thread_start_arg(call.arguments()[0]);
  return;
}

void
goto_symext::intrinsic_set_start_func(code_function_callt &call,
                                      reachability_treet &art)
{

  art.get_cur_state().set_next_thread_start_func(call.arguments()[0]);
  return;
}

void
goto_symext::intrinsic_get_start_arg(code_function_callt &call,
                                     reachability_treet &art)
{

  exprt arg = art.get_cur_state().get_next_thread_start_arg();
  code_assignt assign(call.lhs(), arg);
  assert(call.lhs().type() == arg.type());
  symex_assign(art.get_cur_state().get_active_state(), art.get_cur_state(),
               assign, art.get_cur_state().node_id);
  return;
}

void
goto_symext::intrinsic_get_start_func(code_function_callt &call,
                                      reachability_treet &art)
{

  exprt func = art.get_cur_state().get_next_thread_start_func();
  code_assignt assign(call.lhs(), func);
  assert(call.lhs().type() == func.type());
  symex_assign(art.get_cur_state().get_active_state(), art.get_cur_state(),
               assign, art.get_cur_state().node_id);
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
    art._goto_functions.function_map.find(symname);
  if (it == art._goto_functions.function_map.end()) {
    std::cerr << "Spawning thread \"" << symname << "\": symbol not found";
    std::cerr << std::endl;
    abort();
  }

  if (!it->second.body_available) {
    std::cerr << "Spawning thread \"" << symname << "\": no body" << std::endl;
    abort();
  }

  const goto_programt &prog = it->second.body;
  art.get_cur_state().add_thread(&prog);

  return;
}
