/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com
		Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>

#include <expr_util.h>
#include <i2string.h>
#include <cprover_prefix.h>
#include <prefix.h>
#include <arith_tools.h>
#include <base_type.h>
#include <std_expr.h>

#include <ansi-c/c_types.h>

#include "goto_symex.h"

/*******************************************************************\

Function: goto_symext::get_unwind_recursion

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_symext::get_unwind_recursion(
  const irep_idt &identifier,
  unsigned unwind)
{
  return false;
}

/*******************************************************************\

Function: goto_symext::argument_assignments

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::argument_assignments(
  const code_typet &function_type,
  execution_statet &ex_state,
  const exprt::operandst &arguments)
{
//    statet & state = ex_state.get_active_state();
  // iterates over the operands
  exprt::operandst::const_iterator it1=arguments.begin();

  // these are the types of the arguments
  const code_typet::argumentst &argument_types=function_type.arguments();

  // iterates over the types of the arguments
  for(code_typet::argumentst::const_iterator
      it2=argument_types.begin();
      it2!=argument_types.end();
      it2++)
  {
    // if you run out of actual arguments there was a mismatch
    if(it1==arguments.end())
      throw "function call: not enough arguments";

    const code_typet::argumentt &argument=*it2;

    // this is the type the n-th argument should be
    const typet &arg_type=argument.type();

    const irep_idt &identifier=argument.get_identifier();
        
    if(identifier=="")
      throw "no identifier for function argument";

    const symbolt &symbol=ns.lookup(identifier);
    exprt lhs=symbol_expr(symbol);

    if(it1->is_nil())
    {
    }
    else
    {
      exprt rhs=*it1;

      // it should be the same exact type
      if(!base_type_eq(arg_type, rhs.type(), ns))
      {
        const typet &f_arg_type=ns.follow(arg_type);
        const typet &f_rhs_type=ns.follow(rhs.type());
      
        // we are willing to do some limited conversion
        if((f_arg_type.id()==typet::t_signedbv ||
            f_arg_type.id()==typet::t_unsignedbv ||
            f_arg_type.id()==typet::t_bool ||
            f_arg_type.id()==typet::t_pointer ||
            f_arg_type.id()==typet::t_fixedbv) &&
           (f_rhs_type.id()==typet::t_signedbv ||
            f_rhs_type.id()==typet::t_unsignedbv ||
            f_rhs_type.id()==typet::t_bool ||
            f_rhs_type.id()==typet::t_pointer ||
            f_rhs_type.id()==typet::t_fixedbv))
        {
          rhs.make_typecast(arg_type);
        }
        else
        {
          std::string error="function call: argument \""+
            id2string(identifier)+"\" type mismatch: got "+
            it1->type().to_string()+", expected "+
            arg_type.to_string();
          throw error;
        }
      }
      
      do_simplify(rhs);
      assignment(ex_state, lhs, rhs);
    }

    it1++;
  }

  if(function_type.has_ellipsis())
  {
    for(; it1!=arguments.end(); it1++)
    {
    }
  }
  else if(it1!=arguments.end())
  {
    // we got too many arguments, but we will just ignore them
  }
}

/*******************************************************************\

Function: goto_symext::symex_function_call

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_function_call(
  const goto_functionst &goto_functions,
  execution_statet &ex_state,
  const code_function_callt &code)
{
  const exprt &function=code.function();

  if(function.id()==exprt::symbol)
    symex_function_call_symbol(goto_functions, ex_state, code);
  else
    symex_function_call_deref(goto_functions, ex_state, code);
}

/*******************************************************************\

Function: goto_symext::symex_function_call_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_function_call_symbol(
  const goto_functionst &goto_functions,
  execution_statet &ex_state,
  const code_function_callt &code)
{
    statet & state = ex_state.get_active_state();
  target->location(state.guard, state.source);

  assert(code.function().id()==exprt::symbol);

  const irep_idt &identifier=
    code.function().identifier();
    
  if(identifier=="c::CBMC_trace")
  {
    symex_trace(state, code,ex_state.node_id);
  }
  if(has_prefix(id2string(identifier), CPROVER_FKT_PREFIX))
  {
    symex_fkt(state, code);
  }
  else if(has_prefix(id2string(identifier), CPROVER_MACRO_PREFIX))
  {
    symex_macro(state, code);
  }
  else
    symex_function_call_code(goto_functions, ex_state, code);
}

/*******************************************************************\

Function: goto_symext::symex_function_call_code

  Inputs:

 Outputs:

 Purpose: do function call by inlining

\*******************************************************************/

void goto_symext::symex_function_call_code(
  const goto_functionst &goto_functions,
  execution_statet &ex_state,
  const code_function_callt &call)
{
    statet & state = ex_state.get_active_state();
  const irep_idt &identifier=
    to_symbol_expr(call.function()).get_identifier();
  
  // find code in function map
  
  goto_functionst::function_mapt::const_iterator it=
    goto_functions.function_map.find(identifier);

  if(it==goto_functions.function_map.end())
    throw "failed to find `"+id2string(identifier)+"' in function_map";

  const goto_functionst::goto_functiont &goto_function=it->second;
  
  unsigned &unwinding_counter=state.function_unwind[identifier];

  // see if it's too much
  if(get_unwind_recursion(identifier, unwinding_counter))
  {
    if(!options.get_bool_option("no-unwinding-assertions"))
      claim(false_exprt(), "recursion unwinding assertion", state,ex_state.node_id);

    state.source.pc++;
    return;
  }

  if(!goto_function.body_available)
  {
    no_body(identifier);
  
    if(call.lhs().is_not_nil())
    {
      exprt rhs=exprt("nondet_symbol", call.lhs().type());
      rhs.identifier("symex::"+i2string(ex_state.nondet_count++));
      rhs.location()=call.location();
      code_assignt code(call.lhs(), rhs);
      basic_symext::symex(state, ex_state, code, ex_state.node_id);
    }

    state.source.pc++;
    return;
  }
  
  // read the arguments -- before the locality renaming
  exprt::operandst arguments=call.arguments();
  for(unsigned i=0; i<arguments.size(); i++)
  {
    state.rename(arguments[i], ns,ex_state.node_id);
  }

  // increase unwinding counter
  unwinding_counter++;
  
  // produce a new frame
  assert(!state.call_stack.empty());
  goto_symex_statet::framet &frame=state.new_frame(state.source.thread_nr);
  
  // copy L1 renaming from previous frame
  frame.level1=state.previous_frame().level1;
  frame.level1._thread_id = state.source.thread_nr;
  
  unsigned &frame_nr=state.function_frame[identifier];
  frame_nr++;

  frame.calling_location=state.source;
  
  // preserve locality of local variables
  locality(frame_nr, state, goto_function,ex_state.node_id);

  // assign arguments
  argument_assignments(goto_function.type, ex_state, arguments);

  frame.end_of_function=--goto_function.body.instructions.end();
  frame.return_value=call.lhs();
  frame.function_identifier=identifier;

  state.source.is_set=true;
  state.source.pc=goto_function.body.instructions.begin();
  state.source.prog=&goto_function.body;
}

static std::list<std::pair<guardt,exprt> >
get_function_list(const exprt &expr)
{
  std::list<std::pair<guardt,exprt> > l;

  if (expr.id() == "if") {
    std::list<std::pair<guardt,exprt> > l1, l2;
    exprt guardexpr = expr.op0();
    exprt notguardexpr = not_exprt(guardexpr);

    // Get sub items, them iterate over adding the relevant guard
    l1 = get_function_list(expr.op1());
    for (std::list<std::pair<guardt,exprt> >::iterator it = l1.begin();
         it != l1.end(); it++)
      it->first.add(guardexpr);

    l2 = get_function_list(expr.op2());
    for (std::list<std::pair<guardt,exprt> >::iterator it = l2.begin();
         it != l2.end(); it++)
      it->first.add(notguardexpr);

    l1.splice(l1.begin(), l2);
    return l1;
  } else if (expr.id() == "symbol") {
    guardt guard;
    guard.make_true();
    std::pair<guardt,exprt> p(guard, expr);
    l.push_back(p);
    return l;
  } else {
    std::cerr << "Unexpected irep id " << expr.id() << " in function ptr dereference" << std::endl;
    abort();
  }
}

void
goto_symext::symex_function_call_deref(const goto_functionst &goto_functions,
                                       execution_statet &ex_state,
                                       const code_function_callt &call)
{
  statet &state = ex_state.get_active_state();

  assert(state.top().cur_function_ptr_targets.size() == 0);

  // Indirect function call. The value is dereferenced, so we'll get either an
  // address_of a symbol, or a set of if ireps. For symbols we'll invoke
  // symex_function_call_symbol, when dealing with if's we need to fork and
  // merge.
  if (call.op1().is_nil()) {
    std::cerr << "Function pointer call with no targets; irep: ";
    std::cerr << call.pretty(0) << std::endl;
    abort();
  }

  // Generate a list of functions to call. We'll then proceed to call them,
  // and will later on merge them.
  std::list<std::pair<guardt,exprt> > l = get_function_list(call.op1());

  // Store.
  for (std::list<std::pair<guardt,exprt> >::iterator it = l.begin();
       it != l.end(); it++) {

    goto_functionst::function_mapt::const_iterator fit =
      goto_functions.function_map.find(it->second.identifier());
    if (fit == goto_functions.function_map.end() || !fit->second.body_available) {
      std::cerr << "Couldn't find symbol " << it->second.identifier();
      std::cerr << " or body not available, during function ptr dereference";
      std::cerr << std::endl;
      abort();
    }

    // Set up a merge of the current state into the target function.
    statet::goto_state_listt &goto_state_list =
      state.top().goto_state_map[fit->second.body.instructions.begin()];

    state.top().cur_function_ptr_targets.push_back(
      std::pair<goto_programt::const_targett,exprt>(
        fit->second.body.instructions.begin(),
        it->second)
      );

    goto_state_list.push_back(statet::goto_statet(state));
    statet::goto_statet &new_state = goto_state_list.back();
    new_state.guard.add(it->first.as_expr());
  }

  state.top().function_ptr_call_loc = state.source.pc;
  state.top().function_ptr_combine_target = state.source.pc;
  state.top().function_ptr_combine_target++;
  state.top().orig_func_ptr_call = new code_function_callt(call);

  run_next_function_ptr_target(goto_functions, ex_state);
}

bool
goto_symext::run_next_function_ptr_target(const goto_functionst &goto_functions,
                                          execution_statet &ex_state)
{
  statet &state = ex_state.get_active_state();

  if (state.top().cur_function_ptr_targets.size() == 0)
    return false;

  // Record a merge - when all function ptr target runs are completed, they'll
  // be merged into the state when the instruction after the func call is run.
  statet::goto_state_listt &goto_state_list =
    state.top().goto_state_map[state.top().function_ptr_combine_target];
  goto_state_list.push_back(statet::goto_statet(state));

  // Take one function ptr target out of the list and jump to it. A previously
  // recorded merge will ensure it gets the right state.
  std::pair<goto_programt::const_targett,exprt> p =
    state.top().cur_function_ptr_targets.front();
  state.top().cur_function_ptr_targets.pop_front();

  goto_programt::const_targett target = p.first;
  exprt target_symbol = p.second;

  state.guard.make_false();
  state.source.pc = target;

  // Merge pre-function-ptr-call state in immediately.
  merge_gotos(state, ex_state, ex_state.node_id);

  // Now switch back to the original call location so that the call appears
  // to originate from there...
  state.source.pc = state.top().function_ptr_call_loc;

  // And setup the function call.
  code_function_callt call = *state.top().orig_func_ptr_call;
  call.function() = target_symbol;
  goto_symex_statet::framet &cur_frame = state.top();

  if (state.top().cur_function_ptr_targets.size() == 0)
    delete cur_frame.orig_func_ptr_call;

  symex_function_call_code(goto_functions, ex_state, call);


  return true;
}

/*******************************************************************\

Function: goto_symext::pop_frame

  Inputs:

 Outputs:

 Purpose: pop one call frame

\*******************************************************************/

void goto_symext::pop_frame(statet &state)
{
  assert(!state.call_stack.empty());

  statet::framet &frame=state.top();

  // restore state
  state.source.pc=frame.calling_location.pc;
  state.source.prog=frame.calling_location.prog;

  // clear locals from L2 renaming
  for(statet::framet::local_variablest::const_iterator
      it=frame.local_variables.begin();
      it!=frame.local_variables.end();
      it++)
    state.level2.remove(*it);

  // decrease recursion unwinding counter
  if(frame.function_identifier!="")
    state.function_unwind[frame.function_identifier]--;
  
  state.pop_frame();
}

/*******************************************************************\

Function: goto_symext::symex_end_of_function

  Inputs:

 Outputs:

 Purpose: do function call by inlining

\*******************************************************************/

void goto_symext::symex_end_of_function(statet &state)
{
  pop_frame(state);
}

/*******************************************************************\

Function: goto_symext::locality

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::locality(
  unsigned frame_nr,
  statet &state,
  const goto_functionst::goto_functiont &goto_function,
        unsigned exec_node_id)
{
  goto_programt::local_variablest local_identifiers;

  for(goto_programt::instructionst::const_iterator
      it=goto_function.body.instructions.begin();
      it!=goto_function.body.instructions.end();
      it++)
    local_identifiers.insert(
      it->local_variables.begin(),
      it->local_variables.end());

  statet::framet &frame=state.top();

  for(goto_programt::local_variablest::const_iterator
      it=local_identifiers.begin();
      it!=local_identifiers.end();
      it++)
  {
    frame.level1.rename(*it, frame_nr, exec_node_id);
    irep_idt l1_name=frame.level1(*it,exec_node_id);
    frame.local_variables.insert(l1_name);
  }
}

/*******************************************************************\

Function: goto_symext::return_assignment

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::return_assignment(statet &state, execution_statet &ex_state, unsigned node_id)
{
  statet::framet &frame=state.top();

  const goto_programt::instructiont &instruction=*state.source.pc;
  assert(instruction.is_return());
  const code_returnt &code=to_code_return(instruction.code);

  target->location(state.guard, state.source);

  if(code.operands().size()==1)
  {
    exprt value(code.op0());
    
    dereference(value, state, false,node_id);
    
    if(frame.return_value.is_not_nil())
    {
      code_assignt assignment(frame.return_value, value);

      if (assignment.lhs().type()!=assignment.rhs().type())
        assignment.rhs().make_typecast(assignment.lhs().type());

      //make sure that we assign two expressions of the same type
      assert(assignment.lhs().type()==assignment.rhs().type());
      basic_symext::symex_assign(state, ex_state, assignment, node_id);
    }
  }
  else
  {
    if(frame.return_value.is_not_nil())
      throw "return with unexpected value";
  }
}

/*******************************************************************\

Function: goto_symext::symex_return

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_return(statet &state, execution_statet &ex_state, unsigned node_id)
{
  return_assignment(state, ex_state, node_id);

  // we treat this like an unconditional
  // goto to the end of the function

  // put into state-queue
  statet::goto_state_listt &goto_state_list=
    state.top().goto_state_map[state.top().end_of_function];

  goto_state_list.push_back(statet::goto_statet(state));
  
  // kill this one
  state.guard.make_false();
}
