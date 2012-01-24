/*******************************************************************\

Module: Traces of GOTO Programs

Author: Daniel Kroening

  Date: July 2005

\*******************************************************************/

#include <assert.h>

#include "build_goto_trace.h"

/*******************************************************************\

Function: build_goto_trace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void build_goto_trace(
  const symex_target_equationt &target,
  const prop_convt &prop_conv,
  goto_tracet &goto_trace)
{
  unsigned step_nr=0;

  for(symex_target_equationt::SSA_stepst::const_iterator
      it=target.SSA_steps.begin();
      it!=target.SSA_steps.end();
      it++)
  {
    const symex_target_equationt::SSA_stept &SSA_step=*it;
    tvt result;
    result=prop_conv.prop.l_get(SSA_step.guard_literal);

    //std::cout << "SSA_step.comment: " << SSA_step.comment << "\n";

    //if (result==tvt(true) && SSA_step.comment.compare("arithmetic overflow on *")==0)
      //result = tvt(false);
    //std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
    if(result!=tvt(true) && result!=tvt(tvt::TV_ASSUME))
      continue;
    //std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
    if(it->is_assignment() &&
       SSA_step.assignment_type==symex_target_equationt::HIDDEN)
      continue;
    //std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
    step_nr++;

    goto_trace.steps.push_back(goto_trace_stept());
    goto_trace_stept &goto_trace_step=goto_trace.steps.back();

    goto_trace_step.thread_nr=SSA_step.source.thread_nr;
    goto_trace_step.lhs=SSA_step.lhs;
    goto_trace_step.rhs=SSA_step.rhs;
    goto_trace_step.pc=SSA_step.source.pc;
    goto_trace_step.comment=SSA_step.comment;
    goto_trace_step.original_lhs=SSA_step.original_lhs;
    goto_trace_step.type=SSA_step.type;
    goto_trace_step.step_nr=step_nr;
    goto_trace_step.format_string=SSA_step.format_string;
    goto_trace_step.stack_trace = SSA_step.stack_trace;
    //std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
    if(SSA_step.lhs.is_not_nil())
      goto_trace_step.value=prop_conv.get(SSA_step.lhs);

    for(std::list<exprt>::const_iterator
        j=SSA_step.converted_output_args.begin();
        j!=SSA_step.converted_output_args.end();
        j++)
    {
      const exprt &arg=*j;
      if(arg.is_constant() ||
         arg.id()=="string-constant")
        goto_trace_step.output_args.push_back(arg);
      else
      {
        exprt tmp=prop_conv.get(arg);
        goto_trace_step.output_args.push_back(tmp);
      }
    }

    //std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";

    if(SSA_step.is_assert() ||
       SSA_step.is_assume())
    {
      result = prop_conv.prop.l_get(SSA_step.cond_literal);
      //std::cout << "SSA_step.cond_literal: " << SSA_step.cond_literal.var_no() << "\n";
      if ((result==tvt(tvt::TV_ASSUME) && SSA_step.comment.compare("arithmetic overflow on *")==0) ||
    	 (result==tvt(false) && SSA_step.comment.compare("arithmetic overflow on *")==0))
        goto_trace_step.guard=true;
      else if (result==tvt(tvt::TV_ASSUME) && SSA_step.comment.compare("unwinding assertion loop")==0)
    	goto_trace_step.guard=false;
      else if (result==tvt(tvt::TV_UNKNOWN))
    	goto_trace_step.guard=true;
      else
        goto_trace_step.guard=result.is_true();

      if(!goto_trace_step.guard)
      {
    	if (!SSA_step.is_assert())
    	{
      	  //goto_trace_step.comment="unwinding assertion loop";
      	  goto_trace_step.type=goto_trace_stept::ASSERT;
      	  //std::cout << "\n" << __FUNCTION__ << "[" << __LINE__ << "]" << "\n";
    	}
    	//assert(SSA_step.is_assert());
        break;
      }
    }
  }
}

