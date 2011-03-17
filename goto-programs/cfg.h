/*******************************************************************\

Module: Control Flow Graph

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAMS_CFG_H
#define CPROVER_GOTO_PROGRAMS_CFG_H

#include <std_expr.h>

/*******************************************************************\

   Class: cfgt

 Purpose:

\*******************************************************************/

template<class T>
class cfgt
{
public:
  typedef goto_programt::const_targetst successorst;
  typedef goto_programt::const_targetst predecessorst;

  struct entryt: public T
  {
    // we have edges both ways
    successorst successors;
    predecessorst predecessors;
  };
  
  typedef std::map<goto_programt::const_targett, entryt> entriest;
  entriest entries;

protected:
  void compute_edges(
    const goto_functionst &goto_functions,
    const goto_programt &goto_program,
    goto_programt::const_targett PC);

  void compute_edges(
    const goto_functionst &goto_functions,
    const goto_programt &goto_program);

  void compute_edges(
    const goto_functionst &goto_functions);

public:
  void operator()(
    const goto_functionst &goto_functions)
  {
    compute_edges(goto_functions);
  }

  void operator()(
    const goto_programt &goto_program)
  {
    goto_functionst goto_functions;
    compute_edges(goto_functions, goto_program);
  }

};

/*******************************************************************\

Function: cfgt::compute_edges

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

template<class T>
void cfgt<T>::compute_edges(
  const goto_functionst &goto_functions,
  const goto_programt &goto_program,
  goto_programt::const_targett PC)
{
  const goto_programt::instructiont &instruction=*PC;
  entryt &entry=entries[PC];
  goto_programt::const_targett next_PC(PC);
  next_PC++;

  // compute forward edges first
  if(instruction.is_goto())
  {
    if(next_PC!=goto_program.instructions.end() &&
       !instruction.guard.is_true())
      entry.successors.push_back(next_PC);

    for(goto_programt::instructiont::targetst::const_iterator
        t_it=instruction.targets.begin();
        t_it!=instruction.targets.end();
        t_it++)
    {
      goto_programt::const_targett t=*t_it;
      if(t!=goto_program.instructions.end())
        entry.successors.push_back(t);
    }
  }
  else if(instruction.is_start_thread())
  {
    if(next_PC!=goto_program.instructions.end())
      entry.successors.push_back(PC);
      
    for(goto_programt::instructiont::targetst::const_iterator
        t_it=instruction.targets.begin();
        t_it!=instruction.targets.end();
        t_it++)
    {
      goto_programt::const_targett t=*t_it;
      if(t!=goto_program.instructions.end())
        entry.successors.push_back(t);
    }
  }
  else if(instruction.is_function_call())
  {
    const exprt &function=
      to_code_function_call(instruction.code).function();

    if(function.id()=="symbol")
    {
      const irep_idt &identifier=
        to_symbol_expr(function).get_identifier();

      goto_functionst::function_mapt::const_iterator f_it=
        goto_functions.function_map.find(identifier);

      if(f_it!=goto_functions.function_map.end() &&
         f_it->second.body_available)
      {
        // get the first instruction
        goto_programt::const_targett i_it=
          f_it->second.body.instructions.begin();

        goto_programt::const_targett e_it=
          f_it->second.body.instructions.end();

        if(i_it!=e_it)
          entry.successors.push_back(i_it);

        // add the last instruction as predecessor of the return location
        if(next_PC!=goto_program.instructions.end())
        {
          entries[--e_it].successors.push_back(next_PC);
          entries[next_PC].predecessors.push_back(--e_it);
        }
      }
      else if(next_PC!=goto_program.instructions.end())
        entry.successors.push_back(next_PC);
    }
  }
  else
  {
    if(next_PC!=goto_program.instructions.end())
      entry.successors.push_back(next_PC);
  }

  // now do backward edges
  for(successorst::const_iterator
      s_it=entry.successors.begin();
      s_it!=entry.successors.end();
      s_it++)
  {
    entries[*s_it].predecessors.push_back(PC);
  }
}

/*******************************************************************\

Function: cfgt::compute_edges

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

template<class T>
void cfgt<T>::compute_edges(
  const goto_functionst &goto_functions,
  const goto_programt &goto_program)
{
  forall_goto_program_instructions(it, goto_program)
    compute_edges(goto_functions, goto_program, it);
}

/*******************************************************************\

Function: cfgt::compute_edges

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

template<class T>
void cfgt<T>::compute_edges(
  const goto_functionst &goto_functions)
{
  forall_goto_functions(it, goto_functions)
    if(it->second.body_available)
      compute_edges(goto_functions, it->second.body);
}

#endif
