/*******************************************************************\

Module: Slicer

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <stack>

#include <i2string.h>

#include "slicer.h"
#include "remove_skip.h"
#include "cfg.h"

/*******************************************************************\

   Class: slicert

 Purpose:

\*******************************************************************/

class slicert
{
public:
  void operator()(goto_functionst &goto_functions)
  {
    cfg(goto_functions);
    fixedpoints();
    slice(goto_functions);
  }

  void operator()(goto_programt &goto_program)
  {
    cfg(goto_program);
    fixedpoints();
    slice(goto_program);
  }

protected:
  struct slicer_entryt
  {
    slicer_entryt():reaches_assertion(false), threaded(false)
    {
    }

    bool reaches_assertion, threaded;
  };

  typedef cfgt<slicer_entryt> slicer_cfgt;
  slicer_cfgt cfg;

  typedef std::stack<goto_programt::const_targett> queuet;

  void fixedpoint_assertions();
  void fixedpoint_threads();

  void fixedpoints()
  {
    // do threads first
    fixedpoint_threads();
    fixedpoint_assertions();
  }

  void slice(goto_programt &goto_program);
  void slice(goto_functionst &goto_functions);
};

/*******************************************************************\

Function: slicert::fixedpoint_assertions

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void slicert::fixedpoint_assertions()
{
  queuet queue;

  for(slicer_cfgt::entriest::iterator
      e_it=cfg.entries.begin();
      e_it!=cfg.entries.end();
      e_it++)
    if(e_it->first->is_assert() ||
       e_it->second.threaded)
      queue.push(e_it->first);

  while(!queue.empty())
  {
    goto_programt::const_targett t=queue.top();
    queue.pop();

    slicer_cfgt::entryt &e=cfg.entries[t];

    if(e.reaches_assertion) continue;

    e.reaches_assertion=true;
    
    for(slicer_cfgt::predecessorst::const_iterator
        p_it=e.predecessors.begin();
        p_it!=e.predecessors.end();
        p_it++)
    {
      queue.push(*p_it);
    }
  }
}

/*******************************************************************\

Function: slicert::fixedpoint_threads

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void slicert::fixedpoint_threads()
{
  queuet queue;

  for(slicer_cfgt::entriest::iterator
      e_it=cfg.entries.begin();
      e_it!=cfg.entries.end();
      e_it++)
    if(e_it->first->is_start_thread())
      queue.push(e_it->first);

  while(!queue.empty())
  {
    goto_programt::const_targett t=queue.top();
    queue.pop();

    slicer_cfgt::entryt &e=cfg.entries[t];

    if(e.threaded) continue;

    e.threaded=true;
    
    for(slicer_cfgt::successorst::const_iterator
        p_it=e.successors.begin();
        p_it!=e.successors.end();
        p_it++)
    {
      queue.push(*p_it);
    }
  }
}

/*******************************************************************\

Function: slicert::slice

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void slicert::slice(goto_functionst &goto_functions)
{
  // now remove those instructions that do not reach any assertions

  Forall_goto_functions(f_it, goto_functions)
    if(f_it->second.body_available)
      Forall_goto_program_instructions(i_it, f_it->second.body)
      {
        const slicer_cfgt::entryt &e=cfg.entries[i_it];
        if(!e.reaches_assertion &&
           !i_it->is_end_function())
          i_it->make_skip();
      }
  
  // remove the skips
  remove_skip(goto_functions);
  goto_functions.update();
}

/*******************************************************************\

Function: slicert::slice

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void slicert::slice(goto_programt &goto_program)
{
  // now remove those instructions that do not reach any assertions

  Forall_goto_program_instructions(it, goto_program)
  {
    const slicer_cfgt::entryt &e=cfg.entries[it];
    if(!e.reaches_assertion)
      it->make_skip();
  }
  
  // remove the skips
  remove_skip(goto_program);
  goto_program.update();
}

/*******************************************************************\

Function: slicer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void slicer(goto_programt &goto_program)
{
  slicert()(goto_program);
}

/*******************************************************************\

Function: slicer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void slicer(goto_functionst &goto_functions)
{
  slicert()(goto_functions);
}
