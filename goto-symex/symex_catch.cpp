/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <simplify_expr.h>

#include "goto_symex.h"

/*******************************************************************\

Function: goto_symext::symex_catch

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_catch()
{
  // there are two variants: 'push' and 'pop'
  const goto_programt::instructiont &instruction=*cur_state->source.pc;

  if(instruction.targets.empty()) // pop
  {

  }
  else // push
  {
    goto_symex_statet::exceptiont exception;

    // copy targets
    const irept::subt &exception_list=
      instruction.code.find("exception_list").get_sub();

    assert(exception_list.size()==instruction.targets.size());

    // Fill the map with the catch type and the target
    unsigned i=0;
    for(goto_programt::targetst::const_iterator
        it=instruction.targets.begin();
        it!=instruction.targets.end();
        it++, i++)
      exception.catch_map[exception_list[i].id()]=*it;

    // Stack it
    stack_catch.push(exception);
  }
}

/*******************************************************************\

Function: goto_symext::symex_throw

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_throw()
{
  const goto_programt::instructiont &instruction= *cur_state->source.pc;

  // Get the list of exceptions thrown
  const irept::subt &exceptions_thrown=
    instruction.code.find("exception_list").get_sub();

  // Get the list of catchs
  goto_symex_statet::exceptiont except=stack_catch.top();

  // We check before iterate over the throw list to save time:
  // If there is no catch, we return an error
  if(!except.catch_map.size())
    exception_error(exceptions_thrown.begin()->id(),
      goto_symex_statet::exceptiont::NOCATCH);

}

/*******************************************************************\

Function: goto_symext::update_throw_target

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::exception_error(const irep_idt &id,
  goto_symex_statet::exceptiont::ERROR error)
{
  switch(error)
  {

  case goto_symex_statet::exceptiont::NOCATCH:
  {
    // An un-caught exception. Error
    const std::string &msg="Throwing an exception of type " +
      id.as_string() + " but there is not catch for it.";
    claim(false_exprt(), msg);

    break;
  }

  case goto_symex_statet::exceptiont::NOTALLOWED:
    break;
  }
}


/*******************************************************************\

Function: goto_symext::update_throw_target

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::update_throw_target(goto_symex_statet::framet* frame,
  goto_symex_statet::exceptiont::catch_mapt::const_iterator c_it)
{

}

/*******************************************************************\

Function: goto_symext::handle_throw_decl

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::handle_throw_decl(goto_symex_statet::framet* frame,
  const irep_idt &id)
{

}

/*******************************************************************\

Function: goto_symext::handle_rethrow

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_symext::handle_rethrow(irept::subt exceptions_thrown,
  const goto_programt::instructiont instruction)
{

}

/*******************************************************************\

Function: goto_symext::symex_throw_decl

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_throw_decl()
{

}
