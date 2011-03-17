/*******************************************************************\

Module: 

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "std_code.h"

/*******************************************************************\

Function: codet::make_block

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

code_blockt &codet::make_block()
{
  if(get_statement()=="block") return (code_blockt &)*this;

  exprt tmp;
  tmp.swap(*this);

  *this=codet("block");
  set_statement("block");
  move_to_operands(tmp);
  
  return (code_blockt &)*this;
}

/*******************************************************************\

Function: codet::first_statement

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

codet &codet::first_statement()
{
  const irep_idt &statement=get_statement();

  if(has_operands())
  {
    if(statement=="block")
      return to_code(op0()).first_statement();
    else if(statement=="label")
      return to_code(op0()).first_statement();
  }

  return *this;
}

/*******************************************************************\

Function: first_statement

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const codet &codet::first_statement() const
{
  const irep_idt &statement=get_statement();

  if(has_operands())
  {
    if(statement=="block")
      return to_code(op0()).first_statement();
    else if(statement=="label")
      return to_code(op0()).first_statement();
  }

  return *this;
}

/*******************************************************************\

Function: codet::last_statement

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

codet &codet::last_statement()
{
  const irep_idt &statement=get_statement();

  if(has_operands())
  {
    if(statement=="block")
      return to_code(operands().back()).last_statement();
    else if(statement=="label")
      return to_code(operands().back()).last_statement();
  }

  return *this;
}

/*******************************************************************\

Function: codet::last_statement

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const codet &codet::last_statement() const
{
  const irep_idt &statement=get_statement();

  if(has_operands())
  {
    if(statement=="block")
      return to_code(operands().back()).last_statement();
    else if(statement=="label")
      return to_code(operands().back()).last_statement();
  }

  return *this;
}
