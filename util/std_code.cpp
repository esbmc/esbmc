/*******************************************************************\

Module: 

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "std_code.h"

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
