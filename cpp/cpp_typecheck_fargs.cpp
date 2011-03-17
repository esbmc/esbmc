/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <assert.h>

#include <std_types.h>

#include <ansi-c/c_qualifiers.h>

#include "cpp_typecheck_fargs.h"
#include "cpp_typecheck.h"

/*******************************************************************\

Function: cpp_typecheck_fargst::has_class_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cpp_typecheck_fargst::has_class_type() const
{
  for(exprt::operandst::const_iterator it=operands.begin();
      it!=operands.end();
      it++)
  {
    if(it->type().id()=="struct")
      return true;
  }

  return false;
}

/*******************************************************************\

Function: cpp_typecheck_fargst::build

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheck_fargst::build(
  const side_effect_expr_function_callt &function_call)
{
  in_use=true;

  operands.clear();
  operands.reserve(function_call.op1().operands().size());

  for(unsigned i=0; i<function_call.op1().operands().size(); i++)
    operands.push_back(function_call.op1().operands()[i]);
}

/*******************************************************************\

Function: cpp_typecheck_fargst::exact_match

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cpp_typecheck_fargst::match(
  const code_typet::argumentst &arguments,
  unsigned &distance,
  cpp_typecheckt &cpp_typecheck) const
{
  distance=0;

  exprt::operandst ops = operands;

  if(arguments.size() > ops.size())
  {
    // do ellipsis
    ops.reserve(arguments.size());
    for(unsigned i = ops.size(); i < arguments.size();i++)
    {
      const exprt& default_value =
        arguments[i].default_value();

      if(default_value.is_nil())
        return false;
      ops.push_back(default_value);
    }
  }
  else if(arguments.size() < ops.size())
    return false;

  for(unsigned i=0; i<ops.size(); i++)
  {
    exprt argument = arguments[i];

    exprt& operand=ops[i];

   assert(!is_reference(operand.type()));

    // "this" is a special case
    if(i==0 && argument.get("#base_name")=="this")
    {
      argument.type().set("#reference",true);
      argument.type().set("#this",true);
    }

    unsigned rank = 0;
    exprt new_expr;

    if(!cpp_typecheck.implicit_conversion_sequence(operand,
          argument.type(),
          new_expr, rank))
    {
      return false;
    }
    distance += rank;
  }

  return true;
}
