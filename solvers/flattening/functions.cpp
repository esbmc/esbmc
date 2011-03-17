/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

//#define DEBUG

#include <assert.h>

#include "functions.h"

/*******************************************************************\

Function: functionst::record_function_application

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void functionst::record_function_application(const exprt &expr)
{
  if(expr.operands().size()!=2)
    throw "function application expected to have two operands";

  const exprt &function_expr=expr.op0();

  function_map[function_expr].applications.insert(expr);
}

/*******************************************************************\

Function: functionst::add_function_constraints

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void functionst::add_function_constraints()
{
  for(function_mapt::const_iterator it=
      function_map.begin();
      it!=function_map.end();
      it++)
    add_function_constraints(it->second);
}

/*******************************************************************\

Function: functionst::add_function_constraints

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void functionst::add_function_constraints(const function_infot &info)
{
  // do Ackermann's function reduction

  for(std::set<exprt>::const_iterator it1=info.applications.begin();
      it1!=info.applications.end();
      it1++)
  {
    for(std::set<exprt>::const_iterator it2=info.applications.begin();
        it2!=it1;
        it2++)
    {
      exprt arguments_equal("=", typet("bool"));
      arguments_equal.operands().resize(2);
      arguments_equal.op0()=it1->op1();
      arguments_equal.op1()=it2->op1();

      if(arguments_equal.op0().type()!=
         arguments_equal.op1().type())
      {
        arguments_equal.op1().
          make_typecast(arguments_equal.op0().type());
      }
      
      literalt arguments_equal_lit=convert(arguments_equal);
      
      if(arguments_equal_lit!=const_literal(false))
      {
        exprt values_equal("=", typet("bool"));
        values_equal.copy_to_operands(*it1, *it2);

        bvt implication;
        implication.reserve(2);
        implication.push_back(prop.lnot(arguments_equal_lit));
        implication.push_back(convert(values_equal));
        prop.lcnf(implication);
      }
    }
  }
}
