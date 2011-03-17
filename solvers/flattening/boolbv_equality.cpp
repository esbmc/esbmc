/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <i2string.h>

#include <std_expr.h>

#include "boolbv.h"

/*******************************************************************\

Function: boolbvt::convert_equality

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolbvt::convert_equality(const equality_exprt &expr)
{
  if(expr.op0().type()!=expr.op1().type())
  {
    std::cout << expr.pretty() << std::endl;
    throw "equality without matching types";
  }

  // see if it is an unbounded array
  if(is_unbounded_array(expr.op0().type()))
    return record_array_equality(expr);

  bvt bv0, bv1;
  
  convert_bv(expr.op0(), bv0);
  convert_bv(expr.op1(), bv1);
    
  if(bv0.size()!=bv1.size())
    throw "unexpected size mismatch on equality";

  if(bv0.size()==0)
    throw "got zero-size BV";

  if(expr.op0().type().id()=="verilogbv")
  {
    // TODO
  }
  else
    return bv_utils.equal(bv0, bv1);
  
  return SUB::convert_rest(expr);
}
