/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "boolbv.h"

/*******************************************************************\

Function: boolbvt::convert_shift

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_shift(const exprt &expr, bvt &bv)
{
  if(expr.type().id()!="unsignedbv" &&
     !expr.type().is_signedbv())
    return conversion_failed(expr, bv);

  unsigned width;

  boolbv_get_width(expr.type(), width);

  if(width==0)
    throw "zero length bit vector type: "+expr.type().to_string();

  if(expr.operands().size()!=2)
    throw "shifting takes two operands";

  bvt op, dist;

  convert_bv(expr.op0(), op);
  convert_bv(expr.op1(), dist);

  if(op.size()!=width)
    throw "convert_shift: unexpected operand width";

  bv_utilst::shiftt shift;

  if(expr.id()=="shl")
    shift=bv_utilst::LEFT;
  else if(expr.id()=="ashr")
    shift=bv_utilst::ARIGHT;
  else if(expr.id()=="lshr")
    shift=bv_utilst::LRIGHT;
  else
    throw "unexpected operand";

  bv=bv_utils.shift(op, shift, dist);
}
