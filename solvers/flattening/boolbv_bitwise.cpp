/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "boolbv.h"

/*******************************************************************\

Function: boolbvt::convert_bitwise

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_bitwise(const exprt &expr, bvt &bv)
{
  unsigned width;
  if(boolbv_get_width(expr.type(), width))
    return conversion_failed(expr, bv);

  if(expr.id()=="bitnot")
  {
    if(expr.operands().size()!=1)
      throw "bitnot takes one operand";

    const exprt &op0=expr.op0();
    
    bvt op_bv;
    convert_bv(op0, op_bv);

    bv.resize(width);
    
    if(op_bv.size()!=width)
      throw "convert_bitwise: unexpected operand width";

    for(unsigned i=0; i<width; i++)
      bv[i]=prop.lnot(op_bv[i]);

    return;
  }
  else if(expr.is_bitand() || expr.id()=="bitor" ||
          expr.id()=="bitxor")
  {
    bv.resize(width);

    for(unsigned i=0; i<width; i++)
      bv[i]=const_literal(expr.is_bitand());
    
    forall_operands(it, expr)
    {
      bvt op;

      convert_bv(*it, op);

      if(op.size()!=width)
        throw "convert_bitwise: unexpected operand width";

      for(unsigned i=0; i<width; i++)
      {
        if(expr.is_bitand())
          bv[i]=prop.land(bv[i], op[i]);
        else if(expr.id()=="bitor")
          bv[i]=prop.lor(bv[i], op[i]);
        else if(expr.id()=="bitxor")
          bv[i]=prop.lxor(bv[i], op[i]);
        else
          throw "unexpected operand";
      }
    }    

    return;
  }
 
  throw "unexpected bitwise operand";
}
