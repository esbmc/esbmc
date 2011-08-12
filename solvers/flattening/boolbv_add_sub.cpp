/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <std_types.h>

#include "boolbv.h"

#ifdef HAVE_FLOATBV
#include "../floatbv/float_utils.h"
#endif

/*******************************************************************\

Function: boolbvt::convert_add_sub

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_add_sub(const exprt &expr, bvt &bv)
{
  if(expr.type().id()!="unsignedbv" &&
     expr.type().id()!="signedbv" &&
     !expr.type().is_fixedbv() &&
     !expr.type().is_floatbv() &&
     expr.type().id()!="range")
    return conversion_failed(expr, bv);

  unsigned width;
  if(boolbv_get_width(expr.type(), width))
    return conversion_failed(expr, bv);

  const exprt::operandst &operands=expr.operands();

  if(operands.size()==0)
    throw "operand "+expr.id_string()+" takes at least one operand";

  const exprt &op0=expr.op0();

  if(op0.type()!=expr.type())
  {
    std::cerr << expr.pretty() << std::endl;
    throw "add/sub with mixed types";
  }

  convert_bv(op0, bv);

  if(bv.size()!=width)
    throw "convert_add_sub: unexpected operand width";

  bool subtract=(expr.id()=="-" ||
                 expr.id()=="no-overflow-minus");
                 
  bool no_overflow=(expr.id()=="no-overflow-plus" ||
                    expr.id()=="no-overflow-minus");

  bv_utilst::representationt rep=
    (expr.type().id()=="signedbv" ||
     expr.type().is_fixedbv())?bv_utilst::SIGNED:
                                  bv_utilst::UNSIGNED;

  for(exprt::operandst::const_iterator it=operands.begin()+1;
      it!=operands.end(); it++)
  {
    bvt op;
    
    if(it->type()!=expr.type())
    {
      std::cerr << expr.pretty() << std::endl;
      throw "add/sub with mixed types";
    }

    convert_bv(*it, op);

    if(op.size()!=width)
      throw "convert_add_sub: unexpected operand width";

    if(expr.type().is_floatbv())
    {
      #ifdef HAVE_FLOATBV
      float_utilst float_utils(prop);
      float_utils.spec=to_floatbv_type(expr.type());
      bv=float_utils.add_sub(bv, op, subtract);
      #else
      return conversion_failed(expr, bv);
      #endif
    }
    else if(no_overflow)
      bv=bv_utils.add_sub_no_overflow(bv, op, subtract, rep);
    else
      bv=bv_utils.add_sub(bv, op, subtract);
  }
}

