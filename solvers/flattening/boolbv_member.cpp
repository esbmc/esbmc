/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "boolbv.h"

/*******************************************************************\

Function: boolbvt::convert_member

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_member(const exprt &expr, bvt &bv)
{
  const exprt::operandst &operands=expr.operands();

  if(operands.size()!=1)
    throw "member takes one operand";

  bvt op0_bv;

  convert_bv(expr.op0(), op0_bv);

  if(operands[0].type().id()=="union")
  {
    unsigned width;

    if(boolbv_get_width(expr.type(), width))
      return conversion_failed(expr, bv);

    bv.resize(width);

    if(width>op0_bv.size())
      throw "member: unexpected widths";

    for(unsigned i=0; i<width; i++)
      bv[i]=op0_bv[i];

    return;
  }

  if(operands[0].type().id()!="struct")
    throw "member takes struct operand";

  const irep_idt &component_name=expr.component_name();
  const struct_typet::componentst &components=
    to_struct_type(operands[0].type()).components();

  unsigned offset=0;

  for(struct_typet::componentst::const_iterator
      it=components.begin();
      it!=components.end();
      it++)
  {
    unsigned sub_width;

    const typet &subtype=it->type();

    if(boolbv_get_width(subtype, sub_width))
      sub_width=0;

    if(it->get_name()==component_name)
    {
      if(subtype!=expr.type())
      {
        #if 0
        std::cout << "DEBUG " << expr.pretty() << "\n";
        #endif

        throw "member: component type does not match: "+
          subtype.to_string()+" vs. "+
          expr.type().to_string();
      }

      bv.resize(sub_width);
      for(unsigned i=0; i<sub_width; i++)
        bv[i]=op0_bv[offset+i];

      return;
    }

    offset+=sub_width;
  }

  throw "component "+id2string(component_name)+" not found in structure";
}
