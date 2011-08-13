/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <arith_tools.h>
#include <std_expr.h>

#include "boolbv.h"

/*******************************************************************\

Function: boolbvt::convert_byte_extract

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_byte_extract(const exprt &expr, bvt &bv)
{
  if(expr.operands().size()!=2)
    throw "byte_extract takes two operands";

  unsigned width;
  boolbv_get_width(expr.type(), width);  
  if(width==0) throw "byte_extract without width";
    
  const exprt &op0=expr.op0();
  const exprt &op1=expr.op1();

  bool little_endian;
  
  if(expr.is_byte_extract_little_endian())
    little_endian=true;
  else if(expr.is_byte_extract_big_endian())
    little_endian=false;
  else
    assert(false);
    
  // see if the byte number is constant

  mp_integer index;
  if(!to_integer(op1, index))
    return convert_byte_extract(width, expr.op0(), index, bv, little_endian);

  bvt op0_bv;
  
  convert_bv(op0, op0_bv);

  unsigned bytes=op0_bv.size()/width;
  
  if(prop.has_set_to())
  {
    // free variables

    bv.resize(width);
    for(unsigned i=0; i<width; i++)
      bv[i]=prop.new_variable();

    // add implications

    equality_exprt equality;
    equality.operands().resize(2);
    equality.op0()=op1; // index operand

    typet constant_type=op1.type(); // type of index operand

    bvt equal_bv;
    equal_bv.resize(width);

    for(unsigned i=0; i<bytes; i++)
    {
      equality.op1()=from_integer(i, constant_type);

      unsigned offset=i*width;

      for(unsigned j=0; j<width; j++)
        equal_bv[j]=prop.lequal(bv[j], op0_bv[offset+j]);

      prop.l_set_to_true(
        prop.limplies(convert(equality), prop.land(equal_bv)));
    }
  }
  else
  {
    bv.resize(width);

    equality_exprt equality;
    equality.operands().resize(2);
    equality.op0()=op1; // index operand

    typet constant_type(op1.type()); // type of index operand
    
    for(unsigned i=0; i<bytes; i++)
    {
      equality.op1()=from_integer(i, constant_type);
        
      literalt e=convert(equality);

      unsigned offset=i*width;

      for(unsigned j=0; j<width; j++)
      {
        literalt l=op0_bv[offset+j];

        if(i==0)
          bv[j]=l;
        else
          bv[j]=prop.lselect(e, l, bv[j]);
      }
    }    
  }
}

/*******************************************************************\

Function: boolbvt::convert_byte_extract

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbvt::convert_byte_extract(
  unsigned width,
  const exprt &expr,
  const mp_integer &index,
  bvt &bv,
  bool little_endian)
{
  bv.resize(width);
  
  bvt tmp;
  convert_bv(expr, tmp); // recursive call

  mp_integer offset;
  
  if(little_endian)
    offset=index*width;
  else
    offset=(mp_integer(tmp.size()/width)-index-1)*width;

  if(mp_integer(tmp.size())<offset+width || offset<0)
  {
    // out of bounds
    for(unsigned i=0; i<width; i++)
      bv[i]=prop.new_variable();
  }
  else
  {
    for(unsigned i=0; i<width; i++)
      bv[i]=tmp[integer2long(offset+i)];
  }
}

