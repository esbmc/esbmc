/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "std_expr.h"
#include "expr_util.h"

#include "pointer_arithmetic.h"

pointer_arithmetict::pointer_arithmetict(const exprt &src)
{
  pointer.make_nil();
  offset.make_nil();
  read(src);
}

void pointer_arithmetict::read(const exprt &src)
{
  if(src.id()=="+")
  {
    forall_operands(it, src)
    {
      if(it->type().id()=="pointer")
        read(*it);
      else
        add_to_offset(*it);
    }
  }
  else if(src.id()=="-")
  {
    assert(src.operands().size()==2);
    read(src.op0());
    exprt o=exprt("uminus", src.op1().type());
    o.copy_to_operands(src.op1());
    add_to_offset(o);
  }
  else if(src.is_address_of())
  {
    assert(src.operands().size()==1);
    if(src.op0().id()=="index")
    {
      const index_exprt &index_expr=
        to_index_expr(src.op0());

      if(index_expr.index().is_zero())
        make_pointer(src);
      else
      {
        add_to_offset(index_expr.index());
        // produce &x[0] + i instead of &x[i]
        exprt new_src=src;
        new_src.op0().op1()=gen_zero(index_expr.index().type());
        make_pointer(new_src);
      }
    }
    else
      make_pointer(src);
  }
  else
    make_pointer(src);
}

void pointer_arithmetict::add_to_offset(const exprt &src)
{
  if(offset.is_nil())
    offset=src;
  else if(offset.id()=="+")
    offset.copy_to_operands(src);
  else
  {
    exprt new_offset=exprt("+", offset.type());
    new_offset.copy_to_operands(offset, src);
    if(new_offset.op1().type()!=offset.type())
      new_offset.op1().make_typecast(offset.type());
  }
}

void pointer_arithmetict::make_pointer(const exprt &src)
{
  if(pointer.is_nil())
    pointer=src;
  else
    add_to_offset(src);
}

