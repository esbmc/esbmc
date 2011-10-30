/*******************************************************************\

Module: Pointer Logic

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <expr.h>
#include <arith_tools.h>
#include <std_types.h>

#include "pointer_offset_size.h"

/*******************************************************************\

Function: member_offset

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

mp_integer member_offset(
  const struct_typet &type,
  const irep_idt &member)
{
  const struct_typet::componentst &components=type.components();

  mp_integer result=0;
  unsigned bit_field_bits=0;

  for(struct_typet::componentst::const_iterator
      it=components.begin();
      it!=components.end();
      it++)
  {
    if(it->get_name()==member) break;
    if(it->get_bool("#is_bit_field"))
    {
      bit_field_bits+=binary2integer(it->type().get("width").as_string(), 2).to_long();
    }
    else
    {
      if(bit_field_bits!=0)
      {
        result+=bit_field_bits/8;
        bit_field_bits=0;
      }

      const typet &subtype=it->type();
      mp_integer sub_size=pointer_offset_size(subtype);
      if(sub_size==-1) return -1; // give up
      result+=sub_size;
    }
  }

  return result;
}

/*******************************************************************\

Function: pointer_offset_size

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

mp_integer pointer_offset_size(const typet &type)
{
  if(type.id()=="array")
  {
    mp_integer sub=pointer_offset_size(type.subtype());
  
    // get size
    const exprt &size=(const exprt &)type.find("size");

    // constant?
    mp_integer i;
    
    if(to_integer(size, i))
      return mp_integer(1); // we cannot distinguish the elements
    
    return sub*i;
  }
  else if(type.id()=="struct" ||
          type.id()=="union")
  {
    const irept::subt &components=type.find("components").get_sub();
    
    mp_integer result=1; // for the struct itself

    forall_irep(it, components)
    {
      const typet &subtype=(typet &)it->find("type");
      mp_integer sub_size=pointer_offset_size(subtype);
      result+=sub_size;
    }
    
    return result;
  }
  else
    return mp_integer(1);
}

/*******************************************************************\

Function: compute_pointer_offset

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

mp_integer compute_pointer_offset(
  const namespacet &ns,
  const exprt &expr)
{
  if(expr.id()=="symbol")
    return 0;
  else if(expr.id()=="index")
  {
    assert(expr.operands().size()==2);
    assert(expr.op0().type().id()=="array");
    mp_integer sub_size=pointer_offset_size(expr.op0().type().subtype());

    mp_integer i;

    if(!to_integer(expr.op1(), i))
      return i*sub_size+1;
    
    return 0; // TODO
  }
  else if(expr.id()=="member")
  {
    assert(expr.operands().size()==1);
    const typet &type=ns.follow(expr.op0().type());
    
    assert(type.id()=="struct" ||
           type.id()=="union");

    const irep_idt &component_name=expr.get("component_name");
    const struct_union_typet::componentst &components=
      to_struct_union_type(type).components();
    
    mp_integer result=1; // for the struct itself

    for(struct_union_typet::componentst::const_iterator
        it=components.begin();
        it!=components.end();
        it++)
    {
      if(it->get_name()==component_name) return result;
      const typet &subtype=it->type();
      result+=pointer_offset_size(subtype);
    }
    
    assert(false);
  }
  else if(expr.id()=="string-constant")
  {
    return 0;
  }
  else
  {
    std::cout << expr.pretty() << std::endl;
    assert(false);
  }
  
  return 0;
}
