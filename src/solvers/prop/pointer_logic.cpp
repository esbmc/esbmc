/*******************************************************************\

Module: Pointer Logic

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <config.h>
#include <i2string.h>
#include <arith_tools.h>
#include <type_byte_size.h>
#include <std_expr.h>
#include <prefix.h>

#include "pointer_logic.h"

/*******************************************************************\

Function: pointer_logict::add_object

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

unsigned pointer_logict::add_object(const expr2tc &expr)
{
  // remove any index/member
  
  if (expr->expr_id == expr2t::index_id)
  {
    const index2t &index = static_cast<const index2t &>(*expr.get());
    return add_object(index.source_value);
  }
  else if (expr->expr_id == expr2t::member_id)
  {
    const member2t &memb = static_cast<const member2t &>(*expr.get());
    return add_object(memb.source_value);
  }
  std::pair<objectst::iterator, bool> ret = objects.insert(
                                     std::pair<expr2tc,unsigned int>(expr, 0));
  if (!ret.second)
    return ret.first->second;

  // Initialize object number.
  ret.first->second = objects.size() - 1;
  lookup.push_back(expr);
  assert(lookup.size() == objects.size());
  return objects.size() - 1 + obj_num_offset;
}

/*******************************************************************\

Function: pointer_logict::pointer_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

expr2tc pointer_logict::pointer_expr(
  unsigned object,
  const type2tc &type) const
{
  pointert pointer(object, 0);
  return pointer_expr(pointer, type);
}

/*******************************************************************\

Function: pointer_logict::pointer_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

expr2tc pointer_logict::pointer_expr(
  const pointert &pointer,
  const type2tc &type) const
{
  type2tc pointer_type(new pointer_type2t(type2tc(new empty_type2t())));

  if(pointer.object==null_object) // NULL?
  {
    return symbol2tc(type, irep_idt("NULL"));
  }
  else if(pointer.object==invalid_object) // INVALID?
  {
    return symbol2tc(type, irep_idt("INVALID"));
  }
  
  if(pointer.object>=objects.size())
  {
    return symbol2tc(type, irep_idt("INVALID" + i2string(pointer.object)));
  }

  const expr2tc &object_expr = lookup[pointer.object];

  expr2tc deep_object = object_rec(pointer.offset, type, object_expr);
  
  assert(type->type_id == type2t::pointer_id);
  return address_of2tc(type, deep_object);
}

/*******************************************************************\

Function: pointer_logict::object_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

expr2tc pointer_logict::object_rec(
  const mp_integer &offset,
  const type2tc &pointer_type,
  const expr2tc &src) const
{

  if(src->type->type_id == type2t::array_id)
  {
    const array_type2t &arrtype = dynamic_cast<const array_type2t&>
                                              (*src->type.get());
    mp_integer size=type_byte_size(arrtype.subtype);

    if (size == 0)
      return src;
    
    mp_integer index=offset/size;
    mp_integer rest=offset%size;

    type2tc inttype(new unsignedbv_type2t(config.ansi_c.int_width));
    index2tc newindex(arrtype.subtype, src, constant_int2tc(inttype, index));
    
    return object_rec(rest, pointer_type, newindex);
  }
  else if (is_structure_type(src))
  {
    const struct_union_data &data_ref =
      dynamic_cast<const struct_union_data &>(*src->type);
    const std::vector<type2tc> &members = data_ref.get_structure_members();
    const std::vector<irep_idt> &member_names =
      data_ref.get_structure_member_names();

    assert(offset>=0);
  
    if(offset==0) // the struct itself
      return src;

    mp_integer current_offset=1;

    assert(offset>=current_offset);

    unsigned int idx = 0;
    forall_types(it, members) {
      assert(offset>=current_offset);

      mp_integer sub_size=type_byte_size(*it);

      if(sub_size==0)
        return src;
      
      mp_integer new_offset=current_offset+sub_size;

      if(new_offset>offset)
      {
        // found it
        member2tc tmp(*it, src, member_names[idx]);
        
        return object_rec(offset-current_offset, pointer_type, tmp);
      }
      
      assert(new_offset<=offset);
      current_offset=new_offset;
      assert(current_offset<=offset);
      idx++;
    }
  }
  
  return src;
}

/*******************************************************************\

Function: pointer_logict::pointer_logict

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

pointer_logict::pointer_logict()
{
  obj_num_offset = 0;

  type2tc type(new pointer_type2t(type2tc(new empty_type2t())));
  symbol2tc sym(type, "NULL");

  // add NULL
  null_object = add_object(sym);

  // add INVALID
  symbol2tc invalid(type, "INVALID");
  invalid_object = add_object(invalid);

}

/*******************************************************************\

Function: pointer_logict::~pointer_logict

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

pointer_logict::~pointer_logict()
{
}
