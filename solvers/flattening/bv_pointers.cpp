/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <i2string.h>
#include <config.h>
#include <c_misc.h>
#include <arith_tools.h>
#include <pointer_offset_size.h>
#include <prefix.h>
#include <std_expr.h>

#include "bv_pointers.h"

/*******************************************************************\

Function: bv_pointerst::convert_rest

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt bv_pointerst::convert_rest(const exprt &expr)
{
  if(expr.type().id()!="bool")
    throw "bv_pointerst::convert_rest got non-boolean operand";

  const exprt::operandst &operands=expr.operands();

  if(expr.id()=="invalid-pointer")
  {
    if(operands.size()==1 &&
       is_ptr(operands[0].type()))
    {
      bvt bv;
      convert_bv(operands[0], bv);
      
      if(bv.size()!=0)
      {
        bvt invalid_bv, null_bv;
        encode(pointer_logic.get_invalid_object(), invalid_bv);
        encode(pointer_logic.get_null_object(),    null_bv);

        bvt equal_invalid_bv, equal_null_bv;
        equal_invalid_bv.resize(addr_bits);
        equal_null_bv.resize(addr_bits);

        for(unsigned i=0; i<addr_bits; i++)
        {
          equal_invalid_bv[i]=prop.lequal(bv[offset_bits+i],
                                          invalid_bv[offset_bits+i]);
          equal_null_bv[i]   =prop.lequal(bv[offset_bits+i],
                                          null_bv[offset_bits+i]);

        }

        literalt equal_invalid=prop.land(equal_invalid_bv);
        literalt equal_null=prop.land(equal_null_bv);

        return prop.lor(equal_invalid, equal_null);
      }
    }
  }
  else if(expr.id()=="is_dynamic_object")
  {
    if(operands.size()==1 &&
       is_ptr(operands[0].type()))
    {
      bvt bv;
      convert_bv(operands[0], bv);
      
      {
        bv.erase(bv.begin(), bv.begin()+offset_bits);

        // for now, allocate literal and then do later
        is_dynamic_objectt is_dynamic_object;
        is_dynamic_object.bv=bv;
        is_dynamic_object.l=prop.new_variable();
        
        is_dynamic_object_list.push_back(is_dynamic_object);
        return is_dynamic_object.l;
      }
    }
  }
  else if(expr.id()=="same-object")
  {
    if(operands.size()==2 &&
       is_ptr(operands[0].type()) &&
       is_ptr(operands[1].type()))
    {
      bvt bv0, bv1;
      
      convert_bv(operands[0], bv0);
      convert_bv(operands[1], bv1);

      {
        bvt equal_bv;
        equal_bv.resize(addr_bits);

        for(unsigned i=0; i<addr_bits; i++)
          equal_bv[i]=prop.lequal(bv0[offset_bits+i],
                                  bv1[offset_bits+i]);

        return prop.land(equal_bv);
      }
    }
  }
  else if(expr.id()=="<" || expr.id()=="<=" ||
          expr.id()==">" || expr.id()==">=")
  {
    if(operands.size()==2 &&
       is_ptr(operands[0].type()) &&
       is_ptr(operands[1].type()))
    {
      bvt bv0, bv1;

      convert_bv(operands[0], bv0);
      convert_bv(operands[1], bv1);

      {
        const std::string &rel=expr.id_string();

        if(rel=="<=" || rel=="<")
          return bv_utils.lt_or_le(rel=="<=", bv0, bv1, bv_utilst::UNSIGNED);

        if(rel==">=" || rel==">")
          return bv_utils.lt_or_le(rel==">=", bv1, bv0, bv_utilst::UNSIGNED);
                                              // swapped
      }
    }
  }

  return SUB::convert_rest(expr);
}

/*******************************************************************\

Function: bv_pointerst::bv_pointerst

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bv_pointerst::bv_pointerst(propt &_prop):boolbvt(_prop)
{
  addr_bits=BV_ADDR_BITS;
  offset_bits=config.ansi_c.pointer_width;
  bits=addr_bits+offset_bits;
}

/*******************************************************************\

Function: bv_pointerst::convert_address_of_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bv_pointerst::convert_address_of_rec(
  const exprt &expr,
  bvt &bv)
{
  if(expr.is_symbol())
  {
    add_addr(expr, bv);
    return;
  }
  else if(expr.id()=="NULL-object")
  {
    encode(pointer_logic.get_null_object(), bv);
    return;
  }
  else if(expr.is_index())
  {
    if(expr.operands().size()!=2)
      throw "index takes two operands";

    const exprt &array=expr.op0();
    const exprt &index=expr.op1();
    
    // recursive call

    if(array.type().is_pointer())
      convert_pointer_type(array, bv);
    else
      convert_address_of_rec(array, bv);

    // get size
    mp_integer size;
    
    if(array.type().id()=="string-constant")
      size=1;
    else
      size=pointer_offset_size(array.type().subtype());
    
    if(size==0) return conversion_failed(expr, bv);
    
    return increase_offset(bv, size, index);
  }
  else if(expr.is_member())
  {
    if(expr.operands().size()!=1)
      throw "member takes one operand";

    const exprt &struct_op=expr.op0();

    // recursive call
    convert_address_of_rec(struct_op, bv);

    const irept::subt &components=
      struct_op.type().components().get_sub();
    
    const irep_idt &component_name=expr.component_name();
    
    bool found=false;
    
    mp_integer offset=1; // for the struct itself

    forall_irep(it, components)
    {
      if(component_name==it->name()) { found=true; break; }
      const typet &subtype=it->type();
      mp_integer sub_size=pointer_offset_size(subtype);
      if(sub_size==0)
        return conversion_failed(expr, bv);
      offset+=sub_size;
    }

    if(!found)
      return conversion_failed(expr, bv);

    // add offset
    increase_offset(bv, offset);
    
    return;
  }
  else if(expr.is_constant() ||
          expr.id()=="string-constant" ||
          expr.id()=="zero_string")
  { // constant
    add_addr(expr, bv);
    return;
  }

  return conversion_failed(expr, bv);
}

/*******************************************************************\

Function: bv_pointerst::convert_pointer_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bv_pointerst::convert_pointer_type(const exprt &expr, bvt &bv)
{
  if(!is_ptr(expr.type()))
    throw "convert_pointer_type got non-pointer type";

  bv.resize(bits);

  if(expr.is_symbol())
  {
    const irep_idt &identifier=expr.identifier();
    const typet &type=expr.type();

    for(unsigned i=0; i<bits; i++)
      bv[i]=map.get_literal(identifier, i, type);

    return;
  }
  else if(expr.id()=="nondet_symbol")
  {
    for(unsigned i=0; i<bits; i++)
      bv[i]=prop.new_variable();

    return;
  }
  else if(expr.is_typecast())
  {
    if(expr.operands().size()!=1)
      throw "typecast takes one operand";

    const exprt &op=expr.op0();

    if(op.type().is_pointer())
      return convert_pointer_type(op, bv);
    else
    {
      // produce an invalid pointer
      encode(pointer_logic.get_invalid_object(), bv);

      // set offset to nondet
      for(unsigned i=0; i<offset_bits; i++)
        bv[i]=prop.new_variable();

      return;
    }
  }
  else if(expr.is_if())
  {
    return SUB::convert_if(expr, bv);
  }
  else if(expr.is_index())
  {
    return SUB::convert_index(to_index_expr(expr), bv);
  }
  else if(expr.is_member())
  {
    return SUB::convert_member(expr, bv);
  }
  else if(expr.is_address_of() ||
          expr.id()=="implicit_address_of" ||
          expr.id()=="reference_to")
  {
    if(expr.operands().size()!=1)
      throw expr.id_string()+" takes one operand";
      
    return convert_address_of_rec(expr.op0(), bv);
  }
  else if(expr.is_constant())
  {
    if(expr.value().as_string()!="NULL")
      throw "found non-NULL pointer constant";

    encode(pointer_logic.get_null_object(), bv);

    return;
  }
  else if(expr.id()=="+")
  {
    if(expr.operands().size()==0)
      throw "operator "+expr.id_string()+" takes at least one operand";

    bool found=false;
    mp_integer size;

    forall_operands(it, expr)
    {
      if(it->type().is_pointer())
      {
        if(found)
          throw "found two pointers in sum";

        found=true;
        convert_bv(*it, bv);
          
        size=pointer_offset_size(it->type().subtype());
      }
    }

    if(!found)
      throw "found no pointer in pointer type sum";

    bvt sum;
    sum.resize(offset_bits);

    for(unsigned i=0; i<offset_bits; i++)
      sum[i]=const_literal(false);

    forall_operands(it, expr)
    {
      bvt op;

      if(it->type().is_pointer()) continue;

      if(!it->type().is_unsignedbv() &&
         !it->type().is_signedbv())
        return conversion_failed(expr, bv);

      bv_utilst::representationt rep=
        it->type().is_signedbv()?bv_utilst::SIGNED:
                                    bv_utilst::UNSIGNED;

      convert_bv(*it, op);

      if(op.size()>offset_bits || op.size()==0)
        throw "unexpected pointer arithmetic operand width";

      op=bv_utils.extension(op, offset_bits, rep);

      sum=bv_utils.add(sum, op);
    }

    increase_offset(bv, size, sum);

    return;
  }
  else if(expr.id()=="-")
  {
    if(expr.operands().size()!=2)
      throw "operator "+expr.id_string()+" takes two operands";

    if(!expr.op0().type().is_pointer())
      throw "found no pointer in pointer type in difference";

    bvt bv0, bv1;

    convert_bv(expr.op0(), bv0);
    convert_bv(expr.op1(), bv1);

    bvt sum;
    sum.resize(offset_bits);

    for(unsigned i=0; i<offset_bits; i++)
      sum[i]=bv0[i];

    if(!expr.op1().type().is_unsignedbv() &&
       !expr.op1().type().is_signedbv())
      return conversion_failed(expr, bv);

    if(bv1.size()>offset_bits || bv1.size()==0)
      throw "unexpected pointer arithmetic operand width";

    bv_utilst::representationt rep=
      expr.op1().type().is_signedbv()?bv_utilst::SIGNED:
                                         bv_utilst::UNSIGNED;
 
    bv1=bv_utils.extension(bv1, offset_bits, rep);

    sum=bv_utils.sub(sum, bv1); // subtract

    for(unsigned i=0; i<bv.size(); i++)
      bv[i]=i<offset_bits?sum[i]:bv0[i];

    return;
  }

  return conversion_failed(expr, bv);
}

/*******************************************************************\

Function: bv_pointerst::convert_bitvector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bv_pointerst::convert_bitvector(const exprt &expr, bvt &bv)
{
  if(is_ptr(expr.type()))
    return convert_pointer_type(expr, bv);

  if(expr.id()=="-" && expr.operands().size()==2 &&
     expr.op0().type().is_pointer() &&
     expr.op1().type().is_pointer())
  {
    bvt op0, op1;

    convert_bv(expr.op0(), op0);
    convert_bv(expr.op1(), op1);

    unsigned width;
    if(boolbv_get_width(expr.type(), width))
      return conversion_failed(expr, bv);

    if(width>offset_bits)
      throw "no sign extension on pointer differences";

    op0.resize(width);
    op1.resize(width);

    bv=bv_utils.sub(op0, op1);

    return;
  }
  else if(expr.id()=="pointer_offset" &&
          expr.operands().size()==1 &&
          is_ptr(expr.op0().type()))
  {
    bvt op0;

    convert_bv(expr.op0(), op0);

    unsigned width;
    if(boolbv_get_width(expr.type(), width))
      return conversion_failed(expr, bv);

    if(width>offset_bits)
      throw "no sign extension on pointer differences";

    op0.resize(width);

    bv=op0;

    return;
  }
  else if(expr.id()=="pointer_object" &&
          expr.operands().size()==1 &&
          is_ptr(expr.op0().type()))
  {
    bvt op0;

    convert_bv(expr.op0(), op0);

    unsigned width;
    if(boolbv_get_width(expr.type(), width))
      return conversion_failed(expr, bv);
      
    // erase offset bits
    
    op0.erase(op0.begin()+0, op0.begin()+offset_bits);

    bv=bv_utils.zero_extension(op0, width);

    return;
  }
  else if(expr.is_typecast() &&
          expr.operands().size()==1 &&
          expr.op0().type().is_pointer())
  {
    bvt op0;
    convert_pointer_type(expr.op0(), op0);
  
    // squeeze it in!

    unsigned width;
    if(boolbv_get_width(expr.type(), width))
      return conversion_failed(expr, bv);

    bv.resize(width);
    
    // too small?
    if(width<addr_bits)
      return conversion_failed(expr, bv);

    for(unsigned i=0; i<width; i++)
    {
      if(i>=width-addr_bits)
        bv[i]=op0[i-(width-addr_bits)+offset_bits];
      else if(i>=offset_bits)
        bv[i]=const_literal(false);
      else
        bv[i]=op0[i];
    }

    return;
  }

  return SUB::convert_bitvector(expr, bv);
}

/*******************************************************************\

Function: bv_pointerst::bv_get_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt bv_pointerst::bv_get_rec(
  const bvt &bv,
  const std::vector<bool> &unknown,
  unsigned offset,
  const typet &type) const
{
  if(!is_ptr(type))
    return SUB::bv_get_rec(bv, unknown, offset, type);

  std::string value_addr, value_offset;

  for(unsigned i=0; i<bits; i++)
  {
    char ch;
    unsigned bit_nr=i+offset;

    if(unknown[bit_nr])
      ch='0';
    else
      switch(prop.l_get(bv[bit_nr]).get_value())
      {
       case tvt::TV_FALSE: ch='0'; break;
       case tvt::TV_TRUE:  ch='1'; break;
       case tvt::TV_UNKNOWN: ch='0'; break;
       default: assert(false);
      }

    if(i<offset_bits)
      value_offset=ch+value_offset;
    else
      value_addr=ch+value_addr;
  }

  pointer_logict::pointert pointer;
  pointer.object=integer2long(binary2integer(value_addr, false));
  pointer.offset=binary2integer(value_offset, true);

  return pointer_logic.pointer_expr(pointer, type);
}

/*******************************************************************\

Function: bv_pointerst::encode

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bv_pointerst::encode(unsigned addr, bvt &bv)
{
  bv.resize(bits);

  // set offset to zero
  for(unsigned i=0; i<offset_bits; i++)
    bv[i]=const_literal(false);

  // set variable part
  for(unsigned i=0; i<addr_bits; i++)
    bv[offset_bits+i]=const_literal(addr&(1<<i));
}

/*******************************************************************\

Function: bv_pointerst::increase_offset

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bv_pointerst::increase_offset(bvt &bv, const mp_integer &x)
{
  bvt bv1=bv;
  bv1.resize(offset_bits); // strip down

  bvt bv2=bv_utils.build_constant(x, offset_bits);

  bvt tmp=bv_utils.add(bv1, bv2);

  // copy offset bits
  for(unsigned i=0; i<offset_bits; i++)
    bv[i]=tmp[i];
}

/*******************************************************************\

Function: bv_pointerst::increase_offset

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bv_pointerst::increase_offset(
  bvt &bv,
  const mp_integer &factor,
  const exprt &index)
{
  bvt bv_index;
  convert_bv(index, bv_index);

  bv_utilst::representationt rep=
    index.type().is_signedbv()?bv_utilst::SIGNED:
                                  bv_utilst::UNSIGNED;

  bv_index=bv_utils.extension(bv_index, offset_bits, rep);

  increase_offset(bv, factor, bv_index);
}

/*******************************************************************\

Function: bv_pointerst::increase_offset

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bv_pointerst::increase_offset(
  bvt &bv,
  const mp_integer &factor,
  const bvt &index)
{
  bvt bv_index=index;

  assert(bv_index.size()==offset_bits);
  
  if(factor!=1)
  {
    bvt bv_factor=bv_utils.build_constant(factor, offset_bits);
    bv_index=bv_utils.unsigned_multiplier(bv_index, bv_factor);
  }

  bv_index=bv_utils.zero_extension(bv_index, bv.size());

  bvt bv_tmp=bv_utils.add(bv, bv_index);
  
  // copy result
  for(unsigned i=0; i<offset_bits; i++)
    bv[i]=bv_tmp[i];
}

/*******************************************************************\

Function: bv_pointerst::add_addr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bv_pointerst::add_addr(const exprt &expr, bvt &bv)
{
  unsigned a=pointer_logic.add_object(expr);

  if(a==(unsigned(1)>>addr_bits))
    throw "too many variables";

  encode(a, bv);
}

/*******************************************************************\

Function: bv_pointerst::do_is_dynamic_object

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bv_pointerst::do_is_dynamic_object(
  const is_dynamic_objectt &is_dynamic_object)
{
  const pointer_logict::objectst &objects=
    pointer_logic.objects;
    
  unsigned number=0;
    
  for(pointer_logict::objectst::const_iterator
      it=objects.begin();
      it!=objects.end();
      it++, number++)
  {
    const exprt &expr=*it;
    
    bool is_dynamic=
      expr.type().dynamic() ||
      (expr.is_symbol() &&
       has_prefix(expr.identifier().as_string(), "symex_dynamic::"));
    
    // only compare object part
    bvt bv;
    encode(number, bv);
    
    bv.erase(bv.begin(), bv.begin()+offset_bits);
    
    literalt l1=bv_utils.equal(bv, is_dynamic_object.bv);
    literalt l2=is_dynamic_object.l;
    
    if(!is_dynamic) l2=prop.lnot(l2);
    
    prop.l_set_to(prop.limplies(l1, l2), true);
  }
}

/*******************************************************************\

Function: bv_pointerst::post_process

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bv_pointerst::post_process()
{
  for(is_dynamic_object_listt::const_iterator
      it=is_dynamic_object_list.begin();
      it!=is_dynamic_object_list.end();
      it++)
    do_is_dynamic_object(*it);
  
  is_dynamic_object_list.clear();
  
  SUB::post_process();
}
