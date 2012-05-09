/*******************************************************************\

Module: Value Set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

// XXXjmorse - consider whether or not many more of the type assertions below
// need to take consideration of the data they're working on being an dynamic
// object type, which should match all types.

#include <irep2.h>
#include <migrate.h>

#include <assert.h>

#include <config.h>
#include <context.h>
#include <simplify_expr.h>
#include <expr_util.h>
#include <base_type.h>
#include <std_expr.h>
#include <i2string.h>
#include <prefix.h>
#include <std_code.h>
#include <arith_tools.h>

#include <langapi/language_util.h>
#include <ansi-c/c_types.h>

#include "value_set.h"

const value_sett::object_map_dt value_sett::object_map_dt::empty;
object_numberingt value_sett::object_numbering;

void value_sett::output(
  const namespacet &ns,
  std::ostream &out) const
{
  for(valuest::const_iterator
      v_it=values.begin();
      v_it!=values.end();
      v_it++)
  {
    irep_idt identifier, display_name;
    
    const entryt &e=v_it->second;
  
    if(has_prefix(id2string(e.identifier), "value_set::dynamic_object"))
    {
      display_name=id2string(e.identifier)+e.suffix;
      identifier="";
    }
    else if(e.identifier=="value_set::return_value")
    {
      display_name="RETURN_VALUE"+e.suffix;
      identifier="";
    }
    else
    {
      #if 0
      const symbolt &symbol=ns.lookup(e.identifier);
      display_name=symbol.display_name()+e.suffix;
      identifier=symbol.name;
      #else
      identifier=id2string(e.identifier);
      display_name=id2string(identifier)+e.suffix;
      #endif
    }
    
    out << display_name;

    out << " = { ";

    const object_map_dt &object_map=e.object_map.read();
    
    unsigned width=0;
    
    for(object_map_dt::const_iterator
        o_it=object_map.begin();
        o_it!=object_map.end();
        o_it++)
    {
      const expr2tc &o = object_numbering[o_it->first];
    
      std::string result;

      if (is_invalid2t(o) || is_unknown2t(o))
        result=from_expr(ns, identifier, o);
      else
      {
        result="<"+from_expr(ns, identifier, o)+", ";
      
        if(o_it->second.offset_is_set)
          result+=integer2string(o_it->second.offset)+"";
        else
          result+="*";
        
        result += ", "+from_type(ns, identifier, o->type);
      
        result+=">";
      }

      out << result;

      width+=result.size();
    
      object_map_dt::const_iterator next(o_it);
      next++;

      if(next!=object_map.end())
      {
        out << ", ";
        if(width>=40) out << "\n      ";
      }
    }

    out << " } " << std::endl;
  }
}

exprt value_sett::to_expr(object_map_dt::const_iterator it) const
{
  const expr2tc &object2=object_numbering[it->first];
  const exprt object = migrate_expr_back(object2);
  
  if(object.id()=="invalid" ||
     object.id()=="unknown")
    return object;

  object_descriptor_exprt od;

  od.object()=object;
  
  if(it->second.offset_is_set)
    od.offset()=from_integer(it->second.offset, index_type());

  od.type()=od.object().type();

  return od;
}

bool value_sett::make_union(const value_sett::valuest &new_values)
{
  bool result=false;
  
  for(valuest::const_iterator
      it=new_values.begin();
      it!=new_values.end();
      it++)
  {
    valuest::iterator it2=values.find(it->first);

    if(it2==values.end())
    {
      // we always track these
      if(has_prefix(id2string(it->second.identifier),
           "value_set::dynamic_object") ||
         it->second.identifier=="value_set::return_value")
      {
        values.insert(*it);
        result=true;
      }

      continue;
    }
      
    entryt &e=it2->second;
    const entryt &new_e=it->second;
    
    if(make_union(e.object_map, new_e.object_map))
      result=true;
  }
  
  return result;
}

bool value_sett::make_union(object_mapt &dest, const object_mapt &src) const
{
  bool result=false;
  
  for(object_map_dt::const_iterator it=src.read().begin();
      it!=src.read().end();
      it++)
  {
    if(insert(dest, it))
      result=true;
  }
  
  return result;
}

void value_sett::get_value_set(
  const exprt &expr,
  value_setst::valuest &dest,
  const namespacet &ns) const
{
  object_mapt object_map;
  get_value_set(expr, object_map, ns);
  
  for(object_map_dt::const_iterator
      it=object_map.read().begin();
      it!=object_map.read().end();
      it++)
    dest.push_back(to_expr(it));
}

void value_sett::get_value_set(
  const exprt &expr,
  object_mapt &dest,
  const namespacet &ns) const
{
  exprt tmp(expr);
  simplify(tmp);

  if (tmp.is_nil()) {
    // Hacks, for the --function option, which spews "nil" exprs here, but with
    // a type configured. Create an unknown obj record with that type.
    type2tc cvt_type;
    migrate_type(expr.type(), cvt_type);
    expr2tc tmp = expr2tc(new unknown2t(cvt_type));
    insert(dest, tmp);
    return;
  }

  // Otherwise, continue on as normal.
  expr2tc tmp_expr;
  migrate_expr(tmp, tmp_expr);
  type2tc tmp_type;
  migrate_type(tmp.type(), tmp_type);
  get_value_set_rec(tmp_expr, dest, "", tmp_type, ns);
}

void value_sett::get_value_set_rec(
  const expr2tc &expr,
  object_mapt &dest,
  const std::string &suffix,
  const type2tc &original_type,
  const namespacet &ns) const
{
  const type2tc &expr_type = expr->type;

  if (is_unknown2t(expr) || is_invalid2t(expr))
  {
    insert(dest, expr2tc(new unknown2t(original_type)));
    return;
  }
  else if (is_index2t(expr))
  {
    const index2t &idx = to_index2t(expr);
    const type2tc &source_type = idx.source_value->type;

    assert(is_array_type(source_type) || is_string_type(source_type));
           
    get_value_set_rec(idx.source_value, dest, "[]"+suffix, original_type, ns);
    
    return;
  }
  else if (is_member2t(expr))
  {
    const member2t &memb = to_member2t(expr);
    const type2tc &source_type = memb.source_value->type;

    assert(is_struct_type(source_type) || is_union_type(source_type));
           
    get_value_set_rec(memb.source_value, dest,
                      "." + memb.member.as_string() + suffix,
                      original_type, ns);
    return;
  }
  else if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);

    if (sym.name.as_string() == "NULL" && is_pointer_type(expr->type))
    {
      // XXXjmorse - looks like there's no easy way to avoid this ns.follow
      // for getting the null objects type, without an internal pointer
      // repr reshuffle.
      const pointer_type2t &ptr_ref = to_pointer_type(expr->type);
      typet subtype = migrate_type_back(ptr_ref.subtype);
      if (subtype.id() == "symbol")
        subtype = ns.follow(subtype);

      exprt tmp("NULL-object", subtype);
      expr2tc tmp2;
      migrate_expr(tmp, tmp2);
      insert(dest, tmp2, 0);
      return;
    }

    // look it up
    valuest::const_iterator v_it = values.find(sym.name.as_string() + suffix);
      
    if(v_it!=values.end())
    {
      make_union(dest, v_it->second.object_map);
      return;
    }
  }
  else if (is_if2t(expr))
  {
    const if2t &ifval = to_if2t(expr);

    get_value_set_rec(ifval.true_value, dest, suffix, original_type, ns);
    get_value_set_rec(ifval.false_value, dest, suffix, original_type, ns);

    return;
  }
  else if (is_address_of2t(expr))
  {
    const address_of2t &addrof = to_address_of2t(expr);
    get_reference_set(addrof.ptr_obj, dest, ns);
    return;
  }
  else if (is_dereference2t(expr))
  {
    object_mapt reference_set;
    get_reference_set(expr, reference_set, ns);
    const object_map_dt &object_map=reference_set.read();
    
    if(object_map.begin()!=object_map.end())
    {
      for(object_map_dt::const_iterator
          it1=object_map.begin();
          it1!=object_map.end();
          it1++)
      {
        const expr2tc &object = object_numbering[it1->first];
        get_value_set_rec(object, dest, suffix, original_type, ns);
      }

      return;
    }
  }
  else if(is_constant_expr(expr))
  {
    // Check for null moved to symbol2t.
    return;
  }
  else if (is_typecast2t(expr))
  {
    const typecast2t &cast = to_typecast2t(expr);
    get_value_set_rec(cast.from, dest, suffix, original_type, ns);
    return;
  }
  else if (is_add2t(expr) || is_sub2t(expr))
  {
    if (is_pointer_type(expr->type))
    {
      // find the pointer operand
      // XXXjmorse - polymorphism.
      const expr2tc &op0 = (is_add2t(expr))
                           ? to_add2t(expr).side_1
                           : to_sub2t(expr).side_1;
      const expr2tc &op1 = (is_add2t(expr))
                           ? to_add2t(expr).side_2
                           : to_sub2t(expr).side_2;

      assert(!(is_pointer_type(op0->type) && is_pointer_type(op1->type)) &&
              "Cannot have pointer arithmatic with two pointers as operands");

      const expr2tc &ptr_op= (is_pointer_type(op0->type)) ? op0 : op1;
      const expr2tc &non_ptr_op= (is_pointer_type(op0->type)) ? op1 : op0;

      object_mapt pointer_expr_set;
      get_value_set_rec(ptr_op, pointer_expr_set, "", ptr_op->type, ns);

      for(object_map_dt::const_iterator
          it=pointer_expr_set.read().begin();
          it!=pointer_expr_set.read().end();
          it++)
      {
        objectt object=it->second;

        if (object.offset_is_zero()) {
          if (is_constant_int2t(non_ptr_op)) {
            object.offset = to_constant_int2t(non_ptr_op).constant_value;
          } else {
            object.offset_is_set = false;
          }
        } else {
          object.offset_is_set=false;
        }
          
        insert(dest, it->first, object);
      }

      return;
    }
  }
  else if (is_sideeffect2t(expr))
  {
    const sideeffect2t &side = to_sideeffect2t(expr);
    switch (side.kind) {
    case sideeffect2t::malloc:
      {
      assert(suffix=="");
      const type2tc &dynamic_type = side.alloctype;


      expr2tc locnum = expr2tc(new constant_int2t(type_pool.get_uint(config.ansi_c.int_width), BigInt(location_number)));

      expr2tc dynobj = expr2tc(new dynamic_object2t(dynamic_type,
                                                    locnum, false, false));

      insert(dest, dynobj, 0);
      }
      return;          
 
    case sideeffect2t::cpp_new:
    case sideeffect2t::cpp_new_arr:
      {
      assert(suffix=="");
      assert(is_pointer_type(side.type));

      expr2tc locnum = expr2tc(new constant_int2t(type_pool.get_uint(config.ansi_c.int_width), BigInt(location_number)));

      const pointer_type2t &ptr = to_pointer_type(side.type);

      expr2tc dynobj = expr2tc(new dynamic_object2t(ptr.subtype, locnum,
                                                  false, false));

      insert(dest, dynobj, 0);
      }
      return;
    case sideeffect2t::nondet:
      // XXXjmorse - don't know what to do here, previously wasn't handled,
      // so I won't try to handle it now.
      return;
    default:
      std::cerr << "Unexpected side-effect: " << expr->pretty(0) << std::endl;
      abort();
    }
  }
  else if (is_constant_struct2t(expr))
  {
    // this is like a static struct object
    expr2tc tmp = expr2tc(new address_of2t(expr->type, expr));
    insert(dest, tmp, 0);
    return;
  }
  else if (is_with2t(expr))
  {
    const with2t &with = to_with2t(expr);

    // this is the array/struct
    object_mapt tmp_map0;
    get_value_set_rec(with.source_value, tmp_map0, suffix, original_type, ns);

    // this is the update value -- note NO SUFFIX
    object_mapt tmp_map2;
    get_value_set_rec(with.update_value, tmp_map2, "", original_type, ns);

    make_union(dest, tmp_map0);
    make_union(dest, tmp_map2);
  }
  else if (is_constant_array_of2t(expr) || is_constant_array2t(expr))
  {
    // these are supposed to be done by assign()
    assert(0 && "Encountered array irep in get_value_set_rec");
  }
  else if (is_dynamic_object2t(expr))
  {
    const dynamic_object2t &dyn = to_dynamic_object2t(expr);
  
    // XXXjmorse, could become a uint.
    assert(is_constant_int2t(dyn.instance));
    const constant_int2t &intref = to_constant_int2t(dyn.instance);
    std::string idnum = integer2string(intref.constant_value);
    const std::string name = "value_set::dynamic_object" + idnum + suffix;
  
    // look it up
    valuest::const_iterator v_it=values.find(name);

    if(v_it!=values.end())
    {
      make_union(dest, v_it->second.object_map);
      return;
    }
  }

  expr2tc tmp = expr2tc(new unknown2t(original_type));
  insert(dest, tmp);
}

void value_sett::dereference_rec(
  const exprt &src,
  exprt &dest) const
{
  // remove pointer typecasts
  if(src.id()=="typecast")
  {
    assert(src.type().id()=="pointer");

    if(src.operands().size()!=1)
      throw "typecast expects one operand";
    
    dereference_rec(src.op0(), dest);
  }
  else
    dest=src;
}

void value_sett::get_reference_set(
  const expr2tc &expr,
  value_setst::valuest &dest,
  const namespacet &ns) const
{
  object_mapt object_map;
  get_reference_set(expr, object_map, ns);
  
  for(object_map_dt::const_iterator
      it=object_map.read().begin();
      it!=object_map.read().end();
      it++)
    dest.push_back(to_expr(it));
}

void value_sett::get_reference_set_rec(
  const expr2tc &expr,
  object_mapt &dest,
  const namespacet &ns) const
{

  if (is_symbol2t(expr) || is_dynamic_object2t(expr) ||
      is_constant_string2t(expr))
  {
    if (is_array_type(expr->type) &&
        is_array_type(to_array_type(expr->type).subtype))
      insert(dest, expr);
    else    
      insert(dest, expr, 0);

    return;
  }
  else if (is_dereference2t(expr))
  {
    const dereference2t &deref = to_dereference2t(expr);
    get_value_set_rec(deref.value, dest, "", deref.type, ns);
    return;
  }
  else if (is_index2t(expr))
  {
    const index2t &index = to_index2t(expr);
    
    assert(is_array_type(index.source_value->type) ||
           is_string_type(index.source_value->type));
    
    // fix up array type during migration; it can have location fields and
    // other useless gubbins in it, which get compared against a back-migrated
    // type that drops such information, thus always failing. Work around this
    // by migrating forwards and backwards the array type, thus stripping out
    // any additional information we don't want in a comparison.

    object_mapt array_references;
    get_reference_set(index.source_value, array_references, ns);
        
    const object_map_dt &object_map=array_references.read();
    
    for(object_map_dt::const_iterator
        a_it=object_map.begin();
        a_it!=object_map.end();
        a_it++)
    {
      const expr2tc &object = object_numbering[a_it->first];

      if (is_unknown2t(object)) {
        expr2tc unknown = expr2tc(new unknown2t(expr->type));
        insert(dest, unknown);
      } else {
        type2tc zero_type = type_pool.get_uint(config.ansi_c.int_width);
        expr2tc const_zero = expr2tc(new constant_int2t(zero_type, BigInt(0)));
        expr2tc new_index = expr2tc(new index2t(index.type, object,const_zero));
        
        // adjust type?
        if (object->type != index.source_value->type)
          new_index = expr2tc(new typecast2t(index.source_value->type,
                                             new_index));
        objectt o = a_it->second;

        if (is_constant_int2t(index.index) &&
            to_constant_int2t(index.index).constant_value.is_zero())
          ;
        else if (is_constant_int2t(index.index) && o.offset_is_zero())
          o.offset = to_constant_int2t(index.index).constant_value;
        else
          o.offset_is_set = false;
          
        insert(dest, new_index, o);
      }
    }
    
    return;
  }
  else if (is_member2t(expr))
  {
    const member2t &memb = to_member2t(expr);

    object_mapt struct_references;
    get_reference_set(memb.source_value, struct_references, ns);
    
    const object_map_dt &object_map=struct_references.read();

    for(object_map_dt::const_iterator
        it=object_map.begin();
        it!=object_map.end();
        it++)
    {
      const expr2tc &object = object_numbering[it->first];
      
      if (is_unknown2t(object)) {
        expr2tc unknown = expr2tc(new unknown2t(memb.type));
        insert(dest, unknown);
      } else {
        objectt o=it->second;

        expr2tc new_memb = expr2tc(new member2t(memb.type, object,memb.member));
        
        // adjust type?
        if (memb.source_value->type != object->type)
          new_memb = expr2tc(new typecast2t(memb.source_value->type, new_memb));
        
        insert(dest, new_memb, o);
      }
    }

    return;
  }
  else if (is_if2t(expr))
  {
    const if2t &anif = to_if2t(expr);
    get_reference_set_rec(anif.true_value, dest, ns);
    get_reference_set_rec(anif.false_value, dest, ns);
    return;
  }

  expr2tc unknown = expr2tc(new unknown2t(expr->type));
  insert(dest, unknown);
}

void value_sett::assign(
  const expr2tc &lhs,
  const expr2tc &rhs,
  const namespacet &ns,
  bool add_to_sets)
{

  if (is_if2t(rhs))
  {
    const if2t &ifref = to_if2t(rhs);
    assign(lhs, ifref.true_value, ns, add_to_sets);
    assign(lhs, ifref.false_value, ns, true);
    return;
  }

  assert(!is_symbol_type(lhs->type));
  const type2tc &lhs_type = lhs->type;
  
  if (is_struct_type(lhs_type) || is_union_type(lhs_type))
  {
    const struct_type2t &struct_type = to_struct_type(lhs_type);
    
    unsigned int i = 0;
    for(std::vector<type2tc>::const_iterator c_it = struct_type.members.begin();
        c_it != struct_type.members.end(); c_it++, i++)
    {
      const type2tc &subtype = *c_it;
      const irep_idt &name = struct_type.member_names[i];

      // ignore methods
      if (is_code_type(subtype))
        continue;
    
      expr2tc lhs_member = expr2tc(new member2t(subtype, lhs, name));

      expr2tc rhs_member;
      if (is_unknown2t(rhs))
      {
        rhs_member = expr2tc(new unknown2t(subtype));
      }
      else if (is_invalid2t(rhs))
      {
        rhs_member = expr2tc(new invalid2t(subtype));
      }
      else
      {
        if (is_index2t(rhs)) {
          if (is_symbol2t(lhs)) {
            assign(lhs_member, to_index2t(rhs).source_value, ns, add_to_sets);
            return;
    	  }
    	}

        assert(base_type_eq(rhs->type, lhs_type, ns));

        expr2tc rhs_member;
        exprt rhs_member_tmp = make_member(migrate_expr_back(rhs), name, ns);
        migrate_expr(rhs_member_tmp, rhs_member);
      
        assign(lhs_member, rhs_member, ns, add_to_sets);
      }
    }
  }
  else if (is_array_type(lhs_type))
  {
    const array_type2t &arr_type = to_array_type(lhs_type);
    expr2tc unknown = expr2tc(new unknown2t(index_type2()));
    expr2tc lhs_index = expr2tc(new index2t(arr_type.subtype, lhs, unknown));

    if (is_unknown2t(rhs) || is_invalid2t(rhs))
    {
      // XXXjmorse - was passing rhs as exprt(rhs.id(), type.subtype()),
      // Which discards much data and is probably invalid.
      assign(lhs_index, rhs, ns, add_to_sets);
    }
    else
    {
      assert(base_type_eq(rhs->type, lhs_type, ns));
        
      if (is_constant_array_of2t(rhs))
      {
        assign(lhs_index, to_constant_array_of2t(rhs).initializer,
               ns, add_to_sets);
      }
      else if (is_constant_array2t(rhs) || is_constant_expr(rhs))
      {
        // ...whattt
#if 0
        forall_operands(o_it, rhs)
        {
          assign(lhs_index, *o_it, ns, add_to_sets);
          add_to_sets=true;
        }
#endif
        std::vector<const expr2tc *> operands;
        rhs->list_operands(operands);
        for (std::vector<const expr2tc *>::const_iterator it = operands.begin();
             it != operands.end(); it++) {
          assign(lhs_index, **it, ns, add_to_sets);
          add_to_sets = true;
        }
      }
      else if (is_with2t(rhs))
      {
        const with2t &with = to_with2t(rhs);

        expr2tc unknown = expr2tc(new unknown2t(index_type2()));
        expr2tc idx = expr2tc(new index2t(arr_type.subtype, with.source_value,
                                          unknown));

        assign(lhs_index, idx, ns, add_to_sets);
        assign(lhs_index, with.update_value, ns, true);
      }
      else
      {
        expr2tc unknown = expr2tc(new unknown2t(index_type2()));
        expr2tc rhs_idx = expr2tc(new index2t(arr_type.subtype, rhs, unknown));
        assign(lhs_index, rhs_idx, ns, true);
      }
    }
  }
  else
  {
    // basic type
    object_mapt values_rhs;
    
    get_value_set(migrate_expr_back(rhs), values_rhs, ns);
    
    assign_rec(lhs, values_rhs, "", ns, add_to_sets);
  }
}

void value_sett::do_free(
  const exprt &op,
  const namespacet &ns)
{
  // op must be a pointer
  if(op.type().id()!="pointer")
    throw "free expected to have pointer-type operand";

  // find out what it points to    
  object_mapt value_set;
  get_value_set(op, value_set, ns);
  
  const object_map_dt &object_map=value_set.read();
  
  // find out which *instances* interest us
  expr_sett to_mark;
  
  for(object_map_dt::const_iterator
      it=object_map.begin();
      it!=object_map.end();
      it++)
  {
    const expr2tc &object2=object_numbering[it->first];
    const exprt object = migrate_expr_back(object2);

    if(object.id()=="dynamic_object")
    {
      const dynamic_object_exprt &dynamic_object=
        to_dynamic_object_expr(object);
      
      if(dynamic_object.valid().is_true()) {
        expr2tc tmp;
        migrate_expr(dynamic_object.instance(), tmp);
        to_mark.insert(tmp);
      }
    }
  }
  
  // mark these as 'may be invalid'
  // this, unfortunately, destroys the sharing
  for(valuest::iterator v_it=values.begin();
      v_it!=values.end();
      v_it++)
  {
    object_mapt new_object_map;

    const object_map_dt &old_object_map=
      v_it->second.object_map.read();
      
    bool changed=false;
    
    for(object_map_dt::const_iterator
        o_it=old_object_map.begin();
        o_it!=old_object_map.end();
        o_it++)
    {
      const expr2tc &object2=object_numbering[o_it->first];
      const exprt object = migrate_expr_back(object2);

      if(object.id()=="dynamic_object")
      {
        const exprt &instance=
          to_dynamic_object_expr(object).instance();

        expr2tc inst;
        migrate_expr(instance, inst);
        if(to_mark.count(inst)==0)
          set(new_object_map, o_it);
        else
        {
          // adjust
          objectt o=o_it->second;
          exprt tmp(object);
          to_dynamic_object_expr(tmp).valid()=exprt("unknown", op.type());
          expr2tc tmp2;
          migrate_expr(tmp, tmp2);
          insert(new_object_map, tmp2, o);
          changed=true;
        }
      }
      else
        set(new_object_map, o_it);
    }
    
    if(changed)
      v_it->second.object_map=new_object_map;
  }
}

void value_sett::assign_rec(
  const expr2tc &lhs,
  const object_mapt &values_rhs,
  const std::string &suffix,
  const namespacet &ns,
  bool add_to_sets)
{

  if (is_symbol2t(lhs))
  {
    const irep_idt &identifier = to_symbol2t(lhs).name;
    
    if(add_to_sets)
      make_union(get_entry(identifier, suffix).object_map, values_rhs);
    else
      get_entry(identifier, suffix).object_map=values_rhs;
  }
  else if (is_dynamic_object2t(lhs))
  {
    const dynamic_object2t &dynamic_object = to_dynamic_object2t(lhs);
  
    assert(is_constant_int2t(dynamic_object.instance));
    unsigned int idnum =
      to_constant_int2t(dynamic_object.instance).constant_value.to_long();
    const std::string name = "value_set::dynamic_object" + i2string(idnum);

    make_union(get_entry(name, suffix).object_map, values_rhs);
  }
  else if (is_dereference2t(lhs))
  {
    object_mapt reference_set;
    get_reference_set(lhs, reference_set, ns);

    if(reference_set.read().size()!=1)
      add_to_sets=true;
      
    for(object_map_dt::const_iterator
        it=reference_set.read().begin();
        it!=reference_set.read().end();
        it++)
    {
      // XXXjmorse - some horrible type safety is about to fail
      const expr2tc obj = object_numbering[it->first];

      if (!is_unknown2t(obj))
        assign_rec(obj, values_rhs, suffix, ns, add_to_sets);
    }
  }
  else if (is_index2t(lhs))
  {
    assert(is_array_type(to_index2t(lhs).source_value->type) ||
           is_string_type(to_index2t(lhs).source_value->type) ||
           is_dynamic_object2t(to_index2t(lhs).source_value->type));

    assign_rec(to_index2t(lhs).source_value, values_rhs, "[]"+suffix, ns, true);
  }
  else if (is_member2t(lhs))
  {
    const member2t &member = to_member2t(lhs);
    const std::string &component_name = member.member.as_string();

    assert(is_struct_type(member.source_value->type) ||
           is_union_type(member.source_value->type) ||
           is_dynamic_object2t(member.source_value));
           
    assign_rec(to_member2t(lhs).source_value, values_rhs,
               "."+component_name+suffix, ns, add_to_sets);
  }
  else if (is_zero_string2t(lhs) || is_zero_length_string2t(lhs) ||
           is_constant_string2t(lhs) || is_null_object2t(lhs) ||
           is_valid_object2t(lhs) || is_deallocated_obj2t(lhs) ||
           is_dynamic_size2t(lhs))
  {
    // Ignored
  }
  else if (is_typecast2t(lhs))
  {
    assign_rec(to_typecast2t(lhs).from, values_rhs, suffix, ns, add_to_sets);
  }
  else if (is_byte_extract2t(lhs))
  {
    assign_rec(to_byte_extract2t(lhs).source_value, values_rhs, suffix,
               ns, true);
  }
  else
    throw "assign NYI: `" + get_expr_id(lhs)+ "'";
}

void value_sett::do_function_call(
  const irep_idt &function,
  const exprt::operandst &arguments,
  const namespacet &ns)
{
  const symbolt &symbol=ns.lookup(function);

  const code_typet &type=to_code_type(symbol.type);
  const code_typet::argumentst &argument_types=type.arguments();

  // these first need to be assigned to dummy, temporary arguments
  // and only thereafter to the actuals, in order
  // to avoid overwriting actuals that are needed for recursive
  // calls

  for(unsigned i=0; i<arguments.size(); i++)
  {
    const std::string identifier="value_set::dummy_arg_"+i2string(i);
    add_var(identifier, "");
    exprt dummy_lhs=symbol_exprt(identifier, arguments[i].type());

    exprt tmp_arg = arguments[i];
    if (tmp_arg.id() == "nil") {
      // As a workaround for the "--function" option, which feeds "nil"
      // arguments in here, take the expected function argument type rather
      // than the type from the argument.
      tmp_arg.type() = argument_types[i].type();
    }

    expr2tc migrate_lhs, tmp_arg_2;
    migrate_expr(dummy_lhs, migrate_lhs);
    migrate_expr(tmp_arg, tmp_arg_2);
    assign(migrate_lhs, tmp_arg_2, ns, true);
  }

  // now assign to 'actual actuals'

  unsigned i=0;

  for(code_typet::argumentst::const_iterator
      it=argument_types.begin();
      it!=argument_types.end();
      it++)
  {
    const irep_idt &identifier=it->get_identifier();
    if(identifier=="") continue;

    add_var(identifier, "");
  
    const exprt v_expr=
      symbol_exprt("value_set::dummy_arg_"+i2string(i), it->type());
    
    exprt actual_lhs=symbol_exprt(identifier, it->type());
    expr2tc migrated_lhs, v_expr_2;
    migrate_expr(actual_lhs, migrated_lhs);
    migrate_expr(v_expr, v_expr_2);
    assign(migrated_lhs, v_expr_2, ns, true);
    i++;
  }
}

void value_sett::do_end_function(
  const exprt &lhs,
  const namespacet &ns)
{
  if(lhs.is_nil()) return;

  symbol_exprt rhs("value_set::return_value", lhs.type());

  expr2tc lhs2, rhs2;
  migrate_expr(lhs, lhs2);
  migrate_expr(rhs, rhs2);
  assign(lhs2, rhs2, ns);
}

void value_sett::apply_code(
  const exprt &code,
  const namespacet &ns)
{
  const irep_idt &statement=code.statement();

  if(statement=="block")
  {
    forall_operands(it, code)
      apply_code(*it, ns);
  }
  else if(statement=="function_call")
  {
    // shouldn't be here
    assert(false);
  }
  else if(statement=="assign" ||
          statement=="init")
  {
    if(code.operands().size()!=2)
      throw "assignment expected to have two operands";

    expr2tc op0, op1;
    migrate_expr(code.op0(), op0);
    migrate_expr(code.op1(), op1);
    assign(op0, op1, ns);
  }
  else if(statement=="decl")
  {
    if(code.operands().size()!=1)
      throw "decl expected to have one operand";

    const exprt &lhs=code.op0();

    if(lhs.id()!="symbol")
      throw "decl expected to have symbol on lhs";

    expr2tc lhs2;
    type2tc thetype;
    migrate_expr(lhs, lhs2);
    migrate_type(lhs.type(), thetype);
    expr2tc invalid = expr2tc(new invalid2t(thetype));
    assign(lhs2, invalid, ns);
  }
  else if(statement=="specc_notify" ||
          statement=="specc_wait")
  {
    // ignore, does not change variables
  }
  else if(statement=="expression")
  {
    // can be ignored, we don't expect sideeffects here
  }
  else if(statement=="cpp_delete" ||
          statement=="cpp_delete[]")
  {
    // does nothing
  }
  else if(statement=="free")
  {
    // this may kill a valid bit

    if(code.operands().size()!=1)
      throw "free expected to have one operand";

    do_free(code.op0(), ns);
  }
  else if(statement=="lock" || statement=="unlock")
  {
    // ignore for now
  }
  else if(statement=="asm")
  {
    // ignore for now, probably not safe
  }
  else if(statement=="nondet")
  {
    // doesn't do anything
  }
  else if(statement=="printf")
  {
    // doesn't do anything
  }
  else if(statement=="return")
  {
    // this is turned into an assignment
    if(code.operands().size()==1)
    {
      symbol_exprt lhs("value_set::return_value", code.op0().type());
      expr2tc lhs2, op0;
      migrate_expr(lhs, lhs2);
      migrate_expr(code.op0(), op0);
      assign(lhs2, op0, ns);
    }
  }
  else if(statement=="cpp-try")
  {
    // doesn't do anything
  }
  else
  {
    std::cerr << code.pretty() << std::endl;
    throw "value_sett: unexpected statement: "+id2string(statement);
  }
}

exprt value_sett::make_member(
  const exprt &src,
  const irep_idt &component_name,
  const namespacet &ns)
{
  const struct_union_typet &struct_type=
    to_struct_type(ns.follow(src.type()));

  if(src.id()=="struct" ||
     src.id()=="constant")
  {
    unsigned no=struct_type.component_number(component_name);
    if (no>=src.operands().size()		//component number does not exist,
    	&& src.operands().size()==1)    //we have only one component
      return src.op0();
    else {
      assert(no<src.operands().size());
      return src.operands()[no];
    }
  }
  else if(src.id()=="with")
  {
    assert(src.operands().size()==3);

    // see if op1 is the member we want
    const exprt &member_operand=src.op1();

    if(component_name==member_operand.component_name())
      // yes! just take op2
      return src.op2();
    else
      // no! do this recursively
      return make_member(src.op0(), component_name, ns);
  }
  else if(src.id()=="typecast")
  {
    // push through typecast
    assert(src.operands().size()==1);
    return make_member(src.op0(), component_name, ns);
  }

  // give up
  typet subtype=struct_type.component_type(component_name);
  member_exprt member_expr(subtype);
  member_expr.op0()=src;
  member_expr.set_component_name(component_name);

  return member_expr;
}
