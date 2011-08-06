/*******************************************************************\

Module: Value Set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

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
   
/*******************************************************************\

Function: value_sett::output

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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
      const exprt &o=object_numbering[o_it->first];
    
      std::string result;

      if(o.id()=="invalid" || o.id()=="unknown")
        result=from_expr(ns, identifier, o);
      else
      {
        result="<"+from_expr(ns, identifier, o)+", ";
      
        if(o_it->second.offset_is_set)
          result+=integer2string(o_it->second.offset)+"";
        else
          result+="*";
        
        result+=", "+from_type(ns, identifier, o.type());
      
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

/*******************************************************************\

Function: value_sett::to_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt value_sett::to_expr(object_map_dt::const_iterator it) const
{
  const exprt &object=object_numbering[it->first];
  
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

/*******************************************************************\

Function: value_sett::make_union

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

/*******************************************************************\

Function: value_sett::make_union

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

/*******************************************************************\

Function: value_sett::get_value_set

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

  #if 0
  for(expr_sett::const_iterator it=value_set.begin(); it!=value_set.end(); it++)
    std::cout << "GET_VALUE_SET: " << from_expr(ns, "", *it) << std::endl;
  #endif
}

/*******************************************************************\

Function: value_sett::get_value_set

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void value_sett::get_value_set(
  const exprt &expr,
  object_mapt &dest,
  const namespacet &ns) const
{
  exprt tmp(expr);
  simplify(tmp);

  get_value_set_rec(tmp, dest, "", tmp.type(), ns);
}

/*******************************************************************\

Function: value_sett::get_value_set_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void value_sett::get_value_set_rec(
  const exprt &expr,
  object_mapt &dest,
  const std::string &suffix,
  const typet &original_type,
  const namespacet &ns) const
{
  #if 0
  std::cout << "GET_VALUE_SET_REC EXPR: " << from_expr(ns, "", expr) << std::endl;
  std::cout << "GET_VALUE_SET_REC SUFFIX: " << suffix << std::endl;
  std::cout << std::endl;
  #endif

  if(expr.id()=="unknown" || expr.id()=="invalid")
  {
    insert(dest, exprt("unknown", original_type));
    return;
  }
  else if(expr.id()=="index")
  {
    assert(expr.operands().size()==2);

    const typet &type=ns.follow(expr.op0().type());

    assert(type.is_array() ||
           type.id()=="incomplete_array");
           
    get_value_set_rec(expr.op0(), dest, "[]"+suffix, original_type, ns);
    
    return;
  }
  else if(expr.id()=="member")
  {
    assert(expr.operands().size()==1);

    const typet &type=ns.follow(expr.op0().type());

    assert(type.id()=="struct" ||
           type.id()=="union" ||
           type.id()=="incomplete_struct" ||
           type.id()=="incomplete_union");
           
    const std::string &component_name=
      expr.component_name().as_string();
    
    get_value_set_rec(expr.op0(), dest,
      "."+component_name+suffix, original_type, ns);
      
    return;
  }
  else if(expr.id()=="symbol")
  {
    // look it up
    valuest::const_iterator v_it=
      values.find(expr.identifier().as_string()+suffix);
      
    if(v_it!=values.end())
    {
      make_union(dest, v_it->second.object_map);
      return;
    }
  }
  else if(expr.id()=="if")
  {
    if(expr.operands().size()!=3)
      throw "if takes three operands";

    get_value_set_rec(expr.op1(), dest, suffix, original_type, ns);
    get_value_set_rec(expr.op2(), dest, suffix, original_type, ns);

    return;
  }
  else if(expr.is_address_of())
  {
    if(expr.operands().size()!=1)
      throw expr.id_string()+" expected to have one operand";
      
    get_reference_set(expr.op0(), dest, ns);
    
    return;
  }
  else if(expr.id()=="dereference" ||
          expr.id()=="implicit_dereference")
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
        const exprt &object=object_numbering[it1->first];
        get_value_set_rec(object, dest, suffix, original_type, ns);
      }

      return;
    }
  }
  else if(expr.id()=="reference_to")
  {
    object_mapt reference_set;
    
    get_reference_set(expr, reference_set, ns);
    
    const object_map_dt &object_map=reference_set.read();
 
    if(object_map.begin()!=object_map.end())
    {
      for(object_map_dt::const_iterator
          it=object_map.begin();
          it!=object_map.end();
          it++)
      {
        const exprt &object=object_numbering[it->first];
        get_value_set_rec(object, dest, suffix, original_type, ns);
      }

      return;
    }
  }
  else if(expr.is_constant())
  {
    // check if NULL
    if(expr.value()=="NULL" && expr.type().id()=="pointer")
    {
      insert(dest, exprt("NULL-object", expr.type().subtype()), 0);
      return;
    }
  }
  else if(expr.id()=="typecast")
  {
    if(expr.operands().size()!=1)
      throw "typecast takes one operand";

    get_value_set_rec(expr.op0(), dest, suffix, original_type, ns);
    
    return;
  }
  else if(expr.id()=="+" || expr.id()=="-")
  {
    if(expr.operands().size()<2)
      throw expr.id_string()+" expected to have at least two operands";

    if(expr.type().id()=="pointer")
    {
      // find the pointer operand
      const exprt *ptr_operand=NULL;

      forall_operands(it, expr)
        if(it->type().id()=="pointer")
        {
          if(ptr_operand==NULL)
            ptr_operand=&(*it);
          else
            throw "more than one pointer operand in pointer arithmetic";
        }

      if(ptr_operand==NULL)
        throw "pointer type sum expected to have pointer operand";

      object_mapt pointer_expr_set;
      get_value_set_rec(*ptr_operand, pointer_expr_set, "", ptr_operand->type(), ns);

      for(object_map_dt::const_iterator
          it=pointer_expr_set.read().begin();
          it!=pointer_expr_set.read().end();
          it++)
      {
        objectt object=it->second;
      
        if(object.offset_is_zero() &&
           expr.operands().size()==2)
        {
          if(expr.op0().type().id()!="pointer")
          {
            mp_integer i;
            if(to_integer(expr.op0(), i))
              object.offset_is_set=false;
            else
              object.offset=i;
          }
          else
          {
            mp_integer i;
            if(to_integer(expr.op1(), i))
              object.offset_is_set=false;
            else
              object.offset=i;
          }
        }
        else
          object.offset_is_set=false;
          
        insert(dest, it->first, object);
      }

      return;
    }
  }
  else if(expr.id()=="sideeffect")
  {
    const irep_idt &statement=expr.statement();
    
    if(statement=="function_call")
    {
      // these should be gone
      throw "unexpected function_call sideeffect";
    }
    else if(statement=="malloc")
    {
      assert(suffix=="");
      
      const typet &dynamic_type=
        static_cast<const typet &>(expr.cmt_type());

      dynamic_object_exprt dynamic_object(dynamic_type);
      dynamic_object.instance()=from_integer(location_number, typet("natural"));
      dynamic_object.valid()=true_exprt();

      insert(dest, dynamic_object, 0);
      return;          
    }
    else if(statement=="cpp_new" ||
            statement=="cpp_new[]")
    {
      assert(suffix=="");
      assert(expr.type().id()=="pointer");

      dynamic_object_exprt dynamic_object(expr.type().subtype());
      dynamic_object.instance()=from_integer(location_number, typet("natural"));
      dynamic_object.valid()=true_exprt();

      insert(dest, dynamic_object, 0);
      return;
    }
  }
  else if(expr.id()=="struct")
  {
    // this is like a static struct object
    insert(dest, address_of_exprt(expr), 0);
    return;
  }
  else if(expr.id()=="with" ||
          expr.id()=="array_of" ||
          expr.is_array())
  {
    // these are supposed to be done by assign()
    throw "unexpected value in get_value_set: "+expr.id_string();
  }
  else if(expr.id()=="dynamic_object")
  {
    const dynamic_object_exprt &dynamic_object=
      to_dynamic_object_expr(expr);
  
    const std::string name=
      "value_set::dynamic_object"+
      dynamic_object.instance().value().as_string()+suffix;
  
    // look it up
    valuest::const_iterator v_it=values.find(name);

    if(v_it!=values.end())
    {
      make_union(dest, v_it->second.object_map);
      return;
    }
  }

  insert(dest, exprt("unknown", original_type));
}

/*******************************************************************\

Function: value_sett::dereference_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

/*******************************************************************\

Function: value_sett::get_reference_set

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void value_sett::get_reference_set(
  const exprt &expr,
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

/*******************************************************************\

Function: value_sett::get_reference_set_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void value_sett::get_reference_set_rec(
  const exprt &expr,
  object_mapt &dest,
  const namespacet &ns) const
{
  #if 0
  std::cout << "GET_REFERENCE_SET_REC EXPR: " << from_expr(ns, "", expr) << std::endl;
  #endif

  if(expr.id()=="symbol" ||
     expr.id()=="dynamic_object" ||
     expr.id()=="string-constant")
  {
    if(expr.type().is_array() &&
       expr.type().subtype().is_array())
      insert(dest, expr);
    else    
      insert(dest, expr, 0);

    return;
  }
  else if(expr.id()=="dereference" ||
          expr.id()=="implicit_dereference")
  {
    if(expr.operands().size()!=1)
      throw expr.id_string()+" expected to have one operand";

    get_value_set_rec(expr.op0(), dest, "", expr.op0().type(), ns);

    #if 0
    for(expr_sett::const_iterator it=value_set.begin(); it!=value_set.end(); it++)
      std::cout << "VALUE_SET: " << from_expr(ns, "", *it) << std::endl;
    #endif

    return;
  }
  else if(expr.id()=="index")
  {
    if(expr.operands().size()!=2)
      throw "index expected to have two operands";
  
    const exprt &array=expr.op0();
    const exprt &offset=expr.op1();
    const typet &array_type=ns.follow(array.type());
    
    assert(array_type.is_array() ||
           array_type.id()=="incomplete_array");
    
    object_mapt array_references;
    get_reference_set(array, array_references, ns);
        
    const object_map_dt &object_map=array_references.read();
    
    for(object_map_dt::const_iterator
        a_it=object_map.begin();
        a_it!=object_map.end();
        a_it++)
    {
      const exprt &object=object_numbering[a_it->first];

      if(object.id()=="unknown")
        insert(dest, exprt("unknown", expr.type()));
      else
      {
        index_exprt index_expr(expr.type());
        index_expr.array()=object;
        index_expr.index()=gen_zero(index_type());
        
        // adjust type?
        if(ns.follow(object.type())!=array_type)
          index_expr.make_typecast(array.type());
        
        objectt o=a_it->second;
        mp_integer i;

        if(offset.is_zero())
        {
        }
        else if(!to_integer(offset, i) &&
                o.offset_is_zero())
          o.offset=i;
        else
          o.offset_is_set=false;
          
        insert(dest, index_expr, o);
      }
    }
    
    return;
  }
  else if(expr.id()=="member")
  {
    const irep_idt &component_name=expr.component_name();

    if(expr.operands().size()!=1)
      throw "member expected to have one operand";
  
    const exprt &struct_op=expr.op0();
    
    object_mapt struct_references;
    get_reference_set(struct_op, struct_references, ns);
    
    const object_map_dt &object_map=struct_references.read();

    for(object_map_dt::const_iterator
        it=object_map.begin();
        it!=object_map.end();
        it++)
    {
      const exprt &object=object_numbering[it->first];
      
      if(object.id()=="unknown")
        insert(dest, exprt("unknown", expr.type()));
      else
      {
        objectt o=it->second;

        member_exprt member_expr(expr.type());
        member_expr.op0()=object;
        member_expr.set_component_name(component_name);
        
        // adjust type?
        if(ns.follow(struct_op.type())!=ns.follow(object.type()))
          member_expr.op0().make_typecast(struct_op.type());
        
        insert(dest, member_expr, o);
      }
    }

    return;
  }
  else if(expr.id()=="if")
  {
    if(expr.operands().size()!=3)
      throw "if takes three operands";

    get_reference_set_rec(expr.op1(), dest, ns);
    get_reference_set_rec(expr.op2(), dest, ns);
    return;
  }

  insert(dest, exprt("unknown", expr.type()));
}

/*******************************************************************\

Function: value_sett::assign

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void value_sett::assign(
  const exprt &lhs,
  const exprt &rhs,
  const namespacet &ns,
  bool add_to_sets)
{
  #if 0
  std::cout << "ASSIGN LHS: " << from_expr(ns, "", lhs) << std::endl;
  std::cout << "ASSIGN RHS: " << from_expr(ns, "", rhs) << std::endl;
  #endif

  if(rhs.id()=="if")
  {
    if(rhs.operands().size()!=3)
      throw "if takes three operands";

    assign(lhs, rhs.op1(), ns, add_to_sets);
    assign(lhs, rhs.op2(), ns, true);
    return;
  }

  const typet &type=ns.follow(lhs.type());
  
  if(type.id()=="struct" ||
     type.id()=="union")
  {
    const struct_typet &struct_type=to_struct_type(type);
    
    for(struct_typet::componentst::const_iterator
        c_it=struct_type.components().begin();
        c_it!=struct_type.components().end();
        c_it++)
    {
      const typet &subtype=c_it->type();
      const irep_idt &name=c_it->name();

      // ignore methods
      if(subtype.id()=="code") continue;
    
      member_exprt lhs_member(subtype);
      lhs_member.set_component_name(name);
      lhs_member.op0()=lhs;

      exprt rhs_member;

      if(rhs.id()=="unknown" ||
         rhs.id()=="invalid")
      {
        rhs_member=exprt(rhs.id(), subtype);
      }
      else
      {
        assert(base_type_eq(rhs.type(), type, ns));

        rhs_member=make_member(rhs, name, ns);
      
        assign(lhs_member, rhs_member, ns, add_to_sets);
      }
    }
  }
  else if(type.is_array())
  {
    exprt lhs_index("index", type.subtype());
    lhs_index.copy_to_operands(lhs, exprt("unknown", index_type()));

    if(rhs.id()=="unknown" ||
       rhs.id()=="invalid")
    {
      assign(lhs_index, exprt(rhs.id(), type.subtype()), ns, add_to_sets);
    }
    else
    {
      assert(base_type_eq(rhs.type(), type, ns));
        
      if(rhs.id()=="array_of")
      {
        assert(rhs.operands().size()==1);
        assign(lhs_index, rhs.op0(), ns, add_to_sets);
      }
      else if(rhs.is_array() ||
              rhs.id()=="constant")
      {
        forall_operands(o_it, rhs)
        {
          assign(lhs_index, *o_it, ns, add_to_sets);
          add_to_sets=true;
        }
      }
      else if(rhs.id()=="with")
      {
        assert(rhs.operands().size()==3);

        exprt op0_index("index", type.subtype());
        op0_index.copy_to_operands(rhs.op0(), exprt("unknown", index_type()));

        assign(lhs_index, op0_index, ns, add_to_sets);
        assign(lhs_index, rhs.op2(), ns, true);
      }
      else
      {
        exprt rhs_index("index", type.subtype());
        rhs_index.copy_to_operands(rhs, exprt("unknown", index_type()));
        assign(lhs_index, rhs_index, ns, true);
      }
    }
  }
  else
  {
    // basic type
    object_mapt values_rhs;
    
    get_value_set(rhs, values_rhs, ns);
    
    assign_rec(lhs, values_rhs, "", ns, add_to_sets);
  }
}

/*******************************************************************\

Function: value_sett::do_free

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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
    const exprt &object=object_numbering[it->first];

    if(object.id()=="dynamic_object")
    {
      const dynamic_object_exprt &dynamic_object=
        to_dynamic_object_expr(object);
      
      if(dynamic_object.valid().is_true())
        to_mark.insert(dynamic_object.instance());
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
      const exprt &object=object_numbering[o_it->first];

      if(object.id()=="dynamic_object")
      {
        const exprt &instance=
          to_dynamic_object_expr(object).instance();

        if(to_mark.count(instance)==0)
          set(new_object_map, o_it);
        else
        {
          // adjust
          objectt o=o_it->second;
          exprt tmp(object);
          to_dynamic_object_expr(tmp).valid()=exprt("unknown");
          insert(new_object_map, tmp, o);
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

/*******************************************************************\

Function: value_sett::assign_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void value_sett::assign_rec(
  const exprt &lhs,
  const object_mapt &values_rhs,
  const std::string &suffix,
  const namespacet &ns,
  bool add_to_sets)
{
  #if 0
  std::cout << "ASSIGN_REC LHS: " << from_expr(ns, "", lhs) << std::endl;
  std::cout << "ASSIGN_REC SUFFIX: " << suffix << std::endl;

  for(object_map_dt::const_iterator it=values_rhs.read().begin(); 
      it!=values_rhs.read().end(); 
      it++)
    std::cout << "ASSIGN_REC RHS: " << 
      object_numbering[it->first] << std::endl;
  #endif

  if(lhs.id()=="symbol")
  {
    const irep_idt &identifier=lhs.identifier();
    
    if(add_to_sets)
      make_union(get_entry(identifier, suffix).object_map, values_rhs);
    else
      get_entry(identifier, suffix).object_map=values_rhs;
  }
  else if(lhs.id()=="dynamic_object")
  {
    const dynamic_object_exprt &dynamic_object=
      to_dynamic_object_expr(lhs);
  
    const std::string name=
      "value_set::dynamic_object"+
      dynamic_object.instance().value().as_string();

    make_union(get_entry(name, suffix).object_map, values_rhs);
  }
  else if(lhs.id()=="dereference" ||
          lhs.id()=="implicit_dereference")
  {
    if(lhs.operands().size()!=1)
      throw lhs.id_string()+" expected to have one operand";
      
    object_mapt reference_set;
    get_reference_set(lhs, reference_set, ns);
    
    if(reference_set.read().size()!=1)
      add_to_sets=true;
      
    for(object_map_dt::const_iterator
        it=reference_set.read().begin();
        it!=reference_set.read().end();
        it++)
    {
      const exprt &object=object_numbering[it->first];

      if(object.id()!="unknown")
        assign_rec(object, values_rhs, suffix, ns, add_to_sets);
    }
  }
  else if(lhs.id()=="index")
  {
    if(lhs.operands().size()!=2)
      throw "index expected to have two operands";
      
    const typet &type=ns.follow(lhs.op0().type());
      
    assert(type.is_array() || type.id()=="incomplete_array");

    assign_rec(lhs.op0(), values_rhs, "[]"+suffix, ns, true);
  }
  else if(lhs.id()=="member")
  {
    if(lhs.operands().size()!=1)
      throw "member expected to have one operand";
  
    const std::string &component_name=lhs.component_name().as_string();

    const typet &type=ns.follow(lhs.op0().type());

    assert(type.id()=="struct" ||
           type.id()=="union" ||
           type.id()=="incomplete_struct" ||
           type.id()=="incomplete_union");
           
    assign_rec(lhs.op0(), values_rhs, "."+component_name+suffix, ns, add_to_sets);
  }
  else if(lhs.id()=="valid_object" ||
		  lhs.id()=="deallocated_object" ||
          lhs.id()=="dynamic_size" ||
          lhs.id()=="dynamic_type" ||
          lhs.id()=="is_zero_string" ||
          lhs.id()=="zero_string" ||
          lhs.id()=="zero_string_length")
  {
    // we ignore this here
  }
  else if(lhs.id()=="string-constant")
  {
    // someone writes into a string-constant
    // evil guy
  }
  else if(lhs.id()=="NULL-object")
  {
    // evil as well
  }
  else if(lhs.id()=="typecast")
  {
    const typecast_exprt &typecast_expr=to_typecast_expr(lhs);
  
    assign_rec(typecast_expr.op(), values_rhs, suffix, ns, add_to_sets);
  }
  else if(lhs.id()=="byte_extract_little_endian" ||
          lhs.id()=="byte_extract_big_endian")
  {
    assert(lhs.operands().size()==2);
    assign_rec(lhs.op0(), values_rhs, suffix, ns, true);
  }
  else
    throw "assign NYI: `"+lhs.id_string()+"'";
}

/*******************************************************************\

Function: value_sett::do_function_call

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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
    assign(dummy_lhs, arguments[i], ns, true);
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
    assign(actual_lhs, v_expr, ns, true);
    i++;
  }
}

/*******************************************************************\

Function: value_sett::do_end_function

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void value_sett::do_end_function(
  const exprt &lhs,
  const namespacet &ns)
{
  if(lhs.is_nil()) return;

  symbol_exprt rhs("value_set::return_value", lhs.type());

  assign(lhs, rhs, ns);
}

/*******************************************************************\

Function: value_sett::apply_code

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

    assign(code.op0(), code.op1(), ns);
  }
  else if(statement=="decl")
  {
    if(code.operands().size()!=1)
      throw "decl expected to have one operand";

    const exprt &lhs=code.op0();

    if(lhs.id()!="symbol")
      throw "decl expected to have symbol on lhs";

    assign(lhs, exprt("invalid", lhs.type()), ns);
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
      assign(lhs, code.op0(), ns);
    }
  }
  else
  {
    std::cerr << code.pretty() << std::endl;
    throw "value_sett: unexpected statement: "+id2string(statement);
  }
}

/*******************************************************************\

Function: value_sett::make_member

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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
    assert(no<src.operands().size());
    return src.operands()[no];
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
