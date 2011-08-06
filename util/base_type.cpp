/*******************************************************************\

Module: Base Type Computation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include "std_types.h"
#include "base_type.h"
#include "union_find.h"

/*******************************************************************\

Function: base_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void base_type(typet &type, const namespacet &ns)
{
  if(type.id()=="symbol")
  {
    const symbolt *symbol;

    if(!ns.lookup(type.identifier(), symbol) &&
       symbol->is_type &&
       !symbol->type.is_nil())
    {
      type=symbol->type;
      base_type(type, ns); // recursive call
      return;
    }
  }
  //else if(src.id()=="dependent")
  //  dest=src.find("subtype");
  else if(type.id()=="subtype")
  {
    typet tmp;
    tmp.swap(type.subtype());
    type.swap(tmp);
  }
#if 0
// Disabled by jmorse - no-where else in ESBMC is "predicate" used.
  else if(type.id()=="predicate")
  {
    exprt &predicate=(exprt &)type.add("predicate");
    base_type(predicate.type(), ns);
    assert(predicate.type().id()=="mapping");
    typet tmp;
    tmp.swap(predicate.type().subtypes()[0]);
    type.swap(tmp);
    base_type(type, ns); // recursive call
  }
#endif
  else if(type.id()=="mapping")
  {
    assert(type.subtypes().size()==2);
    base_type(type.subtypes()[0], ns);
    base_type(type.subtypes()[1], ns);
  }
  else if(type.is_array())
  {
    base_type(type.subtype(), ns);
  }
  else if(type.id()=="struct" ||
          type.id()=="class" ||
          type.id()=="union")
  {
    // New subt for manipulating components
    irept::subt components=type.components().get_sub();

    Forall_irep(it, components)
    {
      typet &subtype=it->type();
      base_type(subtype, ns);
    }

    // Set back into type
    irept tmp = type.components();
    tmp.get_sub() = components;
    type.components(tmp);
  }
}

/*******************************************************************\

Function: base_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void base_type(exprt &expr, const namespacet &ns)
{
  base_type(expr.type(), ns);

  Forall_operands(it, expr)
    base_type(*it, ns);
}

/*******************************************************************\

Function: base_type_eqt::base_type_eq_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool base_type_eqt::base_type_eq_rec(
  const typet &type1,
  const typet &type2)
{
  if(type1==type2)
    return true;
    
  #if 0
  std::cout << "T1: " << type1.pretty() << std::endl;
  std::cout << "T2: " << type2.pretty() << std::endl;
  #endif
  
  // loop avoidance
  if(type1.id()=="symbol" &&
     type2.id()=="symbol")
  {
    // already in same set?
    if(identifiers.make_union(
         type1.identifier(),
         type2.identifier()))
      return true;
  }

  if(type1.id()=="symbol")
  {
    const symbolt &symbol=ns.lookup(type1.identifier());

    if(!symbol.is_type)
      throw "symbol "+id2string(symbol.name)+" is not a type";
    
    return base_type_eq_rec(symbol.type, type2);
  }

  if(type2.id()=="symbol")
  {
    const symbolt &symbol=ns.lookup(type2.identifier());

    if(!symbol.is_type)
      throw "symbol "+id2string(symbol.name)+" is not a type";

    return base_type_eq_rec(type1, symbol.type);
  }
  
  if(type1.id()!=type2.id())
    return false;

  if(type1.id()=="struct" ||
     type1.id()=="class" ||
     type1.id()=="union")
  {
    const struct_union_typet::componentst &components1=
      to_struct_union_type(type1).components();

    const struct_union_typet::componentst &components2=
      to_struct_union_type(type2).components();
      
    if(components1.size()!=components2.size())
      return false;

    for(unsigned i=0; i<components1.size(); i++)
    {
      const typet &subtype1=components1[i].type();
      const typet &subtype2=components2[i].type();
      if(!base_type_eq_rec(subtype1, subtype2)) return false;
      if(components1[i].get_name()!=components2[i].get_name()) return false;
    }
    
    return true;
  }
  else if(type1.id()=="incomplete_struct")
  {
    return true;
  }
  else if(type1.id()=="code")
  {
    const irept::subt &arguments1=type1.arguments().get_sub();

    const irept::subt &arguments2=type2.arguments().get_sub();
    
    if(arguments1.size()!=arguments2.size())
      return false;
      
    for(unsigned i=0; i<arguments1.size(); i++)
    {
      const typet &subtype1=arguments1[i].type();
      const typet &subtype2=arguments2[i].type();
      if(!base_type_eq_rec(subtype1, subtype2)) return false;
    }
    
    const typet &return_type1=(typet &)type1.return_type();
    const typet &return_type2=(typet &)type2.return_type();
    
    if(!base_type_eq_rec(return_type1, return_type2))
      return false;
    
    return true;
  }
  else if(type1.id()=="pointer")
  {
    return base_type_eq_rec(type1.subtype(), type2.subtype());
  }
  else if(type1.is_array())
  {
    if(!base_type_eq_rec(type1.subtype(), type2.subtype()))
      return false;
      
    // TODO: check size
      
    return true;
  }
  else if(type1.id()=="incomplete_array")
  {
    return base_type_eq_rec(type1.subtype(), type2.subtype());
  }

  typet tmp1(type1), tmp2(type2);

  base_type(tmp1, ns);
  base_type(tmp2, ns);

  return tmp1==tmp2;  
}

/*******************************************************************\

Function: base_type_eqt::base_type_eq_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool base_type_eqt::base_type_eq_rec(
  const exprt &expr1,
  const exprt &expr2)
{
  if(expr1.id()!=expr2.id())
    return false;
    
  if(!base_type_eq(expr1.type(), expr2.type()))
    return false;

  if(expr1.operands().size()!=expr2.operands().size())
    return false;
    
  for(unsigned i=0; i<expr1.operands().size(); i++)
    if(!base_type_eq(expr1.operands()[i], expr2.operands()[i]))
      return false;
  
  return true;
}

/*******************************************************************\

Function: base_type_eq

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool base_type_eq(
  const typet &type1,
  const typet &type2,
  const namespacet &ns)
{
  base_type_eqt base_type_eq(ns);
  return base_type_eq.base_type_eq(type1, type2);
}

/*******************************************************************\

Function: base_type_eq

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool base_type_eq(
  const exprt &expr1,
  const exprt &expr2,
  const namespacet &ns)
{
  base_type_eqt base_type_eq(ns);
  return base_type_eq.base_type_eq(expr1, expr2);
}
