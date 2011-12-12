/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include "template_map.h"

/*******************************************************************\

Function: template_mapt::apply

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void template_mapt::apply(typet &type) const
{
  if(type.id()=="array")
  {
    apply(type.subtype());
    apply((exprt &)type.add("size"));
  }
  else if(type.id()=="pointer")
  {
    apply(type.subtype());
  }
  else if(type.id()=="struct" ||
          type.id()=="union")
  {
    irept::subt &components=type.add("components").get_sub();

    Forall_irep(it, components)
    {
      typet &subtype=(typet &)it->find("type");
      apply(subtype);
    }
  }
  else if(type.id()=="symbol")
  {
    type_mapt::const_iterator m_it=
      type_map.find(type.get("identifier"));

    if(m_it!=type_map.end())
    {
      type=m_it->second;
      return;
    }
  }
  else if(type.id()=="code")
  {
    apply((typet &)type.add("return_type"));

    irept::subt &arguments=type.add("arguments").get_sub();

    Forall_irep(it, arguments)
    {
      if(it->id()=="argument")
        apply((typet &)(it->add("type")));
    }
  }
  else if(type.id()=="merged_type")
  {
    Forall_subtypes(it, type)
      apply(*it);
  }
}

/*******************************************************************\

Function: template_mapt::apply

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void template_mapt::apply(exprt &expr) const
{
  apply(expr.type());

  if(expr.id()=="symbol")
  {
    expr_mapt::const_iterator m_it=
      expr_map.find(expr.get("identifier"));

    if(m_it!=expr_map.end())
    {
      expr=m_it->second;
      return;
    }
  }

  Forall_operands(it, expr)
    apply(*it);
}

/*******************************************************************\

Function: template_mapt::apply

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt template_mapt::lookup(const irep_idt &identifier) const
{
  type_mapt::const_iterator t_it=
    type_map.find(identifier);

  if(t_it!=type_map.end())
  {
    exprt e("type");
    e.type()=t_it->second;
    return e;
  }

  expr_mapt::const_iterator e_it=
    expr_map.find(identifier);

  if(e_it!=expr_map.end())
    return e_it->second;

  return static_cast<const exprt &>(get_nil_irep());
}

/*******************************************************************\

Function: template_mapt::print

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void template_mapt::print(std::ostream &out) const
{
  for(type_mapt::const_iterator it=type_map.begin();
      it!=type_map.end();
      it++)
    out << it->first << " = " << it->second.pretty() << std::endl;

  for(expr_mapt::const_iterator it=expr_map.begin();
      it!=expr_map.end();
      it++)
    out << it->first << " = " << it->second.pretty() << std::endl;
}

