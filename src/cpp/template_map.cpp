/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/template_map.h>

void template_mapt::apply(typet &type) const
{
  if(type.id() == "array")
  {
    apply(type.subtype());
    apply(static_cast<exprt &>(type.add("size")));
  }
  else if(type.id() == "pointer")
  {
    apply(type.subtype());
  }
  else if(type.id() == "struct" || type.id() == "union")
  {
    irept::subt &components = type.add("components").get_sub();

    Forall_irep(it, components)
    {
      typet &subtype = static_cast<typet &>(it->type());
      apply(subtype);
    }
  }
  else if(type.id() == "symbol")
  {
    type_mapt::const_iterator m_it = type_map.find(type.identifier());

    if(m_it != type_map.end())
    {
      type = m_it->second;
      return;
    }
  }
  else if(type.id() == "code")
  {
    apply(static_cast<typet &>(type.add("return_type")));

    irept::subt &arguments = type.add("arguments").get_sub();

    Forall_irep(it, arguments)
    {
      if(it->id() == "argument")
        apply(static_cast<typet &>(it->type()));
    }
  }
  else if(type.id() == "merged_type")
  {
    Forall_subtypes(it, type)
      apply(*it);
  }
}

void template_mapt::apply(exprt &expr) const
{
  apply(expr.type());

  if(expr.id() == "symbol")
  {
    expr_mapt::const_iterator m_it = expr_map.find(expr.identifier());

    if(m_it != expr_map.end())
    {
      expr = m_it->second;
      return;
    }
  }

  Forall_operands(it, expr)
    apply(*it);
}

exprt template_mapt::lookup(const irep_idt &identifier) const
{
  type_mapt::const_iterator t_it = type_map.find(identifier);

  if(t_it != type_map.end())
  {
    exprt e("type");
    e.type() = t_it->second;
    return e;
  }

  expr_mapt::const_iterator e_it = expr_map.find(identifier);

  if(e_it != expr_map.end())
    return e_it->second;

  return static_cast<const exprt &>(get_nil_irep());
}

typet template_mapt::lookup_type(const irep_idt &identifier) const
{
  type_mapt::const_iterator t_it = type_map.find(identifier);

  if(t_it != type_map.end())
    return t_it->second;

  return static_cast<const typet &>(get_nil_irep());
}

exprt template_mapt::lookup_expr(const irep_idt &identifier) const
{
  expr_mapt::const_iterator e_it = expr_map.find(identifier);

  if(e_it != expr_map.end())
    return e_it->second;

  return static_cast<const exprt &>(get_nil_irep());
}

void template_mapt::print(std::ostream &out) const
{
  for(const auto &it : type_map)
    out << it.first << " = " << it.second.pretty() << std::endl;

  for(const auto &it : expr_map)
    out << it.first << " = " << it.second.pretty() << std::endl;
}

void template_mapt::build(
  const template_typet &template_type,
  const cpp_template_args_tct &template_args)
{
  const template_typet::parameterst &template_parameters =
    template_type.parameters();

  cpp_template_args_tct::argumentst instance = template_args.arguments();

  template_typet::parameterst::const_iterator t_it =
    template_parameters.begin();

  if(instance.size() < template_parameters.size())
  {
    // check for default parameters
    for(unsigned i = instance.size(); i < template_parameters.size(); i++)
    {
      const template_parametert &param = template_parameters[i];

      if(param.has_default_parameter())
        instance.push_back(param.default_parameter());
      else
        break;
    }
  }

  // these should have been typechecked before
  assert(instance.size() == template_parameters.size());

  for(cpp_template_args_tct::argumentst::const_iterator i_it = instance.begin();
      i_it != instance.end();
      i_it++, t_it++)
  {
    assert(t_it != template_parameters.end());
    set(*t_it, *i_it);
  }
}

void template_mapt::set(
  const template_parametert &parameter,
  const exprt &value)
{
  if(parameter.id() == "type")
  {
    if(parameter.id() != "type")
      assert(false); // typechecked before!

    typet tmp = value.type();

    irep_idt identifier = parameter.type().identifier();
    type_map[identifier] = tmp;
  }
  else
  {
    // must be non-type

    if(value.id() == "type")
      assert(false); // typechecked before!

    irep_idt identifier = parameter.identifier();
    expr_map[identifier] = value;
  }
}

void template_mapt::build_unassigned(const template_typet &template_type)
{
  const template_typet::parameterst &template_parameters =
    template_type.parameters();

  for(const auto &t : template_parameters)
  {
    if(t.id() == "type")
    {
      typet tmp("unassigned");
      tmp.identifier(t.type().identifier());
      tmp.location() = t.location();
      type_map[t.type().identifier()] = tmp;
    }
    else
    {
      exprt tmp("unassigned", t.type());
      tmp.identifier(t.identifier());
      tmp.location() = t.location();
      expr_map[t.identifier()] = tmp;
    }
  }
}

cpp_template_args_tct
template_mapt::build_template_args(const template_typet &template_type) const
{
  const template_typet::parameterst &template_parameters =
    template_type.parameters();

  cpp_template_args_tct template_args;
  template_args.arguments().resize(template_parameters.size());

  for(unsigned i = 0; i < template_parameters.size(); i++)
  {
    const exprt &t = template_parameters[i];

    if(t.id() == "type")
    {
      template_args.arguments()[i] =
        exprt("type", lookup_type(t.type().identifier()));
    }
    else
    {
      template_args.arguments()[i] = lookup_expr(t.identifier());
    }
  }

  return template_args;
}
