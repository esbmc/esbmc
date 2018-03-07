/*******************************************************************\

Module: Value Set Propagation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <langapi/language_util.h>
#include <pointer-analysis/value_set_analysis.h>
#include <util/cprover_prefix.h>
#include <util/prefix.h>
#include <util/xml_irep.h>

void value_set_analysist::initialize(const goto_programt &goto_program)
{
  baset::initialize(goto_program);
  add_vars(goto_program);
}

void value_set_analysist::initialize(const goto_functionst &goto_functions)
{
  baset::initialize(goto_functions);
  add_vars(goto_functions);
}

void value_set_analysist::add_vars(const goto_programt &goto_program)
{
  typedef std::list<value_sett::entryt> entry_listt;

  // get the globals
  entry_listt globals;
  get_globals(globals);

  // cache the list for the locals to speed things up
  typedef hash_map_cont<irep_idt, entry_listt, irep_id_hash> entry_cachet;
  entry_cachet entry_cache;

  for(goto_programt::instructionst::const_iterator i_it =
        goto_program.instructions.begin();
      i_it != goto_program.instructions.end();
      i_it++)
  {
    value_sett &v = *(*this)[i_it].value_set;
    v.add_vars(globals);

    for(auto const &l_it : goto_program.local_variables)
    {
      // cache hit?
      entry_cachet::const_iterator e_it = entry_cache.find(l_it);

      if(e_it == entry_cache.end())
      {
        const symbolt &symbol = ns.lookup(l_it);

        std::list<value_sett::entryt> &entries = entry_cache[l_it];
        get_entries(symbol, entries);
        v.add_vars(entries);
      }
      else
        v.add_vars(e_it->second);
    }
  }
}

void value_set_analysist::get_entries(
  const symbolt &symbol,
  std::list<value_sett::entryt> &dest)
{
  get_entries_rec(symbol.name.as_string(), "", symbol.type, dest);
}

void value_set_analysist::get_entries_rec(
  const std::string &identifier,
  const std::string &suffix,
  const typet &type,
  std::list<value_sett::entryt> &dest)
{
  const typet &t = ns.follow(type);

  if(t.id() == "struct" || t.id() == "union")
  {
    const struct_typet &struct_type = to_struct_type(t);

    const struct_typet::componentst &c = struct_type.components();

    for(const auto &it : c)
    {
      get_entries_rec(
        identifier, suffix + "." + it.name().as_string(), it.type(), dest);
    }
  }
  else if(t.is_array())
  {
    get_entries_rec(identifier, suffix + "[]", t.subtype(), dest);
  }
  else if(check_type(t))
  {
    dest.emplace_back(identifier, std::move(suffix));
  }
}

void value_set_analysist::add_vars(const goto_functionst &goto_functions)
{
  // get the globals
  std::list<value_sett::entryt> globals;
  get_globals(globals);

  for(goto_functionst::function_mapt::const_iterator f_it =
        goto_functions.function_map.begin();
      f_it != goto_functions.function_map.end();
      f_it++)

    forall_goto_program_instructions(i_it, f_it->second.body)
    {
      value_sett &v = *(*this)[i_it].value_set;
      v.add_vars(globals);

      for(auto const &l_it : f_it->second.body.local_variables)
      {
        const symbolt &symbol = ns.lookup(l_it);

        std::list<value_sett::entryt> entries;
        get_entries(symbol, entries);
        v.add_vars(entries);
      }
    }
}

void value_set_analysist::get_globals(std::list<value_sett::entryt> &dest)
{
  // static ones
  ns.get_context().foreach_operand([this, &dest](const symbolt &s) {
    if(s.lvalue && s.static_lifetime)
      get_entries(s, dest);
  });
}

bool value_set_analysist::check_type(const typet &type)
{
  if(type.id() == "pointer")
    return true;
  if(type.id() == "struct" || type.id() == "union")
  {
    const struct_typet &struct_type = to_struct_type(type);

    const struct_typet::componentst &components = struct_type.components();

    for(const auto &component : components)
    {
      if(check_type(component.type()))
        return true;
    }
  }
  else if(type.is_array())
    return check_type(type.subtype());
  else if(type.id() == "symbol")
    return check_type(ns.follow(type));

  return false;
}

void value_set_analysist::convert(
  const goto_programt &goto_program,
  const irep_idt &identifier __attribute__((unused)),
  xmlt &dest) const
{
  ::locationt previous_location;

  forall_goto_program_instructions(i_it, goto_program)
  {
    const ::locationt &location = i_it->location;

    if(location == previous_location)
      continue;

    if(location.is_nil() || location.get_file() == "")
      continue;

    // find value set
    const value_sett &value_set = *(*this)[i_it].value_set;

    xmlt &i = dest.new_element("instruction");
    xmlt &xml_location = i.new_element("location");
    ::convert(location, xml_location);
    xml_location.name = "location";

    for(const auto &value : value_set.values)
    {
      xmlt &var = i.new_element("variable");
      var.new_element("identifier").data = value.first.the_string;

#if 0
      const value_sett::expr_sett &expr_set=
        v_it->second.expr_set();

      for(value_sett::expr_sett::const_iterator
          e_it=expr_set.begin();
          e_it!=expr_set.end();
          e_it++)
      {
        std::string value_str=
          from_expr(ns, identifier, *e_it);

        var.new_element("value").data=
          xmlt::escape(value_str);
      }
#endif
    }
  }
}

void convert(
  const goto_functionst &goto_functions,
  const value_set_analysist &value_set_analysis,
  xmlt &dest)
{
  dest = xmlt("value_set_analysis");

  for(const auto &f_it : goto_functions.function_map)
  {
    xmlt &f = dest.new_element("function");

    f.new_element("identifier").data = xmlt::escape(id2string(f_it.first));

    value_set_analysis.convert(f_it.second.body, f_it.first, f);
  }
}

void convert(
  const goto_programt &goto_program,
  const value_set_analysist &value_set_analysis,
  xmlt &dest)
{
  dest = xmlt("value_set_analysis");

  value_set_analysis.convert(goto_program, "", dest.new_element("program"));
}
