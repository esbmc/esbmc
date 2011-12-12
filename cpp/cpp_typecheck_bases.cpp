/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include "cpp_typecheck.h"
#include <set>

/*******************************************************************\

Function: cpp_typecheckt::typcheck_compound_bases

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::typecheck_compound_bases(typet &type)
{
  std::set<irep_idt> bases;
  std::set<irep_idt> vbases;

  if(type.id() == "union"
     && !type.add("bases").get_sub().empty())
  {
    err_location(type);
    str << "error: unions don't have base types";
    throw 0;
  }

  irep_idt default_class_access = type.get_bool("#class")? "private": "public";

  Forall_irep(base_it, type.add("bases").get_sub())
  {
    cpp_namet &name=(cpp_namet &)base_it->add("name");
    exprt symbol_expr;

    resolve(
      name,
      cpp_typecheck_resolvet::TYPE,
      cpp_typecheck_fargst(),
      symbol_expr);

    if(symbol_expr.id()!="type" ||
       symbol_expr.type().id()!="symbol")
    {
      err_location(name.location());
      str << "error: expected type";
      throw 0;
    }

    const symbolt &symbol=lookup(symbol_expr.type());

    if(symbol.type.id()=="incomplete_struct" ||
       symbol.type.id()=="incomplete_class")
    {
      err_location(name.location());
      str << "error: base type is incomplete";
      throw 0;
    }
    else if(symbol.type.id()!="struct")
    {
      err_location(name.location());
      str << "error: expected struct or class, but got `"
          << to_string(symbol.type) << "'";
      throw 0;
    }

    bool virtual_base = base_it->get_bool("virtual");
    irep_idt class_access = base_it->get("protection");

    if(class_access == "")
    {
      class_access = default_class_access;
    }

    symbol_expr.id("base");
    symbol_expr.set("access",class_access);

    if(virtual_base)
      symbol_expr.set("virtual",true);

    base_it->swap(symbol_expr);

    // Add base scopes to the current scopes
    cpp_scopes.current_scope().add_parent(*cpp_scopes.id_map[symbol.name]);

    struct_typet base_struct = to_struct_type(symbol.type);
    add_base_components(base_struct,
                          class_access, to_struct_type(type),
                          bases, vbases, virtual_base);
  }

  if(!vbases.empty())
  {
    // add a flag to determined
    // if this is the most-derived-object
    struct_typet::componentt most_derived;
    most_derived.set("type", "bool");
    most_derived.set("access", "public");
    most_derived.set("base_name", "@most_derived");
    most_derived.set_name(cpp_identifier_prefix(current_mode)+"::"+
                     cpp_scopes.current_scope().prefix+"::"+"@most_derived");
    most_derived.set("pretty_name","@most_derived");
    most_derived.location()=type.location();
    put_compound_into_scope(most_derived);
    to_struct_type(type).components().push_back(most_derived);
  }
}


/*******************************************************************\

Function: cpp_typecheckt::add_base_components

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::add_base_components(
        const struct_typet& from,
        const irep_idt& access,
        struct_typet& to,
        std::set<irep_idt>& bases,
        std::set<irep_idt>& vbases,
        bool is_virtual)
{
  const irep_idt& from_name = from.get("name");

  if(is_virtual && vbases.find(from_name) != vbases.end())
    return;

  if(bases.find(from_name) != bases.end())
  {
    err_location(to);
    str << "error: base class " << from_name << " inherited multiple times";
    throw 0;
  }

  bases.insert(from_name);
  if(is_virtual)
    vbases.insert(from_name);

  // call the parents
  forall_irep(it, from.find("bases").get_sub())
  {

    irep_idt sub_access = it->get("access");
    if(access == "private")
      sub_access = "private";
    else if(access == "protected" && sub_access != "private")
      sub_access = "protected";

    const symbolt&  symb= lookup(it->find("type").get("identifier"));

    bool is_virtual = it->get_bool("virtual");

    add_base_components(to_struct_type(symb.type),
                        sub_access, to, bases, vbases, is_virtual);
  }

  // add the components
  const struct_typet::componentst& src_c = from.components();
  struct_typet::componentst& dest_c = to.components();

  if(access == "public")
  {
    for(struct_typet::componentst::const_iterator it = src_c.begin();
        it != src_c.end(); it++)
    {
      if(it->get_bool("from_base"))
        continue;

      dest_c.push_back(*it);
      exprt &component=(exprt &)dest_c.back();
      component.set("from_base", true);
      if(component.get("access")=="private")
        component.set("access","noaccess");
    }
  }
  else if(access == "protected")
  {
    for(struct_typet::componentst::const_iterator it = src_c.begin();
        it != src_c.end(); it++)
    {
      if(it->get_bool("from_base"))
        continue;

      dest_c.push_back(*it);
      exprt &component=(exprt &)dest_c.back();
      component.set("from_base", true);
      if(component.get("access")=="private")
        component.set("access","noaccess");
      else component.set("access","private");
    }
  }
  else if(access == "private")
  {
    for(struct_typet::componentst::const_iterator it = src_c.begin();
        it != src_c.end(); it++)
    {
      if(it->get_bool("from_base"))
        continue;

      dest_c.push_back(*it);
      exprt &component=(exprt &)dest_c.back();
      component.set("from_base", true);
      irep_idt comp_access = component.get("access");
      if(comp_access == "noaccess" || comp_access == "private")
        component.set("access","noaccess");
      else component.set("access","private");
    }
  }
  else
    assert(0);

}


