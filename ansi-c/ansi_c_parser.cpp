/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "ansi_c_parser.h"

ansi_c_parsert ansi_c_parser;

/*******************************************************************\

Function: ansi_c_parsert::scopet::print

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_parsert::scopet::print(std::ostream &out) const
{
  out << "Prefix: " << prefix << std::endl;

  for(scopet::name_mapt::const_iterator n_it=name_map.begin();
      n_it!=name_map.end();
      n_it++)
  {
    out << "  ID: " << n_it->first
        << " CLASS: " << n_it->second.id_class
        << std::endl;
  }
}

/*******************************************************************\

Function: ansi_c_parsert::lookup

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

ansi_c_id_classt ansi_c_parsert::lookup(std::string &name, bool tag) const
{
  const std::string scope_name=tag?"tag-"+name:name;
  
  for(scopest::const_reverse_iterator it=scopes.rbegin();
      it!=scopes.rend(); it++)
  {
    scopet::name_mapt::const_iterator n_it=it->name_map.find(scope_name);
    if(n_it!=it->name_map.end())
    {
      name=it->prefix+scope_name;
      return n_it->second.id_class;
    }
  }

  return ANSI_C_UNKNOWN;
}

/*******************************************************************\

Function: yyansi_cerror

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

extern char *yyansi_ctext;

int yyansi_cerror(const std::string &error)
{
  ansi_c_parser.parse_error(error, yyansi_ctext);
  return 0;
}

static void insert_subtype(irept &target, const typet &type)
{

  const irept &atype = target.find("subtype");
  if (atype.id() == "nil" || !atype.is_nil()) {
    target.add("subtype") = type;
  } else {
    typet *wheretoadd = &(typet&)target.add("subtype");
    while (wheretoadd->id() != "nil" && !wheretoadd->is_nil())
      wheretoadd = (typet *)&wheretoadd->find("subtype");
    *wheretoadd = type;
  }

  return;
}

/*******************************************************************\

Function: ansi_c_parsert::convert_declarator

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_parsert::convert_declarator(
  irept &declarator,
  const typet &type,
  irept &identifier)
{
  typet *p=(typet *)&declarator;

  if (declarator.find("declarator").id() != "nil") {
    identifier = declarator.find("declarator");
    declarator.remove("declarator");
    insert_subtype(declarator, type);
    return;
  }

  // Otherwise, this is not a normal symbol def, it's a stuct member perhaps
  assert(declarator.id() == "symbol");

  insert_subtype(declarator, type);

  identifier = declarator;
  return;
}

/*******************************************************************\

Function: ansi_c_parsert::new_declaration

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_parsert::new_declaration(
  const irept &type,
  irept &declarator,
  exprt &dest,
  bool is_tag,
  bool put_into_scope)
{
  exprt identifier;

  convert_declarator(declarator, static_cast<const typet &>(type), identifier);
  typet final_type=static_cast<typet &>(declarator);
  
  std::string base_name=identifier.get_string("#base_name");
  
  bool is_global=current_scope().prefix=="";

  ansi_c_id_classt id_class=get_class(final_type);
  
  const std::string scope_name=
    is_tag?"tag-"+base_name:base_name;
    
  if(is_tag)
    final_type.set("tag", base_name);

  std::string name;

  if(base_name!="")
  {  
    name=current_scope().prefix+scope_name;

    if(put_into_scope)
    {
      // see if already in scope
      scopet::name_mapt::const_iterator n_it=
        current_scope().name_map.find(scope_name);
    
      if(n_it==current_scope().name_map.end())
      {
        // add to scope  
        current_scope().name_map[scope_name].id_class=id_class;
      }
    }
  }

  // create dest
  ansi_c_declarationt declaration;

  declaration.type().swap(final_type);
  declaration.set_base_name(base_name);
  declaration.set_name(name);
  declaration.location()=identifier.location();
  declaration.value().make_nil();
  declaration.set_is_type(is_tag || id_class==ANSI_C_TYPEDEF);
  declaration.set_is_typedef(id_class==ANSI_C_TYPEDEF);
  declaration.set_is_global(is_global);
  
  dest.swap(declaration);
}

/*******************************************************************\

Function: ansi_c_parsert::get_class

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
 
ansi_c_id_classt ansi_c_parsert::get_class(const typet &type)
{
  if(type.id()=="typedef")
    return ANSI_C_TYPEDEF;
  else if(type.id()=="struct" ||
          type.id()=="union" ||
          type.id()=="c_enum")
    return ANSI_C_TAG;
  else if(type.id()=="merged_type")
  {
    forall_subtypes(it, type)
      if(get_class(*it)==ANSI_C_TYPEDEF)
        return ANSI_C_TYPEDEF;
  }
  else if(type.has_subtype())
    return get_class(type.subtype());

  return ANSI_C_SYMBOL;
}
