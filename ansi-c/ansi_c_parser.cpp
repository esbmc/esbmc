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

/*******************************************************************\

Function: ansi_c_parsert::convert_declarator

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void
insert_base_type(typet &dest, const typet &base_type)
{
  typet *p = &dest;

  while(true)
  {
    typet &t=*p;

    if(t.is_nil() || t.id() == "")
    {
      t=base_type;
      break;
    }
    else if(t.id()=="merged_type")
    {
      assert(!t.subtypes().empty());
      // Is this the final point in this chain of types? It could be either a
      // further {pointer,array,incomplete_array} or some qualifier. If the
      // former, descend further; if not, insert type here.
      p=&(t.subtypes().back());
      if (p->id() != "pointer" && p->id() != "merged_type" &&
          !p->is_array() && p->id() !=  "incomplete_array") {
        t.subtypes().push_back(typet());
        p=&(t.subtypes().back());
        p->make_nil();
      }
    }
    else
      p=&t.subtype();
  }

  return;
}

void ansi_c_parsert::convert_declarator(
  irept &declarator,
  const typet &type,
  irept &identifier)
{
  typet *p=(typet *)&declarator;

  // In aid of making ireps type safe, declarations with identifiers come in the
  // form of ireps named {declarator,code,array,incomplete_array} with
  // identifier subtypes.

  if (declarator.is_decl_ident_set() && declarator.id() != "symbol") {
    identifier = declarator.decl_ident();
    declarator.remove("decl_ident");

    if (declarator.id() == "merged_type")
      insert_base_type((typet&)declarator, type);
    else
      insert_base_type((typet&)((typet&)declarator).subtype(), type);

    // Plain variables type is the "declarator" subtype. For code/arrays etc,
    // the fact that it's "code" or an array makes a difference.
    if (declarator.id() == "declarator")
      declarator = (exprt&)((typet&)declarator).subtype();
    // else: leave it as it was.
    return;
  }
  
  // walk down subtype until we hit nil or symbol
  while(true)
  {
    typet &t=*p;

    if(t.id()=="symbol")
    {
      identifier=t;
      t=type;
      break;
    }
    else if(t.id()=="")
    {
      std::cout << "D: " << declarator.pretty() << std::endl;
      assert(0);
    }
    else if(t.is_nil())
    {
      identifier.make_nil();
      t=type;
      break;
    }
    else if(t.id()=="merged_type")
    {
      assert(!t.subtypes().empty());
      p=&(t.subtypes().back());
    }
    else
      p=&t.subtype();
  }
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
  
  std::string base_name=identifier.cmt_base_name().as_string();
  
  bool is_global=current_scope().prefix=="";

  ansi_c_id_classt id_class=get_class(final_type);
  
  const std::string scope_name=
    is_tag?"tag-"+base_name:base_name;
    
  if(is_tag)
    final_type.tag(base_name);

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
  declaration.decl_value().make_nil();
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
