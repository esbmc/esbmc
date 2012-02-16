/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <location.h>
#include <expr_util.h>
#include <arith_tools.h>
#include <i2string.h>
#include <replace_symbol.h>
#include <simplify_expr_class.h>
#include <simplify_expr.h>

#include "cpp_type2name.h"
#include "cpp_typecheck.h"
#include "cpp_declarator_converter.h"
#include "cpp_template_type.h"
#include "cpp_convert_type.h"
#include "cpp_template_args.h"

/*******************************************************************\

Function: cpp_typecheckt::check_template_restrictions

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::check_template_restrictions(
  const irept &cpp_name,
  const irep_idt &final_identifier,
  const typet &final_type)
{
  if(final_type.id()=="template")
  {
    // subtype must be class or function

    if(final_type.subtype().id()!="struct" &&
       final_type.subtype().id()!="code")
    {
      err_location(cpp_name);
      str << "template only allowed with classes or functions,"
             " but got `" << to_string(final_type.subtype()) << "'";
      throw 0;
    }
  }
}

/*******************************************************************\

Function: cpp_typecheckt::typecheck_template_class

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::typecheck_template_class(
  cpp_declarationt &declaration)
{
  // Do template arguments. This also sets up the template scope.
  cpp_scopet &template_scope=
    typecheck_template_parameters(declaration.template_type());

  typet &type=declaration.type();
  template_typet &template_type=declaration.template_type();

  bool has_body=type.find("body").is_not_nil();

  const cpp_namet &cpp_name=
    static_cast<const cpp_namet &>(type.find("tag"));

  std::string identifier, base_name;
  cpp_name.convert(identifier, base_name);

  if(identifier!=base_name)
  {
    err_location(cpp_name.location());
    throw "no namespaces allowed here";
  }

  if(base_name.empty())
  {
    err_location(type.location());
    throw "template classes must not be anonymous";
  }

  const cpp_template_args_non_tct &partial_specialization_args=
    declaration.partial_specialization_args();

  const irep_idt symbol_name=
    template_class_identifier(base_name, template_type);

  #if 0
  // Check if the name is already used by a different template
  // in the same scope.
  {
    cpp_scopet::id_sett id_set;
    cpp_scopes.current_scope().lookup(
      base_name,
      cpp_scopet::SCOPE_ONLY,
      cpp_scopet::TEMPLATE,
      id_set);

    if(!id_set.empty())
    {
      const symbolt &previous=lookup((*id_set.begin())->identifier);
      if(previous.name!=symbol_name || id_set.size()>1)
      {
        err_location(cpp_name.location());
        str << "template declaration of `" << base_name.c_str()
            << " does not match previous declaration\n";
        str << "location of previous definition: " << previous.location;
        throw 0;
      }
    }
  }
  #endif

  // check if we have it already

  contextt::symbolst::iterator previous_symbol=
    context.symbols.find(symbol_name);

  if(previous_symbol!=context.symbols.end())
  {
    // there already

    bool previous_has_body=
      previous_symbol->second.type.type().find("body").is_not_nil();

    if(has_body && previous_has_body)
    {
      err_location(cpp_name.location());
      str << "template struct `" << base_name
          << "' defined previously" << std::endl;
      str << "location of previous definition: "
          << previous_symbol->second.location;
      throw 0;
    }

    if(has_body)
    {
      // we replace the template!
      previous_symbol->second.type.swap(declaration);

      // we also replace the template scope (the old one could be deleted)
      cpp_scopes.id_map[symbol_name]=&template_scope;
    }

    assert(cpp_scopes.id_map[symbol_name]->id_class == cpp_idt::TEMPLATE_SCOPE);
    return;
  }

  // it's not there yet

  symbolt symbol;

  symbol.name=symbol_name;
  symbol.base_name=base_name;
  symbol.location=cpp_name.location();
  symbol.mode=current_mode;
  symbol.module=module;
  symbol.type.swap(declaration);
  symbol.is_macro=false;
  symbol.value=exprt("template_decls");

  symbol.pretty_name=
    cpp_scopes.current_scope().prefix+id2string(symbol.base_name);

  symbolt *new_symbol;
  if(context.move(symbol, new_symbol))
    throw "cpp_typecheckt::typecheck_compound_type: context.move() failed";

  // put into current scope
  cpp_idt &id=cpp_scopes.put_into_scope(*new_symbol);
  id.id_class=cpp_idt::TEMPLATE;
  id.prefix=cpp_scopes.current_scope().prefix+
            id2string(new_symbol->base_name);

  // link the template symbol with the template scope
  cpp_scopes.id_map[symbol_name]=&template_scope;
  assert(cpp_scopes.id_map[symbol_name]->id_class==cpp_idt::TEMPLATE_SCOPE);
}

/*******************************************************************\

Function: cpp_typecheckt::typecheck_function_template

  Inputs:

 Outputs:

 Purpose: typecheck function templates

\*******************************************************************/

void cpp_typecheckt::typecheck_function_template(
  cpp_declarationt &declaration)
{
  assert(declaration.declarators().size()==1);

  cpp_declaratort &declarator=declaration.declarators()[0];
  const cpp_namet &cpp_name=to_cpp_name(declarator.add("name"));

  if(cpp_name.is_qualified() ||
     cpp_name.has_template_args())
    return typecheck_template_member_function(declaration);

  // do template arguments
  // this also sets up the template scope
  cpp_scopet &template_scope=
    typecheck_template_parameters(declaration.template_type());

  std::string identifier, base_name;
  cpp_name.convert(identifier, base_name);
  if(identifier!=base_name)
  {
    err_location(declaration);
    str << "namespaces not supported in template declaration";
    throw 0;
  }

  template_typet &template_type=declaration.template_type();

  typet function_type=
    declarator.merge_type(declaration.type());

  cpp_convert_plain_type(function_type);

  irep_idt symbol_name=
    function_template_identifier(
      base_name,
      template_type,
      function_type);

  bool has_value=declarator.find("value").is_not_nil();

  // check if we have it already

  contextt::symbolst::iterator previous_symbol=
    context.symbols.find(symbol_name);

  if(previous_symbol!=context.symbols.end())
  {
    bool previous_has_value =
     to_cpp_declaration(previous_symbol->second.type).
       declarators()[0].find("value").is_not_nil();

    if(has_value && previous_has_value)
    {
      err_location(cpp_name.location());
      str << "function template symbol `" << base_name
          << "' declared previously" << std::endl;
      str << "location of previous definition: "
          << previous_symbol->second.location;
      throw 0;
    }

    if(has_value)
    {
      previous_symbol->second.type.swap(declaration);
      cpp_scopes.id_map[symbol_name]=&template_scope;
    }

    // todo: the old template scope now is useless,
    // and thus, we could delete it
    return;
  }

  symbolt symbol;
  symbol.name=symbol_name;
  symbol.base_name=base_name;
  symbol.location=cpp_name.location();
  symbol.mode=current_mode;
  symbol.module=module;
  symbol.value.make_nil();

  symbol.type.swap(declaration);
  symbol.pretty_name=
    cpp_scopes.current_scope().prefix+id2string(symbol.base_name);

  symbolt *new_symbol;
  if(context.move(symbol, new_symbol))
    throw "cpp_typecheckt::typecheck_compound_type: context.move() failed";

  // put into scope
  cpp_idt &id=cpp_scopes.put_into_scope(*new_symbol);
  id.id_class=cpp_idt::TEMPLATE;
  id.prefix=cpp_scopes.current_scope().prefix+
            id2string(new_symbol->base_name);

  // link the template symbol with the template scope
  assert(template_scope.id_class==cpp_idt::TEMPLATE_SCOPE);
  cpp_scopes.id_map[symbol_name] = &template_scope;
}

/*******************************************************************\

Function: cpp_typecheckt::typecheck_template_member_function

  Inputs:

 Outputs:

 Purpose: typecheck function member templates

\*******************************************************************/

void cpp_typecheckt::typecheck_template_member_function(
  cpp_declarationt &declaration)
{
  assert(declaration.declarators().size()==1);

  cpp_declaratort &declarator=declaration.declarators()[0];
  const cpp_namet &cpp_name=to_cpp_name(declarator.add("name"));

  assert(cpp_name.is_qualified() ||
         cpp_name.has_template_args());

  // must be of the form: name1<template_args>::name2
  // or:                  name1<template_args>::operator X
  if(cpp_name.get_sub().size()==4 &&
     cpp_name.get_sub()[0].id()=="name" &&
     cpp_name.get_sub()[1].id()=="template_args" &&
     cpp_name.get_sub()[2].id()=="::" &&
     cpp_name.get_sub()[3].id()=="name")
  {
  }
  else if(cpp_name.get_sub().size()==5 &&
          cpp_name.get_sub()[0].id()=="name" &&
          cpp_name.get_sub()[1].id()=="template_args" &&
          cpp_name.get_sub()[2].id()=="::" &&
          cpp_name.get_sub()[3].id()=="operator")
  {
  }
  else
  {
    err_location(cpp_name);
    str << "bad template name";
    throw 0;
  }

  // let's find the template class this function template belongs to.
  cpp_scopet::id_sett id_set;

  cpp_scopes.current_scope().lookup(
    cpp_name.get_sub().front().identifier(),
    id_set);

  if(id_set.empty())
  {
    str << cpp_scopes.current_scope();
    err_location(cpp_name);
    str << "template function/member identifier `"
        << cpp_name.get_sub().front().identifier()
        << "' not found";
    throw 0;
  }
  else if(id_set.size()>1)
  {
    err_location(cpp_name);
    str << "template function/member identifier `"
        << cpp_name.get_sub().front().identifier()
        << "' is ambiguous";
    throw 0;
  }
  else if((*(id_set.begin()))->id_class!=cpp_idt::TEMPLATE)
  {
    std::cerr << *(*id_set.begin()) << std::endl;
    err_location(cpp_name);
    str << "template function/member identifier `"
        << cpp_name.get_sub().front().identifier()
        << "' is not a template";
    throw 0;
  }

  const cpp_idt &cpp_id=**(id_set.begin());
  symbolt &template_symbol=
    context.symbols.find(cpp_id.identifier)->second;

  exprt &template_methods=static_cast<exprt &>(
    template_symbol.value.add("template_methods"));

  template_methods.copy_to_operands(declaration);

  // save current scope
  cpp_save_scopet cpp_saved_scope(cpp_scopes);

  const irept &instantiated_with =
    template_symbol.value.add("instantiated_with");

  for(unsigned i=0; i<instantiated_with.get_sub().size(); i++)
  {
    const cpp_template_args_tct &tc_template_args=
      static_cast<const cpp_template_args_tct &>(instantiated_with.get_sub()[i]);

    cpp_declarationt decl_tmp=declaration;

    // do template arguments
    // this also sets up the template scope of the method
    cpp_scopet &method_scope=
      typecheck_template_parameters(decl_tmp.template_type());

    cpp_scopes.go_to(method_scope);

    // mapping from template arguments to values/types
    template_map.build(decl_tmp.template_type(), tc_template_args);

    decl_tmp.remove("template_type");
    decl_tmp.remove("is_template");

    convert(decl_tmp);
    cpp_saved_scope.restore();
  }
}

/*******************************************************************\

Function: cpp_typecheckt::template_class_identifier

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string cpp_typecheckt::template_class_identifier(
  const irep_idt &base_name,
  const template_typet &template_type)
{
  std::string identifier=
    cpp_identifier_prefix(current_mode)+"::"+
      cpp_scopes.current_scope().prefix+
      "template."+id2string(base_name) + "<";

  int counter=0;

  // these are probably not needed -- templates
  // should be unique in a namespace
  for(template_typet::parameterst::const_iterator
      it=template_type.parameters().begin();
      it!=template_type.parameters().end();
      it++)
  {
    if(counter!=0) identifier+=",";

    if(it->id()=="type")
      identifier+="Type"+i2string(counter);
    else
      identifier+="Non_Type"+i2string(counter);

    counter++;
  }

  identifier += ">";
  return identifier;
}



/*******************************************************************\

Function: cpp_typecheckt::function_template_identifier

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string cpp_typecheckt::function_template_identifier(
  const irep_idt &base_name,
  const template_typet &template_type,
  const typet &function_type)
{
  // we first build something without function arguments
  cpp_template_args_non_tct partial_specialization_args;
  std::string identifier=
    template_class_identifier(base_name, template_type);

  // we must also add the signature of the function to the identifier
  identifier+=cpp_type2name(function_type);

  return identifier;
}

/*******************************************************************\

Function: cpp_typecheckt::convert_template_declaration

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::convert_template_declaration(
  cpp_declarationt &declaration)
{
  if(declaration.member_spec().is_virtual())
  {
    err_location(declaration);
    str <<  "invalid use of 'virtual' in template declaration";
    throw 0;
  }

  if(convert_typedef(declaration.type()))
  {
    err_location(declaration);
    str << "template declaration for typedef";
    throw 0;
  }

  typet &type=declaration.type();

  // there are 1) function templates and 2) template classes

  if(declaration.is_template_class())
  {
    if(type.id()!="struct")
    {
      err_location(declaration);
      throw "expected template class";
    }

    typecheck_template_class(declaration);
    return;
  }
  else // template function, maybe member function
  {
    typecheck_function_template(declaration);
    return;
  }
}


/*******************************************************************\

Function: cpp_typecheckt::convert_template_specialization

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const symbolt& cpp_typecheckt::convert_template_specialization(
  cpp_declarationt &declaration)
{
  cpp_save_scopet saved_scope(cpp_scopes);

  typet& type = declaration.type();

  if(type.id()=="struct")
  {
    cpp_namet& cpp_name = static_cast<cpp_namet&> (type.add("tag"));

    if (cpp_name.is_qualified())
    {
      err_location(cpp_name.location());
      str << "qualifiers not excpected here";
      throw 0;
    }

    if (cpp_name.get_sub().size() != 2
      || cpp_name.get_sub()[0].id() != "name"
      || cpp_name.get_sub()[1].id() != "template_args")
    {
      err_location(cpp_name.location());
      str << "bad template-sepcialization name"; // currently we are more restrictive
                                                 // than the standard
      throw 0;
    }

    std::string base_name =
      cpp_name.get_sub()[0].identifier().c_str();

    cpp_scopest::id_sett id_set;
    cpp_scopes.get_ids(base_name, cpp_idt::TEMPLATE, id_set ,true);

    if(id_set.empty())
    {
        err_location(type.location());
        str << "template `" << base_name << "' not found\n";
        throw 0;
    }
    else if(id_set.size() > 1)
    {
        while(!id_set.empty())
        {
          std::cerr<< (*id_set.begin())->identifier << std::endl;
          id_set.erase(id_set.begin());
        }

        //str << cpp_scopes.current_scope();
        err_location(type);
        str << "template `" << base_name << "' is ambiguous\n";
        throw 0;
    }

    irept template_args = cpp_name.get_sub()[1];
    cpp_name.get_sub().pop_back();

    return instantiate_template(cpp_name.location(),
      (*id_set.begin())->identifier,
      template_args,
      type);
  }
  else
  {
    assert(declaration.declarators().size() == 1);
    cpp_declaratort declarator = declaration.declarators().front();
    cpp_namet& cpp_name = declarator.name();

    if (cpp_name.is_qualified())
    {
      err_location(cpp_name.location());
      str << "qualifiers not excpected here";
      throw 0;
    }

    if (cpp_name.get_sub().size() != 2
      || cpp_name.get_sub()[0].id() != "name"
      || cpp_name.get_sub()[1].id() != "template_args")
    {
      err_location(cpp_name.location());
      str << "bad template-sepcialization name"; // currently we are more restrictive
                                                 // than the standard
      throw 0;
    }

    std::string base_name =
      cpp_name.get_sub()[0].identifier().c_str();

    cpp_scopest::id_sett id_set;
    cpp_scopes.get_ids(base_name, id_set, true);

    if(id_set.empty())
    {
        err_location(cpp_name.location());
        str << "template `" << base_name << "' not found\n";
        throw 0;
    }
    else if(id_set.size() > 1)
    {
        err_location(cpp_name.location());
        str << "template `" << base_name << "' is ambiguous\n";
    }

    irept template_args = cpp_name.get_sub()[1];
    cpp_name.get_sub().pop_back();

    typet specialization;
    specialization.swap(declarator);


    return instantiate_template(cpp_name.location(),
      (*id_set.begin())->identifier,
      template_args,
      specialization);
  }
}


/*******************************************************************\

Function: cpp_typecheckt::typecheck_template_parameters

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

cpp_scopet &cpp_typecheckt::typecheck_template_parameters(
  template_typet &type)
{
  cpp_save_scopet cpp_saved_scope(cpp_scopes);

  assert(type.id()=="template");

  std::string id_suffix="template::"+i2string(template_counter++);

  // produce a new scope for the template parameters
  cpp_scopet &template_scope=
    cpp_scopes.current_scope().new_scope(
      cpp_scopes.current_scope().prefix+id_suffix);

  template_scope.prefix=template_scope.get_parent().prefix+id_suffix;
  template_scope.id_class=cpp_idt::TEMPLATE_SCOPE;

  cpp_scopes.go_to(template_scope);

  // put template parameters into this scope
  template_typet::parameterst &parameters=type.parameters();

  unsigned anon_count=0;

  for(template_typet::parameterst::iterator
      it=parameters.begin();
      it!=parameters.end();
      it++)
  {
    exprt &parameter=*it;

    cpp_declarationt declaration;
    declaration.swap(static_cast<cpp_declarationt &>(parameter));

    cpp_declarator_convertert cpp_declarator_converter(*this);

    // there must be _one_ declarator
    assert(declaration.declarators().size()==1);

    cpp_declaratort &declarator=declaration.declarators().front();

    // it may be anonymous
    if(declarator.name().is_nil())
    {
      irept name("name");
      name.identifier("anon#"+i2string(++anon_count));
      declarator.name()=cpp_namet();
      declarator.name().get_sub().push_back(name);
    }

    cpp_declarator_converter.is_typedef=declaration.get_bool("is_type");
    cpp_declarator_converter.is_template_argument=true;

    // There might be a default type or value.
    // We store it for later, as it can't be typechecked now
    // because of dependencies on earlier parameters!
    exprt default_value=declarator.value();
    declarator.value().make_nil();

    const symbolt &symbol=
      cpp_declarator_converter.convert(declaration, declarator);

    if(cpp_declarator_converter.is_typedef)
    {
      parameter=exprt("type", typet("symbol"));
      parameter.type().identifier(symbol.name);
      parameter.type().location()=declaration.find_location();
    }
    else
      parameter=symbol_expr(symbol);

    // set (non-typechecked) default value
    if(default_value.is_not_nil())
      parameter.add("#default")=default_value;

    parameter.location()=declaration.find_location();
  }

  // continue without adding to the prefix
  template_scope.prefix=template_scope.get_parent().prefix;

  return template_scope;
}

/*******************************************************************\

Function: cpp_typecheckt::typecheck_template_args

  Inputs: location, non-typechecked template arguments

 Outputs: typechecked template arguments

 Purpose:

\*******************************************************************/

void cpp_typecheckt::typecheck_template_args(
  irept &template_args)
{

  if(template_args.id()=="already_typechecked")
  {
      exprt args =
        static_cast<exprt&>(template_args);
      assert(args.operands().size()==1);
      template_args.swap(args.op0());
      return;
  }

  irept& args = template_args.add("arguments");

  Forall_irep(args_it, args.get_sub())
  {

    exprt &t=(exprt &)*args_it;

    if(t.id()=="type")
      typecheck_type(t.type());
    else if(t.id() == "ambiguous")
    {
      // it can be either a template argument or a type
      exprt res=
        resolve(
          to_cpp_name(t.type()),
          cpp_typecheck_resolvet::BOTH,
          cpp_typecheck_fargst());

       t.swap(res);
    }
    else
    {
      exprt tmp(t);
      typecheck_expr(tmp);
      simplify(tmp);
      t.swap(tmp);


    }
  }
}


/*******************************************************************\

Function: cpp_typecheckt::build_template_map

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::build_template_map(
  const typet &type,
  const irept &template_args)
{
  const irept &type_arguments=type.find("arguments");
  irept instance=template_args.find("arguments");

  irept::subt::const_iterator t_it=
    type_arguments.get_sub().begin();

  #if 0
  std::cout << "A: " << type_arguments.pretty() << std::endl;
  std::cout << "I: " << instance.pretty() << std::endl;
  #endif

   if(instance.get_sub().size() < type_arguments.get_sub().size())
   {
	   // check for default parameters
	   for(unsigned i = instance.get_sub().size();
		i < type_arguments.get_sub().size();
		i++)
	   {
		   const exprt& arg =
			   static_cast<const exprt&>(type_arguments.get_sub()[i]);
		   exprt value = static_cast<const exprt&>(arg.find("#default"));
		   if(value.is_not_nil())
			   instance.get_sub().push_back(value);
		   else break;
	   }
   }


  if(instance.get_sub().size()!=type_arguments.get_sub().size())
  {

    //ps_irep("type",type);
    //ps_irep("template_args",template_args);
    err_location(template_args);
    str << "wrong number of template arguments "
        << "(expected " << type_arguments.get_sub().size()
        << ", but got " << instance.get_sub().size()
        << ").\n"
        << "Expected: " << type_arguments.pretty() << std::endl
        << "But got: "  << instance.pretty() << std::endl;
    throw 0;
  }

  forall_irep(i_it, instance.get_sub())
  {
    assert(t_it!=type_arguments.get_sub().end());

    const exprt &t=(exprt &)*t_it;
    const exprt &i=(exprt &)*i_it;

    if(t.id()=="type")
    {
      if(i.id()!="type")
      {
        err_location(i);
        str << "expected type, but got expression";
        throw 0;
      }

      typet tmp(i.type());

      template_map.type_map[t.type().identifier()] = tmp;
      //template_map.type_map.insert(std::pair<irep_idt, typet>
      //  (t.type().identifier(), tmp));
    }
    else
    {
      if(i.id()=="type")
      {
        err_location(*i_it);
        str << "expected expression, but got type";
        throw 0;
      }

      exprt tmp(i);
      implicit_typecast(tmp, t.type());

      template_map.expr_map.insert(std::pair<irep_idt, exprt>
        (t.identifier(), tmp));
    }

    t_it++;
  }
}
