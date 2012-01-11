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
  //ps_irep("typecheck_template_class",declaration);
  // do tempalte arguments
  // this also sets up the template scope
  cpp_scopet& template_scope = typecheck_template_arguments(declaration.template_type());

  typet &type=declaration.type();
  template_typet &template_type = to_template_type(declaration.template_type());

  bool has_body = type.find("body").is_not_nil();

  cpp_namet &cpp_name=(cpp_namet &)type.add("tag");

  std::string identifier, base_name;
  cpp_name.convert(identifier, base_name);

  if(identifier!=base_name)
  {
    err_location(cpp_name.location());
    throw "no namespaces allowed here";
  }

  if(base_name.empty())
  {
    err_location(cpp_name.location());
    throw "template classes must not be anonymous";
  }

  const irep_idt symbol_name=
    template_class_identifier(base_name, template_type);

  // check if the name is already used
  cpp_scopet::id_sett id_set;
  cpp_scopes.current_scope().lookup(base_name, cpp_scopet::TEMPLATE, id_set);
  if(!id_set.empty())
  {
    const symbolt& previous = lookup((*id_set.begin())->identifier);
    if(previous.name != symbol_name || id_set.size() > 1)
    {
      err_location(cpp_name.location());
      str << "template declaration of `" << base_name.c_str() << "'\n";
      str << "does not match previous declaration\n";
      str << "location of previous definition: " << previous.location;
      throw 0;
    }
  }

  // check if we have it

  symbolst::iterator previous_symbol=
    context.symbols.find(symbol_name);

  if(previous_symbol!=context.symbols.end())
  {
    bool previous_has_body =
      previous_symbol->second.type.subtype().find("body").is_not_nil();

    if (has_body && previous_has_body)
    {
      err_location(cpp_name.location());
      str << "error: template struct symbol `" << base_name
          << "' declared previously" << std::endl;
      str << "location of previous definition: "
          << previous_symbol->second.location;
      throw 0;
    }

    if(has_body)
    {
      previous_symbol->second.type.swap(declaration);
    }

    // todo: the new template scope is useless thus, delete it
    assert(cpp_scopes.id_map[symbol_name]->id_class == cpp_idt::TEMPLATE_SCOPE);
    return;
  }

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

Function: cpp_typecheckt::typecheck_template_function

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::typecheck_template_function(
       cpp_declarationt &declaration)
{
  assert(declaration.declarators().size()==1);

  cpp_declaratort& declarator = declaration.declarators()[0];

  const cpp_namet &cpp_name = to_cpp_name(declarator.add("name"));
  std::string identifier, base_name;

  if(cpp_name.is_qualified() ||
     cpp_name.has_template_args())
  {
    // musst be of the form: name1<template_args>::name2;
    if(cpp_name.get_sub().size() != 4 ||
       cpp_name.get_sub()[0].id() != "name" ||
       cpp_name.get_sub()[1].id() != "template_args" ||
       cpp_name.get_sub()[2].id() != "::" ||
       cpp_name.get_sub()[3].id() != "name")
    {

      if(cpp_name.get_sub().size() != 5 ||
       cpp_name.get_sub()[0].id() != "name" ||
       cpp_name.get_sub()[1].id() != "template_args" ||
       cpp_name.get_sub()[2].id() != "::" ||
       cpp_name.get_sub()[3].id() != "operator")
      {
        err_location(cpp_name);
        str << "bad template name";
        throw 0;
      }
    }

    // let's find the template structre this template function belongs to.
    cpp_scopet::id_sett id_set;
    cpp_scopes.current_scope().lookup(cpp_name.get_sub().front().get("identifier"),id_set);

    if(id_set.empty())
    {
      str << cpp_scopes.current_scope();
      err_location(cpp_name);
      str << "identifier`" << cpp_name.get_sub().front().get("identifier") << "' not found";
      throw 0;
    }
    else if(id_set.size() > 1)
    {
      err_location(cpp_name);
      str << "identifier `" << cpp_name.get_sub().front().get("identifier") << "' is ambigious";
      throw 0;
    }
    else if((*(id_set.begin()))->id_class != cpp_idt::TEMPLATE)
    {
      std::cerr << *(*id_set.begin()) << std::endl;
      err_location(cpp_name);
      str << "identifier `" << cpp_name.get_sub().front().get("identifier") << "' is not a template";
      throw 0;
    }

    const cpp_idt& cpp_id  = **(id_set.begin());
    symbolt& template_symbol = context.symbols.find(cpp_id.identifier)->second;


    exprt& template_methods = static_cast<exprt&>(template_symbol.value.add("template_methods"));
    template_methods.copy_to_operands(declaration);

    // save current scope
    cpp_save_scopet cpp_saved_scope(cpp_scopes);

    const irept& instantiated_with = template_symbol.value.add("instantiated_with");
    for(unsigned i = 0; i < instantiated_with.get_sub().size(); i++)
    {
      const irept& tc_template_args = instantiated_with.get_sub()[i];
      cpp_declarationt decl_tmp = declaration;

      // do tempalte arguments
      // this also sets up the template scope of the method
      cpp_scopet& method_scope = typecheck_template_arguments(decl_tmp.template_type());
      cpp_scopes.go_to(method_scope);

      // mapping from template arguments to values/types
      build_template_map(decl_tmp.template_type(), tc_template_args);

      decl_tmp.remove("template_type");
      decl_tmp.remove("is_template");

      convert(decl_tmp);
      cpp_saved_scope.restore();
    }

    return;
  }

  // do template arguments
  // this also sets up the template scope
  cpp_scopet& template_scope = typecheck_template_arguments(declaration.template_type());

  cpp_name.convert(identifier, base_name);
  if(identifier != base_name)
  {
    err_location(declaration);
    str << "namespaces not supported in template declaration";
    throw 0;
  }

  template_typet& template_type = to_template_type(declaration.template_type());

  irep_idt symbol_name = template_function_identifier(base_name,
                                                      template_type,
                                                      declarator.type(),
                                                      declaration.type());

  bool has_value = declarator.find("value").is_not_nil();

  // check if we have it

  symbolst::iterator previous_symbol=
    context.symbols.find(symbol_name);

  if(previous_symbol!=context.symbols.end())
  {
    bool previous_has_value =
     to_cpp_declaration(previous_symbol->second.type).declarators()[0].find("value").is_not_nil();

    if (has_value && previous_has_value)
    {
      err_location(cpp_name.location());
      str << "error: template function symbol `" << base_name
          << "' declared previously" << std::endl;
      str << "location of previous definition: "
          << previous_symbol->second.location;
      throw 0;
    }

    if(has_value)
    {
      previous_symbol->second.type = template_type;
    }

    // todo: the new template scope is useless thus, delete it
    assert(cpp_scopes.id_map[symbol_name]->id_class == cpp_idt::TEMPLATE_SCOPE);
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
  template_scope.id_class = cpp_idt::TEMPLATE_SCOPE;
  cpp_scopes.id_map[symbol_name] = &template_scope;
}


/*******************************************************************\

Function: cpp_typecheckt::template_class_identifier

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string cpp_typecheckt::template_class_identifier(
                                       const irep_idt& base_name,
                                       const template_typet& template_type)
{
  std::string identifier =  cpp_identifier_prefix(current_mode)+"::"+
      cpp_scopes.current_scope().prefix+
      "template."+id2string(base_name) + "<";

  int counter = 0;

  forall_irep(it, template_type.arguments().get_sub())
  {
    std::string arg;
    if(it->id_string() == "type")
      identifier += "Type" + i2string(counter);
    else
      identifier += "Non_Type"+ i2string(counter);

    counter++;

    if(it - template_type.get_sub().end() > 1)
      identifier +=",";
  }
  identifier += ">";
  return identifier;
}



/*******************************************************************\

Function: cpp_typecheckt::template_function_identifier

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string cpp_typecheckt::template_function_identifier(
  const irep_idt &base_name,
  const template_typet &template_type,
  const typet &function_type,
  const typet &return_type)
{
  // we first build something without function arguments
  cpp_template_args_non_tct partial_specialization_args;
  std::string identifier = template_class_identifier(base_name, template_type);

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

  // template specialization
  if((declaration.template_type().find("arguments")).get_sub().size() == 0)
  {
    convert_template_specialization(declaration);
    return;
  }

  typet &type=declaration.type();

  // there are template functions
  // and template classes

  if(declaration.declarators().empty())
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
    typecheck_template_function(declaration);
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
      cpp_name.get_sub()[0].get("identifier").c_str();

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
      cpp_name.get_sub()[0].get("identifier").c_str();

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

Function: cpp_typecheckt::typecheck_template_arguments

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

cpp_scopet& cpp_typecheckt::typecheck_template_arguments(typet &type)
{
  cpp_save_scopet cpp_saved_scope(cpp_scopes);


  assert(type.id()=="template");

  std::string id_suffix = "template::"+i2string(template_counter++);

  // produce a new scope
  cpp_scopet &template_scope=
    cpp_scopes.current_scope().new_scope(cpp_scopes.current_scope().prefix + id_suffix);
  cpp_scopes.go_to(template_scope);

  template_scope.prefix=template_scope.get_parent().prefix + id_suffix;

  template_scope.id_class=cpp_idt::TEMPLATE_SCOPE;


  // put template arguments into this scope

  irept::subt &arguments=type.add("arguments").get_sub();

  unsigned anon_count=0;

  forall_irep(it, arguments)
  {
    exprt &arg=(exprt &)*it;
    cpp_declarationt declaration;
    declaration.swap((cpp_declarationt &)arg);

    cpp_declarator_convertert cpp_declarator_converter(*this);

    // there must be one declarator
    assert(declaration.declarators().size()==1);

    cpp_declaratort &declarator=declaration.declarators().front();

    // there might be a default type
    exprt default_value = static_cast<const exprt&>(declarator.find("value"));

    // it may be anonymous
    if(declarator.name().is_nil())
    {
      irept name("name");
      name.set("identifier", "anon#"+i2string(++anon_count));
      declarator.name()=cpp_namet();
      declarator.name().get_sub().push_back(name);
    }

    cpp_declarator_converter.is_typedef=declaration.get_bool("is_type");
    cpp_declarator_converter.is_template_argument=true;

    const symbolt &symbol=
      cpp_declarator_converter.convert(declaration, declarator);

    if(cpp_declarator_converter.is_typedef)
    {
      arg=exprt("type", typet("symbol"));
      arg.type().set("identifier", symbol.name);

      	arg.type().location()=declaration.find_location();

     }
    else
      arg=symbol_expr(symbol);

    if(default_value.is_not_nil())
	    arg.add("#default") = default_value;

    arg.location()=declaration.find_location();
  }

  // continue without adding to the prefix
  template_scope.prefix=template_scope.get_parent().prefix;

  return template_scope;
}


/*******************************************************************\

Function: cpp_typecheckt::typecheck_template_args

  Inputs:

 Outputs:

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

      template_map.type_map[t.type().get("identifier")] = tmp;
      //template_map.type_map.insert(std::pair<irep_idt, typet>
      //  (t.type().get("identifier"), tmp));
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
        (t.get("identifier"), tmp));
    }

    t_it++;
  }
}

/*******************************************************************\

Function: cpp_typecheckt::template_suffix

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string cpp_typecheckt::template_suffix(
  const irept &template_args)
{
  // quick hack
  std::string result="<";
  bool first=true;

  const irept &instance=template_args.find("arguments");

  forall_irep(it, instance.get_sub())
  {
    if(first) first=false; else result+=",";

    exprt expr=(const exprt &)*it;

    if(expr.id()=="ambiguous")
      expr.id("type");

    if(expr.id()=="type")
    {
      const typet &type=expr.type();
      if(type.id()=="symbol")
        result+=type.get_string("identifier");
      else
      {
//        std::string tmp;
//        irep2name(type, tmp);
//        result+=tmp;
      }
    }
    else // expression
    {
      exprt e(expr);
      simplify_exprt simplify;
      simplify.simplify(e);

      // this must be scalar
      mp_integer i;

      if(to_integer(e, i))
      {
        err_location(*it);
        throw "template argument expression expected to be scalar";
      }

      result+=integer2string(i);
    }
  }

  result+='>';

  return result;
}

/*******************************************************************\

Function: cpp_typecheckt::instantiate_template

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const symbolt &cpp_typecheckt::instantiate_template(
  const locationt &location,
  const irep_idt &identifier,
  const irept &template_args,
  const typet& specialization)
{
  cpp_save_scopet cpp_saved_scope(cpp_scopes);
  cpp_saved_template_mapt saved_map(template_map);

  symbolt &old_symbol=context.symbols.find(identifier)->second;
  bool is_specialization = specialization.is_not_nil();

  // do we have args?
  if(template_args.is_nil())
  {
    err_location(location);
    str << "`" << old_symbol.base_name
        << "' is a template; thus, expected template arguments";
    throw 0;
  }

  // typecheck template arguments
  irept tc_template_args(template_args);
  typecheck_template_args(tc_template_args);

  // produce new symbol name
  std::string suffix=template_suffix(tc_template_args);

  // we need the template scope
  cpp_scopet *template_scope = static_cast<cpp_scopet*>(cpp_scopes.id_map[old_symbol.name]);
  if(template_scope == NULL)
  {
    err_location(location);
    str << "identifier: " << identifier << std::endl;
    throw "instantiation error";
  }
  assert(template_scope!=NULL);

  // new declaration
  cpp_declarationt  new_decl = to_cpp_declaration(old_symbol.type);
  new_decl.remove("is_template");

  // the type of the template
  template_typet template_type;
  template_type = static_cast<const template_typet&>(new_decl.template_type());
  new_decl.remove("template_type");

  // save old scope
  cpp_save_scopet saved_scope(cpp_scopes);

  // mapping from template arguments to values/types
  build_template_map(template_type,
                     tc_template_args);

  // enter correct scope
  cpp_scopes.go_to(*template_scope);

  // is it a template method?
  bool is_template_method = cpp_scopes.current_scope().get_parent().is_class() &&
    new_decl.type().id() != "struct";
  irep_idt class_name;
  if(is_template_method)
    class_name = cpp_scopes.current_scope().get_parent().identifier;


  // sub scope for fixing the prefix
  std::string subscope_name = template_scope->identifier.c_str() + suffix;

  cpp_scopest::id_mapt::iterator scope_it = cpp_scopes.id_map.find(subscope_name);
  if(scope_it != cpp_scopes.id_map.end())
  {
    cpp_scopet& scope = cpp_scopes.get_scope(subscope_name);

    // It has already been instantianted..
    cpp_scopet::id_sett id_set;
    scope.lookup(old_symbol.base_name, id_set);
    assert(id_set.size()==1);
    const cpp_idt& cpp_id = **id_set.begin();
    assert(cpp_id.id_class == cpp_idt::CLASS || cpp_id.id_class == cpp_idt::SYMBOL);

    const symbolt& symb = lookup(cpp_id.identifier);

    // continue if the type is incomplete only
    if(cpp_id.id_class == cpp_idt::CLASS &&
       symb.type.id() == "struct")
    {
      return symb;
    }
    else if(symb.value.is_not_nil())
    {
      return symb;
    }
    cpp_scopes.go_to(scope);
  }
  else
  {
    std::string prefix = template_scope->get_parent().prefix + suffix;
    cpp_scopet &sub_scope =
      cpp_scopes.current_scope().new_scope(subscope_name);
    sub_scope.prefix = prefix;
    cpp_scopes.go_to(sub_scope);
    cpp_scopes.id_map.insert(cpp_scopest::id_mapt::value_type(subscope_name,&sub_scope));
  }


  // store the information that the template has
  // been instantiated using these arguments
  irept& instantiated_with = old_symbol.value.add("instantiated_with");
  irept tc_template_args_tmp = tc_template_args;
  instantiated_with.move_to_sub(tc_template_args_tmp);

  #if 0
  std::cout << "MAP:" << std::endl;
  template_map.print(std::cout);
  #endif

  // fix the type
  {
    typet declaration_type;

    declaration_type = new_decl.type();

    if(declaration_type.id() == "struct" && is_specialization)
    {
      declaration_type = specialization;
      declaration_type.location() = location;
    }
    else if(is_specialization)
    {
      irept tmp = specialization;
      new_decl.declarators()[0].swap(tmp);
    }

    template_map.apply(declaration_type);
    new_decl.type().swap(declaration_type);
  }

  if(new_decl.type().id() == "struct")
  {
    convert(new_decl);
    const symbolt& new_symb = lookup(new_decl.type().get("identifier"));

    // instantiate the template methods
    const exprt& template_methods = static_cast<const exprt&>(old_symbol.value.find("template_methods"));
    for(unsigned i = 0; i < template_methods.operands().size();i++)
    {
      cpp_saved_scope.restore();

      cpp_declarationt method_decl =
        static_cast<const cpp_declarationt&>(
          static_cast<const irept&>(template_methods.operands()[i]));

      // the type of the template method
      template_typet method_type;
      method_type = static_cast<const template_typet&>(method_decl.template_type());

      // do tempalte arguments
      // this also sets up the template scope of the method
      cpp_scopet& method_scope = typecheck_template_arguments(method_type);
      cpp_scopes.go_to(method_scope);

      // mapping from template arguments to values/types
      build_template_map(method_type, tc_template_args);

      method_decl.remove("template_type");
      method_decl.remove("is_template");

      convert(method_decl);
    }
    return new_symb;
  }

  if(is_template_method)
  {
    contextt::symbolst::iterator it = 
      context.symbols.find(class_name);

    assert(it != context.symbols.end());

    symbolt& symb = it->second;



    assert( new_decl.declarators().size() == 1);

    if(new_decl.member_spec().is_virtual())
    {
      err_location(new_decl);
      str <<  "invalid use of 'virtual' in template declaration";
      throw 0;
    }

    if(convert_typedef(new_decl.type()))
    {
      err_location(new_decl);
      str << "template declaration for typedef";
      throw 0;
    }


    if(new_decl.storage_spec().is_extern()
         || new_decl.storage_spec().is_auto()
       || new_decl.storage_spec().is_register()
       || new_decl.storage_spec().is_mutable())
    {
        err_location(new_decl);
        str << "invalid storage class specified for template field";
        throw 0;
    }

    bool is_static=new_decl.storage_spec().is_static();

    irep_idt access = new_decl.get("#access");
    assert(access != "");

    typecheck_compound_declarator(symb,new_decl, new_decl.declarators()[0],
        to_struct_type(symb.type).components(),access,is_static,false,false);

    return lookup(to_struct_type(symb.type).components().back().get("name"));
  }
  convert(new_decl);
  const symbolt& symb = lookup(new_decl.declarators()[0].get("identifier"));
  return symb;
}
