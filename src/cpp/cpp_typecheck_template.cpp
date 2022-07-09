/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_convert_type.h>
#include <cpp/cpp_declarator_converter.h>
#include <cpp/cpp_template_args.h>
#include <cpp/cpp_template_type.h>
#include <cpp/cpp_type2name.h>
#include <cpp/cpp_typecheck.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/simplify_expr.h>
#include <util/simplify_expr_class.h>

void cpp_typecheckt::salvage_default_parameters(
  const template_typet &old_type,
  template_typet &new_type)
{
  const template_typet::parameterst &old_parameters = old_type.parameters();
  template_typet::parameterst &new_parameters = new_type.parameters();

  for(unsigned i = 0; i < new_parameters.size(); i++)
  {
    if(
      i < old_parameters.size() && old_parameters[i].has_default_parameter() &&
      !new_parameters[i].has_default_parameter())
    {
      // TODO! The default may depend on previous parameters!!
      new_parameters[i].default_parameter() =
        old_parameters[i].default_parameter();
    }
  }
}

void cpp_typecheckt::typecheck_class_template(cpp_declarationt &declaration)
{
  // Do template parameters. This also sets up the template scope.
  cpp_scopet &template_scope =
    typecheck_template_parameters(declaration.template_type());

  typet &type = declaration.type();
  template_typet &template_type = declaration.template_type();

  bool has_body = type.body().is_not_nil();

  const cpp_namet &cpp_name = static_cast<const cpp_namet &>(type.find("tag"));

  std::string identifier, base_name;
  cpp_name.convert(identifier, base_name);

  if(identifier != base_name)
  {
    err_location(cpp_name.location());
    throw "no namespaces allowed here";
  }

  if(base_name.empty())
  {
    err_location(type.location());
    throw "template classes must not be anonymous";
  }

  const cpp_template_args_non_tct &partial_specialization_args =
    declaration.partial_specialization_args();

  const irep_idt symbol_name = class_template_identifier(
    base_name, template_type, partial_specialization_args);

  // Check if the name is already used by a different template
  // in the same scope.
  {
    cpp_scopet::id_sett id_set;
    cpp_scopes.current_scope().lookup(base_name, cpp_scopet::TEMPLATE, id_set);

    if(!id_set.empty())
    {
      // It is ok to be share the name if it's an specialization
      if(declaration.get_specialization_of() == "")
      {
        const symbolt &previous = *lookup((*id_set.begin())->identifier);
        if(previous.id != symbol_name || id_set.size() > 1)
        {
          err_location(cpp_name.location());
          str << "template declaration of `" << base_name.c_str()
              << " does not match previous declaration\n";
          str << "location of previous definition: " << previous.location;
          throw 0;
        }
      }
    }
  }

  // check if we have it already

  symbolt *previous_symbol = context.find_symbol(symbol_name);

  if(previous_symbol != nullptr)
  {
    // there already
    cpp_declarationt &previous_declaration =
      to_cpp_declaration(previous_symbol->type);

    bool previous_has_body = previous_declaration.type().body().is_not_nil();

    // check if we have 2 bodies
    if(has_body && previous_has_body)
    {
      err_location(cpp_name.location());
      str << "template struct `" << base_name << "' defined previously"
          << std::endl;
      str << "location of previous definition: " << previous_symbol->location;
      throw 0;
    }

    if(has_body)
    {
      // We replace the template!
      // We have to retain any default parameters from the previous declaration.
      salvage_default_parameters(
        previous_declaration.template_type(), declaration.template_type());
      previous_symbol->type.swap(declaration);

      // We also replace the template scope (the old one could be deleted).
      cpp_scopes.id_map[symbol_name] = &template_scope;

      // We also fix the parent scope in order to see the new
      // template arguments
    }
    else
    {
      // just update any default parameters
      salvage_default_parameters(
        declaration.template_type(), previous_declaration.template_type());
    }

    assert(cpp_scopes.id_map[symbol_name]->id_class == cpp_idt::TEMPLATE_SCOPE);
    return;
  }

  // it's not there yet

  symbolt symbol;

  symbol.id = symbol_name;
  symbol.name = base_name;
  symbol.location = cpp_name.location();
  symbol.mode = current_mode;
  symbol.module = module;
  symbol.type.swap(declaration);
  symbol.is_macro = false;
  symbol.value = exprt("template_decls");

  symbolt *new_symbol;
  if(context.move(symbol, new_symbol))
    throw "cpp_typecheckt::typecheck_compound_type: context.move() failed";

  // put into current scope
  cpp_idt &id = cpp_scopes.put_into_scope(*new_symbol);
  id.id_class = cpp_idt::TEMPLATE;
  id.prefix = cpp_scopes.current_scope().prefix + id2string(new_symbol->name);

  // link the template symbol with the template scope
  cpp_scopes.id_map[symbol_name] = &template_scope;
  assert(cpp_scopes.id_map[symbol_name]->id_class == cpp_idt::TEMPLATE_SCOPE);
}

void cpp_typecheckt::typecheck_function_template(cpp_declarationt &declaration)
{
  assert(declaration.declarators().size() == 1);

  cpp_declaratort &declarator = declaration.declarators()[0];
  const cpp_namet &cpp_name = to_cpp_name(declarator.add("name"));

  // do template arguments
  // this also sets up the template scope
  cpp_scopet &template_scope =
    typecheck_template_parameters(declaration.template_type());

  // Record that this template is 'using' the scope of the parent class. This
  // prevents the template identifier hiding other methods / types.
  template_scope.using_set.insert(&template_scope.get_parent(0));

  std::string identifier, base_name;
  cpp_name.convert(identifier, base_name);
  if(identifier != base_name)
  {
    err_location(declaration);
    str << "namespaces not supported in template declaration";
    throw 0;
  }

  template_typet &template_type = declaration.template_type();

  typet function_type = declarator.merge_type(declaration.type());

  cpp_convert_plain_type(function_type);

  irep_idt symbol_name =
    function_template_identifier(base_name, template_type, function_type);

  bool has_value = declarator.find("value").is_not_nil();

  // check if we have it already

  symbolt *previous_symbol = context.find_symbol(symbol_name);

  if(previous_symbol != nullptr)
  {
    bool previous_has_value = to_cpp_declaration(previous_symbol->type)
                                .declarators()[0]
                                .find("value")
                                .is_not_nil();

    if(has_value && previous_has_value)
    {
      err_location(cpp_name.location());
      str << "function template symbol `" << base_name
          << "' declared previously" << std::endl;
      str << "location of previous definition: " << previous_symbol->location;
      throw 0;
    }

    if(has_value)
    {
      previous_symbol->type.swap(declaration);
      cpp_scopes.id_map[symbol_name] = &template_scope;
    }

    // todo: the old template scope now is useless,
    // and thus, we could delete it
    return;
  }

  symbolt symbol;
  symbol.id = symbol_name;
  symbol.name = base_name;
  symbol.location = cpp_name.location();
  symbol.mode = current_mode;
  symbol.module = module;
  symbol.value.make_nil();

  symbol.type.swap(declaration);

  symbolt *new_symbol;
  if(context.move(symbol, new_symbol))
    throw "cpp_typecheckt::typecheck_compound_type: context.move() failed";

  // put into scope
  cpp_idt &id = cpp_scopes.put_into_scope(*new_symbol);
  id.id_class = cpp_idt::TEMPLATE;
  id.prefix = cpp_scopes.current_scope().prefix + id2string(new_symbol->name);

  // link the template symbol with the template scope
  assert(template_scope.id_class == cpp_idt::TEMPLATE_SCOPE);
  cpp_scopes.id_map[symbol_name] = &template_scope;
}

void cpp_typecheckt::typecheck_class_template_member(
  cpp_declarationt &declaration)
{
  assert(declaration.declarators().size() == 1);

  cpp_declaratort &declarator = declaration.declarators()[0];
  const cpp_namet &cpp_name = to_cpp_name(declarator.add("name"));

  assert(cpp_name.is_qualified() || cpp_name.has_template_args());

  // must be of the form: name1<template_args>::name2
  // or:                  name1<template_args>::operator X
  if(
    cpp_name.get_sub().size() == 4 && cpp_name.get_sub()[0].id() == "name" &&
    cpp_name.get_sub()[1].id() == "template_args" &&
    cpp_name.get_sub()[2].id() == "::" && cpp_name.get_sub()[3].id() == "name")
  {
  }
  else if(
    cpp_name.get_sub().size() == 5 && cpp_name.get_sub()[0].id() == "name" &&
    cpp_name.get_sub()[1].id() == "template_args" &&
    cpp_name.get_sub()[2].id() == "::" &&
    cpp_name.get_sub()[3].id() == "operator")
  {
  }
  else if(declaration.is_destructor())
  {
  }
  else
  {
    return; // TODO
    err_location(cpp_name);
    str << "bad template name";
    throw 0;
  }

  // let's find the class template this function template belongs to.
  cpp_scopet::id_sett id_set;

  cpp_scopes.current_scope().lookup(
    cpp_name.get_sub().front().identifier(), id_set);

  if(id_set.empty())
  {
    str << cpp_scopes.current_scope();
    err_location(cpp_name);
    str << "class template `" << cpp_name.get_sub().front().identifier()
        << "' not found";
    throw 0;
  }
  if(id_set.size() > 1)
  {
    err_location(cpp_name);
    str << "class template `" << cpp_name.get_sub().front().identifier()
        << "' is ambiguous";
    throw 0;
  }
  else if((*(id_set.begin()))->id_class != cpp_idt::TEMPLATE)
  {
    std::cerr << *(*id_set.begin()) << std::endl;
    err_location(cpp_name);
    str << "class template `" << cpp_name.get_sub().front().identifier()
        << "' is not a template";
    throw 0;
  }

  const cpp_idt &cpp_id = **(id_set.begin());
  symbolt &template_symbol = *context.find_symbol(cpp_id.identifier);

  exprt *template_methods =
    &static_cast<exprt &>(template_symbol.value.add("template_methods"));
  template_methods->copy_to_operands(declaration);

  // save current scope
  cpp_save_scopet cpp_saved_scope(cpp_scopes);

  const irept &instantiated_with =
    template_symbol.value.add("instantiated_with");

  for(const auto &i : instantiated_with.get_sub())
  {
    const cpp_template_args_tct &tc_template_args =
      static_cast<const cpp_template_args_tct &>(i);

    cpp_declarationt decl_tmp = declaration;

    // do template arguments
    // this also sets up the template scope of the method
    cpp_scopet &method_scope =
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

std::string cpp_typecheckt::class_template_identifier(
  const irep_idt &base_name,
  const template_typet &template_type,
  const cpp_template_args_non_tct &partial_specialization_args)
{
  std::string identifier = cpp_scopes.current_scope().prefix + "template." +
                           id2string(base_name) + "<";

  int counter = 0;

  // these are probably not needed -- templates
  // should be unique in a namespace
  for(const auto &it : template_type.parameters())
  {
    if(counter != 0)
      identifier += ",";

    if(it.id() == "type")
      identifier += "Type" + i2string(counter);
    else
      identifier += "Non_Type" + i2string(counter);

    counter++;
  }

  identifier += ">";

  if(!partial_specialization_args.arguments().empty())
  {
    identifier += "_specialized_to_<";

    counter = 0;
    for(cpp_template_args_non_tct::argumentst::const_iterator it =
          partial_specialization_args.arguments().begin();
        it != partial_specialization_args.arguments().end();
        it++, counter++)
    {
      if(counter != 0)
        identifier += ",";

      if(it->id() == "type" || it->id() == "ambiguous")
        identifier += cpp_type2name(it->type());
      else
        identifier += cpp_expr2name(*it);
    }

    identifier += ">";
  }

  return identifier;
}

std::string cpp_typecheckt::function_template_identifier(
  const irep_idt &base_name,
  const template_typet &template_type,
  const typet &function_type)
{
  // we first build something without function arguments
  cpp_template_args_non_tct partial_specialization_args;
  std::string identifier = class_template_identifier(
    base_name, template_type, partial_specialization_args);

  // we must also add the signature of the function to the identifier
  identifier += cpp_type2name(function_type);

  return identifier;
}

void cpp_typecheckt::convert_class_template_specialization(
  cpp_declarationt &declaration)
{
  cpp_save_scopet saved_scope(cpp_scopes);

  typet &type = declaration.type();

  assert(type.id() == "struct");

  cpp_namet &cpp_name = static_cast<cpp_namet &>(type.add("tag"));

  if(cpp_name.is_qualified())
  {
    err_location(cpp_name.location());
    str << "qualifiers not excpected here";
    throw 0;
  }

  if(
    cpp_name.get_sub().size() != 2 || cpp_name.get_sub()[0].id() != "name" ||
    cpp_name.get_sub()[1].id() != "template_args")
  {
    // currently we are more restrictive
    // than the standard
    err_location(cpp_name.location());
    str << "bad template-class-sepcialization name";
    throw 0;
  }

  irep_idt base_name = cpp_name.get_sub()[0].get("identifier");

  // copy the template arguments
  const cpp_template_args_non_tct template_args_non_tc =
    to_cpp_template_args_non_tc(cpp_name.get_sub()[1]);

  // Remove the template arguments from the name.
  cpp_name.get_sub().pop_back();

  // get the template symbol

  cpp_scopest::id_sett id_set;
  cpp_scopes.get_ids(base_name, cpp_idt::TEMPLATE, id_set, true);

  // remove any specializations
  for(cpp_scopest::id_sett::iterator it = id_set.begin();
      it != id_set.end();) // no it++
  {
    cpp_scopest::id_sett::iterator next = it;
    next++;

    if(lookup((*it)->identifier)->type.find("specialization_of").is_not_nil())
      id_set.erase(it);

    it = next;
  }

  // only one should be left
  if(id_set.empty())
  {
    err_location(type.location());
    str << "class template `" << base_name << "' not found";
    throw 0;
  }
  if(id_set.size() > 1)
  {
    err_location(type);
    str << "class template `" << base_name << "' is ambiguous";
    throw 0;
  }

  symbolt *s = context.find_symbol((*id_set.begin())->identifier);
  assert(s != nullptr);

  symbolt &template_symbol = *s;

  if(!template_symbol.type.get_bool("is_template"))
  {
    err_location(type);
    str << "expected a template";
  }

  // partial -- we typecheck
  declaration.partial_specialization_args() = template_args_non_tc;
  declaration.set_specialization_of(template_symbol.id);

  // We can't typecheck arguments yet, they are used
  // for guessing later. But we can check the number.
  if(
    template_args_non_tc.arguments().size() !=
    to_cpp_declaration(template_symbol.type)
      .template_type()
      .parameters()
      .size())
  {
    err_location(cpp_name.location());
    throw "template specialization with wrong number of arguments";
  }

  typecheck_class_template(declaration);
}

void cpp_typecheckt::convert_template_function_or_member_specialization(
  cpp_declarationt &declaration)
{
  cpp_save_scopet saved_scope(cpp_scopes);

  if(
    declaration.declarators().size() != 1 ||
    declaration.declarators().front().type().id() != "function_type")
  {
    err_location(declaration.type());
    str << "expected function template specialization";
    throw 0;
  }

  assert(declaration.declarators().size() == 1);
  cpp_declaratort declarator = declaration.declarators().front();
  cpp_namet &cpp_name = declarator.name();

  if(cpp_name.is_qualified())
  {
    err_location(cpp_name.location());
    str << "qualifiers not excpected here";
    throw 0;
  }

  // There is specialization (instantiation with template arguments)
  // but also function overloading (no template arguments)

  assert(!cpp_name.get_sub().empty());

  if(cpp_name.get_sub().back().id() == "template_args")
  {
    // proper specialization with arguments
    if(
      cpp_name.get_sub().size() != 2 || cpp_name.get_sub()[0].id() != "name" ||
      cpp_name.get_sub()[1].id() != "template_args")
    {
      // currently we are more restrictive
      // than the standard
      err_location(cpp_name.location());
      str << "bad template-function-specialization name";
      throw 0;
    }

    std::string base_name = cpp_name.get_sub()[0].get("identifier").c_str();

    cpp_scopest::id_sett id_set;

    cpp_scopes.get_ids(base_name, id_set, true);

    if(id_set.empty())
    {
      err_location(cpp_name.location());
      str << "template function `" << base_name << "' not found";
      throw 0;
    }
    if(id_set.size() > 1)
    {
      err_location(cpp_name.location());
      str << "template function `" << base_name << "' is ambiguous";
    }

    const symbolt &template_symbol = *lookup((*id_set.begin())->identifier);

    cpp_template_args_tct template_args = typecheck_template_args(
      declaration.location(),
      template_symbol,
      to_cpp_template_args_non_tc(cpp_name.get_sub()[1]));

    cpp_name.get_sub().pop_back();

    typet specialization;
    specialization.swap(declarator);

    instantiate_template(
      cpp_name.location(),
      template_symbol,
      template_args,
      template_args,
      specialization);
  }
  else
  {
    // Just overloading, but this is still a template
    // for disambiguation purposes!
    // http://www.gotw.ca/publications/mill17.htm
    cpp_declarationt new_declaration = declaration;

    new_declaration.remove("template_type");
    new_declaration.remove("is_template");
    new_declaration.set("#template", ""); // todo, get identifier

    convert_non_template_declaration(new_declaration);
  }
}

cpp_scopet &cpp_typecheckt::typecheck_template_parameters(template_typet &type)
{
  cpp_save_scopet cpp_saved_scope(cpp_scopes);

  assert(type.id() == "template");

  std::string id_suffix = "template::" + i2string(template_counter++);

  // produce a new scope for the template parameters
  cpp_scopet &template_scope = cpp_scopes.current_scope().new_scope(
    cpp_scopes.current_scope().prefix + id_suffix);

  template_scope.prefix = template_scope.get_parent().prefix + id_suffix;
  template_scope.id_class = cpp_idt::TEMPLATE_SCOPE;

  cpp_scopes.go_to(template_scope);

  // put template parameters into this scope
  template_typet::parameterst &parameters = type.parameters();

  unsigned anon_count = 0;

  for(auto &parameter : parameters)
  {
    cpp_declarationt declaration;
    declaration.swap(parameter);

    cpp_declarator_convertert cpp_declarator_converter(*this);

    // there must be _one_ declarator
    assert(declaration.declarators().size() == 1);

    cpp_declaratort &declarator = declaration.declarators().front();

    // it may be anonymous
    if(declarator.name().is_nil())
    {
      irept name("name");
      name.identifier("anon#" + i2string(++anon_count));
      declarator.name() = cpp_namet();
      declarator.name().get_sub().push_back(name);
    }

    cpp_declarator_converter.is_typedef = declaration.is_type();
    cpp_declarator_converter.is_template_argument = true;

    // There might be a default type or value.
    // We store it for later, as it can't be typechecked now
    // because of dependencies on earlier parameters!
    exprt default_value = declarator.value();
    declarator.value().make_nil();

    const symbolt &symbol =
      cpp_declarator_converter.convert(declaration, declarator);

    if(cpp_declarator_converter.is_typedef)
    {
      parameter = template_parametert("type", typet("symbol"));
      parameter.type().identifier(symbol.id);
      parameter.type().location() = declaration.find_location();
    }
    else
    {
      parameter = template_parametert("symbol", symbol.type);
      parameter.identifier(symbol.id);
    }

    // set (non-typechecked) default value
    if(default_value.is_not_nil())
      parameter.add("#default") = default_value;

    parameter.location() = declaration.find_location();
  }

  // continue without adding to the prefix
  template_scope.prefix = template_scope.get_parent().prefix;

  return template_scope;
}

cpp_template_args_tct cpp_typecheckt::typecheck_template_args(
  const locationt &location,
  const symbolt &template_symbol,
  const cpp_template_args_non_tct &template_args)
{
  // old stuff
  assert(template_args.id() != "already_typechecked");

  assert(template_symbol.type.get_bool("is_template"));

  const template_typet &template_type =
    to_cpp_declaration(template_symbol.type).template_type();

  // bad re-cast, but better than copying the args one by one
  cpp_template_args_tct result = (const cpp_template_args_tct &)(template_args);

  cpp_template_args_tct::argumentst &args = result.arguments();

  const template_typet::parameterst &parameters = template_type.parameters();

  if(parameters.size() < args.size())
  {
    err_location(location);
    str << "too many template arguments (expected " << parameters.size()
        << ", but got " << args.size() << ")";
    throw 0;
  }

  // we will modify the template map
  template_mapt old_template_map;
  old_template_map = template_map;

  // check for default parameters
  for(unsigned i = 0; i < parameters.size(); i++)
  {
    const template_parametert &parameter = parameters[i];
    cpp_save_scopet cpp_saved_scope(cpp_scopes);

    if(i >= args.size())
    {
      // Check for default parameter.
      // These may depend on previous arguments.
      if(!parameter.has_default_parameter())
      {
        err_location(location);
        str << "not enough template arguments (expected " << parameters.size()
            << ", but got " << args.size() << ")";
        throw 0;
      }

      args.push_back(parameter.default_parameter());

      // these need to be typechecked in the scope of the template,
      // not in the current scope!
      cpp_idt *template_scope = cpp_scopes.id_map[template_symbol.id];
      assert(template_scope != nullptr);
      cpp_scopes.go_to(*template_scope);
    }

    assert(i < args.size());

    exprt &arg = args[i];

    if(parameter.id() == "type")
    {
      if(arg.id() == "type")
      {
        typecheck_type(arg.type());
      }
      else if(arg.id() == "ambiguous")
      {
        typecheck_type(arg.type());
        typet t = arg.type();
        arg = exprt("type", t);
      }
      else
      {
        err_location(arg);
        str << "expected type, but got expression";
        throw 0;
      }
    }
    else // expression
    {
      if(arg.id() == "type")
      {
        err_location(arg);
        str << "expected expression, but got type";
        throw 0;
      }
      if(arg.id() == "ambiguous")
      {
        exprt e;
        e.swap(arg.type());
        arg.swap(e);
      }

      typecheck_expr(arg);
      simplify(arg);
      implicit_typecast(arg, parameter.type());
    }

    // set right away -- this is for the benefit of default
    // parameters

    template_map.set(parameter, arg);
  }

  // restore template map
  template_map.swap(old_template_map);

  // now the numbers should match
  assert(args.size() == parameters.size());

  return result;
}

void cpp_typecheckt::convert_template_declaration(cpp_declarationt &declaration)
{
  assert(declaration.is_template());

  if(declaration.member_spec().is_virtual())
  {
    err_location(declaration);
    str << "invalid use of 'virtual' in template declaration";
    throw 0;
  }

  if(convert_typedef(declaration.type()))
  {
    err_location(declaration);
    str << "template declaration for typedef";
    throw 0;
  }

  typet &type = declaration.type();

  // there are
  // 1) function templates
  // 2) class templates
  // 3) members of class templates (static or methods)

  if(declaration.is_class_template())
  {
    // there should not be declarators
    if(!declaration.declarators().empty())
    {
      err_location(declaration);
      throw "class template not expected to have declarators";
    }

    // it needs to be a class template
    if(type.id() != "struct")
    {
      err_location(declaration);
      throw "expected class template";
    }

    // Is it class template specialization?
    // We can tell if there are template arguments in the class name,
    // like template<...> class tag<stuff> ...
    if((static_cast<const cpp_namet &>(type.find("tag"))).has_template_args())
    {
      convert_class_template_specialization(declaration);
      return;
    }

    typecheck_class_template(declaration);
    return;
  }
  // maybe function template, maybe class template member

  // there should be declarators in either case
  if(declaration.declarators().empty())
  {
    err_location(declaration);
    throw "function template or class template member expected to have declarator";
  }

  // Is it function template specialization?
  // Only full specialization is allowed!
  if(declaration.template_type().parameters().empty())
  {
    convert_template_function_or_member_specialization(declaration);
    return;
  }

  // Explicit qualification is forbidden for function templates,
  // which we can use to distinguish them.

  assert(declaration.declarators().size() >= 1);

  cpp_declaratort &declarator = declaration.declarators()[0];
  const cpp_namet &cpp_name = to_cpp_name(declarator.add("name"));

  if(cpp_name.is_qualified() || cpp_name.has_template_args())
    return typecheck_class_template_member(declaration);

  // must be function template
  typecheck_function_template(declaration);
  return;
}
