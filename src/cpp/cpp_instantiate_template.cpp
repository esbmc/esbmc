/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_type2name.h>
#include <cpp/cpp_typecheck.h>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/simplify_expr.h>
#include <util/simplify_expr_class.h>

/*******************************************************************\

Function: cpp_typecheckt::template_suffix

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string cpp_typecheckt::template_suffix(
  const cpp_template_args_tct &template_args)
{
  // quick hack
  std::string result="<";
  bool first=true;

  const cpp_template_args_tct::argumentst &arguments=
    template_args.arguments();

  for(cpp_template_args_tct::argumentst::const_iterator
      it=arguments.begin();
      it!=arguments.end();
      it++)
  {
    if(first) first=false; else result+=",";

    const exprt expr=*it;

    assert(expr.id()!="ambiguous");

    if(expr.id()=="type")
    {
      const typet &type=expr.type();
      if(type.id()=="symbol")
        result+=type.get_string("identifier");
      else
        result+=cpp_type2name(type);
    }
    else if(expr.id()=="sideeffect")
    {
      const typet &type=expr.type();
      if(type.id()=="symbol")
        result+=type.get_string("identifier");
      else
        result+=cpp_type2name(type);
    }
    else // expression
    {
      exprt e=expr;
      make_constant(e);

      // this must be a constant, which includes true/false
      mp_integer i;

      if(e.is_true())
        i=1;
      else if(e.is_false())
        i=0;
      else if(to_integer(e, i))
      {
        err_location(*it);
        str << "template argument expression expected to be "
               "scalar constant, but got `"
            << to_string(e) << "'";
        throw 0;
      }

      result+=integer2string(i);
    }
  }

  result+='>';

  return result;
}

/*******************************************************************\

Function: cpp_typecheckt::show_instantiation_stack

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::show_instantiation_stack(std::ostream &out)
{
  for(instantiation_stackt::const_iterator
      s_it=instantiation_stack.begin();
      s_it!=instantiation_stack.end();
      s_it++)
  {
    const symbolt &symbol=lookup(s_it->identifier);
    out << "instantiating `" << symbol.pretty_name << "' with <";

    forall_expr(a_it, s_it->full_template_args.arguments())
    {
      if(a_it!=s_it->full_template_args.arguments().begin())
        out << ", ";

      if(a_it->id()=="type")
        out << a_it->type().id();
      else
        out << a_it->id();
    }

    out << "> at " << s_it->location << std::endl;
  }
}

const symbolt *
cpp_typecheckt::is_template_instantiated(
    const irep_idt &template_symbol_name,
    const irep_idt &template_pattern_name) const
{

  // Check whether the instance already exists. The 'template_instances' irep
  // contains a list of already instantiated patterns, and the symbol names
  // where the resulting thing is.
  const symbolt &template_symbol = lookup(template_symbol_name);
  const irept &instances = template_symbol.value.find("template_instances");
  if (!instances.is_nil()) {
    if (instances.get(template_pattern_name) != "") {

      // It has already been instantianted! Look up the symbol.
      const symbolt &symb=lookup(instances.get(template_pattern_name));

      // continue if the type is incomplete only -- it might now be complete(?).
      if (symb.type.id() != "incomplete_struct" || symb.value.is_not_nil())
        return &symb;
    }
  }

  return NULL;
}

void cpp_typecheckt::mark_template_instantiated(
    const irep_idt &template_symbol_name,
    const irep_idt &template_pattern_name,
    const irep_idt &instantiated_symbol_name)
{
  symbolt* s = context.find_symbol(template_symbol_name);
  assert(s != nullptr);

  // Set a flag in the template's value indicating that this has been
  // instantiated, and what the instantiated things symbol is.
  irept &new_instances = s->value.add("template_instances");
  new_instances.set(template_pattern_name, instantiated_symbol_name);
  return;
}

const symbolt *
cpp_typecheckt::handle_recursive_template_instance(
    const symbolt &template_symbol,
    const cpp_template_args_tct &full_template_args,
    const exprt &new_decl)
{
  // Recursive template uses are fine if it doesn't lead to a cyclic
  // definition, in the same way that C structs can't be recursively defined.
  // It's OK for this to happen within a method though; those get typechecked
  // later.
  // Detect recursive instantiations, then resolve them to a symbolic type.
  // Anything that attempts to find a concrete value from that type should
  // mean that the template is defined cyclicly, and so it's a program error.

  // The first item on the instantiation stack is the one we're wondering is
  // recursive. Skip it.
  instantiation_stackt::const_reverse_iterator it =
    instantiation_stack.rbegin();
  it++;

  // Look for this template being instantiated.
  for (; it != instantiation_stack.rend(); it++) {
    if (it->identifier == template_symbol.name) {
      // OK, we found it. Now, are the types equivalent?
      typedef cpp_template_args_baset::argumentst argumentst;
      const argumentst &src_args = it->full_template_args.arguments();
      const argumentst &cur_args = full_template_args.arguments();

      if (src_args.size() != cur_args.size())
        continue;

      bool match = true;
      for (unsigned int i = 0; i != src_args.size(); i++) {
        if (!base_type_eq(src_args[i].type(), cur_args[i].type(), *this)) {
          match = false;
          break;
        }
      }

      if (!match)
        continue;

      // OK: Recursion detected. For the moment, only deal with structs.
      assert(new_decl.type().id() == "struct");
      std::string instance = fetch_compound_name(new_decl.type());

      // We have the name this is /going/ to resolve to once it's instantiated.
      // Now create a temporary symbol that links back to it if it's followed.
      // (Unless it already exists).

      irep_idt link_symbol = instance + "$recurse";
      const symbolt* s = context.find_symbol(link_symbol);
      if(s != nullptr)
        return s;

      // Nope; create it.
      symbolt symbol;
      symbol.name = link_symbol;
      symbol.base_name = template_symbol.base_name;
      symbol.value = exprt();
      symbol.location = locationt();
      symbol.mode = mode; // uhu.
      symbol.module = module; // uuuhu.
      symbol.type = symbol_typet(instance);
      symbol.is_macro = false;
      symbol.is_type = true;
      symbol.pretty_name = template_symbol.base_name;

      // Insert.
      symbolt *new_symbol;
      if (context.move(symbol, new_symbol))
        throw "cpp_typecheckt::handle_recurse_templ: context.move() failed";

      return new_symbol;
    }
  }

  return NULL;
}

bool cpp_typecheckt::has_incomplete_args(
  cpp_template_args_tct template_args_tc)
{
  const cpp_template_args_tct::argumentst &_arguments =
    template_args_tc.arguments();

  for (cpp_template_args_tct::argumentst::const_iterator it =
    _arguments.begin(); it != _arguments.end(); it++)
  {
    const typet& e = it->type();

    symbolt* arg_sym = context.find_symbol(e.identifier());
    if(arg_sym != nullptr)
    {
      if (arg_sym->type.id() == "incomplete_struct")
      {
        std::cerr << "**** WARNING: template instantiation with incomplete type "
          << arg_sym->pretty_name << " at "<< arg_sym->location << std::endl;
        return true;
      }
    }
  }

  return false;
}

/*******************************************************************\

Function: cpp_typecheckt::instantiate_template

  Inputs: location of the instantiation,
          the identifier of the template symbol,
          typechecked template arguments,
          an (optional) specialization

 Outputs:

 Purpose:

\*******************************************************************/

const symbolt &cpp_typecheckt::instantiate_template(
  const locationt &location,
  const symbolt &template_symbol,
  const cpp_template_args_tct &specialization_template_args,
  const cpp_template_args_tct &full_template_args,
  const typet &specialization)
{

  if(instantiation_stack.size()==50)
  {
    err_location(location);
    throw "reached maximum template recursion depth";
  }

  instantiation_levelt i_level(instantiation_stack);
  instantiation_stack.back().location=location;
  instantiation_stack.back().identifier=template_symbol.name;
  instantiation_stack.back().full_template_args=full_template_args;

  #if 0
  std::cout << "L: " << location << std::endl;
  std::cout << "I: " << template_symbol.name << std::endl;
  #endif

  cpp_save_scopet cpp_saved_scope(cpp_scopes);
  cpp_saved_template_mapt saved_map(template_map);

  bool specialization_given=specialization.is_not_nil();

  // we should never get 'unassigned' here
  assert(!specialization_template_args.has_unassigned());
  assert(!full_template_args.has_unassigned());

  // do we have args?
  if(full_template_args.arguments().empty())
  {
    err_location(location);
    str << "`" << template_symbol.base_name
        << "' is a template; thus, expected template arguments";
    throw 0;
  }

  // produce new symbol name
  std::string suffix=template_suffix(full_template_args);

  // we need the template scope to see the parameters
  cpp_scopet *template_scope=
    static_cast<cpp_scopet *>(cpp_scopes.id_map[template_symbol.name]);

  if(template_scope==NULL)
  {
    err_location(location);
    str << "identifier: " << template_symbol.name << std::endl;
    throw "template instantiation error: scope not found";
  }

  assert(template_scope!=NULL);

  // produce new declaration
  cpp_declarationt new_decl=to_cpp_declaration(template_symbol.type);

  // the new one is not a template any longer, but we remember the
  // template type
  template_typet template_type=new_decl.template_type();
  new_decl.remove("is_template");
  new_decl.remove("template_type");
  new_decl.set("#template", template_symbol.name);
  new_decl.set("#template_arguments", specialization_template_args);

  // Let's check if the arguments are incompletes (they might have been
  // forward declared)
  if(has_incomplete_args(specialization_template_args))
  {
    // This happens when the arguments were not declared yet but the
    // code tried to use the template. This can happen when
    // typedefing for example, check esbmc-cpp/esbmc-cbmc/Templates39 for
    // an example
    //
    // Hack: let's remove the template body so nothing will be instantiated
    // When an object is instantiated in the future, it will create the
    // right instantiated template, or will throw an error if the argument
    // isn't declared yet
    new_decl.type().remove("body");
  }

  // save old scope
  cpp_save_scopet saved_scope(cpp_scopes);

  // mapping from template parameters to values/types
  template_map.build(template_type, specialization_template_args);

  // enter the template scope
  cpp_scopes.go_to(*template_scope);

  // Is it a template method?
  // It's in the scope of a class, and not a class itself.
  bool is_template_method=
    cpp_scopes.current_scope().get_parent().is_class() &&
    new_decl.type().id()!="struct";

  irep_idt class_name;

  if(is_template_method)
    class_name=cpp_scopes.current_scope().get_parent().identifier;

  // sub-scope for fixing the prefix
  std::string subscope_name=id2string(template_scope->identifier)+suffix;

  // Does it already exist?
  const symbolt *existing_template_instance =
    is_template_instantiated(template_symbol.name, subscope_name);
  if (existing_template_instance) {
    // continue if the type is incomplete only -- it might now be complete(?).
//      if (symb.type.id() != "incomplete_struct" || symb.value.is_not_nil())
      return *existing_template_instance;
  }

  // set up a scope as subscope of the template scope
  std::string prefix=template_scope->get_parent().prefix+suffix;
  cpp_scopet &sub_scope=
    cpp_scopes.current_scope().new_scope(subscope_name);
  sub_scope.prefix=prefix;
  cpp_scopes.go_to(sub_scope);
  cpp_scopes.id_map.insert(
    cpp_scopest::id_mapt::value_type(subscope_name, &sub_scope));

  // store the information that the template has
  // been instantiated using these arguments
  {
    // need non-const handle on template symbol
    symbolt &s = *context.find_symbol(template_symbol.name);
    irept &instantiated_with=s.value.add("instantiated_with");
    instantiated_with.get_sub().push_back(specialization_template_args);
  }

  #if 0
  std::cout << "MAP:" << std::endl;
  template_map.print(std::cout);
  #endif

  // fix the type
  {
    typet declaration_type=new_decl.type();

    // specialization?
    if(specialization_given)
    {
      if(declaration_type.id()=="struct")
      {
        declaration_type=specialization;
        declaration_type.location()=location;
      }
      else
      {
        irept tmp=specialization;
        new_decl.declarators()[0].swap(tmp);
      }
    }

    template_map.apply(declaration_type);
    new_decl.type().swap(declaration_type);
  }

  // Before properly typechecking this instance: are we already doing that
  // right now, recursively? If so, this will explode, so generate a symbolic
  // type instead. Currently only rated for structs.
  if(new_decl.type().id()=="struct") {
    const symbolt *recurse_sym = handle_recursive_template_instance(
        template_symbol, full_template_args, new_decl);
    if (recurse_sym)
      return *recurse_sym;
  }

  // We're definitely instantiating this; put the template types into scope.
  put_template_args_in_scope(template_type, specialization_template_args);

  if(new_decl.type().id()=="struct")
  {
    convert(new_decl);

    symbolt &new_symb=
      const_cast<symbolt&>(lookup(new_decl.type().identifier()));

    // Mark template as instantiated before instantiating template methods,
    // as they might then go and instantiate recursively.
    mark_template_instantiated(template_symbol.name, subscope_name,
                               new_symb.name);

    // also instantiate all the template methods
    const exprt &template_methods=
      static_cast<const exprt &>(
        template_symbol.value.find("template_methods"));

    for(unsigned i=0; i<template_methods.operands().size(); i++)
    {
      cpp_saved_scope.restore();

      cpp_declarationt method_decl=
        static_cast<const cpp_declarationt &>(
          static_cast<const irept &>(template_methods.operands()[i]));

      // copy the type of the template method
      template_typet method_type=
        method_decl.template_type();

      // do template arguments
      // this also sets up the template scope of the method
      cpp_scopet &method_scope=
        typecheck_template_parameters(method_type);

      cpp_scopes.go_to(method_scope);

      // mapping from template arguments to values/types
      template_map.build(method_type, specialization_template_args);

      method_decl.remove("template_type");
      method_decl.remove("is_template");

      convert(method_decl);
    }

    // any template instance to remember?
    if(new_decl.find("#template").is_not_nil())
    {
      new_symb.type.set("#template", new_decl.find("#template"));
      new_symb.type.set("#template_arguments", new_decl.find("#template_arguments"));
    }

    // Put the template we're instantiating from into the class scope. The class
    // is entitled to use its own template with different template args, and in
    // that circumstance it needs to be able to resolve the classname to the
    // template, not just the instantiated class.
    cpp_scopet &class_scope =
      cpp_scopes.get_scope(new_decl.type().identifier());
    cpp_idt &identifier=
      cpp_scopes.put_into_scope(template_symbol, class_scope, false);
    identifier.id_class = cpp_idt::TEMPLATE;

    return new_symb;
  }

  if(is_template_method)
  {
    symbolt* s = context.find_symbol(class_name);
    assert(s != nullptr);

    symbolt &symb = *s;
    assert(new_decl.declarators().size() == 1);

    if(new_decl.member_spec().is_virtual())
    {
      err_location(new_decl);
      str <<  "invalid use of `virtual' in template declaration";
      throw 0;
    }

    if(convert_typedef(new_decl.type()))
    {
      err_location(new_decl);
      str << "template declaration for typedef";
      throw 0;
    }

    if(new_decl.storage_spec().is_extern() ||
       new_decl.storage_spec().is_auto() ||
       new_decl.storage_spec().is_register() ||
       new_decl.storage_spec().is_mutable())
    {
      err_location(new_decl);
      str << "invalid storage class specified for template field";
      throw 0;
    }

    bool is_static=new_decl.storage_spec().is_static();
    irep_idt access = new_decl.get("#access");

    assert(access!=irep_idt());
    assert(symb.type.id()=="struct");

    typecheck_compound_declarator(
      symb,
      new_decl,
      new_decl.declarators()[0],
      to_struct_type(symb.type).components(),
      access,
      is_static,
      false,
      false);

    irep_idt sym_name = to_struct_type(symb.type).components().back().name();
    mark_template_instantiated(template_symbol.name, subscope_name, sym_name);
    symbolt &final_sym = *context.find_symbol(sym_name);

    // Propagate the '#template' attributes
    final_sym.type.set("#template", new_decl.find("#template"));
    final_sym.type.set("#template_arguments",
                       new_decl.find("#template_arguments"));
    return final_sym;
  }

  // not a class template, not a class template method,
  // it must be a function template!

  assert(new_decl.declarators().size()==1);

  convert_non_template_declaration(new_decl);

  const irep_idt &new_sym_name = new_decl.declarators()[0].identifier();
  mark_template_instantiated(template_symbol.name, subscope_name, new_sym_name);
  return lookup(new_sym_name);
}

void
cpp_typecheckt::put_template_args_in_scope(
    const template_typet &template_type,
    const cpp_template_args_tct &template_args)
{
  const template_typet::parameterst &template_parameters=
    template_type.parameters();

  cpp_template_args_tct::argumentst instance=
    template_args.arguments();

  template_typet::parameterst::const_iterator t_it=
    template_parameters.begin();

  if(instance.size()<template_parameters.size())
  {
    // check for default parameters
    for(unsigned i=instance.size();
        i<template_parameters.size();
        i++)
    {
      const template_parametert &param=template_parameters[i];

      if(param.has_default_parameter())
        instance.push_back(param.default_parameter());
      else
        break;
    }
  }

  // these should have been typechecked before
  assert(instance.size()==template_parameters.size());

  for(cpp_template_args_tct::argumentst::const_iterator
      i_it=instance.begin();
      i_it!=instance.end();
      i_it++, t_it++)
  {
    put_template_arg_into_scope(*t_it, *i_it);
  }
}

void
cpp_typecheckt::put_template_arg_into_scope(
      const template_parametert &template_param,
      const exprt &argument)
{
  symbolt symbol;

  // Fetch useful information for the following symbol construction
  cpp_scopet *cur_scope = &cpp_scopes.current_scope();
  std::string cur_scope_prefix = cur_scope->identifier.as_string();

  // Template parameter is either a type with a symbol type; or it's an
  // expression that's a symbol.
  irep_idt templ_param_id;
  if (template_param.id() == "symbol") {
    symbol.is_type = false;
    templ_param_id = template_param.identifier();
  } else {
    symbol.is_type = true;
    const typet &templ_param_type = template_param.type();
    assert(templ_param_type.id() == "symbol");
    templ_param_id = templ_param_type.identifier();
  }

  // Find declaration of that templated type.
  const symbolt &orig_symbol = lookup(templ_param_id);

  // Construct a new, concrete type symbol, with the base name as the templated
  // type name, and with the current scopes prefix.
  symbol.name = cur_scope_prefix + "::" + orig_symbol.base_name.as_string();
  symbol.base_name = orig_symbol.base_name;
  symbol.value = argument;
  symbol.location = argument.location();
  symbol.mode = mode; // uhu.
  symbol.module = module; // uuuhu.
  symbol.type = argument.type(); // BAM
  symbol.is_macro = false;
  symbol.pretty_name = orig_symbol.base_name;

  // Install this concrete type symbol into the context.
  symbolt *new_symbol;
  if (context.move(symbol, new_symbol)) {
    // Normally this is a good indicator that something to do with recursive
    // or nested templates has gone wrong. However, it becomes much more complex
    // when incomplete structs turn up, which can be instantiated multiple times
    // unsuccessfully. To guard against this, don't make this a fatal error
    // (for now).
    //throw "cpp_typecheckt::put_template_arg_in_scope: context.move() failed";
    return;
  }

  // And install it into the templates scope too.
  cpp_idt &identifier=
    cpp_scopes.put_into_scope(*new_symbol, *cur_scope, false);

  // Mark it as being a template argument
  identifier.id_class = cpp_idt::TEMPLATE_ARGUMENT;

  // Resolver code will pick up the fact that this argument exists; look up its
  // fully qualified name and get the above symbolt; and then pick out either
  // the type or value this resolves to.
}
