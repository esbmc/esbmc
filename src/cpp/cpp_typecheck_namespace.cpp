/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_typecheck.h>
#include <util/location.h>

/*******************************************************************\

Function: cpp_typecheckt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::convert(cpp_namespace_spect &namespace_spec)
{
  // save the scope
  cpp_save_scopet saved_scope(cpp_scopes);

  const irep_idt &name=namespace_spec.get_namespace();

  if(name=="")
  {
    // "unique namespace"
    err_location(namespace_spec);
    throw "unique namespace not supported yet";
  }

  irep_idt final_name(name);

  std::string identifier=
    cpp_scopes.current_scope().prefix+id2string(final_name);

  symbolt* s = context.find_symbol(identifier);

  if(s != nullptr)
  {
    if(namespace_spec.alias().is_not_nil())
    {
      err_location(namespace_spec);
      str << "namespace alias `" << final_name
          << "' previously declared" << std::endl;
      str << "location of previous declaration: "
          << s->location;
      throw 0;
    }

    if(s->type.id() != "namespace")
    {
      err_location(namespace_spec);
      str << "namespace `" << final_name
          << "' previously declared" << std::endl;
      str << "location of previous declaration: "
          << s->location;
      throw 0;
    }

    // enter that scope
    cpp_scopes.set_scope(identifier);
  }
  else
  {
    symbolt symbol;

    symbol.name=identifier;
    symbol.base_name=final_name;
    symbol.value.make_nil();
    symbol.location=namespace_spec.location();
    symbol.mode=current_mode;
    symbol.module=module;
    symbol.type=typet("namespace");

    if(context.move(symbol))
      throw "cpp_typecheckt::convert_namespace: context.move() failed";

    cpp_scopes.new_namespace(final_name);
  }

  /*if(namespace_spec.alias().is_not_nil())
  {
    cpp_typecheck_resolvet resolver(*this);
    cpp_scopet &s=resolver.resolve_namespace(namespace_spec.alias());
    cpp_scopes.current_scope().add_using_scope(s);
  }
  else
  {*/
    // do the declarations
    for(cpp_namespace_spect::itemst::iterator
        it=namespace_spec.items().begin();
        it!=namespace_spec.items().end();
        it++)
      convert(*it);
//  }
}
