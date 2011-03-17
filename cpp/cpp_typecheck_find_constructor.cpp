/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <type_eq.h>

#include "cpp_typecheck.h"

/*******************************************************************\

Function: cpp_typecheckt::find_constructor

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::find_constructor(
  const typet &start_dest_type,
  exprt &constructor_expr)
{
  constructor_expr.make_nil();

  typet dest_type(start_dest_type);
  follow_symbol(dest_type);

  if(dest_type.id()!="struct")
    return;

  const irept::subt &components=
    dest_type.find("components").get_sub();

  forall_irep(it, components)
  {
    const exprt &component=(exprt &)*it;
    const typet &type=component.type();

    if(type.find("return_type").id()=="constructor")
    {
      const irept::subt &arguments=type.find("arguments").get_sub();

      namespacet ns(context);

      if(arguments.size()==1)
      {
        const exprt &argument=(exprt &)arguments.front();
        const typet &arg_type=argument.type();

        if(arg_type.id()=="pointer" &&
           type_eq(arg_type.subtype(), dest_type, ns))
        {
          // found!
          const irep_idt &identifier=
            component.get("name");

          if(identifier=="")
            throw "constructor without identifier";

          constructor_expr=exprt("symbol", type);
          constructor_expr.set("identifier", identifier);
          return;
        }
      }
    }
  }
}
