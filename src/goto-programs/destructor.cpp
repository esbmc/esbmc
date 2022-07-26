#include <goto-programs/destructor.h>

bool get_destructor(
  const namespacet &ns,
  const typet &type,
  code_function_callt &destructor)
{
  if(type.id() == "symbol")
  {
    return get_destructor(ns, ns.follow(type), destructor);
  }

  if(type.id() == "struct")
  {
    const struct_typet &struct_type = to_struct_type(type);

    const struct_typet::componentst &components = struct_type.methods();

    for(const auto &component : components)
    {
      if(component.type().is_code())
      {
        const code_typet &code_type = to_code_type(component.type());

        if(
          code_type.return_type().id() == "destructor" &&
          code_type.arguments().size() == 1)
        {
          const typet &arg_type = code_type.arguments().front().type();

          if(
            arg_type.id() == "pointer" && ns.follow(arg_type.subtype()) == type)
          {
            exprt symbol_expr("symbol", component.type());
            symbol_expr.identifier(component.name());

            code_function_callt function_call;
            function_call.function() = symbol_expr;

            destructor = function_call;
            return true;
          }
        }
      }
    }
  }

  return false;
}
