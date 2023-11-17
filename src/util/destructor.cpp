#include <std_types.h>
#include <std_code.h>
#include <util/destructor.h>

code_function_callt get_destructor(const namespacet &ns, const typet &type)
{
  if (type.id() == "symbol")
    return get_destructor(ns, ns.follow(type));

  if (type.id() == "struct")
  {
    const struct_typet &struct_type = to_struct_type(type);

    auto it = ns.get_context().symbol_base_map.find(
      "~" + struct_type.tag().as_string());
    if (it != ns.get_context().symbol_base_map.end())
    {
      const symbolt *cpp_delete = ns.get_context().find_symbol(it->second);

      code_function_callt function_call;
      function_call.function() = symbol_exprt(cpp_delete->id, cpp_delete->type);

      return function_call;
    }
  }

  return static_cast<const code_function_callt &>(get_nil_irep());
}
