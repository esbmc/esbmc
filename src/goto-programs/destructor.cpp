#include <goto-programs/destructor.h>

bool get_destructor(
  const namespacet &ns,
  const typet &type,
  code_function_callt &destructor)
{
  if (type.id() == "symbol")
  {
    return get_destructor(ns, ns.follow(type), destructor);
  }

  if (type.id() == "struct")
  {
    // An inline struct type embedded in a function body can be a degraded copy
    // of the class type: the --irep2-bodies round-trip strips the C++ `methods`
    // sub and component attributes, because IREP2 struct types store only data
    // members. Resolve it back to the authoritative type symbol via its tag so
    // the scan below sees the full method list and the self-reference
    // comparison matches. Falls back to the inline type for anonymous structs
    // and when no symbol is bound (the legacy flag-off type already carries its
    // methods, so this is a no-op there).
    const irep_idt &tag = type.get("tag");
    const symbolt *type_sym =
      tag.empty() ? nullptr : ns.lookup("tag-" + id2string(tag));
    const typet &resolved =
      (type_sym != nullptr && type_sym->get_type().id() == "struct")
        ? type_sym->get_type()
        : type;

    const struct_typet &struct_type = to_struct_type(resolved);

    const struct_typet::componentst &components = struct_type.methods();

    for (const auto &component : components)
    {
      if (component.type().is_code())
      {
        const code_typet &code_type = to_code_type(component.type());

        if (
          code_type.return_type().id() == "destructor" &&
          code_type.arguments().size() == 1)
        {
          const typet &arg_type = code_type.arguments().front().type();

          if (
            arg_type.id() == "pointer" &&
            ns.follow(arg_type.subtype()) == resolved)
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
