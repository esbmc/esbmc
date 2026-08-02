#include <goto-programs/destructor.h>

const struct_typet *resolve_class_type(const namespacet &ns, const typet &type)
{
  if (type.id() == "symbol")
    return resolve_class_type(ns, ns.follow(type));

  if (type.id() != "struct")
    return nullptr;

  // An inline struct type embedded in a function body can be a degraded copy
  // of the class type: the --irep2-bodies round-trip strips the C++ `methods`
  // sub and component attributes, because IREP2 struct types store only data
  // members. Resolve it back to the authoritative type symbol via its tag so
  // callers see the full method list and the self-reference comparison
  // matches. Falls back to the inline type for anonymous structs and when no
  // symbol is bound (the legacy flag-off type already carries its methods, so
  // this is a no-op there).
  const irep_idt &tag = type.get("tag");
  const symbolt *type_sym =
    tag.empty() ? nullptr : ns.lookup("tag-" + id2string(tag));

  return &to_struct_type(
    (type_sym != nullptr && type_sym->get_type().id() == "struct")
      ? type_sym->get_type()
      : type);
}

const struct_typet::componentt *
get_destructor_component(const namespacet &ns, const struct_typet &struct_type)
{
  for (const auto &component : struct_type.methods())
  {
    if (!component.type().is_code())
      continue;

    const code_typet &code_type = to_code_type(component.type());

    if (
      code_type.return_type().id() != "destructor" ||
      code_type.arguments().size() != 1)
      continue;

    const typet &arg_type = code_type.arguments().front().type();

    if (
      arg_type.id() == "pointer" &&
      ns.follow(arg_type.subtype()) == struct_type)
      return &component;
  }

  return nullptr;
}

bool get_destructor(
  const namespacet &ns,
  const typet &type,
  code_function_callt &destructor)
{
  const struct_typet *struct_type = resolve_class_type(ns, type);
  if (struct_type == nullptr)
    return false;

  const struct_typet::componentt *component =
    get_destructor_component(ns, *struct_type);

  if (component == nullptr)
    return false;

  exprt symbol_expr("symbol", component->type());
  symbol_expr.identifier(component->name());

  code_function_callt function_call;
  function_call.function() = symbol_expr;

  destructor = function_call;
  return true;
}
