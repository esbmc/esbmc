#include <goto-programs/add_restrict_assertions.h>

#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/migrate.h>
#include <util/namespace.h>
#include <util/std_types.h>
#include <util/type_byte_size.h>

// A restrict-qualified pointer parameter, reduced to the data the entry
// assertion needs: the pointer value (as a level-0 symbol) and the byte size of
// one pointed-to element.
namespace
{
struct restrict_paramt
{
  expr2tc pointer;
  BigInt element_size;
};
} // namespace

// Byte size of one pointed-to element. Incomplete/opaque (symbolic), function,
// and dynamically/infinitely sized subtypes have no concrete size computable
// here — sizing them would abort or throw — so fall back to one byte. That keeps
// the overlap test a sound under-approximation (it may miss some overlaps, never
// invent one).
static BigInt element_size(const type2tc &pointer_type, const namespacet &ns)
{
  const type2tc &subtype = to_pointer_type(pointer_type).subtype;
  if (is_symbol_type(subtype) || is_code_type(subtype))
    return 1;

  try
  {
    BigInt size = type_byte_size(subtype, &ns);
    return size <= 0 ? BigInt(1) : size;
  }
  catch (...)
  {
    return 1;
  }
}

// Collect the restrict-qualified pointer parameters of a function. The
// `#restricted` qualifier is only preserved on the legacy code_typet, so the
// parameters are read from the function symbol's type.
static std::vector<restrict_paramt>
collect_restrict_params(const symbolt &func_symbol, const namespacet &ns)
{
  std::vector<restrict_paramt> params;
  const code_typet &code_type = to_code_type(func_symbol.get_type());

  for (const code_typet::argumentt &arg : code_type.arguments())
  {
    const typet &arg_type = arg.type();
    if (!arg_type.is_pointer() || !arg_type.restricted())
      continue;

    // A named parameter is required to reference its value in the body.
    const irep_idt &ident = arg.get_identifier();
    if (ident.empty())
      continue;

    const type2tc pointer_type = migrate_type(arg_type);
    params.push_back(
      {symbol2tc(pointer_type, ident), element_size(pointer_type, ns)});
  }

  return params;
}

// Build the disjointness assertion for a pair of restrict pointers. The pointers
// alias illegally when both designate an object (are non-null), point into the
// same object, and their element footprints overlap:
//   !(a != NULL && b != NULL && same_object(a,b) &&
//     [oa, oa+sa) overlaps [ob, ob+sb))
// The non-null guard avoids a false alarm on unused/null parameters, which
// designate no object and so cannot violate the restrict contract.
static expr2tc
disjoint_assertion(const restrict_paramt &a, const restrict_paramt &b)
{
  const type2tc offset_type = get_int_type(config.ansi_c.address_width);
  expr2tc off_a = pointer_offset2tc(offset_type, a.pointer);
  expr2tc off_b = pointer_offset2tc(offset_type, b.pointer);
  expr2tc end_a =
    add2tc(offset_type, off_a, constant_int2tc(offset_type, a.element_size));
  expr2tc end_b =
    add2tc(offset_type, off_b, constant_int2tc(offset_type, b.element_size));

  expr2tc both_non_null = and2tc(
    notequal2tc(a.pointer, gen_zero(a.pointer->type)),
    notequal2tc(b.pointer, gen_zero(b.pointer->type)));
  expr2tc ranges_overlap =
    and2tc(lessthan2tc(off_a, end_b), lessthan2tc(off_b, end_a));

  expr2tc overlap = and2tc(
    both_non_null,
    and2tc(same_object2tc(a.pointer, b.pointer), ranges_overlap));

  return not2tc(overlap);
}

void add_restrict_assertions(contextt &context, goto_functionst &goto_functions)
{
  const namespacet ns(context);

  Forall_goto_functions (f_it, goto_functions)
  {
    goto_programt &body = f_it->second.body;
    if (body.instructions.empty())
      continue;

    const symbolt *func_symbol = context.find_symbol(f_it->first);
    if (func_symbol == nullptr || !func_symbol->get_type().is_code())
      continue;

    std::vector<restrict_paramt> params =
      collect_restrict_params(*func_symbol, ns);
    if (params.size() < 2)
      continue;

    goto_programt::targett first = body.instructions.begin();
    for (std::size_t i = 0; i < params.size(); i++)
      for (std::size_t j = i + 1; j < params.size(); j++)
      {
        goto_programt::targett t = body.insert(first);
        t->make_assertion(disjoint_assertion(params[i], params[j]));
        t->location = first->location;
        t->location.user_provided(false);
        t->location.comment("restrict pointer aliasing");
      }
  }

  goto_functions.update();
}
