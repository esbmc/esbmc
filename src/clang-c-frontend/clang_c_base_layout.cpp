#include <clang-c-frontend/clang_c_base_layout.h>
#include <util/base/prefix.h>
#include <util/expr/type_byte_size.h>
#include <util/irep/migrate.h>
#include <util/symtab/base_subobject.h>

// How far `c`, a component of the base, has moved in the derived struct.
// Nothing if the derived has no such member: matching is on type and on which
// class declared the member, not just on the name, because
// is_duplicate_component merges by name alone -- two bases with a same-named
// member share one slot, and a name-only match would confidently land in the
// other base's storage. A member the base itself declares is owned by it.
static std::optional<BigInt> member_delta(
  const namespacet &ns,
  const struct_typet &ds,
  const type2tc &d2,
  const type2tc &b2,
  const struct_typet::componentt &c,
  const irep_idt &base_id)
{
  const struct_typet::componentt &dc = ds.get_component(c.get_name());
  const irep_idt owner =
    c.get("#base_owner").empty() ? base_id : c.get("#base_owner");
  if (dc.is_nil() || dc.type() != c.type() || dc.get("#base_owner") != owner)
    return std::nullopt;

  return member_offset(d2, c.get_name(), &ns) -
         member_offset(b2, c.get_name(), &ns);
}

// Sum the offsets of the "@base@" components leading from `derived` down to
// the struct symbol `base_id`. Offsets come from ESBMC's own layout, so they
// agree with the member path the derived->base cast builds. adjust() fixes up
// every type symbol before any value, so padding is already in place here.
static bool base_subobject_offset(
  const namespacet &ns,
  const typet &derived,
  const irep_idt &base_id,
  BigInt &offset)
{
  const typet &d = ns.follow(derived);
  if (!d.is_struct())
    return false;

  const struct_typet &st = to_struct_type(d);
  const irep_idt want = base_subobject_name(base_id.as_string());

  for (const auto &c : st.components())
  {
    if (!has_prefix(c.get_name(), BASE_SUBOBJECT_PREFIX))
      continue;

    BigInt nested = 0;
    if (
      c.get_name() != want &&
      !base_subobject_offset(ns, c.type(), base_id, nested))
      continue;

    offset += member_offset(migrate_type(st), c.get_name(), &ns) + nested;
    return true;
  }

  return false;
}

// get_base_components_methods copies a base's components into the derived
// struct -- and stamps each with its declaring class -- only for the flattened
// layout it falls back to when the hierarchy contains a virtual base.
static bool uses_flattened_layout(const namespacet &ns, const typet &derived)
{
  const typet &d = ns.follow(derived);
  if (!d.is_struct())
    return false;

  for (const auto &c : to_struct_type(d).components())
    if (!c.get("#base_owner").empty())
      return true;
  return false;
}

// Displacement of `base_id`'s members inside the flattened `derived` layout,
// which get_base_components_methods produces for any hierarchy containing a
// virtual base. Every member of the base must appear in the derived struct at
// one common delta; a virtual base shared by two sibling bases has no such
// delta, so the caller keeps the unadjusted pointer. Padding is already in
// place here: adjust() completes every type symbol before any value.
static bool flattened_base_offset(
  const namespacet &ns,
  const typet &derived,
  const irep_idt &base_id,
  BigInt &offset)
{
  const typet &d = ns.follow(derived);
  const symbolt *base_sym = ns.lookup(base_id);
  if (!d.is_struct() || !base_sym)
    return false;

  const typet &b = ns.follow(base_sym->get_type());
  if (!b.is_struct())
    return false;

  const struct_typet &ds = to_struct_type(d);
  const struct_typet &bs = to_struct_type(b);

  // A base that keeps the nested layout reaches its own bases through an
  // "@base@" component, yet get_base_components_methods *also* copies those
  // ancestors' fields flat into the derived struct. The two copies alias only
  // at displacement zero, which is what every other access assumes (the
  // <ios>/<istream>/<ostream> models rely on it), so leave such a base alone.
  for (const auto &c : bs.components())
    if (c.get_bool("is_base_subobject"))
      return false;

  const type2tc d2 = migrate_type(ds);
  const type2tc b2 = migrate_type(bs);

  bool seen = false;
  BigInt delta = 0;
  for (const auto &c : bs.components())
  {
    if (c.get_is_padding() || c.get_is_unnamed_bitfield())
      continue;

    const std::optional<BigInt> d = member_delta(ns, ds, d2, b2, c, base_id);
    if (!d)
      return false;
    if (seen && *d != delta)
      return false;
    delta = *d;
    seen = true;
  }

  if (!seen || delta < 0)
    return false;

  offset = delta;
  return true;
}

// Displacement of `base_id`'s subobject inside `derived`, from ESBMC's own
// layout -- the only one the base-offset paths may use. Pick the oracle by
// layout, never by whichever answers first: a flattened struct also carries
// the "@base@" components it copied out of a nested-layout base, and walking
// those lands on storage duplicated at displacement zero (the <ios> models
// depend on that aliasing).
bool base_displacement(
  const namespacet &ns,
  const typet &derived,
  const irep_idt &base_id,
  BigInt &offset)
{
  return uses_flattened_layout(ns, derived)
           ? flattened_base_offset(ns, derived, base_id, offset)
           : base_subobject_offset(ns, derived, base_id, offset);
}
