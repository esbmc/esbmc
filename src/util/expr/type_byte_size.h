#ifndef CPROVER_POINTER_OFFSET_SIZE_H
#define CPROVER_POINTER_OFFSET_SIZE_H

#include <util/irep/expr.h>
#include <irep2/irep2.h>
#include <util/arith/mp_arith.h>
#include <util/symtab/namespace.h>
#include <util/irep/std_types.h>
#include <string>
#include <vector>

BigInt member_offset_bits(
  const type2tc &type,
  const irep_idt &member,
  const namespacet *ns = nullptr);
BigInt member_offset(
  const type2tc &type,
  const irep_idt &member,
  const namespacet *ns = nullptr);

/* The field paths of `type` at which a sub-object of type `target` sits,
 * spelled as value-set suffixes (".m" for a member, "[]" for an element).
 * These invert the offset walks above: members are accumulated in bits, as
 * member_offset_bits does, and array elements in bytes, as the index arm of
 * value_sett::get_reference_set_rec does. They live here so the two directions
 * share a file and the conventions they must agree on -- padding as explicit
 * members, unions overlaid at zero, ns.follow before measuring -- are held in
 * one place rather than by comment (esbmc/esbmc#6981, R31/R32). */
std::vector<std::string> member_paths_at_offset(
  const type2tc &type,
  const BigInt &offset,
  const type2tc &target,
  const namespacet &ns);

/* As above, but for a descriptor carrying no constant offset: no single path is
 * selected, so every path of the right type is possible. No size is consulted,
 * there being no offset to place, which keeps a target inside a
 * variable-length element reachable where the offset walk has to drop it. */
std::vector<std::string> member_paths_of_type(
  const type2tc &type,
  const type2tc &target,
  const namespacet &ns);

/* These can throw array_type2t::inf_sized_array_excp or
 * array_type2t::dyn_sized_array_excp */
BigInt type_byte_size_bits(const type2tc &type, const namespacet *ns = nullptr);
BigInt type_byte_size(const type2tc &type, const namespacet *ns = nullptr);
BigInt type_byte_size_default(
  const type2tc &type,
  const BigInt &defaultval,
  const namespacet *ns = nullptr);

/* type_byte_size*_expr() can throw array_type2t::inf_sized_array_excp */
expr2tc
type_byte_size_bits_expr(const type2tc &type, const namespacet *ns = nullptr);
expr2tc
type_byte_size_expr(const type2tc &type, const namespacet *ns = nullptr);

expr2tc
compute_pointer_offset(const expr2tc &expr, const namespacet *ns = nullptr);
expr2tc compute_pointer_offset_bits(
  const expr2tc &expr,
  const namespacet *ns = nullptr);

const expr2tc &get_base_object(const expr2tc &expr);

/* Number of bytes an ExtInt is represented with: the smallest power of two
 * that holds its width. */
std::size_t ext_int_representation_bytes(const typet &type);

/* The alignment of `type` in bytes: an explicit "alignment" attribute when
 * present, 1 when packed, otherwise the natural alignment implied by the
 * type's layout. */
BigInt alignment(const typet &type, const namespacet &ns);

#endif
