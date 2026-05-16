#pragma once
// Generic switch-dispatch helpers for the irep2 hierarchy flattening
// (issue #4560).
//
// Include AFTER irep2_expr.h; do NOT pull this into irep2.h or irep2_expr.h —
// it depends on the full definitions of all concrete kinds.
//
// Each kind that has been migrated to the flat struct layout exposes a
// `static constexpr auto fields` tuple of member pointers covering its
// user-visible fields. The `has_fields_v<K>` trait detects that declaration
// and the `generic_*` functions provide the corresponding operation bodies.
// The v2 switch dispatchers in irep2_expr.cpp use
//   if constexpr (esbmct::has_fields_v<kind##2t>)
//     return esbmct::generic_*(static_cast<const kind##2t &>(*this), ...);
//   else
//     return static_cast<const kind##2t &>(*this).op(...);  // virtual path
// so unmigrated kinds still go through the existing virtual methods.

#include <tuple>
#include <type_traits>
#include <irep2/irep2_template_utils.h>
#include <irep2/irep2_templates.h>

namespace esbmct
{

// --------------------------------------------------------------------------
// has_fields_v<K>: true once K::fields has been declared.
// --------------------------------------------------------------------------
template <class K, class = void>
inline constexpr bool has_fields_v = false;

template <class K>
inline constexpr bool has_fields_v<K, std::void_t<decltype(K::fields)>> = true;

// --------------------------------------------------------------------------
// generic_cmp: structural equality over K::fields.
// Precondition: a.expr_id == o.expr_id (checked at the switch boundary).
// --------------------------------------------------------------------------
template <class K>
bool generic_cmp(const K &a, const expr2t &o)
{
  const K &b = static_cast<const K &>(o);
  bool eq = true;
  std::apply(
    [&](auto... mp) { ((eq = eq && do_type_cmp(a.*mp, b.*mp)), ...); },
    K::fields);
  return eq;
}

// --------------------------------------------------------------------------
// generic_lt: trinary ordering over K::fields.
// Precondition: a.expr_id == o.expr_id.
// --------------------------------------------------------------------------
template <class K>
int generic_lt(const K &a, const expr2t &o)
{
  const K &b = static_cast<const K &>(o);
  int r = 0;
  std::apply(
    [&](auto... mp) {
      ((r == 0 ? (void)(r = do_type_lt(a.*mp, b.*mp)) : (void)0), ...);
    },
    K::fields);
  return r;
}

// --------------------------------------------------------------------------
// generic_clone: copy via make_irep<K>.
// --------------------------------------------------------------------------
template <class K>
expr2tc generic_clone(const K &a)
{
  return make_irep<K>(a);
}

// --------------------------------------------------------------------------
// generic_do_crc: mix expr_id then each field in K::fields.
// For notype kinds (e.g. not2t) K::fields has no type slot, mirroring
// expr2t_traits_notype.  For kinds with dynamic type, K::fields must list
// &expr2t::type first so the CRC includes it.
// --------------------------------------------------------------------------
template <class K>
size_t generic_do_crc(const K &a)
{
  if (size_t cached = a.crc_val.load(std::memory_order_acquire); cached != 0)
    return cached;
  size_t v = 0;
  hash_combine(v, do_type_crc(a.expr_id));
  std::apply(
    [&](auto... mp) { (hash_combine(v, do_type_crc(a.*mp)), ...); },
    K::fields);
  a.crc_val.store(v, std::memory_order_release);
  return v;
}

// --------------------------------------------------------------------------
// generic_hash: ingest expr_id and each field into a SHA-1 state.
// --------------------------------------------------------------------------
template <class K>
void generic_hash(const K &a, crypto_hash &h)
{
  do_type_hash(a.expr_id, h);
  std::apply(
    [&](auto... mp) { (do_type_hash(a.*mp, h), ...); }, K::fields);
}

// --------------------------------------------------------------------------
// generic_tostring: build the pretty-print member list from K::fields.
// K::field_names[i] names the i-th user field (0-based).
// --------------------------------------------------------------------------
template <class K>
list_of_memberst generic_tostring(const K &a, unsigned int indent)
{
  list_of_memberst vec;
  unsigned int idx = 0;
  std::apply(
    [&](auto... mp) {
      (..., (do_type2string(a.*mp, idx++, K::field_names, vec, indent)));
    },
    K::fields);
  return vec;
}

// --------------------------------------------------------------------------
// generic_get_sub_expr: return pointer to the idx-th expr2tc child.
// --------------------------------------------------------------------------
template <class K>
const expr2tc *generic_get_sub_expr(const K &a, size_t idx)
{
  const expr2tc *result = nullptr;
  size_t cur = 0;
  std::apply(
    [&](auto... mp) {
      (void)(... || do_get_sub_expr(a.*mp, idx, cur, result));
    },
    K::fields);
  return result;
}

// --------------------------------------------------------------------------
// generic_get_sub_expr_nc: non-const version.
// --------------------------------------------------------------------------
template <class K>
expr2tc *generic_get_sub_expr_nc(K &a, size_t idx)
{
  expr2tc *result = nullptr;
  size_t cur = 0;
  std::apply(
    [&](auto... mp) {
      (void)(... || do_get_sub_expr_nc(a.*mp, idx, cur, result));
    },
    K::fields);
  return result;
}

// --------------------------------------------------------------------------
// generic_get_num_sub_exprs: count all expr2tc children in K::fields.
// --------------------------------------------------------------------------
template <class K>
size_t generic_get_num_sub_exprs(const K &a)
{
  size_t total = 0;
  std::apply(
    [&](auto... mp) { (void)((total += do_count_sub_exprs(a.*mp)), ...); },
    K::fields);
  return total;
}

// --------------------------------------------------------------------------
// generic_foreach_operand_impl_const: const delegate iteration.
// --------------------------------------------------------------------------
template <class K>
void generic_foreach_operand_impl_const(
  const K &a,
  expr2t::const_op_delegate &f)
{
  std::apply(
    [&](auto... mp) { (call_expr_delegate(a.*mp, f), ...); }, K::fields);
}

// --------------------------------------------------------------------------
// generic_foreach_operand_impl: non-const delegate iteration.
// --------------------------------------------------------------------------
template <class K>
void generic_foreach_operand_impl(K &a, expr2t::op_delegate &f)
{
  std::apply(
    [&](auto... mp) { (call_expr_delegate(a.*mp, f), ...); }, K::fields);
}

} // namespace esbmct
