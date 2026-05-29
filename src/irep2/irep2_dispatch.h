#pragma once
// Generic switch-dispatch helpers for irep2's flat node layout, plus
// the per-field operations they invoke.
//
// Include AFTER irep2_expr.h: this depends on the full definitions of
// every concrete kind. Do NOT pull this into irep2.h or irep2_expr.h.
//
// Each concrete kind exposes a `static constexpr auto fields` tuple of
// member pointers covering its user-visible fields, plus a static
// `field_names` array naming them in tuple order. The generic_*<K>
// helpers below walk that tuple via std::apply to implement cmp/lt/crc/
// hash/tostring/clone/get_sub_expr/foreach_operand uniformly. The
// switch-on-id dispatchers on expr2t / type2t pick the right helper
// per kind from the X-macro manifests (`expr_kinds.inc`,
// `type_kinds.inc`).

#include <tuple>
#include <type_traits>
#include <util/crypto_hash.h>
#include <util/fixedbv.h>
#include <util/i2string.h>
#include <util/ieee_float.h>
#include <util/migrate.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>

// ============================================================================
// Per-field operations the generic dispatchers invoke on every K::fields
// entry: pretty-printing, structural cmp/lt, CRC, SHA-1 ingestion, and
// sub-expression / delegate iteration. Primary templates cover trivially-
// comparable / hashable field types; explicit overloads handle BigInt
// sign-aware hashing, std::vector<...>, null-safe expr2tc/type2tc dispatch,
// irep_idt's interned-index hashing, and the *_ids dummies that mix the
// kind id into the CRC.
// ============================================================================

std::string type_to_string(const bool &thebool, int);
std::string type_to_string(const sideeffect_allockind &data, int);
std::string type_to_string(const unsigned int &theval, int);
std::string type_to_string(const constant_string_kindt &theval, int);
std::string type_to_string(const printf_kindt &theval, int);
std::string type_to_string(const symbol_renaming_level &theval, int);
std::string type_to_string(const BigInt &theint, int);
std::string type_to_string(const fixedbvt &theval, int);
std::string type_to_string(const ieee_floatt &theval, int);
std::string type_to_string(const std::vector<expr2tc> &theval, int indent);
std::string type_to_string(const std::vector<type2tc> &theval, int indent);
std::string type_to_string(const std::vector<irep_idt> &theval, int indent);
std::string type_to_string(const expr2tc &theval, int indent);
std::string type_to_string(const type2tc &theval, int indent);
std::string type_to_string(const irep_idt &theval, int);

// Structural equality on operator==.
template <class T>
inline bool do_type_cmp(const T &side1, const T &side2)
{
  return side1 == side2;
}

// Trinary -1/0/1 ordering on operator<.
template <class T>
inline int do_type_lt(const T &side1, const T &side2)
{
  if (side1 < side2)
    return -1;
  if (side2 < side1)
    return 1;
  return 0;
}

// std::hash<T> for the non-enum case. For enums, hash through uint8_t
// so the value matches the byte the CRC mix actually folds into the
// state — on-disk caches and state hashes depend on this width.
template <class T>
inline size_t do_type_crc(const T &theval)
{
  if constexpr (std::is_enum_v<T>)
    return std::hash<uint8_t>{}(static_cast<uint8_t>(theval));
  else
    return std::hash<T>{}(theval);
}

// Raw POD bytes into the SHA-1 stream.
template <class T>
inline void do_type_hash(const T &theval, crypto_hash &hash)
{
  hash.ingest((void *)&theval, sizeof(theval));
}

// Explicit overloads for the field types whose semantics differ.

int do_type_lt(const BigInt &side1, const BigInt &side2);
int do_type_lt(
  const std::vector<expr2tc> &side1,
  const std::vector<expr2tc> &side2);
int do_type_lt(
  const std::vector<type2tc> &side1,
  const std::vector<type2tc> &side2);
int do_type_lt(const expr2tc &side1, const expr2tc &side2);
int do_type_lt(const type2tc &side1, const type2tc &side2);

size_t do_type_crc(const BigInt &theint);
size_t do_type_crc(const fixedbvt &theval);
size_t do_type_crc(const ieee_floatt &theval);
size_t do_type_crc(const std::vector<expr2tc> &theval);
size_t do_type_crc(const std::vector<type2tc> &theval);
size_t do_type_crc(const std::vector<irep_idt> &theval);
size_t do_type_crc(const expr2tc &theval);
size_t do_type_crc(const type2tc &theval);
size_t do_type_crc(const irep_idt &theval);
size_t do_type_crc(const type2t::type_ids &i);
size_t do_type_crc(const expr2t::expr_ids &i);

void do_type_hash(const BigInt &theint, crypto_hash &hash);
void do_type_hash(const fixedbvt &theval, crypto_hash &hash);
void do_type_hash(const ieee_floatt &theval, crypto_hash &hash);
void do_type_hash(const std::vector<expr2tc> &theval, crypto_hash &hash);
void do_type_hash(const std::vector<type2tc> &theval, crypto_hash &hash);
void do_type_hash(const std::vector<irep_idt> &theval, crypto_hash &hash);
void do_type_hash(const expr2tc &theval, crypto_hash &hash);
void do_type_hash(const type2tc &theval, crypto_hash &hash);
void do_type_hash(const irep_idt &theval, crypto_hash &hash);
void do_type_hash(const type2t::type_ids &, crypto_hash &);
void do_type_hash(const expr2t::expr_ids &, crypto_hash &);

template <typename T>
void do_type2string(
  const T &thething,
  unsigned int idx,
  std::string (&names)[esbmct::num_type_fields],
  list_of_memberst &vec,
  unsigned int indent)
{
  vec.push_back(member_entryt(names[idx], type_to_string(thething, indent)));
}

template <class T>
bool do_get_sub_expr(const T &, size_t, size_t &, const expr2tc *&)
{
  return false;
}

template <>
bool do_get_sub_expr<expr2tc>(
  const expr2tc &item,
  size_t idx,
  size_t &it,
  const expr2tc *&ptr);

template <>
bool do_get_sub_expr<std::vector<expr2tc>>(
  const std::vector<expr2tc> &item,
  size_t idx,
  size_t &it,
  const expr2tc *&ptr);

template <class T>
size_t do_count_sub_exprs(T &)
{
  return 0;
}

template <>
size_t do_count_sub_exprs<const expr2tc>(const expr2tc &);

template <>
size_t do_count_sub_exprs<const std::vector<expr2tc>>(
  const std::vector<expr2tc> &item);

// Field-dispatch for expr2t::foreach_operand. Primary template is a no-op;
// the specialisations below match expr2tc and std::vector<expr2tc> fields.
template <typename T, typename U>
void call_expr_delegate(T &, U &)
{
}

template <>
void call_expr_delegate<const expr2tc, expr2t::const_op_delegate>(
  const expr2tc &ref,
  expr2t::const_op_delegate &f);

template <>
void call_expr_delegate<expr2tc, expr2t::op_delegate>(
  expr2tc &ref,
  expr2t::op_delegate &f);

template <>
void call_expr_delegate<const std::vector<expr2tc>, expr2t::const_op_delegate>(
  const std::vector<expr2tc> &ref,
  expr2t::const_op_delegate &f);

template <>
void call_expr_delegate<std::vector<expr2tc>, expr2t::op_delegate>(
  std::vector<expr2tc> &ref,
  expr2t::op_delegate &f);

// Same shape as call_expr_delegate, but for type2t::foreach_subtype.
template <typename T, typename U>
void call_type_delegate(T &, U &)
{
}

template <>
void call_type_delegate<const type2tc, type2t::const_subtype_delegate>(
  const type2tc &ref,
  type2t::const_subtype_delegate &f);

template <>
void call_type_delegate<type2tc, type2t::subtype_delegate>(
  type2tc &ref,
  type2t::subtype_delegate &f);

template <>
void call_type_delegate<
  const std::vector<type2tc>,
  type2t::const_subtype_delegate>(
  const std::vector<type2tc> &ref,
  type2t::const_subtype_delegate &f);

template <>
void call_type_delegate<std::vector<type2tc>, type2t::subtype_delegate>(
  std::vector<type2tc> &ref,
  type2t::subtype_delegate &f);

// ============================================================================
// generic_*<K>: switch-dispatch helpers that walk K::fields.
// ============================================================================

namespace esbmct
{
// --------------------------------------------------------------------------
// generic_cmp: structural equality over K::fields.
// Precondition: a.expr_id == o.expr_id (checked at the switch boundary).
//
// Also acts as the canonical instantiation site for the per-kind
// `assert_kind_invariants<K>()` static_assert chain: every expr dispatcher
// case lands here, so adding a new kind without a valid `fields` tuple
// fails to compile at the dispatcher, not silently at cmp/crc/hash time.
// --------------------------------------------------------------------------
template <class K>
bool generic_cmp(const K &a, const expr2t &o)
{
  static_assert(assert_kind_invariants<K>());
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
// generic_do_crc: mix expr_id then each field in K::fields. Notype kinds
// (e.g. not2t) have no type slot in K::fields; kinds with dynamic type
// list &expr2t::type first so the CRC includes it.
// --------------------------------------------------------------------------
template <class K>
size_t generic_do_crc(const K &a)
{
  if (size_t cached = a.crc_val.load(std::memory_order_acquire); cached != 0)
    return cached;
  size_t v = 0;
  hash_combine(v, do_type_crc(a.expr_id));
  std::apply(
    [&](auto... mp) { (hash_combine(v, do_type_crc(a.*mp)), ...); }, K::fields);
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
  std::apply([&](auto... mp) { (do_type_hash(a.*mp, h), ...); }, K::fields);
}

// --------------------------------------------------------------------------
// generic_tostring: build the pretty-print member list from K::fields.
// K::field_names[i] names the i-th user field (0-based). When K is an expr
// kind, any type2tc slot in K::fields is skipped — the type is shown by
// the pretty-print banner above the member list, not as a named field.
// --------------------------------------------------------------------------
template <class K>
list_of_memberst generic_tostring(const K &a, unsigned int indent)
{
  list_of_memberst vec;
  unsigned int idx = 0;
  std::apply(
    [&](auto... mp) {
      (..., ([&]() {
         using FieldT = std::remove_cvref_t<decltype(a.*mp)>;
         if constexpr (
           std::is_same_v<FieldT, type2tc> && std::is_base_of_v<expr2t, K>)
           (void)indent; // skip — shown by the type banner
         else
           do_type2string(a.*mp, idx++, K::field_names, vec, indent);
       }()));
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

// --------------------------------------------------------------------------
// Type-side generic helpers. Identical shape to the expr-side ones above
// but typed against type2t and folding type_id (instead of expr_id) into
// crc/hash. Type kinds' `fields` tuples list only user-visible fields —
// type_id is mixed in here at the head of crc/hash and is short-circuited
// by the outer switch boundary for cmp/lt.
// --------------------------------------------------------------------------

template <class K>
bool generic_cmp_type(const K &a, const type2t &o)
{
  // Canonical instantiation site for the per-kind invariant chain on the
  // type side; see generic_cmp() above for rationale.
  static_assert(assert_kind_invariants<K>());
  const K &b = static_cast<const K &>(o);
  bool eq = true;
  std::apply(
    [&](auto... mp) { ((eq = eq && do_type_cmp(a.*mp, b.*mp)), ...); },
    K::fields);
  return eq;
}

template <class K>
int generic_lt_type(const K &a, const type2t &o)
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

template <class K>
size_t generic_do_crc_type(const K &a)
{
  if (size_t cached = a.crc_val.load(std::memory_order_acquire); cached != 0)
    return cached;
  size_t v = 0;
  hash_combine(v, do_type_crc(a.type_id));
  std::apply(
    [&](auto... mp) { (hash_combine(v, do_type_crc(a.*mp)), ...); }, K::fields);
  a.crc_val.store(v, std::memory_order_release);
  return v;
}

template <class K>
void generic_hash_type(const K &a, crypto_hash &h)
{
  do_type_hash(a.type_id, h);
  std::apply([&](auto... mp) { (do_type_hash(a.*mp, h), ...); }, K::fields);
}

template <class K>
list_of_memberst generic_tostring_type(const K &a, unsigned int indent)
{
  list_of_memberst vec;
  unsigned int idx = 0;
  std::apply(
    [&](auto... mp) {
      (..., ([&]() {
         do_type2string(a.*mp, idx++, K::field_names, vec, indent);
       }()));
    },
    K::fields);
  return vec;
}

template <class K>
void generic_foreach_subtype_const(
  const K &a,
  type2t::const_subtype_delegate &f)
{
  std::apply(
    [&](auto... mp) { (call_type_delegate(a.*mp, f), ...); }, K::fields);
}

template <class K>
void generic_foreach_subtype(K &a, type2t::subtype_delegate &f)
{
  std::apply(
    [&](auto... mp) { (call_type_delegate(a.*mp, f), ...); }, K::fields);
}

} // namespace esbmct
