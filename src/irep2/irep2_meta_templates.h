#pragma once

// Out-of-line definitions of irep_methods2's templated methods.
//
// The class declaration lives in irep2.h. The bodies live here because
// they call per-field-type helpers (do_type_cmp, do_type_lt, do_type_crc,
// do_type_hash, do_type2string, do_get_sub_expr*, do_count_sub_exprs,
// call_expr_delegate, call_type_delegate) that need expr2tc / type2tc /
// data-class types declared before they can be defined — irep2.h can't
// pull them in without a cycle. Per-node templates/*.cpp files include
// this header so the helpers are visible at the point of instantiation.
//
// The implementations walk traits::fields (a std::tuple of
// field_traits<R, C, R C::*> entries) via the for_each_field helper
// defined on irep_methods2.

#include <memory>
#include <type_traits>
#include <irep2/irep2_type.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <irep2/irep2_template_utils.h>
#include <irep2/irep2_templates.h>

namespace esbmct
{

template <class derived, class baseclass, typename traits>
auto irep_methods2<derived, baseclass, traits>::clone() const
  -> base_container2tc
{
  // Single allocation via make_irep: refcount=0 from `new`, then the
  // container adopts and bumps to 1. No separate control block — the
  // count lives in the node itself.
  return make_irep<derived>(*static_cast<const derived *>(this));
}

template <class derived, class baseclass, typename traits>
list_of_memberst
irep_methods2<derived, baseclass, traits>::tostring(unsigned int indent) const
{
  const derived *self = static_cast<const derived *>(this);
  list_of_memberst vec;
  unsigned int idx = 0;
  for_each_field([&](auto field) {
    using F = decltype(field);
    using R = typename F::result_type;
    // Skip the type2tc slot on expr nodes: pretty-printing handles the
    // type banner separately and the historic field_names[] tables
    // don't allocate a slot for it.
    if constexpr (
      std::is_same_v<R, type2tc> && std::is_base_of_v<expr2t, derived>)
    {
      (void)indent;
      return;
    }
    else
    {
      do_type2string<R>(
        self->*F::value, idx, derived::field_names, vec, indent);
      ++idx;
    }
  });
  return vec;
}

template <class derived, class baseclass, typename traits>
bool irep_methods2<derived, baseclass, traits>::cmp(const base2t &ref) const
{
  const derived *self = static_cast<const derived *>(this);
  const derived *other = static_cast<const derived *>(&ref);
  bool eq = true;
  for_each_field([&](auto field) {
    if (!eq)
      return;
    using F = decltype(field);
    if (!do_type_cmp(self->*F::value, other->*F::value))
      eq = false;
  });
  return eq;
}

template <class derived, class baseclass, typename traits>
int irep_methods2<derived, baseclass, traits>::lt(const base2t &ref) const
{
  const derived *self = static_cast<const derived *>(this);
  const derived *other = static_cast<const derived *>(&ref);
  int result = 0;
  for_each_field([&](auto field) {
    if (result != 0)
      return;
    using F = decltype(field);
    result = do_type_lt(self->*F::value, other->*F::value);
  });
  return result;
}

template <class derived, class baseclass, typename traits>
size_t irep_methods2<derived, baseclass, traits>::do_crc() const
{
  // Check the cache first. The acquire load also synchronises with the
  // release store at the bottom of this function — if another reader
  // sees a non-zero value, it sees the full state that fed into it.
  if (size_t cached = this->crc_val.load(std::memory_order_acquire);
      cached != 0)
    return cached;

  const derived *self = static_cast<const derived *>(this);
  // Compute on a local so hash_combine doesn't see partial writes to
  // the cache. Publishing once at the end means concurrent readers
  // either see 0 (and recompute, getting the same value) or the final
  // value — never a half-mixed state.
  size_t v = 0;
  for_each_field([&](auto field) {
    using F = decltype(field);
    esbmct::hash_combine(v, do_type_crc(self->*F::value));
  });
  this->crc_val.store(v, std::memory_order_release);
  return v;
}

template <class derived, class baseclass, typename traits>
void irep_methods2<derived, baseclass, traits>::hash(crypto_hash &h) const
{
  const derived *self = static_cast<const derived *>(this);
  for_each_field([&](auto field) {
    using F = decltype(field);
    do_type_hash(self->*F::value, h);
  });
}

template <class derived, class baseclass, typename traits>
const expr2tc *
irep_methods2<derived, baseclass, traits>::get_sub_expr_impl(size_t desired)
  const
{
  const derived *self = static_cast<const derived *>(this);
  const expr2tc *result = nullptr;
  size_t cur = 0;
  for_each_field([&](auto field) {
    if (result)
      return;
    using F = decltype(field);
    using R = typename F::result_type;
    if constexpr (
      std::is_same_v<R, expr2tc> ||
      std::is_same_v<R, std::vector<expr2tc>>)
    {
      const expr2tc *ptr = nullptr;
      if (do_get_sub_expr(self->*F::value, desired, cur, ptr))
        result = ptr;
    }
  });
  return result;
}

template <class derived, class baseclass, typename traits>
expr2tc *
irep_methods2<derived, baseclass, traits>::get_sub_expr_nc_impl(size_t desired)
{
  derived *self = static_cast<derived *>(this);
  expr2tc *result = nullptr;
  size_t cur = 0;
  for_each_field([&](auto field) {
    if (result)
      return;
    using F = decltype(field);
    using R = typename F::result_type;
    if constexpr (
      std::is_same_v<R, expr2tc> ||
      std::is_same_v<R, std::vector<expr2tc>>)
    {
      expr2tc *ptr = nullptr;
      if (do_get_sub_expr_nc(self->*F::value, desired, cur, ptr))
        result = ptr;
    }
  });
  return result;
}

template <class derived, class baseclass, typename traits>
size_t irep_methods2<derived, baseclass, traits>::get_num_sub_exprs_impl() const
{
  const derived *self = static_cast<const derived *>(this);
  size_t total = 0;
  for_each_field([&](auto field) {
    using F = decltype(field);
    total += do_count_sub_exprs(self->*F::value);
  });
  return total;
}

template <class derived, class baseclass, typename traits>
template <typename Delegate>
void irep_methods2<derived, baseclass, traits>::
  foreach_operand_impl_const_inner(Delegate &f) const
{
  const derived *self = static_cast<const derived *>(this);
  for_each_field([&](auto field) {
    using F = decltype(field);
    call_expr_delegate(self->*F::value, f);
  });
}

template <class derived, class baseclass, typename traits>
template <typename Delegate>
void irep_methods2<derived, baseclass, traits>::foreach_operand_impl_inner(
  Delegate &f)
{
  derived *self = static_cast<derived *>(this);
  for_each_field([&](auto field) {
    using F = decltype(field);
    call_expr_delegate(self->*F::value, f);
  });
}

template <class derived, class baseclass, typename traits>
template <typename Delegate>
void irep_methods2<derived, baseclass, traits>::
  foreach_subtype_impl_const_inner(Delegate &f) const
{
  const derived *self = static_cast<const derived *>(this);
  for_each_field([&](auto field) {
    using F = decltype(field);
    call_type_delegate(self->*F::value, f);
  });
}

template <class derived, class baseclass, typename traits>
template <typename Delegate>
void irep_methods2<derived, baseclass, traits>::foreach_subtype_impl_inner(
  Delegate &f)
{
  derived *self = static_cast<derived *>(this);
  for_each_field([&](auto field) {
    using F = decltype(field);
    call_type_delegate(self->*F::value, f);
  });
}

} // namespace esbmct
