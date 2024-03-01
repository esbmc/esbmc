#pragma once

#include <memory>
#include <boost/functional/hash.hpp>
#include <util/fixedbv.h>
#include <util/i2string.h>
#include <util/ieee_float.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <irep2/irep2_template_utils.h>
#include <irep2/irep2_templates.h>
#include <util/migrate.h>
#include <util/std_types.h>

/************************ Second attempt at irep templates ********************/

// Implementations of common methods, recursively.

// Top level type method definition (above recursive def)
// exprs

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
const expr2tc *
esbmct::expr_methods2<derived, baseclass, traits, enable, fields>::get_sub_expr(
  unsigned int i) const
{
  return superclass::get_sub_expr_rec(0, i); // Skips expr_id
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
expr2tc *esbmct::expr_methods2<derived, baseclass, traits, enable, fields>::
  get_sub_expr_nc(unsigned int i)
{
  return superclass::get_sub_expr_nc_rec(0, i); // Skips expr_id
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
unsigned int esbmct::expr_methods2<derived, baseclass, traits, enable, fields>::
  get_num_sub_exprs() const
{
  return superclass::get_num_sub_exprs_rec(); // Skips expr_id
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::expr_methods2<derived, baseclass, traits, enable, fields>::
  foreach_operand_impl_const(expr2t::const_op_delegate &f) const
{
  superclass::foreach_operand_impl_const_rec(f);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::expr_methods2<derived, baseclass, traits, enable, fields>::
  foreach_operand_impl(expr2t::op_delegate &f)
{
  superclass::foreach_operand_impl_rec(f);
}

// Types

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::type_methods2<derived, baseclass, traits, enable, fields>::
  foreach_subtype_impl_const(type2t::const_subtype_delegate &f) const
{
  superclass::foreach_subtype_impl_const_rec(f);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::type_methods2<derived, baseclass, traits, enable, fields>::
  foreach_subtype_impl(type2t::subtype_delegate &f)
{
  superclass::foreach_subtype_impl_rec(f);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
auto esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::clone()
  const -> base_container2tc
{
  const derived *derived_this = static_cast<const derived *>(this);
  // Use std::make_shared to clone this with one allocation, it puts the ref
  // counting block ahead of the data object itself. This necessitates making
  // a bare std::shared_ptr first, and then feeding that into an expr2tc
  // container.
  // Generally, storing an irep in a bare std::shared_ptr loses the detach
  // facility and breaks everything, this is an exception.
  return base_container2tc(ksptr::make_shared<derived>(*derived_this));
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
list_of_memberst
esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::tostring(
  unsigned int indent) const
{
  list_of_memberst thevector;

  superclass::tostring_rec(0, thevector, indent); // Skips type_id / expr_id
  return thevector;
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
bool esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::cmp(
  const base2t &ref) const
{
  return cmp_rec(ref); // _includes_ type_id / expr_id
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
int esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::lt(
  const base2t &ref) const
{
  return lt_rec(ref); // _includes_ type_id / expr_id
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
size_t
esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::do_crc()
  const
{
  if (this->crc_val != 0)
    return this->crc_val;

  // Starting from 0, pass a crc value through all the sub-fields of this
  // expression. Store it into crc_val.
  assert(this->crc_val == 0);

  do_crc_rec(); // _includes_ type_id / expr_id

  // Finally, combine the crc of this expr with the input , and return
  return this->crc_val;
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::hash(
  crypto_hash &hash) const
{
  hash_rec(hash); // _includes_ type_id / expr_id
}

// The, *actual* recursive defs

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::
  tostring_rec(unsigned int idx, list_of_memberst &vec, unsigned int indent)
    const
{
  // Skip over type fields in expressions. Alas, this is a design oversight,
  // without this we would screw up the field name list.
  // It escapes me why this isn't printed here anyway, it gets printed in the
  // end.
  if (
    std::is_same<cur_type, type2tc>::value &&
    std::is_base_of<expr2t, derived>::value)
  {
    superclass::tostring_rec(idx, vec, indent);
    return;
  }

  // Insert our particular member to string list.
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;
  do_type2string<cur_type>(
    derived_this->*m_ptr, idx, derived_this->field_names, vec, indent);

  // Recurse
  superclass::tostring_rec(idx + 1, vec, indent);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
bool esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::cmp_rec(
  const base2t &ref) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);
  auto m_ptr = membr_ptr::value;

  if (!do_type_cmp(derived_this->*m_ptr, ref2->*m_ptr))
    return false;

  return superclass::cmp_rec(ref);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
int esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::lt_rec(
  const base2t &ref) const
{
  int tmp;
  const derived *derived_this = static_cast<const derived *>(this);
  const derived *ref2 = static_cast<const derived *>(&ref);
  auto m_ptr = membr_ptr::value;

  tmp = do_type_lt(derived_this->*m_ptr, ref2->*m_ptr);
  if (tmp != 0)
    return tmp;

  return superclass::lt_rec(ref);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::
  do_crc_rec() const
{
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;

  size_t tmp = do_type_crc(derived_this->*m_ptr);
  boost::hash_combine(this->crc_val, tmp);

  superclass::do_crc_rec();
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::
  hash_rec(crypto_hash &hash) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;
  do_type_hash(derived_this->*m_ptr, hash);

  superclass::hash_rec(hash);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
const expr2tc *
esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::
  get_sub_expr_rec(unsigned int cur_idx, unsigned int desired) const
{
  const expr2tc *ptr;
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;

  // XXX -- this takes a _reference_ to cur_idx, and maybe modifies.
  if (do_get_sub_expr(derived_this->*m_ptr, desired, cur_idx, ptr))
    return ptr;

  return superclass::get_sub_expr_rec(cur_idx, desired);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
expr2tc *esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::
  get_sub_expr_nc_rec(unsigned int cur_idx, unsigned int desired)
{
  expr2tc *ptr;
  derived *derived_this = static_cast<derived *>(this);
  auto m_ptr = membr_ptr::value;

  // XXX -- this takes a _reference_ to cur_idx, and maybe modifies.
  if (do_get_sub_expr_nc(derived_this->*m_ptr, desired, cur_idx, ptr))
    return ptr;

  return superclass::get_sub_expr_nc_rec(cur_idx, desired);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
unsigned int esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::
  get_num_sub_exprs_rec() const
{
  unsigned int num = 0;
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;

  num = do_count_sub_exprs(derived_this->*m_ptr);
  return num + superclass::get_num_sub_exprs_rec();
}

// Operand iteration specialized for expr2tc: call delegate.
template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::
  foreach_operand_impl_const_rec(expr2t::const_op_delegate &f) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_expr_delegate(derived_this->*m_ptr, f);

  superclass::foreach_operand_impl_const_rec(f);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::
  foreach_operand_impl_rec(expr2t::op_delegate &f)
{
  derived *derived_this = static_cast<derived *>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_expr_delegate(derived_this->*m_ptr, f);

  superclass::foreach_operand_impl_rec(f);
}

// Misery cakes to add readwrite modifier for non-const fields.
namespace esbmct
{
template <class Reg, class targfield>
struct magical_mystery_modifier
{
public:
  void operator()(Reg &, const char *, targfield)
  {
  }
};

template <class Reg, class foo, class bar>
struct magical_mystery_modifier<Reg, foo bar::*>
{
public:
  void operator()(
    Reg &reg,
    const char *fname,
    foo bar::*baz,
    typename boost::disable_if<typename boost::is_const<foo>::type, bool>::type
      qux = false)
  {
    (void)qux;
    reg.def_readwrite(fname, baz);
  }
};

template <class Reg, class foo, class bar>
struct magical_mystery_modifier<Reg, const foo bar::*>
{
public:
  void operator()(Reg &, const char *, const foo bar::*)
  {
  }
};
} // namespace esbmct

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::
  foreach_subtype_impl_const_rec(type2t::const_subtype_delegate &f) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_type_delegate(derived_this->*m_ptr, f);

  superclass::foreach_subtype_impl_const_rec(f);
}

template <
  class derived,
  class baseclass,
  typename traits,
  typename enable,
  typename fields>
void esbmct::irep_methods2<derived, baseclass, traits, enable, fields>::
  foreach_subtype_impl_rec(type2t::subtype_delegate &f)
{
  derived *derived_this = static_cast<derived *>(this);
  auto m_ptr = membr_ptr::value;

  // Call delegate
  call_type_delegate(derived_this->*m_ptr, f);

  superclass::foreach_subtype_impl_rec(f);
}
