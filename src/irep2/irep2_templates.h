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
#include <util/migrate.h>
#include <util/std_types.h>

template <typename T>
class register_irep_methods;

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

template <>
void do_type2string<type2t::type_ids>(
  const type2t::type_ids &,
  unsigned int,
  std::string (&)[esbmct::num_type_fields],
  list_of_memberst &,
  unsigned int);

template <>
void do_type2string<const expr2t::expr_ids>(
  const expr2t::expr_ids &,
  unsigned int,
  std::string (&)[esbmct::num_type_fields],
  list_of_memberst &,
  unsigned int);

template <class T>
bool do_get_sub_expr(const T &, unsigned int, unsigned int &, const expr2tc *&)
{
  return false;
}

template <>
bool do_get_sub_expr<expr2tc>(
  const expr2tc &item,
  unsigned int idx,
  unsigned int &it,
  const expr2tc *&ptr);

template <>
bool do_get_sub_expr<std::vector<expr2tc>>(
  const std::vector<expr2tc> &item,
  unsigned int idx,
  unsigned int &it,
  const expr2tc *&ptr);

template <class T>
bool do_get_sub_expr_nc(T &, unsigned int, unsigned int &, expr2tc *&)
{
  return false;
}

// Non-const versions of the above.
template <>
bool do_get_sub_expr_nc<expr2tc>(
  expr2tc &item,
  unsigned int idx,
  unsigned int &it,
  expr2tc *&ptr);

template <>
bool do_get_sub_expr_nc<std::vector<expr2tc>>(
  std::vector<expr2tc> &item,
  unsigned int idx,
  unsigned int &it,
  expr2tc *&ptr);

template <class T>
unsigned int do_count_sub_exprs(T &)
{
  return 0;
}

template <>
unsigned int do_count_sub_exprs<const expr2tc>(const expr2tc &);

template <>
unsigned int do_count_sub_exprs<const std::vector<expr2tc>>(
  const std::vector<expr2tc> &item);

// Local template for implementing delegate calling, with type dependency.
// Can't easily extend to cover types because field type is _already_ abstracted
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

// Repeat of call_expr_delegate, but for types
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