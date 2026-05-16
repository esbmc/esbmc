#include <irep2/irep2_templates.h>

std::string indent_str_irep2(unsigned int indent)
{
  return std::string(indent, ' ');
}

// For CRCing to actually be accurate, expr/type ids mustn't overflow out of
// a byte. If this happens then a) there are too many exprs, and b) the expr
// crcing code has to change.
static_assert(type2t::end_type_id <= 256, "Type id overflow");
static_assert(expr2t::end_expr_id <= 256, "Expr id overflow");

template <>
void do_type2string<type2t::type_ids>(
  const type2t::type_ids &,
  unsigned int,
  std::string (&)[esbmct::num_type_fields],
  list_of_memberst &,
  unsigned int)
{
  // Do nothing; this is a dummy member.
}

template <>
void do_type2string<const expr2t::expr_ids>(
  const expr2t::expr_ids &,
  unsigned int,
  std::string (&)[esbmct::num_type_fields],
  list_of_memberst &,
  unsigned int)
{
  // Do nothing; this is a dummy member.
}

template <>
bool do_get_sub_expr<expr2tc>(
  const expr2tc &item,
  size_t idx,
  size_t &it,
  const expr2tc *&ptr)
{
  if (idx == it)
  {
    ptr = &item;
    return true;
  }
  else
  {
    it++;
    return false;
  }
}

template <>
bool do_get_sub_expr<std::vector<expr2tc>>(
  const std::vector<expr2tc> &item,
  size_t idx,
  size_t &it,
  const expr2tc *&ptr)
{
  if (idx < it + item.size())
  {
    ptr = &item[idx - it];
    return true;
  }
  else
  {
    it += item.size();
    return false;
  }
}

// Non-const versions of the above.
template <>
bool do_get_sub_expr_nc<expr2tc>(
  expr2tc &item,
  size_t idx,
  size_t &it,
  expr2tc *&ptr)
{
  if (idx == it)
  {
    ptr = &item;
    return true;
  }
  else
  {
    it++;
    return false;
  }
}

template <>
bool do_get_sub_expr_nc<std::vector<expr2tc>>(
  std::vector<expr2tc> &item,
  size_t idx,
  size_t &it,
  expr2tc *&ptr)
{
  if (idx < it + item.size())
  {
    ptr = &item[idx - it];
    return true;
  }
  else
  {
    it += item.size();
    return false;
  }
}

template <>
size_t do_count_sub_exprs<const expr2tc>(const expr2tc &)
{
  return 1;
}

template <>
size_t
do_count_sub_exprs<const std::vector<expr2tc>>(const std::vector<expr2tc> &item)
{
  return item.size();
}

template <>
void call_expr_delegate<const expr2tc, expr2t::const_op_delegate>(
  const expr2tc &ref,
  expr2t::const_op_delegate &f)
{
  f(ref);
}

template <>
void call_expr_delegate<expr2tc, expr2t::op_delegate>(
  expr2tc &ref,
  expr2t::op_delegate &f)
{
  f(ref);
}

template <>
void call_expr_delegate<const std::vector<expr2tc>, expr2t::const_op_delegate>(
  const std::vector<expr2tc> &ref,
  expr2t::const_op_delegate &f)
{
  for (const expr2tc &r : ref)
    f(r);
}

template <>
void call_expr_delegate<std::vector<expr2tc>, expr2t::op_delegate>(
  std::vector<expr2tc> &ref,
  expr2t::op_delegate &f)
{
  for (expr2tc &r : ref)
    f(r);
}

template <>
void call_type_delegate<const type2tc, type2t::const_subtype_delegate>(
  const type2tc &ref,
  type2t::const_subtype_delegate &f)
{
  f(ref);
}

template <>
void call_type_delegate<type2tc, type2t::subtype_delegate>(
  type2tc &ref,
  type2t::subtype_delegate &f)
{
  f(ref);
}

template <>
void call_type_delegate<
  const std::vector<type2tc>,
  type2t::const_subtype_delegate>(
  const std::vector<type2tc> &ref,
  type2t::const_subtype_delegate &f)
{
  for (const type2tc &r : ref)
    f(r);
}

template <>
void call_type_delegate<std::vector<type2tc>, type2t::subtype_delegate>(
  std::vector<type2tc> &ref,
  type2t::subtype_delegate &f)
{
  for (type2tc &r : ref)
    f(r);
}
