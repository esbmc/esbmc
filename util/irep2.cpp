#include "irep2.h"

#include <solvers/prop/prop_conv.h>

/*************************** Base type2t definitions **************************/

type2t::type2t(type_ids id)
  : type_id(id)
{
}

template<class derived>
void
type_body<derived>::convert_smt_type(prop_convt &obj, void *&arg) const
{
  const derived *derived_this = static_cast<const derived *>(this);
  obj.convert_smt_type(*derived_this, arg);
}

bv_type2t::bv_type2t(type2t::type_ids id, unsigned int _width)
  : type_body<bv_type2t>(id),
    width(_width)
{
}

unsigned int
bv_type2t::get_width(void) const
{
  return width;
}

bool_type2t::bool_type2t(void)
  : type_body<bool_type2t>(bool_id)
{
}

unsigned int
bool_type2t::get_width(void) const
{
  return 1;
}

signedbv_type2t::signedbv_type2t(unsigned int width)
  : type_body<signedbv_type2t>(signedbv_id, width)
{
}

unsignedbv_type2t::unsignedbv_type2t(unsigned int width)
  : type_body<unsignedbv_type2t>(unsignedbv_id, width)
{
}

array_type2t::array_type2t(const type2tc t, const expr2tc s, bool inf)
  : type_body<array_type2t>(array_id), subtype(t), array_size(s),
    size_is_infinite(inf)
{
}

unsigned int
array_type2t::get_width(void) const
{
  // Two edge cases: the array can have infinite size, or it can have a dynamic
  // size that's determined by the solver.
  if (size_is_infinite)
    throw new inf_sized_array_excp();

  if (array_size->expr_id != expr2t::constant_int_id)
    throw new dyn_sized_array_excp();

  // Otherwise, we can multiply the size of the subtype by the number of elements.
  unsigned int sub_width = subtype->get_width();

  expr2t *elem_size = array_size.get();
  constant_int2t *const_elem_size = dynamic_cast<constant_int2t*>(elem_size);
  assert(const_elem_size != NULL);
  unsigned int num_elems = const_elem_size->as_uint();

  return num_elems * sub_width;
}

pointer_type2t::pointer_type2t(type2tc _sub)
  : type_body<pointer_type2t>(pointer_id), subtype(_sub)
{
}

unsigned int
pointer_type2t::get_width(void) const
{
  return config.ansi_c.pointer_width;
}

/*************************** Base expr2t definitions **************************/

expr2t::expr2t(const type2tc _type, expr_ids id)
  : expr_id(id), type(_type)
{
}

expr2t::expr2t(const expr2t &ref)
  : expr_id(ref.expr_id),
    type(ref.type)
{
}

void expr2t::convert_smt(prop_convt &obj, void *&arg) const
{ obj.convert_smt_expr(*this, arg); }

/***************************** Templated expr body ****************************/

template <class derived>
expr_body<derived>::expr_body(const expr_body<derived> &ref)
  : expr2t(ref)
{
}

template <class derived>
void
expr_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  const derived *new_this = static_cast<const derived*>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc
expr_body<derived>::clone(void) const
{
  const derived *derived_this = static_cast<const derived*>(this);
  derived *new_obj = new derived(*derived_this);
  return expr2tc(new_obj);
}

/**************************** Expression constructors *************************/

symbol2t::symbol2t(const type2tc type, irep_idt _name)
  : expr_body<symbol2t>(type, symbol_id),
    name(_name)
{
}

symbol2t::symbol2t(const symbol2t &ref)
  : expr_body<symbol2t>(ref),
    name(ref.name)
{
}

unsigned int
constant_int2t::as_uint(void) const
{
  // XXXjmorse - add assertion that we don't exceed machine word width?
  return constant_value.to_ulong();
}
