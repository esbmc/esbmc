#include "irep2.h"

#include <solvers/prop/prop_conv.h>

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

template <class derived>
void
expr_body<derived>::convert_smt(prop_convt &obj, void *&arg) const
{
  derived *new_this = static_cast<derived>(this);
  obj.convert_smt_expr(*new_this, arg);
  return;
}

template <class derived>
expr2tc
expr_body<derived>::clone(void) const
{
  derived *new_obj = new derived(*static_cast<derived>(this));
  return expr2tc(new_obj);
}

symbol2t::symbol2t(const type2tc type, irep_idt _name)
  : expr2t(type, symbol_id),
    name(_name)
{
}

symbol2t::symbol2t(const symbol2t &ref)
  : expr2t(ref),
    name(ref.name)
{
}

expr2tc symbol2t::clone(void) const
{ return expr2tc(new symbol2t(*this)); }

void symbol2t::convert_smt(prop_convt &obj, void *&arg) const
{ obj.convert_smt_expr(*this, arg); }
