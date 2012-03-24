#include "irep2.h"

#include <solvers/prop/prop_conv.h>

expr2t::expr2t(const type2tc _type, expr_ids id)
  : expr_id(id), type(_type)
{
}

void expr2t::convert_smt(prop_convt &obj, void *&arg) const
{ obj.convert_smt_expr(*this, arg); }
