#include <util/goto_expr_factory.h>

expr2tc create_value_expr(int value, const type2tc &type)
{
  return constant_int2tc(type, BigInt(value));
}

expr2tc create_lessthanequal_relation(expr2tc &lhs, expr2tc &rhs)
{
  expr2tc lhs_typecast = typecast2tc(lhs->type, lhs);
  expr2tc rhs_typecast = typecast2tc(lhs->type, rhs);

  return lessthanequal2tc(lhs_typecast, rhs_typecast);
}

expr2tc create_greaterthanequal_relation(expr2tc &lhs, expr2tc &rhs)
{
  expr2tc lhs_typecast = typecast2tc(lhs->type, lhs);
  expr2tc rhs_typecast = typecast2tc(lhs->type, rhs);

  return greaterthanequal2tc(lhs_typecast, rhs_typecast);
}
