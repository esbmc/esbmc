#include <util/goto_expr_factory.h>

constant_int2tc create_value_expr(int value, type2tc type)
{
  BigInt num(value);
  constant_int2tc expression(type, num);
  return expression;
}

lessthanequal2tc create_lessthanequal_relation(expr2tc &lhs, expr2tc &rhs)
{
  typecast2tc lhs_typecast(lhs->type, lhs);
  typecast2tc rhs_typecast(lhs->type, rhs);

  lessthanequal2tc relation(lhs_typecast, rhs_typecast);
  return relation;
}

greaterthanequal2tc create_greaterthanequal_relation(expr2tc &lhs, expr2tc &rhs)
{
  typecast2tc lhs_typecast(lhs->type, lhs);
  typecast2tc rhs_typecast(lhs->type, rhs);

  greaterthanequal2tc relation(lhs_typecast, rhs_typecast);
  return relation;
}
