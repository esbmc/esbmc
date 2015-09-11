/*
 * typecast.cpp
 *
 *  Created on: Sep 11, 2015
 *      Author: mramalho
 */

#include "typecast.h"

#include <simplify_expr_class.h>

void gen_typecast(
  exprt &expr,
  typet type)
{
  if(expr.type() != type)
  {
    exprt new_expr = typecast_exprt(expr, type);

    simplify_exprt simplify;
    simplify.simplify(new_expr);

    new_expr.location() = expr.location();
    expr.swap(new_expr);
  }
}
