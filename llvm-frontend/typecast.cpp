/*
 * typecast.cpp
 *
 *  Created on: Sep 11, 2015
 *      Author: mramalho
 */

#include "typecast.h"

#include <simplify_expr_class.h>
#include <ansi-c/c_typecast.h>

void gen_typecast(
  namespacet ns,
  exprt &dest,
  typet type)
{
  if(dest.type() != type)
  {
    c_typecastt c_typecast(ns);
    c_typecast.implicit_typecast(dest, type);
  }
}
