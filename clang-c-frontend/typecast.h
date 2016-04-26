/*
 * typecast.h
 *
 *  Created on: Sep 11, 2015
 *      Author: mramalho
 */

#ifndef TYPECAST_H_
#define TYPECAST_H_

#include <std_expr.h>
#include <namespace.h>

extern void gen_typecast(
  namespacet ns,
  exprt &dest,
  typet type);

extern void gen_typecast_bool(
  namespacet ns,
  exprt &dest);

extern void gen_typecast_arithmetic(
  namespacet ns,
  exprt &expr1,
  exprt &expr2);

extern void gen_typecast_arithmetic(
  namespacet ns,
  exprt &expr);

#endif /* TYPECAST_H_ */
