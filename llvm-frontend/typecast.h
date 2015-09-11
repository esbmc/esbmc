/*
 * typecast.h
 *
 *  Created on: Sep 11, 2015
 *      Author: mramalho
 */

#ifndef LLVM_FRONTEND_TYPECAST_H_
#define LLVM_FRONTEND_TYPECAST_H_

#include <std_expr.h>

extern void gen_typecast(
  exprt &expr,
  typet type);

#endif /* LLVM_FRONTEND_TYPECAST_H_ */
