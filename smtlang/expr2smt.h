/*******************************************************************\

Module: SMT-LIB Frontend, expression conversion

Author: CM Wintersteiger

\*******************************************************************/

#ifndef EXPR2SMT_H_
#define EXPR2SMT_H_

#include "expr.h"

bool expr2smt(const exprt &expr, std::string &code);
bool type2smt(const typet &type, std::string &code);

std::string ii2string(const irept &ident);

#endif /*EXPR2SMT_H_*/
