/*******************************************************************\

Module: Obtain the pretty name of a given code

Author: Lucas Cordeiro, lucas.cordeiro@manchester.ac.uk

\*******************************************************************/

#ifndef CPROVER_UTIL_PRETTY_H
#define CPROVER_UTIL_PRETTY_H

#include <string>

#include <util/irep2.h>
#include <util/irep2_expr.h>

inline std::string get_pretty_name(const expr2tc code)
{
  return to_code_decl2t(code).value.as_string().substr(
    to_code_decl2t(code).value.as_string().find_last_of('@') + 1);
}

#endif // CPROVER_UTIL_PRETTY_H
