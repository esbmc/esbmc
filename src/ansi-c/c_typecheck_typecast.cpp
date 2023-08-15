/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/c_typecheck_base.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/expr_util.h>

void c_typecheck_baset::implicit_typecast(exprt &expr, const typet &type)
{
  c_typecastt c_typecast(*this);

  typet original_expr_type(expr.type());

  c_typecast.implicit_typecast(expr, type);

  for(std::list<std::string>::const_iterator it = c_typecast.errors.begin();
      it != c_typecast.errors.end();
      it++)
  {
    err_location(expr);
    str << "conversion from `" << to_string(original_expr_type) << "' to `"
        << to_string(type) << "': " << *it;
    log_error("{}", str.str());
    abort();
  }

  if(!c_typecast.errors.empty())
    throw 0; // give up

  for(std::list<std::string>::const_iterator it = c_typecast.warnings.begin();
      it != c_typecast.warnings.end();
      it++)
  {
    err_location(expr);
    str << "warning: conversion from `" << to_string(original_expr_type)
        << "' to `" << to_string(type) << "': " << *it;
    log_warning("{}", str.str());
  }
}

void c_typecheck_baset::implicit_typecast_arithmetic(exprt &expr1, exprt &expr2)
{
  c_typecastt c_typecast(*this);
  c_typecast.implicit_typecast_arithmetic(expr1, expr2);
}

void c_typecheck_baset::implicit_typecast_arithmetic(exprt &expr)
{
  c_typecastt c_typecast(*this);
  c_typecast.implicit_typecast_arithmetic(expr);
}
