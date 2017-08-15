/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_LANGUAGE_UTIL_H
#define CPROVER_LANGUAGE_UTIL_H

#include <util/irep2.h>
#include <util/language.h>
#include <util/migrate.h>
#include <util/namespace.h>

std::string from_expr(
  const namespacet &ns,
  const irep_idt &identifier,
  const exprt &expr,
  const bool fullname = false);

inline std::string from_expr(
  const namespacet &ns,
  const irep_idt &identifier,
  const expr2tc &expr,
  const bool fullname = false)
{
  return from_expr(ns, identifier, migrate_expr_back(expr), fullname);
}

std::string from_expr(const exprt &expr);

inline std::string from_expr(const expr2tc &expr)
{
  return from_expr(migrate_expr_back(expr));
}

std::string from_type(
  const namespacet &ns,
  const irep_idt &identifier,
  const typet &type,
  const bool fullname = false);

inline std::string from_type(
  const namespacet &ns,
  const irep_idt &identifier,
  const type2tc &type,
  const bool fullname = false)
{
  return from_type(ns, identifier, migrate_type_back(type), fullname);
}

std::string from_type(const typet &type);

inline std::string from_type(const type2tc &type)
{
  return from_type(migrate_type_back(type));
}

#endif
