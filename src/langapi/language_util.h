#ifndef CPROVER_LANGUAGE_UTIL_H
#define CPROVER_LANGUAGE_UTIL_H

#include <irep2/irep2.h>
#include <util/language.h>
#include <util/migrate.h>
#include <util/namespace.h>
#include <util/symbol.h>

std::unique_ptr<languaget> language_from_symbol(const symbolt &symbol);

std::string from_expr(
  const namespacet &ns,
  const irep_idt &identifier,
  const exprt &exprt,
  presentationt target = presentationt::HUMAN);

inline std::string from_expr(
  const namespacet &ns,
  const irep_idt &identifier,
  const expr2tc &expr,
  presentationt target = presentationt::HUMAN)
{
  return from_expr(ns, identifier, migrate_expr_back(expr), target);
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
  presentationt target = presentationt::HUMAN);

inline std::string from_type(
  const namespacet &ns,
  const irep_idt &identifier,
  const type2tc &type,
  presentationt target = presentationt::HUMAN)
{
  return from_type(ns, identifier, migrate_type_back(type), target);
}

std::string from_type(const typet &type);

inline std::string from_type(const type2tc &type)
{
  return from_type(migrate_type_back(type));
}

#endif
