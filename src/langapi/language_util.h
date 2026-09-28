#ifndef CPROVER_LANGUAGE_UTIL_H
#define CPROVER_LANGUAGE_UTIL_H

#include <irep2/irep2.h>
#include <util/lang/language.h>
#include <util/irep/migrate.h>
#include <util/symtab/namespace.h>
#include <util/symtab/symbol.h>

std::unique_ptr<languaget> language_from_symbol(const symbolt &symbol);

/// The language to render expressions in when there is no symbol to take a mode
/// from -- VCC dumps, SSA dumps and witness generation. Uses whatever the
/// frontend parsed, falling back to C, rather than always C: dumping C++
/// through expr2c prints references as pointers (esbmc/esbmc#782).
language_idt configured_language();

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
