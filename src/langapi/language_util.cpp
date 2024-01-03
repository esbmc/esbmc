#include <langapi/language_util.h>
#include <langapi/mode.h>
#include <memory>
#include <util/message.h>

std::unique_ptr<languaget> language_from_symbol(const symbolt &symbol)
{
  language_idt lang = language_idt::C;
  if (symbol.mode != "")
    lang = language_id_by_name(id2string(symbol.mode));

  if (lang != language_idt::NONE)
    return std::unique_ptr<languaget>(new_language(lang));

  log_error("symbol '{}' has unknown mode '{}'", symbol.name, symbol.mode);
  abort();
}

static std::unique_ptr<languaget>
language_from_symbol_id(const namespacet &ns, const irep_idt &id)
{
  if (id != "")
  {
    const symbolt *s = ns.lookup(id);
    if (s)
      return language_from_symbol(*s);
  }
  return std::unique_ptr<languaget>(new_language(language_idt::C));
}

std::string from_expr(
  const namespacet &ns,
  const irep_idt &identifier,
  const exprt &expr,
  presentationt target)
{
  std::unique_ptr<languaget> p = language_from_symbol_id(ns, identifier);
  std::string result;
  p->from_expr(expr, result, ns, target);
  return result;
}

std::string from_type(
  const namespacet &ns,
  const irep_idt &identifier,
  const typet &type,
  presentationt target)
{
  std::unique_ptr<languaget> p = language_from_symbol_id(ns, identifier);
  std::string result;
  p->from_type(type, result, ns, target);
  return result;
}

std::string from_expr(const exprt &expr)
{
  contextt context;
  return from_expr(namespacet(context), "", expr);
}

std::string from_type(const typet &type)
{
  contextt context;
  return from_type(namespacet(context), "", type);
}
