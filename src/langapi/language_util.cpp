#include <langapi/language_util.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <util/message.h>

static language_idt language_id_from_mode(irep_idt mode)
{
  return mode.empty() ? language_idt::C : language_id_by_name(id2string(mode));
}

std::unique_ptr<languaget> language_from_symbol(const symbolt &symbol)
{
  language_idt lang = language_id_from_mode(symbol.mode);
  if (lang != language_idt::NONE)
    return new_language(lang);

  log_error("symbol '{}' has unknown mode '{}'", symbol.name, symbol.mode);
  abort();
}

static languagest
languages_from_symbol_id(const namespacet &ns, const irep_idt &id)
{
  language_idt lang = language_idt::C;
  if (!id.empty())
    if (const symbolt *s = ns.lookup(id))
      lang = language_id_from_mode(s->mode);
  return languagest(ns, lang);
}

std::string from_expr(
  const namespacet &ns,
  const irep_idt &identifier,
  const exprt &expr,
  presentationt target)
{
  languagest langs = languages_from_symbol_id(ns, identifier);
  std::string result;
  langs.from_expr(expr, result, target);
  return result;
}

std::string from_type(
  const namespacet &ns,
  const irep_idt &identifier,
  const typet &type,
  presentationt target)
{
  languagest langs = languages_from_symbol_id(ns, identifier);
  std::string result;
  langs.from_type(type, result, target);
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
