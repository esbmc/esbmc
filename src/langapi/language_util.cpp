#include <langapi/language_util.h>
#include <langapi/mode.h>
#include <memory>
#include <util/message.h>

static int mode_from_symbol(const symbolt *symbol)
{
  if(!symbol)
    return 0;

  if(symbol->mode == "")
    return 0;

  if(int mode = get_mode(id2string(symbol->mode)); mode >= 0)
    return mode;

  log_error("symbol '{}' has unknown mode '{}'", symbol->name, symbol->mode);
  abort();
}

static std::unique_ptr<languaget>
language_from_symbol_id(const namespacet &ns, const irep_idt &id)
{
  int mode = id == "" ? 0 : mode_from_symbol(ns.lookup(id));
  return std::unique_ptr<languaget>(mode_table[mode].new_language());
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
