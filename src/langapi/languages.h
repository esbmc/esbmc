#ifndef CPROVER_LANGUAGES_H
#define CPROVER_LANGUAGES_H

#include <langapi/mode.h>
#include <util/language.h>

class languagest
{
public:
  // conversion of expressions

  bool from_expr(
    const exprt &expr,
    std::string &code,
    presentationt target = presentationt::HUMAN)
  {
    return language->from_expr(expr, code, ns, target);
  }

  bool from_type(
    const typet &type,
    std::string &code,
    presentationt target = presentationt::HUMAN)
  {
    return language->from_type(type, code, ns, target);
  }

  // constructor / destructor

  languagest(const namespacet &_ns, language_idt lang);
  virtual ~languagest() = default;

protected:
  const namespacet &ns;
  std::unique_ptr<languaget> language;
};

#endif
