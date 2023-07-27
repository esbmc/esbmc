#ifndef CPROVER_LANGUAGES_H
#define CPROVER_LANGUAGES_H

#include <langapi/mode.h>
#include <util/language.h>
#include <util/migrate.h>
class languagest
{
public:
  // conversion of expressions

  bool from_expr(const exprt &expr, std::string &code)
  {
    return language->from_expr(expr, code, ns);
  }

  bool from_expr(const expr2tc &expr, std::string &code)
  {
    return language->from_expr(migrate_expr_back(expr), code, ns);
  }

  bool from_type(const typet &type, std::string &code)
  {
    return language->from_type(type, code, ns);
  }

  // constructor / destructor

  languagest(const namespacet &_ns, language_idt lang);
  virtual ~languagest();

protected:
  const namespacet &ns;
  languaget *language;
};

#endif
