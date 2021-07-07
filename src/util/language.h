/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_LANGUAGE_H
#define CPROVER_LANGUAGE_H

#include <cstdio>
#include <set>
#include <util/context.h>
#include <util/message/message.h>
#include <util/namespace.h>

class languaget
{
public:
  // parse file

  virtual bool parse(const std::string &path, const messaget &msg) = 0;

  // add external dependencies of a given module to set

  virtual void dependencies()
  {
  }

  // add modules provided by currently parsed file to set

  virtual void modules_provided(std::set<std::string> &)
  {
  }

  // final adjustments, e.g., initialization and call to main()
  virtual bool final(contextt &, const messaget &)
  {
    return false;
  }

  // type check interfaces of currently parsed file
  virtual bool interfaces()
  {
    return false;
  }

  // type check a module in the currently parsed file
  virtual bool typecheck(
    contextt &context,
    const std::string &module,
    const messaget &msg) = 0;

  // language id / description
  virtual std::string id() const
  {
    return "";
  }
  virtual std::string description() const
  {
    return "";
  }

  // show parse tree

  virtual void show_parse(std::ostream &out) = 0;

  // conversion of expressions
  virtual bool
  from_expr(const exprt &expr, std::string &code, const namespacet &ns) = 0;

  virtual bool
  from_type(const typet &type, std::string &code, const namespacet &ns) = 0;

  virtual languaget *new_language(const messaget &msg) = 0;

  // constructor / destructor

  explicit languaget(const messaget &msg) : msg(msg)
  {
  }
  virtual ~languaget() = default;

protected:
  const messaget &msg;
};
#endif
