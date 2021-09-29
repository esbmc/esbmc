#include <assert.h>
#include <jimple-frontend/jimple-language.h>
#include <util/message/format.h>
#include <c2goto/cprover_library.h>
languaget *new_jimple_language(const messaget &msg)
{
  return new jimple_languaget(msg);

}
bool jimple_languaget::final(contextt &context, const messaget &msg)
{
  add_cprover_library(context, msg);
  return false;
}
bool jimple_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  //code = expr2c(expr, ns);
  return false;
}
bool jimple_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  //code = expr2c(expr, ns);
  return false;
}


