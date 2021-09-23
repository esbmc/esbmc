#include <assert.h>
#include <jimple-frontend/jimple-language.h>
#include <util/message/format.h>

languaget *new_jimple_language(const messaget &msg)
{
  return new jimple_languaget(msg);

}
bool jimple_languaget::final(contextt &context, const messaget &msg)
{
  msg.debug("I don't know what final does");
  return languaget::final(context, msg);
}
bool jimple_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  msg.debug(fmt::format("From Type: {}",code));
  return false;
}
bool jimple_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  msg.debug(fmt::format("From Expr: {}",code));
  return false;
}


