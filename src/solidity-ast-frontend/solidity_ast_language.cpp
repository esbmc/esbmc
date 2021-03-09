/*******************************************************************\

Module: Solidity AST module

\*******************************************************************/

#include <solidity-ast-frontend/solidity_ast_language.h>

languaget *new_solidity_ast_language()
{
  return new solidity_ast_languaget;
}

solidity_ast_languaget::solidity_ast_languaget()
{
    printf("TODO: solidity_ast_languaget constructor ...\n");
}

bool solidity_ast_languaget::parse(
  const std::string &path,
  message_handlert &message_handler)
{
  assert(!"come back and continue - solidity_ast_languaget::parse");
  return false;
}

bool solidity_ast_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  assert(!"come back and continue - solidity_ast_languaget::typecheck");
  return false;
}

void solidity_ast_languaget::show_parse(std::ostream &)
{
  assert(!"come back and continue - solidity_ast_languaget::show_parse");
}

bool solidity_ast_languaget::final(
  contextt &context,
  message_handlert &message_handler)
{
  assert(!"come back and continue - solidity_ast_languaget::final");
  return false;
}

bool solidity_ast_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  assert(!"come back and continue - solidity_ast_languaget::from_expr");
  return false;
}

bool solidity_ast_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  assert(!"come back and continue - solidity_ast_languaget::parse");
  return false;
}
