/*******************************************************************\

Module: Solidity AST module

\*******************************************************************/

#include <solidity-ast-frontend/solidity_ast_language.h>
#include <solidity-ast-frontend/solidity_ast_convert.h>

languaget *new_solidity_ast_language()
{
  return new solidity_ast_languaget;
}

bool solidity_ast_languaget::parse(
  const std::string &path,
  message_handlert &message_handler)
{
    // function to get parse tree (AST):
    //   - In clang_c_language.cpp counterpart, we generate the AST of C programs using clang's utilities.
    //   - For Solidity AST files, we do nothing. Because we are already using Solidity's AST.
    parse_path = path;
    printf("Parsing... Already using Solidity AST\n");
    return false;
}

bool solidity_ast_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
    if(solidity_ast_convert(parse_tree, module, message_handler))
        return true;

    assert(!"come back and continue!");
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
  assert(!"come back and continue - solidity_ast_languaget::from_type");
  return false;
}
