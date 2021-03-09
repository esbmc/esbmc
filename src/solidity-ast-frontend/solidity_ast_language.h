/*******************************************************************\

Module: Solidity AST module

\*******************************************************************/

#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_AST_LANGUAGE_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_AST_LANGUAGE_H_

#include <util/language.h>

class solidity_ast_languaget : public languaget
{
public:
  bool
  parse(const std::string &path, message_handlert &message_handler) override;

  bool final(contextt &context, message_handlert &message_handler) override;

  bool typecheck(
    contextt &context,
    const std::string &module,
    message_handlert &message_handler) override;

  std::string id() const override
  {
    return "solidity_ast";
  }

  void show_parse(std::ostream &out) override;

  // conversion from expression into string
  bool from_expr(const exprt &expr, std::string &code, const namespacet &ns)
    override;

  // conversion from type into string
  bool from_type(const typet &type, std::string &code, const namespacet &ns)
    override;

  languaget *new_language() override
  {
    return new solidity_ast_languaget;
  }

  // constructor, destructor
  ~solidity_ast_languaget() override = default;
  solidity_ast_languaget();
};

languaget *new_solidity_ast_language();

#endif
