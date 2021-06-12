/*******************************************************************\

Module: Solidity AST module

\*******************************************************************/

#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_AST_LANGUAGE_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_AST_LANGUAGE_H_

#include <clang-c-frontend/clang_c_language.h>
#include <util/language.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <libSif/ASTAnalyser.hpp>
#include <libUtils/Utils.hpp>
#include <nlohmann/json.hpp>

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
  ~solidity_ast_languaget();
  solidity_ast_languaget();

  // store AST json in nlohmann::json data structure
  nlohmann::json ast_json;
  nlohmann::json intrinsic_json;
  void print_json(const nlohmann::json &json_in);

  languaget * clang_c_module;

private:
  std::string internal_additions();
};

languaget *new_solidity_ast_language();

#endif
