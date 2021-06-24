/*******************************************************************\

Module: Solidity AST module

\*******************************************************************/

#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_AST_LANGUAGE_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_AST_LANGUAGE_H_

#include <c2goto/cprover_library.h>
#include <clang-c-frontend/clang_c_language.h>
#include <clang-c-frontend/clang_c_adjust.h> // for context adjust in typecheck
#include <clang-c-frontend/clang_c_main.h>
#include <util/c_link.h> // for c_link
#include <util/language.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <libSif/ASTAnalyser.hpp>
#include <libUtils/Utils.hpp>
#include <nlohmann/json.hpp>

// Forward dec, to avoid bringing in clang headers
namespace clang
{
class ASTUnit;
} // namespace clang

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

//protected:
};

languaget *new_solidity_ast_language();

#endif
