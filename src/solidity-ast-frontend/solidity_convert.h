#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_

//#define __STDC_LIMIT_MACROS
//#define __STDC_FORMAT_MACROS

#include <util/context.h>
#include <util/namespace.h>
#include <util/std_types.h>
#include <nlohmann/json.hpp>

class solidity_convertert
{
public:
  solidity_convertert(
    contextt &_context,
    nlohmann::json &_ast_json,
    nlohmann::json &_intrinsic_json);
  virtual ~solidity_convertert() = default;

  bool convert();

protected:
  contextt &context;
  nlohmann::json &ast_json; // json for Solidity AST. Use vector for multiple contracts
  nlohmann::json &intrinsic_json; // json for ESBMC intrinsics.
  void print_json_element(const nlohmann::json &json_in, const unsigned index,
    const std::string &key, const std::string& json_name);

  // functions to get declarations for different parts
  bool get_decl_intrinsics(
      const nlohmann::json& decl, exprt &new_expr,
      const unsigned index, const std::string &key, const std::string &json_name);

  unsigned int current_scope_var_num; // tracking scope while getting declarations
};

#endif /* SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_ */
