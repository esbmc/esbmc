#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_

//#define __STDC_LIMIT_MACROS
//#define __STDC_FORMAT_MACROS

#include <memory>
#include <util/context.h>
#include <util/namespace.h>
#include <util/std_types.h>
#include <nlohmann/json.hpp>
#include <solidity-ast-frontend/solidity_type.h>
#include <solidity-ast-frontend/solidity_decl_tracker.h>

typedef std::shared_ptr<decl_function_tracker> DeclTrackerPtr; // to tracker json object when getting declarations

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

  // functions to get declarations of various types
  bool get_decl_intrinsics(
      const nlohmann::json& decl, exprt &new_expr,
      const unsigned index, const std::string &key, const std::string &json_name);
  bool get_function(DeclTrackerPtr& json_tracker); // process function decl and add symbols

  unsigned int current_scope_var_num; // tracking scope while getting declarations
};

#endif /* SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_ */
