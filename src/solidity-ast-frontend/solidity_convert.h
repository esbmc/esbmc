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

using jsonTrackerRef = std::shared_ptr<decl_function_tracker>&;

class solidity_convertert
{
public:
  solidity_convertert(
    contextt &_context,
    nlohmann::json &_ast_json);
  virtual ~solidity_convertert() = default;

  bool convert();

protected:
  contextt &context;
  nlohmann::json &ast_json; // json for Solidity AST. Use vector for multiple contracts
  void print_json_element(nlohmann::json &json_in, const unsigned index,
    const std::string &key, const std::string& json_name);
  void print_json_array_element(nlohmann::json &json_in,
      const std::string& node_type, const unsigned index);

  unsigned int current_scope_var_num; // tracking scope while getting declarations

  void convert_ast_nodes(nlohmann::json &contract_def);
};

#endif /* SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_ */
