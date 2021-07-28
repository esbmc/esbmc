#ifndef SOLIDITY_FRONTEND_PATTERN_CHECK_H_
#define SOLIDITY_FRONTEND_PATTERN_CHECK_H_

//#define __STDC_LIMIT_MACROS
//#define __STDC_FORMAT_MACROS

#include <memory>
#include <util/context.h>
#include <util/namespace.h>
#include <util/std_types.h>
#include <nlohmann/json.hpp>
#include <solidity-frontend/solidity_type.h>
#include <solidity-frontend/solidity_decl_tracker.h>

using varDeclTrackerPtr = std::shared_ptr<VarDeclTracker>&;
using funDeclTrackerPtr = std::shared_ptr<FunctionDeclTracker>&;

class pattern_checker
{
public:
  pattern_checker(
    const nlohmann::json &_ast_nodes,
    const std::string &_target_func,
    const messaget &msg);
  virtual ~pattern_checker() = default;

  bool do_pattern_check();
  bool start_pattern_based_check(const nlohmann::json &func);
  void check_authorization_through_tx_origin(const nlohmann::json &func);

protected:
  const nlohmann::json &ast_nodes;
  const std::string target_func; // function to be verified
  const messaget &msg;
  void print_json_element(const nlohmann::json &json_in);
};

#endif /* SOLIDITY_FRONTEND_PATTERN_CHECK_H_ */
