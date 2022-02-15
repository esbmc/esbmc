#include <solidity-frontend/pattern_check.h>
#include <stdlib.h>

pattern_checker::pattern_checker(
  const nlohmann::json &_ast_nodes,
  const std::string &_target_func,
  const messaget &msg)
  : ast_nodes(_ast_nodes), target_func(_target_func), msg(msg)
{
}

bool pattern_checker::do_pattern_check()
{
  // TODO: add more functions here to perform more pattern-based checks
  msg.status(fmt::format("Checking function {} ...", target_func.c_str()));

  unsigned index = 0;
  for(nlohmann::json::const_iterator itr = ast_nodes.begin();
      itr != ast_nodes.end();
      ++itr, ++index)
  {
    if(
      (*itr).contains("kind") && (*itr).contains("nodeType") &&
      (*itr).contains("name"))
    {
      // locate the target function
      if(
        (*itr)["kind"].get<std::string>() == "function" &&
        (*itr)["nodeType"].get<std::string>() == "FunctionDefinition" &&
        (*itr)["name"].get<std::string>() == target_func)
        return start_pattern_based_check(*itr);
    }
  }

  return false;
}

bool pattern_checker::start_pattern_based_check(const nlohmann::json &func)
{
  // SWC-115: Authorization through tx.origin
  check_authorization_through_tx_origin(func);
  return false;
}

void pattern_checker::check_authorization_through_tx_origin(
  const nlohmann::json &func)
{
  // looking for the pattern require(tx.origin == <VarDeclReference>)
  const nlohmann::json &body_stmt = func["body"]["statements"];
  msg.progress(
    "  - Pattern-based checking: SWC-115 Authorization through tx.origin");
  msg.debug("statements in function body array ... \n");

  unsigned index = 0;

  for(nlohmann::json::const_iterator itr = body_stmt.begin();
      itr != body_stmt.end();
      ++itr, ++index)
  {
    msg.status(fmt::format(" checking function body stmt {}", index));
    if(itr->contains("nodeType"))
    {
      if((*itr)["nodeType"].get<std::string>() == "ExpressionStatement")
      {
        const nlohmann::json &expr = (*itr)["expression"];
        if(expr["nodeType"] == "FunctionCall")
        {
          if(expr["kind"] == "functionCall")
          {
            check_require_call(expr);
          }
        }
      }
    }
  }
}

void pattern_checker::check_require_call(const nlohmann::json &expr)
{
  // Checking the authorization argument of require() function
  if(expr["expression"]["nodeType"].get<std::string>() == "Identifier")
  {
    if(expr["expression"]["name"].get<std::string>() == "require")
    {
      const nlohmann::json &call_args = expr["arguments"];
      // Search for tx.origin in BinaryOperation (==) as used in require(tx.origin == <VarDeclReference>)
      // There should be just one argument, the BinaryOpration expression.
      // Checking 1 argument as in require(<leftExpr> == <rightExpr>)
      if(call_args.size() == 1)
      {
        check_require_argument(call_args);
      }
    }
  }
}

void pattern_checker::check_require_argument(const nlohmann::json &call_args)
{
  // This function is used to check the authorization argument of require() funciton
  const nlohmann::json &arg_expr = call_args[0];

  // look for BinaryOperation "=="
  if(arg_expr["nodeType"].get<std::string>() == "BinaryOperation")
  {
    if(arg_expr["operator"].get<std::string>() == "==")
    {
      const nlohmann::json &left_expr = arg_expr["leftExpression"];
      // Search for "tx", "." and "origin". First, confirm the nodeType is MemeberAccess
      // If the nodeType was NOT MemberAccess, accessing "memberName" would throw an exception !
      if(
        left_expr["nodeType"].get<std::string>() ==
        "MemberAccess") // tx.origin is of the type MemberAccess expression
      {
        check_tx_origin(left_expr);
      }
    } // end of "=="
  }   // end of "BinaryOperation"
}

void pattern_checker::check_tx_origin(const nlohmann::json &left_expr)
{
  // This function is used to check the Tx.origin pattern used in BinOp expr
  if(left_expr["memberName"].get<std::string>() == "origin")
  {
    if(left_expr["expression"]["nodeType"].get<std::string>() == "Identifier")
    {
      if(left_expr["expression"]["name"].get<std::string>() == "tx")
      {
        //assert(!"Found vulnerability SWC-115 Authorization through tx.origin");
        msg.error(
          "Found vulnerability SWC-115 Authorization through tx.origin");
        msg.error("VERIFICATION FAILED");
        exit(EXIT_SUCCESS);
      }
    }
  }
}
