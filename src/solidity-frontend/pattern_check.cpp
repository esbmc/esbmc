#include <solidity-frontend/pattern_check.h>

pattern_checker::pattern_checker(const nlohmann::json &_ast_nodes,
    const std::string &_target_func, const messaget &msg):
    ast_nodes(_ast_nodes),
    target_func(_target_func),
    msg(msg)
{
}

bool pattern_checker::pattern_checker::do_pattern_check()
{
  // add more functions here to perform more pattern-based checks
  //print_json_element(ast_nodes);
  printf("Checking function %s ...\n", target_func.c_str());

  unsigned index = 0;
  nlohmann::json::const_iterator itr = ast_nodes.begin();
  for (; itr != ast_nodes.end(); ++itr, ++index)
  {
    if ((*itr).contains("kind") && (*itr).contains("nodeType") && (*itr).contains("name"))
    {
      // locate the target function
      if ((*itr)["kind"].get<std::string>() == "function" &&
          (*itr)["nodeType"].get<std::string>() == "FunctionDefinition" &&
          (*itr)["name"].get<std::string>() == target_func)
      {
        return start_pattern_based_check(*itr);
      }
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

void pattern_checker::check_authorization_through_tx_origin(const nlohmann::json &func)
{
  // looking for the pattern require(tx.origin == <VarDeclReference>)
  const nlohmann::json &body_stmt = func["body"]["statements"];
  msg.progress("  - Pattern-based checking: SWC-115 Authorization through tx.origin");
  printf("statements in function body array ... \n");
  //print_json_element(body_stmt);

  unsigned index = 0;
  nlohmann::json::const_iterator itr = body_stmt.begin();
  for (; itr != body_stmt.end(); ++itr, ++index)
  {
    printf("@@ checking body stmt %u\n", index);
    if (itr->contains("nodeType"))
    {
      if ((*itr)["nodeType"].get<std::string>() == "ExpressionStatement")
      {
        const nlohmann::json &expr = (*itr)["expression"];
        //print_json_element(expr);
        if (expr["nodeType"] == "FunctionCall")
        {
          if (expr["kind"] == "functionCall")
          {
            // require(.) authorization
            if (expr["expression"]["nodeType"].get<std::string>() == "Identifier")
            {
              if (expr["expression"]["name"].get<std::string>() == "require")
              {
                const nlohmann::json &call_args = expr["arguments"];
                // Search for tx.origin in BinaryOperation (==) as used in require(tx.origin == <VarDeclReference>)
                // There should be just one argument, the BinaryOpration expression
                if (call_args.size() == 1)
                {
                  const nlohmann::json &arg_expr = call_args[0];
                  // look for BinaryOperation "=="
                  if (arg_expr["nodeType"].get<std::string>() == "BinaryOperation")
                  {
                    if (arg_expr["operator"].get<std::string>() == "==")
                    {
                      const nlohmann::json &left_expr = arg_expr["leftExpression"];
                      // search for "tx", "." and "origin"
                      if (left_expr["nodeType"].get<std::string>() == "MemberAccess") // tx.origin is of the type MemberAccess expression
                      {
                        // do NOT conbine this predicate with the one above. If the nodeType was NOT MemberAccess,
                        // accessing "memberName" would throw an exception !
                        if (left_expr["memberName"].get<std::string>() == "origin")
                        {
                          if (left_expr["expression"]["nodeType"].get<std::string>() == "Identifier")
                          {
                            if (left_expr["expression"]["name"].get<std::string>() == "tx")
                            {
                              msg.progress("\t- Found vulnerability SWC-115 Authorization through tx.origin");
                            }
                          }
                        }
                      } // end of MemberAccess as in tx.origin
                    } // end of "=="
                  } // end of "BinaryOperation"
                } // end of checking 1 argument as in require(<leftExpr> == <rightExpr>);
              } // end of checking "require(.)"
            } // end of checking Identifier "require"
          } // end of "kind" == "functionCall"
        } // end of top-level expr["nodeType"] == "FunctionCall"
      }
    }
  }
}

void pattern_checker::print_json_element(const nlohmann::json &json_in)
{
  //printf("### %s element[%u] content: key=%s, size=%lu ###\n",
  //    json_name.c_str(), index, key.c_str(), json_in.size());
  printf("@@ ast_nodes in pattern_checker: \n");
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}
