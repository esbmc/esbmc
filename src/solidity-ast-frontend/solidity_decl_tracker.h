#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_

#include <nlohmann/json.hpp>
#include <solidity-ast-frontend/solidity_type.h>
#include <iostream>
#include <iomanip>
#include <vector>

class decl_function_tracker
{
public:
  decl_function_tracker(nlohmann::json& _decl_func)
  {
      clear_all();
  }

  decl_function_tracker(const decl_function_tracker &rhs) = default; // default copy constructor
  // move constructor
  decl_function_tracker(decl_function_tracker &&rhs) = default;
  // copy assignment operator - TODO: Since we have a reference member, it can be tricky in this case.
  //    Let's not do it for the time being. Leave it for future improvement
  decl_function_tracker& operator=(const decl_function_tracker &rhs) { assert(!"copy assignment is not allowed at the moment"); }
  // move assignment operator - TODO: Since we have a reference member, it can be tricky in this case.
  //    Let's not do it for the time being. Leave it for future improvement
  decl_function_tracker& operator=(decl_function_tracker &&rhs) { assert(!"move assignment is not allowed at the moment"); }

  void clear_all() // reset to default values
  {
    assert(!"populate tracker");
  }

  // for debug print
  void print_decl_func_json();

  // config this tracker based on json values
  void config();

//private:
};

#endif // END of SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
