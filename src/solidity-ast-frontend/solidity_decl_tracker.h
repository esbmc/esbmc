#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_

#include <nlohmann/json.hpp>
#include <solidity-ast-frontend/solidity_type.h>

class solidity_decl_tracker
{
public:
  solidity_decl_tracker()
  {
      clear_all();
  }

  solidity_decl_tracker(const solidity_decl_tracker &rhs) = default; // default copy constructor
  solidity_decl_tracker(solidity_decl_tracker &&rhs) = default;      // default move constructor
  solidity_decl_tracker& operator=(const solidity_decl_tracker &rhs) = default; // default copy assignment operator
  solidity_decl_tracker& operator=(solidity_decl_tracker &&rhs) = default;      // default move assignment operator

  void clear_all()
  {
    printf("making decl tracker ...\n");
    test = false;
  }

  unsigned test_number() { return 12345; }

private:
  bool test;
};

#endif // END of SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
