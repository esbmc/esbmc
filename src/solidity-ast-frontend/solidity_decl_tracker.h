#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_

#include <nlohmann/json.hpp>
#include <solidity-ast-frontend/solidity_type.h>

class decl_function_tracker;
typedef std::shared_ptr<decl_function_tracker> DeclTrackerPtr;
namespace ConfigureTracker
{
  // contains functions to parse the json value and configure the trackers accordingly
  void configre_decl_function_tracker(const nlohmann::json& decl, DeclTrackerPtr& json_tracker); // configure tracker based on json values
};

class decl_function_tracker
{
public:
  decl_function_tracker()
  {
      clear_all();
  }

  decl_function_tracker(const decl_function_tracker &rhs) = default; // default copy constructor
  decl_function_tracker(decl_function_tracker &&rhs) = default;      // default move constructor
  decl_function_tracker& operator=(const decl_function_tracker &rhs) = default; // default copy assignment operator
  decl_function_tracker& operator=(decl_function_tracker &&rhs) = default;      // default move assignment operator

  void clear_all()
  {
    isImplicit = false;
  }

  // setters
  void set_isImplicit(bool _isImplicit) { isImplicit = _isImplicit; }

  // getters
  bool get_isImplicit() { return isImplicit; }

private:
  bool isImplicit;

};

#endif // END of SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
