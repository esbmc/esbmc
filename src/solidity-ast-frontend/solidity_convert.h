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
  void print_json_element(nlohmann::json &json_in, const unsigned index,
    const std::string &key, const std::string& json_name);

  // functions to get declarations of various types
  bool get_decl_intrinsics(
      nlohmann::json& decl, exprt &new_expr,
      const unsigned index, const std::string &key, const std::string &json_name);
  bool get_function(std::shared_ptr<decl_function_tracker>& json_tracker); // process function decl and add symbols
  bool get_type(std::shared_ptr<decl_function_tracker>& json_tracker, typet &new_type);
  bool get_type(const SolidityTypes::typeClass the_type,
      typet &new_type, std::shared_ptr<decl_function_tracker>& json_tracker);
  bool get_builtin_type(SolidityTypes::builInTypes the_blti_type, typet &new_type);
  void get_location_from_decl(std::shared_ptr<decl_function_tracker>& json_tracker,
      locationt &location);
  unsigned get_presumed_location(std::shared_ptr<decl_function_tracker>& json_tracker);
  void set_location(unsigned PLoc, std::string &function_name, locationt &location);
  std::string get_filename_from_path();
  void get_decl_name(std::shared_ptr<decl_function_tracker>& json_tracker, std::string &name, std::string &id);

  unsigned int current_scope_var_num; // tracking scope while getting declarations
};

#endif /* SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_ */
