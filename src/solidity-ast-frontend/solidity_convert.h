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

  // functions to get declarations of various types
  bool get_decl_intrinsics(
      nlohmann::json& decl, exprt &new_expr,
      const unsigned index, const std::string &key, const std::string &json_name);
  bool get_function(jsonTrackerRef json_tracker); // process function decl and add symbols
  bool get_type(jsonTrackerRef json_tracker, typet &new_type);
  bool get_type(const SolidityTypes::typeClass the_type,
      typet &new_type, jsonTrackerRef json_tracker);
  bool get_builtin_type(SolidityTypes::builInTypes the_blti_type, typet &new_type);
  void get_location_from_decl(jsonTrackerRef json_tracker,
      locationt &location);
  unsigned get_presumed_location(jsonTrackerRef json_tracker);
  void set_location(unsigned PLoc, std::string &function_name, locationt &location);
  std::string get_filename_from_path();
  void get_decl_name(jsonTrackerRef json_tracker, std::string &name, std::string &id);
  void get_default_symbol(symbolt &symbol, std::string module_name,
    typet type, std::string name, std::string id, locationt location);
  std::string get_modulename_from_path(jsonTrackerRef json_tracker);
  symbolt *move_symbol_to_context(symbolt &symbol);

  unsigned int current_scope_var_num; // tracking scope while getting declarations

  void convert_ast_nodes(nlohmann::json &contract_def);
};

#endif /* SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_ */
