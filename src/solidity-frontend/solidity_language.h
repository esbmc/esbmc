/// \file solidity_language.h
/// \brief Solidity language frontend interface for ESBMC.
///
/// Defines solidity_languaget, the top-level entry point for parsing Solidity
/// source files. Inherits from clang_cpp_languaget and orchestrates the full
/// frontend pipeline: invoking the solc compiler to produce a JSON AST,
/// converting the AST to ESBMC's GOTO intermediate representation via
/// solidity_convertert, and performing type-checking and finalization.

#ifndef SOLIDITY_FRONTEND_SOLIDITY_AST_LANGUAGE_H_
#define SOLIDITY_FRONTEND_SOLIDITY_AST_LANGUAGE_H_

#include <clang-c-frontend/clang_c_language.h>
#include <clang-cpp-frontend/clang_cpp_language.h>
#include <util/language.h>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>

class solidity_languaget : public clang_cpp_languaget
{
public:
  solidity_languaget();

  bool parse(const std::string &path) override;

  bool final(contextt &context) override;

  bool typecheck(contextt &context, const std::string &module) override;

  std::string id() const override
  {
    return "solidity_ast";
  }

  void show_parse(std::ostream &out) override;

  // temp file for Clang to parse ESBMC intrinsic symbols
  std::string temp_path;
  std::string get_temp_file();
  bool convert_intrinsics(contextt &context);

  // solc auto-invocation support
  std::string find_solc() const;
  std::string get_solc_version(const std::string &solc) const;
  bool invoke_solc(const std::string &sol_path, std::string &solast_path);
  bool parse_solast(const std::string &solast_path);

  languaget *new_language() const override
  {
    return new solidity_languaget;
  }

  // contract name for verification, allow multiple inputs.
  std::string contract_names;

  // function name for verification that requires this information before GOTO conversion phase.
  std::string func_name;

  // focus function name: like func_name, but the full contract harness
  // (constructor + state init) still runs and only this function is
  // dispatched in the nondet extcall loop. Empty means feature disabled.
  std::string focus_func_name;

  // smart contract source
  std::string contract_path;

  // store AST json in nlohmann::json data structure
  nlohmann::json src_ast_json_array = nlohmann::json::array();
  nlohmann::json intrinsic_json;

  languaget *clang_c_module;
};

languaget *new_solidity_language();

#endif /* SOLIDITY_FRONTEND_SOLIDITY_AST_LANGUAGE_H_ */
