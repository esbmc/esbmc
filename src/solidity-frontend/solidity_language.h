#ifndef SOLIDITY_FRONTEND_SOLIDITY_AST_LANGUAGE_H_
#define SOLIDITY_FRONTEND_SOLIDITY_AST_LANGUAGE_H_

#include <clang-cpp-frontend/clang_cpp_language.h>
#include <util/language.h>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>

class solidity_languaget : public clang_cpp_languaget
{
public:
  bool parse(const std::string &path) override;

  bool final(contextt &context) override;

  bool typecheck(contextt &context, const std::string &module) override;

  std::string id() const override
  {
    return "solidity_ast";
  }

  void show_parse(std::ostream &out) override;

  // temp file used by clang-c-frontend
  std::string temp_path;

  // Functions to handle temp C file used by clang-c-frontend
  std::string get_temp_file();
  std::string temp_c_file();

  languaget *new_language() const override
  {
    return new solidity_languaget;
  }

  bool convert_intrinsics(contextt &context);

  // store AST json in nlohmann::json data structure
  nlohmann::json ast_json;
  nlohmann::json intrinsic_json;

  languaget *clang_cpp_module;
};

languaget *new_solidity_language();

#endif /* SOLIDITY_FRONTEND_SOLIDITY_AST_LANGUAGE_H_ */
