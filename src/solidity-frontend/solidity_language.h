/*******************************************************************\

Module: Solidity Language module

Author: Kunjian Song, kunjian.song@postgrad.manchester.ac.uk

\*******************************************************************/

#ifndef SOLIDITY_FRONTEND_SOLIDITY_AST_LANGUAGE_H_
#define SOLIDITY_FRONTEND_SOLIDITY_AST_LANGUAGE_H_

#include <clang-c-frontend/clang_c_language.h>
#include <util/language.h>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>

class solidity_languaget : public clang_c_languaget
{
public:
  bool parse(const std::string &path, const messaget &msg) override;

  bool final(contextt &context, const messaget &msg) override;

  bool typecheck(
    contextt &context,
    const std::string &module,
    const messaget &msg) override;

  std::string id() const override
  {
    return "solidity_ast";
  }

  void show_parse(std::ostream &out) override;

  // temp file used by clang-c-frontend
  std::string temp_path;

  // Functions to handle temp C file used by clang-c-frontend
  std::string get_temp_file(const messaget &msg);
  std::string temp_c_file();

  languaget *new_language(const messaget &msg) const override
  {
    return new solidity_languaget(msg);
  }

  // constructor, destructor
  ~solidity_languaget() override = default;
  explicit solidity_languaget(const messaget &msg);

  bool convert_intrinsics(contextt &context, const messaget &msg);

  // store AST json in nlohmann::json data structure
  nlohmann::json ast_json;
  nlohmann::json intrinsic_json;

  languaget *clang_c_module;
};

languaget *new_solidity_language(const messaget &msg);

#endif /* SOLIDITY_FRONTEND_SOLIDITY_AST_LANGUAGE_H_ */
