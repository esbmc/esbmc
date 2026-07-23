#pragma once

#include <util/language.h>
#include <python-frontend/module/global_scope.h>

#include <nlohmann/json.hpp>

#include <vector>

class python_languaget : public languaget
{
public:
  bool parse(const std::string &path) override;

  bool final(contextt &context) override;

  bool typecheck(contextt &context, const std::string &module) override;

  bool from_expr(
    const exprt &expr,
    std::string &code,
    const namespacet &ns,
    unsigned flags) override;

  bool from_type(
    const typet &type,
    std::string &code,
    const namespacet &ns,
    unsigned flags) override;

  unsigned default_flags(presentationt target) const override;

  std::string id() const override
  {
    return "python";
  }

  void show_parse(std::ostream &out) override;

  languaget *new_language() const override
  {
    return new python_languaget;
  }

  /// Additional positional Python files passed on the command line, beyond
  /// the first (github #6211). Each entry is a fully parsed and annotated
  /// module AST, merged into the same program by python_converter.
  const std::vector<nlohmann::json> &get_extra_asts() const
  {
    return extra_asts;
  }

private:
  std::string ast_output_dir;
  nlohmann::json ast;
  std::vector<nlohmann::json> extra_asts;
  global_scope global_scope_;
};

languaget *new_python_language();
