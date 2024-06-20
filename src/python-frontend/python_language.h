#pragma once

#include <util/language.h>

#include <nlohmann/json.hpp>

class python_languaget : public languaget
{
public:
  bool parse(const std::string &path) override;

  bool final(contextt &context) override;

  bool typecheck(contextt &context) override;

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

private:
  std::string ast_output_dir;
  nlohmann::json ast;
};

languaget *new_python_language();
