#pragma once

#include <util/language.h>

class python_languaget : public languaget
{
public:
  bool parse(const std::string &path) override;

  bool final(contextt &context) override;

  bool typecheck(contextt &context, const std::string &module) override;

  bool from_expr(const exprt &expr, std::string &code, const namespacet &ns)
    override;

  bool from_type(const typet &type, std::string &code, const namespacet &ns)
    override;

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
};

languaget *new_python_language();
