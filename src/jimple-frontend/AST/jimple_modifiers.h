#ifndef ESBMC_JIMPLE_MODIFIERS_H
#define ESBMC_JIMPLE_MODIFIERS_H

#include <jimple-frontend/AST/jimple_ast.h>

class jimple_modifiers : public jimple_ast
{
public:
  virtual void from_json(const json& j) override;
  enum class modifier {
    Abstract,
    Final,
    Native,
    Public,
    Protected,
    Private,
    Static,
    Synchronized,
    Transient,
    Volatile,
    StrictFp,
    Enum,
    Annotation
  };
  virtual modifier from_string(const std::string &name);
  virtual std::string to_string(const modifier &ft);
  virtual std::string to_string() override;

protected:
  std::vector<modifier> m;
};

#endif //ESBMC_JIMPLE_MODIFIERS_H
