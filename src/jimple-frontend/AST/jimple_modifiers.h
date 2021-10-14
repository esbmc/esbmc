#ifndef ESBMC_JIMPLE_MODIFIERS_H
#define ESBMC_JIMPLE_MODIFIERS_H

#include <jimple-frontend/AST/jimple_ast.h>

/**
 * @brief AST parsing for modifiers
 * 
 * This should hold every modifier e.g. final, private
 * and etc...
 */
class jimple_modifiers : public jimple_ast
{
public:
  // overrides
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override;

  enum class modifier
  {
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
  virtual modifier from_string(const std::string &name) const;
  virtual std::string to_string(const modifier &ft) const;

protected:
  std::vector<modifier> m;

private:
  std::map<std::string, modifier> from_map = {
    {"Abstract", modifier::Abstract},
    {"Final", modifier::Final},
    {"Native", modifier::Native},
    {"Public", modifier::Public},
    {"Protected", modifier::Protected},
    {"Private", modifier::Private},
    {"Static", modifier::Static},
    {"Synchronized", modifier::Synchronized},
    {"Transient", modifier::Transient},
    {"Volatile", modifier::Volatile},
    {"StrictFp", modifier::StrictFp},
    {"Enum", modifier::Enum},
    {"Annotation", modifier::Annotation}};

  std::map<modifier, std::string> to_map = {
    {modifier::Abstract, "Abstract"},
    {modifier::Final, "Final"},
    {modifier::Native, "Native"},
    {modifier::Public, "Public"},
    {modifier::Protected, "Protected"},
    {modifier::Private, "Private"},
    {modifier::Static, "Static"},
    {modifier::Synchronized, "Synchronized"},
    {modifier::Transient, "Transient"},
    {modifier::Volatile, "Volatile"},
    {modifier::StrictFp, "StrictFp"},
    {modifier::Enum, "Enum"},
    {modifier::Annotation, "Annotation"}};
};

#endif //ESBMC_JIMPLE_MODIFIERS_H
