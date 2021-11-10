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

  modifier at(const int i) const
  {
    return m[i];
  }

protected:
  std::vector<modifier> m;

private:
  std::map<std::string, modifier> from_map = {
    {"abstract", modifier::Abstract},
    {"final", modifier::Final},
    {"native", modifier::Native},
    {"public", modifier::Public},
    {"protected", modifier::Protected},
    {"private", modifier::Private},
    {"static", modifier::Static},
    {"synchronized", modifier::Synchronized},
    {"transient", modifier::Transient},
    {"volatile", modifier::Volatile},
    {"strictFp", modifier::StrictFp},
    {"enum", modifier::Enum},
    {"annotation", modifier::Annotation}};

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
