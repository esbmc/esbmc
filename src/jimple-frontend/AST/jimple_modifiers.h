#ifndef ESBMC_JIMPLE_MODIFIERS_H
#define ESBMC_JIMPLE_MODIFIERS_H

#include <jimple-frontend/AST/jimple_ast.h>

/**
 * @brief AST parsing for modifiers
 *
 * This should hold every modifier e.g. final, private
 * and etc...
 *
 * We will probably only care for Static and Native thought
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
    Native, // C/C++ calls through JNI
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

  bool contains(modifier other) const
  {
    return std::find(modifiers.begin(), modifiers.end(), other) !=
           modifiers.end();
  }

  bool is_static() const
  {
    return contains(modifier::Static);
  }

  bool is_public() const
  {
    return contains(modifier::Public);
  }

  bool is_private() const
  {
    return contains(modifier::Private);
  }

  modifier at(const int i) const
  {
    return modifiers[i];
  }

protected:
  std::vector<modifier> modifiers;

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
