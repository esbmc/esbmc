#pragma once

#include <util/language.h>
#include <ld-frontend/parser/ld_ast.h>
#include <ld-frontend/property/yaml_property_parser.h>
#include <string>
#include <vector>

class ld_languaget : public languaget
{
public:
  // Parse PLCopen XML → LdAst; run type checker.
  bool parse(const std::string &path) override;

  // Run ld_converter: populate contextt with symbolt entries and the
  // scan-loop GOTO function body.  Mirrors python_languaget::typecheck().
  bool typecheck(contextt &context, const std::string &module) override;

  bool final(contextt &) override
  {
    return false;
  }

  std::string id() const override
  {
    return "ld";
  }

  void show_parse(std::ostream &out) override;

  languaget *new_language() const override
  {
    return new ld_languaget;
  }

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

  // Set the path to the YAML property file (used by ld-verify CLI).
  void set_props_path(const std::string &path)
  {
    props_path_ = path;
  }

private:
  LdAst ast_;
  std::string props_path_;
  std::vector<LdProperty> props_;
};

languaget *new_ld_language();
