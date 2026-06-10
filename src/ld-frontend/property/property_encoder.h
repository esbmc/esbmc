#pragma once

#include <ld-frontend/property/yaml_property_parser.h>
#include <util/context.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <string>
#include <vector>

// property_encoder translates LdProperty objects into code_assertt nodes
// which are appended to the end of the scan-loop body by ld_converter.
class property_encoder
{
public:
  property_encoder(contextt &context, const std::string &source_file);

  // Encode all properties and return a code_blockt of assertions.
  // Throws std::runtime_error if a property references an undeclared variable.
  code_blockt encode(const std::vector<LdProperty> &props);

private:
  contextt &context_;
  std::string source_file_;

  symbol_exprt var_expr(const std::string &name, const LdProperty &p) const;
  exprt parse_bool_expr(const std::string &expr_str, const LdProperty &p) const;

  code_assertt make_assert(const exprt &cond, const LdProperty &p) const;

  code_blockt encode_mutual_exclusion(const LdProperty &p);
  code_blockt encode_invariant(const LdProperty &p);
  code_blockt encode_absence(const LdProperty &p);
  code_blockt encode_reachability(const LdProperty &p);
  code_blockt encode_response(const LdProperty &p);
};
