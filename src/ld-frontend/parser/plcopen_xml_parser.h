#pragma once

#include <ld-frontend/parser/ld_ast.h>
#include <stdexcept>
#include <string>

// Thrown when the PLCopen XML file cannot be parsed or has a schema error.
struct LdParseError : std::runtime_error
{
  explicit LdParseError(const std::string &msg) : std::runtime_error(msg)
  {
  }
};

// Thrown when an unsupported Tier-2+ construct is encountered.
struct UnsupportedConstructError : std::runtime_error
{
  std::string construct;
  int tier;
  UnsupportedConstructError(const std::string &name, int t)
    : std::runtime_error(
        "UnsupportedConstruct(" + name + ", tier=" + std::to_string(t) + ")"),
      construct(name),
      tier(t)
  {
  }
};

class PlcopenXmlParser
{
public:
  // Parse a PLCopen XML file and return the typed AST.
  // Throws LdParseError on schema or structural errors.
  // Throws UnsupportedConstructError for Tier-2+ constructs.
  LdAst parse(const std::string &path);

private:
  // Schema normalisation: absorb vendor-specific deviations before building AST.
  void normalise(struct pugi_doc_wrapper &doc);

  VarKind var_kind_from_string(const std::string &type_str);
  ContactKind contact_kind_from_string(const std::string &s);
  CoilKind coil_kind_from_string(const std::string &s);
  FBKind fb_kind_from_string(const std::string &s);

  // Parse helpers
  VarDecl parse_var_decl(const void *xml_node);
  RungElement parse_rung_element(const void *xml_node);
  RungNode parse_rung(const void *xml_node);
  NetworkNode parse_network(const void *xml_node);

  std::string source_file_;

  // Variables the graphical resolver has to invent: function-block pins and
  // the accumulators that OR parallel paths together. Appended to the AST's
  // declarations once every network has been resolved.
  std::vector<VarDecl> synth_vars_;

  // Cyclic task period in milliseconds, used to convert IEC duration literals
  // into the scan ticks the fixed-tick timer model counts. Zero means the
  // program declares no task, in which case one tick is taken to be one ms.
  unsigned scan_interval_ms_ = 0;
};
