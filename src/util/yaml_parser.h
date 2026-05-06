#pragma once

#include <goto-symex/witnesses.h>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

class yaml_parser
{
public:
  static std::vector<invariant> read_invariants(const std::string &path);

  // Reads the source file at `source_path` and returns a new source string
  // with witness intrinsic calls inserted before each annotated line.
  // `#line` directives after each injection refer to `original_path` so that
  // Clang source locations point back to the real file.
  static std::string build_injected_source(
    const std::string &source_path,
    const std::string &original_path,
    const std::vector<invariant> &invariants);

private:
  static invariant parse_invariant(const YAML::Node &node);
  static invariant::Type type_from_string(const std::string &s);
};
