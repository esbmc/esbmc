#pragma once

#include <goto-symex/witnesses.h>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

class yaml_parser
{
public:
  static std::vector<invariant> read_invariants(const std::string &path);
  static std::vector<waypoint> get_waypoints(const std::string &path);
  /// Returns true and sets @p out to the target waypoint if the witness has
  /// one; otherwise returns false.  Triggers parsing if not yet cached.
  static bool get_target_waypoint(const std::string &path, waypoint &out);

  // Reads the source file at `source_path` and returns a new source string
  // with witness intrinsic calls inserted before each annotated line.
  // `#line` directives after each injection refer to `original_path` so that
  // Clang source locations point back to the real file.
  static std::string build_injected_source(
    const std::string &source_path,
    const std::string &original_path,
    const std::vector<invariant> &invariants);

  // Reads the source file at `source_path` and returns a new source string
  // with `__ESBMC_witness_assume` calls injected before each assumption
  // waypoint line.  Multiple assumptions at the same line are injected in
  // order so that a stateful queue in symex can drain them correctly.
  static std::string build_violation_witness_source(
    const std::string &source_path,
    const std::string &original_path,
    const std::vector<waypoint> &waypoints);

private:
  static invariant parse_invariant(const YAML::Node &node);
  static invariant::Type type_from_string(const std::string &s);

  static waypoint parse_waypoint(const YAML::Node &node);
  static waypoint::Type waypoint_type_from_string(const std::string &s);
  static waypoint::Action action_from_string(const std::string &s);
};
