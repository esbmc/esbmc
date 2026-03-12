#pragma once

#include <goto-programs/goto_functions.h>
#include <goto-symex/witnesses.h>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

class yaml_parser
{
public:
  explicit yaml_parser(const std::string &path, contextt &ns, optionst &options);
  ~yaml_parser() = default;

  // load the witness file in YAML format
  bool load_file();
  // extract loop invariant from YAML witness file
  bool get_invariants();
  // inject loop invariants to goto functions
  bool inject_loop_invariants(goto_functionst &goto_functions);

private:
  // path to witness file
  std::string file_path_;
  YAML::Node root_;
  contextt &context_;
  optionst &options_;
  typedef std::vector<invariant> invariantst;
  invariantst parsed_invariants_;
  invariant parse_invariant(const YAML::Node &node) const;
  invariant::Type type_from_string(const std::string &s) const;
};
