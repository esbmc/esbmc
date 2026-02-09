#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <goto-symex/witnesses.h>

class yaml_parser
{
public:
  explicit yaml_parser(const std::string &path);
  ~yaml_parser() = default;

  bool load_file();
  std::vector<invariant> get_invariants() const;

private:
  std::string file_path;
  YAML::Node root;
  invariant parse_invariant(const YAML::Node &node) const;
  invariant::Type type_from_string(const std::string &s) const;
};
