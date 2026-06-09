#pragma once

#include <stdexcept>
#include <string>
#include <vector>

struct LdPropertyParseError : std::runtime_error
{
  explicit LdPropertyParseError(const std::string &msg)
    : std::runtime_error(msg)
  {
  }
};

enum class PropertyKind
{
  mutual_exclusion,
  invariant,
  response,
  absence,
  reachability,
};

struct LdProperty
{
  std::string id;
  PropertyKind kind;
  std::string description;
  std::string justification; // required for response / reachability

  // mutual_exclusion
  std::vector<std::string> variables;

  // invariant / absence / reachability
  std::string expression;

  // response
  std::string trigger;
  std::string response_var;
  unsigned max_scans = 0;
};

class YamlPropertyParser
{
public:
  // Parse a YAML property specification file.
  // Throws LdPropertyParseError on schema or validation errors.
  std::vector<LdProperty> parse(const std::string &path);

private:
  static PropertyKind kind_from_string(const std::string &s);
};
