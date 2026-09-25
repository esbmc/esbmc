#pragma once

#include <unordered_set>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class python_class
{
public:
  python_class() = default;

  void parse(const json &class_def);

  const std::string &name() const
  {
    return name_;
  }

  const std::unordered_set<std::string> &methods() const
  {
    return methods_;
  }

  const std::unordered_set<std::string> &attributes() const
  {
    return attrs_;
  }

  const std::unordered_set<std::string> &bases() const
  {
    return bases_;
  }

private:
  std::string name_;
  std::unordered_set<std::string> methods_;
  std::unordered_set<std::string> attrs_;
  std::unordered_set<std::string> bases_;
};
