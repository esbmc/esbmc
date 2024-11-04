#pragma once

#include <unordered_set>
#include <string>

class global_scope
{
public:
  global_scope() = default;

  void add_class(const std::string &class_name)
  {
    classes_.insert(class_name);
  }

  void add_variable(const std::string &var_name)
  {
    variables_.insert(var_name);
  }

  const std::unordered_set<std::string> &classes() const noexcept
  {
    return classes_;
  }

  const std::unordered_set<std::string> &variables() const noexcept
  {
    return variables_;
  }

private:
  std::unordered_set<std::string> classes_;
  std::unordered_set<std::string> variables_;
};
