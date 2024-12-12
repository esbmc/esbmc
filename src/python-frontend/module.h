#pragma once

#include <string>
#include <vector>
#include <algorithm>

struct function
{
  std::string name_;
  std::string return_type_;
};

class module
{
public:
  module(const std::string &name) : name_(name)
  {
  }

  const std::string &name() const
  {
    return name_;
  }

  void add_function(const function &func)
  {
    functions_.push_back(func);
  }

  void add_functions(const std::vector<function> &other_functions)
  {
    std::copy(
      other_functions.begin(),
      other_functions.end(),
      std::back_inserter(functions_));
  }

  void add_submodule(const std::shared_ptr<module> mod)
  {
    submodules_.insert(mod);
  }

  function get_function(const std::string &func_name) const
  {
    auto it = std::find_if(
      functions_.begin(), functions_.end(), [&](const function &func) {
        return func.name_ == func_name;
      });

    if (it != functions_.end())
      return *it;

    return {};
  }

  const std::vector<function> &functions() const
  {
    return functions_;
  }

  const std::unordered_set<std::shared_ptr<module>> &submodules() const
  {
    return submodules_;
  }

private:
  std::string name_;
  std::vector<function> functions_;
  std::unordered_set<std::shared_ptr<module>> submodules_;
};
