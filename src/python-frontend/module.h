#pragma once

#include <string>
#include <vector>
#include <algorithm>

struct function
{
  std::string name_;
  std::string return_type_;

  bool operator==(const function &other) const
  {
    return name_ == other.name_;
  }
};

struct function_hash
{
  std::size_t operator()(const function &f) const
  {
    return std::hash<std::string>()(f.name_);
  }
};

using FunctionsList = std::unordered_set<function, function_hash>;
/* Keeping submodules as shared_ptr since ownership needs to be shared in
 * module_manager::get_module_from_dir */
using SubmodulesList = std::unordered_set<std::shared_ptr<module>>;

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
    functions_.insert(func);
  }

  void add_functions(const FunctionsList &other_functions)
  {
    functions_.insert(other_functions.begin(), other_functions.end());
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

  const FunctionsList &functions() const
  {
    return functions_;
  }

  const SubmodulesList &submodules() const
  {
    return submodules_;
  }

private:
  std::string name_;
  FunctionsList functions_;
  SubmodulesList submodules_;
};
