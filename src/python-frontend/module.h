#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <nlohmann/json.hpp>

class module;

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

using OverloadList = std::vector<nlohmann::json>;

struct class_definition
{
  std::string name_;
  std::vector<std::string> bases_;
  std::vector<std::string> methods_;

  bool operator==(const class_definition &other) const
  {
    return name_ == other.name_;
  }
};

struct class_definition_hash
{
  std::size_t operator()(const class_definition &c) const
  {
    return std::hash<std::string>()(c.name_);
  }
};

using ClassesList = std::unordered_set<class_definition, class_definition_hash>;

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

  void add_overload(const nlohmann::json &ast)
  {
    overloads_.push_back(ast);
  }

  void add_class(const class_definition &cls)
  {
    classes_.insert(cls);
  }

  void add_classes(const ClassesList &other_classes)
  {
    classes_.insert(other_classes.begin(), other_classes.end());
  }

  class_definition get_class(const std::string &class_name) const
  {
    // Use find() for O(1) lookup
    class_definition key;
    key.name_ = class_name;
    auto it = classes_.find(key);

    if (it != classes_.end())
      return *it;

    return {};
  }

  function get_function(const std::string &func_name) const
  {
    // Use find() for O(1) lookup
    function key;
    key.name_ = func_name;
    auto it = functions_.find(key);

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

  const OverloadList &overloads() const
  {
    return overloads_;
  }

  const ClassesList &classes() const
  {
    return classes_;
  }

private:
  std::string name_;
  FunctionsList functions_;
  SubmodulesList submodules_;
  OverloadList overloads_;
  ClassesList classes_;
};
