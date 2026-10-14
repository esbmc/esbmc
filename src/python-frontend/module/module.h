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

  /// Record a JSON AST this module's contents come from. The file is parsed
  /// on first access, not here: a run imports a handful of the modules the
  /// parser emits, and parsing the rest cost ~0.15s of every Python run.
  ///
  /// Invariant, load-bearing for the lazy split: parsing a source may only
  /// add functions, classes and overloads. It must never add a source or a
  /// submodule, because module_manager::find_module walks submodules_ WITHOUT
  /// hydrating, so a dotted lookup traverses unhydrated nodes. Following
  /// `Import` nodes here would make every dotted lookup silently miss.
  void add_source(std::string json_path)
  {
    sources_.push_back(std::move(json_path));
  }

  const std::vector<std::string> &sources() const
  {
    return sources_;
  }

  bool hydrated() const
  {
    return hydrated_;
  }

  void mark_hydrated()
  {
    hydrated_ = true;
  }

  /// False once hydration has run and every source failed to parse.
  bool readable() const
  {
    return readable_;
  }

  void set_readable(bool yes)
  {
    readable_ = yes;
  }

  const std::string &name() const
  {
    return name_;
  }

  void add_function(const function &func)
  {
    functions_.insert(func);
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

  /// @brief Retrieve a class definition by name
  /// @param class_name The name of the class to find
  /// @return The class definition, or empty class_definition if not found
  /// @complexity O(1) average case via hash lookup
  class_definition get_class(const std::string &class_name) const
  {
    class_definition key;
    key.name_ = class_name;
    auto it = classes_.find(key);

    if (it != classes_.end())
      return *it;

    return {};
  }

  /// @brief Retrieve a function definition by name
  /// @param func_name The name of the function to find
  /// @return The function definition, or empty function if not found
  /// @complexity O(1) average case via hash lookup
  function get_function(const std::string &func_name) const
  {
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
  std::vector<std::string> sources_;
  bool hydrated_ = false;
  bool readable_ = true;
};
