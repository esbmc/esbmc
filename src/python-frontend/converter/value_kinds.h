#pragma once

#include <nlohmann/json.hpp>

#include <set>
#include <string>
#include <utility>
#include <vector>

/// The coarse kinds of value (str, number, bool, None) a Python expression
/// may produce, read from the module AST alone: the bindings of the names it
/// reads, the arguments a parameter takes at every call site and its default,
/// and what the functions and lambdas it calls return, following a function
/// passed down a chain of parameters. What it cannot read stays unknown, so
/// two kinds are reported only when the program really mixes them.
class value_kinds
{
public:
  explicit value_kinds(const nlohmann::json &module);

  /// Whether `expr`, read in `scope` (a function definition or the module),
  /// may produce values of two different known kinds.
  bool diverges(const nlohmann::json &expr, const nlohmann::json &scope);

  /// The top-level function called `function_name`, or the module.
  const nlohmann::json &scope_of(const std::string &function_name) const;

private:
  using scoped = std::pair<const nlohmann::json *, const nlohmann::json *>;

  void kinds_of(
    const nlohmann::json &expr,
    const nlohmann::json &scope,
    std::set<std::string> &out,
    unsigned depth);
  void call_kinds(
    const nlohmann::json &call,
    const nlohmann::json &scope,
    std::set<std::string> &out,
    unsigned depth);
  std::vector<scoped>
  name_values(const std::string &name, const nlohmann::json &scope) const;
  std::vector<scoped> callees(
    const nlohmann::json &callee,
    const nlohmann::json &scope,
    unsigned depth) const;
  void parameter_values(
    const nlohmann::json &def,
    const std::string &name,
    std::vector<scoped> &out) const;
  const nlohmann::json *top_level_def(const std::string &name) const;
  const nlohmann::json *enclosing_scope(const nlohmann::json &call) const;

  const nlohmann::json &module_;
};
