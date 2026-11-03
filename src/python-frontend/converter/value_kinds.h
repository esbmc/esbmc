#pragma once

#include <nlohmann/json.hpp>

#include <set>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

/// The coarse kinds of value (str, number, bool, None) a Python expression
/// may produce, read from the module AST alone: the bindings of the names it
/// reads, the arguments a parameter takes at every call that may reach its
/// function and its default, and what the functions and lambdas it calls
/// return. A function is followed through names, parameters, nested scopes
/// and the calls that return it (#9319). What it cannot read stays unknown.
class value_kinds
{
public:
  explicit value_kinds(const nlohmann::json &module);

  /// Whether `expr`, read in `scope` (a function definition or the module),
  /// may produce a value that is not a number: it has sources of two
  /// different known kinds, or one of them is a str or None.
  bool
  may_be_non_number(const nlohmann::json &expr, const nlohmann::json &scope);

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
  std::vector<scoped> name_values(
    const std::string &name,
    const nlohmann::json &scope,
    unsigned depth) const;
  std::vector<scoped> callees(
    const nlohmann::json &callee,
    const nlohmann::json &scope,
    unsigned depth) const;
  std::vector<const nlohmann::json *>
  return_values(const nlohmann::json &fn, bool &returns, bool &yields) const;
  std::vector<scoped> returned_functions(
    const nlohmann::json &call,
    const nlohmann::json &scope,
    unsigned depth) const;
  void parameter_values(
    const nlohmann::json &def,
    const std::string &name,
    std::vector<scoped> &out) const;
  std::vector<const nlohmann::json *> callers(const nlohmann::json &def) const;
  const nlohmann::json *top_level_def(const std::string &name) const;
  const nlohmann::json &enclosing_scope(const nlohmann::json &node) const;
  void index(const nlohmann::json &node, const nlohmann::json *scope);

  const nlohmann::json &module_;
  /// The function or lambda whose own scope holds each node, or the module.
  std::unordered_map<const nlohmann::json *, const nlohmann::json *> scope_;
  std::vector<const nlohmann::json *> calls_;
  mutable std::
    unordered_map<const nlohmann::json *, std::vector<const nlohmann::json *>>
      callers_;
  mutable std::set<const nlohmann::json *> finding_callers_;
};
