#pragma once

#include <nlohmann/json.hpp>

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/// The functions a Python expression may evaluate to, gathered from every
/// binding of the names it reads: assignments in any branch, the arguments a
/// parameter takes at every call site, the elements of a list literal it
/// indexes and the values a function it calls returns. The set is complete or
/// not given: a binding of any other shape, a use that could change what a
/// name holds, or a read that may come before the name is bound gives none.
class function_values
{
public:
  /// A function definition and the module AST it belongs to.
  struct candidate
  {
    const nlohmann::json *def;
    const nlohmann::json *module;
  };

  /// `others` are the program's other modules, searched for uses of this
  /// module's functions.
  function_values(
    const nlohmann::json &module,
    const std::vector<const nlohmann::json *> &others);

  /// The functions `expr` may evaluate to. `expr` may be a copy of a node of
  /// the module: it is found by its kind and source range.
  std::optional<std::vector<candidate>> of(const nlohmann::json &expr);

private:
  /// The module, then each function the expression is nested in.
  using chaint = std::vector<const nlohmann::json *>;
  using valuet = std::pair<const nlohmann::json *, chaint>;

  bool resolve(
    const nlohmann::json &expr,
    const chaint &chain,
    std::vector<candidate> &out,
    unsigned depth);
  bool name_values(
    const nlohmann::json &name,
    const chaint &chain,
    std::vector<valuet> &values,
    std::vector<candidate> &out);
  bool local_values(
    const nlohmann::json &name,
    const chaint &chain,
    std::vector<valuet> &values);
  bool module_name_values(
    const nlohmann::json &name,
    const chaint &chain,
    std::vector<valuet> &values,
    std::vector<candidate> &out);
  bool parameter_values(
    const nlohmann::json &def,
    const chaint &chain,
    const std::string &name,
    std::vector<valuet> &values);
  /// Where `def` is called, as (scope, name it is called by): the module by
  /// its own name, and each function it is passed to and only called
  /// through, by that parameter's name. None when it is used otherwise.
  std::optional<std::vector<std::pair<const nlohmann::json *, std::string>>>
  callers_of(const nlohmann::json &def, const nlohmann::json &module) const;
  /// Collects the argument each call of `callee` in `scope` passes for
  /// positional parameter `index`, called `name`; whether every call does.
  bool bound_arguments(
    const nlohmann::json &module,
    const nlohmann::json *scope,
    const std::string &callee,
    const nlohmann::json &args,
    std::size_t index,
    const std::string &name,
    std::vector<valuet> &values) const;
  /// Whether every use of `def` is a call `module` makes.
  bool
  only_called(const nlohmann::json &def, const nlohmann::json &module) const;
  bool element_values(
    const nlohmann::json &subscript,
    const chaint &chain,
    std::vector<valuet> &values);
  bool returned_values(
    const std::vector<candidate> &callees,
    std::vector<valuet> &values);
  bool imported_function(
    const nlohmann::json &module,
    const nlohmann::json &import_from,
    const nlohmann::json &alias,
    std::vector<candidate> &out);
  /// Whether a module other than `module` mentions `name`.
  bool mentioned_elsewhere(
    const nlohmann::json &module,
    const std::string &name) const;

  const nlohmann::json &module_;
  std::vector<const nlohmann::json *> others_;
  /// Imported modules' ASTs, by file.
  std::map<std::string, nlohmann::json> loaded_;
};
