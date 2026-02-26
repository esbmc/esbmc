#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

/// Represents a constant Python value for compile-time evaluation.
struct PyConstValue
{
  enum Kind
  {
    NONE,
    BOOL,
    INT,
    FLOAT,
    STRING
  };

  Kind kind = NONE;
  bool bool_val = false;
  long long int_val = 0;
  double float_val = 0.0;
  std::string string_val;

  static PyConstValue make_none()
  {
    return {NONE, false, 0, 0.0, ""};
  }
  static PyConstValue make_bool(bool v)
  {
    return {BOOL, v, 0, 0.0, ""};
  }
  static PyConstValue make_int(long long v)
  {
    return {INT, false, v, 0.0, ""};
  }
  static PyConstValue make_float(double v)
  {
    return { FLOAT, false, 0, v, "" };
  }
  static PyConstValue make_string(const std::string &s)
  {
    return {STRING, false, 0, 0.0, s};
  }

  bool is_truthy() const;
};

/// Lightweight Python AST evaluator for compile-time function evaluation.
/// When a pure function is called with all-constant arguments, this evaluator
/// interprets the function body at conversion time and returns the result,
/// eliminating loops from the GOTO program entirely.
class python_consteval
{
public:
  explicit python_consteval(
    const nlohmann::json &ast,
    int max_iterations = 50000);

  /// Try to evaluate a function call with constant arguments.
  /// Returns std::nullopt if evaluation cannot be completed
  /// (unsupported construct, timeout, recursion limit, etc.).
  std::optional<PyConstValue> try_eval_call(
    const std::string &func_name,
    const std::vector<PyConstValue> &args);

private:
  const nlohmann::json &ast_;
  int iteration_budget_;
  int call_depth_ = 0;
  static constexpr int MAX_CALL_DEPTH = 30;

  using Env = std::unordered_map<std::string, PyConstValue>;

  struct StmtResult
  {
    enum Type
    {
      NORMAL,
      RETURN,
      BREAK_STMT,
      CONTINUE_STMT
    };
    Type type = NORMAL;
    PyConstValue value;
  };

  std::optional<StmtResult> exec_block(
    const nlohmann::json &body,
    Env &env);
  std::optional<StmtResult> exec_stmt(
    const nlohmann::json &stmt,
    Env &env);
  std::optional<PyConstValue> eval_expr(
    const nlohmann::json &node,
    const Env &env);

  // String slice with full Python semantics
  static std::string slice_string(
    const std::string &s,
    std::optional<long long> start,
    std::optional<long long> stop,
    long long step);

  // Find function definition in top-level AST body
  const nlohmann::json *find_function(const std::string &name) const;
};
