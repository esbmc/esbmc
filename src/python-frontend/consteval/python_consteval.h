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
    STRING,
    TUPLE,
    LIST
  };

  Kind kind = NONE;
  bool bool_val = false;
  long long int_val = 0;
  double float_val = 0.0;
  std::string string_val;
  // Element storage for both TUPLE and LIST values; `kind` disambiguates so
  // that (1,) and [1] never compare equal (Python semantics).
  std::vector<PyConstValue> tuple_val;

  static PyConstValue make_none()
  {
    return {NONE, false, 0, 0.0, "", {}};
  }
  static PyConstValue make_bool(bool v)
  {
    return {BOOL, v, 0, 0.0, "", {}};
  }
  static PyConstValue make_int(long long v)
  {
    return {INT, false, v, 0.0, "", {}};
  }
  static PyConstValue make_float(double v)
  {
    return {FLOAT, false, 0, v, "", {}};
  }
  static PyConstValue make_string(const std::string &s)
  {
    return {STRING, false, 0, 0.0, s, {}};
  }
  static PyConstValue make_tuple(const std::vector<PyConstValue> &values)
  {
    return {TUPLE, false, 0, 0.0, "", values};
  }
  static PyConstValue make_list(const std::vector<PyConstValue> &values)
  {
    return {LIST, false, 0, 0.0, "", values};
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

  /// Try to evaluate an arbitrary expression at module scope, seeding the
  /// environment with module-level constant globals first. Used to fold whole
  /// assertion tests such as `f(GLOBAL) == [literal]` whose operands reference
  /// module globals (which try_eval_call alone cannot see). Returns
  /// std::nullopt if any sub-expression is not constant-foldable.
  std::optional<PyConstValue> try_eval_global_expr(const nlohmann::json &expr);

private:
  const nlohmann::json &ast_;
  int iteration_budget_;
  int call_depth_ = 0;
  static constexpr int MAX_CALL_DEPTH = 30;

  // When false (the default, used by the call-site pre-scan in converter_funcall
  // that folds f(literal) in expression position), functions containing control
  // flow (If/For/While) are NOT folded, so their branches stay in the GOTO
  // program for coverage/branch analysis. The whole-assertion folder
  // (try_eval_global_expr) sets this true: an assert over fully-constant
  // operands has a deterministic single path, so folding it loses no coverage.
  bool allow_control_flow_ = false;

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

  // Seed `env` with module-level constant globals (best-effort; non-foldable
  // assignments are simply skipped).
  void seed_globals(Env &env);

  std::optional<StmtResult> exec_block(const nlohmann::json &body, Env &env);
  std::optional<StmtResult> exec_stmt(const nlohmann::json &stmt, Env &env);

  // Bind an assignment target (Name, or Tuple/List for unpacking) to a value.
  // Returns false on an unsupported target shape or arity mismatch.
  bool bind_target(
    const nlohmann::json &target,
    const PyConstValue &value,
    Env &env);

  // Resolve the optional (start, end) window arguments of a str method
  // (args[1], args[2]) for a string of length `sz`, applying Python's
  // negative-index and clamping rules. Defaults: start=0, end=sz. Returns
  // false if a present window argument is not a constant int.
  bool resolve_str_window(
    const nlohmann::json &args,
    const Env &env,
    long long sz,
    long long &start,
    long long &end);
  std::optional<PyConstValue>
  eval_expr(const nlohmann::json &node, const Env &env);

  // String slice with full Python semantics
  static std::string slice_string(
    const std::string &s,
    std::optional<long long> start,
    std::optional<long long> stop,
    long long step);

  // Find function definition in top-level AST body
  const nlohmann::json *find_function(const std::string &name) const;
};
