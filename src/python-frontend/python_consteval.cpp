#include <python-frontend/python_consteval.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>

static double round_ties_to_even_consteval(const double value)
{
  const double lower = std::floor(value);
  const double diff = value - lower;
  constexpr double tie_eps = 1e-12;

  if (diff < 0.5 - tie_eps)
    return lower;
  if (diff > 0.5 + tie_eps)
    return lower + 1.0;

  const double parity = std::fmod(std::fabs(lower), 2.0);
  const bool lower_is_even =
    parity < tie_eps || std::fabs(parity - 2.0) < tie_eps;
  return lower_is_even ? lower : lower + 1.0;
}

static double
round_to_ndigits_ties_even_consteval(const double value, const int ndigits)
{
  auto round_ld_ties_even = [](const long double v) -> long double {
    const long double lower = std::floor(v);
    const long double diff = v - lower;
    constexpr long double tie_eps = 1e-15L;

    if (diff < 0.5L - tie_eps)
      return lower;
    if (diff > 0.5L + tie_eps)
      return lower + 1.0L;

    const long double parity = std::fmod(std::fabs(lower), 2.0L);
    const bool lower_is_even =
      parity < tie_eps || std::fabs(parity - 2.0L) < tie_eps;
    return lower_is_even ? lower : lower + 1.0L;
  };

  long double scale = 1.0L;
  if (ndigits >= 0)
  {
    for (int i = 0; i < ndigits; ++i)
      scale *= 10.0L;
    return static_cast<double>(
      round_ld_ties_even(static_cast<long double>(value) * scale) / scale);
  }

  for (int i = 0; i < -ndigits; ++i)
    scale *= 10.0L;
  return static_cast<double>(
    round_ld_ties_even(static_cast<long double>(value) / scale) * scale);
}

bool PyConstValue::is_truthy() const
{
  switch (kind)
  {
  case NONE:
    return false;
  case BOOL:
    return bool_val;
  case INT:
    return int_val != 0;
  case FLOAT:
    return float_val != 0.0;
  case STRING:
    return !string_val.empty();
  case TUPLE:
  case LIST:
    return !tuple_val.empty();
  }
  return false;
}

static bool pyconst_equal(const PyConstValue &lhs, const PyConstValue &rhs)
{
  if (lhs.kind != rhs.kind)
    return false;

  switch (lhs.kind)
  {
  case PyConstValue::NONE:
    return true;
  case PyConstValue::BOOL:
    return lhs.bool_val == rhs.bool_val;
  case PyConstValue::INT:
    return lhs.int_val == rhs.int_val;
  case PyConstValue::FLOAT:
    return lhs.float_val == rhs.float_val;
  case PyConstValue::STRING:
    return lhs.string_val == rhs.string_val;
  case PyConstValue::TUPLE:
  case PyConstValue::LIST:
    if (lhs.tuple_val.size() != rhs.tuple_val.size())
      return false;
    for (size_t i = 0; i < lhs.tuple_val.size(); ++i)
    {
      if (!pyconst_equal(lhs.tuple_val[i], rhs.tuple_val[i]))
        return false;
    }
    return true;
  }
  return false;
}

python_consteval::python_consteval(
  const nlohmann::json &ast,
  int max_iterations)
  : ast_(ast), iteration_budget_(max_iterations)
{
}

const nlohmann::json *
python_consteval::find_function(const std::string &name) const
{
  if (!ast_.contains("body") || !ast_["body"].is_array())
    return nullptr;

  for (const auto &node : ast_["body"])
  {
    if (
      node.contains("_type") && node["_type"] == "FunctionDef" &&
      node.contains("name") && node["name"] == name)
      return &node;
  }
  return nullptr;
}

/// Generic recursive scan: true if any statement (or nested block) has a
/// `_type` listed in `kinds`.
static bool body_has_stmt_kind(
  const nlohmann::json &body,
  std::initializer_list<const char *> kinds)
{
  for (const auto &stmt : body)
  {
    if (!stmt.contains("_type"))
      continue;
    const std::string &t = stmt["_type"];
    for (const char *k : kinds)
      if (t == k)
        return true;
    if (stmt.contains("body") && stmt["body"].is_array())
      if (body_has_stmt_kind(stmt["body"], kinds))
        return true;
    if (stmt.contains("orelse") && stmt["orelse"].is_array())
      if (body_has_stmt_kind(stmt["orelse"], kinds))
        return true;
  }
  return false;
}

/// Bodies carrying properties that must be verified independently: an Assert
/// (must reach the solver) or a nested FunctionDef (closures are unmodeled).
/// Always blocks folding, regardless of control-flow permission.
static bool body_has_unfoldable_stmts(const nlohmann::json &body)
{
  return body_has_stmt_kind(body, {"Assert", "FunctionDef"});
}

/// Control flow whose branches matter for coverage/branch analysis. Blocks
/// folding only on the conservative call-site pre-scan path (see
/// python_consteval::allow_control_flow_).
static bool body_has_control_flow(const nlohmann::json &body)
{
  return body_has_stmt_kind(body, {"If", "For", "While"});
}

std::optional<PyConstValue> python_consteval::try_eval_call(
  const std::string &func_name,
  const std::vector<PyConstValue> &args)
{
  const nlohmann::json *func_node = find_function(func_name);
  if (!func_node)
    return std::nullopt;

  // Don't fold functions whose bodies carry independently-verifiable
  // properties (asserts) or nested function definitions.
  if (
    func_node->contains("body") && (*func_node)["body"].is_array() &&
    body_has_unfoldable_stmts((*func_node)["body"]))
    return std::nullopt;

  // On the conservative pre-scan path, also decline functions with control
  // flow so their branches remain in the GOTO program for coverage analysis.
  if (
    !allow_control_flow_ && func_node->contains("body") &&
    (*func_node)["body"].is_array() &&
    body_has_control_flow((*func_node)["body"]))
    return std::nullopt;

  if (!func_node->contains("args") || !(*func_node)["args"].contains("args"))
    return std::nullopt;

  const auto &func_args = (*func_node)["args"]["args"];
  if (func_args.size() != args.size())
    return std::nullopt;

  // Create environment with parameter bindings
  Env env;
  for (size_t i = 0; i < args.size(); ++i)
  {
    std::string param_name = func_args[i]["arg"].get<std::string>();

    // Check type annotation if present — decline folding on mismatch
    // so that --strict-types and similar checks can still detect errors.
    if (func_args[i].contains("annotation"))
    {
      const auto &ann = func_args[i]["annotation"];
      if (ann.contains("id"))
      {
        const std::string &ann_type = ann["id"].get<std::string>();
        bool mismatch = false;
        if (ann_type == "int" && args[i].kind != PyConstValue::INT)
          mismatch = true;
        else if (ann_type == "str" && args[i].kind != PyConstValue::STRING)
          mismatch = true;
        else if (ann_type == "float" && args[i].kind != PyConstValue::FLOAT)
          mismatch = true;
        else if (ann_type == "bool" && args[i].kind != PyConstValue::BOOL)
          mismatch = true;
        if (mismatch)
          return std::nullopt;
      }
    }

    env[param_name] = args[i];
  }

  ++call_depth_;
  if (call_depth_ > MAX_CALL_DEPTH)
  {
    --call_depth_;
    return std::nullopt;
  }

  auto result = exec_block((*func_node)["body"], env);
  --call_depth_;

  if (!result)
    return std::nullopt;

  if (result->type == StmtResult::RETURN)
  {
    // Check return type annotation — decline folding on mismatch
    if (func_node->contains("returns") && !(*func_node)["returns"].is_null())
    {
      const auto &ret_ann = (*func_node)["returns"];
      if (ret_ann.contains("id"))
      {
        const std::string &ret_type = ret_ann["id"].get<std::string>();
        bool mismatch = false;
        if (ret_type == "int" && result->value.kind != PyConstValue::INT)
          mismatch = true;
        else if (
          ret_type == "str" && result->value.kind != PyConstValue::STRING)
          mismatch = true;
        else if (
          ret_type == "float" && result->value.kind != PyConstValue::FLOAT)
          mismatch = true;
        else if (ret_type == "bool" && result->value.kind != PyConstValue::BOOL)
          mismatch = true;
        if (mismatch)
          return std::nullopt;
      }
    }
    return result->value;
  }

  // Function didn't return explicitly → returns None
  return PyConstValue::make_none();
}

void python_consteval::seed_globals(Env &env)
{
  if (!ast_.contains("body") || !ast_["body"].is_array())
    return;

  // Count every top-level write to each name (any Assign/AnnAssign/AugAssign
  // target, including names bound through tuple/list unpacking). A name written
  // more than once does not have a single stable module-scope value, so seeding
  // it with its final value could mis-evaluate an assert that runs at an
  // earlier program point — only single-write names are safe to seed.
  std::unordered_map<std::string, int> write_count;
  std::function<void(const nlohmann::json &)> count_target =
    [&](const nlohmann::json &tgt) {
      if (!tgt.contains("_type"))
        return;
      const std::string &tt = tgt["_type"];
      if (tt == "Name")
        ++write_count[tgt["id"].get<std::string>()];
      else if ((tt == "Tuple" || tt == "List") && tgt.contains("elts"))
        for (const auto &e : tgt["elts"])
          count_target(e);
    };
  for (const auto &stmt : ast_["body"])
  {
    if (!stmt.contains("_type"))
      continue;
    const std::string &t = stmt["_type"];
    if (t == "Assign" && stmt.contains("targets"))
      for (const auto &tgt : stmt["targets"])
        count_target(tgt);
    else if ((t == "AnnAssign" || t == "AugAssign") && stmt.contains("target"))
      count_target(stmt["target"]);
  }

  auto seed_one =
    [&](const nlohmann::json &target, const nlohmann::json &value) {
      if (!target.contains("_type") || target["_type"] != "Name")
        return;
      const std::string name = target["id"].get<std::string>();
      if (write_count[name] != 1)
        return;
      auto val = eval_expr(value, env);
      if (val)
        env[name] = *val;
    };

  for (const auto &stmt : ast_["body"])
  {
    if (!stmt.contains("_type"))
      continue;
    const std::string &t = stmt["_type"];

    if (t == "Assign")
    {
      if (
        !stmt.contains("targets") || stmt["targets"].size() != 1 ||
        !stmt.contains("value"))
        continue;
      seed_one(stmt["targets"][0], stmt["value"]);
    }
    else if (t == "AnnAssign")
    {
      if (
        !stmt.contains("target") || !stmt.contains("value") ||
        stmt["value"].is_null())
        continue;
      seed_one(stmt["target"], stmt["value"]);
    }
  }
}

std::optional<PyConstValue>
python_consteval::try_eval_global_expr(const nlohmann::json &expr)
{
  // A whole-assertion fold over fully-constant operands has a single
  // deterministic path, so control-flow functions may be folded here without
  // losing branch/condition coverage.
  allow_control_flow_ = true;
  Env env;
  seed_globals(env);
  return eval_expr(expr, env);
}

std::optional<python_consteval::StmtResult>
python_consteval::exec_block(const nlohmann::json &body, Env &env)
{
  for (const auto &stmt : body)
  {
    auto result = exec_stmt(stmt, env);
    if (!result)
      return std::nullopt;
    if (result->type != StmtResult::NORMAL)
      return result;
  }
  return StmtResult{StmtResult::NORMAL, {}};
}

bool python_consteval::bind_target(
  const nlohmann::json &target,
  const PyConstValue &value,
  Env &env)
{
  if (!target.contains("_type"))
    return false;

  const std::string &t = target["_type"];

  if (t == "Name")
  {
    if (!target.contains("id"))
      return false;
    env[target["id"].get<std::string>()] = value;
    return true;
  }

  // Tuple/list unpacking: a, b = <sequence>. Starred targets are unsupported.
  if (t == "Tuple" || t == "List")
  {
    if (value.kind != PyConstValue::TUPLE && value.kind != PyConstValue::LIST)
      return false;
    if (!target.contains("elts"))
      return false;
    const auto &elts = target["elts"];
    if (elts.size() != value.tuple_val.size())
      return false;
    for (size_t i = 0; i < elts.size(); ++i)
      if (!bind_target(elts[i], value.tuple_val[i], env))
        return false;
    return true;
  }

  return false;
}

bool python_consteval::resolve_str_window(
  const nlohmann::json &args,
  const Env &env,
  long long sz,
  long long &start,
  long long &end)
{
  start = 0;
  end = sz;
  if (args.size() >= 2)
  {
    auto a = eval_expr(args[1], env);
    if (!a || a->kind != PyConstValue::INT)
      return false;
    start = a->int_val;
  }
  if (args.size() >= 3)
  {
    auto a = eval_expr(args[2], env);
    if (!a || a->kind != PyConstValue::INT)
      return false;
    end = a->int_val;
  }
  if (start < 0)
    start += sz;
  if (end < 0)
    end += sz;
  start = std::max<long long>(0, std::min(start, sz));
  end = std::max<long long>(0, std::min(end, sz));
  if (end < start)
    end = start;
  return true;
}

std::optional<python_consteval::StmtResult>
python_consteval::exec_stmt(const nlohmann::json &stmt, Env &env)
{
  if (!stmt.contains("_type"))
    return std::nullopt;

  const std::string &type = stmt["_type"];

  // Return statement
  if (type == "Return")
  {
    if (!stmt.contains("value") || stmt["value"].is_null())
      return StmtResult{StmtResult::RETURN, PyConstValue::make_none()};

    auto val = eval_expr(stmt["value"], env);
    if (!val)
      return std::nullopt;
    return StmtResult{StmtResult::RETURN, *val};
  }

  // Assignment: x = expr  or  x: type = expr
  if (type == "Assign")
  {
    if (
      !stmt.contains("targets") || stmt["targets"].empty() ||
      !stmt.contains("value"))
      return std::nullopt;

    auto val = eval_expr(stmt["value"], env);
    if (!val)
      return std::nullopt;

    for (const auto &target : stmt["targets"])
    {
      if (!bind_target(target, *val, env))
        return std::nullopt;
    }
    return StmtResult{StmtResult::NORMAL, {}};
  }

  if (type == "AnnAssign")
  {
    if (!stmt.contains("target") || stmt["target"]["_type"] != "Name")
      return std::nullopt;

    if (stmt.contains("value") && !stmt["value"].is_null())
    {
      auto val = eval_expr(stmt["value"], env);
      if (!val)
        return std::nullopt;
      env[stmt["target"]["id"].get<std::string>()] = *val;
    }
    return StmtResult{StmtResult::NORMAL, {}};
  }

  // Augmented assignment: x += expr, x -= expr, etc.
  if (type == "AugAssign")
  {
    if (
      !stmt.contains("target") || stmt["target"]["_type"] != "Name" ||
      !stmt.contains("value") || !stmt.contains("op"))
      return std::nullopt;

    const std::string &var_name = stmt["target"]["id"].get<std::string>();
    auto it = env.find(var_name);
    if (it == env.end())
      return std::nullopt;

    auto rhs = eval_expr(stmt["value"], env);
    if (!rhs)
      return std::nullopt;

    const std::string &op = stmt["op"]["_type"];
    PyConstValue &lhs = it->second;

    if (op == "Add")
    {
      if (lhs.kind == PyConstValue::INT && rhs->kind == PyConstValue::INT)
        lhs.int_val += rhs->int_val;
      else if (
        lhs.kind == PyConstValue::STRING && rhs->kind == PyConstValue::STRING)
        lhs.string_val += rhs->string_val;
      else
        return std::nullopt;
    }
    else if (op == "Sub")
    {
      if (lhs.kind == PyConstValue::INT && rhs->kind == PyConstValue::INT)
        lhs.int_val -= rhs->int_val;
      else
        return std::nullopt;
    }
    else if (op == "Mult")
    {
      if (lhs.kind == PyConstValue::INT && rhs->kind == PyConstValue::INT)
        lhs.int_val *= rhs->int_val;
      else
        return std::nullopt;
    }
    else
      return std::nullopt;

    return StmtResult{StmtResult::NORMAL, {}};
  }

  // If statement
  if (type == "If")
  {
    auto cond = eval_expr(stmt["test"], env);
    if (!cond)
      return std::nullopt;

    if (cond->is_truthy())
      return exec_block(stmt["body"], env);
    else if (stmt.contains("orelse") && !stmt["orelse"].empty())
      return exec_block(stmt["orelse"], env);

    return StmtResult{StmtResult::NORMAL, {}};
  }

  // While loop
  if (type == "While")
  {
    while (true)
    {
      if (--iteration_budget_ <= 0)
        return std::nullopt; // Prevent infinite loops

      auto cond = eval_expr(stmt["test"], env);
      if (!cond)
        return std::nullopt;

      if (!cond->is_truthy())
        break;

      auto result = exec_block(stmt["body"], env);
      if (!result)
        return std::nullopt;

      if (result->type == StmtResult::RETURN)
        return result;
      if (result->type == StmtResult::BREAK_STMT)
        break;
      // CONTINUE_STMT just continues to next iteration
    }
    return StmtResult{StmtResult::NORMAL, {}};
  }

  // For loop: for var in iterable
  if (type == "For")
  {
    if (stmt["target"]["_type"] != "Name")
      return std::nullopt;

    const std::string &var_name = stmt["target"]["id"].get<std::string>();
    auto iter_val = eval_expr(stmt["iter"], env);
    if (!iter_val)
      return std::nullopt;

    // Support for range()
    if (
      stmt["iter"]["_type"] == "Call" &&
      stmt["iter"]["func"]["_type"] == "Name" &&
      stmt["iter"]["func"]["id"] == "range")
    {
      // range() already evaluated — we need the actual range values
      // We'll handle range in eval_expr and iterate a list
      return std::nullopt; // TODO: implement range iteration if needed
    }

    // Iterate over string characters
    if (iter_val->kind == PyConstValue::STRING)
    {
      for (char c : iter_val->string_val)
      {
        if (--iteration_budget_ <= 0)
          return std::nullopt;

        env[var_name] = PyConstValue::make_string(std::string(1, c));
        auto result = exec_block(stmt["body"], env);
        if (!result)
          return std::nullopt;
        if (result->type == StmtResult::RETURN)
          return result;
        if (result->type == StmtResult::BREAK_STMT)
          break;
      }
      return StmtResult{StmtResult::NORMAL, {}};
    }

    return std::nullopt;
  }

  // Expression statement (e.g., standalone function calls)
  if (type == "Expr")
  {
    // In-place list mutation: <name>.append(<expr>). Handled here (not in
    // eval_expr) because it writes back into the environment.
    const auto &value = stmt["value"];
    if (
      value.contains("_type") && value["_type"] == "Call" &&
      value.contains("func") && value["func"].contains("_type") &&
      value["func"]["_type"] == "Attribute" && value["func"].contains("attr") &&
      value["func"]["attr"] == "append" && value["func"].contains("value") &&
      value["func"]["value"].contains("_type") &&
      value["func"]["value"]["_type"] == "Name" && value.contains("args") &&
      value["args"].size() == 1)
    {
      const std::string &name = value["func"]["value"]["id"].get<std::string>();
      auto it = env.find(name);
      if (it != env.end() && it->second.kind == PyConstValue::LIST)
      {
        auto arg = eval_expr(value["args"][0], env);
        if (!arg)
          return std::nullopt;
        it->second.tuple_val.push_back(*arg);
        return StmtResult{StmtResult::NORMAL, {}};
      }
    }

    auto val = eval_expr(stmt["value"], env);
    if (!val)
      return std::nullopt;
    return StmtResult{StmtResult::NORMAL, {}};
  }

  // Break/Continue
  if (type == "Break")
    return StmtResult{StmtResult::BREAK_STMT, {}};
  if (type == "Continue")
    return StmtResult{StmtResult::CONTINUE_STMT, {}};

  // Pass
  if (type == "Pass")
    return StmtResult{StmtResult::NORMAL, {}};

  // Assert — decline const-eval if the assertion would fail, so the
  // normal verification path can detect the bug.
  if (type == "Assert")
  {
    if (stmt.contains("test"))
    {
      auto test_val = eval_expr(stmt["test"], env);
      if (!test_val)
        return std::nullopt; // Unsupported expression — decline folding
      if (!test_val->is_truthy())
        return std::nullopt; // Assert would fail — decline folding
    }
    return StmtResult{StmtResult::NORMAL, {}};
  }

  return std::nullopt; // Unsupported statement type
}

std::optional<PyConstValue>
python_consteval::eval_expr(const nlohmann::json &node, const Env &env)
{
  if (!node.contains("_type"))
    return std::nullopt;

  const std::string &type = node["_type"];

  // Constants
  if (type == "Constant")
  {
    // Bignum literals tagged by parser.py (issue #4642) carry value:null
    // alongside the `_bigint` digit string. Decline to fold so the call
    // falls back through to get_literal, which raises the overflow
    // diagnostic. Without this guard a bignum literal in const-foldable
    // position would be silently treated as Python None.
    if (node.contains("_bigint"))
      return std::nullopt;
    const auto &val = node["value"];
    if (val.is_null())
      return PyConstValue::make_none();
    if (val.is_boolean())
      return PyConstValue::make_bool(val.get<bool>());
    if (val.is_number_integer())
      return PyConstValue::make_int(val.get<long long>());
    if (val.is_number_float())
      return PyConstValue{
        PyConstValue::FLOAT, false, 0, val.get<double>(), "", {}};
    if (val.is_string())
      return PyConstValue::make_string(val.get<std::string>());
    return std::nullopt;
  }

  if (type == "Tuple" || type == "List")
  {
    std::vector<PyConstValue> values;
    for (const auto &elt : node["elts"])
    {
      auto value = eval_expr(elt, env);
      if (!value)
        return std::nullopt;
      values.push_back(*value);
    }
    return type == "List" ? PyConstValue::make_list(values)
                          : PyConstValue::make_tuple(values);
  }

  // Variable lookup
  if (type == "Name")
  {
    const std::string &name = node["id"].get<std::string>();

    // Built-in constants
    if (name == "True")
      return PyConstValue::make_bool(true);
    if (name == "False")
      return PyConstValue::make_bool(false);
    if (name == "None")
      return PyConstValue::make_none();

    auto it = env.find(name);
    if (it == env.end())
      return std::nullopt;
    return it->second;
  }

  // Unary operations: not, -, +
  if (type == "UnaryOp")
  {
    auto operand = eval_expr(node["operand"], env);
    if (!operand)
      return std::nullopt;

    const std::string &op = node["op"]["_type"];

    if (op == "Not")
      return PyConstValue::make_bool(!operand->is_truthy());

    if (op == "USub")
    {
      if (operand->kind == PyConstValue::INT)
        return PyConstValue::make_int(-operand->int_val);
      if (operand->kind == PyConstValue::FLOAT)
        return PyConstValue{
          PyConstValue::FLOAT, false, 0, -operand->float_val, "", {}};
      return std::nullopt;
    }

    if (op == "UAdd")
    {
      if (
        operand->kind == PyConstValue::INT ||
        operand->kind == PyConstValue::FLOAT)
        return operand;
      return std::nullopt;
    }

    return std::nullopt;
  }

  // Binary operations: +, -, *, //, %, **
  if (type == "BinOp")
  {
    auto left = eval_expr(node["left"], env);
    auto right = eval_expr(node["right"], env);
    if (!left || !right)
      return std::nullopt;

    const std::string &op = node["op"]["_type"];

    // String concatenation
    if (
      op == "Add" && left->kind == PyConstValue::STRING &&
      right->kind == PyConstValue::STRING)
      return PyConstValue::make_string(left->string_val + right->string_val);

    // String repetition
    if (
      op == "Mult" && left->kind == PyConstValue::STRING &&
      right->kind == PyConstValue::INT)
    {
      if (right->int_val <= 0)
        return PyConstValue::make_string("");
      const std::size_t unit_len = left->string_val.size();
      if (unit_len == 0)
        return PyConstValue::make_string("");
      const long long repeat = right->int_val;
      if (repeat > static_cast<long long>(10000 / unit_len))
        return std::nullopt; // Result too large for safe constant-folding
      std::string result;
      result.reserve(unit_len * static_cast<std::size_t>(repeat));
      for (long long i = 0; i < repeat; ++i)
        result += left->string_val;
      return PyConstValue::make_string(result);
    }

    // Integer arithmetic
    if (left->kind == PyConstValue::INT && right->kind == PyConstValue::INT)
    {
      long long l = left->int_val, r = right->int_val;
      if (op == "Add")
        return PyConstValue::make_int(l + r);
      if (op == "Sub")
        return PyConstValue::make_int(l - r);
      if (op == "Mult")
        return PyConstValue::make_int(l * r);
      if (op == "FloorDiv")
      {
        if (r == 0)
          return std::nullopt;
        return PyConstValue::make_int(l / r);
      }
      if (op == "Mod")
      {
        if (r == 0)
          return std::nullopt;
        long long result = l % r;
        // Python mod always returns non-negative for positive divisor
        if (result != 0 && ((result < 0) != (r < 0)))
          result += r;
        return PyConstValue::make_int(result);
      }
      if (op == "Pow")
      {
        if (r < 0)
          return std::nullopt; // Would produce float
        if (r > 64)
          return std::nullopt; // Exponent too large for safe const-eval
        long long result = 1;
        for (long long i = 0; i < r; ++i)
          result *= l;
        return PyConstValue::make_int(result);
      }
    }

    return std::nullopt;
  }

  // Boolean operations: and, or
  if (type == "BoolOp")
  {
    const std::string &op = node["op"]["_type"];
    const auto &values = node["values"];

    if (op == "And")
    {
      PyConstValue result = PyConstValue::make_bool(true);
      for (const auto &v : values)
      {
        auto val = eval_expr(v, env);
        if (!val)
          return std::nullopt;
        if (!val->is_truthy())
          return val; // Short-circuit: return the falsy value
        result = *val;
      }
      return result; // Return last truthy value
    }

    if (op == "Or")
    {
      PyConstValue result = PyConstValue::make_bool(false);
      for (const auto &v : values)
      {
        auto val = eval_expr(v, env);
        if (!val)
          return std::nullopt;
        if (val->is_truthy())
          return val; // Short-circuit: return the truthy value
        result = *val;
      }
      return result; // Return last falsy value
    }

    return std::nullopt;
  }

  // Comparison: ==, !=, <, >, <=, >=
  if (type == "Compare")
  {
    auto left_val = eval_expr(node["left"], env);
    if (!left_val)
      return std::nullopt;

    const auto &ops = node["ops"];
    const auto &comparators = node["comparators"];

    if (ops.size() != comparators.size())
      return std::nullopt;

    for (size_t i = 0; i < ops.size(); ++i)
    {
      auto right_val = eval_expr(comparators[i], env);
      if (!right_val)
        return std::nullopt;

      const std::string &cmp_op = ops[i]["_type"];
      bool result = false;

      // String comparison
      if (
        left_val->kind == PyConstValue::STRING &&
        right_val->kind == PyConstValue::STRING)
      {
        int cmp = left_val->string_val.compare(right_val->string_val);
        if (cmp_op == "Eq")
          result = cmp == 0;
        else if (cmp_op == "NotEq")
          result = cmp != 0;
        else if (cmp_op == "Lt")
          result = cmp < 0;
        else if (cmp_op == "LtE")
          result = cmp <= 0;
        else if (cmp_op == "Gt")
          result = cmp > 0;
        else if (cmp_op == "GtE")
          result = cmp >= 0;
        else
          return std::nullopt;
      }
      // Integer comparison
      else if (
        left_val->kind == PyConstValue::INT &&
        right_val->kind == PyConstValue::INT)
      {
        long long l = left_val->int_val, r = right_val->int_val;
        if (cmp_op == "Eq")
          result = l == r;
        else if (cmp_op == "NotEq")
          result = l != r;
        else if (cmp_op == "Lt")
          result = l < r;
        else if (cmp_op == "LtE")
          result = l <= r;
        else if (cmp_op == "Gt")
          result = l > r;
        else if (cmp_op == "GtE")
          result = l >= r;
        else
          return std::nullopt;
      }
      // Boolean == None, None == None, etc.
      else if (
        left_val->kind == PyConstValue::NONE ||
        right_val->kind == PyConstValue::NONE)
      {
        if (cmp_op == "Is" || cmp_op == "Eq")
          result = (left_val->kind == right_val->kind);
        else if (cmp_op == "IsNot" || cmp_op == "NotEq")
          result = (left_val->kind != right_val->kind);
        else
          return std::nullopt;
      }
      // Bool comparison
      else if (
        left_val->kind == PyConstValue::BOOL &&
        right_val->kind == PyConstValue::BOOL)
      {
        if (cmp_op == "Eq")
          result = left_val->bool_val == right_val->bool_val;
        else if (cmp_op == "NotEq")
          result = left_val->bool_val != right_val->bool_val;
        else
          return std::nullopt;
      }
      // List/tuple equality (element-wise; kind must match)
      else if (
        (left_val->kind == PyConstValue::LIST ||
         left_val->kind == PyConstValue::TUPLE) &&
        (cmp_op == "Eq" || cmp_op == "NotEq"))
      {
        bool eq = pyconst_equal(*left_val, *right_val);
        result = (cmp_op == "Eq") ? eq : !eq;
      }
      else
        return std::nullopt;

      if (!result)
        return PyConstValue::make_bool(false);

      left_val = right_val; // For chained comparisons
    }
    return PyConstValue::make_bool(true);
  }

  // Subscript (string slicing and indexing)
  if (type == "Subscript")
  {
    auto obj = eval_expr(node["value"], env);
    if (!obj || obj->kind != PyConstValue::STRING)
      return std::nullopt;

    const auto &slice = node["slice"];

    // Simple index: s[i]
    if (slice.contains("_type") && slice["_type"] == "Constant")
    {
      if (!slice["value"].is_number_integer())
        return std::nullopt;

      long long idx = slice["value"].get<long long>();
      long long len = static_cast<long long>(obj->string_val.size());

      if (idx < 0)
        idx += len;
      if (idx < 0 || idx >= len)
        return std::nullopt; // Index out of range

      return PyConstValue::make_string(
        std::string(1, obj->string_val[static_cast<size_t>(idx)]));
    }

    // Variable index: s[var]
    if (slice.contains("_type") && slice["_type"] == "Name")
    {
      auto idx_val = eval_expr(slice, env);
      if (!idx_val || idx_val->kind != PyConstValue::INT)
        return std::nullopt;

      long long idx = idx_val->int_val;
      long long len = static_cast<long long>(obj->string_val.size());

      if (idx < 0)
        idx += len;
      if (idx < 0 || idx >= len)
        return std::nullopt;

      return PyConstValue::make_string(
        std::string(1, obj->string_val[static_cast<size_t>(idx)]));
    }

    // Slice: s[start:stop:step]
    if (slice.contains("_type") && slice["_type"] == "Slice")
    {
      std::optional<long long> start, stop;
      long long step = 1;

      if (slice.contains("lower") && !slice["lower"].is_null())
      {
        auto v = eval_expr(slice["lower"], env);
        if (!v || v->kind != PyConstValue::INT)
          return std::nullopt;
        start = v->int_val;
      }

      if (slice.contains("upper") && !slice["upper"].is_null())
      {
        auto v = eval_expr(slice["upper"], env);
        if (!v || v->kind != PyConstValue::INT)
          return std::nullopt;
        stop = v->int_val;
      }

      if (slice.contains("step") && !slice["step"].is_null())
      {
        auto v = eval_expr(slice["step"], env);
        if (!v || v->kind != PyConstValue::INT)
          return std::nullopt;
        step = v->int_val;
        if (step == 0)
          return std::nullopt; // ValueError in Python
      }

      return PyConstValue::make_string(
        slice_string(obj->string_val, start, stop, step));
    }

    return std::nullopt;
  }

  // Function call
  if (type == "Call")
  {
    if (!node.contains("func"))
      return std::nullopt;

    if (
      node["func"]["_type"] == "Attribute" && node["func"].contains("attr") &&
      node["func"]["attr"] == "index" && node["func"].contains("value"))
    {
      auto recv = eval_expr(node["func"]["value"], env);
      // Both LIST and TUPLE store their elements in tuple_val; a list literal
      // receiver folds the same way a tuple one does. Without covering LIST,
      // `[...].index(x)` fell through to the OM, whose value matching returns
      // a wrong index for a literal receiver.
      if (
        !recv ||
        (recv->kind != PyConstValue::TUPLE &&
         recv->kind != PyConstValue::LIST) ||
        node["args"].size() != 1)
        return std::nullopt;

      auto needle = eval_expr(node["args"][0], env);
      if (!needle)
        return std::nullopt;

      for (size_t i = 0; i < recv->tuple_val.size(); ++i)
      {
        if (pyconst_equal(recv->tuple_val[i], *needle))
          return PyConstValue::make_int(static_cast<long long>(i));
      }
      return std::nullopt;
    }

    // list/tuple .count(x) on a constant receiver folds at conversion time.
    // The literal-receiver OM path returns 0 for a present element, so fold
    // it here. String .count is handled by the string-method block below, so
    // only act on a LIST/TUPLE receiver and otherwise fall through.
    if (
      node["func"]["_type"] == "Attribute" &&
      node["func"].value("attr", std::string()) == "count" &&
      node["func"].contains("value"))
    {
      auto recv = eval_expr(node["func"]["value"], env);
      if (
        recv &&
        (recv->kind == PyConstValue::TUPLE ||
         recv->kind == PyConstValue::LIST) &&
        node["args"].size() == 1)
      {
        auto needle = eval_expr(node["args"][0], env);
        if (needle)
        {
          long long c = 0;
          for (const auto &e : recv->tuple_val)
            if (pyconst_equal(e, *needle))
              ++c;
          return PyConstValue::make_int(c);
        }
      }
    }

    // String methods on a STRING receiver: fold pure, ASCII-only operations
    // at conversion time. This lets handle_string_<method> see a constant
    // receiver downstream, avoiding `<method>() requires constant string`
    // errors for code like `flip_case(s)` whose body is `return s.swapcase()`.
    if (
      node["func"]["_type"] == "Attribute" && node["func"].contains("attr") &&
      node["func"].contains("value"))
    {
      auto recv = eval_expr(node["func"]["value"], env);
      if (recv && recv->kind == PyConstValue::STRING)
      {
        const std::string &s = recv->string_val;
        const std::string &m = node["func"]["attr"].get<std::string>();
        const auto &args_arr = node["args"];

        auto unary = [&](auto &&op) -> std::optional<PyConstValue> {
          if (!args_arr.empty())
            return std::nullopt;
          return op(s);
        };

        if (m == "swapcase")
          return unary([](const std::string &x) {
            std::string out(x);
            for (char &c : out)
            {
              auto uc = static_cast<unsigned char>(c);
              if (std::islower(uc))
                c = static_cast<char>(std::toupper(uc));
              else if (std::isupper(uc))
                c = static_cast<char>(std::tolower(uc));
            }
            return PyConstValue::make_string(out);
          });

        if (m == "upper")
          return unary([](const std::string &x) {
            std::string out(x);
            for (char &c : out)
              c =
                static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            return PyConstValue::make_string(out);
          });

        if (m == "lower" || m == "casefold")
          return unary([](const std::string &x) {
            std::string out(x);
            for (char &c : out)
              c =
                static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return PyConstValue::make_string(out);
          });

        if (m == "capitalize")
          return unary([](const std::string &x) {
            std::string out(x);
            for (size_t i = 0; i < out.size(); ++i)
            {
              auto uc = static_cast<unsigned char>(out[i]);
              out[i] =
                static_cast<char>(i == 0 ? std::toupper(uc) : std::tolower(uc));
            }
            return PyConstValue::make_string(out);
          });

        // title: a letter starts a new word iff the previous character is
        // uncased (CPython semantics — digits are uncased, so they *end* a
        // word: "3d movie".title() == "3D Movie", "a1a".title() == "A1A").
        if (m == "title")
          return unary([](const std::string &x) {
            std::string out(x);
            bool prev_cased = false;
            for (char &c : out)
            {
              auto uc = static_cast<unsigned char>(c);
              c = static_cast<char>(
                prev_cased ? std::tolower(uc) : std::toupper(uc));
              prev_cased = std::isalpha(uc) != 0;
            }
            return PyConstValue::make_string(out);
          });

        // is*() predicates — Python returns False on the empty string, and
        // require *all* chars to satisfy the predicate. `isupper`/`islower`
        // additionally require at least one cased character.
        auto pred_all =
          [&](auto &&char_ok, bool empty_ok) -> std::optional<PyConstValue> {
          if (!args_arr.empty())
            return std::nullopt;
          if (s.empty())
            return PyConstValue::make_bool(empty_ok);
          for (char c : s)
            if (!char_ok(static_cast<unsigned char>(c)))
              return PyConstValue::make_bool(false);
          return PyConstValue::make_bool(true);
        };

        if (m == "isalpha")
          return pred_all(
            [](unsigned char c) { return std::isalpha(c) != 0; }, false);
        if (m == "isdigit")
          return pred_all(
            [](unsigned char c) { return std::isdigit(c) != 0; }, false);
        if (m == "isalnum")
          return pred_all(
            [](unsigned char c) { return std::isalnum(c) != 0; }, false);
        if (m == "isspace")
          return pred_all(
            [](unsigned char c) { return std::isspace(c) != 0; }, false);
        // isascii/isprintable return True on the empty string (unlike the
        // other predicates); isdecimal returns False. Over byte strings the
        // ASCII range coincides with CPython for isdecimal (the '0'..'9'
        // digits) and for the printable/ascii classification.
        if (m == "isascii")
          return pred_all([](unsigned char c) { return c < 128; }, true);
        if (m == "isdecimal")
          return pred_all(
            [](unsigned char c) { return c >= '0' && c <= '9'; }, false);
        if (m == "isprintable")
          return pred_all(
            [](unsigned char c) { return std::isprint(c) != 0; }, true);
        if (m == "isupper" || m == "islower")
        {
          if (!args_arr.empty())
            return std::nullopt;
          bool seen_cased = false;
          for (char c : s)
          {
            auto uc = static_cast<unsigned char>(c);
            if (std::isupper(uc))
            {
              if (m == "islower")
                return PyConstValue::make_bool(false);
              seen_cased = true;
            }
            else if (std::islower(uc))
            {
              if (m == "isupper")
                return PyConstValue::make_bool(false);
              seen_cased = true;
            }
          }
          return PyConstValue::make_bool(seen_cased);
        }

        // istitle: uppercase letters may only follow uncased characters,
        // lowercase letters only cased ones, and at least one cased
        // character must be present (CPython semantics, byte-level ASCII).
        if (m == "istitle")
        {
          if (!args_arr.empty())
            return std::nullopt;
          bool cased = false;
          bool prev_cased = false;
          for (char c : s)
          {
            auto uc = static_cast<unsigned char>(c);
            if (std::isupper(uc))
            {
              if (prev_cased)
                return PyConstValue::make_bool(false);
              prev_cased = true;
              cased = true;
            }
            else if (std::islower(uc))
            {
              if (!prev_cased)
                return PyConstValue::make_bool(false);
              prev_cased = true;
              cased = true;
            }
            else
              prev_cased = false;
          }
          return PyConstValue::make_bool(cased);
        }

        if (m == "startswith" || m == "endswith")
        {
          if (args_arr.size() != 1)
            return std::nullopt;
          auto sub = eval_expr(args_arr[0], env);
          if (!sub || sub->kind != PyConstValue::STRING)
            return std::nullopt;
          const std::string &needle = sub->string_val;
          if (needle.size() > s.size())
            return PyConstValue::make_bool(false);
          bool ok =
            (m == "startswith")
              ? s.compare(0, needle.size(), needle) == 0
              : s.compare(s.size() - needle.size(), needle.size(), needle) == 0;
          return PyConstValue::make_bool(ok);
        }

        if (m == "count")
        {
          if (args_arr.empty() || args_arr.size() > 3)
            return std::nullopt;
          auto sub = eval_expr(args_arr[0], env);
          if (!sub || sub->kind != PyConstValue::STRING)
            return std::nullopt;
          const std::string &needle = sub->string_val;
          long long start = 0;
          long long end = 0;
          if (!resolve_str_window(
                args_arr, env, static_cast<long long>(s.size()), start, end))
            return std::nullopt;
          long long c = 0;
          if (needle.empty())
            return PyConstValue::make_int(end - start + 1);
          size_t pos = static_cast<size_t>(start);
          auto limit = static_cast<size_t>(end);
          while (pos + needle.size() <= limit)
          {
            if (s.compare(pos, needle.size(), needle) == 0)
            {
              ++c;
              pos += needle.size();
            }
            else
              ++pos;
          }
          return PyConstValue::make_int(c);
        }

        if (m == "find" || m == "rfind" || m == "index" || m == "rindex")
        {
          if (args_arr.empty() || args_arr.size() > 3)
            return std::nullopt;
          auto sub = eval_expr(args_arr[0], env);
          if (!sub || sub->kind != PyConstValue::STRING)
            return std::nullopt;
          const std::string &needle = sub->string_val;

          // Optional start/end search window (Python str.find(sub, start, end)).
          long long start = 0;
          long long end = 0;
          if (!resolve_str_window(
                args_arr, env, static_cast<long long>(s.size()), start, end))
            return std::nullopt;

          // Search within s[start:end]; matches are reported as indices into
          // the original string (offset by `start`).
          const std::string window = s.substr(
            static_cast<size_t>(start), static_cast<size_t>(end - start));
          size_t pos = (m == "rfind" || m == "rindex") ? window.rfind(needle)
                                                       : window.find(needle);
          if (pos == std::string::npos)
          {
            if (m == "index" || m == "rindex")
              return std::nullopt; // would raise ValueError — leave to BMC
            return PyConstValue::make_int(-1);
          }
          return PyConstValue::make_int(start + static_cast<long long>(pos));
        }

        if (m == "strip" || m == "lstrip" || m == "rstrip")
        {
          if (!args_arr.empty())
            return std::nullopt;
          size_t lo = 0;
          size_t hi = s.size();
          if (m != "rstrip")
            while (lo < hi && std::isspace(static_cast<unsigned char>(s[lo])))
              ++lo;
          if (m != "lstrip")
            while (hi > lo &&
                   std::isspace(static_cast<unsigned char>(s[hi - 1])))
              --hi;
          return PyConstValue::make_string(s.substr(lo, hi - lo));
        }

        // Unsupported str method — fall through to the normal codegen path.
        return std::nullopt;
      }
    }

    // Only support simple function calls (Name), not other methods
    if (node["func"]["_type"] != "Name")
      return std::nullopt;

    const std::string &func_name = node["func"]["id"].get<std::string>();

    // Built-in: len()
    if (func_name == "len")
    {
      if (node["args"].size() != 1)
        return std::nullopt;
      auto arg = eval_expr(node["args"][0], env);
      if (!arg)
        return std::nullopt;
      if (arg->kind == PyConstValue::STRING)
        return PyConstValue::make_int(
          static_cast<long long>(arg->string_val.size()));
      return std::nullopt;
    }

    // Built-in: str()
    if (func_name == "str")
    {
      if (node["args"].size() != 1)
        return std::nullopt;
      auto arg = eval_expr(node["args"][0], env);
      if (!arg)
        return std::nullopt;
      if (arg->kind == PyConstValue::STRING)
        return arg;
      if (arg->kind == PyConstValue::INT)
        return PyConstValue::make_string(std::to_string(arg->int_val));
      return std::nullopt;
    }

    // Built-in: int()
    if (func_name == "int")
    {
      if (node["args"].size() != 1)
        return std::nullopt;
      auto arg = eval_expr(node["args"][0], env);
      if (!arg)
        return std::nullopt;
      if (arg->kind == PyConstValue::INT)
        return arg;
      if (arg->kind == PyConstValue::BOOL)
        return PyConstValue::make_int(arg->bool_val ? 1 : 0);
      return std::nullopt;
    }

    // Built-in: bool()
    if (func_name == "bool")
    {
      if (node["args"].size() != 1)
        return std::nullopt;
      auto arg = eval_expr(node["args"][0], env);
      if (!arg)
        return std::nullopt;
      return PyConstValue::make_bool(arg->is_truthy());
    }

    // Built-in: abs()
    if (func_name == "abs")
    {
      if (node["args"].size() != 1)
        return std::nullopt;
      auto arg = eval_expr(node["args"][0], env);
      if (!arg)
        return std::nullopt;
      if (arg->kind == PyConstValue::INT)
        return PyConstValue::make_int(
          arg->int_val < 0 ? -arg->int_val : arg->int_val);
      return std::nullopt;
    }

    // Built-in: round()
    if (func_name == "round")
    {
      if (node["args"].size() < 1 || node["args"].size() > 2)
        return std::nullopt;
      auto arg = eval_expr(node["args"][0], env);
      if (!arg)
        return std::nullopt;

      if (node["args"].size() == 1)
      {
        // round(x) -> int
        if (arg->kind == PyConstValue::INT)
          return PyConstValue::make_int(arg->int_val);
        if (arg->kind == PyConstValue::FLOAT)
          return PyConstValue::make_int(static_cast<long long>(
            round_ties_to_even_consteval(arg->float_val)));
        return std::nullopt;
      }

      // round(x, n) -> float (or int if n <= 0, but keep float for simplicity)
      auto ndigits = eval_expr(node["args"][1], env);
      if (!ndigits || ndigits->kind != PyConstValue::INT)
        return std::nullopt;
      int n = static_cast<int>(ndigits->int_val);
      double val = (arg->kind == PyConstValue::INT)
                     ? static_cast<double>(arg->int_val)
                     : arg->float_val;
      double rounded = round_to_ndigits_ties_even_consteval(val, n);
      return PyConstValue::make_float(rounded);
    }

    // Built-in: min(), max() for two int arguments
    if (func_name == "min" || func_name == "max")
    {
      if (node["args"].size() != 2)
        return std::nullopt;
      auto a = eval_expr(node["args"][0], env);
      auto b = eval_expr(node["args"][1], env);
      if (
        !a || !b || a->kind != PyConstValue::INT ||
        b->kind != PyConstValue::INT)
        return std::nullopt;
      if (func_name == "min")
        return PyConstValue::make_int(std::min(a->int_val, b->int_val));
      return PyConstValue::make_int(std::max(a->int_val, b->int_val));
    }

    // User-defined function call — recurse
    std::vector<PyConstValue> args;
    for (const auto &arg_node : node["args"])
    {
      auto arg = eval_expr(arg_node, env);
      if (!arg)
        return std::nullopt;
      args.push_back(*arg);
    }

    return try_eval_call(func_name, args);
  }

  // IfExp (ternary): x if cond else y
  if (type == "IfExp")
  {
    auto cond = eval_expr(node["test"], env);
    if (!cond)
      return std::nullopt;
    if (cond->is_truthy())
      return eval_expr(node["body"], env);
    return eval_expr(node["orelse"], env);
  }

  // JoinedStr (f-string)
  if (type == "JoinedStr")
  {
    std::string result;
    for (const auto &part : node["values"])
    {
      if (part["_type"] == "Constant" && part["value"].is_string())
      {
        result += part["value"].get<std::string>();
      }
      else if (part["_type"] == "FormattedValue")
      {
        // A format spec or a !r/!a conversion changes the rendered text
        // ("{x:03d}" pads to "007"; "{s!r}" quotes); folding while
        // ignoring it would produce a wrong value that can satisfy a
        // false assertion — decline instead and let the string handler
        // deal with it. !s (115) renders the same text as the default.
        if (part.contains("format_spec") && !part["format_spec"].is_null())
          return std::nullopt;
        if (
          part.contains("conversion") && part["conversion"].is_number() &&
          part["conversion"].get<int>() != -1 &&
          part["conversion"].get<int>() != 's')
          return std::nullopt;
        auto val = eval_expr(part["value"], env);
        if (!val)
          return std::nullopt;
        if (val->kind == PyConstValue::STRING)
          result += val->string_val;
        else if (val->kind == PyConstValue::INT)
          result += std::to_string(val->int_val);
        else
          return std::nullopt;
      }
      else
        return std::nullopt;
    }
    return PyConstValue::make_string(result);
  }

  return std::nullopt; // Unsupported expression type
}

std::string python_consteval::slice_string(
  const std::string &s,
  std::optional<long long> start,
  std::optional<long long> stop,
  long long step)
{
  long long len = static_cast<long long>(s.size());

  long long actual_start, actual_stop;

  if (step > 0)
  {
    actual_start = start.value_or(0);
    actual_stop = stop.value_or(len);
  }
  else
  {
    actual_start = start.value_or(len - 1);
    actual_stop = stop.value_or(-(len + 1));
  }

  // Resolve negative indices
  if (actual_start < 0)
    actual_start += len;
  if (actual_stop < 0)
    actual_stop += len;

  // Clamp to valid range
  if (step > 0)
  {
    if (actual_start < 0)
      actual_start = 0;
    if (actual_start > len)
      actual_start = len;
    if (actual_stop < 0)
      actual_stop = 0;
    if (actual_stop > len)
      actual_stop = len;
  }
  else
  {
    if (actual_start < -1)
      actual_start = -1;
    if (actual_start >= len)
      actual_start = len - 1;
    if (actual_stop < -1)
      actual_stop = -1;
    if (actual_stop >= len)
      actual_stop = len;
  }

  std::string result;
  if (step > 0)
  {
    for (long long i = actual_start; i < actual_stop; i += step)
      result += s[static_cast<size_t>(i)];
  }
  else
  {
    for (long long i = actual_start; i > actual_stop; i += step)
      result += s[static_cast<size_t>(i)];
  }

  return result;
}
