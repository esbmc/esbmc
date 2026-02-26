#include <python-frontend/python_consteval.h>
#include <algorithm>
#include <cmath>

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

std::optional<PyConstValue> python_consteval::try_eval_call(
  const std::string &func_name,
  const std::vector<PyConstValue> &args)
{
  const nlohmann::json *func_node = find_function(func_name);
  if (!func_node)
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
    return result->value;

  // Function didn't return explicitly → returns None
  return PyConstValue::make_none();
}

std::optional<python_consteval::StmtResult> python_consteval::exec_block(
  const nlohmann::json &body,
  Env &env)
{
  for (const auto &stmt : body)
  {
    auto result = exec_stmt(stmt, env);
    if (!result)
      return std::nullopt;
    if (result->type != StmtResult::CONTINUE)
      return result;
  }
  return StmtResult{StmtResult::CONTINUE, {}};
}

std::optional<python_consteval::StmtResult> python_consteval::exec_stmt(
  const nlohmann::json &stmt,
  Env &env)
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
      if (target["_type"] != "Name")
        return std::nullopt; // Only support simple variable assignment
      env[target["id"].get<std::string>()] = *val;
    }
    return StmtResult{StmtResult::CONTINUE, {}};
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
    return StmtResult{StmtResult::CONTINUE, {}};
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

    return StmtResult{StmtResult::CONTINUE, {}};
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

    return StmtResult{StmtResult::CONTINUE, {}};
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
    return StmtResult{StmtResult::CONTINUE, {}};
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
      return StmtResult{StmtResult::CONTINUE, {}};
    }

    return std::nullopt;
  }

  // Expression statement (e.g., standalone function calls)
  if (type == "Expr")
  {
    auto val = eval_expr(stmt["value"], env);
    if (!val)
      return std::nullopt;
    return StmtResult{StmtResult::CONTINUE, {}};
  }

  // Break/Continue
  if (type == "Break")
    return StmtResult{StmtResult::BREAK_STMT, {}};
  if (type == "Continue")
    return StmtResult{StmtResult::CONTINUE_STMT, {}};

  // Pass
  if (type == "Pass")
    return StmtResult{StmtResult::CONTINUE, {}};

  // Assert — evaluate but don't enforce (we're just computing the result)
  if (type == "Assert")
    return StmtResult{StmtResult::CONTINUE, {}};

  return std::nullopt; // Unsupported statement type
}

std::optional<PyConstValue> python_consteval::eval_expr(
  const nlohmann::json &node,
  const Env &env)
{
  if (!node.contains("_type"))
    return std::nullopt;

  const std::string &type = node["_type"];

  // Constants
  if (type == "Constant")
  {
    const auto &val = node["value"];
    if (val.is_null())
      return PyConstValue::make_none();
    if (val.is_boolean())
      return PyConstValue::make_bool(val.get<bool>());
    if (val.is_number_integer())
      return PyConstValue::make_int(val.get<long long>());
    if (val.is_number_float())
      return PyConstValue{
        PyConstValue::FLOAT, false, 0, val.get<double>(), ""};
    if (val.is_string())
      return PyConstValue::make_string(val.get<std::string>());
    return std::nullopt;
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
          PyConstValue::FLOAT, false, 0, -operand->float_val, ""};
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
      std::string result;
      for (long long i = 0;
           i < right->int_val && result.size() < 10000;
           ++i)
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
        long long result = 1;
        for (long long i = 0; i < r && i < 64; ++i)
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

    // Only support simple function calls (Name), not methods
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

    // Built-in: min(), max() for two int arguments
    if (func_name == "min" || func_name == "max")
    {
      if (node["args"].size() != 2)
        return std::nullopt;
      auto a = eval_expr(node["args"][0], env);
      auto b = eval_expr(node["args"][1], env);
      if (!a || !b || a->kind != PyConstValue::INT ||
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
