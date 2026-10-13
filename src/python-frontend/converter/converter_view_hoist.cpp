#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/python_converter.h>

#include <set>
#include <string>
#include <vector>

// A NumPy view passed to, or returned by, a user function carries its layout
// through a name. A view expression or a view-returning call used inside a
// larger expression is therefore bound to a fresh name before the statement,
// when moving it there cannot change what the statement computes.

namespace
{
using json_pointer = nlohmann::json::json_pointer;

bool is_name(const nlohmann::json &node)
{
  return node.is_object() && node.value("_type", "") == "Name" &&
         node.contains("id");
}

bool is_named_call(const nlohmann::json &node)
{
  return node.is_object() && node.value("_type", "") == "Call" &&
         node.contains("func") && is_name(node["func"]) &&
         node.contains("args") && node["args"].is_array();
}

// The expression a simple statement evaluates; "" for any other statement.
std::string statement_expression_key(const nlohmann::json &statement)
{
  static const std::set<std::string> with_value = {
    "Assign", "AnnAssign", "AugAssign", "Expr", "Return"};
  const std::string kind = statement.value("_type", "");
  // An `if` test runs once, before either branch.
  const std::string key = kind == "Assert" || kind == "If" ? "test"
                          : with_value.count(kind) != 0    ? "value"
                                                           : "";
  return !key.empty() && statement.contains(key) && statement[key].is_object()
           ? key
           : "";
}

// A body that runs later, repeatedly or not at all.
bool is_deferred_scope(const std::string &type)
{
  static const std::set<std::string> types = {
    "Lambda", "ListComp", "SetComp", "DictComp", "GeneratorExp"};
  return types.count(type) != 0;
}

// Whether `key` of a node of `type` holds operands that may not be evaluated.
bool is_conditional_operand(const std::string &type, const std::string &key)
{
  return type == "IfExp" && key != "test";
}

// Pointers to every `name[...]` argument of a named call that is evaluated
// exactly once when the expression at `path` is.
void collect_subscript_arguments(
  const nlohmann::json &node,
  const json_pointer &path,
  std::vector<json_pointer> &out)
{
  if (!node.is_object() && !node.is_array())
    return;
  const std::string type = node.is_object() ? node.value("_type", "") : "";
  if (is_deferred_scope(type))
    return;
  if (type == "BoolOp")
    return collect_subscript_arguments(
      node["values"][0], json_pointer(path.to_string() + "/values/0"), out);
  if (is_named_call(node))
    for (std::size_t i = 0; i < node["args"].size(); ++i)
      if (
        node["args"][i].value("_type", "") == "Subscript" &&
        is_name(node["args"][i]["value"]))
        out.push_back(path / "args" / i);
  std::size_t index = 0;
  for (auto it = node.begin(); it != node.end(); ++it, ++index)
  {
    const std::string key = node.is_object() ? it.key() : std::to_string(index);
    if (!is_conditional_operand(type, key))
      collect_subscript_arguments(*it, path / key, out);
  }
}

void collect_calls(
  const nlohmann::json &node,
  std::vector<const nlohmann::json *> &out)
{
  if (!node.is_object() && !node.is_array())
    return;
  if (node.is_object() && node.value("_type", "") == "Call")
    out.push_back(&node);
  for (const auto &child : node)
    collect_calls(child, out);
}

bool starts_before(const nlohmann::json &a, const nlohmann::json &b)
{
  const int a_line = a.value("lineno", 0), b_line = b.value("lineno", 0);
  return a_line < b_line || (a_line == b_line && a.value("col_offset", 0) <
                                                   b.value("col_offset", 0));
}

bool ends_before_end_of(const nlohmann::json &a, const nlohmann::json &b)
{
  const int a_line = a.value("end_lineno", 0),
            b_line = b.value("end_lineno", 0);
  return a_line < b_line ||
         (a_line == b_line &&
          a.value("end_col_offset", 0) < b.value("end_col_offset", 0));
}

// Whether some call of `expression` runs before `node` is evaluated: one
// that starts before it without enclosing it. A node without a source
// position counts as preceded.
bool is_preceded_by_call(
  const nlohmann::json &expression,
  const nlohmann::json &node)
{
  if (node.value("lineno", 0) == 0)
    return true;
  std::vector<const nlohmann::json *> calls;
  collect_calls(expression, calls);
  for (const nlohmann::json *call : calls)
    if (
      call->value("lineno", 0) == 0 ||
      (starts_before(*call, node) && ends_before_end_of(*call, node)))
      return true;
  return false;
}

// The next node on the path to what `node` evaluates first: the object of a
// subscript or attribute, the left operand, the first argument of a call to
// a plain name or to a module function.
std::string first_evaluated_child(
  const nlohmann::json &node,
  const std::unordered_map<std::string, std::string> &modules)
{
  const std::string type = node.value("_type", "");
  if (type == "Subscript" || type == "Attribute")
    return "/value";
  if (type == "Compare" || type == "BinOp")
    return "/left";
  if (type != "Call" || !node.contains("func"))
    return "";
  const nlohmann::json &func = node["func"];
  const bool module_function =
    func.value("_type", "") == "Attribute" && is_name(func["value"]) &&
    modules.count(func["value"]["id"].get<std::string>()) != 0;
  if (is_name(func) || module_function)
    return node["args"].empty() ? "" : "/args/0";
  return "/func";
}
} // namespace

// Binds `value` to a fresh name with an assignment emitted before the
// statement, and returns the name to use in its place.
nlohmann::json python_converter::bind_numpy_view_temporary(
  const nlohmann::json &statement,
  const nlohmann::json &value,
  const std::string &prefix,
  codet &target_block)
{
  nlohmann::json name = {
    {"_type", "Name"},
    {"id", prefix + std::to_string(numpy_view_arg_count_++)},
    {"ctx", {{"_type", "Load"}}}};
  for (const char *key :
       {"lineno", "col_offset", "end_lineno", "end_col_offset"})
    if (value.contains(key))
      name[key] = value[key];
  nlohmann::json binding = statement;
  binding.erase("test");
  binding.erase("msg");
  binding.erase("target");
  binding.erase("op");
  binding.erase("annotation");
  binding["_type"] = "Assign";
  binding["targets"] = nlohmann::json::array({name});
  binding["targets"][0]["ctx"] = {{"_type", "Store"}};
  binding["value"] = value;
  bind_runtime_numpy_view_arguments(binding, target_block);
  get_folded_var_assign(binding, target_block);
  return name;
}

// `f(a[lo:hi])` where f binds that parameter as a view: the subscript is
// first bound to a fresh name, as `t = a[lo:hi]; f(t)`, so the callee writes
// through the view instead of a temporary. Inside a larger expression this
// holds only when no call runs before the subscript is evaluated.
bool python_converter::hoist_numpy_view_arguments(
  const std::string &key,
  nlohmann::json &rewritten,
  codet &target_block)
{
  const nlohmann::json statement = rewritten;
  std::vector<json_pointer> arguments;
  collect_subscript_arguments(
    statement[key], json_pointer("/" + key), arguments);
  bool changed = false;
  for (const json_pointer &argument : arguments)
  {
    const json_pointer call = argument.parent_pointer().parent_pointer();
    if (
      !is_bound_numpy_view_param(
        statement[call]["func"]["id"].get<std::string>(),
        std::stoul(argument.back())) ||
      is_preceded_by_call(statement[key], statement[argument]))
      continue;
    rewritten[argument] = bind_numpy_view_temporary(
      statement, statement[argument], "$numpy_view_arg$", target_block);
    changed = true;
  }
  return changed;
}

// Whether `node` holds something that must not run before the operands to
// its left are read: a call that needs binding, or a call passing a view to
// a bound view parameter, through which the callee may write.
bool python_converter::needs_numpy_view_hoist(const nlohmann::json &node)
{
  if (!node.is_object() && !node.is_array())
    return false;
  if (needs_numpy_view_call_binding(node))
    return true;
  if (is_named_call(node))
    for (std::size_t i = 0; i < node["args"].size(); ++i)
      if (
        (node["args"][i].value("_type", "") == "Subscript" ||
         is_name(node["args"][i])) &&
        is_bound_numpy_view_param(node["func"]["id"].get<std::string>(), i))
        return true;
  for (const auto &child : node)
    if (needs_numpy_view_hoist(child))
      return true;
  return false;
}

// Binds the operand at `path` to a fresh name, so that what follows it in
// the statement can be moved before the statement without overtaking it.
// False for an operand a name would not preserve: an array or object, which
// a later call may change through an alias.
bool python_converter::freeze_numpy_view_operand(
  const json_pointer &path,
  nlohmann::json &rewritten,
  codet &target_block)
{
  const nlohmann::json operand = rewritten[path];
  const std::string type = operand.value("_type", "");
  if (type == "Constant")
    return true;
  if (type == "Attribute")
    return false;
  if (type == "Name")
  {
    const std::string id = operand.value("id", "");
    if (id.rfind("$numpy_view_", 0) == 0)
      return true;
    const symbolt *symbol =
      symbol_table_.find_symbol(resolve_name_symbol_id(id));
    const typet symbol_type = symbol ? ns.follow(symbol->get_type()) : typet();
    if (!is_number(symbol_type) && !symbol_type.is_bool())
      return false;
  }
  rewritten[path] = bind_numpy_view_temporary(
    rewritten, operand, "$numpy_view_tmp$", target_block);
  return true;
}

// A call that must be a statement of its own: one returning its view
// parameter, whose result carries the view through a name, or one whose
// callee reads a view layout the caller stores right before the call.
bool python_converter::needs_numpy_view_call_binding(const nlohmann::json &node)
{
  return is_named_call(node) &&
         (returned_numpy_view_argument(node) != nullptr ||
          !runtime_numpy_view_params(node["func"]["id"].get<std::string>())
             .empty());
}

// `f(v)[0]`, `len(f(v))`, `f(v).tolist()` where f returns its view
// parameter: the call is first bound to a fresh name, which then carries the
// view. Only what the expression at `path` evaluates first is moved; the
// right operand of a binary operation follows once the left one is bound to
// a name. `skip` leaves the node at `path` itself in place.
bool python_converter::hoist_first_evaluated_numpy_view_call(
  const json_pointer &path,
  bool skip,
  nlohmann::json &rewritten,
  codet &target_block)
{
  if (!rewritten.contains(path) || !rewritten[path].is_object())
    return false;
  const nlohmann::json node = rewritten[path];
  if (!skip && needs_numpy_view_call_binding(node))
  {
    rewritten[path] = bind_numpy_view_temporary(
      rewritten, node, "$numpy_view_ret$", target_block);
    return true;
  }
  const std::string step = first_evaluated_child(node, imported_modules);
  if (step.empty())
    return false;
  const json_pointer first(path.to_string() + step);
  if (
    hoist_first_evaluated_numpy_view_call(
      first, false, rewritten, target_block))
    return true;
  const std::string type = node.value("_type", "");
  const bool binary =
    type == "BinOp" || (type == "Compare" && node["comparators"].size() == 1);
  const json_pointer second(
    path.to_string() + (type == "BinOp" ? "/right" : "/comparators/0"));
  if (!binary || !needs_numpy_view_hoist(rewritten[second]))
    return false;
  const nlohmann::json before = rewritten[first];
  if (!freeze_numpy_view_operand(first, rewritten, target_block))
    return false;
  return rewritten[first] != before ||
         hoist_first_evaluated_numpy_view_call(
           second, false, rewritten, target_block);
}

// `x = a if c else f(v[lo:hi])`: the view expression may only be built when
// its operand runs, so the assignment becomes an `if` with one assignment
// per branch, where each is handled on its own.
bool python_converter::split_conditional_numpy_view_assignment(
  const nlohmann::json &statement,
  nlohmann::json &rewritten)
{
  const std::string kind = statement.value("_type", "");
  if (
    (kind != "Assign" && kind != "AnnAssign") || !statement.contains("value") ||
    !statement["value"].is_object() ||
    statement["value"].value("_type", "") != "IfExp")
    return false;
  const nlohmann::json &choice = statement["value"];
  std::vector<json_pointer> arguments;
  for (const char *branch : {"body", "orelse"})
    collect_subscript_arguments(choice[branch], json_pointer(), arguments);
  if (arguments.empty())
    return false;
  nlohmann::json taken = statement, other = statement;
  taken["value"] = choice["body"];
  other["value"] = choice["orelse"];
  rewritten = statement;
  rewritten.erase("value");
  rewritten.erase("targets");
  rewritten.erase("target");
  rewritten.erase("annotation");
  rewritten["_type"] = "If";
  rewritten["test"] = choice["test"];
  rewritten["body"] = nlohmann::json::array({taken});
  rewritten["orelse"] = nlohmann::json::array({other});
  return true;
}

const nlohmann::json &python_converter::hoist_unnamed_numpy_view_arguments(
  const nlohmann::json &statement,
  nlohmann::json &rewritten,
  codet &target_block)
{
  if (split_conditional_numpy_view_assignment(statement, rewritten))
    return rewritten;
  const std::string key = statement_expression_key(statement);
  if (key.empty() || !needs_numpy_view_hoist(statement[key]))
    return statement;
  // A call that is the whole value of these statements is bound, discarded
  // or returned where the statement is converted.
  static const std::set<std::string> handles_root = {
    "Assign", "AnnAssign", "Expr", "Return"};
  const bool skip_root = handles_root.count(statement.value("_type", "")) != 0;
  nlohmann::json work = statement;
  // Each round moves one thing before the statement, which may let the
  // next one follow.
  for (unsigned round = 0; round < 16; ++round)
    if (
      !hoist_numpy_view_arguments(key, work, target_block) &&
      !hoist_first_evaluated_numpy_view_call(
        json_pointer("/" + key), skip_root, work, target_block))
      break;
  if (work != statement)
    rewritten = std::move(work);
  return rewritten.is_null() ? statement : rewritten;
}
