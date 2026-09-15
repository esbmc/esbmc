#include <python-frontend/runtime/runtime_converter.h>
#include <util/arith/arith_tools.h>
#include <util/expr/expr_util.h>
#include <util/irep/std_expr.h>
#include <util/lang/c_types.h>

#include <stdexcept>

namespace
{
const char *const object_tag = "tag-struct __pyrt_object";
const char *const long_tag = "tag-struct __pyrt_long";

bool is_type(const nlohmann::json &node, const char *type)
{
  return node.is_object() && node.contains("_type") && node["_type"] == type;
}

/// Py_LT..Py_GE in pyrt.h, or -1 for an operator richcompare does not take.
int richcompare_op(const std::string &op)
{
  static const std::map<std::string, int> ops = {
    {"Lt", 0}, {"LtE", 1}, {"Eq", 2}, {"NotEq", 3}, {"Gt", 4}, {"GtE", 5}};
  auto it = ops.find(op);
  return it == ops.end() ? -1 : it->second;
}
} // namespace

python_runtime_converter::python_runtime_converter(
  contextt &context,
  const nlohmann::json &ast)
  : context_(context),
    ast_(ast),
    file_(ast["filename"].get<std::string>()),
    object_type_(pointer_typet(symbol_typet(object_tag)))
{
}

void python_runtime_converter::unsupported(const json &node) const
{
  std::string what = node.contains("_type") ? node["_type"].get<std::string>()
                                            : std::string("node");
  if (node.contains("lineno") && node["lineno"].is_number())
    what += " at line " + std::to_string(node["lineno"].get<int>());
  throw std::runtime_error("unsupported under --python-runtime: " + what);
}

locationt python_runtime_converter::location(const json &node) const
{
  locationt loc;
  loc.set_file(file_);
  if (node.contains("lineno") && node["lineno"].is_number())
    loc.set_line(node["lineno"].get<unsigned>());
  if (node.contains("col_offset") && node["col_offset"].is_number())
    loc.set_column(node["col_offset"].get<unsigned>());
  if (!function_.empty())
    loc.set_function(function_);
  return loc;
}

const symbolt &python_runtime_converter::lookup(const std::string &id) const
{
  const symbolt *symbol = context_.find_symbol(id);
  if (!symbol)
    throw std::runtime_error("--python-runtime: symbol " + id + " is missing");
  return *symbol;
}

symbolt &python_runtime_converter::add_symbol(symbolt &symbol)
{
  symbol.mode = "Python";
  symbol.module = file_;
  if (symbol.name.empty())
    symbol.name = symbol.id;
  return *context_.move_symbol_to_context(symbol);
}

std::string python_runtime_converter::global_id(const std::string &name) const
{
  return "py:" + file_ + "@" + name;
}

std::string python_runtime_converter::function_id(const std::string &name) const
{
  return "py:" + file_ + "@F@" + name;
}

std::string python_runtime_converter::local_id(
  const std::string &function,
  const std::string &name) const
{
  return function_id(function) + "@" + name;
}

exprt python_runtime_converter::runtime_object(const std::string &name) const
{
  return typecast_exprt(
    address_of_exprt(symbol_expr(lookup("c:@" + name))), object_type_);
}

exprt python_runtime_converter::new_temporary(
  const typet &type,
  const locationt &loc)
{
  symbolt symbol;
  symbol.id =
    local_id(function_.empty() ? "python_user_main" : function_, "$pyrt$") +
    std::to_string(temporaries_++);
  symbol.set_type(type);
  symbol.location = loc;
  symbol.lvalue = true;
  symbol.file_local = true;
  exprt temporary = symbol_expr(add_symbol(symbol));

  code_declt decl(temporary);
  decl.location() = loc;
  block_->copy_to_operands(decl);
  return temporary;
}

exprt python_runtime_converter::call(
  const std::string &function,
  const std::vector<exprt> &arguments,
  const locationt &loc)
{
  const symbolt &callee = lookup("c:@F@" + function);
  code_function_callt call;
  call.function() = symbol_expr(callee);
  call.arguments() = arguments;
  call.location() = loc;

  exprt result = nil_exprt();
  const typet &return_type = to_code_type(callee.get_type()).return_type();
  if (return_type.id() != "empty")
  {
    result = new_temporary(return_type, loc);
    call.lhs() = result;
  }
  block_->copy_to_operands(call);
  return result;
}

void python_runtime_converter::raise(
  const std::string &message,
  const locationt &loc)
{
  code_assertt assertion;
  assertion.assertion() = false_exprt();
  assertion.location() = loc;
  assertion.location().comment(message);
  block_->copy_to_operands(assertion);

  codet assumption("assume");
  assumption.copy_to_operands(false_exprt());
  assumption.location() = loc;
  block_->copy_to_operands(assumption);
}

void python_runtime_converter::emit_if(
  const exprt &cond,
  codet then_case,
  const locationt &loc)
{
  code_ifthenelset branch;
  branch.cond() = cond;
  branch.then_case() = std::move(then_case);
  branch.location() = loc;
  block_->copy_to_operands(branch);
}

exprt python_runtime_converter::expr(const json &node)
{
  const std::string type = node["_type"];
  if (type == "Constant")
    return constant(node);
  if (type == "Name")
    return name(node);
  if (type == "BinOp")
    return binop(node);
  if (type == "UnaryOp")
    return unaryop(node);
  if (type == "Compare")
    return compare(node);
  if (type == "BoolOp")
    return boolop(node);
  if (type == "Call")
    return call_expr(node);
  if (type == "List")
    return list(node);
  if (type == "Subscript")
    return subscript(node);
  unsupported(node);
}

exprt python_runtime_converter::truth(const json &node)
{
  return call("pyrt_is_true", {expr(node)}, location(node));
}

exprt python_runtime_converter::constant(const json &node)
{
  const json &value = node["value"];
  if (value.is_null())
    return runtime_object("pyrt_None");
  if (value.is_boolean())
    return runtime_object(value.get<bool>() ? "pyrt_True" : "pyrt_False");
  if (
    value.is_number_integer() && !node.contains("_bigint") &&
    (!value.is_number_unsigned() ||
     value.get<uint64_t>() <= uint64_t(INT64_MAX)))
    return int_constant(value.get<int64_t>(), location(node));
  unsupported(node);
}

/// An int literal is one static object per value, so evaluating it allocates
/// nothing and every occurrence shares the same address.
exprt python_runtime_converter::int_constant(
  int64_t value,
  const locationt &loc)
{
  const std::string id = global_id("$pyrt_int$" + std::to_string(value));
  if (!context_.find_symbol(id))
  {
    const struct_typet &layout = to_struct_type(lookup(long_tag).get_type());
    struct_exprt init{symbol_typet(long_tag)};
    init.copy_to_operands(typecast_exprt(
      address_of_exprt(symbol_expr(lookup("c:@PyRtLong_Type"))),
      layout.components()[0].type()));
    init.copy_to_operands(from_integer(value, layout.components()[1].type()));

    symbolt symbol;
    symbol.id = id;
    symbol.set_type(symbol_typet(long_tag));
    symbol.set_value(init);
    symbol.location = loc;
    symbol.lvalue = true;
    symbol.static_lifetime = true;
    add_symbol(symbol);
  }
  return typecast_exprt(
    address_of_exprt(symbol_expr(lookup(id))), object_type_);
}

exprt python_runtime_converter::name(const json &node)
{
  const std::string id = node["id"];
  if (locals_.count(id))
    return symbol_expr(lookup(local_id(function_, id)));
  if (globals_.count(id))
    return symbol_expr(lookup(global_id(id)));
  raise("NameError: name '" + id + "' is not defined", location(node));
  return gen_zero(object_type_);
}

exprt python_runtime_converter::binop(const json &node)
{
  static const std::map<std::string, std::string> ops = {
    {"Add", "pyrt_number_add"},
    {"Sub", "pyrt_number_subtract"},
    {"Mult", "pyrt_number_multiply"}};
  auto it = ops.find(node["op"]["_type"]);
  if (it == ops.end())
    unsupported(node);
  exprt left = expr(node["left"]);
  exprt right = expr(node["right"]);
  return call(it->second, {left, right}, location(node));
}

exprt python_runtime_converter::unaryop(const json &node)
{
  const std::string op = node["op"]["_type"];
  const locationt loc = location(node);
  if (op == "USub")
    return call("pyrt_number_negative", {expr(node["operand"])}, loc);
  if (op == "Not")
    return call("pyrt_bool_from", {not_exprt(truth(node["operand"]))}, loc);
  unsupported(node);
}

exprt python_runtime_converter::compare(const json &node)
{
  if (node["ops"].size() != 1)
    unsupported(node);
  const locationt loc = location(node);
  exprt left = expr(node["left"]);
  exprt right = expr(node["comparators"][0]);

  const std::string op = node["ops"][0]["_type"];
  if (op == "Is" || op == "IsNot")
  {
    exprt same = equality_exprt(left, right);
    return call("pyrt_bool_from", {op == "Is" ? same : not_exprt(same)}, loc);
  }
  int richcompare = richcompare_op(op);
  if (richcompare < 0)
    unsupported(node);
  return call(
    "pyrt_richcompare",
    {left, right, from_integer(richcompare, int_type())},
    loc);
}

/// `a and b` is `a` when `a` is false and `b` otherwise, evaluating `b` only
/// in the second case; `or` mirrors it.
exprt python_runtime_converter::boolop(const json &node)
{
  const locationt loc = location(node);
  exprt result = new_temporary(object_type_, loc);
  const json &values = node["values"];
  block_->copy_to_operands(code_assignt(result, expr(values[0])));
  boolop_rest(values, 1, result, node["op"]["_type"] == "And", loc);
  return result;
}

void python_runtime_converter::boolop_rest(
  const json &values,
  size_t index,
  const exprt &result,
  bool is_and,
  const locationt &loc)
{
  if (index == values.size())
    return;
  exprt is_true = call("pyrt_is_true", {result}, loc);

  code_blockt next;
  code_blockt *outer = block_;
  block_ = &next;
  block_->copy_to_operands(code_assignt(result, expr(values[index])));
  boolop_rest(values, index + 1, result, is_and, loc);
  block_ = outer;

  emit_if(is_and ? is_true : not_exprt(is_true), next, loc);
}

exprt python_runtime_converter::call_expr(const json &node)
{
  if (!node["keywords"].empty())
    unsupported(node);
  const json &func = node["func"];
  const locationt loc = location(node);

  if (is_type(func, "Attribute"))
  {
    if (func["attr"] != "append" || node["args"].size() != 1)
      unsupported(node);
    exprt receiver = expr(func["value"]);
    call("pyrt_list_append", {receiver, expr(node["args"][0])}, loc);
    return runtime_object("pyrt_None");
  }
  if (!is_type(func, "Name"))
    unsupported(node);

  std::vector<exprt> arguments;
  for (const json &arg : node["args"])
  {
    if (is_type(arg, "Starred"))
      unsupported(arg);
    arguments.push_back(expr(arg));
  }

  const std::string callee = func["id"];
  if (locals_.count(callee) || globals_.count(callee))
    unsupported(node);

  auto function = functions_.find(callee);
  if (function != functions_.end())
  {
    const size_t arity = (*function->second)["args"]["args"].size();
    if (arguments.size() != arity)
    {
      raise(
        "TypeError: " + callee + "() takes " + std::to_string(arity) +
          " positional arguments but " + std::to_string(arguments.size()) +
          " were given",
        loc);
      return runtime_object("pyrt_None");
    }
    exprt result = new_temporary(object_type_, loc);
    code_function_callt call;
    call.function() = symbol_expr(lookup(function_id(callee)));
    call.arguments() = arguments;
    call.lhs() = result;
    call.location() = loc;
    block_->copy_to_operands(call);
    return result;
  }
  if (callee == "len" && arguments.size() == 1)
    return call("pyrt_builtin_len", arguments, loc);
  if (callee == "print")
    return runtime_object("pyrt_None");
  unsupported(node);
}

exprt python_runtime_converter::list(const json &node)
{
  const locationt loc = location(node);
  exprt result = call("pyrt_list_new", {}, loc);
  for (const json &element : node["elts"])
    call("pyrt_list_append", {result, expr(element)}, loc);
  return result;
}

exprt python_runtime_converter::subscript(const json &node)
{
  if (is_type(node["slice"], "Slice"))
    unsupported(node);
  exprt container = expr(node["value"]);
  exprt key = expr(node["slice"]);
  return call("pyrt_getitem", {container, key}, location(node));
}

void python_runtime_converter::statements(const json &body, code_blockt &block)
{
  code_blockt *outer = block_;
  block_ = &block;
  for (const json &node : body)
    statement(node);
  block_ = outer;
}

void python_runtime_converter::statement(const json &node)
{
  const std::string type = node["_type"];
  const locationt loc = location(node);
  if (type == "Expr")
    expr(node["value"]);
  else if (type == "Assign")
  {
    exprt value = expr(node["value"]);
    for (const json &target : node["targets"])
      store(target, value, loc);
  }
  else if (type == "AugAssign")
  {
    if (!is_type(node["target"], "Name"))
      unsupported(node);
    json binop_node = node;
    binop_node["_type"] = "BinOp";
    binop_node["left"] = node["target"];
    binop_node["right"] = node["value"];
    store(node["target"], binop(binop_node), loc);
  }
  else if (type == "If")
    if_statement(node);
  else if (type == "While")
    while_statement(node);
  else if (type == "Assert")
    assert_statement(node);
  else if (type == "Return")
  {
    if (function_.empty())
      unsupported(node);
    code_returnt ret;
    ret.return_value() = node["value"].is_null() ? runtime_object("pyrt_None")
                                                 : expr(node["value"]);
    ret.location() = loc;
    block_->copy_to_operands(ret);
  }
  else if (type == "Break")
    block_->copy_to_operands(code_breakt());
  else if (type == "Continue")
    block_->copy_to_operands(code_continuet());
  else if (type == "FunctionDef" && function_.empty())
    return;
  else if (type != "Pass" && type != "Global")
    unsupported(node);
}

void python_runtime_converter::store(
  const json &target,
  const exprt &value,
  const locationt &loc)
{
  if (is_type(target, "Name"))
  {
    const std::string id = target["id"];
    const std::string symbol =
      locals_.count(id) ? local_id(function_, id) : global_id(id);
    code_assignt assign(symbol_expr(lookup(symbol)), value);
    assign.location() = loc;
    block_->copy_to_operands(assign);
  }
  else if (is_type(target, "Subscript") && !is_type(target["slice"], "Slice"))
  {
    exprt container = expr(target["value"]);
    exprt key = expr(target["slice"]);
    call("pyrt_setitem", {container, key, value}, loc);
  }
  else
    unsupported(target);
}

void python_runtime_converter::if_statement(const json &node)
{
  const locationt loc = location(node);
  code_ifthenelset branch;
  branch.cond() = truth(node["test"]);
  code_blockt then_case;
  statements(node["body"], then_case);
  branch.then_case() = then_case;
  if (!node["orelse"].empty())
  {
    code_blockt else_case;
    statements(node["orelse"], else_case);
    branch.else_case() = else_case;
  }
  branch.location() = loc;
  block_->copy_to_operands(branch);
}

/// The test is re-evaluated inside the loop body, ahead of the user
/// statements, so a `continue` reaches it again.
void python_runtime_converter::while_statement(const json &node)
{
  if (!node["orelse"].empty())
    unsupported(node);
  const locationt loc = location(node);

  code_blockt body;
  code_blockt *outer = block_;
  block_ = &body;
  emit_if(not_exprt(truth(node["test"])), code_breakt(), loc);
  block_ = outer;
  statements(node["body"], body);

  codet loop;
  loop.set_statement("while");
  loop.copy_to_operands(true_exprt(), body);
  loop.location() = loc;
  block_->copy_to_operands(loop);
}

void python_runtime_converter::assert_statement(const json &node)
{
  code_assertt assertion;
  assertion.assertion() = truth(node["test"]);
  assertion.location() = location(node);
  const json &msg = node["msg"];
  assertion.location().comment(
    is_type(msg, "Constant") && msg["value"].is_string()
      ? msg["value"].get<std::string>()
      : std::string("assertion"));
  block_->copy_to_operands(assertion);
}

void python_runtime_converter::collect_assigned(
  const json &body,
  std::set<std::string> &assigned,
  std::set<std::string> &declared_global) const
{
  for (const json &node : body)
  {
    if (is_type(node, "Assign"))
    {
      for (const json &target : node["targets"])
        if (is_type(target, "Name"))
          assigned.insert(target["id"].get<std::string>());
    }
    else if (is_type(node, "AugAssign") && is_type(node["target"], "Name"))
      assigned.insert(node["target"]["id"].get<std::string>());
    else if (is_type(node, "Global"))
      for (const json &global : node["names"])
        declared_global.insert(global.get<std::string>());
    else if (is_type(node, "If") || is_type(node, "While"))
    {
      collect_assigned(node["body"], assigned, declared_global);
      collect_assigned(node["orelse"], assigned, declared_global);
    }
  }
}

void python_runtime_converter::declare_variable(
  const std::string &id,
  const std::string &name,
  bool is_global,
  const locationt &loc)
{
  symbolt symbol;
  symbol.id = id;
  symbol.name = name;
  symbol.set_type(object_type_);
  symbol.location = loc;
  symbol.lvalue = true;
  symbol.static_lifetime = is_global;
  symbol.file_local = !is_global;
  if (is_global)
    symbol.set_value(gen_zero(object_type_));
  add_symbol(symbol);
}

void python_runtime_converter::declare_function(const json &def)
{
  const json &args = def["args"];
  if (
    !def["decorator_list"].empty() || !args["posonlyargs"].empty() ||
    !args["kwonlyargs"].empty() || !args["defaults"].empty() ||
    !args["vararg"].is_null() || !args["kwarg"].is_null())
    unsupported(def);

  const std::string name = def["name"];
  const locationt loc = location(def);
  code_typet type;
  type.return_type() = object_type_;
  for (const json &arg : args["args"])
  {
    const std::string arg_name = arg["arg"];
    const std::string arg_id = local_id(name, arg_name);
    code_typet::argumentt parameter(object_type_);
    parameter.set_identifier(arg_id);
    parameter.set_base_name(arg_name);
    type.arguments().push_back(parameter);
    declare_variable(arg_id, arg_name, false, loc);
  }

  symbolt symbol;
  symbol.id = function_id(name);
  symbol.name = name;
  symbol.set_type(type);
  symbol.location = loc;
  symbol.lvalue = true;
  add_symbol(symbol);
  functions_[name] = &def;
}

void python_runtime_converter::define_function(const json &def)
{
  const std::string name = def["name"];
  function_ = name;

  std::set<std::string> assigned, declared_global;
  collect_assigned(def["body"], assigned, declared_global);
  locals_.clear();
  for (const json &arg : def["args"]["args"])
    locals_.insert(arg["arg"].get<std::string>());

  code_blockt body;
  block_ = &body;
  const locationt loc = location(def);
  for (const std::string &local : assigned)
  {
    if (declared_global.count(local) || locals_.count(local))
      continue;
    const std::string id = local_id(name, local);
    declare_variable(id, local, false, loc);
    locals_.insert(local);
    exprt variable = symbol_expr(lookup(id));
    body.copy_to_operands(code_declt(variable));
    body.copy_to_operands(code_assignt(variable, gen_zero(object_type_)));
  }

  statements(def["body"], body);
  code_returnt fall_off;
  fall_off.return_value() = runtime_object("pyrt_None");
  body.copy_to_operands(fall_off);

  context_.find_symbol(function_id(name))->set_value(body);
  function_.clear();
  locals_.clear();
  block_ = nullptr;
}

/// The memory model's globals, which the Python symbol table does not get from
/// the C models (cf. python_converter::load_c_intrisics).
void python_runtime_converter::add_c_intrinsics()
{
  const std::pair<const char *, typet> intrinsics[] = {
    {"__ESBMC_alloc", array_typet(bool_type(), exprt("infinity"))},
    {"__ESBMC_is_dynamic", array_typet(bool_type(), exprt("infinity"))},
    {"__ESBMC_alloc_size", array_typet(size_type(), exprt("infinity"))}};
  for (const auto &[name, type] : intrinsics)
  {
    symbolt symbol;
    symbol.id = std::string("c:@") + name;
    symbol.name = name;
    symbol.mode = "C";
    symbol.set_type(type);
    symbol.lvalue = true;
    symbol.static_lifetime = true;
    exprt zero = gen_zero(type, true);
    zero.zero_initializer(true);
    symbol.set_value(zero);
    context_.move_symbol_to_context(symbol);
  }

  code_typet void_void;
  void_void.return_type() = empty_typet();
  symbolt yield;
  yield.id = "c:@F@__ESBMC_yield";
  yield.name = "__ESBMC_yield";
  yield.mode = "C";
  yield.set_type(void_void);
  if (!context_.find_symbol(yield.id))
    context_.move_symbol_to_context(yield);
}

void python_runtime_converter::add_entry_points(code_blockt &user_code)
{
  const locationt loc = location(ast_);
  code_typet void_void;
  void_void.return_type() = empty_typet();

  symbolt user_main;
  user_main.id = user_main.name = "python_user_main";
  user_main.set_type(void_void);
  user_main.lvalue = true;
  user_main.location = loc;
  user_main.set_value(user_code);
  context_.move_symbol_to_context(user_main);

  code_blockt main_body;
  context_.foreach_operand_in_order([&main_body](const symbolt &s) {
    if (s.static_lifetime && !s.get_value().is_nil() && !s.get_type().is_code())
    {
      code_assignt assign(symbol_expr(s), s.get_value());
      assign.location() = s.location;
      main_body.copy_to_operands(assign);
    }
  });
  code_function_callt call_user_main;
  call_user_main.function() = symbol_expr(lookup("python_user_main"));
  main_body.copy_to_operands(call_user_main);
  main_body.end_location(loc);

  symbolt main;
  main.id = main.name = "__ESBMC_main";
  main.set_type(void_void);
  main.lvalue = true;
  main.location = loc;
  main.set_value(main_body);
  if (context_.move(main))
    throw std::runtime_error(
      "The main function is already defined in another module");
}

void python_runtime_converter::convert()
{
  if (!context_.find_symbol(object_tag))
    throw std::runtime_error(
      "--python-runtime: the runtime models are missing");
  add_c_intrinsics();

  const json &body = ast_["body"];
  std::set<std::string> declared_global;
  collect_assigned(body, globals_, declared_global);
  for (const json &node : body)
    if (is_type(node, "FunctionDef"))
    {
      std::set<std::string> assigned;
      collect_assigned(node["body"], assigned, declared_global);
      for (const std::string &name : assigned)
        if (declared_global.count(name))
          globals_.insert(name);
    }

  const locationt loc = location(ast_);
  for (const std::string &name : globals_)
    declare_variable(global_id(name), name, true, loc);

  for (const json &node : body)
    if (is_type(node, "FunctionDef"))
    {
      if (globals_.count(node["name"]))
        unsupported(node);
      declare_function(node);
    }
  for (const auto &[name, def] : functions_)
    define_function(*def);

  code_blockt user_code;
  statements(body, user_code);
  add_entry_points(user_code);
}
