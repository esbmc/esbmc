#include <python-frontend/runtime/runtime_converter.h>
#include <util/arith/arith_tools.h>
#include <util/expr/expr_util.h>
#include <util/irep/std_expr.h>
#include <util/expr/string_constant.h>
#include <util/lang/c_types.h>

#include <algorithm>
#include <optional>
#include <stdexcept>

namespace
{
const char *const object_tag = "tag-struct __pyrt_object";
const char *const long_tag = "tag-struct __pyrt_long";
const char *const type_tag = "tag-struct __pyrt_type";
const char *const attrs_tag = "tag-struct __pyrt_attrs";
const char *const function_tag = "tag-struct __pyrt_function";
const char *const str_tag = "tag-struct __pyrt_str";
const char *const float_tag = "tag-struct __pyrt_float";
const char *const args_tag = "tag-struct __pyrt_args";
const size_t max_args = 6;
const unsigned heaptype_flag = 1;

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

std::optional<int64_t> literal_int(const nlohmann::json &node)
{
  if (
    is_type(node, "Constant") && node["value"].is_number_integer() &&
    !node.contains("_bigint"))
    return node["value"].get<int64_t>();
  if (is_type(node, "UnaryOp") && node["op"]["_type"] == "USub")
    if (auto inner = literal_int(node["operand"]))
      return -*inner;
  return std::nullopt;
}

/// Type object for a builtin name, or "" when the name does not name one.
std::string builtin_type_symbol(const std::string &name)
{
  static const std::map<std::string, std::string> types = {
    {"int", "c:@PyRtLong_Type"},
    {"bool", "c:@PyRtBool_Type"},
    {"list", "c:@PyRtList_Type"},
    {"dict", "c:@PyRtDict_Type"},
    {"str", "c:@PyRtStr_Type"},
    {"float", "c:@PyRtFloat_Type"},
    {"object", "c:@PyRtObject_Type"},
    {"type", "c:@PyRtType_Type"}};
  auto it = types.find(name);
  return it == types.end() ? std::string() : it->second;
}

std::string binop_function(const std::string &op)
{
  static const std::map<std::string, std::string> ops = {
    {"Add", "pyrt_number_add"},
    {"Sub", "pyrt_number_subtract"},
    {"Mult", "pyrt_number_multiply"},
    {"Div", "pyrt_number_true_divide"},
    {"FloorDiv", "pyrt_number_floor_divide"},
    {"Mod", "pyrt_number_remainder"},
    {"Pow", "pyrt_number_power"}};
  auto it = ops.find(op);
  return it == ops.end() ? std::string() : it->second;
}
} // namespace

python_runtime_converter::python_runtime_converter(
  contextt &context,
  const nlohmann::json &ast,
  bool check_annotations)
  : context_(context),
    ast_(ast),
    file_(ast["filename"].get<std::string>()),
    object_type_(pointer_typet(symbol_typet(object_tag))),
    check_annotations_(check_annotations)
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
  if (!function_name_.empty())
    loc.set_function(function_name_);
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

std::string python_runtime_converter::code_id(
  const std::string &cls,
  const std::string &name) const
{
  return "py:" + file_ + (cls.empty() ? "" : "@C@" + cls) + "@F@" + name;
}

std::string python_runtime_converter::local_id(const std::string &name) const
{
  return (code_id_.empty() ? code_id("", "python_user_main") : code_id_) +
         "@" + name;
}

std::string python_runtime_converter::function_object_id(
  const std::string &cls,
  const std::string &name) const
{
  return global_id("$pyrt_function$" + (cls.empty() ? name : cls + "." + name));
}

std::string
python_runtime_converter::type_object_id(const std::string &cls) const
{
  return global_id("$pyrt_type$" + cls);
}

exprt python_runtime_converter::address(const std::string &id) const
{
  return typecast_exprt(
    address_of_exprt(symbol_expr(lookup(id))), object_type_);
}

exprt python_runtime_converter::type_pointer(const std::string &cls) const
{
  return typecast_exprt(
    address_of_exprt(symbol_expr(lookup(type_object_id(cls)))),
    pointer_typet(symbol_typet(type_tag)));
}

/// Attribute names compare by address: the runtime interns the ones it looks
/// for itself (pyrt_names.c), and every other name gets one symbol here.
exprt python_runtime_converter::name_pointer(const std::string &name)
{
  const typet char_pointer = pointer_typet(char_type());
  if (const symbolt *interned = context_.find_symbol("c:@pyrt_str_" + name))
  {
    exprt first(exprt::index, interned->get_type().subtype());
    first.copy_to_operands(
      symbol_expr(*interned), from_integer(0, index_type()));
    return typecast_exprt(address_of_exprt(first), char_pointer);
  }

  const std::string id = global_id("$pyrt_str$" + name);
  if (!context_.find_symbol(id))
    declare_variable(id, name, char_type(), true, locationt());
  return typecast_exprt(
    address_of_exprt(symbol_expr(lookup(id))), char_pointer);
}

exprt python_runtime_converter::struct_value(
  const char *tag,
  const std::map<std::string, exprt> &fields) const
{
  const struct_typet &layout = to_struct_type(lookup(tag).get_type());
  struct_exprt value{symbol_typet(tag)};
  for (const auto &component : layout.components())
  {
    auto field = fields.find(component.get_name().as_string());
    if (field != fields.end())
    {
      value.copy_to_operands(typecast_exprt(field->second, component.type()));
      continue;
    }
    exprt zero = gen_zero(component.type(), true);
    zero.zero_initializer(true);
    value.copy_to_operands(zero);
  }
  return value;
}

void python_runtime_converter::add_static_object(
  const std::string &id,
  const char *tag,
  const locationt &loc)
{
  symbolt symbol;
  symbol.id = id;
  symbol.set_type(symbol_typet(tag));
  symbol.location = loc;
  symbol.lvalue = true;
  symbol.static_lifetime = true;
  add_symbol(symbol);
}

exprt python_runtime_converter::new_temporary(
  const typet &type,
  const locationt &loc)
{
  symbolt symbol;
  symbol.id = local_id("$pyrt$") + std::to_string(temporaries_++);
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

std::vector<exprt> python_runtime_converter::arguments(const json &call_node)
{
  if (!call_node["keywords"].empty())
    unsupported(call_node);
  std::vector<exprt> values;
  for (const json &arg : call_node["args"])
  {
    if (is_type(arg, "Starred"))
      unsupported(arg);
    values.push_back(expr(arg));
  }
  return values;
}

exprt python_runtime_converter::arguments_struct(
  const std::vector<exprt> &arguments,
  const json &call_node) const
{
  if (arguments.size() > max_args)
    unsupported(call_node);
  std::map<std::string, exprt> fields;
  for (size_t i = 0; i < arguments.size(); ++i)
    fields["a" + std::to_string(i)] = arguments[i];
  return struct_value(args_tag, fields);
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
  {
    exprt left = expr(node["left"]);
    exprt right = expr(node["right"]);
    return binop(node["op"]["_type"], left, right, node);
  }
  if (type == "UnaryOp")
    return unaryop(node);
  if (type == "Compare")
    return compare(node);
  if (type == "BoolOp")
    return boolop(node);
  if (type == "IfExp")
    return ifexp(node);
  if (type == "Call")
    return call_expr(node);
  if (type == "List")
    return list(node);
  if (type == "Dict")
    return dict_literal(node);
  if (type == "Subscript")
    return subscript(node);
  if (type == "Attribute")
  {
    exprt object = expr(node["value"]);
    return call(
      "pyrt_getattr", {object, name_pointer(node["attr"])}, location(node));
  }
  unsupported(node);
}

exprt python_runtime_converter::truth(const json &node)
{
  return call("pyrt_is_true", {expr(node)}, location(node));
}

exprt python_runtime_converter::constant(const json &node)
{
  const json &value = node["value"];
  /* JSON has no bytes or complex, so ast2json spells both as plain strings
   * indistinguishable from str: b"ab" arrives as "ab" and 0j as "0j", which
   * would box as a non-empty -- hence truthy -- str. The parser tags them. */
  if (node.contains("esbmc_type_annotation"))
  {
    const std::string kind = node["esbmc_type_annotation"];
    if (kind == "bytes" || kind == "complex")
      unsupported(node);
  }
  if (value.is_null())
    return address("c:@pyrt_None");
  if (value.is_boolean())
    return address(value.get<bool>() ? "c:@pyrt_True" : "c:@pyrt_False");
  if (value.is_string())
    return str_constant(value.get<std::string>(), location(node));
  if (value.is_number_float() && !node.contains("value_nonfinite"))
    return float_constant(value.get<double>(), location(node));
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
    add_static_object(id, long_tag, loc);
    context_.find_symbol(id)->set_value(struct_value(
      long_tag,
      {{"ob_type", address_of_exprt(symbol_expr(lookup("c:@PyRtLong_Type")))},
       {"value", from_integer(value, long_long_int_type())}}));
  }
  return address(id);
}

/// One static object per distinct literal, with its characters in a static
/// array beside it, so evaluating a literal allocates nothing.
exprt python_runtime_converter::str_constant(
  const std::string &value,
  const locationt &loc)
{
  auto existing = string_literals_.find(value);
  if (existing != string_literals_.end())
    return address(existing->second);

  const std::string index = std::to_string(string_literals_.size());
  const std::string data_id = global_id("$pyrt_strdata$" + index);
  const std::string object_id = global_id("$pyrt_strlit$" + index);

  string_constantt characters(value);
  symbolt data;
  data.id = data_id;
  data.name = data_id;
  data.set_type(characters.type());
  data.set_value(characters);
  data.location = loc;
  data.lvalue = true;
  data.static_lifetime = true;
  add_symbol(data);

  exprt first(exprt::index, char_type());
  first.copy_to_operands(
    symbol_expr(lookup(data_id)), from_integer(0, index_type()));

  add_static_object(object_id, str_tag, loc);
  context_.find_symbol(object_id)->set_value(struct_value(
    str_tag,
    {{"ob_type", address_of_exprt(symbol_expr(lookup("c:@PyRtStr_Type")))},
     {"length", from_integer(value.size(), long_long_int_type())},
     {"data", address_of_exprt(first)}}));
  string_literals_[value] = object_id;
  return address(object_id);
}

exprt python_runtime_converter::float_constant(
  double value,
  const locationt &loc)
{
  auto existing = float_literals_.find(value);
  if (existing != float_literals_.end())
    return address(existing->second);

  const std::string id =
    global_id("$pyrt_floatlit$" + std::to_string(float_literals_.size()));
  add_static_object(id, float_tag, loc);
  context_.find_symbol(id)->set_value(struct_value(
    float_tag,
    {{"ob_type", address_of_exprt(symbol_expr(lookup("c:@PyRtFloat_Type")))},
     {"value", from_double(value, double_type())}}));
  float_literals_[value] = id;
  return address(id);
}

exprt python_runtime_converter::name(const json &node)
{
  const std::string id = node["id"];
  if (locals_.count(id))
    return symbol_expr(lookup(local_id(id)));
  if (globals_.count(id))
    return symbol_expr(lookup(global_id(id)));
  if (functions_.count(id))
    return address(function_object_id("", id));
  const std::string builtin = builtin_type_symbol(id);
  if (!builtin.empty())
    return address(builtin);
  raise("NameError: name '" + id + "' is not defined", location(node));
  return gen_zero(object_type_);
}

exprt python_runtime_converter::binop(
  const std::string &op,
  exprt left,
  exprt right,
  const json &node)
{
  const std::string function = binop_function(op);
  if (function.empty())
    unsupported(node);
  return call(function, {left, right}, location(node));
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
  if (op == "In" || op == "NotIn")
  {
    exprt contains = call("pyrt_contains", {right, left}, loc);
    if (op == "In")
      return contains;
    return call(
      "pyrt_bool_from",
      {not_exprt(call("pyrt_is_true", {contains}, loc))},
      loc);
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

/// `a if c else b` evaluates only the arm it takes, so each arm's code goes in
/// its own block.
exprt python_runtime_converter::ifexp(const json &node)
{
  const locationt loc = location(node);
  exprt result = new_temporary(object_type_, loc);
  exprt condition = truth(node["test"]);

  code_blockt then_case, else_case;
  code_blockt *outer = block_;
  block_ = &then_case;
  block_->copy_to_operands(code_assignt(result, expr(node["body"])));
  block_ = &else_case;
  block_->copy_to_operands(code_assignt(result, expr(node["orelse"])));
  block_ = outer;

  code_ifthenelset branch;
  branch.cond() = condition;
  branch.then_case() = then_case;
  branch.else_case() = else_case;
  branch.location() = loc;
  block_->copy_to_operands(branch);
  return result;
}

exprt python_runtime_converter::call_expr(const json &node)
{
  const json &func = node["func"];
  const locationt loc = location(node);
  const auto argc = [](const std::vector<exprt> &values) {
    return from_integer(values.size(), long_long_int_type());
  };

  if (is_type(func, "Attribute"))
  {
    exprt receiver = expr(func["value"]);
    std::vector<exprt> values = arguments(node);
    return call(
      "pyrt_call_method",
      {receiver,
       name_pointer(func["attr"]),
       arguments_struct(values, node),
       argc(values)},
      loc);
  }

  if (is_type(func, "Name"))
  {
    const std::string callee = func["id"];
    if (!locals_.count(callee) && !globals_.count(callee))
    {
      auto function = functions_.find(callee);
      if (function != functions_.end())
      {
        std::vector<exprt> values = arguments(node);
        const size_t arity = (*function->second)["args"]["args"].size();
        if (values.size() != arity)
        {
          raise(
            "TypeError: " + callee + "() takes " + std::to_string(arity) +
              " positional arguments but " + std::to_string(values.size()) +
              " were given",
            loc);
          return address("c:@pyrt_None");
        }
        exprt result = new_temporary(object_type_, loc);
        code_function_callt direct;
        direct.function() = symbol_expr(lookup(code_id("", callee)));
        direct.arguments().push_back(arguments_struct(values, node));
        direct.lhs() = result;
        direct.location() = loc;
        block_->copy_to_operands(direct);
        return result;
      }
      if (callee == "type" && node["args"].size() == 1)
        return call("pyrt_type_of", {expr(node["args"][0])}, loc);
      if (callee == "isinstance" && node["args"].size() == 2)
      {
        exprt object = expr(node["args"][0]);
        exprt cls = expr(node["args"][1]);
        return call(
          "pyrt_bool_from", {call("pyrt_isinstance", {object, cls}, loc)}, loc);
      }
      if (
        (callee == "getattr" || callee == "hasattr" || callee == "setattr") &&
        node["args"].size() >= 2)
      {
        const json &attribute = node["args"][1];
        if (!(is_type(attribute, "Constant") && attribute["value"].is_string()))
          unsupported(node);
        exprt object = expr(node["args"][0]);
        exprt name = name_pointer(attribute["value"].get<std::string>());
        if (callee == "getattr" && node["args"].size() == 2)
          return call("pyrt_getattr", {object, name}, loc);
        if (callee == "hasattr" && node["args"].size() == 2)
          return call(
            "pyrt_bool_from",
            {call("pyrt_hasattr", {object, name}, loc)},
            loc);
        if (callee == "setattr" && node["args"].size() == 3)
        {
          call("pyrt_setattr", {object, name, expr(node["args"][2])}, loc);
          return address("c:@pyrt_None");
        }
        unsupported(node);
      }
      if (callee == "nondet_bool" || callee == "nondet_int")
        return call("pyrt_" + callee, {}, loc);
      if (callee == "len")
      {
        std::vector<exprt> values = arguments(node);
        if (values.size() != 1)
          unsupported(node);
        return call("pyrt_builtin_len", values, loc);
      }
      if (callee == "print")
      {
        arguments(node);
        return address("c:@pyrt_None");
      }
      if (callee == "bool" && node["args"].size() == 1)
        return call("pyrt_bool_from", {truth(node["args"][0])}, loc);
      static const std::map<std::string, std::string> iterable_builtins = {
        {"abs", "pyrt_builtin_abs"},
        {"all", "pyrt_builtin_all"},
        {"any", "pyrt_builtin_any"},
        {"sum", "pyrt_builtin_sum"},
        {"min", "pyrt_builtin_min_iter"},
        {"max", "pyrt_builtin_max_iter"}};
      auto builtin = iterable_builtins.find(callee);
      if (builtin != iterable_builtins.end())
      {
        std::vector<exprt> values = arguments(node);
        if (values.size() == 1)
          return call(builtin->second, values, loc);
        if ((callee == "min" || callee == "max") && values.size() >= 2)
        {
          const std::string fold =
            callee == "min" ? "pyrt_builtin_min2" : "pyrt_builtin_max2";
          exprt result = values[0];
          for (size_t i = 1; i < values.size(); ++i)
            result = call(fold, {result, values[i]}, loc);
          return result;
        }
        if (callee == "sum" && values.size() == 2)
          return call("pyrt_builtin_sum_start", values, loc);
        /* A bad arity is a TypeError in Python, so it stays a claim. Refusing
         * the conversion here would reject the whole program instead. */
        raise("TypeError: bad argument count for " + callee + "()", loc);
        return address("c:@pyrt_None");
      }
    }
  }

  exprt callable = expr(func);
  std::vector<exprt> values = arguments(node);
  return call(
    "pyrt_call",
    {callable, arguments_struct(values, node), argc(values)},
    loc);
}

exprt python_runtime_converter::list(const json &node)
{
  const locationt loc = location(node);
  exprt result = call("pyrt_list_new", {}, loc);
  for (const json &element : node["elts"])
    call("pyrt_list_append", {result, expr(element)}, loc);
  return result;
}

exprt python_runtime_converter::dict_literal(const json &node)
{
  const locationt loc = location(node);
  exprt result = call("pyrt_dict_new", {}, loc);
  const json &keys = node["keys"];
  for (size_t i = 0; i < keys.size(); ++i)
  {
    if (keys[i].is_null())
      unsupported(node);
    exprt key = expr(keys[i]);
    exprt value = expr(node["values"][i]);
    call("pyrt_setitem", {result, key, value}, loc);
  }
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

/// The type object an annotation names, or nil when the runtime cannot
/// express it. Any, TypeVar, Callable, a generic, a forward reference and
/// anything unresolved all yield nil and so produce no claim at all: refusing
/// them would reject most annotated programs.
exprt python_runtime_converter::annotation_type(const json &annotation) const
{
  if (is_type(annotation, "Constant") && annotation["value"].is_null())
    return address("c:@PyRtNone_Type");
  if (!is_type(annotation, "Name"))
    return nil_exprt();
  const std::string id = annotation["id"];
  const std::string builtin = builtin_type_symbol(id);
  if (!builtin.empty())
    return address(builtin);
  if (classes_.count(id))
    return address(type_object_id(id));
  return nil_exprt();
}

/// CPython does not enforce an annotation, so a wrong one is a defect the
/// static path silently trusts. Under --python-check-annotations the object
/// carries its type, so the annotation becomes a claim instead.
void python_runtime_converter::check_annotation(
  const json &annotation,
  const exprt &value,
  const std::string &what,
  const locationt &loc)
{
  if (!check_annotations_ || annotation.is_null())
    return;
  exprt cls = annotation_type(annotation);
  if (cls.is_nil())
    return;
  code_assertt assertion;
  assertion.assertion() = call("pyrt_isinstance", {value, cls}, loc);
  assertion.location() = loc;
  assertion.location().comment(
    "TypeError: " + what + " does not match its annotation");
  block_->copy_to_operands(assertion);
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
  const bool module_level = code_id_.empty();
  if (type == "Expr")
  {
    const json &value = node["value"];
    if (!(is_type(value, "Constant") && value["value"].is_string()))
      expr(value);
  }
  else if (type == "Assign")
  {
    exprt value = expr(node["value"]);
    for (const json &target : node["targets"])
      store(target, value, loc);
  }
  else if (type == "AnnAssign")
  {
    /* A bare `x: int` binds nothing, as in CPython. */
    if (!node["value"].is_null())
    {
      const json &target = node["target"];
      exprt value = expr(node["value"]);
      store(target, value, loc);
      check_annotation(
        node["annotation"],
        value,
        is_type(target, "Name")
          ? "'" + target["id"].get<std::string>() + "'"
          : std::string("value"),
        loc);
    }
  }
  else if (type == "AugAssign")
    aug_assign(node);
  else if (type == "If")
    if_statement(node);
  else if (type == "While")
    while_statement(node);
  else if (type == "For")
    for_statement(node);
  else if (type == "Assert")
    assert_statement(node);
  else if (type == "Return")
  {
    if (module_level)
      unsupported(node);
    exprt value = node["value"].is_null() ? address("c:@pyrt_None")
                                          : expr(node["value"]);
    if (return_annotation_)
      check_annotation(*return_annotation_, value, "the return value", loc);
    code_returnt ret;
    ret.return_value() = value;
    ret.location() = loc;
    block_->copy_to_operands(ret);
  }
  else if (type == "Break")
    block_->copy_to_operands(code_breakt());
  else if (type == "Continue")
    block_->copy_to_operands(code_continuet());
  else if (type == "ClassDef" && module_level)
    class_statement(node);
  else if (type == "FunctionDef" && module_level)
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
    const std::string symbol = locals_.count(id) ? local_id(id) : global_id(id);
    code_assignt assign(symbol_expr(lookup(symbol)), value);
    assign.location() = loc;
    block_->copy_to_operands(assign);
  }
  else if (is_type(target, "Attribute"))
  {
    exprt object = expr(target["value"]);
    call("pyrt_setattr", {object, name_pointer(target["attr"]), value}, loc);
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

void python_runtime_converter::aug_assign(const json &node)
{
  const json &target = node["target"];
  const std::string op = node["op"]["_type"];
  const locationt loc = location(node);
  if (is_type(target, "Name"))
  {
    exprt current = name(target);
    exprt result = binop(op, current, expr(node["value"]), node);
    store(target, result, loc);
  }
  else if (is_type(target, "Attribute"))
  {
    exprt object = expr(target["value"]);
    exprt attr = name_pointer(target["attr"]);
    exprt current = call("pyrt_getattr", {object, attr}, loc);
    exprt result = binop(op, current, expr(node["value"]), node);
    call("pyrt_setattr", {object, attr, result}, loc);
  }
  else
    unsupported(node);
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

/// `for i in range(...)` counts on an unboxed integer; anything else walks
/// indices over the length taken once, which matches CPython for the
/// containers this runtime has. The loop is emitted as init/condition/step so
/// `continue` still advances it.
void python_runtime_converter::for_statement(const json &node)
{
  if (!node["orelse"].empty())
    unsupported(node);
  const json &target = node["target"];
  if (!is_type(target, "Name"))
    unsupported(node);

  const locationt loc = location(node);
  const json &iterable = node["iter"];
  const bool over_range = is_type(iterable, "Call") &&
                          is_type(iterable["func"], "Name") &&
                          iterable["func"]["id"] == "range" &&
                          !locals_.count("range") && !globals_.count("range") &&
                          !functions_.count("range");

  const typet counter = long_long_int_type();
  exprt index = new_temporary(counter, loc);
  exprt limit = new_temporary(counter, loc);
  exprt container;
  exprt start = from_integer(0, counter);
  int64_t stride = 1;

  if (over_range)
  {
    const json &args = iterable["args"];
    if (args.empty() || args.size() > 3 || !iterable["keywords"].empty())
      unsupported(iterable);
    exprt stop;
    if (args.size() == 1)
      stop = call("pyrt_as_index", {expr(args[0])}, loc);
    else
    {
      start = call("pyrt_as_index", {expr(args[0])}, loc);
      stop = call("pyrt_as_index", {expr(args[1])}, loc);
    }
    if (args.size() == 3)
    {
      // The direction decides the loop condition, so the step has to be known
      // here rather than computed.
      auto literal = literal_int(args[2]);
      if (!literal || *literal == 0)
        unsupported(iterable);
      stride = *literal;
    }
    block_->copy_to_operands(code_assignt(limit, stop));
  }
  else
  {
    container = new_temporary(object_type_, loc);
    block_->copy_to_operands(code_assignt(container, expr(iterable)));
    block_->copy_to_operands(
      code_assignt(limit, call("pyrt_iter_length", {container}, loc)));
  }

  exprt condition(stride > 0 ? "<" : ">", bool_typet());
  condition.copy_to_operands(index, limit);

  code_blockt body;
  code_blockt *outer = block_;
  block_ = &body;
  store(
    target,
    over_range ? call("pyrt_long_from", {index}, loc)
               : call("pyrt_iter_item", {container, index}, loc),
    loc);
  block_ = outer;
  statements(node["body"], body);

  exprt next = plus_exprt(index, from_integer(stride, counter));
  next.type() = counter;

  codet loop;
  loop.set_statement("for");
  loop.copy_to_operands(code_assignt(index, start));
  loop.copy_to_operands(condition);
  loop.copy_to_operands(code_assignt(index, next));
  loop.copy_to_operands(body);
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

/// The class statement readies the static type object, then sets each body
/// member through pyrt_setattr, so defining a special method takes the same
/// copy-on-write path as assigning one later.
void python_runtime_converter::class_statement(const json &node)
{
  const std::string cls = node["name"];
  const locationt loc = location(node);
  exprt type_object = address(type_object_id(cls));
  call("pyrt_type_ready", {type_pointer(cls)}, loc);

  for (const json &member : node["body"])
  {
    const locationt member_loc = location(member);
    if (is_type(member, "FunctionDef"))
    {
      const std::string method = member["name"];
      call(
        "pyrt_setattr",
        {type_object,
         name_pointer(method),
         address(function_object_id(cls, method))},
        member_loc);
    }
    else if (
      is_type(member, "Assign") && member["targets"].size() == 1 &&
      is_type(member["targets"][0], "Name"))
    {
      exprt value = expr(member["value"]);
      call(
        "pyrt_setattr",
        {type_object, name_pointer(member["targets"][0]["id"]), value},
        member_loc);
    }
    else if (
      !is_type(member, "Pass") &&
      !(is_type(member, "Expr") && is_type(member["value"], "Constant")))
      unsupported(member);
  }

  code_assignt bind(symbol_expr(lookup(global_id(cls))), type_object);
  bind.location() = loc;
  block_->copy_to_operands(bind);
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
    else if (
      (is_type(node, "AugAssign") || is_type(node, "AnnAssign")) &&
      is_type(node["target"], "Name"))
      assigned.insert(node["target"]["id"].get<std::string>());
    else if (is_type(node, "ClassDef"))
      assigned.insert(node["name"].get<std::string>());
    else if (is_type(node, "Global"))
      for (const json &global : node["names"])
        declared_global.insert(global.get<std::string>());
    else if (is_type(node, "If") || is_type(node, "While"))
    {
      collect_assigned(node["body"], assigned, declared_global);
      collect_assigned(node["orelse"], assigned, declared_global);
    }
    else if (is_type(node, "For"))
    {
      if (is_type(node["target"], "Name"))
        assigned.insert(node["target"]["id"].get<std::string>());
      collect_assigned(node["body"], assigned, declared_global);
      collect_assigned(node["orelse"], assigned, declared_global);
    }
  }
}

void python_runtime_converter::declare_variable(
  const std::string &id,
  const std::string &name,
  const typet &type,
  bool is_global,
  const locationt &loc)
{
  symbolt symbol;
  symbol.id = id;
  symbol.name = name;
  symbol.set_type(type);
  symbol.location = loc;
  symbol.lvalue = true;
  symbol.static_lifetime = is_global;
  symbol.file_local = !is_global;
  if (is_global)
    symbol.set_value(gen_zero(type));
  add_symbol(symbol);
}

void python_runtime_converter::check_signature(const json &def) const
{
  const json &args = def["args"];
  if (
    !def["decorator_list"].empty() || !args["posonlyargs"].empty() ||
    !args["kwonlyargs"].empty() || !args["defaults"].empty() ||
    !args["vararg"].is_null() || !args["kwarg"].is_null() ||
    args["args"].size() > max_args)
    unsupported(def);
}

void python_runtime_converter::declare_function(
  const json &def,
  const std::string &cls)
{
  check_signature(def);
  const std::string name = def["name"];
  const std::string id = code_id(cls, name);
  const locationt loc = location(def);
  const typet args_type = symbol_typet(args_tag);

  code_typet type;
  type.return_type() = object_type_;
  code_typet::argumentt parameter(args_type);
  parameter.set_identifier(id + "@$args");
  parameter.set_base_name("$args");
  type.arguments().push_back(parameter);
  declare_variable(id + "@$args", "$args", args_type, false, loc);

  symbolt symbol;
  symbol.id = id;
  symbol.name = name;
  symbol.set_type(type);
  symbol.location = loc;
  symbol.lvalue = true;
  const symbolt &function = add_symbol(symbol);

  const std::string object_id = function_object_id(cls, name);
  add_static_object(object_id, function_tag, loc);
  context_.find_symbol(object_id)->set_value(struct_value(
    function_tag,
    {{"ob_type", address_of_exprt(symbol_expr(lookup("c:@PyRtFunction_Type")))},
     {"arity", from_integer(def["args"]["args"].size(), long_long_int_type())},
     {"code", address_of_exprt(symbol_expr(function))}}));
}

void python_runtime_converter::define_function(
  const json &def,
  const std::string &cls)
{
  const std::string name = def["name"];
  code_id_ = code_id(cls, name);
  function_name_ = name;

  std::set<std::string> assigned, declared_global;
  collect_assigned(def["body"], assigned, declared_global);
  std::vector<std::string> parameters;
  for (const json &arg : def["args"]["args"])
    parameters.push_back(arg["arg"].get<std::string>());
  locals_ = std::set<std::string>(parameters.begin(), parameters.end());
  for (const std::string &local : assigned)
    if (!declared_global.count(local))
      locals_.insert(local);

  code_blockt body;
  block_ = &body;
  const locationt loc = location(def);
  const exprt args = symbol_expr(lookup(code_id_ + "@$args"));
  for (const std::string &local : locals_)
  {
    declare_variable(local_id(local), local, object_type_, false, loc);
    exprt variable = symbol_expr(lookup(local_id(local)));
    body.copy_to_operands(code_declt(variable));

    exprt initial = gen_zero(object_type_);
    auto position = std::find(parameters.begin(), parameters.end(), local);
    if (position != parameters.end())
      initial = member_exprt(
        args,
        "a" + std::to_string(position - parameters.begin()),
        object_type_);
    body.copy_to_operands(code_assignt(variable, initial));
  }

  const json &returns = def["returns"];
  return_annotation_ = returns.is_null() ? nullptr : &returns;
  for (size_t i = 0; i < parameters.size(); ++i)
  {
    const json &argument = def["args"]["args"][i];
    if (!argument["annotation"].is_null())
      check_annotation(
        argument["annotation"],
        symbol_expr(lookup(local_id(parameters[i]))),
        "parameter '" + parameters[i] + "'",
        loc);
  }

  statements(def["body"], body);
  /* Falling off the end returns None, which a return annotation usually
   * forbids. That is the defect this catches most often. */
  exprt implicit = address("c:@pyrt_None");
  if (return_annotation_)
    check_annotation(*return_annotation_, implicit, "the return value", loc);
  code_returnt fall_off;
  fall_off.return_value() = implicit;
  body.copy_to_operands(fall_off);
  return_annotation_ = nullptr;

  context_.find_symbol(code_id_)->set_value(body);
  code_id_.clear();
  function_name_.clear();
  locals_.clear();
  block_ = nullptr;
}

/// A class is a static type object with its own attribute table. Direct
/// subclasses are threaded through tp_subclass/tp_sibling so a slot write can
/// reach them.
void python_runtime_converter::declare_class(const json &def)
{
  const std::string cls = def["name"];
  if (
    !def["decorator_list"].empty() || !def["keywords"].empty() ||
    def["bases"].size() > 1)
    unsupported(def);
  if (!def["bases"].empty())
  {
    const json &base = def["bases"][0];
    if (
      !is_type(base, "Name") ||
      (base["id"] != "object" && !classes_.count(base["id"])))
      unsupported(def);
  }

  const locationt loc = location(def);
  const std::string attrs_id = global_id("$pyrt_attrs$" + cls);
  add_static_object(attrs_id, attrs_tag, loc);
  context_.find_symbol(attrs_id)->set_value(struct_value(attrs_tag, {}));
  add_static_object(type_object_id(cls), type_tag, loc);
  classes_[cls] = &def;

  for (const json &member : def["body"])
    if (is_type(member, "FunctionDef"))
      declare_function(member, cls);
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
    throw std::runtime_error("--python-runtime: the runtime models are missing");
  add_c_intrinsics();

  const json &body = ast_["body"];
  std::set<std::string> declared_global;
  collect_assigned(body, globals_, declared_global);
  for (const json &node : body)
  {
    if (!is_type(node, "FunctionDef") && !is_type(node, "ClassDef"))
      continue;
    std::set<std::string> assigned;
    for (const json &member : node["body"])
      if (is_type(member, "FunctionDef"))
        collect_assigned(member["body"], assigned, declared_global);
    collect_assigned(node["body"], assigned, declared_global);
  }
  for (const std::string &name : declared_global)
    globals_.insert(name);

  const locationt loc = location(ast_);
  for (const std::string &name : globals_)
    declare_variable(global_id(name), name, object_type_, true, loc);

  for (const json &node : body)
    if (is_type(node, "FunctionDef"))
    {
      if (globals_.count(node["name"]))
        unsupported(node);
      functions_[node["name"]] = &node;
      declare_function(node, "");
    }
    else if (is_type(node, "ClassDef"))
      declare_class(node);

  // Type objects are filled in once every class exists, since each points at
  // its base and its first direct subclass and next sibling.
  std::map<std::string, std::vector<std::string>> subclasses;
  for (const json &node : body)
    if (is_type(node, "ClassDef") && !node["bases"].empty())
    {
      const std::string base = node["bases"][0]["id"];
      if (base != "object")
        subclasses[base].push_back(node["name"]);
    }
  for (const auto &[cls, def] : classes_)
  {
    std::string base;
    if (!(*def)["bases"].empty() && (*def)["bases"][0]["id"] != "object")
      base = (*def)["bases"][0]["id"];
    std::map<std::string, exprt> fields = {
      {"ob_type", address_of_exprt(symbol_expr(lookup("c:@PyRtType_Type")))},
      {"tp_name", name_pointer(cls)},
      {"tp_base",
       base.empty() ? address_of_exprt(symbol_expr(lookup("c:@PyRtObject_Type")))
                    : type_pointer(base)},
      {"tp_attrs",
       address_of_exprt(symbol_expr(lookup(global_id("$pyrt_attrs$" + cls))))},
      {"tp_flags", from_integer(heaptype_flag, uint_type())}};
    const auto &children = subclasses[cls];
    if (!children.empty())
      fields["tp_subclass"] = type_pointer(children.front());
    const auto &siblings = subclasses[base];
    auto self = std::find(siblings.begin(), siblings.end(), cls);
    if (!base.empty() && self != siblings.end() && self + 1 != siblings.end())
      fields["tp_sibling"] = type_pointer(*(self + 1));
    context_.find_symbol(type_object_id(cls))
      ->set_value(struct_value(type_tag, fields));
  }

  for (const auto &[name, def] : functions_)
    define_function(*def, "");
  for (const auto &[cls, def] : classes_)
    for (const json &member : (*def)["body"])
      if (is_type(member, "FunctionDef"))
        define_function(member, cls);

  code_blockt user_code;
  statements(body, user_code);
  add_entry_points(user_code);
}
