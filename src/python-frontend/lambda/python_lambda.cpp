#include <python-frontend/lambda/python_lambda.h>
#include <python-frontend/python-list/python_list.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/type/type_handler.h>
#include <util/arith/arith_tools.h>
#include <util/lang/c_types.h>
#include <util/irep/std_code.h>

using namespace python_expr;

// Initialize static counter
int python_lambda::lambda_counter_ = 0;

python_lambda::python_lambda(
  python_converter &converter,
  contextt &context,
  type_handler &type_handler)
  : converter_(converter), context_(context), type_handler_(type_handler)
{
}

std::string python_lambda::generate_unique_lambda_name()
{
  return "lam" + std::to_string(++lambda_counter_);
}

bool python_lambda::is_lambda_assignment(const nlohmann::json &ast_node) const
{
  return ast_node.contains("value") && ast_node["value"].contains("_type") &&
         ast_node["value"]["_type"] == "Lambda";
}

void python_lambda::handle_lambda_assignment(
  symbolt *lhs_symbol,
  exprt &lhs,
  exprt &rhs)
{
  if (!lhs_symbol || !rhs.is_symbol())
    return;

  const symbolt *lambda_func_symbol = context_.find_symbol(rhs.identifier());

  if (!lambda_func_symbol || !lambda_func_symbol->get_type().is_code())
  {
    throw std::runtime_error("Lambda function symbol does not have code type");
  }

  // Create function pointer type
  typet func_ptr_type = gen_pointer_type(lambda_func_symbol->get_type());
  lhs_symbol->set_type(func_ptr_type);
  lhs.type() = func_ptr_type;

  // Convert lambda symbol to address
  rhs = build_address_of(rhs);
}

static bool is_param_used_as_string(
  const nlohmann::json &body_node,
  const std::string &param_name)
{
  if (!body_node.contains("_type"))
    return false;

  std::string body_type = body_node["_type"].get<std::string>();

  // Check if param is in string concatenation: param + "string" or "string" + param
  if (
    body_type == "BinOp" && body_node.contains("op") &&
    body_node["op"].contains("_type") && body_node["op"]["_type"] == "Add")
  {
    auto is_string_literal = [](const nlohmann::json &node) {
      return node.contains("_type") && node["_type"] == "Constant" &&
             node.contains("value") && node["value"].is_string();
    };

    auto is_param = [&](const nlohmann::json &node) {
      return node.contains("_type") && node["_type"] == "Name" &&
             node.contains("id") && node["id"] == param_name;
    };

    if (body_node.contains("left") && body_node.contains("right"))
    {
      if (
        (is_param(body_node["left"]) &&
         is_string_literal(body_node["right"])) ||
        (is_string_literal(body_node["left"]) && is_param(body_node["right"])))
        return true;
    }
  }

  // Check IfExp branches recursively
  if (body_type == "IfExp")
  {
    if (
      body_node.contains("body") &&
      is_param_used_as_string(body_node["body"], param_name))
      return true;
    if (
      body_node.contains("orelse") &&
      is_param_used_as_string(body_node["orelse"], param_name))
      return true;
  }

  return false;
}

// True when `param_name` appears in callee position anywhere in the body.
// Parameters are typed `double` below, so lowering such a call would build a
// call through a non-code operand and hand the solver an ill-sorted term
// (esbmc/esbmc#7074).
static bool is_param_used_as_callee(
  const nlohmann::json &node,
  const std::string &param_name)
{
  if (node.is_array())
  {
    for (const auto &child : node)
      if (is_param_used_as_callee(child, param_name))
        return true;
    return false;
  }

  if (!node.is_object())
    return false;

  if (node.value("_type", "") == "Call" && node.contains("func"))
  {
    const auto &callee = node["func"];
    if (
      callee.is_object() && callee.value("_type", "") == "Name" &&
      callee.value("id", "") == param_name)
      return true;
  }

  for (const auto &entry : node.items())
    if (is_param_used_as_callee(entry.value(), param_name))
      return true;

  return false;
}

static void refuse_called_lambda_parameter(
  const nlohmann::json &body_node,
  const std::string &arg_name)
{
  if (!is_param_used_as_callee(body_node, arg_name))
    return;

  throw std::runtime_error(
    "calling the lambda parameter '" + arg_name +
    "' is not supported: higher-order lambda parameters have no inferred "
    "signature");
}

typet python_lambda::infer_lambda_return_type(
  [[maybe_unused]] const nlohmann::json &body_node)
{
  // Check if body is a string operation
  if (body_node.contains("_type"))
  {
    std::string body_type = body_node["_type"].get<std::string>();

    // String concatenation (BinOp with Add and string constant)
    if (
      body_type == "BinOp" && body_node.contains("op") &&
      body_node["op"].contains("_type") && body_node["op"]["_type"] == "Add")
    {
      // Check if the right operand is a string constant
      if (
        body_node.contains("right") && body_node["right"].contains("_type") &&
        body_node["right"]["_type"] == "Constant" &&
        body_node["right"].contains("value") &&
        body_node["right"]["value"].is_string())
      {
        return gen_pointer_type(signed_char_type());
      }
    }

    // Handle IfExp (ternary expressions)
    if (body_type == "IfExp")
    {
      // Recursively check if any branch contains a string literal
      std::function<bool(const nlohmann::json &)> has_string_literal =
        [&](const nlohmann::json &node) -> bool {
        if (!node.contains("_type"))
          return false;

        std::string node_type = node["_type"].get<std::string>();

        // Direct string constant
        if (
          node_type == "Constant" && node.contains("value") &&
          node["value"].is_string())
          return true;

        // Nested IfExp - check recursively
        if (node_type == "IfExp")
        {
          return (node.contains("body") && has_string_literal(node["body"])) ||
                 (node.contains("orelse") &&
                  has_string_literal(node["orelse"]));
        }

        return false;
      };

      // If any branch has a string literal, return string pointer type
      if (
        (body_node.contains("body") && has_string_literal(body_node["body"])) ||
        (body_node.contains("orelse") &&
         has_string_literal(body_node["orelse"])))
      {
        return gen_pointer_type(signed_char_type());
      }
    }
  }

  // Default to double for numeric expressions
  return double_type();
}

symbolt python_lambda::create_symbol(
  const std::string &id,
  const std::string &name,
  const typet &type,
  const locationt &location,
  const std::string &module_name,
  bool file_local,
  bool is_parameter)
{
  symbolt symbol;
  symbol.id = id;
  symbol.name = name;
  symbol.set_type(type);
  symbol.location = location;
  symbol.mode = "Python";
  symbol.module = module_name;
  symbol.lvalue = true;
  symbol.is_parameter = is_parameter;
  symbol.file_local = file_local;
  symbol.static_lifetime = false;
  symbol.is_extern = false;

  return symbol;
}

namespace
{
bool same_position(const nlohmann::json &a, const nlohmann::json &b)
{
  return a.value("lineno", -1) == b.value("lineno", -2) &&
         a.value("col_offset", -1) == b.value("col_offset", -2);
}

// The statement list holding `lambda_node`'s binding, plus the name it binds.
// A lambda is lowered eagerly at its assignment, so this is the only place a
// later call through that name is still visible (#7328).
const nlohmann::json *find_binding_scope(
  const nlohmann::json &node,
  const nlohmann::json &lambda_node,
  std::string &bound_name)
{
  if (node.is_array())
  {
    for (const auto &stmt : node)
    {
      if (
        stmt.value("_type", "") == "Assign" && stmt.contains("value") &&
        stmt["value"].value("_type", "") == "Lambda" &&
        same_position(stmt["value"], lambda_node) && stmt.contains("targets") &&
        stmt["targets"].is_array() && stmt["targets"].size() == 1 &&
        stmt["targets"][0].value("_type", "") == "Name")
      {
        bound_name = stmt["targets"][0].value("id", "");
        return &node;
      }
    }
  }

  if (!node.is_structured())
    return nullptr;

  for (const auto &child : node.items())
  {
    const nlohmann::json *found =
      find_binding_scope(child.value(), lambda_node, bound_name);
    if (found != nullptr)
      return found;
  }
  return nullptr;
}

// Argument `index` of every call to `name` under `node`. All of them, not just
// the first: a lambda called with two different types has one frozen signature,
// so committing to the first call's type would mistype the rest (#7328).
void collect_call_arguments(
  const nlohmann::json &node,
  const std::string &name,
  size_t index,
  std::vector<const nlohmann::json *> &out)
{
  if (node.is_object() && node.value("_type", "") == "Call")
  {
    const nlohmann::json &func = node["func"];
    if (
      func.value("_type", "") == "Name" && func.value("id", "") == name &&
      node.contains("args") && node["args"].is_array() &&
      node["args"].size() > index)
      out.push_back(&node["args"][index]);
  }

  if (!node.is_structured())
    return;

  for (const auto &child : node.items())
    collect_call_arguments(child.value(), name, index, out);
}
} // namespace

std::vector<typet>
python_lambda::call_site_argument_types(const nlohmann::json &element) const
{
  std::vector<typet> types;
  if (
    !element.contains("args") || !element["args"].contains("args") ||
    !element["args"]["args"].is_array())
    return types;

  std::string bound_name;
  const nlohmann::json *scope =
    find_binding_scope(converter_.ast(), element, bound_name);
  if (scope == nullptr || bound_name.empty())
    return types;

  const locationt location = converter_.get_location_from_decl(element);
  const std::string prefix = "py:" + location.get_file().as_string() + "@F@" +
                             converter_.get_current_func_name() + "@";

  const size_t count = element["args"]["args"].size();
  for (size_t i = 0; i < count; ++i)
  {
    typet resolved;
    resolved.make_nil();

    std::vector<const nlohmann::json *> args;
    collect_call_arguments(*scope, bound_name, i, args);

    for (const nlohmann::json *arg : args)
    {
      typet from_arg;
      from_arg.make_nil();

      // A subscript yields an element, whose registered type list_type_map
      // recorded when the list literal was converted. A list-valued element is
      // left alone: the list object pointer is not usable as a parameter type
      // here, and typing it as one makes the solver reject the formula.
      if (
        arg->value("_type", "") == "Subscript" && arg->contains("value") &&
        (*arg)["value"].value("_type", "") == "Name")
      {
        typet elem = python_list::get_list_element_type(
          prefix + (*arg)["value"].value("id", ""), 0);
        if (
          elem != typet() && elem != empty_typet() &&
          elem != type_handler_.get_list_type())
          from_arg = elem;
      }

      // Every call has to agree: one disagreeing call means the single frozen
      // signature cannot serve them all, so leave the parameter as it was.
      if (from_arg.is_nil() || (resolved.is_not_nil() && resolved != from_arg))
      {
        resolved.make_nil();
        break;
      }
      resolved = from_arg;
    }
    types.push_back(resolved);
  }
  return types;
}

// A lambda parameter is typed by its annotation, else by string usage in the
// body, else by the value the call site passes -- the `double` default rejects
// any body that indexes its parameter (#7328).
static typet lambda_parameter_type(
  const nlohmann::json &arg,
  const nlohmann::json &body_node,
  const std::string &arg_name,
  const typet &from_call_site)
{
  if (arg.contains("annotation") && !arg["annotation"].is_null())
    return arg["annotation"].get<std::string>() == "str"
             ? gen_pointer_type(signed_char_type())
             : double_type();

  if (is_param_used_as_string(body_node, arg_name))
    return gen_pointer_type(signed_char_type());

  if (from_call_site.is_not_nil() && from_call_site.id() != irep_idt())
    return from_call_site;

  return double_type();
}

void python_lambda::process_lambda_parameters(
  const nlohmann::json &args_node,
  code_typet &lambda_type,
  [[maybe_unused]] const std::string &lambda_id,
  const std::string &param_scope_id,
  const locationt &location,
  const nlohmann::json &body_node,
  const std::vector<typet> &call_site_types)
{
  if (!args_node.contains("args") || !args_node["args"].is_array())
    return;

  std::string module_name = location.get_file().as_string();

  size_t arg_index = 0;
  for (const auto &arg : args_node["args"])
  {
    const size_t this_index = arg_index++;
    std::string arg_name = arg["arg"].get<std::string>();

    refuse_called_lambda_parameter(body_node, arg_name);

    const typet param_type = lambda_parameter_type(
      arg,
      body_node,
      arg_name,
      this_index < call_site_types.size() ? call_site_types[this_index]
                                          : typet());

    // Each lambda parameter is modelled as two symbols:
    //
    //  1. closure_id  (lam@x): a static symbol that is never passed to
    //     symex_decl, so it never ends up in frame.local_variables and is
    //     therefore NOT cleared when the function frame is popped.  Inner
    //     lambdas look up free variables by name and find this symbol.
    //
    //  2. actual_param_id (lam@x$param): the real parameter symbol that
    //     goto_symex::argument_assignments assigns from the call-site argument.
    //     It lives in the function's local frame (symex_decl adds it to
    //     local_variables) and is cleaned up on return as normal.
    //
    // The lambda body starts with ASSIGN lam@x = lam@x$param (see
    // get_lambda_expr), which copies the transient parameter value into the
    // persistent closure variable via a plain symex_assign (no symex_decl),
    // preserving it for any inner lambda that captures it.
    std::string closure_id = param_scope_id + "@" + arg_name;
    std::string actual_param_id = closure_id + "$param";

    // Create function argument – points to the actual parameter symbol so
    // that goto_symex assigns the call argument to lam@x$param.
    code_typet::argumentt argument;
    argument.type() = param_type;
    argument.cmt_base_name(arg_name);
    argument.cmt_identifier(actual_param_id);
    argument.location() = location;
    lambda_type.arguments().push_back(argument);

    // Static closure variable: persists after the enclosing function returns.
    symbolt closure_symbol = create_symbol(
      closure_id,
      arg_name,
      param_type,
      location,
      module_name,
      true, // file_local
      false // not a parameter – keeps it out of symex frame locals
    );
    closure_symbol.static_lifetime = true;
    context_.add(closure_symbol);

    // Actual parameter symbol: assigned by argument_assignments at call site.
    symbolt param_symbol = create_symbol(
      actual_param_id,
      arg_name + "$param",
      param_type,
      location,
      module_name,
      true, // file_local
      true  // is_parameter
    );

    context_.add(param_symbol);
  }

  // Trailing positional parameters may carry default values
  // (lambda x, y=2: ...). Record them on the argument slots so the call site
  // fills omitted arguments, matching process_function_arguments for defs;
  // without this the omitted parameter is left nondet.
  if (
    args_node.contains("defaults") && args_node["defaults"].is_array() &&
    !args_node["defaults"].empty())
  {
    const auto &defaults = args_node["defaults"];
    const size_t n_args = lambda_type.arguments().size();
    const size_t defaults_count = defaults.size();
    if (defaults_count <= n_args)
    {
      for (size_t i = 0; i < defaults_count; ++i)
      {
        if (defaults[i].is_null())
          continue;
        auto &arg = lambda_type.arguments()[n_args - defaults_count + i];
        exprt default_expr = converter_.get_expr(defaults[i]);
        // String/aggregate defaults need the string_constantt + address-of
        // conversion that finalize_call applies to def parameters, which is not
        // yet wired through the lambda indirect-call path. Record only scalar
        // defaults; a string default is left as the existing nondet rather than
        // a mis-cast pointer.
        if (default_expr.type().is_array() || arg.type().is_pointer())
          continue;
        if (default_expr.type() != arg.type())
          default_expr = typecast_exprt(default_expr, arg.type());
        arg.default_value() = default_expr;
      }
    }
  }
}

exprt python_lambda::process_lambda_body(
  const nlohmann::json &body_node,
  const locationt &location)
{
  // Get the body expression through the converter
  exprt body_expr = converter_.get_expr(body_node);

  // If the body is a nested lambda (inner function), take its address so
  // the outer lambda returns a function pointer, not a bare code symbol.
  if (body_expr.type().is_code() && body_expr.is_symbol())
    body_expr = build_address_of(body_expr);

  // Create return statement
  code_returnt return_stmt;
  return_stmt.return_value() = body_expr;
  return_stmt.location() = location;

  // Wrap in a block
  code_blockt lambda_block;
  lambda_block.copy_to_operands(return_stmt);

  return lambda_block;
}

exprt python_lambda::get_lambda_expr(const nlohmann::json &element)
{
  // Generate unique lambda name
  std::string lambda_name = generate_unique_lambda_name();

  locationt location = converter_.get_location_from_decl(element);
  std::string module_name = location.get_file().as_string();

  // Save the original function context
  std::string old_func = converter_.get_current_func_name();

  // Resolve call-site argument types while the enclosing scope is still
  // current: the names they mention are invisible from the lambda's own scope.
  const std::vector<typet> call_site_types = call_site_argument_types(element);

  // Determine if we're in a lambda (function name starts with "lam")
  bool in_lambda = (old_func.find("lam") == 0);

  // Determine the scope for parameters: use first lambda's scope for all nested lambdas
  std::string param_scope;
  if (in_lambda)
  {
    // Nested lambda: use parent lambda's scope for all parameters
    param_scope = old_func;
  }
  else
  {
    // Top-level lambda: use this lambda's name as the scope
    param_scope = lambda_name;
    converter_.set_current_func_name(lambda_name);
  }

  // Create function type with inferred return type
  code_typet lambda_type;
  typet return_type = double_type();

  if (element.contains("body"))
  {
    return_type = infer_lambda_return_type(element["body"]);
    converter_.set_current_element_type(return_type);
  }

  lambda_type.return_type() = return_type;

  // Lambda function symbol is always top-level: py:module@F@lambda_name
  std::string lambda_id = "py:" + module_name + "@F@" + lambda_name;

  // Parameters are created in param_scope (shared for nested lambdas)
  std::string param_scope_id = "py:" + module_name + "@F@" + param_scope;

  // Process lambda parameters: pass body for type inference
  if (element.contains("args"))
    process_lambda_parameters(
      element["args"],
      lambda_type,
      lambda_id,
      param_scope_id,
      location,
      element.contains("body") ? element["body"] : nlohmann::json(),
      call_site_types);

  // Create lambda function symbol
  symbolt lambda_symbol = create_symbol(
    lambda_id,
    lambda_name,
    lambda_type,
    location,
    module_name,
    false, // file_local
    false  // is_parameter
  );

  symbolt *added_symbol = context_.move_symbol_to_context(lambda_symbol);
  assert(added_symbol);

  // Process lambda body
  if (element.contains("body"))
  {
    exprt lambda_body = process_lambda_body(element["body"], location);

    // If the body returns a function pointer (nested lambda) or an Optional[T]
    // struct (ternary with one None branch), update this lambda's declared
    // return type to match the actual return value type so that callers
    // (e.g. g = f(5); g(10)) receive the correct type.
    // The RETURN statement is lambda_body.operands()[0] at this point (before
    // we prepend the closure assignments below).
    if (!lambda_body.operands().empty())
    {
      const exprt &ret_stmt = lambda_body.operands()[0];
      if (
        ret_stmt.id() == "code" && ret_stmt.get("statement") == "return" &&
        !ret_stmt.operands().empty())
      {
        const typet &actual_ret = ret_stmt.operands()[0].type();
        bool is_optional_struct =
          actual_ret.is_struct() && actual_ret.get("tag").as_string().find(
                                      "Optional_") != std::string::npos;
        if (
          (actual_ret.is_pointer() && actual_ret.subtype().is_code()) ||
          is_optional_struct)
        {
          typet t = added_symbol->get_type();
          to_code_type(t).return_type() = actual_ret;
          added_symbol->set_type(std::move(t));
        }
      }
    }

    // Prepend closure assignments: lam@x = lam@x$param for each parameter.
    // This copies the transient argument value (which lives in the symex
    // local frame and is cleared on return) into the static closure variable
    // (which is never in frame.local_variables and therefore persists).
    // Inner lambdas then read the static variable and see the correct value.
    if (
      element.contains("args") && element["args"].contains("args") &&
      element["args"]["args"].is_array() && !element["args"]["args"].empty())
    {
      code_blockt closure_body;
      for (const auto &arg : element["args"]["args"])
      {
        std::string arg_name = arg["arg"].get<std::string>();
        std::string closure_id = param_scope_id + "@" + arg_name;
        std::string actual_param_id = closure_id + "$param";

        const symbolt *closure_sym = context_.find_symbol(closure_id);
        const symbolt *param_sym = context_.find_symbol(actual_param_id);
        if (closure_sym && param_sym)
        {
          code_assignt assign(
            symbol_expr(*closure_sym), symbol_expr(*param_sym));
          assign.location() = location;
          closure_body.copy_to_operands(assign);
        }
      }
      for (const auto &op : lambda_body.operands())
        closure_body.copy_to_operands(op);
      lambda_body = closure_body;
    }

    added_symbol->set_value(lambda_body);
  }

  // Restore context only if we changed it (top-level lambda only)
  if (!in_lambda)
    converter_.set_current_func_name(old_func);

  return symbol_expr(*added_symbol);
}