#include <python-frontend/lambda/python_lambda.h>
#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <python-frontend/python-list/python_list.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/type/element_type_registry.h>
#include <python-frontend/type/type_handler.h>
#include <python-frontend/json_utils.h>
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

// Whether a lambda's signature takes the type its body actually returns
// rather than the inferred default: a function pointer (nested lambda), an
// Optional[T] struct, or an integral value, which a `double` default rounds
// above 2**53 (#7745).
static bool declares_body_return_type(const typet &actual)
{
  const bool is_optional_struct =
    actual.is_struct() &&
    actual.get("tag").as_string().find("Optional_") != std::string::npos;
  return (actual.is_pointer() && actual.subtype().is_code()) ||
         is_optional_struct || actual.is_signedbv() || actual.is_unsignedbv() ||
         actual.is_bool();
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
  // Left legacy: migrate_type drops #cpp_type, and the python type checker
  // reads it -- a bool default otherwise lowers as double (#4715).
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

/// The one name target a plain `x = v` or `x: T = v` statement binds, or null.
const nlohmann::json *binding_target(const nlohmann::json &stmt)
{
  const std::string kind = stmt.value("_type", "");
  const nlohmann::json *target = nullptr;
  if (
    kind == "Assign" && stmt.contains("targets") &&
    stmt["targets"].is_array() && stmt["targets"].size() == 1)
    target = &stmt["targets"][0];
  else if (kind == "AnnAssign" && stmt.contains("target"))
    target = &stmt["target"];
  return target != nullptr && target->is_object() ? target : nullptr;
}

// A lambda is lowered eagerly at its assignment, so the statement list holding
// that assignment is the only place a later call through the bound name is
// still visible (#7328).
const nlohmann::json *find_binding_scope(
  const nlohmann::json &node,
  const nlohmann::json &lambda_node,
  std::string &bound_name)
{
  if (node.is_array())
  {
    for (const auto &stmt : node)
    {
      // Not every ast2json array holds objects: Global/Nonlocal carry raw
      // strings in `names`, and value() throws on those (#7328).
      if (!stmt.is_object())
        continue;

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

struct call_argument_scan
{
  std::vector<const nlohmann::json *> args;
  bool foreign_scope_call = false;
};

// Python opens a new binding scope at each of these, so `name`'s argument
// inside one may denote a different object than the enclosing prefix builds.
bool opens_binding_scope(const std::string &kind)
{
  static const std::set<std::string> kinds = {
    "FunctionDef",
    "AsyncFunctionDef",
    "Lambda",
    "ListComp",
    "SetComp",
    "DictComp",
    "GeneratorExp"};
  return kinds.count(kind) != 0;
}

// Argument `index` of every call to `name` under `node`. All of them, not just
// the first: a lambda called with two different types has one frozen signature,
// so committing to the first call's type would mistype the rest (#7328).
void collect_call_arguments(
  const nlohmann::json &node,
  const std::string &name,
  size_t index,
  bool nested,
  call_argument_scan &out)
{
  const std::string kind = node.is_object() ? node.value("_type", "") : "";

  if (kind == "Call")
  {
    const auto func = node.find("func");
    if (
      func != node.end() && func->is_object() &&
      func->value("_type", "") == "Name" && func->value("id", "") == name &&
      node.contains("args") && node["args"].is_array() &&
      node["args"].size() > index)
    {
      if (nested)
        out.foreign_scope_call = true;
      else
        out.args.push_back(&node["args"][index]);
    }
  }

  if (!node.is_structured())
    return;

  const bool child_nested = nested || opens_binding_scope(kind);

  for (const auto &child : node.items())
    collect_call_arguments(child.value(), name, index, child_nested, out);
}

// Only a list literal materialises each element into a symbol whose type is
// what `name[k]` yields. Every other builder -- concatenation above all --
// records the source expression's type while the list object stores something
// else, so typing a parameter from that entry aborts the frontend on the
// argument check ("got struct, expected struct") (#7328).
bool binds_list_literal(const nlohmann::json &scope, const std::string &name)
{
  bool found = false;
  for (const auto &stmt : scope)
  {
    if (!stmt.is_object())
      continue;

    // The frontend annotates a plain `x = [...]` into an AnnAssign, so both
    // spellings have to be recognised here.
    const nlohmann::json *target = binding_target(stmt);

    if (target == nullptr || target->value("id", "") != name)
      continue;

    // A bare annotation (`cars: list`) is an AnnAssign whose value is JSON
    // null; reading _type off it throws.
    if (
      !stmt.contains("value") || !stmt["value"].is_object() ||
      stmt["value"].value("_type", "") != "List")
      return false;
    found = true;
  }
  return found;
}

/// True when @p node stores to (or deletes) @p name anywhere inside it.
bool stores_name(const nlohmann::json &node, const std::string &name)
{
  if (node.is_array())
  {
    for (const auto &child : node)
      if (stores_name(child, name))
        return true;
    return false;
  }

  if (!node.is_object())
    return false;

  if (
    node.value("_type", "") == "Name" && node.value("id", "") == name &&
    node.contains("ctx") && node["ctx"].is_object())
  {
    const std::string ctx = node["ctx"].value("_type", "");
    if (ctx == "Store" || ctx == "Del")
      return true;
  }

  for (const auto &child : node.items())
    if (stores_name(child.value(), name))
      return true;

  return false;
}

/// True when @p name is also bound somewhere this scan cannot read: a loop
/// target, a with-item, a walrus, a del. Answering from the assignments alone
/// would ignore those bindings.
bool bound_opaquely(const nlohmann::json &node, const std::string &name)
{
  static const std::set<std::string> opaque_kinds = {
    "For", "AsyncFor", "With", "AsyncWith", "NamedExpr", "Delete"};

  if (node.is_array())
  {
    for (const auto &child : node)
      if (bound_opaquely(child, name))
        return true;
    return false;
  }

  if (!node.is_object())
    return false;

  if (opaque_kinds.count(node.value("_type", "")) && stores_name(node, name))
    return true;

  for (const auto &child : node.items())
    if (bound_opaquely(child.value(), name))
      return true;

  return false;
}

/// Every value assigned to @p name in @p scope. Callers require the bindings to
/// agree rather than demanding a single one: rebinding a name to the same thing
/// is ordinary Python and says as much about its type as binding it once.
std::vector<const nlohmann::json *>
assigned_values(const nlohmann::json &scope, const std::string &name)
{
  std::vector<const nlohmann::json *> values;
  for (const auto &stmt : scope)
  {
    if (!stmt.is_object())
      continue;

    const nlohmann::json *target = binding_target(stmt);

    // A bare annotation (`a: Car`) is an AnnAssign carrying a JSON null, which
    // binds nothing -- and which throws if read as an object.
    if (
      target == nullptr || target->value("id", "") != name ||
      !stmt.contains("value") || !stmt["value"].is_object())
      continue;

    values.push_back(&stmt["value"]);
  }
  return values;
}

/// The class @p elt is an instance of, as the AST states it. @p hop allows one
/// step through a name binding, for the `a = Car(); xs = [a]` spelling; the
/// recursive call disallows it, so this cannot chain or cycle.
std::optional<typet> element_class_type(
  const nlohmann::json &elt,
  const nlohmann::json &scope,
  const nlohmann::json &ast,
  bool hop)
{
  const std::string kind = elt.value("_type", "");

  if (kind == "Name")
  {
    const std::string bound_name = elt.value("id", "");
    if (!hop || bound_name.empty() || bound_opaquely(scope, bound_name))
      return std::nullopt;

    std::optional<typet> agreed;
    for (const nlohmann::json *bound : assigned_values(scope, bound_name))
    {
      const std::optional<typet> one =
        element_class_type(*bound, scope, ast, false);
      if (!one || (agreed && *agreed != *one))
        return std::nullopt;
      agreed = one;
    }
    return agreed;
  }

  if (kind != "Call" || !elt.contains("func") || !elt["func"].is_object())
    return std::nullopt;

  const std::string class_name = elt["func"].value("id", "");
  if (class_name.empty() || !json_utils::is_class(class_name, ast))
    return std::nullopt;

  return gen_pointer_type(symbol_typet("tag-" + class_name));
}

/// The class a list literal's @p index element is an instance of, read from the
/// AST. The registry is filled in conversion order, so a lambda bound before
/// the list it is called with finds no element id there; the literal does not
/// move (#7745). An id is also absent when the recorded type came from an
/// annotation rather than a concrete element, so this runs then too -- in both
/// cases the old code answered nothing at all. Only class instances are
/// answered; every other element kind keeps the registry's verdict.
std::optional<typet> literal_element_class_type(
  const nlohmann::json &scope,
  const std::string &name,
  size_t index,
  const nlohmann::json &ast)
{
  if (bound_opaquely(scope, name))
    return std::nullopt;

  std::optional<typet> agreed;
  for (const nlohmann::json *bound : assigned_values(scope, name))
  {
    if (
      bound->value("_type", "") != "List" || !bound->contains("elts") ||
      !(*bound)["elts"].is_array() || index >= (*bound)["elts"].size())
      return std::nullopt;

    const std::optional<typet> one =
      element_class_type((*bound)["elts"][index], scope, ast, true);
    if (!one || (agreed && *agreed != *one))
      return std::nullopt;
    agreed = one;
  }
  return agreed;
}

/// Whether @p node itself binds @p name through a construct other than a
/// Name: a parameter, a def/class, an import, an except or match capture, a
/// global/nonlocal declaration. `from m import *` may bind any name.
bool binds_by_field(const nlohmann::json &node, const std::string &name)
{
  static const std::map<std::string, std::string> name_field = {
    {"arg", "arg"},
    {"FunctionDef", "name"},
    {"AsyncFunctionDef", "name"},
    {"ClassDef", "name"},
    {"ExceptHandler", "name"},
    {"MatchAs", "name"},
    {"MatchStar", "name"},
    {"MatchMapping", "rest"},
    {"Global", "names"},
    {"Nonlocal", "names"}};

  const std::string kind = node.value("_type", "");
  if (kind == "alias")
  {
    const nlohmann::json &as =
      node.contains("asname") && node["asname"].is_string() ? node["asname"]
                                                            : node["name"];
    const std::string bound = as.get<std::string>();
    return bound == "*" || bound.substr(0, bound.find('.')) == name;
  }
  const auto it = name_field.find(kind);
  if (it == name_field.end() || !node.contains(it->second))
    return false;
  const nlohmann::json &field = node[it->second];
  return field.is_array()
           ? std::find(field.begin(), field.end(), name) != field.end()
           : field == name;
}

/// Every construct under @p node that binds @p name, in any nested scope too.
size_t count_bindings(const nlohmann::json &node, const std::string &name)
{
  size_t n = 0;
  if (node.is_object())
    n += node.value("_type", "") == "Name"
           ? node.value("id", "") == name &&
               node["ctx"].value("_type", "") != "Load"
           : binds_by_field(node, name);
  for (const auto &child : node)
    if (child.is_structured())
      n += count_bindings(child, name);
  return n;
}

/// The function, lambda, class or module whose names the code at @p lambda
/// resolves: Python scoping follows these, not the statement list (an `if`
/// body, say) the lambda happens to be bound in.
const nlohmann::json *enclosing_scope(
  const nlohmann::json &node,
  const nlohmann::json &lambda,
  const nlohmann::json *root)
{
  static const std::set<std::string> scope_kinds = {
    "FunctionDef", "AsyncFunctionDef", "Lambda", "ClassDef"};

  if (node.is_object())
  {
    const std::string kind = node.value("_type", "");
    if (kind == "Lambda" && same_position(node, lambda))
      return root;
    if (scope_kinds.count(kind))
      root = &node;
  }
  for (const auto &child : node)
  {
    if (!child.is_structured())
      continue;
    if (const nlohmann::json *found = enclosing_scope(child, lambda, root))
      return found;
  }
  return nullptr;
}

std::optional<typet>
literal_type(const nlohmann::json &value, type_handler &types)
{
  if (!value.is_object() || value.value("_type", "") != "Constant")
    return std::nullopt;
  const nlohmann::json &literal = value["value"];
  if (literal.is_boolean())
    return types.get_typet(std::string("bool"));
  if (literal.is_number_integer())
    return types.get_typet(std::string("int"));
  if (literal.is_number_float())
    return types.get_typet(std::string("float"));
  return std::nullopt;
}

/// The scalar type an assignment gives its target. A literal keeps its own
/// type, as ESBMC's variables do, so an annotation contradicting it (`x: int =
/// 2.5`) answers nothing rather than truncating.
std::optional<typet>
assigned_scalar_type(const nlohmann::json &stmt, type_handler &types)
{
  const std::optional<typet> literal = literal_type(stmt["value"], types);
  if (stmt.value("_type", "") != "AnnAssign")
    return literal;

  const std::string annotation = stmt["annotation"].value("id", "");
  if (annotation != "int" && annotation != "bool" && annotation != "float")
    return std::nullopt;
  const typet annotated = types.get_typet(annotation);
  if (literal && *literal != annotated)
    return std::nullopt;
  return annotated;
}

/// The scalar type of a call argument: a literal, or a name every binding of
/// which, in the scope the lambda resolves names in, is a top-level
/// assignment agreeing on one scalar type. Any other binding -- a parameter,
/// a `global` write, an augmented or branch-local assignment -- makes the
/// binding count exceed the top-level assignments, and answers nothing
/// (#7745).
std::optional<typet> scalar_argument_type(
  const nlohmann::json &arg,
  const nlohmann::json &scope,
  type_handler &types)
{
  if (arg.value("_type", "") != "Name")
    return literal_type(arg, types);
  if (!scope.contains("body") || !scope["body"].is_array())
    return std::nullopt;

  const std::string name = arg.value("id", "");
  std::optional<typet> agreed;
  size_t bindings = 0;
  for (const auto &stmt : scope["body"])
  {
    const nlohmann::json *target = binding_target(stmt);
    if (target == nullptr || target->value("id", "") != name)
      continue;

    const std::optional<typet> one = assigned_scalar_type(stmt, types);
    if (!one || (agreed && *agreed != *one))
      return std::nullopt;
    agreed = one;
    ++bindings;
  }
  if (bindings != count_bindings(scope, name))
    return std::nullopt;
  return agreed;
}

// The element type a `name[k]` argument selects, or nothing when it cannot be
// pinned down. A non-literal index names no single element type, and a
// list-valued element is left alone: the list object pointer is not usable as
// a parameter type here, and typing it as one makes the solver reject the
// formula (#7328).
std::optional<typet> subscript_element_type(
  const nlohmann::json &arg,
  const nlohmann::json &scope,
  const std::string &prefix,
  const typet &list_type,
  const element_type_registry &elem_types,
  const nlohmann::json &ast)
{
  if (
    arg.value("_type", "") != "Subscript" || !arg.contains("value") ||
    !arg["value"].is_object() || arg["value"].value("_type", "") != "Name" ||
    !arg.contains("slice"))
    return std::nullopt;

  const nlohmann::json &slice = arg["slice"];
  if (
    !slice.is_object() || slice.value("_type", "") != "Constant" ||
    !slice.contains("value") || !slice["value"].is_number_unsigned())
    return std::nullopt;

  const std::string base_name = arg["value"].value("id", "");
  if (!binds_list_literal(scope, base_name))
    return std::nullopt;

  const std::string list_id = prefix + base_name;
  const size_t index = slice["value"].get<size_t>();

  // element_type() clamps an out-of-range index to element 0, so the recorded
  // element id is what says the index names a real element.
  if (elem_types.element_id(list_id, index).empty())
    return literal_element_class_type(scope, base_name, index, ast);

  const typet elem = elem_types.element_type(list_id, index);
  if (elem == typet() || elem == empty_typet() || elem == list_type)
    return std::nullopt;
  return elem;
}
} // namespace

std::vector<std::optional<typet>>
python_lambda::call_site_argument_types(const nlohmann::json &element) const
{
  std::vector<std::optional<typet>> types;
  if (
    !element.contains("args") || !element["args"].contains("args") ||
    !element["args"]["args"].is_array())
    return types;

  std::string bound_name;
  const nlohmann::json *scope =
    find_binding_scope(converter_.ast(), element, bound_name);
  if (scope == nullptr || bound_name.empty())
    return types;

  const nlohmann::json *name_scope =
    enclosing_scope(converter_.ast(), element, &converter_.ast());

  const locationt location = converter_.get_location_from_decl(element);
  const std::string prefix = "py:" + location.get_file().as_string() + "@F@" +
                             converter_.get_current_func_name() + "@";

  const size_t count = element["args"]["args"].size();
  for (size_t i = 0; i < count; ++i)
  {
    std::optional<typet> resolved;

    call_argument_scan scan;
    collect_call_arguments(*scope, bound_name, i, false, scan);
    if (scan.foreign_scope_call)
    {
      types.push_back(std::nullopt);
      continue;
    }

    for (const nlohmann::json *arg : scan.args)
    {
      std::optional<typet> from_arg = subscript_element_type(
        *arg,
        *scope,
        prefix,
        type_handler_.get_list_type(),
        converter_.get_element_type_registry(),
        converter_.ast());
      if (!from_arg && name_scope != nullptr)
        from_arg = scalar_argument_type(*arg, *name_scope, type_handler_);

      // Every call has to agree: one disagreeing call means the single frozen
      // signature cannot serve them all, so leave the parameter as it was.
      if (!from_arg || (resolved && *resolved != *from_arg))
      {
        resolved.reset();
        break;
      }
      resolved = from_arg;
    }
    types.push_back(resolved);
  }
  return types;
}

// The `double` default rejects any body that indexes its parameter (#7328), so
// the call site is consulted when neither an annotation nor string usage says
// otherwise.
static typet lambda_parameter_type(
  const nlohmann::json &arg,
  const nlohmann::json &body_node,
  const std::string &arg_name,
  const std::optional<typet> &from_call_site)
{
  if (arg.contains("annotation") && !arg["annotation"].is_null())
    return arg["annotation"].get<std::string>() == "str"
             ? gen_pointer_type(signed_char_type())
             : double_type();

  if (is_param_used_as_string(body_node, arg_name))
    return gen_pointer_type(signed_char_type());

  if (from_call_site)
    return *from_call_site;

  return double_type();
}

void python_lambda::process_lambda_parameters(
  const nlohmann::json &args_node,
  code_typet &lambda_type,
  [[maybe_unused]] const std::string &lambda_id,
  const std::string &param_scope_id,
  const locationt &location,
  const nlohmann::json &body_node,
  const std::vector<std::optional<typet>> &call_site_types)
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
                                          : std::nullopt);

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
  const std::vector<std::optional<typet>> call_site_types =
    call_site_argument_types(element);

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

    // Callers (e.g. g = f(5); g(10)) read the declared return type, so it
    // follows the body where declares_body_return_type says so.
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
        if (declares_body_return_type(actual_ret))
        {
          typet t = added_symbol->get_type();
          to_code_type(t).return_type() = actual_ret;
          // Legacy type: migrate_type drops the parameters' default values.
          added_symbol->set_type(t);
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

    added_symbol->set_value(migrate_expr(lambda_body));
  }

  // Restore context only if we changed it (top-level lambda only)
  if (!in_lambda)
    converter_.set_current_func_name(old_func);

  return symbol_expr(*added_symbol);
}