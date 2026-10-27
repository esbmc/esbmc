#include <python-frontend/json_utils.h>
#include <python-frontend/python_converter.h>
#include <filesystem>
#include <python-frontend/symbol_id.h>
#include <util/arith/arith_tools.h>
#include <util/message/message.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <regex>
#include <set>
#include <utility>

void python_converter::update_symbol(const exprt &expr) const
{
  symbolt *sym = nullptr;

  // For symbol_exprt, the full symbol ID (e.g. "py:file@A") is stored in
  // identifier(), not in name(). Try identifier-based lookup first.
  if (!expr.identifier().empty())
  {
    sym = symbol_table_.find_symbol(expr.identifier());
  }

  // Fall back to name-based lookup (name() = base name, e.g. "A").
  // This handles expressions that carry only the base name.
  if (sym == nullptr && !expr.name().empty())
  {
    symbol_id sid = create_symbol_id();
    sid.set_object(expr.name().c_str());

    // Try scoped ID first (py:file@F@func@var), then global (py:file@var)
    sym = symbol_table_.find_symbol(sid.to_string());
    if (sym == nullptr)
      sym = symbol_table_.find_symbol(sid.global_to_string());
  }

  if (sym == nullptr)
  {
    log_debug(
      "python-frontend",
      "[update_symbol]: symbol not found for name='{}' identifier='{}'",
      expr.name().as_string(),
      expr.identifier().as_string());
    return;
  }

  // Update the type of the symbol and its value.
  const typet &expr_type = expr.type();
  sym->set_type(migrate_type(expr_type));
  {
    exprt v = sym->get_value();
    // Stays legacy: this retypes the root only, so migrating eagerly would
    // build an arith node over operands of the old type and trip
    // assert_arith_2ops_consistency (docs/roadmap/scope-python-irep2.md §6.2).
    v.type() = expr_type;
    sym->set_value(std::move(v));
  }

  // Check if the symbol has a constant or bitvector value.
  if (
    sym->get_value().is_constant() || sym->get_value().is_signedbv() ||
    sym->get_value().is_unsignedbv())
  {
    const std::string &binary_value_str = sym->get_value().value().c_str();

    // Only attempt binary conversion if the string is non-empty and consists
    // solely of '0' and '1' characters (i.e., it is a valid binary string).
    // Character or decimal values stored in the symbol will not satisfy this
    // check and must be left unchanged to avoid a stoll conversion failure.
    bool is_binary_string =
      !binary_value_str.empty() &&
      binary_value_str.find_first_not_of("01") == std::string::npos;

    if (is_binary_string)
    {
      try
      {
        // Convert binary string to integer. binary2integer accepts any width,
        // so we don't lose precision on bignum constants emitted by the
        // --ir-gated ** widening for #1964 Part 2 / #4642.
        const bool is_signed = expr_type.is_signedbv();
        const BigInt int_val = binary2integer(binary_value_str, is_signed);

        // Create a new constant expression with the converted value and type.
        exprt new_value = from_integer(int_val, expr_type);

        // Assign the new value to the symbol.
        sym->set_value(migrate_expr(new_value));
      }
      catch (const std::exception &e)
      {
        log_error(
          "update_symbol: Failed to convert binary value '{}' to integer for "
          "symbol '{}'. Error: {}",
          binary_value_str,
          sym->id.as_string(),
          e.what());
      }
    }
  }
}

/// Module file and the name that module defines, for the top-level
/// `from <module> import <bound>` binding @p bound; an empty path when none
/// does. A bare `from m import B` leaves only the class name at the use site,
/// so the module cannot be read off the base expression the way `m.B` spells
/// it (#6745). Python binds the last import of a name, so later statements
/// win, and an `as` alias hides the name the module really defines.
static std::pair<std::string, std::string>
from_import_binding(const nlohmann::json &ast, const std::string &bound)
{
  std::pair<std::string, std::string> binding, wildcard;

  for (const auto &stmt : ast["body"])
  {
    if (
      stmt.value("_type", "") != "ImportFrom" || !stmt.contains("names") ||
      !stmt["names"].is_array() || !stmt.contains("full_path") ||
      stmt["full_path"].is_null())
      continue;

    for (const auto &alias : stmt["names"])
    {
      const std::string name = alias.value("name", "");
      // `from m import *` binds every name m defines, this one included, but
      // an explicit import of the name still wins over it (#7399).
      if (name == "*")
      {
        wildcard = {stmt["full_path"].get<std::string>(), bound};
        continue;
      }
      const bool aliased =
        alias.contains("asname") && !alias["asname"].is_null();
      if ((aliased ? alias["asname"].get<std::string>() : name) == bound)
        binding = {stmt["full_path"].get<std::string>(), name};
    }
  }

  return binding.first.empty() ? wildcard : binding;
}

/// Class name a `bases` entry denotes. A qualified base (`module.Class`)
/// parses as an Attribute carrying `attr` instead of `id` -- reading `id`
/// unconditionally aborted on any model-provided base such as
/// unittest.TestCase (#6745). The symbol id uses the bare name either way.
static std::string base_class_name(const nlohmann::json &base_class_node)
{
  return base_class_node.contains("id")
           ? base_class_node["id"].get<std::string>()
           : base_class_node.value("attr", "");
}

std::pair<const nlohmann::json *, std::string>
python_converter::find_imported_class_module(
  const std::string &class_name) const
{
  std::pair<const nlohmann::json *, std::string> found{nullptr, {}};

  for (const auto &[module_name, module_ast] : module_ast_pool_)
  {
    const std::string path = get_imported_module_path(module_name);
    if (path.empty())
      continue;
    if (json_utils::find_class(module_ast["body"], class_name).is_null())
      continue;
    if (found.first)
      return {nullptr, {}}; // defined in 2+ modules -- don't guess
    found = {&module_ast, path};
  }

  return found;
}

symbolt *python_converter::find_function_in_imported_base_classes(
  const std::string &class_name,
  const std::string &method_name,
  bool is_ctor) const
{
  const auto [module_ast, module_path] = find_imported_class_module(class_name);
  if (!module_ast)
    return nullptr;

  nlohmann::json class_node =
    json_utils::find_class((*module_ast)["body"], class_name);

  for (const auto &base_class_node : class_node["bases"])
  {
    const std::string base_class = base_class_name(base_class_node);
    if (base_class.empty())
      continue;

    class symbol_id base_id(
      module_path, base_class, is_ctor ? base_class : method_name);
    if (symbolt *func = symbol_table_.find_symbol(base_id.to_string()))
      return func;

    if (
      symbolt *func = find_function_in_imported_base_classes(
        base_class, method_name, is_ctor))
      return func;
  }

  return nullptr;
}

/// A base reached through an import lives in that module's file, so an id
/// rewritten against the current file cannot name it. Both spellings reach
/// here: `module.Base` names its module at the use site, while
/// `from module import Base` leaves only the bare name (#6745).
symbolt *python_converter::find_method_in_imported_base(
  const nlohmann::json &base_class_node,
  const std::string &base_class,
  const std::string &method_name,
  bool is_ctor) const
{
  const bool qualified = base_class_node.contains("value") &&
                         base_class_node["value"].is_object() &&
                         base_class_node["value"].contains("id");

  std::string module_path;
  std::string defined_name = base_class;

  if (qualified)
    module_path = get_imported_module_path(
      base_class_node["value"]["id"].get<std::string>());
  // A class this file defines itself shadows an import of the same name;
  // reaching into the module would bind the method off an unrelated class.
  else if (
    json_utils::find_class((*ast_json)["body"], base_class) == nlohmann::json())
  {
    const auto binding = from_import_binding(*ast_json, base_class);
    module_path = binding.first;
    defined_name = binding.second;
  }

  if (module_path.empty())
    return nullptr;

  class symbol_id base_id(
    module_path, defined_name, is_ctor ? defined_name : method_name);
  return symbol_table_.find_symbol(base_id.to_string());
}

symbolt *python_converter::find_method_in_model_base(
  const std::string &base_class,
  const std::string &sym_id) const
{
  if (!json_utils::find_class((*ast_json)["body"], base_class).is_null())
    return nullptr;
  return find_model_symbol(sym_id);
}

/// Whether import alias @p alias binds @p name: `import a.b` binds `a`, and
/// `from m import *` may bind any name.
static bool
alias_binds_name(const nlohmann::json &alias, const std::string &name)
{
  const std::string imported = alias.value("name", "");
  const nlohmann::json asname = alias.value("asname", nlohmann::json());
  return imported == "*" ||
         (asname.is_string() ? asname.get<std::string>()
                             : imported.substr(0, imported.find('.'))) == name;
}

/// Whether @p node of type @p type names @p name as what it binds: an import
/// alias, a parameter, or a def or class statement.
static bool binds_by_name(
  const nlohmann::json &node,
  const std::string &type,
  const std::string &name,
  bool any_scope)
{
  if (type == "alias")
    return alias_binds_name(node, name);
  if (type == "arg")
    return any_scope && node.value("arg", "") == name;
  const nlohmann::json bound = node.value("name", nlohmann::json());
  return bound.is_string() && bound.get<std::string>() == name;
}

static bool may_bind_name(
  const nlohmann::json &node,
  const std::string &name,
  bool any_scope);

/// Whether anything @p node of type @p type holds may bind @p name, reading a
/// node that binds by itself rather than through a child.
static bool may_bind_within(
  const nlohmann::json &node,
  const std::string &type,
  const std::string &name,
  bool any_scope)
{
  const auto binds = [&](const nlohmann::json &child) {
    return may_bind_name(child, name, any_scope);
  };
  if (type == "Name")
    return node.value("id", "") == name &&
           node.value("ctx", nlohmann::json::object()).value("_type", "") !=
             "Load";
  // A bare annotation (`m: int`) records a type and binds nothing.
  if (type == "AnnAssign" && node.value("value", nlohmann::json()).is_null())
    return false;
  return std::any_of(node.begin(), node.end(), binds);
}

/// Whether @p node, a statement or part of one, may bind @p name. Bindings in
/// nested function and class bodies count only with @p any_scope, which a call
/// site needs: it sees the scope it stands in as well as the module's.
static bool may_bind_name(
  const nlohmann::json &node,
  const std::string &name,
  bool any_scope)
{
  if (node.is_array())
    return std::any_of(
      node.begin(), node.end(), [&](const nlohmann::json &child) {
        return may_bind_name(child, name, any_scope);
      });
  if (!node.is_object())
    return false;

  const std::string type = node.value("_type", "");
  if (binds_by_name(node, type, name, any_scope))
    return true;
  if (type == "alias" || type == "arg")
    return false;
  if (type == "FunctionDef" || type == "AsyncFunctionDef" || type == "ClassDef")
    return any_scope &&
           std::any_of(
             node.begin(), node.end(), [&](const nlohmann::json &child) {
               return may_bind_name(child, name, any_scope);
             });

  return may_bind_within(node, type, name, any_scope);
}

bool python_converter::class_binds_name(
  const std::string &class_name,
  const std::string &name) const
{
  const nlohmann::json *node = sole_binding(class_name);
  return node &&
         may_bind_name(node->value("body", nlohmann::json()), name, false);
}

const nlohmann::json *python_converter::sole_binding(
  const std::string &name,
  const std::string &statement) const
{
  const nlohmann::json *found = nullptr;
  for (const auto &stmt : (*ast_json)["body"])
  {
    // Any scope: a name the call site's own function binds is not this class.
    if (!may_bind_name(stmt, name, true))
      continue;
    if (found || stmt.value("_type", "") != statement)
      return nullptr;
    found = &stmt;
  }
  return found;
}

using scope_bindings = python_converter::scope_bindings;

static bool names_contain(const nlohmann::json &names, const std::string &name)
{
  return std::find(names.begin(), names.end(), name) != names.end();
}

/// Records an import of @p name, or a declaration handing it to an outer
/// scope. Whether @p node is such a statement.
static bool record_declaration(
  const nlohmann::json &node,
  const std::string &type,
  const std::string &name,
  scope_bindings &found)
{
  const nlohmann::json names = node.value("names", nlohmann::json::array());
  if (type == "Import" || type == "ImportFrom")
  {
    for (const auto &alias : names)
      found.by_import = found.by_import || alias_binds_name(alias, name);
    return true;
  }
  if (type == "Global" || type == "Nonlocal")
  {
    found.declared_outer = found.declared_outer || names_contain(names, name);
    return true;
  }
  return false;
}

/// Records how @p node binds @p name in its own scope. Function, class, lambda
/// and comprehension bodies are scopes of their own and are not entered.
static void collect_scope_bindings(
  const nlohmann::json &node,
  const std::string &name,
  scope_bindings &found)
{
  if (node.is_array())
  {
    for (const auto &child : node)
      collect_scope_bindings(child, name, found);
    return;
  }
  if (!node.is_object())
    return;
  const std::string type = node.value("_type", "");
  if (record_declaration(node, type, name, found))
    return;
  if (type == "Name")
  {
    found.by_other = found.by_other || may_bind_within(node, type, name, false);
    return;
  }
  found.by_other = found.by_other || binds_by_name(node, type, name, false);
  static const std::set<std::string> own_scopes = {
    "FunctionDef",
    "AsyncFunctionDef",
    "ClassDef",
    "Lambda",
    "ListComp",
    "SetComp",
    "DictComp",
    "GeneratorExp"};
  // A bare annotation (`m: int`) records a type and binds nothing.
  if (
    own_scopes.count(type) ||
    (type == "AnnAssign" && node.value("value", nlohmann::json()).is_null()))
    return;
  for (const auto &child : node)
    collect_scope_bindings(child, name, found);
}

static bool declares_global(const nlohmann::json &node, const std::string &name)
{
  if (
    node.is_object() && node.value("_type", "") == "Global" &&
    names_contain(node["names"], name))
    return true;
  return node.is_structured() &&
         std::any_of(node.begin(), node.end(), [&](const nlohmann::json &c) {
           return declares_global(c, name);
         });
}

static std::runtime_error
undecided_module_binding(const std::string &name, const nlohmann::json &use)
{
  return std::runtime_error(
    fmt::format(
      "'{}' at line {} is bound both by an import and by another statement, "
      "and the Python frontend cannot tell which binding it reads",
      name,
      use.value("lineno", 0)));
}

static std::pair<int, int> source_position(const nlohmann::json &node, bool end)
{
  return {
    node.value(end ? "end_lineno" : "lineno", 0),
    node.value(end ? "end_col_offset" : "col_offset", 0)};
}

/// Whether the code at @p at, under @p node, runs only when something defined
/// there is called or iterated, which need not follow the statements before it.
static bool deferred_at(const nlohmann::json &node, std::pair<int, int> at)
{
  static const std::set<std::string> deferring = {
    "FunctionDef", "AsyncFunctionDef", "ClassDef", "Lambda", "GeneratorExp"};
  if (
    node.is_object() && deferring.count(node.value("_type", "")) &&
    source_position(node, false) <= at && at <= source_position(node, true))
    return true;
  return node.is_structured() &&
         std::any_of(node.begin(), node.end(), [&](const nlohmann::json &c) {
           return deferred_at(c, at);
         });
}

/// At module level, with every binding of @p name a plain top-level
/// statement and @p use run in order, the last binding to complete before
/// @p use is the one it reads. Whether that is an import; throws otherwise.
static bool module_level_binding_is_import(
  const nlohmann::json &body,
  const std::string &name,
  const nlohmann::json &use)
{
  const std::pair<int, int> at = source_position(use, false);
  if (deferred_at(body, at))
    throw undecided_module_binding(name, use);
  std::optional<bool> is_import;
  for (const auto &stmt : body)
  {
    scope_bindings found;
    collect_scope_bindings(stmt, name, found);
    if (!found.by_import && !found.by_other)
      continue;
    const std::string type = stmt.value("_type", "");
    const bool definition =
      type == "FunctionDef" || type == "AsyncFunctionDef" || type == "ClassDef";
    if (
      type == "Delete" ||
      (!definition && (stmt.contains("body") || stmt.contains("cases"))))
      throw undecided_module_binding(name, use);
    if (source_position(stmt, true) <= at)
      is_import = found.by_import;
  }
  if (!is_import)
    throw undecided_module_binding(name, use);
  return *is_import;
}

/// How @p function binds @p name: by an import alone (true), as a parameter or
/// another local (false), or not at all (empty).
static std::optional<bool> function_binding_is_import(
  const nlohmann::json &function,
  const std::string &name,
  const nlohmann::json &use)
{
  if (may_bind_name(function["args"], name, true))
    return false;
  scope_bindings local;
  collect_scope_bindings(function["body"], name, local);
  if (local.declared_outer || (!local.by_import && !local.by_other))
    return std::nullopt;
  if (local.by_import && local.by_other)
    throw undecided_module_binding(name, use);
  return local.by_import;
}

std::optional<bool> python_converter::binding_is_import(
  const nlohmann::json &module,
  const scope_bindings &global,
  const nlohmann::json *function,
  bool in_class,
  bool in_order,
  const std::string &name,
  const nlohmann::json &use)
{
  if (function)
    if (const auto local = function_binding_is_import(*function, name, use))
      return local;

  const bool rebound = global.by_other || global.declared_outer;
  if (!global.by_import)
    return rebound ? std::optional<bool>(false) : std::nullopt;
  if (!rebound && !in_order)
    return true;
  if (global.declared_outer || function || in_class)
    throw undecided_module_binding(name, use);
  return module_level_binding_is_import(module["body"], name, use);
}

python_converter::scope_bindings python_converter::module_scope_bindings(
  const nlohmann::json &module,
  const std::string &name)
{
  scope_bindings found;
  collect_scope_bindings(module["body"], name, found);
  found.declared_outer = declares_global(module, name);
  return found;
}

bool python_converter::name_shadows_module(
  const nlohmann::json &receiver,
  const nlohmann::json &use) const
{
  if (
    receiver.value("_type", "") != "Name" ||
    !receiver.value("id", nlohmann::json()).is_string())
    return false;
  const std::string name = receiver["id"].get<std::string>();

  const std::optional<bool> is_import = binding_is_import(
    *ast_json,
    module_bindings(name),
    enclosing_function_node_,
    !current_class_name_.empty(),
    false,
    name,
    use);
  return is_import && !*is_import;
}

const python_converter::scope_bindings &
python_converter::module_bindings(const std::string &name) const
{
  const auto [entry, inserted] =
    module_bindings_memo_.try_emplace({ast_json, name});
  if (inserted)
    entry->second = module_scope_bindings(*ast_json, name);
  return entry->second;
}

bool python_converter::is_module_receiver(
  const std::string &module_name,
  const nlohmann::json &receiver,
  const nlohmann::json &use) const
{
  return is_imported_module(module_name) && !name_shadows_module(receiver, use);
}

const nlohmann::json *python_converter::sole_method_def(
  const std::string &class_name,
  const std::string &method) const
{
  const nlohmann::json *cls = sole_binding(class_name);
  if (!cls || !cls->contains("body"))
    return nullptr;
  const nlohmann::json *found = nullptr;
  for (const auto &stmt : cls->at("body"))
  {
    if (!may_bind_name(stmt, method, false))
      continue;
    if (found || stmt.value("_type", "") != "FunctionDef")
      return nullptr;
    found = &stmt;
  }
  return found;
}

bool python_converter::overridden_below(
  const std::string &class_name,
  const std::string &method) const
{
  std::function<bool(const nlohmann::json &)> visit = [&](
                                                        const nlohmann::json
                                                          &node) {
    if (node.is_array())
      return std::any_of(node.begin(), node.end(), visit);
    if (!node.is_object())
      return false;
    if (
      node.value("_type", "") == "ClassDef" &&
      node.value("name", "") != class_name &&
      may_bind_name(node.value("body", nlohmann::json()), method, false))
    {
      const auto mro = class_mro(node["name"].get<std::string>());
      if (!mro || std::find(mro->begin(), mro->end(), class_name) != mro->end())
        return true;
    }
    return std::any_of(node.begin(), node.end(), visit);
  };
  return visit((*ast_json)["body"]);
}

std::optional<std::vector<std::string>>
python_converter::c3_merge(std::vector<std::vector<std::string>> seqs)
{
  const auto in_a_tail = [&](const std::string &cls) {
    return std::any_of(seqs.begin(), seqs.end(), [&](const auto &seq) {
      return std::find(seq.begin() + 1, seq.end(), cls) != seq.end();
    });
  };

  std::vector<std::string> merged;
  for (;;)
  {
    seqs.erase(
      std::remove_if(
        seqs.begin(), seqs.end(), [](const auto &seq) { return seq.empty(); }),
      seqs.end());
    if (seqs.empty())
      return merged;

    const auto good =
      std::find_if(seqs.begin(), seqs.end(), [&](const auto &seq) {
        return !in_a_tail(seq.front());
      });
    if (good == seqs.end())
      return std::nullopt;

    const std::string head = good->front();
    for (auto &seq : seqs)
      if (seq.front() == head)
        seq.erase(seq.begin());
    merged.push_back(head);
  }
}

std::optional<std::vector<std::string>>
python_converter::class_mro(const std::string &class_name, mro_memo &memo) const
{
  // An entry stays empty while its class is being linearised, so reaching it
  // again (a cycle) yields no order; a finished entry is reused.
  const auto [entry, inserted] = memo.try_emplace(class_name);
  if (!inserted)
    return entry->second;

  const nlohmann::json *node = sole_binding(class_name);
  if (!node)
    return std::nullopt;

  std::vector<std::vector<std::string>> seqs;
  std::vector<std::string> bases;
  for (const auto &base : node->value("bases", nlohmann::json::array()))
  {
    const std::string base_name =
      base.value("_type", "") == "Name" ? base.value("id", "") : std::string();
    // object ends every MRO and declares no user method.
    if (base_name == "object")
      continue;
    std::optional<std::vector<std::string>> mro =
      base_name.empty() ? std::nullopt : class_mro(base_name, memo);
    if (!mro)
      return std::nullopt;
    seqs.push_back(std::move(*mro));
    bases.push_back(base_name);
  }

  seqs.push_back(std::move(bases));
  std::optional<std::vector<std::string>> mro = c3_merge(std::move(seqs));
  if (mro)
    mro->insert(mro->begin(), class_name);
  entry->second = mro;
  return mro;
}

std::optional<std::vector<std::string>>
python_converter::class_mro(const std::string &class_name) const
{
  mro_memo memo;
  return class_mro(class_name, memo);
}

static void collect_class_nodes(
  const nlohmann::json &node,
  std::vector<const nlohmann::json *> &classes)
{
  if (node.is_object() && node.value("_type", "") == "ClassDef")
    classes.push_back(&node);
  if (node.is_object() || node.is_array())
    for (const auto &child : node)
      collect_class_nodes(child, classes);
}

std::set<std::string>
python_converter::subclasses(const std::string &class_name) const
{
  std::vector<const nlohmann::json *> classes;
  collect_class_nodes((*ast_json)["body"], classes);
  std::set<std::string> family{class_name};
  for (bool grew = true; grew;)
  {
    grew = false;
    for (const nlohmann::json *cls : classes)
      for (const auto &base : (*cls)["bases"])
        grew |= family.count(base.value("id", "")) &&
                family.insert((*cls)["name"].get<std::string>()).second;
  }
  return family;
}

std::optional<std::string> python_converter::bound_method_id(
  const std::string &owner,
  const std::string &m) const
{
  const auto resolve = [&](const std::string &cls) {
    std::optional<std::string> found;
    if (const auto mro = class_mro(cls))
      for (auto c = mro->begin(); c != mro->end() && !found; ++c)
        if (class_binds_name(*c, m))
          found = *c;
    return found;
  };
  const std::optional<std::string> defining = resolve(owner);
  if (!defining)
    return std::nullopt;

  const nlohmann::json &body = (*sole_binding(*defining))["body"];
  const auto binds_m = [&](const nlohmann::json &stmt) {
    return may_bind_name(stmt, m, false);
  };
  const auto def = std::find_if(body.begin(), body.end(), binds_m);
  if (
    std::count_if(body.begin(), body.end(), binds_m) != 1 ||
    def->value("_type", "") != "FunctionDef" ||
    !(*def)["decorator_list"].empty())
    return std::nullopt;

  // `self.m` binds the method the object's own class resolves, so every
  // subclass must resolve it to the same one.
  const std::set<std::string> family = subclasses(owner);
  if (std::any_of(family.begin(), family.end(), [&](const std::string &cls) {
        return resolve(cls) != defining;
      }))
    return std::nullopt;

  return symbol_id(
           current_python_file, *defining, m == "__init__" ? *defining : m)
    .to_string();
}

std::optional<std::pair<bool, std::string>> python_converter::stored_function(
  const nlohmann::json &target,
  const nlohmann::json &value,
  const std::string &owner,
  const nlohmann::json *method) const
{
  const std::string kind = value.value("_type", "");
  const std::string name = value.value(kind == "Name" ? "id" : "attr", "");
  const nlohmann::json *def =
    kind == "Name" ? sole_binding(name, "FunctionDef") : nullptr;
  if (def && (*def)["name"] == name)
    return std::make_pair(
      false, symbol_id(current_python_file, "", name).to_string());
  if (
    kind != "Attribute" || !method || !(*method)["decorator_list"].empty() ||
    (*method)["args"]["args"].empty())
    return std::nullopt;

  const std::string self = (*method)["args"]["args"][0].value("arg", "");
  if (
    value["value"].value("id", "") != self ||
    target["value"].value("id", "") != self ||
    may_bind_name((*method)["body"], self, true))
    return std::nullopt;
  if (const auto id = bound_method_id(owner, name))
    return std::make_pair(true, *id);
  return std::nullopt;
}

/// Whether @p node names a hook (setattr, __dict__, ...) that can reach any
/// attribute.
static bool names_attribute_hook(const nlohmann::json &node)
{
  static const std::set<std::string> hooks{
    "setattr",
    "delattr",
    "__setattr__",
    "__delattr__",
    "__dict__",
    "__getattr__",
    "__getattribute__"};
  if (node.is_object())
    for (const char *key : {"id", "attr", "name"})
      if (
        node.contains(key) && node[key].is_string() &&
        hooks.count(node[key].get<std::string>()))
        return true;
  return false;
}

/// Whether @p node is not known to be only read.
static bool may_store(const nlohmann::json &node)
{
  const nlohmann::json ctx = node.value("ctx", nlohmann::json());
  return !ctx.is_object() || ctx.value("_type", "") != "Load";
}

/// The attribute an assignment statement stores a value into, or null.
static const nlohmann::json *
attribute_store_target(const nlohmann::json &node, const std::string &type)
{
  const nlohmann::json *target =
    type == "AnnAssign"                               ? &node["target"]
    : type == "Assign" && node["targets"].size() == 1 ? &node["targets"][0]
                                                      : nullptr;
  return target && target->value("_type", "") == "Attribute" &&
             !node["value"].is_null()
           ? target
           : nullptr;
}

void python_converter::record_function_store(
  const std::string &attr,
  const std::pair<bool, std::string> &stored)
{
  auto &held = attribute_functions_[ast_json]
                 ->try_emplace(attr, attribute_functions{stored.first, {}})
                 .first->second;
  if (held && held->bound == stored.first)
    held->ids.push_back(stored.second);
  else
    held.reset();
}

void python_converter::collect_attribute_stores(
  const nlohmann::json &node,
  const std::string &owner,
  const nlohmann::json *method)
{
  auto &module = attribute_functions_[ast_json];
  if (!module || names_attribute_hook(node))
    return module.reset();
  const std::string type = node.is_object() ? node.value("_type", "") : "";
  if (type == "Attribute" && may_store(node))
    (*module)[node["attr"].get<std::string>()].reset();

  if (type == "ClassDef")
    return collect_attribute_stores(
      node["body"], node["name"].get<std::string>(), nullptr);
  // A def nested in a method sees the method's own `self`.
  if (type == "FunctionDef" && !method)
    return collect_attribute_stores(node["body"], owner, &node);

  if (const nlohmann::json *target = attribute_store_target(node, type))
    if (
      const auto stored =
        stored_function(*target, node["value"], owner, method))
      return record_function_store(
        (*target)["attr"].get<std::string>(), *stored);

  if (node.is_structured())
    for (const auto &child : node)
      collect_attribute_stores(child, owner, method);
}

const python_converter::attribute_functions *
python_converter::function_attribute(const std::string &attr)
{
  const auto [module, unscanned] =
    attribute_functions_.try_emplace(ast_json, std::in_place);
  if (unscanned)
    collect_attribute_stores((*ast_json)["body"], "", nullptr);
  if (!module->second)
    return nullptr;
  const auto held = module->second->find(attr);
  return held == module->second->end() || !held->second ? nullptr
                                                        : &*held->second;
}

symbolt *python_converter::find_function_in_base_classes(
  const std::string &class_name,
  const std::string &symbol_id,
  std::string method_name,
  bool is_ctor) const
{
  // Find class node in the AST
  auto class_node = json_utils::find_class((*ast_json)["body"], class_name);

  if (class_node == nlohmann::json())
    return nullptr;

  // Name of the callee under the current class: for a constructor it is the
  // class name itself, otherwise the method name. The caller only reaches this
  // for methods/constructors, so symbol_id always carries the "@C@<class>@F@"
  // marker; pos and span are loop-invariant.
  const std::string current_func_name = is_ctor ? class_name : method_name;
  const std::size_t pos = symbol_id.rfind("@C@" + class_name);
  const std::size_t span =
    std::string("@C@" + class_name + "@F@" + current_func_name).length();

  // Under multiple inheritance the bases are searched in C3 order, which a
  // depth-first walk breaks: `class D(B, C)` with `B(A)`, `C(A)` must find
  // C's method before A's.
  if (const auto mro = class_mro(class_name))
    for (auto cls = std::next(mro->begin()); cls != mro->end(); ++cls)
    {
      std::string sym_id = symbol_id;
      sym_id.replace(
        pos, span, "@C@" + *cls + "@F@" + (is_ctor ? *cls : method_name));
      if (symbolt *func = symbol_table_.find_symbol(sym_id.c_str()))
        return func;
    }

  // Search the method in every base class, transitively (walk the MRO).
  // Python enforces acyclic inheritance, so this recursion terminates.
  for (const auto &base_class_node : class_node["bases"])
  {
    const std::string base_class = base_class_name(base_class_node);
    if (base_class.empty())
      continue;

    // Under the base class, a constructor is named after that base.
    const std::string base_func_name = is_ctor ? base_class : method_name;

    std::string sym_id = symbol_id;
    sym_id.replace(pos, span, "@C@" + base_class + "@F@" + base_func_name);

    if (symbolt *func = symbol_table_.find_symbol(sym_id.c_str()))
      return func;

    if (
      symbolt *func = find_method_in_imported_base(
        base_class_node, base_class, method_name, is_ctor))
      return func;

    if (symbolt *func = find_method_in_model_base(base_class, sym_id))
      return func;

    // Not defined directly in this base: descend into its own bases so a
    // method inherited from a grandparent (or higher) still resolves.
    if (
      symbolt *func = find_function_in_base_classes(
        base_class, sym_id, base_func_name, is_ctor))
      return func;

    // The base itself is defined in an imported module, whose classes this
    // AST does not hold, so that descent stopped one level too early (#7398).
    if (
      symbolt *func = find_function_in_imported_base_classes(
        base_class, method_name, is_ctor))
      return func;
  }

  return nullptr;
}

/// The name @p id asks for, as an import would spell it. from_string parses
/// the @C@/@F@ markers but not the trailing segment; when an id carries one,
/// that segment is the name and the class/function are only its scope. Reading
/// the function instead made every name used inside `def acos()` resolve to
/// `math.acos` from an unrelated `import math` (#6895).
static std::string
import_lookup_name(const std::string &id, const ::symbol_id &parsed)
{
  ::symbol_id scope = parsed;
  scope.set_object("");
  const std::string prefix = scope.to_string();
  if (
    id.size() > prefix.size() && id.compare(0, prefix.size(), prefix) == 0 &&
    id[prefix.size()] == '@')
    return id.substr(prefix.size() + 1);

  if (!parsed.get_class().empty())
    return parsed.get_class();
  if (!parsed.get_function().empty())
    return parsed.get_function();
  return parsed.get_object();
}

/// The imports of the module being converted, top-level ones first and then
/// those directly inside its functions. `from pkg import mod` names pkg's own
/// file, so each submodule it binds follows it as an import of that
/// submodule's file, held in @p submodules (#8375).
std::vector<const nlohmann::json *>
python_converter::import_nodes(std::deque<nlohmann::json> &submodules) const
{
  std::vector<const nlohmann::json *> imports;
  auto add = [&](const nlohmann::json &node) {
    const std::string type = node.value("_type", "");
    if (type != "Import" && type != "ImportFrom")
      return;
    imports.push_back(&node);
    if (type != "ImportFrom" || node["module"].is_null())
      return;
    for (const auto &alias : node["names"])
    {
      const auto submodule = imported_modules.find(
        node["module"].get<std::string>() + "." +
        alias["name"].get<std::string>());
      if (submodule == imported_modules.end())
        continue;
      nlohmann::json &import = submodules.emplace_back();
      import["_type"] = "Import";
      import["full_path"] = submodule->second;
      imports.push_back(&import);
    }
  };

  const nlohmann::json &body = (*ast_json)["body"];
  for (const auto &node : body)
    add(node);
  for (const auto &node : body)
    if (
      node["_type"] == "FunctionDef" && node.contains("body") &&
      node["body"].is_array())
      for (const auto &stmt : node["body"])
        add(stmt);
  return imports;
}

// In user code, `import numpy` binds only the name `numpy`: a bare `sum`,
// `min` or `max` is the builtin, not the operational model's function of
// that name, reached as `numpy.sum` under the model's own file. The numpy
// models themselves call each other by bare name.
bool python_converter::import_hides_builtin(
  const nlohmann::json &import,
  const std::string &symbol_id,
  const std::string &name) const
{
  const std::string module_file = import["full_path"].get<std::string>();
  const std::string numpy_model = "/models/numpy.py";
  return import["_type"] == "Import" &&
         symbol_id.find("/models/") == std::string::npos &&
         symbol_table_.find_symbol(
           "py:/esbmc-vfs/python/models/builtins.py@F@" + name) &&
         module_file.size() >= numpy_model.size() &&
         module_file.compare(
           module_file.size() - numpy_model.size(),
           numpy_model.size(),
           numpy_model) == 0 &&
         symbol_id.rfind("py:" + module_file + "@", 0) != 0;
}

// The name the import statement `obj` binds `lookup_name` to in its module:
// `lookup_name` itself, or for `from m import orig as alias` the original
// name. nullopt when the statement does not import it: a `from m import`
// only binds the names it lists, so `from other import sum` does not also
// hijack `max`.
static std::optional<std::string>
imported_module_name(const nlohmann::json &obj, const std::string &lookup_name)
{
  if (
    obj["_type"] != "ImportFrom" || !obj.contains("names") ||
    lookup_name.empty())
    return lookup_name;
  for (const auto &name : obj["names"])
  {
    const std::string &n = name["name"].get<std::string>();
    if (n == "*" || n == lookup_name)
      return lookup_name;
    if (
      name.contains("asname") && !name["asname"].is_null() &&
      name["asname"].get<std::string>() == lookup_name)
      return n;
  }
  return std::nullopt;
}

// The symbol `symbol_id` names in the module file `module_file`, where the
// module defines it as `name`.
symbolt *python_converter::find_symbol_in_module(
  const std::string &symbol_id,
  const std::string &module_file,
  const std::string &lookup_name,
  const std::string &name) const
{
  std::regex pattern("py:(.*?)@");
  std::string imported_symbol =
    std::regex_replace(symbol_id, pattern, "py:" + module_file + "@");

  // For an aliased import the trailing component of symbol_id is the alias;
  // swap it for the original name so the direct lookup targets the real
  // symbol (e.g. py:<collections>@deque, not py:<collections>@Queue).
  if (name != lookup_name && !lookup_name.empty())
  {
    const std::size_t at = imported_symbol.rfind('@');
    if (at != std::string::npos)
      imported_symbol.replace(at + 1, std::string::npos, name);
  }

  if (symbolt *func_symbol = symbol_table_.find_symbol(imported_symbol.c_str()))
    return func_symbol;
  if (name.empty())
    return nullptr;

  // Imported free functions are often looked up as object symbols in the
  // caller scope (e.g., py:main@replace). Also probe the equivalent
  // function-id form in the imported module (py:module@F@replace).
  ::symbol_id imported_sid = ::symbol_id::from_string(imported_symbol);
  imported_sid.set_class("");
  imported_sid.set_object("");
  imported_sid.set_attribute("");
  imported_sid.set_function(name);
  if (
    symbolt *func_symbol =
      symbol_table_.find_symbol(imported_sid.to_string().c_str()))
    return func_symbol;

  imported_sid.set_function("");
  imported_sid.set_object(name);
  return symbol_table_.find_symbol(imported_sid.to_string().c_str());
}

// The symbol `symbol_id` names in the module the import statement `obj`
// imports, as `lookup_name`; nullptr when it does not import it.
symbolt *python_converter::find_symbol_in_import(
  const nlohmann::json &obj,
  const std::string &symbol_id,
  const std::string &lookup_name) const
{
  if (
    (obj["_type"] != "ImportFrom" && obj["_type"] != "Import") ||
    !obj.contains("full_path") || obj["full_path"].is_null())
    return nullptr;
  const std::optional<std::string> name =
    imported_module_name(obj, lookup_name);
  if (!name || import_hides_builtin(obj, symbol_id, lookup_name))
    return nullptr;
  return find_symbol_in_module(
    symbol_id, obj["full_path"].get<std::string>(), lookup_name, *name);
}

symbolt *
python_converter::find_imported_symbol(const std::string &symbol_id) const
{
  // Extract the name being looked up from the symbol ID.
  // When the symbol has a class component (py:main@C@Foo@F@bar),
  // use the class name for matching against import names.
  auto parsed = ::symbol_id::from_string(symbol_id);
  std::string lookup_name = import_lookup_name(symbol_id, parsed);

  // symbol_id::from_string currently parses class/function components but not
  // trailing object names (e.g. py:file@replace). Recover that case from raw
  // text so imported free functions are still resolved.
  if (lookup_name.empty())
  {
    const std::size_t at = symbol_id.rfind('@');
    if (at != std::string::npos && at + 1 < symbol_id.size())
    {
      lookup_name = symbol_id.substr(at + 1);
      if (lookup_name == "C" || lookup_name == "F")
        lookup_name.clear();
    }
  }

  auto find_in_import = [&](const nlohmann::json &obj) -> symbolt * {
    return find_symbol_in_import(obj, symbol_id, lookup_name);
  };

  // A `from m import name` binds the name itself, so it takes precedence over
  // a plain `import m`, which reaches it only through the module (#9046).
  std::deque<nlohmann::json> submodules;
  const std::vector<const nlohmann::json *> imports = import_nodes(submodules);
  for (const bool from_import : {true, false})
    for (const nlohmann::json *import : imports)
      if (((*import)["_type"] == "ImportFrom") == from_import)
        if (symbolt *func_symbol = find_in_import(*import))
          return func_symbol;

  return nullptr;
}

symbolt *
python_converter::find_nested_function_symbol(const std::string &name) const
{
  if (name.empty() || current_func_name_.empty())
    return nullptr;

  for (std::string scope = current_func_name_; !scope.empty();)
  {
    ::symbol_id nested_func_sid(current_python_file, "", scope + "@F@" + name);
    if (
      symbolt *nested_func_symbol =
        symbol_table_.find_symbol(nested_func_sid.to_string()))
    {
      if (nested_func_symbol->get_type().is_code())
        return nested_func_symbol;
    }

    const std::size_t sep = scope.rfind("@F@");
    if (sep == std::string::npos)
      break;
    scope.resize(sep);
  }

  return nullptr;
}

symbolt *
python_converter::find_symbol_through_module(const std::string &sym_id) const
{
  // Imports bind names in the module being converted only: an id in another
  // module's file names that module's symbol (#9046).
  const std::string file = symbol_id::from_string(sym_id).get_filename();
  const std::size_t at = sym_id.rfind('@');
  if (!file.empty() && file != current_python_file && at != std::string::npos)
  {
    const std::string name = sym_id.substr(at + 1);
    return find_symbol_in_module(sym_id, file, name, name);
  }

  // User imports (e.g. "from other import sum") shadow builtin model
  // functions that are registered in the main file's namespace. Don't let
  // model-module stubs (e.g. esbmc.py's nondet_list stub) shadow operational
  // models (e.g. nondet.py's nondet_list).
  symbolt *imported = find_imported_symbol(sym_id);
  if (
    imported &&
    imported->id.as_string().find("/models/") != std::string::npos &&
    symbol_table_.find_symbol(sym_id))
    return nullptr;
  return imported;
}

symbolt *python_converter::find_symbol(const std::string &sym_id) const
{
  if (!is_loading_models)
    if (symbolt *symbol = find_symbol_through_module(sym_id))
      return symbol;

  if (symbolt *symbol = symbol_table_.find_symbol(sym_id))
    return symbol;

  symbol_id nested_sym_id = symbol_id::from_string(sym_id);
  std::size_t pos = sym_id.rfind('@');
  std::string obj = (pos == std::string::npos) ? "" : sym_id.substr(pos + 1);
  nested_sym_id.set_object(obj);

  for (int i = scope_stack_.size(); i >= 0; --i)
  {
    // Build id prefix from module + outer scopes
    for (int j = 0; j < i; ++j)
    {
      std::string func = scope_stack_[j].substr(3); // drop "@F@"
      nested_sym_id.set_function(func);
      std::string candidate = nested_sym_id.to_string();

      if (symbolt *symbol = symbol_table_.find_symbol(candidate))
        return symbol;
    }
  }

  if (symbolt *symbol = find_symbol_in_global_scope(sym_id))
    return symbol;

  if (symbolt *symbol = find_model_symbol(sym_id))
    return symbol;

  return find_imported_symbol(sym_id);
}

/// A model-provided name referenced from user code forms an id under the
/// caller's file; the model defining it lives in its own namespace. Retried
/// only for an id that names a file: an unresolved reference carries an empty
/// filename -- `random.random()` without `import random` forms `py:@F@random`
/// -- and resolving that would accept a module the program never imported.
symbolt *python_converter::find_model_symbol(const std::string &sym_id) const
{
  if (is_loading_models || sym_id.rfind("py:", 0) != 0)
    return nullptr;

  const std::size_t at = sym_id.find('@');
  if (at == std::string::npos || at <= 3)
    return nullptr;

  const std::string suffix = sym_id.substr(at);
  for (const std::string &ns : model_namespaces_)
    if (symbolt *symbol = symbol_table_.find_symbol("py:" + ns + suffix))
      return symbol;

  return nullptr;
}

symbolt *python_converter::find_symbol_in_global_scope(
  const std::string &symbol_id) const
{
  std::size_t class_start_pos = symbol_id.find("@C@");
  std::size_t func_start_pos = symbol_id.find("@F@");
  std::string sid = symbol_id;

  // Remove class name from symbol
  if (class_start_pos != std::string::npos)
    sid.erase(class_start_pos, func_start_pos - class_start_pos);

  func_start_pos = sid.find("@F@");
  std::size_t func_end_pos = sid.rfind("@");

  // Remove function name from symbol
  if (func_start_pos != std::string::npos)
    sid.erase(func_start_pos, func_end_pos - func_start_pos);

  return symbol_table_.find_symbol(sid);
}

bool python_converter::is_imported_module(const std::string &module_name) const
{
  if (imported_modules.find(module_name) != imported_modules.end())
    return true;

  // A module never imports itself, so its own name is not a module reference
  // inside it and a variable may legitimately carry that name. The fallback
  // below only asks whether a module of this name exists on disk, and the file
  // being converted always does -- so without this, `foo` in foo.py resolves to
  // the module and every method call on it fails to dispatch (#6639).
  if (module_name == current_module_name())
    return false;

  return json_utils::is_module(module_name, *ast_json);
}

/// Stem of the file being converted, i.e. the module name it would be imported
/// under.
std::string python_converter::current_module_name() const
{
  return std::filesystem::path(current_python_file).stem().string();
}

std::string python_converter::current_ast_scope() const
{
  if (
    current_class_name_.empty() || current_func_name_.empty() ||
    current_func_name_.find("@C@") != std::string::npos)
    return current_func_name_;

  const std::string outer =
    json_utils::split_function_path(current_func_name_).front();
  // __init__ is converted under its class's name.
  const std::string method =
    current_class_name_ + "@C@" +
    (outer == current_class_name_ ? "__init__" : outer);
  // A class nested in a function or class is not reachable as Cls@C@m.
  if (json_utils::find_function_by_path(*ast_json, {method}).empty())
    return current_func_name_;
  return method + current_func_name_.substr(outer.size());
}

symbolt &python_converter::create_tmp_symbol(
  const nlohmann::json &element,
  const std::string var_name,
  const typet &symbol_type,
  const exprt &symbol_value)
{
  return create_tmp_symbol(
    get_location_from_decl(element), var_name, symbol_type, symbol_value);
}

symbolt &python_converter::create_tmp_symbol(
  const locationt &location,
  const std::string var_name,
  const typet &symbol_type,
  const exprt &symbol_value)
{
  std::string path = location.file().as_string();
  std::string name_prefix =
    path + ":" + location.get_line().as_string() + var_name;
  symbolt &cl =
    sym_generator_.new_symbol(symbol_table_, symbol_type, name_prefix);
  cl.mode = "Python";
  std::string module_name = location.get_file().as_string();
  cl.module = module_name;
  cl.location = location;
  cl.static_lifetime = false;
  cl.is_extern = false;
  cl.file_local = true;
  if (symbol_value != exprt())
    cl.set_value(migrate_expr(symbol_value));

  return cl;
}
