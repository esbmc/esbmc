#include <python-frontend/function_call/builder.h>
#include <python-frontend/exception/python_exception_handler.h>
#include <python-frontend/function_call/expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/numpy/numpy_call_expr.h>
#include <python-frontend/python-list/python_list.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type/type_utils.h>
#include <util/arith/arith_tools.h>
#include <util/lang/c_types.h>
#include <util/lang/python_types.h>
#include <util/message/message.h>
#include <util/irep/std_expr.h>
#include <python-frontend/python_expr_builder.h>

#include <boost/algorithm/string/predicate.hpp>
#include <optional>
#include <set>
#include <unordered_map>

using namespace python_expr;

// Which OM class a tagged scalar's no-arg instance method belongs to.
// Extend this map, not the call site, when a new int/float OM method is
// added.
static const std::unordered_map<std::string, std::string> &
tagged_scalar_method_class()
{
  static const std::unordered_map<std::string, std::string> class_by_method = {
    {"bit_length", "int"},
    {"bit_count", "int"},
    {"conjugate", "int"},
    {"is_integer", "float"},
  };
  return class_by_method;
}

// True for a bare `:` slice, i.e. Slice(lower=None, upper=None, step=None).
// Mirrors converter_expr.cpp's helper of the same name: used here to tell
// whether a Tuple-sliced Subscript (`a[i, j]`) is one of the two supported
// 2-D slicing patterns (which produce an array) or a fully scalar multi-index
// (which does not), so len() dispatch doesn't misroute the latter.
static bool is_full_slice_node(const nlohmann::json &node)
{
  if (!(node.contains("_type") && node["_type"] == "Slice"))
    return false;
  auto is_absent = [&](const char *key) {
    return !node.contains(key) || node[key].is_null();
  };
  return is_absent("lower") && is_absent("upper") && is_absent("step");
}

static typet normalize_pylist_candidate_type(typet type, const namespacet &ns)
{
  if (type.is_pointer())
    type = type.subtype();
  if (type.id() == "symbol")
    type = ns.follow(type);
  return type;
}

static std::optional<struct_typet>
try_get_pylist_struct_type(const typet &type, const namespacet &ns)
{
  typet actual_type = normalize_pylist_candidate_type(type, ns);
  if (!actual_type.is_struct())
    return std::nullopt;

  const struct_typet &struct_type = to_struct_type(actual_type);
  const std::string tag = struct_type.tag().as_string();
  if (tag.find("__ESBMC_PyListObj") == std::string::npos)
    return std::nullopt;

  return struct_type;
}

static bool is_pylist_object_type(const typet &type, const namespacet &ns)
{
  return try_get_pylist_struct_type(type, ns).has_value();
}

/// `len(list)`. A contract clause cannot name the result of a call, so there it
/// reads the size field directly.
static exprt list_length(
  const exprt &list,
  const struct_typet &list_struct,
  bool in_contract_clause)
{
  if (in_contract_clause && list.type().is_pointer())
    return build_deref_member(list, "size", size_type());

  code_typet list_size_type;
  list_size_type.return_type() = size_type();
  code_typet::argumentt arg_type;
  arg_type.type() = pointer_typet(list_struct);
  list_size_type.arguments().push_back(arg_type);

  symbol_exprt list_size_func("c:@F@__ESBMC_list_size", list_size_type);

  return build_call(
    list_size_func,
    size_type(),
    {list.type().is_pointer() ? list : build_address_of(list)});
}

/// Python types that define no __len__. Named rather than tested inline so the
/// len() lowering stays one decision point wide.
static bool is_unsized_python_type(const std::string &python_type)
{
  return python_type == "int" || python_type == "float" ||
         python_type == "bool" || python_type == "complex";
}

const std::string kGetObjectSize = "__ESBMC_get_object_size";
const std::string kStrlen = "strlen";
const std::string kEsbmcAssume = "__ESBMC_assume";
const std::string kVerifierAssume = "__VERIFIER_assume";
const std::string kEsbmcAssert = "__ESBMC_assert";
const std::string kEsbmcUnreachable = "__ESBMC_unreachable";
const std::string kLoopInvariant = "__loop_invariant";
const std::string kEsbmcLoopInvariant = "__ESBMC_loop_invariant";
const std::string kEsbmcRequires = "__ESBMC_requires";
const std::string kEsbmcEnsures = "__ESBMC_ensures";
const std::string kEsbmcAssigns = "__ESBMC_assigns";
const std::string kEsbmcOld = "__ESBMC_old";
const std::string kEsbmcOldRaw = "__ESBMC_old_raw";
const std::string kEsbmcReturnValue = "__ESBMC_return_value";
const std::string kEsbmcCover = "__ESBMC_cover";
const std::string kEsbmcAtomicBegin = "__ESBMC_atomic_begin";
const std::string kEsbmcAtomicEnd = "__ESBMC_atomic_end";
const std::string kEsbmcSpawnThread = "__ESBMC_spawn_thread";
const std::string kPytInitTid = "__pyt_init_tid";
const std::string kPytJoin = "__pyt_join";
const std::string kPytTerminate = "__pyt_terminate";
const std::string kPyLockBlockAndCheck = "__ESBMC_pylock_block_and_check";
const std::string kPyLockReleaseWaiters = "__pyt_lock_release_waiters";

typet function_call_builder::enclosing_return_type() const
{
  symbol_id sid(
    converter_.python_file(),
    converter_.current_classname(),
    converter_.current_function_name());

  const symbolt *sym = converter_.symbol_table().find_symbol(sid.to_string());
  if (!sym || !sym->get_type().is_code())
    return typet();

  return to_code_type(sym->get_type()).return_type();
}

// Loop invariants and function-contract clauses share one shape: a bodiless
// `void f(bool)` whose call goto_convert lowers into a LOOP_INVARIANT or a
// marked ASSUME.
static bool needs_bool_intrinsic_symbol(const std::string &name)
{
  return name == kEsbmcLoopInvariant || name == kEsbmcRequires ||
         name == kEsbmcEnsures;
}

// Contract intrinsics the Python frontend does not lower yet. Named so that a
// clause mentioning one says why, rather than reporting only that the clause is
// impure and leaving the reader to guess which call was the problem.
static bool is_unsupported_contract_intrinsic(const std::string &name)
{
  return name == "__ESBMC_is_fresh" || name == "__ESBMC_forall" ||
         name == "__ESBMC_exists";
}

void function_call_builder::check_contract_call(
  const symbol_id &function_id) const
{
  const std::string &clause = function_id.get_function();

  if (clause == kEsbmcAssigns)
  {
    check_assigns_targets();
    return;
  }

  if (clause != kEsbmcRequires && clause != kEsbmcEnsures)
    return;

  // goto_convert aborts on any other arity, so reject it here where the user
  // still gets a line number and a suggestion.
  if (call_["args"].size() != 1)
    throw std::runtime_error(
      fmt::format(
        "{} at line {} takes exactly one argument, got {}; combine conditions "
        "with 'and'",
        clause,
        call_.value("lineno", 0),
        call_["args"].size()));

  check_contract_clause(call_["args"], clause);
}

// Node kinds a clause may contain. An allow-list rather than a deny-list: the
// clause becomes one ASSUME/ASSERT, so anything needing a temporary, a branch
// or an iteration does not survive it, and a construct nobody enumerated
// should fail loudly instead of lowering into something else. Denying only
// Call and Subscript let comprehensions and the walrus operator through.
// Operator and context nodes carry their own `_type`, so they are listed too.
// `In` / `NotIn` are absent because they are container operations.
static bool is_pure_clause_node(const std::string &node_type)
{
  static const std::set<std::string> pure = {
    "Name",  "Constant",  "BoolOp", "UnaryOp", "BinOp",  "Compare",
    "IfExp", "Attribute", "Load",   "And",     "Or",     "Not",
    "UAdd",  "USub",      "Invert", "Add",     "Sub",    "Mult",
    "Div",   "FloorDiv",  "Mod",    "Pow",     "LShift", "RShift",
    "BitOr", "BitXor",    "BitAnd", "Eq",      "NotEq",  "Lt",
    "LtE",   "Gt",        "GtE",    "Is",      "IsNot"};
  return pure.count(node_type) != 0;
}

/// Names the Python construct rather than the AST spelling where they differ.
static std::string describe_clause_node(const std::string &node_type)
{
  if (node_type == "Call")
    return "a function call";
  if (node_type == "Subscript")
    return "a subscript";
  if (node_type == "NamedExpr")
    return "an assignment expression";
  if (
    node_type == "ListComp" || node_type == "SetComp" ||
    node_type == "DictComp" || node_type == "GeneratorExp")
    return "a comprehension";
  return "a " + node_type + " expression";
}

/// Scalars the wrapper can snapshot by value. Anything else is a pointer into
/// the Python object model, whose extent and aliasing the contract layer has no
/// way to state yet (#6938 P4).
static bool is_snapshottable_scalar(const typet &type)
{
  return type.is_signedbv() || type.is_unsignedbv() || type.is_floatbv() ||
         type.is_bool();
}

std::optional<exprt> function_call_builder::build_contract_intrinsic(
  const symbol_id &function_id) const
{
  check_contract_call(function_id);

  if (function_id.get_function() == kEsbmcOld)
    return build_old_snapshot();

  if (function_id.get_function() == kEsbmcAssigns)
    return build_assigns_call();

  // Loop invariants and function-contract clauses share one shape: a bodiless
  // `void f(bool)` whose call goto_convert lowers into a LOOP_INVARIANT or a
  // marked ASSUME. The symbol must stay valueless: builtin_functions.cpp
  // routes any callee carrying a body to a plain FUNCTION_CALL instead.
  if (!needs_bool_intrinsic_symbol(function_id.get_function()))
    return std::nullopt;

  const std::string &func_symbol_id = function_id.to_string();
  if (converter_.symbol_table().find_symbol(func_symbol_id.c_str()) != nullptr)
    return std::nullopt;

  code_typet code_type;
  code_type.return_type() = empty_typet();
  code_typet::argumentt arg;
  arg.type() = bool_type();
  code_type.arguments().push_back(arg);

  symbolt symbol = converter_.create_symbol(
    converter_.python_file(),
    function_id.get_function(),
    func_symbol_id,
    converter_.get_location_from_decl(call_),
    code_type);
  converter_.add_symbol(symbol);
  return std::nullopt;
}

// A frame names places the caller can observe: a module-level global, or an
// attribute of an object the function reaches. A local or a parameter is
// invisible to the caller, so naming one states nothing.
void function_call_builder::check_assigns_targets() const
{
  for (const auto &target : call_["args"])
  {
    const std::string kind = target.value("_type", "");
    const std::string name = target.value("id", "");
    if (kind == "Attribute")
      continue;
    if (
      kind == "Name" && names_module_global(name) &&
      !names_enclosing_parameter(name))
      continue;
    throw std::runtime_error(
      fmt::format(
        "{} at line {} names {}; a frame target must be a module-level "
        "global or an attribute",
        kEsbmcAssigns,
        call_.value("lineno", 0),
        kind == "Name" ? "'" + name + "'" : describe_clause_node(kind)));
  }
}

// Lowered as the C macro spells it: __ESBMC_assigns_impl(&t1, &t2, ...), or
// __ESBMC_assigns_impl((void *)0) for an empty frame.
exprt function_call_builder::build_assigns_call() const
{
  const std::string impl_id = "c:@F@__ESBMC_assigns_impl";
  const locationt location = converter_.get_location_from_decl(call_);
  const symbolt *impl = converter_.symbol_table().find_symbol(impl_id);
  if (impl == nullptr)
  {
    code_typet code_type;
    code_type.return_type() = empty_typet();
    code_type.make_ellipsis();
    symbolt symbol = converter_.create_symbol(
      "", "__ESBMC_assigns_impl", impl_id, location, code_type);
    impl = converter_.add_symbol_and_get_ptr(symbol);
  }

  code_function_callt call;
  call.function() = symbol_expr(*impl);
  call.location() = location;
  for (const auto &target : call_["args"])
    call.arguments().push_back(address_of_exprt(converter_.get_expr(target)));
  if (call.arguments().empty())
    call.arguments().push_back(gen_zero(pointer_typet(empty_typet())));
  return call;
}

// Whether \p name is a parameter of the function being converted. Only a
// parameter or a module-level global has a value from before the body ran; a
// local does not exist yet at the point a snapshot would be taken.
bool function_call_builder::names_enclosing_parameter(
  const std::string &name) const
{
  symbol_id sid(
    converter_.python_file(),
    converter_.current_classname(),
    converter_.current_function_name());

  const symbolt *fn = converter_.symbol_table().find_symbol(sid.to_string());
  if (!fn || !fn->get_type().is_code())
    return false;

  sid.set_object(name);
  const std::string param_id = sid.to_string();
  for (const auto &argument : to_code_type(fn->get_type()).arguments())
    if (argument.get_identifier() == param_id)
      return true;

  return false;
}

bool function_call_builder::names_module_global(const std::string &name) const
{
  symbol_id sid(
    converter_.python_file(),
    converter_.current_classname(),
    converter_.current_function_name());
  sid.set_object(name);
  return converter_.symbol_table().find_symbol(sid.global_to_string()) !=
         nullptr;
}

exprt function_call_builder::build_old_snapshot() const
{
  const int line = call_.value("lineno", 0);

  // Outside a clause there is no "before" for the snapshot to name, and the
  // instruction it plants would have no reader. Reporting it beats leaving a
  // statement that looks like it did something.
  if (!converter_.in_contract_clause())
    throw std::runtime_error(
      fmt::format(
        "{} at line {} is only meaningful inside a contract clause",
        kEsbmcOld,
        line));

  if (call_["args"].size() != 1)
    throw std::runtime_error(
      fmt::format(
        "{} at line {} takes exactly one argument, got {}",
        kEsbmcOld,
        line,
        call_["args"].size()));

  const nlohmann::json &arg = call_["args"][0];
  if (arg.value("_type", "") != "Name")
    throw std::runtime_error(
      fmt::format(
        "{} at line {} takes a variable; {} cannot be snapshotted",
        kEsbmcOld,
        line,
        describe_clause_node(arg.value("_type", ""))));

  const std::string &name = arg.value("id", "");
  if (!names_enclosing_parameter(name) && !names_module_global(name))
    throw std::runtime_error(
      fmt::format(
        "{} at line {} takes '{}', which is neither a parameter nor a "
        "module-level global; a local has no pre-call value to snapshot",
        kEsbmcOld,
        line,
        name));

  exprt target = converter_.get_expr(arg);
  if (!is_snapshottable_scalar(target.type()))
    throw std::runtime_error(
      fmt::format(
        "{} at line {} takes '{}', which is not an int, float or bool; only "
        "scalars can be snapshotted by the Python frontend yet",
        kEsbmcOld,
        line,
        arg.value("id", "")));

  // Build what the C macro expands to,
  // `*(__typeof__(x)*)__ESBMC_old_raw((void*)(&x))`, which Python cannot spell
  // but the frontend can. Going through the same intrinsic keeps the snapshot
  // instruction, the wrapper materialisation and the call-site one shared with
  // C rather than reimplemented here.
  const std::string raw_id = "c:@F@" + kEsbmcOldRaw;
  const typet void_ptr = pointer_typet(empty_typet());
  locationt location = converter_.get_location_from_decl(call_);

  code_typet code_type;
  code_type.return_type() = void_ptr;
  code_typet::argumentt argument;
  argument.type() = void_ptr;
  code_type.arguments().push_back(argument);

  if (converter_.symbol_table().find_symbol(raw_id.c_str()) == nullptr)
  {
    symbolt symbol = converter_.create_symbol(
      converter_.python_file(), kEsbmcOldRaw, raw_id, location, code_type);
    converter_.add_symbol(symbol);
  }

  exprt address = address_of_exprt(target);
  address.make_typecast(void_ptr);

  side_effect_expr_function_callt call;
  call.function() = symbol_exprt(raw_id, code_type);
  call.arguments() = {address};
  call.type() = void_ptr;
  call.location() = location;

  exprt typed = call;
  typed.make_typecast(pointer_typet(target.type()));

  exprt snapshot("dereference", target.type());
  snapshot.copy_to_operands(typed);
  snapshot.location() = location;
  return snapshot;
}

const symbolt *
function_call_builder::clause_name_symbol(const std::string &name) const
{
  symbol_id sid(
    converter_.python_file(),
    converter_.current_classname(),
    converter_.current_function_name());
  sid.set_object(name);

  // A clause may also name a module-level global, which is not in the
  // function's scope.
  const symbolt *sym = converter_.symbol_table().find_symbol(sid.to_string());
  if (!sym)
    sym = converter_.symbol_table().find_symbol(sid.global_to_string());
  return sym;
}

static bool is_int_literal(const nlohmann::json &node)
{
  if (
    node.value("_type", "") == "UnaryOp" &&
    node["op"].value("_type", "") == "USub")
    return is_int_literal(node["operand"]);
  return node.value("_type", "") == "Constant" &&
         node["value"].is_number_integer();
}

// `len(l)` and `l[k]`, for a list `l` and an int literal `k`, lower to reads
// of the list object with no call or branch, so a clause may use them (#7629).
bool function_call_builder::is_pure_list_query(const nlohmann::json &node) const
{
  const std::string kind = node.value("_type", "");
  const nlohmann::json *base = nullptr;
  if (
    kind == "Call" && node["func"].value("id", "") == "len" &&
    node["args"].size() == 1 &&
    node.value("keywords", nlohmann::json()).empty())
    base = &node["args"][0];
  else if (kind == "Subscript" && is_int_literal(node["slice"]))
    base = &node["value"];

  if (base == nullptr || base->value("_type", "") != "Name")
    return false;
  const symbolt *sym = clause_name_symbol(base->value("id", ""));
  return sym && is_pylist_object_type(sym->get_type(), converter_.ns);
}

void function_call_builder::check_clause_name(
  const nlohmann::json &node,
  const std::string &clause) const
{
  const std::string name = node.value("id", "");

  if (name == kEsbmcReturnValue)
  {
    if (clause == kEsbmcRequires)
      throw std::runtime_error(
        fmt::format(
          "{} clause at line {} references {}; a precondition cannot mention "
          "the return value",
          clause,
          node.value("lineno", 0),
          kEsbmcReturnValue));

    if (returns_no_value(enclosing_return_type()))
      throw std::runtime_error(
        fmt::format(
          "{} clause at line {} references {}, but '{}' returns None",
          clause,
          node.value("lineno", 0),
          kEsbmcReturnValue,
          converter_.current_function_name()));
  }

  const symbolt *sym = clause_name_symbol(name);
  if (sym && sym->get_type() == any_type())
    throw std::runtime_error(
      fmt::format(
        "{} clause at line {} references '{}', whose type could not be "
        "determined; annotate the parameter so the clause constrains its value",
        clause,
        node.value("lineno", 0),
        name));
}

// A contract clause is lowered into one ASSUME/ASSERT, so its argument has to
// survive as a pure expression. Three Python constructs do not: a call binds
// to an SSA temporary that the contract wrapper never declares, so the clause
// silently becomes nondeterministic; a subscript expands into a bounds-check
// branch, which collapses the clause to a constant; and a parameter whose type
// could not be inferred is a `void *`, so `o > 0` compares a pointer rather
// than the value the source reads as. Reject all three instead of verifying a
// different property than the one written. A parameter whose type *is* known,
// including a list or str, stays allowed, as do `len(l)` and `l[k]` over a
// list, which lower without a temporary or a branch.
void function_call_builder::check_contract_clause(
  const nlohmann::json &node,
  const std::string &clause) const
{
  if (!node.is_object())
  {
    if (node.is_array())
      for (const auto &child : node)
        check_contract_clause(child, clause);
    return;
  }

  const std::string node_type = node.value("_type", "");
  const std::string callee =
    node_type == "Call" ? node["func"].value("id", "") : std::string();

  // __ESBMC_old(x) is part of the clause language rather than a call whose
  // result the clause depends on: it names the value x held before the body
  // ran, and lowers to a snapshot taken there.
  if (
    !node_type.empty() && !is_pure_clause_node(node_type) &&
    callee != kEsbmcOld && !is_pure_list_query(node))
  {
    throw std::runtime_error(
      fmt::format(
        "{} clause at line {} contains {}; a contract clause must be a pure "
        "expression{}",
        clause,
        node.value("lineno", 0),
        describe_clause_node(node_type),
        is_unsupported_contract_intrinsic(callee)
          ? fmt::format(
              " ({} is not supported by the Python frontend yet)", callee)
          : ""));
  }

  // A precondition already speaks about the pre-state, and nothing rewrites a
  // snapshot in it: `replace_old_in_expr` is applied to the ensures alone, so
  // the requires would be asserted over a symbol no instruction defines.
  if (callee == kEsbmcOld && clause == kEsbmcRequires)
    throw std::runtime_error(
      fmt::format(
        "{} at line {} says nothing in a precondition, which already speaks "
        "about the pre-state",
        kEsbmcOld,
        node.value("lineno", 0)));

  if (node_type == "Name")
    check_clause_name(node, clause);

  for (const auto &child : node)
    check_contract_clause(child, clause);
}

// A statement planted while converting a clause runs in the body, not where
// the clause is checked, so the clause would read a value it never computed.
void function_call_builder::check_clause_planted_nothing(
  const std::string &clause,
  size_t planted) const
{
  if (
    (clause == kEsbmcRequires || clause == kEsbmcEnsures) &&
    converter_.current_block_statement_count() != planted)
    throw std::runtime_error(
      fmt::format(
        "{} clause at line {} needs a statement to evaluate; a contract clause "
        "must be a pure expression",
        clause,
        call_.value("lineno", 0)));
}

function_call_builder::function_call_builder(
  python_converter &converter,
  const nlohmann::json &call)
  : converter_(converter), call_(call)
{
}

bool function_call_builder::is_cover_call(const symbol_id &function_id) const
{
  const std::string &func_name = function_id.get_function();
  return (func_name == kEsbmcCover);
}

bool function_call_builder::is_numpy_call(const symbol_id &function_id) const
{
  const std::string &function = function_id.get_function();
  if (
    type_utils::is_builtin_type(function) || function == "isinstance" ||
    function == "hasattr" ||
    boost::algorithm::starts_with(function, "nondet_") ||
    boost::algorithm::starts_with(function, "__VERIFIER_nondet_"))
    return false;

  const std::string &filename = function_id.get_filename();

  return boost::algorithm::ends_with(filename, "/models/numpy.py") ||
         filename.find("/numpy/linalg") != std::string::npos;
}

bool function_call_builder::is_assume_call(const symbol_id &function_id) const
{
  const std::string &func_name = function_id.get_function();
  return (func_name == kEsbmcAssume || func_name == kVerifierAssume);
}

bool function_call_builder::is_assert_call(const symbol_id &function_id) const
{
  return function_id.get_function() == kEsbmcAssert;
}

bool function_call_builder::is_unreachable_call(
  const symbol_id &function_id) const
{
  return function_id.get_function() == kEsbmcUnreachable;
}

bool function_call_builder::is_len_call(const symbol_id &function_id) const
{
  const std::string &func_name = function_id.get_function();
  return func_name == kGetObjectSize || func_name == kStrlen;
}

/// True when @p arg is `name[...]` whose slice node is of kind @p slice_kind
/// and whose base is a plain Name; the caller then tests that name's type.
static bool is_name_subscript(const nlohmann::json &arg, const char *slice_kind)
{
  return arg["_type"] == "Subscript" && arg.contains("slice") &&
         arg["slice"].is_object() &&
         arg["slice"].value("_type", "") == slice_kind &&
         arg.contains("value") && arg["value"].is_object() &&
         arg["value"].value("_type", "") == "Name";
}

namespace
{
/// How a list's name is used across the module, as far as it decides whether
/// the element types recorded for the list are all the list can hold.
struct list_name_uses
{
  int bindings = 0;
  /// The list[Class] element of each same-named parameter; "" when a
  /// parameter is not annotated so.
  std::set<std::string> params;
  /// A use the registry cannot follow: an item store, an alias, an argument.
  bool escapes = false;
};
} // namespace

static bool is_name(const nlohmann::json &node, const std::string &name)
{
  return node.is_object() && node.value("_type", "") == "Name" &&
         node.value("id", "") == name;
}

static void scan_list_uses(
  const nlohmann::json &node,
  const std::string &name,
  list_name_uses &uses);

static void scan_store_target(
  const nlohmann::json &target,
  const std::string &name,
  list_name_uses &uses)
{
  if (is_name(target, name))
    ++uses.bindings;
  else if (
    target.value("_type", "") == "Subscript" && is_name(target["value"], name))
    uses.escapes = true;
  else
    scan_list_uses(target, name, uses);
}

static std::string list_param_class(const nlohmann::json &arg)
{
  if (!arg.contains("annotation"))
    return std::string();
  const nlohmann::json &annotation = arg["annotation"];
  if (
    annotation.is_object() && annotation.value("_type", "") == "Subscript" &&
    is_name(annotation["value"], "list") &&
    annotation["slice"].value("_type", "") == "Name")
    return annotation["slice"].value("id", "");
  return std::string();
}

/// Records a bare use of the list's name: a parameter, or an occurrence no
/// enclosing node accounted for, which lets the list escape.
static bool scan_name_use(
  const nlohmann::json &node,
  const std::string &name,
  list_name_uses &uses)
{
  if (is_name(node, name))
  {
    uses.escapes = true;
    return true;
  }
  if (node.value("_type", "") == "arg" && node.value("arg", "") == name)
  {
    uses.params.insert(list_param_class(node));
    return true;
  }
  return false;
}

/// Whether the node reads an item of the list, calls a method on it or takes
/// its length: the registry follows the first two and the last reads nothing.
static bool scan_tracked_use(
  const nlohmann::json &node,
  const std::string &name,
  list_name_uses &uses)
{
  const std::string type = node.value("_type", "");
  if (
    (type == "Subscript" || type == "Attribute") &&
    is_name(node["value"], name))
  {
    if (node.contains("slice"))
      scan_list_uses(node["slice"], name, uses);
    return true;
  }
  return type == "Call" && is_name(node["func"], "len") &&
         node["args"].size() == 1 && is_name(node["args"][0], name);
}

static void scan_list_uses(
  const nlohmann::json &node,
  const std::string &name,
  list_name_uses &uses)
{
  if (node.is_array())
  {
    for (const auto &child : node)
      scan_list_uses(child, name, uses);
    return;
  }
  if (
    !node.is_object() || scan_name_use(node, name, uses) ||
    scan_tracked_use(node, name, uses))
    return;

  for (const auto &[key, child] : node.items())
  {
    if (key == "targets" && child.is_array())
      for (const auto &target : child)
        scan_store_target(target, name, uses);
    else if (key == "target")
      scan_store_target(child, name, uses);
    else if (!(key == "iter" && is_name(child, name)))
      scan_list_uses(child, name, uses);
  }
}

/// The classes of the elements recorded for a list, "" standing for an
/// element that is not a class instance.
static std::set<std::string> recorded_element_classes(
  const python_converter &converter,
  const std::string &id)
{
  std::set<std::string> classes;
  const auto *recorded = converter.get_element_type_registry().find(id);
  if (!recorded)
    return classes;

  for (const auto &elem : *recorded)
  {
    typet type = elem.second.is_pointer() ? elem.second.subtype() : elem.second;
    type = converter.name_space().follow(type);
    std::string tag =
      type.is_struct() ? to_struct_type(type).tag().as_string() : std::string();
    if (tag.rfind("tag-", 0) == 0)
      tag = tag.substr(4);
    classes.insert(json_utils::is_class(tag, converter.ast()) ? tag : "");
  }
  return classes;
}

std::string function_call_builder::subscript_receiver_class(
  const nlohmann::json &receiver) const
{
  if (
    receiver.value("_type", "") != "Subscript" ||
    receiver["value"].value("_type", "") != "Name")
    return "str";

  const std::string name = receiver["value"].value("id", "");
  symbol_id base_sid(
    converter_.python_file(),
    converter_.current_classname(),
    converter_.current_function_name());
  base_sid.set_object(name);
  const symbolt *base_sym = converter_.find_symbol(base_sid.to_string());
  if (
    !base_sym ||
    base_sym->get_type() != converter_.get_type_handler().get_list_type())
    return "str";

  const std::string list_id = base_sym->id.as_string();
  const std::set<std::string> classes =
    recorded_element_classes(converter_, list_id);
  if (classes.empty() || classes == std::set<std::string>{""})
    return "str";

  const std::string message = fmt::format(
    "method call on an element of list '{}' at line {} is not supported: its "
    "elements are not known to share one class",
    name,
    call_.value("lineno", 0));
  if (classes.size() > 1)
    throw std::runtime_error(message);

  // The registry follows pushes as they are converted, so the class it
  // reports holds only for a list that nothing else can fill: one binding (or
  // a parameter annotated with that class), no item stores and no aliases.
  const std::string &cls = *classes.begin();
  list_name_uses uses;
  scan_list_uses(converter_.ast()["body"], name, uses);
  const bool tracked =
    !uses.escapes &&
    ((uses.bindings == 1 && uses.params.empty()) ||
     (uses.bindings == 0 && uses.params == std::set<std::string>{cls}));
  if (!tracked)
    throw std::runtime_error(message);

  // A push converted later, e.g. further down a loop body, must not change
  // the class this call was dispatched on.
  converter_.defer_check([&converter = converter_, list_id, cls, message]() {
    if (recorded_element_classes(converter, list_id) != std::set{cls})
      throw std::runtime_error(message);
  });
  return cls;
}

// The class whose bases super() searches: the class its first argument
// names, as in super(C, self), or the enclosing class for a bare super().
static std::string super_start_class(
  const nlohmann::json &super_call,
  const std::string &current_class,
  const nlohmann::json &ast)
{
  if (!super_call.contains("args") || super_call["args"].empty())
    return current_class;
  const std::string named = super_call["args"][0].value("id", "");
  return json_utils::is_class(named, ast) ? named : current_class;
}

// `id`, a method's symbol id under class `from`, rewritten to name the same
// method under class `to`. A constructor is named after its class.
static std::string rebase_method_id(
  std::string id,
  const std::string &from,
  const std::string &to,
  const std::string &method,
  bool is_ctor)
{
  const std::string old_tail = "@C@" + from + "@F@" + (is_ctor ? from : method);
  const std::size_t pos = id.rfind(old_tail);
  if (from == to || pos == std::string::npos)
    return id;
  return id.replace(
    pos, old_tail.size(), "@C@" + to + "@F@" + (is_ctor ? to : method));
}

symbol_id function_call_builder::build_function_id() const
{
  const std::string &python_file = converter_.python_file();
  const std::string &current_class_name = converter_.current_classname();
  const std::string &current_function_name = converter_.current_function_name();
  const auto &ast = converter_.ast();
  type_handler th(converter_);

  bool is_member_function_call = false;

  const auto &func_json = call_["func"];

  const std::string &func_type = func_json["_type"];

  std::string func_name, obj_name, class_name;

  symbol_id function_id(python_file, current_class_name, current_function_name);

  if (func_type == "Name")
  {
    func_name = func_json["id"];

    // Map Python loop invariant name to ESBMC internal name
    if (func_name == kLoopInvariant)
      func_name = kEsbmcLoopInvariant;

    // `Alias = bytes` resolves fine as an annotation but not as a call
    // target; without this it crashes calling the alias as a function.
    if (
      !type_utils::is_builtin_type(func_name) &&
      !json_utils::is_class(func_name, ast))
    {
      const std::string resolved = th.resolve_builtin_alias(func_name);
      if (!resolved.empty())
        func_name = resolved;
    }
  }
  else if (func_type == "Attribute") // Handling obj_name.func_name() calls
  {
    is_member_function_call = true;
    func_name = func_json["attr"];

    // Get object name
    if (func_json["value"]["_type"] == "Attribute")
    {
      /* Handle nested attribute chains (e.g., self.f.foo(), a.b.c.method())
       *
       * When calling a method through an attribute chain, we need to determine
       * the class type of the intermediate object. For example, in
       * self.f.foo(), we need to know that 'f' has type Foo to correctly
       * resolve Foo.foo().
       *
       * Strategy: Recursively walk the attribute chain from left to right,
       * resolving each component's type by looking up struct members in the
       * symbol table, until we reach the final object whose class we need.
       */

      // Tracks the dotted module path being built during type resolution.
      // Set as a side-effect in the Name base case and extended in Attribute
      // nodes when type resolution fails, so that pkg.mod4.MyClass() can be
      // recognised as module "pkg.mod4" without a separate traversal pass.
      std::string module_path_candidate;

      std::function<typet(const nlohmann::json &)> resolve_attr_type =
        [&](const nlohmann::json &node) -> typet {
        if (node["_type"] == "Name")
        {
          // Base case: resolve variable to its type
          std::string name = node.value("id", "");
          if (name.empty())
            return typet();

          // Track the name for dotted-path reconstruction (used when type
          // resolution fails and we need to check for a module path).
          module_path_candidate = name;

          // Skip module names - let the module handling logic deal with them
          if (converter_.is_module_receiver(name, node, call_))
            return typet();

          symbol_id var_sid(
            python_file, current_class_name, current_function_name);
          var_sid.set_object(name);
          symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());
          return var_symbol ? var_symbol->get_type() : typet();
        }
        else if (node["_type"] == "Attribute")
        {
          // Recursive case: resolve base, then look up member
          typet base_type = resolve_attr_type(node["value"]);
          std::string attr = node.value("attr", "");
          if (attr.empty())
            return typet();
          if (base_type.id().empty())
          {
            // Type resolution failed for the base — extend the dotted path
            // candidate so the caller can check is_imported_module on the
            // full path (e.g., "pkg" + "mod4" -> "pkg.mod4").
            if (!module_path_candidate.empty())
              module_path_candidate += "." + attr;
            return typet();
          }

          // Normalize type (dereference pointers, follow symbol types)
          if (base_type.is_pointer())
            base_type = base_type.subtype();
          if (base_type.id() == "symbol")
            base_type = converter_.ns.follow(base_type);

          // Look up the member in the struct
          if (base_type.is_struct())
          {
            const struct_typet &struct_type = to_struct_type(base_type);
            if (struct_type.has_component(attr))
              return struct_type.get_component(attr).type();
            // Not an instance member — check for class-level symbol
            // (e.g. mutable_attr = [] declared in the class body).
            symbol_id cls_sid(python_file, struct_type.tag().as_string(), "");
            cls_sid.set_object(attr);
            symbolt *cls_sym = converter_.find_symbol(cls_sid.to_string());
            if (cls_sym)
              return cls_sym->get_type();
            return typet();
          }
          return typet();
        }
        return typet();
      };

      // Resolve the full attribute chain type
      typet obj_type = resolve_attr_type(func_json["value"]);

      if (!obj_type.id().empty())
      {
        // If the resolved type is a list (e.g. class-level mutable_attr = []),
        // map directly to the "list" builtin name.
        if (is_pylist_object_type(obj_type, converter_.ns))
        {
          obj_name = "list";
        }
        else
        {
          // Normalize the resolved type
          if (obj_type.is_pointer())
            obj_type = obj_type.subtype();
          if (obj_type.id() == "symbol")
            obj_type = converter_.ns.follow(obj_type);

          // Extract class name from struct type
          if (obj_type.is_struct())
          {
            std::string tag = to_struct_type(obj_type).tag().as_string();
            obj_name = (tag.find("tag-") == 0) ? tag.substr(4) : tag;
          }
          else
          {
            obj_name = th.type_to_string(obj_type);
          }
        }
      }
      else
      {
        // Type resolution failed. module_path_candidate was built as a
        // side-effect of resolve_attr_type; check whether it names a module
        // that is directly registered (e.g., "pkg.mod4" for
        // pkg.mod4.MyClass()). We require get_imported_module_path() to be
        // non-empty rather than just is_imported_module(), because
        // is_imported_module() can return true based solely on a JSON file
        // existing (is_module fallback), even for submodules like "os.path"
        // that are registered under their stem
        // ("path") rather than their dotted name. If the dotted name has no
        // direct path mapping, fall back to the last attribute component
        // (e.g., "path" for os.path.exists()) which IS in imported_modules.
        if (
          !module_path_candidate.empty() &&
          !converter_.get_imported_module_path(module_path_candidate).empty())
          obj_name = module_path_candidate;
        else
          obj_name = func_json["value"].value("attr", "");
      }
    }
    else if (
      func_json["value"]["_type"] == "Constant" &&
      func_json["value"]["value"].is_string())
    {
      obj_name = "str";
    }
    else if (func_json["value"]["_type"] == "BinOp")
    {
      // Mixed-type operands (e.g. a tagged int subclass mixed with a plain
      // int literal) are valid Python. get_operand_type already resolves a
      // BinOp node gracefully, falling back to whichever side is non-empty.
      obj_name = th.get_operand_type(func_json["value"]);
      // CPython's default int arithmetic always returns a plain `int`,
      // even with a subclass operand; none of the consensus int-subclasses
      // override an arithmetic dunder.
      if (type_utils::is_consensus_type(obj_name))
        obj_name = "int";
    }
    else if (func_json["value"]["_type"] == "Call")
    {
      const auto &inner_call = func_json["value"];
      if (
        inner_call.contains("func") && inner_call["func"].contains("_type") &&
        inner_call["func"]["_type"] == "Name" &&
        inner_call["func"].contains("id") &&
        inner_call["func"]["id"].is_string())
      {
        obj_name = inner_call["func"]["id"].get<std::string>();
      }
      else
      {
        // Nested calls (e.g., u.encode(...).decode(...)) do not always have
        // func.id. Use inferred call result type instead of indexing JSON
        // fields that may be null.
        obj_name = th.get_operand_type(inner_call);
      }

      if (obj_name.empty())
        obj_name = "str";

      if (obj_name == "nondet_str")
        obj_name = "str";

      if (obj_name == "super")
      {
        // For __init__, use is_ctor=true: parent's __init__ is stored under
        // the class name (e.g. Vehicle::Vehicle), not the literal "__init__".
        bool is_init_call = (func_name == "__init__");
        const std::string start =
          super_start_class(inner_call, current_class_name, ast);
        symbolt *base_class_func = converter_.find_function_in_base_classes(
          start,
          rebase_method_id(
            function_id.to_string(),
            current_class_name,
            start,
            func_name,
            is_init_call),
          func_name,
          is_init_call);
        if (base_class_func)
        {
          return symbol_id::from_string(base_class_func->id.as_string());
        }
      }
    }
    else
    {
      if (
        func_json["value"]["_type"] == "Name" &&
        func_json["value"].contains("id"))
        obj_name = func_json["value"]["id"];
      else
        obj_name = subscript_receiver_class(func_json["value"]);
    }

    obj_name = json_utils::get_object_alias(ast, obj_name);

    if (
      !json_utils::is_class(obj_name, ast) &&
      converter_.is_module_receiver(obj_name, func_json["value"], call_))
    {
      const auto &module_path = converter_.get_imported_module_path(obj_name);

      function_id =
        symbol_id(module_path, current_class_name, current_function_name);

      is_member_function_call = false;
    }
  }

  // build symbol_id
  if (func_name == "len")
  {
    const auto &arg = call_["args"][0];
    func_name = kStrlen;

    // Handle len(range(...))
    if (arg["_type"] == "Call")
    {
      if (
        arg.contains("func") && arg["func"].contains("id") &&
        arg["func"]["id"] == "range")
        func_name = kGetObjectSize;
      else if (
        arg.contains("func") && arg["func"].contains("id") &&
        arg["func"]["id"] == "set")
        func_name = kGetObjectSize;
      else if (
        arg.contains("func") && arg["func"].contains("id") &&
        arg["func"]["id"] == "dict")
      {
        // len(dict(...))
        // route to dict-aware size handler.
        func_name = "__ESBMC_len_dict";
        function_id.clear();
        function_id.set_prefix("esbmc:");
        function_id.set_function(func_name);
        return function_id;
      }
      else if (
        arg.contains("func") && arg["func"].is_object() &&
        arg["func"].value("_type", "") == "Attribute" &&
        arg["func"].value("attr", "") == "encode")
        // len(s.encode()): the encoded bytes are a wide-int array whose zero
        // high bytes make strlen stop at the first element, so size by element
        // count instead (matches the bytes variable and bytes-slice paths).
        func_name = kGetObjectSize;
    }
    else if (arg["_type"] == "List")
      func_name = kGetObjectSize;
    else if (arg["_type"] == "Dict")
    {
      // len({"a": 1}) -- a dict literal is dict-typed just as dict(...) is,
      // so it needs the same size handler. Without this it fell to strlen over
      // the dict struct and reported a wrong size; a dict bound to a name was
      // never affected, because it reaches the dict-aware path by type.
      func_name = "__ESBMC_len_dict";
      function_id.clear();
      function_id.set_prefix("esbmc:");
      function_id.set_function(func_name);
      return function_id;
    }
    else if (
      is_name_subscript(arg, "Slice") &&
      th.get_var_type(arg["value"]["id"].get<std::string>()) == "bytes")
    {
      // Inline len(b[a:b]) where b is bytes: the slice is a wide-int array, so
      // count elements rather than running strlen over the byte representation
      // (which stops at the first element's zero high bytes). String slices
      // keep the strlen path (their result is a null-terminated char array).
      func_name = kGetObjectSize;
    }
    else if (
      is_name_subscript(arg, "List") &&
      th.get_var_type(arg["value"]["id"].get<std::string>()) != "str")
    {
      // len(a[[0, 2]]): fancy indexing always selects multiple elements into
      // a fresh numeric array, never a null-terminated string; count elements
      // instead of running strlen.
      func_name = kGetObjectSize;
    }
    else if (
      arg["_type"] == "Subscript" && arg.contains("slice") &&
      arg["slice"].is_object() && arg["slice"].value("_type", "") == "Tuple" &&
      arg["slice"].contains("elts") && arg["slice"]["elts"].is_array() &&
      arg["slice"]["elts"].size() == 2 &&
      is_full_slice_node(arg["slice"]["elts"][0]) !=
        is_full_slice_node(arg["slice"]["elts"][1]) &&
      arg.contains("value") && arg["value"].is_object() &&
      arg["value"].value("_type", "") == "Name" &&
      th.get_var_type(arg["value"]["id"].get<std::string>()) != "str")
    {
      // len(a[i, :]) / len(a[:, j]): 2-D slicing on a numpy array produces a
      // fresh numeric array, not a null-terminated string; count elements
      // instead of running strlen. Only the two supported 2-D slicing
      // patterns (exactly one axis fully sliced) reach this branch -- a
      // fully scalar multi-index like `a[0, 1]` produces a scalar and must
      // keep falling through to normal scalar handling below.
      func_name = kGetObjectSize;
    }
    else if (arg["_type"] == "Name")
    {
      const std::string &var_type = th.get_var_type(arg["id"]);
      const std::string var_name = arg["id"].get<std::string>();
      symbol_id var_sid(python_file, current_class_name, current_function_name);
      var_sid.set_object(var_name);
      symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());

      auto symbol_points_to_list = [&](const symbolt *sym) -> bool {
        return sym && is_pylist_object_type(sym->get_type(), converter_.ns);
      };

      const bool var_symbol_is_list = symbol_points_to_list(var_symbol);

      // A dict-typed symbol may carry no "dict" frontend annotation: a dict
      // comprehension ({k: v for ...}) or a dict-returning call assigns the
      // __python_dict__ struct directly without recording the type in the
      // annotation map. Detect the struct type on the symbol so len() routes
      // to the dict-size handler rather than falling through to strlen, which
      // would read the struct's bytes as a C string and yield a wrong size.
      const bool var_symbol_is_dict =
        var_symbol && converter_.get_dict_handler()->is_dict_type(
                        converter_.ns.follow(var_symbol->get_type()));

      auto is_list_slice_assignment = [&]() -> bool {
        nlohmann::json var_value = json_utils::get_var_value(
          var_name,
          converter_.get_current_func_name(),
          converter_.get_ast_json());
        if (
          var_value.empty() || !var_value.contains("value") ||
          !var_value["value"].is_object())
          return false;

        const auto &value_node = var_value["value"];
        if (
          !value_node.contains("_type") || value_node["_type"] != "Subscript" ||
          !value_node.contains("slice") || !value_node["slice"].is_object() ||
          !value_node["slice"].contains("_type") ||
          value_node["slice"]["_type"] != "Slice" ||
          !value_node.contains("value") || !value_node["value"].is_object() ||
          !value_node["value"].contains("_type"))
          return false;

        const auto &base = value_node["value"];
        if (base["_type"] == "List")
          return true;

        if (base["_type"] == "Name" && base.contains("id"))
        {
          const std::string base_name = base["id"].get<std::string>();
          const std::string base_type = th.get_var_type(base_name);
          if (base_type == "list" || base_type == "List")
            return true;
          if (base_type == "str")
            return false;

          symbol_id base_sid(
            python_file, current_class_name, current_function_name);
          base_sid.set_object(base_name);
          symbolt *base_symbol = converter_.find_symbol(base_sid.to_string());
          return symbol_points_to_list(base_symbol);
        }

        return false;
      };

      // Check if this is a tuple by looking up the variable's type
      if (var_type == "tuple" || var_type.empty())
      {
        if (var_symbol && var_symbol->get_type().id() == "struct")
        {
          const struct_typet &struct_type =
            to_struct_type(var_symbol->get_type());

          // Check if this is a tuple by examining the tag
          if (struct_type.tag().as_string().find("tag-tuple") == 0)
          {
            // Mark this as a tuple len() call
            func_name = "__ESBMC_len_tuple";
            function_id.clear();
            function_id.set_prefix("esbmc:");
            function_id.set_function(func_name);
            return function_id;
          }
        }
      }
      if (var_type == "dict" || var_symbol_is_dict)
      {
        // Mark this as a dict len() call
        func_name = "__ESBMC_len_dict";
        function_id.clear();
        function_id.set_prefix("esbmc:");
        function_id.set_function(func_name);
        return function_id;
      }
      else if (
        var_type == "bytes" || var_type == "list" || var_type == "List" ||
        var_type == "set" || var_type == "Sequence" || var_type == "range")
      {
        func_name = kGetObjectSize;
      }
      else if (var_symbol_is_list)
      {
        // Prefer symbol type over inferred frontend type when they conflict.
        func_name = kGetObjectSize;
      }
      else if (var_type.empty() && is_list_slice_assignment())
      {
        // list slicing assigned to a variable may have no annotation.
        // Use list size semantics for len(sub) where sub = lst[a:b].
        func_name = kGetObjectSize;
      }
      else if (
        var_symbol && converter_.ns.follow(var_symbol->get_type()).is_array() &&
        converter_.ns.follow(var_symbol->get_type()).subtype() != char_type())
      {
        // A non-char array reaching here unannotated — e.g. a bytes slice
        // (s = b[a:b]), which loses its "bytes" frontend type — must be sized
        // by element count, not strlen: bytes elements are wide ints whose zero
        // high bytes make strlen stop at the first element.
        func_name = kGetObjectSize;
      }
    }
    function_id.clear();
    function_id.set_prefix("c:");
  }
  else if (type_utils::is_builtin_type(obj_name))
  {
    class_name = obj_name;
    function_id = symbol_id(python_file, class_name, func_name);
  }
  else if (
    is_assume_call(function_id) || is_assert_call(function_id) ||
    is_cover_call(function_id) || is_unreachable_call(function_id))
  {
    function_id.clear();
  }

  // Insert class name in the symbol id
  if (obj_name == "super")
  {
    class_name = current_class_name;
  }
  else if (th.is_constructor_call(call_))
  {
    // An explicit `Base.__init__(self, ...)` call names the receiver class and
    // passes self explicitly. is_constructor_call() flags any `.__init__`, but
    // here the constructor is the named class's own, stored under the class
    // name (@C@Base@F@Base), not the literal "__init__". Resolve both the class
    // and the (renamed) function to the receiver so the symbol is found; a
    // direct `Base()` call keeps func_name as the class name already.
    if (func_name == "__init__" && json_utils::is_class(obj_name, ast))
    {
      class_name = obj_name;
      func_name = obj_name;
    }
    else
      class_name = func_name;
  }
  else if (is_member_function_call)
  {
    if (
      type_utils::is_builtin_type(obj_name) ||
      json_utils::is_class(obj_name, ast))
    {
      class_name = obj_name;
    }
    else
    {
      // Look up variable type from symbol table instead of AST
      symbol_id var_sid(python_file, current_class_name, current_function_name);
      var_sid.set_object(obj_name);
      symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());

      if (!var_symbol)
      {
        // Reaching here means the attribute-chain resolver above could not
        // determine a struct type for the receiver (most commonly because a
        // function parameter is unannotated and python_annotation could not
        // infer it from any call site), and `obj_name` ended up as the
        // trailing attribute name rather than a real variable. Surface a
        // diagnostic that names the unresolved member call and points at
        // the likely missing annotation, instead of the cryptic
        // "Variable <attr> not found" the call previously emitted.
        throw std::runtime_error(
          "Could not resolve type of receiver in member call '." + obj_name +
          "." + func_name +
          "()'. Add a type annotation to the function parameter or local "
          "variable so the attribute chain can be typed.");
      }

      // A tagged receiver's struct tag isn't a real class; isinstance()
      // already narrowed its runtime type_id, so dispatch by method name.
      if (th.is_tagged_scalar_type(var_symbol->get_type()))
      {
        auto it = tagged_scalar_method_class().find(func_name);
        class_name = it != tagged_scalar_method_class().end() ? it->second : "";
      }
      else
      {
        // Extract class name from the type, following symbol references
        typet var_type = var_symbol->get_type().is_pointer()
                           ? var_symbol->get_type().subtype()
                           : var_symbol->get_type();

        // Follow symbol type references using the converter's namespace
        var_type = converter_.ns.follow(var_type);

        if (var_type.is_struct())
        {
          const struct_typet &struct_type = to_struct_type(var_type);
          class_name = struct_type.tag().as_string();
          use_imported_class_file(function_id, *var_symbol, class_name);
        }
        else
          class_name = th.type_to_string(var_type);
      }
    }
  }

  if (!class_name.empty())
  {
    function_id.set_class(class_name);
  }

  function_id.set_function(func_name);

  // Check if this is a nested function call
  if (func_type == "Name" && !current_function_name.empty())
  {
    // Self-recursive call (e.g., bar() inside foo@F@bar)
    if (current_function_name.ends_with("@F@" + func_name))
    {
      function_id.set_function(current_function_name);
      return function_id;
    }

    // Walk nesting chain from deepest to shallowest
    for (std::string ctx = current_function_name; !ctx.empty();)
    {
      std::string nested_id = ctx + "@F@" + func_name;
      symbol_id sid(python_file, current_class_name, nested_id);

      if (nested_function_exists(sid))
      {
        function_id.set_function(nested_id);
        return function_id;
      }

      size_t pos = ctx.rfind("@F@");
      ctx = (pos != std::string::npos) ? ctx.substr(0, pos) : "";
    }
  }

  return function_id;
}

// A receiver bound by `from m import obj` has its class defined in m, so its
// methods live under m's file, not this one (#8375).
void function_call_builder::use_imported_class_file(
  symbol_id &function_id,
  const symbolt &receiver,
  const std::string &class_name) const
{
  if (
    receiver.module == converter_.python_file() ||
    !json_utils::find_class(converter_.ast()["body"], class_name).empty())
    return;
  const std::string module_path =
    converter_.find_imported_class_module(class_name).second;
  if (!module_path.empty())
    function_id.set_filename(module_path);
}

bool function_call_builder::nested_function_exists(const symbol_id &sid) const
{
  // A sibling defined later in the enclosing body is not converted yet.
  return converter_.symbol_table().find_symbol(sid.to_string()) ||
         !json_utils::find_function_by_path(
            converter_.ast(),
            json_utils::split_function_path(sid.get_function()))
            .empty();
}

exprt function_call_builder::len_of_bitvector_operand(
  const nlohmann::json &arg) const
{
  // A single character also lowers to a bitvector and does have length 1, so
  // tell the two apart by the Python type. Anything the annotator could not
  // type keeps the historical answer: get_operand_type reports a BinOp as its
  // left operand's type, so `3 * "x,"` looks like an int, and refusing on that
  // alone would reject a real str/list.
  const std::string arg_py_type =
    converter_.get_type_handler().get_operand_type(arg);

  if (is_unsized_python_type(arg_py_type))
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "object of type '" + arg_py_type + "' has no len()");

  return from_integer(1, long_long_int_type());
}

static std::string none_has_no_attribute(const nlohmann::json &call)
{
  return "'NoneType' object has no attribute '" +
         call["func"]["attr"].get<std::string>() + "'";
}

bool function_call_builder::receiver_is_bound_to_none() const
{
  const nlohmann::json &receiver = call_["func"]["value"];
  if (receiver.value("_type", "") != "Name")
    return false;
  const std::string &name = receiver["id"].get_ref<const std::string &>();
  if (json_utils::count_name_bindings(converter_.ast(), name) != 1)
    return false;
  const nlohmann::json decl = json_utils::find_var_decl(
    name, converter_.current_function_name(), converter_.ast());
  return decl.contains("value") && decl["value"].is_object() &&
         decl["value"].value("_type", "") == "Constant" &&
         decl["value"]["value"].is_null();
}

static bool annotation_is_optional(const nlohmann::json &ann)
{
  return (ann.value("_type", "") == "Subscript" &&
          ann["value"].value("id", "") == "Optional") ||
         type_utils::annotation_admits_none(ann);
}

// A None value, directly or as a branch of a conditional expression.
static bool is_none_valued(const nlohmann::json &value)
{
  if (!value.is_object())
    return false;
  if (value.value("_type", "") == "IfExp")
    return is_none_valued(value["body"]) || is_none_valued(value["orelse"]);
  return value.value("_type", "") == "Constant" && value["value"].is_null();
}

// Whether `node` is a parameter, or a single-target assignment, of `name`.
static bool binds_name(const nlohmann::json &node, const std::string &name)
{
  const std::string kind = node.value("_type", "");
  if (kind == "arg")
    return node["arg"] == name;
  const nlohmann::json *target = nullptr;
  if (kind == "AnnAssign")
    target = &node["target"];
  else if (kind == "Assign" && node["targets"].size() == 1)
    target = &node["targets"][0];
  return target && target->value("_type", "") == "Name" &&
         (*target)["id"] == name;
}

// Whether some parameter, declaration or assignment of `name` under `node`
// admits None, through its annotation or its value.
static bool may_hold_none(const nlohmann::json &node, const std::string &name)
{
  if (
    node.is_object() && binds_name(node, name) &&
    ((node.contains("annotation") && node["annotation"].is_object() &&
      annotation_is_optional(node["annotation"])) ||
     (node.contains("value") && is_none_valued(node["value"]))))
    return true;
  for (const auto &child : node)
    if (child.is_structured() && may_hold_none(child, name))
      return true;
  return false;
}

void function_call_builder::raise_if_none_receiver() const
{
  const nlohmann::json &receiver = call_["func"]["value"];
  if (
    receiver.value("_type", "") != "Name" ||
    !converter_.can_emit_runtime_guard() ||
    !may_hold_none(converter_.ast(), receiver["id"].get<std::string>()))
    return;

  symbol_id sid = converter_.create_symbol_id();
  sid.set_object(receiver["id"].get<std::string>());
  const symbolt *symbol = converter_.find_symbol(sid.to_string());
  if (!symbol || !symbol->get_type().is_pointer())
    return;

  const typet &pointee = symbol->get_type().subtype();
  const std::string tag =
    pointee.id() == "symbol" ? pointee.identifier().as_string() : "";
  const bool is_instance =
    boost::starts_with(tag, "tag-") &&
    json_utils::is_class(tag.substr(4), converter_.ast());
  if (!is_instance && !type_utils::is_string_type(symbol->get_type()))
    return;

  const exprt ref = symbol_expr(*symbol);
  converter_.emit_guarded_raise(
    equality_exprt(ref, gen_zero(ref.type())),
    "AttributeError",
    none_has_no_attribute(call_),
    converter_.get_location_from_decl(call_));
}

std::string
function_call_builder::instance_class_name(const nlohmann::json &node) const
{
  const std::string kind = node.value("_type", "");
  if (kind == "Call")
  {
    const nlohmann::json &func = node["func"];
    if (
      func.value("_type", "") == "Name" &&
      json_utils::is_class(func["id"].get<std::string>(), converter_.ast()))
      return func["id"].get<std::string>();
    return "";
  }

  typet type;
  if (kind == "Name")
  {
    symbol_id sid = converter_.create_symbol_id();
    sid.set_object(node["id"].get<std::string>());
    const symbolt *symbol = converter_.find_symbol(sid.to_string());
    if (!symbol)
      return "";
    type = symbol->get_type();
  }
  else if (kind == "Attribute")
  {
    const std::string base = instance_class_name(node["value"]);
    const symbolt *base_class = converter_.find_symbol("tag-" + base);
    const std::string attr = node["attr"].get<std::string>();
    if (
      base.empty() || !base_class || !base_class->get_type().is_struct() ||
      !to_struct_type(base_class->get_type()).has_component(attr))
      return "";
    type = to_struct_type(base_class->get_type()).get_component(attr).type();
  }

  if (type.is_pointer())
    type = type.subtype();
  std::string tag;
  if (type.id() == "symbol")
    tag = type.identifier().as_string();
  else if (type.is_struct())
    tag = to_struct_type(type).tag().as_string();
  return boost::starts_with(tag, "tag-") ? tag.substr(4) : tag;
}

bool function_call_builder::receiver_class_defines_method() const
{
  const auto &ast = converter_.ast();
  const std::string cls = instance_class_name(call_["func"]["value"]);
  return json_utils::is_class(cls, ast) &&
         json_utils::class_defines_method(
           ast["body"], cls, call_["func"]["attr"].get<std::string>());
}

exprt function_call_builder::build_method_call_on_receiver() const
{
  if (receiver_is_bound_to_none())
    return converter_.get_exception_handler().gen_exception_raise(
      "AttributeError", none_has_no_attribute(call_));
  raise_if_none_receiver();
  if (receiver_class_defines_method())
    return nil_exprt();
  exprt number_error = str_method_on_number_error();
  if (!number_error.is_nil())
    return number_error;
  return converter_.get_string_handler().handle_string_attribute_call(call_);
}

static std::string constant_number_type_name(const nlohmann::json &value)
{
  if (value.is_boolean())
    return "bool";
  if (value.is_number_float())
    return "float";
  return value.is_number() ? "int" : "";
}

exprt function_call_builder::str_method_on_number_error() const
{
  // set(dir(str)) - set(dir(int)) - set(dir(float)), public names only.
  static const std::set<std::string> str_only_methods = {
    "capitalize",   "casefold",     "center",    "count",     "encode",
    "endswith",     "expandtabs",   "find",      "format",    "format_map",
    "index",        "isalnum",      "isalpha",   "isascii",   "isdecimal",
    "isdigit",      "isidentifier", "islower",   "isnumeric", "isprintable",
    "isspace",      "istitle",      "isupper",   "join",      "ljust",
    "lower",        "lstrip",       "maketrans", "partition", "removeprefix",
    "removesuffix", "replace",      "rfind",     "rindex",    "rjust",
    "rpartition",   "rsplit",       "rstrip",    "split",     "splitlines",
    "startswith",   "strip",        "swapcase",  "title",     "translate",
    "upper",        "zfill"};
  const std::string method = call_["func"]["attr"].get<std::string>();
  if (!str_only_methods.count(method))
    return nil_exprt();

  const nlohmann::json &recv = call_["func"]["value"];
  std::string type_name;
  if (recv.value("_type", "") == "Constant")
    type_name = constant_number_type_name(recv["value"]);
  else if (recv.value("_type", "") == "Name")
  {
    symbol_id sid = converter_.create_symbol_id();
    sid.set_object(recv["id"].get<std::string>());
    const symbolt *symbol = converter_.find_symbol(sid.to_string());
    const type_handler &types = converter_.get_type_handler();
    // get_expr follows a retype to str (`x = 3; x = "ab"`), the raw symbol not.
    if (symbol && types.is_numeric_scalar_type(symbol->get_type()))
    {
      const typet current = converter_.get_expr(recv).type();
      if (types.is_numeric_scalar_type(current))
        type_name = types.get_python_type_name(current);
    }
  }
  if (type_name.empty())
    return nil_exprt();

  exprt raise = converter_.get_exception_handler().gen_exception_raise(
    "AttributeError",
    "'" + type_name + "' object has no attribute '" + method + "'");
  raise.location() = converter_.get_location_from_decl(call_);
  return raise;
}

exprt function_call_builder::build() const
{
  if (call_["func"]["_type"] == "Attribute")
  {
    exprt handled_method = build_method_call_on_receiver();
    if (!handled_method.is_nil())
      return handled_method;
  }

  symbol_id function_id = build_function_id();
  if (is_len_call(function_id) && !call_["args"].empty())
  {
    exprt arg_expr = converter_.get_expr(call_["args"][0]);

    // If len() argument is a list-typed symbol, force list-size semantics.
    // This avoids misrouting to strlen() when variable annotations are absent
    // (e.g., sub = lst[a:b]; len(sub)).
    if (arg_expr.is_symbol())
    {
      symbolt *arg_symbol = converter_.find_symbol(
        to_symbol_expr(arg_expr).get_identifier().as_string());
      if (arg_symbol)
      {
        auto list_struct =
          try_get_pylist_struct_type(arg_symbol->get_type(), converter_.ns);
        if (list_struct)
          return list_length(
            arg_expr, *list_struct, converter_.in_contract_clause());
      }
    }

    if (arg_expr.type().is_signedbv() || arg_expr.type().is_unsignedbv())
      return len_of_bitvector_operand(call_["args"][0]);

    typet len_arg_type = converter_.ns.follow(arg_expr.type());
    if (len_arg_type.is_array() && len_arg_type.subtype() != char_type())
      return to_array_type(len_arg_type).size();

    // len() of a tuple-typed expression (e.g. an inline str.partition() result
    // that is not bound to a Name, so the __ESBMC_len_tuple routing above never
    // fires) is the number of components. Without this the call falls through
    // to strlen() and reports the wrong length (#5114).
    if (arg_expr.type().id() == "struct")
    {
      const struct_typet &struct_type = to_struct_type(arg_expr.type());
      if (struct_type.tag().as_string().find("tag-tuple") == 0)
        return from_integer(struct_type.components().size(), size_type());
    }

    exprt len_string_fast_path =
      converter_.get_string_handler().try_handle_len_string_fast_path(
        call_, arg_expr);
    if (!len_string_fast_path.is_nil())
      return len_string_fast_path;
  }

  if (function_id.get_function() == "__ESBMC_len_tuple")
  {
    const auto &arg = call_["args"][0];
    exprt obj_expr = converter_.get_expr(arg);

    if (obj_expr.type().id() == "struct")
    {
      const struct_typet &struct_type = to_struct_type(obj_expr.type());
      size_t tuple_len = struct_type.components().size();
      return from_integer(tuple_len, size_type());
    }

    // Fallback
    return from_integer(0, size_type());
  }

  if (function_id.get_function() == "__ESBMC_len_dict")
  {
    const auto &arg = call_["args"][0];
    exprt obj_expr = converter_.get_expr(arg);

    // Check actual type: could be dict or list (e.g., from d.keys())
    // If it's actually a list, call list_size directly
    auto list_struct =
      try_get_pylist_struct_type(obj_expr.type(), converter_.ns);
    if (list_struct)
    {
      // It's a list, not a dict: call list_size on it directly
      code_typet list_size_type;
      list_size_type.return_type() = size_type();
      code_typet::argumentt arg_type;
      arg_type.type() = pointer_typet(*list_struct);
      list_size_type.arguments().push_back(arg_type);

      symbol_exprt list_size_func("c:@F@__ESBMC_list_size", list_size_type);

      return build_call(list_size_func, size_type(), {obj_expr});
    }

    // It's genuinely a dict: get the keys member
    typet keys_type = pointer_typet(struct_typet());
    exprt keys_member = build_member(obj_expr, "keys", keys_type);

    // Create the list_get_size function symbol
    code_typet list_get_size_type;
    list_get_size_type.return_type() = size_type();
    code_typet::argumentt arg_type;
    arg_type.type() = keys_type;
    list_get_size_type.arguments().push_back(arg_type);

    symbol_exprt list_get_size_func(
      "c:@F@__ESBMC_list_size", list_get_size_type);

    // Create the function call
    return build_call(list_get_size_func, size_type(), {keys_member});
  }

  // Special handling for assume calls: convert to code_assume instead of
  // function call
  if (is_assume_call(function_id))
  {
    if (call_["args"].empty())
      throw std::runtime_error("__ESBMC_assume requires one boolean argument");

    exprt condition = converter_.get_expr(call_["args"][0]);
    if (!condition.type().is_bool())
      condition = build_typecast(condition, bool_type());

    // Create code_assume statement
    codet assume_code("assume");
    assume_code.copy_to_operands(condition);
    assume_code.location() = converter_.get_location_from_decl(call_);

    return assume_code;
  }

  // __ESBMC_assert(cond[, msg]) — same message lowering as native
  // `assert cond, msg` in converter_stmt.cpp.
  if (is_assert_call(function_id))
  {
    const auto &args = call_["args"];
    if (args.empty() || args.size() > 2)
      throw std::runtime_error(
        "__ESBMC_assert takes one or two arguments (cond[, msg])");

    exprt condition = converter_.get_expr(args[0]);
    if (!condition.type().is_bool())
      condition = build_typecast(condition, bool_type());

    code_assertt assert_code(condition);
    assert_code.location() = converter_.get_location_from_decl(call_);

    if (args.size() == 2)
    {
      const auto &msg_node = args[1];
      std::string msg;
      if (
        msg_node["_type"] == "Constant" && msg_node.contains("value") &&
        msg_node["value"].is_string())
        msg = msg_node["value"].get<std::string>();
      else if (msg_node["_type"] == "JoinedStr")
        msg = "<formatted string message>";

      if (!msg.empty())
        assert_code.location().comment(msg);
    }

    return assert_code;
  }

  // cover calls convert to code_assert statement
  // cover semantics: cover(cond) behaves as assert(!cond)
  // - failure (counterexample) means the condition is satisfiable
  // - success (proof) means the condition is not satisfiable
  if (is_cover_call(function_id))
  {
    if (call_["args"].empty())
      throw std::runtime_error("__ESBMC_cover requires one boolean argument");

    exprt condition = converter_.get_expr(call_["args"][0]);

    // Negate the condition: cover(cond) = assert(!cond)
    exprt negated_condition = gen_not(condition);

    // Create code_assert statement with cover property
    code_assertt cover_code(negated_condition);
    locationt loc = converter_.get_location_from_decl(call_);
    cover_code.location() = loc;

    return cover_code;
  }

  // __ESBMC_unreachable: emit a call to the C-style symbol so symex's
  // intrinsic handler in symex_main.cpp fires under
  // --enable-unreachability-intrinsic. Lowering matches the atomic_*
  // pattern below: register a void(void) symbol id and call it.
  if (is_unreachable_call(function_id))
  {
    if (!call_["args"].empty())
      throw std::runtime_error("__ESBMC_unreachable takes no arguments");

    code_typet fn_type;
    fn_type.return_type() = empty_typet();

    locationt location = converter_.get_location_from_decl(call_);
    converter_.ensure_void_void_intrinsic(kEsbmcUnreachable, location);

    code_function_callt call;
    call.function() = symbol_exprt("c:@F@" + kEsbmcUnreachable, fn_type);
    call.type() = empty_typet();
    call.location() = location;
    return call;
  }

  // __ESBMC_atomic_begin / __ESBMC_atomic_end: thin wrappers around the C
  // intrinsics with the same names. Emitting a call to a symbol whose
  // base name matches is enough; goto-symex lowers it to the atomic-section
  // builtins in src/goto-programs/builtin_functions.cpp.
  {
    const std::string &func_name = function_id.get_function();
    const bool is_atomic_begin = func_name == kEsbmcAtomicBegin;
    const bool is_atomic_end = func_name == kEsbmcAtomicEnd;
    if (is_atomic_begin || is_atomic_end)
    {
      if (!call_["args"].empty())
        throw std::runtime_error(func_name + " takes no arguments");

      code_typet atomic_type;
      atomic_type.return_type() = empty_typet();

      locationt location = converter_.get_location_from_decl(call_);
      converter_.ensure_void_void_intrinsic(func_name, location);
      // do_atomic_begin emits a call to __ESBMC_yield to force a context
      // switch before entering the atomic section, dereferencing the symbol
      // unconditionally; the Python frontend does not pull in the C
      // intrinsics header, so register the yield symbol here too. Only
      // atomic_begin needs it; atomic_end does not call yield.
      if (is_atomic_begin)
        converter_.ensure_void_void_intrinsic("__ESBMC_yield", location);

      code_function_callt atomic_call;
      atomic_call.function() = symbol_exprt("c:@F@" + func_name, atomic_type);
      atomic_call.type() = empty_typet();
      atomic_call.location() = location;
      return atomic_call;
    }
  }

  // __ESBMC_spawn_thread: the symex-level thread-spawn intrinsic used
  // by the Python frontend's threading.Thread lowering (see parser.py
  // lower_threading_thread_usage). Symex's intrinsic_spawn_thread
  // (builtin_functions/threads.cpp) requires its operand to be a
  // literal address_of(symbol_expr(trampoline_function)); the Python
  // converter receives a bare Name → symbol_exprt and wraps it here.
  {
    const std::string &func_name = function_id.get_function();
    if (func_name == kEsbmcSpawnThread)
    {
      auto &symbol_table = converter_.symbol_table();
      locationt location = converter_.get_location_from_decl(call_);

      if (call_["args"].size() != 1)
        throw std::runtime_error(func_name + " takes exactly one argument");
      exprt arg = converter_.get_expr(call_["args"][0]);
      if (arg.type().is_code())
        arg = build_address_of(arg);

      // intrinsic_spawn_thread requires a *literal* address_of(symbol); a
      // typecast around it trips its is_address_of2t assertion. The trampoline
      // returns None, whose GOTO return type is none_type() rather than void
      // (issue #5914), so derive the parameter type from the actual address
      // expression instead of forcing pointer-to-void() and typecasting.
      typet param_type = arg.type();

      code_typet fn_type;
      fn_type.return_type() = uint_type();
      fn_type.arguments().push_back(code_typet::argumentt(param_type));

      const std::string symbol_id = "c:@F@" + func_name;
      if (symbol_table.find_symbol(symbol_id.c_str()) == nullptr)
      {
        symbolt symbol = converter_.create_symbol(
          converter_.python_file(), func_name, symbol_id, location, fn_type);
        converter_.add_symbol(symbol);
      }

      code_function_callt call;
      call.function() = symbol_exprt(symbol_id, fn_type);
      call.type() = fn_type.return_type();
      call.location() = location;
      call.arguments().push_back(arg);
      return call;
    }
  }

  // __pyt_init_tid / __pyt_join / __pyt_terminate /
  // __ESBMC_pylock_block_and_check: C-side bookkeeping helpers for the Python
  // threading lowering (defined in pthread_lib.c, pulled into the GOTO program
  // via the python_c_models whitelist in cprover_library.cpp). Register the
  // matching C-style symbol id so the call resolves to the linked body rather
  // than a freshly-created Python-frontend symbol with no implementation.
  {
    const std::string &func_name = function_id.get_function();
    const bool is_init_tid = func_name == kPytInitTid;
    const bool is_join = func_name == kPytJoin;
    const bool is_terminate = func_name == kPytTerminate;
    const bool is_lock_block = func_name == kPyLockBlockAndCheck;
    const bool is_lock_release = func_name == kPyLockReleaseWaiters;
    if (
      is_init_tid || is_join || is_terminate || is_lock_block ||
      is_lock_release)
    {
      auto &symbol_table = converter_.symbol_table();
      locationt location = converter_.get_location_from_decl(call_);

      const bool takes_uint_arg = is_init_tid || is_join || is_lock_release;

      code_typet fn_type;
      fn_type.return_type() = empty_typet();
      if (takes_uint_arg)
        fn_type.arguments().push_back(code_typet::argumentt(uint_type()));

      const std::string symbol_id = "c:@F@" + func_name;
      if (symbol_table.find_symbol(symbol_id.c_str()) == nullptr)
      {
        symbolt symbol = converter_.create_symbol(
          converter_.python_file(), func_name, symbol_id, location, fn_type);
        converter_.add_symbol(symbol);
      }

      code_function_callt call;
      call.function() = symbol_exprt(symbol_id, fn_type);
      call.type() = empty_typet();
      call.location() = location;

      if (takes_uint_arg)
      {
        if (call_["args"].size() != 1)
          throw std::runtime_error(func_name + " takes exactly one argument");
        exprt arg = converter_.get_expr(call_["args"][0]);
        if (arg.type() != uint_type())
          arg = build_typecast(arg, uint_type());
        call.arguments().push_back(arg);
      }
      else if (!call_["args"].empty())
      {
        throw std::runtime_error(func_name + " takes no arguments");
      }

      return call;
    }
  }

  // Add len function to symbol table
  if (is_len_call(function_id))
  {
    const auto &symbol_table = converter_.symbol_table();
    const std::string &func_symbol_id = function_id.to_string();

    if (symbol_table.find_symbol(func_symbol_id.c_str()) == nullptr)
    {
      code_typet code_type;
      code_type.return_type() = long_long_int_type();
      code_type.arguments().push_back(pointer_typet(empty_typet()));

      const std::string &python_file = converter_.python_file();
      const std::string &func_name = function_id.get_function();
      locationt location = converter_.get_location_from_decl(call_);

      symbolt symbol = converter_.create_symbol(
        python_file, func_name, func_symbol_id, location, code_type);

      converter_.add_symbol(symbol);
    }
  }

  if (std::optional<exprt> intrinsic = build_contract_intrinsic(function_id))
    return *intrinsic;

  // Handle NumPy functions
  if (is_numpy_call(function_id))
  {
    // Adjust the function ID when reusing functions from the C models
    if (type_utils::is_c_model_func(function_id.get_function()))
    {
      function_id.set_prefix("c:");
      function_id.set_filename("");
    }

    return numpy_call_expr::build_result(function_id, call_, converter_);
  }

  const bool saved = converter_.enter_contract_clause(
    needs_bool_intrinsic_symbol(function_id.get_function()));
  const size_t planted = converter_.current_block_statement_count();
  function_call_expr call_expr(function_id, call_, converter_);
  exprt result = call_expr.get();
  converter_.restore_contract_clause(saved);
  check_clause_planted_nothing(function_id.get_function(), planted);
  return result;
}
