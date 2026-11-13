#include "python_dict_internal.h"

using namespace python_expr;

/// A subscript whose base can be converted again to retype the read. A call
/// to a module-level function was typed by its first conversion, and
/// converting it again would call it twice.
bool python_dict_handler::is_retypable_subscript(const nlohmann::json &node)
{
  return type_utils::is_dict_subscript(node) &&
         !converter_.call_dict_value_type(node["value"]).has_value();
}

/// A dict read whose value type is still unknown. A str value is already read
/// as its char pointer.
static bool is_untyped_value_ptr(const typet &t)
{
  return t.is_pointer() && !type_utils::is_char_storage(t.subtype());
}

void python_dict_handler::resolve_dict_subscript_types(
  const nlohmann::json &left,
  const nlohmann::json &right,
  exprt &lhs,
  exprt &rhs)
{
  bool lhs_is_dict_subscript = is_retypable_subscript(left);
  bool rhs_is_dict_subscript = is_retypable_subscript(right);

  bool lhs_is_ptr = lhs.type().is_pointer();
  bool rhs_is_ptr = rhs.type().is_pointer();

  auto is_primitive_type = [](const typet &t) {
    return t.is_signedbv() || t.is_unsignedbv() || t.is_bool() ||
           t.is_floatbv();
  };

  bool lhs_is_primitive = is_primitive_type(lhs.type());
  bool rhs_is_primitive = is_primitive_type(rhs.type());

  // Case 1: LHS is dict subscript (returning pointer) and RHS is primitive
  if (lhs_is_dict_subscript && lhs_is_ptr && rhs_is_primitive)
  {
    exprt dict_expr = converter_.get_expr(left["value"]);
    if (dict_expr.type().is_struct() && is_dict_type(dict_expr.type()))
    {
      lhs = handle_dict_subscript(dict_expr, left["slice"], rhs.type());
      // Dereference the pointer to get the actual value
      if (lhs.type().is_pointer())
        lhs = build_dereference(lhs, lhs.type().subtype());
    }
  }

  // Case 2: RHS is dict subscript (returning pointer) and LHS is primitive
  if (rhs_is_dict_subscript && rhs_is_ptr && lhs_is_primitive)
  {
    exprt dict_expr = converter_.get_expr(right["value"]);
    if (dict_expr.type().is_struct() && is_dict_type(dict_expr.type()))
    {
      rhs = handle_dict_subscript(dict_expr, right["slice"], lhs.type());
      // Dereference the pointer to get the actual value
      if (rhs.type().is_pointer())
        rhs = build_dereference(rhs, rhs.type().subtype());
    }
  }

  // Case 3: Both sides are dict subscripts (returning pointers)
  // Default to long_int_type for dict-to-dict comparisons
  if (
    lhs_is_dict_subscript && rhs_is_dict_subscript &&
    is_untyped_value_ptr(lhs.type()) && is_untyped_value_ptr(rhs.type()))
  {
    typet default_type = long_int_type();

    exprt lhs_dict = converter_.get_expr(left["value"]);
    if (lhs_dict.type().is_struct() && is_dict_type(lhs_dict.type()))
    {
      lhs = handle_dict_subscript(lhs_dict, left["slice"], default_type);
      // Dereference the pointer to get the actual value
      if (lhs.type().is_pointer())
        lhs = build_dereference(lhs, lhs.type().subtype());
    }

    exprt rhs_dict = converter_.get_expr(right["value"]);
    if (rhs_dict.type().is_struct() && is_dict_type(rhs_dict.type()))
    {
      rhs = handle_dict_subscript(rhs_dict, right["slice"], default_type);
      // Dereference the pointer to get the actual value
      if (rhs.type().is_pointer())
      {
        rhs = build_dereference(rhs, rhs.type().subtype());
      }
    }
  }
}

typet python_dict_handler::get_dict_value_type_from_annotation(
  const nlohmann::json &annotation_node)
{
  // Get the slice which contains the key and value types
  if (!annotation_node.contains("slice"))
    return empty_typet();

  const auto &slice = annotation_node["slice"];

  // For dict[K, V], slice is a Tuple with two elements
  if (
    slice.contains("_type") && slice["_type"] == "Tuple" &&
    slice.contains("elts") && slice["elts"].size() >= 2)
  {
    // Return the value type (second element of the tuple)
    const auto &value_type_node = slice["elts"][1];
    return converter_.get_type_from_annotation(
      value_type_node, annotation_node);
  }

  return empty_typet();
}

/// Whether `node` builds an empty dict: `{}` or `dict()`.
static bool is_empty_dict_init(const nlohmann::json &node)
{
  if (!node.is_object())
    return false;
  const std::string type = node.value("_type", std::string());
  if (type == "Dict")
    return !node.contains("values") || !node["values"].is_array() ||
           node["values"].empty();
  return type == "Call" && node.contains("func") && node["func"].is_object() &&
         node["func"].value("id", std::string()) == "dict" &&
         node.value("args", nlohmann::json::array()).empty() &&
         node.value("keywords", nlohmann::json::array()).empty();
}

/// Whether `node` is the Name `name`.
static bool is_name(const nlohmann::json &node, const std::string &name)
{
  return node.is_object() && node.value("_type", "") == "Name" &&
         node.value("id", "") == name;
}

/// Whether `node` reads a value out of the dict `name`: `name[k]`,
/// `name.pop(k)` or `name.get(k)`.
static bool reads_from_dict(const nlohmann::json &node, const std::string &name)
{
  if (!node.is_object())
    return false;
  const std::string kind = node.value("_type", "");
  if (kind == "Subscript")
    return is_name(node["value"], name) &&
           node["slice"].value("_type", "") != "Slice";
  const std::string method =
    kind == "Call" ? node["func"].value("attr", "") : std::string();
  return (method == "pop" || method == "get") &&
         node["func"].value("_type", "") == "Attribute" &&
         is_name(node["func"]["value"], name) && node["args"].size() == 1 &&
         node["keywords"].empty();
}

/// Whether `node` is `name.popitem()`, which removes an entry and stores none.
static bool pops_item_from(const nlohmann::json &node, const std::string &name)
{
  return node.is_object() && node.value("_type", "") == "Call" &&
         node["func"].value("_type", "") == "Attribute" &&
         node["func"].value("attr", "") == "popitem" &&
         is_name(node["func"]["value"], name) && node["args"].empty();
}

/// The Names `name` under `node`, and how many of them, outside a def, class
/// or lambda, are the dict operand of a subscript or of a `pop` call. The
/// scan of stores skips those scopes.
static void count_dict_uses(
  const nlohmann::json &node,
  const std::string &name,
  bool nested,
  std::size_t &all,
  std::size_t &indexed)
{
  if (!node.is_structured())
    return;
  const std::string kind =
    node.is_object() ? node.value("_type", "") : std::string();
  if (is_name(node, name))
    ++all;
  if (
    !nested && (kind == "Subscript"
                  ? is_name(node["value"], name)
                  : reads_from_dict(node, name) || pops_item_from(node, name)))
    ++indexed;
  nested = nested || kind == "FunctionDef" || kind == "AsyncFunctionDef" ||
           kind == "ClassDef" || kind == "Lambda";
  for (const auto &child : node)
    count_dict_uses(child, name, nested, all, indexed);
}

/// Whether `value` names a variable bound only once, by a read out of the dict
/// `name`, itself bound only once and otherwise only subscripted or popped:
/// storing it back into `name` adds no new kind of value.
static bool reads_back_from(
  const nlohmann::json &ast,
  const nlohmann::json &value,
  const std::string &name)
{
  if (value.value("_type", "") != "Name")
    return false;
  const std::string var = value.value("id", "");
  if (
    json_utils::count_name_bindings(ast, var) != 1 ||
    json_utils::count_name_bindings(ast, name) != 1)
    return false;
  const nlohmann::json decl = json_utils::find_var_decl(var, "", ast);
  const nlohmann::json *read = json_utils::plain_assigned_value(decl, var);
  if (!read || !reads_from_dict(*read, name))
    return false;
  // Any other use, such as `name.update(...)` or an alias, can store a value
  // of another kind, which `var` may then read back.
  std::size_t all = 0, indexed = 0;
  count_dict_uses(ast, name, false, all, indexed);
  return all == indexed + 1;
}

/// The kind of the stored expression `rhs`: that of a literal, "?" otherwise.
/// A key distinguishes int from float.
static std::string stored_expr_kind(const nlohmann::json &rhs, bool by_key)
{
  const std::string type = rhs.value("_type", "");
  if (type == "List")
    return "list";
  if (type != "Constant")
    return "?";
  const auto &literal = rhs["value"];
  if (by_key && literal.is_number_float())
    return "float";
  if (by_key && literal.is_number_integer())
    return "int";
  return literal.type_name();
}

/// The node whose bindings of `var` are the ones a read in the scope `func`
/// sees: `func` itself when it binds `var`, else the whole module (#10489).
static nlohmann::json binding_scope(
  const nlohmann::json &ast,
  const std::string &var,
  const std::string &func)
{
  const nlohmann::json function = json_utils::find_function_by_path(
    ast, json_utils::split_function_path(func));
  return function.is_object() && json_utils::count_name_bindings(function, var)
           ? function
           : ast;
}

/// The annotation that types the value `rhs` in the scope `func`: that of a
/// variable bound once, or the return annotation of a module-level function.
/// Null for any other value (#10432).
static nlohmann::json stored_value_annotation(
  const nlohmann::json &ast,
  const nlohmann::json &rhs,
  const std::string &func)
{
  const std::string type = rhs.value("_type", "");
  const std::string var = rhs.value("id", "");
  if (
    type == "Name" &&
    json_utils::count_name_bindings(binding_scope(ast, var, func), var) == 1)
  {
    const nlohmann::json decl = json_utils::find_var_decl(var, func, ast);
    return decl.is_object() ? decl.value("annotation", nlohmann::json())
                            : nlohmann::json();
  }
  if (type != "Call" || rhs["func"].value("_type", "") != "Name")
    return nlohmann::json();
  const nlohmann::json def =
    json_utils::try_find_function(ast["body"], rhs["func"].value("id", ""));
  return def.is_object() ? def.value("returns", nlohmann::json())
                         : nlohmann::json();
}

/// The scalar type the annotation `node` names: `int`, `float`, `bool` or
/// `str`, else empty.
static typet
scalar_annotation_type(const nlohmann::json &node, type_handler &handler)
{
  const std::string id = node.is_object() && node.value("_type", "") == "Name"
                           ? node.value("id", "")
                           : "";
  if (id == "bool")
    return bool_type();
  if (id == "int" || id == "float")
    return handler.get_typet(id, 0);
  if (id == "str")
    return gen_pointer_type(char_type());
  return empty_typet();
}

/// The literal kind (see stored_expr_kind) of a value of the scalar type
/// `node` annotates, "?" for any other annotation.
static std::string annotated_kind(const nlohmann::json &node)
{
  static const std::map<std::string, std::string> kinds = {
    {"bool", "boolean"},
    {"int", "number"},
    {"float", "number"},
    {"str", "string"}};
  const auto it = kinds.find(
    node.is_object() && node.value("_type", "") == "Name" ? node.value("id", "")
                                                          : "");
  return it == kinds.end() ? "?" : it->second;
}

/// The kind of the stored value `rhs` that is not a literal: that of the
/// scalar type it is annotated with (see annotated_kind), or the class that a
/// constructor call builds.
static std::string
stored_value_kind(const nlohmann::json &ast, const nlohmann::json &rhs)
{
  const std::string callee =
    rhs.value("_type", "") == "Call" && rhs["func"].value("_type", "") == "Name"
      ? rhs["func"].value("id", "")
      : "";
  if (!callee.empty() && json_utils::is_class(callee, ast))
    return "class " + callee;
  return annotated_kind(stored_value_annotation(ast, rhs, ""));
}

/// The kind of the literal that the statement `stmt` stores into the dict
/// `name` by subscript, "?" for any other value, and "" if it stores nothing.
/// With @p by_key it is the kind of the stored key instead.
static std::string stored_literal_kind(
  const nlohmann::json &ast,
  const nlohmann::json &stmt,
  const std::string &name,
  bool by_key = false)
{
  if (
    stmt.value("_type", "") != "Assign" || !stmt.contains("targets") ||
    stmt["targets"].empty())
    return "";
  const auto &target = stmt["targets"][0];
  if (
    !target.is_object() || target.value("_type", "") != "Subscript" ||
    !target["value"].is_object() || target["value"].value("id", "") != name)
    return "";
  const auto &rhs = by_key ? target["slice"] : stmt["value"];
  if (!by_key && reads_back_from(ast, rhs, name))
    return "";
  const std::string kind = stored_expr_kind(rhs, by_key);
  return kind == "?" && !by_key ? stored_value_kind(ast, rhs) : kind;
}

/// Whether every statement of `body` that stores into the dict `name` by
/// subscript stores a literal of one kind, so the last store names the type.
static bool stores_one_literal_kind(
  const nlohmann::json &ast,
  const nlohmann::json &body,
  const std::string &name,
  std::string &kind,
  bool by_key = false)
{
  if (!body.is_array())
    return true;
  for (const auto &stmt : body)
  {
    if (!stmt.is_object())
      continue;
    const std::string type = stmt.value("_type", "");
    if (type == "FunctionDef" || type == "ClassDef" || type == "Lambda")
      continue;
    const std::string stored = stored_literal_kind(ast, stmt, name, by_key);
    if (stored == "?" || (!stored.empty() && !kind.empty() && stored != kind))
      return false;
    if (!stored.empty())
      kind = stored;
    for (const char *key : {"body", "orelse", "finalbody", "handlers"})
      if (
        stmt.contains(key) &&
        !stores_one_literal_kind(ast, stmt[key], name, kind, by_key))
        return false;
  }
  return true;
}

/// The value type that storing the literal `rhs` (`d[k] = 5`, `d[k] = "x"`,
/// `d[k] = []`, ...) gives a dict, or empty for any other value.
static typet
stored_literal_type(const nlohmann::json &rhs, type_handler &handler)
{
  if (rhs.value("_type", std::string()) == "List")
    return handler.get_list_type();
  const nlohmann::json lit = rhs.value("value", nlohmann::json());
  if (lit.is_boolean())
    return bool_type();
  if (lit.is_number_integer() || lit.is_number_unsigned())
    return handler.get_typet("int", 0);
  if (lit.is_number_float())
    return handler.get_typet("float", 0);
  if (lit.is_string())
    return gen_pointer_type(char_type());
  return empty_typet();
}

/// The value type that storing `rhs` in the scope `func` gives a dict: that
/// of a literal (a list only when @p only_lists), or the scalar type a
/// variable or a call is annotated with. Empty for any other value.
static typet stored_value_type(
  const nlohmann::json &ast,
  const nlohmann::json &rhs,
  const std::string &func,
  bool only_lists,
  type_handler &handler)
{
  const std::string kind = rhs.value("_type", "");
  if (kind == "Constant" || (kind == "List" && only_lists))
    return stored_literal_type(rhs, handler);
  return scalar_annotation_type(
    stored_value_annotation(ast, rhs, func), handler);
}

typet python_dict_handler::resolve_stored_key_type(const exprt &dict_expr)
{
  const symbolt *sym = dict_expr.is_symbol()
                         ? symbol_table_.find_symbol(dict_expr.identifier())
                         : nullptr;
  if (!sym || !converter_.get_current_func_name().empty())
    return empty_typet();
  const std::string name = sym->name.as_string();
  const nlohmann::json decl = json_utils::find_var_decl(
    name, converter_.get_current_func_name(), converter_.get_ast_json());
  const nlohmann::json &ast = converter_.get_ast_json();
  std::string kind;
  if (
    decl.empty() || !decl.contains("value") ||
    !is_empty_dict_init(decl["value"]) ||
    !stores_one_literal_kind(ast, ast["body"], name, kind, true))
    return empty_typet();
  if (kind == "boolean")
    return bool_type();
  if (kind == "int")
    return type_handler_.get_typet("int", 0);
  if (kind == "float")
    return type_handler_.get_typet("float", 0);
  return empty_typet();
}

typet python_dict_handler::key_type_or_default(
  const typet &key_type,
  const exprt &dict_expr)
{
  typet key = key_type;
  if (key.is_nil() || key.is_empty())
    key = resolve_stored_key_type(dict_expr);
  if (key.is_nil() || key.is_empty())
    key = gen_pointer_type(char_type());
  return key;
}

typet python_dict_handler::resolve_expected_type_for_dict_subscript(
  const exprt &dict_expr)
{
  if (dict_expr.id() == "member")
  {
    const auto &member = to_member_expr(dict_expr);
    typet class_type = member.struct_op().type();
    namespacet ns(symbol_table_);

    if (class_type.is_pointer())
      class_type = ns.follow(class_type.subtype());
    else
      class_type = ns.follow(class_type);

    assert(class_type.is_struct());

    // For self.data[key], look up the dict value type from the class
    // annotation before falling back to the usual symbol-based path.
    const std::string class_name = converter_.extract_class_name_from_tag(
      to_struct_type(class_type).tag().as_string());
    const nlohmann::json class_node =
      json_utils::find_class(converter_.get_ast_json()["body"], class_name);

    if (!class_node.empty() && class_node.contains("body"))
    {
      for (const auto &class_elem : class_node["body"])
      {
        if (
          class_elem.contains("_type") &&
          class_elem["_type"] == "FunctionDef" && class_elem.contains("name") &&
          class_elem["name"] == "__init__" && class_elem.contains("body"))
        {
          for (const auto &stmt : class_elem["body"])
          {
            if (
              stmt.contains("_type") && stmt["_type"] == "AnnAssign" &&
              stmt.contains("target") && stmt["target"].is_object() &&
              stmt["target"].contains("_type") &&
              stmt["target"]["_type"] == "Attribute" &&
              stmt["target"].contains("value") &&
              stmt["target"]["value"].is_object() &&
              stmt["target"]["value"].contains("id") &&
              stmt["target"]["value"]["id"] == "self" &&
              stmt["target"].contains("attr") &&
              stmt["target"]["attr"].get<std::string>() ==
                member.get_component_name().as_string() &&
              stmt.contains("annotation"))
            {
              typet result =
                get_dict_value_type_from_annotation(stmt["annotation"]);
              if (!result.is_nil() && !result.is_empty())
                return result;
            }
          }
        }
      }
    }
  }

  // Only works if dict_expr is a symbol (variable reference)
  if (!dict_expr.is_symbol())
    return empty_typet();

  const symbolt *sym = symbol_table_.find_symbol(dict_expr.identifier());
  if (!sym)
    return empty_typet();

  // Look up the variable's declaration in the AST to get its annotation
  std::string var_name = sym->name.as_string();
  nlohmann::json var_decl = json_utils::find_var_decl(
    var_name, converter_.get_current_func_name(), converter_.get_ast_json());

  if (var_decl.empty())
    return empty_typet();

  // No annotation or bare `dict`:
  // peek at the literal's first value to infer the value type.
  const bool no_annotation =
    !var_decl.contains("annotation") || var_decl["annotation"].is_null();
  const bool bare_dict_annotation =
    !no_annotation && var_decl["annotation"].is_object() &&
    var_decl["annotation"].value("_type", std::string()) == "Name" &&
    var_decl["annotation"].value("id", std::string()) == "dict" &&
    !var_decl["annotation"].contains("slice");
  if (no_annotation || bare_dict_annotation)
  {
    typet recorded = recorded_tuple_value_type(dict_expr);
    if (!recorded.is_nil() && !recorded.is_empty())
      return recorded;

    if (
      var_decl.contains("value") && var_decl["value"].is_object() &&
      var_decl["value"].value("_type", std::string()) == "Dict" &&
      var_decl["value"].contains("values") &&
      var_decl["value"]["values"].is_array() &&
      !var_decl["value"]["values"].empty())
    {
      const std::string kind =
        var_decl["value"]["values"][0].value("_type", std::string());
      if (kind == "List")
        return type_handler_.get_list_type();
      if (kind == "Dict")
        return get_dict_struct_type();
    }

    // Dict comprehension `d = {k: v for ...}`: the variable carries no
    // annotation and its AST value is a DictComp, so the Dict-literal peek
    // above does not fire. Infer the subscript value type from the
    // comprehension's value expression. Without this the read falls through
    // to the char* default and returns the raw PyObject value pointer rather
    // than the stored scalar — a wrong verdict on `d[k]` reads, e.g. inside a
    // function body where no target type flows in (#5222).
    if (
      var_decl.contains("value") && var_decl["value"].is_object() &&
      var_decl["value"].value("_type", std::string()) == "DictComp" &&
      var_decl["value"].contains("value") &&
      var_decl["value"]["value"].is_object())
    {
      const auto &val_node = var_decl["value"]["value"];
      const std::string val_kind = val_node.value("_type", std::string());
      if (val_kind == "List")
        return type_handler_.get_list_type();
      if (val_kind == "Dict" || val_kind == "DictComp")
        return get_dict_struct_type();
      // Scalar literal value: classify exactly as the value is stored by the
      // comprehension (Python int -> int, float -> float, bool -> bool). Only
      // numeric/boolean literal constants are inferred; string, bytes and any
      // other value expression are left to the existing path, which already
      // resolves them to the char* default (unchanged behaviour).
      if (val_kind == "Constant" && val_node.contains("value"))
      {
        const auto &lit = val_node["value"];
        if (lit.is_boolean())
          return bool_type();
        if (lit.is_number_integer() || lit.is_number_unsigned())
          return type_handler_.get_typet("int", 0);
        if (lit.is_number_float())
          return type_handler_.get_typet("float", 0);
      }
    }
    // Empty literal `d = {}`:
    // scan the enclosing function's top-level statements and pick the most
    // recent `d[k] = Klass(...)` assignment as the subscript value type.
    // A later `d = {}` re-init resets the candidate so the next assignment
    // dominates — keeps the inferred type aligned with the live dict.
    // Without it, d[k] reads on an empty literal default to char*.
    if (var_decl.contains("value") && is_empty_dict_init(var_decl["value"]))
    {
      std::vector<std::string> fn_path =
        json_utils::split_function_path(converter_.get_current_func_name());
      std::string store_kind;
      if (
        !fn_path.empty() || stores_one_literal_kind(
                              converter_.get_ast_json(),
                              converter_.get_ast_json()["body"],
                              var_name,
                              store_kind))
      {
        const nlohmann::json function =
          fn_path.empty() ? nlohmann::json()
                          : json_utils::find_function_by_path(
                              converter_.get_ast_json(), fn_path);
        const nlohmann::json &func_node =
          fn_path.empty() ? converter_.get_ast_json() : function;
        if (
          !func_node.empty() && func_node.contains("body") &&
          func_node["body"].is_array())
        {
          typet candidate;
          bool have_candidate = false;
          // A list store names the type only when every store is a list: the
          // last store need not be the one a later read sees.
          std::string body_kind;
          const bool only_lists = stores_one_literal_kind(
                                    converter_.get_ast_json(),
                                    func_node["body"],
                                    var_name,
                                    body_kind) &&
                                  body_kind == "list";
          // Walk statements in document order, descending into For/While/If/
          // Try/With nested bodies — the d[k] = factory() assignments inserted
          // by the defaultdict preprocessor (see
          // preprocessor/loop_mixin.py:_lower_defaultdict_reads_in_expr) sit
          // inside the containing statement, which can be arbitrarily deep
          // inside nested loops (e.g. quixbugs/shortest_path_lengths).
          std::function<void(const nlohmann::json &)> visit_stmt;
          auto visit_body = [&](const nlohmann::json &body) {
            if (!body.is_array())
              return;
            for (const auto &s : body)
              visit_stmt(s);
          };
          visit_stmt = [&](const nlohmann::json &stmt) {
            if (!stmt.is_object() || !stmt.contains("_type"))
              return;
            const std::string &kind = stmt["_type"].get<std::string>();
            if (kind == "For" || kind == "AsyncFor" || kind == "While")
            {
              if (stmt.contains("body"))
                visit_body(stmt["body"]);
              if (stmt.contains("orelse"))
                visit_body(stmt["orelse"]);
              return;
            }
            if (kind == "If")
            {
              if (stmt.contains("body"))
                visit_body(stmt["body"]);
              if (stmt.contains("orelse"))
                visit_body(stmt["orelse"]);
              return;
            }
            if (kind == "With" || kind == "AsyncWith")
            {
              if (stmt.contains("body"))
                visit_body(stmt["body"]);
              return;
            }
            if (kind == "Try" || kind == "TryStar")
            {
              if (stmt.contains("body"))
                visit_body(stmt["body"]);
              if (stmt.contains("orelse"))
                visit_body(stmt["orelse"]);
              if (stmt.contains("finalbody"))
                visit_body(stmt["finalbody"]);
              if (stmt.contains("handlers") && stmt["handlers"].is_array())
                for (const auto &h : stmt["handlers"])
                  if (h.is_object() && h.contains("body"))
                    visit_body(h["body"]);
              return;
            }
            // Don't descend into nested FunctionDef/Lambda/ClassDef — their
            // assignments belong to a different scope.
            if (kind != "Assign")
              return;
            if (
              !stmt.contains("targets") || !stmt["targets"].is_array() ||
              stmt["targets"].empty())
              return;
            const auto &tgt = stmt["targets"][0];

            // `d = {}` re-init clears the candidate;
            // the next subscript assignment dominates.
            if (
              tgt.is_object() && tgt.value("_type", std::string()) == "Name" &&
              tgt.value("id", std::string()) == var_name &&
              stmt.contains("value") && is_empty_dict_init(stmt["value"]))
            {
              have_candidate = false;
              return;
            }

            if (
              !tgt.is_object() ||
              tgt.value("_type", std::string()) != "Subscript" ||
              !tgt.contains("value") || !tgt["value"].is_object() ||
              tgt["value"].value("_type", std::string()) != "Name" ||
              tgt["value"].value("id", std::string()) != var_name ||
              !stmt.contains("value") || !stmt["value"].is_object())
              return;
            const auto &rhs = stmt["value"];

            const typet stored = stored_value_type(
              converter_.get_ast_json(),
              rhs,
              converter_.get_current_func_name(),
              only_lists,
              type_handler_);
            if (!stored.is_empty())
            {
              candidate = stored;
              have_candidate = true;
              return;
            }

            if (
              rhs.value("_type", std::string()) == "Call" &&
              rhs.contains("func") && rhs["func"].is_object())
            {
              const auto &func = rhs["func"];

              // Lambda factory: `defaultdict(lambda: float('inf'))` is
              // lowered by the preprocessor to `d[k] = (<lambda>)()`. Inspect
              // the lambda body to determine the value type.
              if (
                func.value("_type", std::string()) == "Lambda" &&
                func.contains("body") && func["body"].is_object())
              {
                const auto &body = func["body"];
                if (
                  body.value("_type", std::string()) == "Constant" &&
                  body.contains("value"))
                {
                  const auto &lit = body["value"];
                  if (lit.is_boolean())
                  {
                    candidate = bool_type();
                    have_candidate = true;
                  }
                  else if (lit.is_number_integer() || lit.is_number_unsigned())
                  {
                    candidate = type_handler_.get_typet("int", 0);
                    have_candidate = true;
                  }
                  else if (lit.is_number_float())
                  {
                    candidate = type_handler_.get_typet("float", 0);
                    have_candidate = true;
                  }
                }
                else if (
                  body.value("_type", std::string()) == "Call" &&
                  body.contains("func") && body["func"].is_object() &&
                  body["func"].value("_type", std::string()) == "Name" &&
                  body["func"].contains("id"))
                {
                  const std::string body_callee =
                    body["func"]["id"].get<std::string>();
                  // `lambda: float('inf')`, `lambda: int()`, ...
                  if (body_callee == "float")
                  {
                    candidate = type_handler_.get_typet("float", 0);
                    have_candidate = true;
                  }
                  else if (body_callee == "int")
                  {
                    candidate = type_handler_.get_typet("int", 0);
                    have_candidate = true;
                  }
                  else if (body_callee == "bool")
                  {
                    candidate = bool_type();
                    have_candidate = true;
                  }
                  else if (body_callee == "str")
                  {
                    candidate = gen_pointer_type(char_type());
                    have_candidate = true;
                  }
                }
                return;
              }

              if (
                !func.contains("_type") || func["_type"] != "Name" ||
                !func.contains("id"))
                return;

              const std::string callee = func["id"].get<std::string>();

              // Built-in type constructors: `int()`, `float()`, `bool()`,
              // `str()` — emitted by the preprocessor when lowering
              // `defaultdict(<builtin>)` reads (see
              // preprocessor/loop_mixin.py:_make_defaultdict_missing_check).
              if (
                callee == "int" || callee == "float" || callee == "bool" ||
                callee == "str")
              {
                if (callee == "str")
                  candidate = gen_pointer_type(char_type());
                else
                  candidate = type_handler_.get_typet(callee, 0);
                have_candidate = true;
                return;
              }

              if (json_utils::is_class(callee, converter_.get_ast_json()))
              {
                candidate = symbol_typet("tag-" + callee);
                have_candidate = true;
              }
            }
          };
          visit_body(func_node["body"]);
          if (have_candidate)
            return candidate;
        }
      }
    }
    if (no_annotation)
      return empty_typet();
  }

  // Check if the annotation is just a simple type (e.g., "dict")
  // If so, try to get the full type from the RHS (function call)
  if (
    var_decl["annotation"]["_type"] == "Name" &&
    var_decl["annotation"]["id"] == "dict" && var_decl.contains("value") &&
    var_decl["value"]["_type"] == "Call")
  {
    const auto &call_node = var_decl["value"];
    if (call_node["func"]["_type"] == "Name")
    {
      std::string func_name = call_node["func"]["id"].get<std::string>();

      // Find the function definition to get its return type annotation
      nlohmann::json func_def = json_utils::try_find_function(
        converter_.get_ast_json()["body"], func_name);

      if (
        !func_def.empty() && func_def.contains("returns") &&
        !func_def["returns"].is_null())
      {
        // Extract the value type from the function's return annotation
        typet result = get_dict_value_type_from_annotation(func_def["returns"]);
        if (!result.is_nil())
          return result;
      }
    }
  }

  // Extract the value type from the dict annotation
  typet result = get_dict_value_type_from_annotation(var_decl["annotation"]);
  if (result.is_nil() || result.is_empty())
  {
    typet recorded = recorded_tuple_value_type(dict_expr);
    if (!recorded.is_nil() && !recorded.is_empty())
      return recorded;
  }
  return result;
}

/// The type every value of a literal dict was stored with, recorded at
/// construction; nil when the dict was not built from a literal or its values
/// differ in type. A float is left out: a value the literal's AST does not
/// show to be a float is stored by bit pattern, not in the float buffer a
/// float list read uses.
typet python_dict_handler::recorded_value_type(const exprt &dict_expr) const
{
  const std::string &vals_id =
    get_internal_list_id(dict_expr.identifier().as_string(), false);
  if (vals_id.empty())
    return typet();
  const typet uniform =
    converter_.get_element_type_registry().uniform_element_type(
      vals_id, type_slot::dict_value_types);
  return uniform.is_floatbv() ? typet() : uniform;
}

/// Homogeneous tuple values: the struct type recorded at construction
/// (tagged and string-padded), which neither an annotation nor an AST
/// literal peek reproduces. Empty when the dict was not built from a
/// literal or its value types are not a single tuple type.
typet python_dict_handler::recorded_tuple_value_type(
  const exprt &dict_expr) const
{
  const typet uniform = recorded_value_type(dict_expr);
  return converter_.get_tuple_handler().is_tuple_type(uniform)
           ? uniform
           : static_cast<typet>(empty_typet());
}

typet python_dict_handler::uniform_list_dict_value_type(
  const std::string &list_id) const
{
  const element_type_registry &registry =
    converter_.get_element_type_registry();
  const element_type_registry::entries *dicts = registry.find(list_id);
  if (!dicts)
    return empty_typet();

  typet value_type;
  for (const auto &dict : *dicts)
  {
    const std::string &vals_id = get_internal_list_id(dict.first, false);
    const typet uniform =
      vals_id.empty()
        ? typet()
        : registry.uniform_element_type(vals_id, type_slot::dict_value_types);
    if (uniform == typet() || (value_type != typet() && uniform != value_type))
      return empty_typet();
    value_type = uniform;
  }
  // Aggregates share the empty type name, so their type_id cannot be checked.
  return type_handler_.type_to_string(value_type).empty()
           ? static_cast<typet>(empty_typet())
           : value_type;
}

typet python_dict_handler::get_dict_key_type_from_annotation(
  const nlohmann::json &annotation_node)
{
  if (!annotation_node.contains("slice"))
    return empty_typet();

  const auto &slice = annotation_node["slice"];
  if (
    slice.contains("_type") && slice["_type"] == "Tuple" &&
    slice.contains("elts") && slice["elts"].size() >= 2)
  {
    return converter_.get_type_from_annotation(
      slice["elts"][0], annotation_node);
  }

  return empty_typet();
}
