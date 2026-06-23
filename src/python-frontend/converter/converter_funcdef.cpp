#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/function_call/expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_annotation.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_lambda.h>
#include <python-frontend/python_list.h>
#include <python-frontend/python_typechecking.h>
#include <util/encoding.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/python_types.h>
#include <util/std_code.h>
#include <util/symbolic_types.h>

#include <functional>
#include <set>

using namespace json_utils;

size_t python_converter::get_type_size(const nlohmann::json &ast_node)
{
  size_t type_size = 0;

  // Handle lambda functions - they don't have a meaningful size
  if (
    ast_node.contains("value") && ast_node["value"].contains("_type") &&
    ast_node["value"]["_type"] == "Lambda")
    return 0;

  if (ast_node.contains("value") && ast_node["value"].contains("value"))
  {
    // Handle bytes literals
    if (
      ast_node.contains("annotation") &&
      ast_node["annotation"].contains("id") &&
      ast_node["annotation"]["id"] == "bytes")
    {
      if (ast_node["value"].contains("encoded_bytes"))
      {
        const std::string &str =
          ast_node["value"]["encoded_bytes"].get<std::string>();
        std::vector<uint8_t> decoded = base64_decode(str);
        type_size = decoded.size();
      }
      else if (ast_node["value"]["value"].is_string())
      {
        // Direct bytes literal such as b'A'
        type_size = ast_node["value"]["value"].get<std::string>().size();
      }
    }
    else if (ast_node["value"]["value"].is_string())
      type_size = ast_node["value"]["value"].get<std::string>().size();
  }
  else if (
    ast_node["value"].contains("args") &&
    ast_node["value"]["args"].is_array() &&
    ast_node["value"]["args"].size() > 0 &&
    ast_node["value"]["args"][0].contains("value") &&
    ast_node["value"]["args"][0]["value"].is_string())
  {
    type_size = ast_node["value"]["args"][0]["value"].get<std::string>().size();
  }
  else if (
    ast_node["value"].contains("_type") && ast_node["value"]["_type"] == "List")
  {
    type_size = ast_node["value"]["elts"].size();
  }
  // Handle cases where size cannot be determined from AST structure
  else if (
    ast_node["value"].contains("value") &&
    ast_node["value"]["value"].is_string())
  {
    // Fallback for direct string values
    type_size = ast_node["value"]["value"].get<std::string>().size();
  }

  return type_size;
}

symbolt python_converter::create_return_temp_variable(
  const typet &return_type,
  const locationt &location,
  const std::string &func_name)
{
  static int temp_counter = 0;
  temp_counter++;

  symbol_id temp_sid = create_symbol_id();
  std::string temp_name =
    "return_value$_" + func_name + "$" + std::to_string(temp_counter);
  temp_sid.set_object(temp_name);

  symbolt temp_symbol;
  temp_symbol.id = temp_sid.to_string();
  temp_symbol.name = temp_sid.to_string();
  temp_symbol.set_type(return_type);
  temp_symbol.lvalue = true;
  temp_symbol.static_lifetime = false;
  temp_symbol.location = location;
  temp_symbol.mode = "Python";
  temp_symbol.module = location.get_file().as_string();
  temp_symbol.file_local = true;
  temp_symbol.is_extern = false;

  return temp_symbol;
}

const nlohmann::json &get_return_statement(const nlohmann::json &function)
{
  for (const auto &stmt : function["body"])
  {
    if (python_frontend::get_statement_type(stmt) == StatementType::RETURN)
      return stmt;
  }

  throw std::runtime_error(
    "Function " + function["name"].get<std::string>() +
    " has no return statement");
}

bool python_converter::function_has_missing_return_paths(
  const nlohmann::json &function_node)
{
  const auto &body = function_node["body"];
  if (body.empty())
    return true;

  // Check if the last statement is a return
  const auto &last_stmt = body.back();
  if (last_stmt["_type"] == "Return")
    return false;

  // Check for if-else structures at the end
  if (last_stmt["_type"] == "If")
  {
    // Check if both if and else branches have returns
    bool if_has_return = false;
    bool else_has_return = false;

    // Check if branch
    if (!last_stmt["body"].empty())
    {
      const auto &if_last = last_stmt["body"].back();
      if_has_return = (if_last["_type"] == "Return");
    }

    // Check else branch
    if (last_stmt.contains("orelse") && !last_stmt["orelse"].empty())
    {
      const auto &else_last = last_stmt["orelse"].back();
      else_has_return = (else_last["_type"] == "Return");
    }

    return !(if_has_return && else_has_return);
  }

  return true; // No explicit return found
}

TypeFlags
python_converter::infer_types_from_returns(const nlohmann::json &function_body)
{
  TypeFlags flags;

  std::function<void(const nlohmann::json &)> scan = [&](const nlohmann::json
                                                           &body) {
    for (const auto &stmt : body)
    {
      if (stmt["_type"] == "Return" && stmt["value"].is_null())
      {
        // Bare "return" (no value) is semantically "return None"
        flags.has_none = true;
      }
      else if (stmt["_type"] == "Return" && !stmt["value"].is_null())
      {
        const auto &val = stmt["value"];

        if (val["_type"] == "Constant")
        {
          const auto &constant_val = val["value"];
          // Bignum literal (issue #4642): tagged Constants carry `_bigint`
          // alongside a null `value`. Classify as int rather than None so
          // type inference still produces an int return type; the actual
          // overflow diagnostic fires later in get_literal.
          if (val.contains("_bigint"))
            flags.has_int = true;
          else if (constant_val.is_number_float())
            flags.has_float = true;
          else if (constant_val.is_number_integer())
            flags.has_int = true;
          else if (constant_val.is_boolean())
            flags.has_bool = true;
          else if (constant_val.is_null())
            flags.has_none = true;
          else
          {
            // A string/object/array return literal in an Any-typed function
            // cannot be folded into the numeric TypeFlags hierarchy. Don't
            // abort the whole conversion (issue #2848): skip this return so
            // any numeric sibling return still drives the inferred type, and
            // an all-non-numeric body falls back to the caller's default.
            std::string type_name = constant_val.is_string()   ? "string"
                                    : constant_val.is_object() ? "object"
                                    : constant_val.is_array()  ? "array"
                                                               : "unknown";
            log_debug(
              "python",
              "infer_types_from_returns: ignoring unsupported return literal "
              "of type '{}' for Any inference",
              type_name);
          }
        }
        else if (val["_type"] == "BinOp" || val["_type"] == "UnaryOp")
        {
          flags.has_float = true; // Default for expressions
        }
        else if (val["_type"] == "Call")
        {
          // For return <func_call>(), look up the called function's returns
          // to infer the value type being propagated through the call
          const auto &func = val["func"];
          bool resolved = false;
          if (func.contains("id") && ast_json)
          {
            std::string called_name = func["id"].get<std::string>();
            const auto &module_body = (*ast_json)["body"];
            for (const auto &item : module_body)
            {
              if (item["_type"] == "FunctionDef" && item["name"] == called_name)
              {
                // Scan the called function's return statements directly
                // (one level only to avoid infinite recursion)
                for (const auto &s : item["body"])
                {
                  if (
                    s["_type"] == "Return" && !s["value"].is_null() &&
                    s["value"]["_type"] == "Constant" &&
                    !s["value"]["value"].is_null())
                  {
                    const auto &cv = s["value"]["value"];
                    if (cv.is_number_float())
                      flags.has_float = true;
                    else if (cv.is_number_integer())
                      flags.has_int = true;
                    else if (cv.is_boolean())
                      flags.has_bool = true;
                    resolved = true;
                  }
                }
                break;
              }
            }
          }
          if (!resolved)
            flags.has_int = true; // Default to int for unresolvable calls
        }
        else if (val["_type"] == "Name")
        {
          // return <variable> — indicates a value return of unknown type
          flags.has_int = true;
        }
      }

      if (stmt.contains("body") && stmt["body"].is_array())
        scan(stmt["body"]);
      if (stmt.contains("orelse") && stmt["orelse"].is_array())
        scan(stmt["orelse"]);
    }
  };

  scan(function_body);
  return flags;
}

namespace
{
bool is_list_literal_value(const nlohmann::json &v)
{
  return v.is_object() && v.contains("_type") &&
         (v["_type"] == "List" || v["_type"] == "ListComp");
}

// Collect names bound to a list value within 'body': either assigned a list
// literal / comprehension, or annotated `list`/`List`. The comprehension
// desugaring runs before return-type inference, so a `return [..][1:]` reaches
// here as `return ESBMC_listcomp_0[1:]` with `ESBMC_listcomp_0: list = [...]`
// hoisted above it — the name must be tracked to recognise the list slice.
void collect_list_bound_names(
  const nlohmann::json &body,
  std::set<std::string> &names)
{
  if (!body.is_array())
    return;

  for (const auto &stmt : body)
  {
    if (!stmt.is_object())
      continue;

    const std::string st = stmt.value("_type", std::string());
    if (
      st == "Assign" && stmt.contains("value") &&
      is_list_literal_value(stmt["value"]) && stmt.contains("targets"))
    {
      for (const auto &tgt : stmt["targets"])
        if (tgt.is_object() && tgt.value("_type", std::string()) == "Name")
          names.insert(tgt.value("id", std::string()));
    }
    else if (st == "AnnAssign" && stmt.contains("target"))
    {
      const bool list_ann =
        stmt.contains("annotation") && stmt["annotation"].is_object() &&
        (stmt["annotation"].value("id", std::string()) == "list" ||
         stmt["annotation"].value("id", std::string()) == "List");
      const bool list_val =
        stmt.contains("value") && is_list_literal_value(stmt["value"]);
      const auto &tgt = stmt["target"];
      if (
        (list_ann || list_val) && tgt.is_object() &&
        tgt.value("_type", std::string()) == "Name")
        names.insert(tgt.value("id", std::string()));
    }

    for (const char *key : {"body", "orelse"})
      if (stmt.contains(key))
        collect_list_bound_names(stmt[key], names);
  }
}

// Return true if a return expression denotes a list value: a list literal /
// comprehension, a name bound to one, or a slice of either (a slice of a list
// is a list, whereas an index t[i] is an element). Used to keep an unannotated,
// list-returning function from defaulting its return type to scalar double,
// which would retype the returned list's elements and break list equality
// (humaneval_62).
bool return_value_is_list(
  const nlohmann::json &val,
  const std::set<std::string> &list_names)
{
  if (!val.is_object() || !val.contains("_type"))
    return false;

  const std::string t = val["_type"].get<std::string>();
  if (t == "List" || t == "ListComp")
    return true;
  if (t == "Name")
    return list_names.count(val.value("id", std::string())) > 0;
  if (
    t == "Subscript" && val.contains("slice") && val["slice"].is_object() &&
    val["slice"].value("_type", std::string()) == "Slice" &&
    val.contains("value"))
    return return_value_is_list(val["value"], list_names);

  return false;
}

// Return true if any return statement in 'body' returns a list value. Mirrors
// the body/orelse recursion of infer_types_from_returns so the two agree on the
// statement set being inspected.
bool body_returns_list_value(const nlohmann::json &body)
{
  std::set<std::string> list_names;
  collect_list_bound_names(body, list_names);

  std::function<bool(const nlohmann::json &)> check =
    [&](const nlohmann::json &b) -> bool {
    if (!b.is_array())
      return false;
    for (const auto &stmt : b)
    {
      if (!stmt.is_object())
        continue;
      if (
        stmt.value("_type", std::string()) == "Return" &&
        stmt.contains("value") && !stmt["value"].is_null() &&
        return_value_is_list(stmt["value"], list_names))
        return true;
      for (const char *key : {"body", "orelse"})
        if (stmt.contains(key) && check(stmt[key]))
          return true;
    }
    return false;
  };

  return check(body);
}
} // namespace

// Return true if 'param_name' has any attribute written (x.attr = ...)
// anywhere in 'body' (recursive scan over nested blocks).
static bool param_is_mutated_in_body(
  const std::string &param_name,
  const nlohmann::json &body)
{
  if (!body.is_array())
    return false;

  for (const auto &stmt : body)
  {
    if (!stmt.is_object())
      continue;

    const std::string &stype =
      stmt.contains("_type") ? stmt["_type"].get<std::string>() : "";

    // x.attr = value  (plain assignment)
    if (stype == "Assign" && stmt.contains("targets"))
    {
      for (const auto &tgt : stmt["targets"])
      {
        if (
          tgt.contains("_type") && tgt["_type"] == "Attribute" &&
          tgt.contains("value") && tgt["value"].contains("_type") &&
          tgt["value"]["_type"] == "Name" && tgt["value"].contains("id") &&
          tgt["value"]["id"] == param_name)
          return true;
      }
    }
    // x.attr: T = value  (annotated assignment)
    else if (stype == "AnnAssign" && stmt.contains("target"))
    {
      const auto &tgt = stmt["target"];
      if (
        tgt.contains("_type") && tgt["_type"] == "Attribute" &&
        tgt.contains("value") && tgt["value"].contains("_type") &&
        tgt["value"]["_type"] == "Name" && tgt["value"].contains("id") &&
        tgt["value"]["id"] == param_name)
        return true;
    }

    // Recurse into nested blocks (if/while/for bodies, else branches)
    for (const char *key : {"body", "orelse", "handlers", "finalbody"})
    {
      if (stmt.contains(key) && stmt[key].is_array())
      {
        if (param_is_mutated_in_body(param_name, stmt[key]))
          return true;
      }
    }
  }
  return false;
}

// Collect the names of `self.<attr>` slots that `param_name` is stored
// into anywhere in `body` (recursive). When such an attribute is itself
// pointer-typed (see get_attributes_from_self / GitHub #4831), the
// parameter value used as the RHS must also be a pointer — otherwise the
// field-store creates a pointer/struct mismatch when downstream code
// reassigns the same attribute via a chained reference.
static void collect_self_attr_stores_of_param(
  const std::string &param_name,
  const nlohmann::json &body,
  std::vector<std::string> &out)
{
  if (!body.is_array())
    return;

  auto rhs_is_param = [&](const nlohmann::json &v) {
    return v.is_object() && v.value("_type", "") == "Name" &&
           v.contains("id") && v["id"] == param_name;
  };
  auto extract_self_attr = [](const nlohmann::json &t) -> std::string {
    if (
      t.is_object() && t.value("_type", "") == "Attribute" &&
      t.contains("value") && t["value"].is_object() &&
      t["value"].value("_type", "") == "Name" && t["value"].contains("id") &&
      t["value"]["id"] == "self" && t.contains("attr") && t["attr"].is_string())
      return t["attr"].get<std::string>();
    return "";
  };

  for (const auto &stmt : body)
  {
    if (!stmt.is_object())
      continue;

    const std::string &stype =
      stmt.contains("_type") ? stmt["_type"].get<std::string>() : "";

    // self.attr = param  (plain assignment)
    if (
      stype == "Assign" && stmt.contains("targets") && stmt.contains("value") &&
      rhs_is_param(stmt["value"]))
    {
      for (const auto &tgt : stmt["targets"])
      {
        const std::string name = extract_self_attr(tgt);
        if (!name.empty())
          out.push_back(name);
      }
    }
    // self.attr: T = param  (annotated assignment)
    else if (
      stype == "AnnAssign" && stmt.contains("target") &&
      stmt.contains("value") && rhs_is_param(stmt["value"]))
    {
      const std::string name = extract_self_attr(stmt["target"]);
      if (!name.empty())
        out.push_back(name);
    }

    // Recurse into nested blocks
    for (const char *key : {"body", "orelse", "handlers", "finalbody"})
    {
      if (stmt.contains(key) && stmt[key].is_array())
        collect_self_attr_stores_of_param(param_name, stmt[key], out);
    }
  }
}

static bool node_uses_param_as_list_like(
  const std::string &param_name,
  const nlohmann::json &node)
{
  if (!node.is_object())
    return false;

  if (node.contains("_type") && node["_type"].is_string())
  {
    const std::string node_type = node["_type"].get<std::string>();

    // x[i]
    if (
      node_type == "Subscript" && node.contains("value") &&
      node["value"].is_object() && node["value"].value("_type", "") == "Name" &&
      node["value"].value("id", "") == param_name)
      return true;

    if (node_type == "Call")
    {
      // len(x)
      if (
        node.contains("func") && node["func"].is_object() &&
        node["func"].value("_type", "") == "Name" &&
        node["func"].value("id", "") == "len" && node.contains("args") &&
        node["args"].is_array() && !node["args"].empty() &&
        node["args"][0].is_object() &&
        node["args"][0].value("_type", "") == "Name" &&
        node["args"][0].value("id", "") == param_name)
      {
        return true;
      }

      // x.append(...), x.pop(...), ...
      if (
        node.contains("func") && node["func"].is_object() &&
        node["func"].value("_type", "") == "Attribute" &&
        node["func"].contains("value") && node["func"]["value"].is_object() &&
        node["func"]["value"].value("_type", "") == "Name" &&
        node["func"]["value"].value("id", "") == param_name &&
        node["func"].contains("attr") && node["func"]["attr"].is_string())
      {
        const std::string attr = node["func"]["attr"].get<std::string>();
        if (
          attr == "append" || attr == "extend" || attr == "insert" ||
          attr == "pop" || attr == "remove" || attr == "clear" ||
          attr == "sort" || attr == "reverse")
          return true;
      }
    }
  }

  for (auto it = node.begin(); it != node.end(); ++it)
  {
    const auto &child = it.value();
    if (child.is_object())
    {
      if (node_uses_param_as_list_like(param_name, child))
        return true;
    }
    else if (child.is_array())
    {
      for (const auto &elem : child)
      {
        if (elem.is_object() && node_uses_param_as_list_like(param_name, elem))
          return true;
      }
    }
  }

  return false;
}

static bool param_is_list_like_in_body(
  const std::string &param_name,
  const nlohmann::json &body)
{
  if (!body.is_array())
    return false;

  for (const auto &stmt : body)
  {
    if (node_uses_param_as_list_like(param_name, stmt))
      return true;
  }

  return false;
}

size_t python_converter::register_function_argument(
  const nlohmann::json &element,
  code_typet &type,
  const symbol_id &id,
  const locationt &location,
  bool is_keyword_only)
{
  (void)is_keyword_only;

  // Extract the argument name and resolve its type from the annotation.
  // Special cases: `self` and `cls` are modelled as pointers to the current class
  std::string arg_name = element["arg"].get<std::string>();
  typet arg_type;

  if (arg_name == "self")
    arg_type = gen_pointer_type(type_handler_.get_typet(current_class_name_));
  else if (arg_name == "cls")
    arg_type = any_type();
  else
  {
    if (!element.contains("annotation") || element["annotation"].is_null())
    {
      // Python does not require type annotations; treat unannotated parameters
      // as Any (void*) to follow Python semantics.
      arg_type = any_type();
    }
    else
      arg_type = get_type_from_annotation(element["annotation"], element);
  }

  // Arrays are converted to pointers so that the backend receives the same
  // representation regardless of how the parameter is declared.
  if (arg_type.is_array())
    arg_type = gen_pointer_type(arg_type.subtype());

  // Object-model migration (#3067/#4773): a class-typed parameter receives a
  // migrated `Class*` instance, and Python passes objects by reference. Type
  // the formal as `Class*` (like `self`) so the call signature matches the
  // pointer argument and mutations through the parameter are visible to the
  // caller. A by-value struct formal would mismatch the pointer argument and
  // produce a malformed call expression.
  if (is_user_class_struct_type(arg_type))
    arg_type = gen_pointer_type(arg_type);

  assert(arg_type != typet());

  code_typet::argumentt arg;
  arg.type() = arg_type;
  arg.cmt_base_name(arg_name);

  // Build a unique identifier for the parameter. The identifier mirrors the
  // scheme used elsewhere in the converter (function-id@parameter-name)
  std::string arg_id = id.to_string() + "@" + arg_name;
  arg.cmt_identifier(arg_id);
  arg.identifier(arg_id);
  arg.location() = get_location_from_decl(element);

  type.arguments().push_back(arg);
  size_t inserted_index = type.arguments().size() - 1;

  // Materialise a symbol for the parameter so that subsequent passes (e.g.
  // attribute access on instances) can resolve it.
  symbolt param_symbol = create_symbol(
    location.get_file().as_string(),
    arg_name,
    arg_id,
    arg.location(),
    arg_type);
  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;
  param_symbol.static_lifetime = false;
  param_symbol.is_extern = false;
  symbol_table_.add(param_symbol);
  symbolt *stored_param = symbol_table_.find_symbol(arg_id);
  if (
    stored_param != nullptr && element.contains("annotation") &&
    !element["annotation"].is_null())
  {
    get_typechecker().cache_annotation_types(
      *stored_param, element["annotation"]);

    if (
      element["annotation"].contains("_type") &&
      element["annotation"]["_type"] == "Subscript" &&
      element["annotation"].contains("value") &&
      element["annotation"]["value"].contains("id"))
    {
      const std::string container_name =
        element["annotation"]["value"]["id"].get<std::string>();
      if (container_name == "List" || container_name == "list")
      {
        typet elem_type = type_handler_.get_list_type(element).subtype();
        if (!elem_type.is_empty())
          python_list::add_type_info_entry(arg_id, "", elem_type);
      }
    }
  }

  // If the parameter is class-typed (e.g. Foo), copy instance attributes from
  // the class’ synthetic `self` symbol so method bodies can access members via
  // this parameter.
  if (arg_name != "self" && arg_name != "cls")
  {
    typet base_type = arg_type.is_pointer() ? arg_type.subtype() : arg_type;
    if (base_type.id() == "symbol")
      base_type = ns.follow(base_type);

    if (base_type.is_struct())
    {
      const struct_typet &struct_type = to_struct_type(base_type);
      std::string class_tag = struct_type.tag().as_string();

      std::string class_name = extract_class_name_from_tag(class_tag);

      symbol_id self_sid(
        location.get_file().as_string(), class_name, class_name);
      self_sid.set_object("self");

      copy_instance_attributes(self_sid.to_string(), arg_id);

      std::string normalized_key = create_normalized_self_key(class_tag);
      copy_instance_attributes(normalized_key, arg_id);
    }
  }

  return inserted_index;
}

void python_converter::process_function_arguments(
  const nlohmann::json &function_node,
  code_typet &type,
  const symbol_id &id,
  const locationt &location)
{
  std::vector<size_t> positional_indices;
  std::vector<size_t> kwonly_indices;

  // Extract args node to avoid repeated access
  const nlohmann::json &args_node = function_node["args"];

  // Process regular arguments
  for (const nlohmann::json &element : args_node["args"])
  {
    size_t index =
      register_function_argument(element, type, id, location, false);
    positional_indices.push_back(index);
  }

  // Process keyword-only arguments (parameters after * separator)
  if (args_node.contains("kwonlyargs") && !args_node["kwonlyargs"].is_null())
  {
    for (const nlohmann::json &element : args_node["kwonlyargs"])
    {
      size_t index =
        register_function_argument(element, type, id, location, true);
      kwonly_indices.push_back(index);
    }
  }

  if (
    args_node.contains("defaults") && args_node["defaults"].is_array() &&
    !args_node["defaults"].empty() && !positional_indices.empty())
  {
    const auto &defaults = args_node["defaults"];
    size_t defaults_count = defaults.size();

    if (defaults_count <= positional_indices.size())
    {
      for (size_t i = 0; i < defaults_count; ++i)
      {
        size_t positional_index =
          positional_indices[positional_indices.size() - defaults_count + i];
        if (!defaults[i].is_null())
        {
          exprt default_expr = get_expr(defaults[i]);
          type.arguments()[positional_index].default_value() = default_expr;

          // If the default is a function pointer and the parameter was
          // annotated as Any (void*), upgrade the parameter type to match.
          // This enables indirect-call resolution for function-alias defaults
          // like def h(op=g) where g = f (a named function).
          if (
            default_expr.type().is_pointer() &&
            default_expr.type().subtype().is_code())
          {
            auto &param_arg = type.arguments()[positional_index];
            if (param_arg.type() == any_type())
            {
              param_arg.type() = default_expr.type();
              std::string param_id = param_arg.cmt_identifier().as_string();
              if (!param_id.empty())
              {
                symbolt *param_sym = symbol_table_.find_symbol(param_id);
                if (param_sym)
                  param_sym->set_type(default_expr.type());
              }
            }
          }
        }
      }
    }
  }

  if (
    args_node.contains("kw_defaults") && args_node["kw_defaults"].is_array() &&
    args_node["kw_defaults"].size() == kwonly_indices.size())
  {
    const auto &kw_defaults = args_node["kw_defaults"];
    for (size_t i = 0; i < kw_defaults.size(); ++i)
    {
      if (!kw_defaults[i].is_null())
      {
        exprt default_expr = get_expr(kw_defaults[i]);
        type.arguments()[kwonly_indices[i]].default_value() = default_expr;
      }
    }
  }

  // Python object reference semantics: if a non-enum class parameter is
  // mutated inside the function (x.attr = ...), model it as a pointer so
  // that mutations are visible to the caller (same as 'self' for methods).
  if (!function_node.contains("body"))
    return;
  const nlohmann::json &body = function_node["body"];

  // Refine unannotated Any parameters to list model type when body usage
  // clearly matches list semantics (len(x), x[i], list mutator methods).
  // Restrict this refinement to functions from the main source file to avoid
  // affecting imported module internals.
  if (location.get_file().as_string() == main_python_file)
  {
    for (auto &param_arg : type.arguments())
    {
      const std::string param_name = param_arg.get_base_name().as_string();
      if (param_name == "self" || param_name == "cls" || param_name.empty())
        continue;

      if (param_arg.type() != any_type())
        continue;

      if (!param_is_list_like_in_body(param_name, body))
        continue;

      typet list_t = type_handler_.get_list_type();
      param_arg.type() = list_t;

      const std::string param_id = param_arg.cmt_identifier().as_string();
      if (!param_id.empty())
      {
        symbolt *param_sym = symbol_table_.find_symbol(param_id);
        if (param_sym)
          param_sym->set_type(list_t);
      }
    }
  }

  for (auto &param_arg : type.arguments())
  {
    const std::string param_name = param_arg.get_base_name().as_string();
    if (param_name == "self" || param_name == "cls" || param_name.empty())
      continue;

    // Only applies to user-defined (non-enum) class-typed parameters.
    typet ptype = param_arg.type();
    if (ptype.id() == "symbol")
      ptype = ns.follow(ptype);
    if (!ptype.is_struct())
      continue;
    const std::string class_tag = to_struct_type(ptype).tag().as_string();
    const std::string class_name = extract_class_name_from_tag(class_tag);
    if (
      !json_utils::is_class(class_name, *ast_json) ||
      python_frontend::is_enum_class(class_name, *ast_json))
      continue;

    // Upgrade to pointer when (a) the function body mutates this parameter
    // (param.attr = ...), or (b) the parameter is stored into a `self.attr`
    // slot whose declared field type is itself a pointer — the field-store
    // would otherwise create a struct/pointer mismatch. (See GitHub #4831.)
    bool stored_to_ptr_attr = false;
    if (!current_class_name_.empty())
    {
      std::vector<std::string> stored_attrs;
      collect_self_attr_stores_of_param(param_name, body, stored_attrs);
      if (!stored_attrs.empty())
      {
        const symbolt *cls_sym =
          symbol_table_.find_symbol("tag-" + current_class_name_);
        if (cls_sym && cls_sym->get_type().is_struct())
        {
          const struct_typet &cs = to_struct_type(cls_sym->get_type());
          for (const std::string &a : stored_attrs)
          {
            if (cs.has_component(a) && cs.get_component(a).type().is_pointer())
            {
              stored_to_ptr_attr = true;
              break;
            }
          }
        }
      }
    }

    if (!param_is_mutated_in_body(param_name, body) && !stored_to_ptr_attr)
      continue;

    // Upgrade the parameter to a pointer and update the parameter symbol.
    typet ptr_type = gen_pointer_type(param_arg.type());
    param_arg.type() = ptr_type;
    const std::string param_id = param_arg.cmt_identifier().as_string();
    if (!param_id.empty())
    {
      symbolt *param_sym = symbol_table_.find_symbol(param_id);
      if (param_sym)
        param_sym->set_type(ptr_type);
    }
  }
}

void python_converter::validate_return_paths(
  const nlohmann::json &function_node,
  const code_typet &type,
  exprt &function_body)
{
  // Skip validation for void returns and constructors
  if (
    type.return_type().is_empty() ||
    type.return_type().id() == typet::t_empty ||
    type.return_type().id() == "constructor" ||
    !function_has_missing_return_paths(function_node))
  {
    return;
  }

  locationt loc = get_location_from_decl(function_node);

  code_assertt missing_return_assert;
  // V.3: build the always-fail assert condition in IREP2.
  missing_return_assert.assertion() = migrate_expr_back(gen_false_expr());
  missing_return_assert.location() = loc;
  missing_return_assert.location().comment(
    "Missing return statement detected in function '" + current_func_name_ +
    "'");

  function_body.copy_to_operands(missing_return_assert);
}

typet python_converter::infer_return_type_from_body(const nlohmann::json &body)
{
  auto infer_constant_type = [](const nlohmann::json &constant_value) -> typet {
    if (constant_value.is_number_float())
      return double_type();
    if (constant_value.is_number_integer())
      return long_long_int_type();
    if (constant_value.is_boolean())
      return bool_type();
    if (constant_value.is_string())
      return gen_pointer_type(char_type());
    if (constant_value.is_null())
      return none_type();
    return empty_typet();
  };

  for (const auto &stmt : body)
  {
    if (stmt["_type"] == "Return" && !stmt["value"].is_null())
    {
      const auto &ret_val = stmt["value"];

      // `return self` in a method: surface the class's struct value type so
      // callers can assign to a `Class`-typed local. Without this, fallback
      // inference picks up `Class *` from `self` and the call-site assignment
      // becomes a pointer-to-struct mismatch that trips an assertion in
      // value_set::make_member on later member access. See GitHub #4514.
      if (
        ret_val["_type"] == "Name" && ret_val.contains("id") &&
        ret_val["id"] == "self" && !current_class_name_.empty())
        return type_handler_.get_typet(current_class_name_);

      // If returning a tuple, infer its type. This routine runs twice:
      // once before function body conversion (so return statements can
      // be typed in the right context), and once afterwards.
      // tuple_handler evaluates each element via get_expr, which aborts
      // on a Name that refers to a function-local variable not yet
      // declared in the symbol table. Restrict the pre-body call to
      // tuples whose elements are all Constants (or otherwise
      // resolvable without symbol lookup); for tuples containing
      // locals, leave the return type empty and let the post-body
      // second pass infer it. See #4807.
      if (ret_val["_type"] == "Tuple")
      {
        bool all_constant = true;
        if (ret_val.contains("elts") && ret_val["elts"].is_array())
        {
          for (const auto &elt : ret_val["elts"])
          {
            if (!elt.contains("_type") || elt["_type"] != "Constant")
            {
              all_constant = false;
              break;
            }
          }
        }
        if (all_constant)
          return tuple_handler_->get_tuple_expr(ret_val).type();
        // Fall through -- post-body inference will pick up the real
        // type once the local symbols have been declared.
      }

      // Constant returns (including strings)
      if (ret_val["_type"] == "Constant" && ret_val.contains("value"))
      {
        // Bignum literal (issue #4642): tagged Constant has null value but
        // the function genuinely returns an int. Use int rather than fall
        // through to the null → none branch in infer_constant_type.
        if (ret_val.contains("_bigint"))
          return long_long_int_type();
        typet inferred = infer_constant_type(ret_val["value"]);
        if (!inferred.is_empty())
          return inferred;
      }

      // Heuristic: return dict.get(key, default) -> infer from default literal.
      if (
        ret_val["_type"] == "Call" && ret_val.contains("func") &&
        ret_val["func"].contains("_type") &&
        ret_val["func"]["_type"] == "Attribute" &&
        ret_val["func"].contains("attr") && ret_val["func"]["attr"] == "get" &&
        ret_val.contains("args") && ret_val["args"].is_array() &&
        ret_val["args"].size() >= 2)
      {
        const auto &default_arg = ret_val["args"][1];
        if (
          default_arg.contains("_type") && default_arg["_type"] == "Constant" &&
          default_arg.contains("value"))
        {
          typet inferred = infer_constant_type(default_arg["value"]);
          if (!inferred.is_empty())
            return inferred;
        }
      }
    }
  }

  return empty_typet();
}

void python_converter::get_function_definition(
  const nlohmann::json &function_node)
{
  // Function return type
  code_typet type;
  const nlohmann::json &return_node = function_node["returns"];

  // Tracks annotations that already encode Optional (e.g. Optional[T] or
  // T | None). When true, the later body_has_none_return pass must not
  // re-wrap the return type as Optional<existing-type>.
  bool annotation_is_optional = false;

  // Determine return type
  if (
    return_node.is_null() ||
    (return_node["_type"] == "Constant" && return_node["value"].is_null()))
  {
    type.return_type() = empty_typet();
  }
  else if (return_node.contains("id") || return_node["_type"] == "Subscript")
  {
    const nlohmann::json &return_type = (return_node["_type"] == "Subscript")
                                          ? return_node["value"]["id"]
                                          : return_node["id"];

    if (return_type == "Any")
    {
      // Infer type from return statements
      TypeFlags flags = infer_types_from_returns(function_node["body"]);

      if (
        !flags.has_float && !flags.has_int && !flags.has_bool &&
        body_returns_list_value(function_node["body"]))
      {
        // No scalar type could be inferred but the body returns a list whose
        // element type is unknown. Defaulting to scalar double would retype the
        // returned list's elements and break equality against, e.g., an int
        // list (humaneval_62). Use a generic list type, as an explicit
        // `-> list` annotation would.
        type.return_type() = type_handler_.get_list_type();
      }
      else
      {
        type.return_type() =
          type_utils::select_widest_type(flags, double_type());

        if (!flags.has_float && !flags.has_int && !flags.has_bool)
          log_warning("Default to double since no type could be inferred");
      }
    }
    else if (return_type == "Union")
    {
      // Extract Union member types
      TypeFlags flags = type_utils::extract_union_types(return_node["slice"]);
      type.return_type() = type_utils::select_widest_type(flags, any_type());

      if (!flags.has_float && !flags.has_int && !flags.has_bool)
        log_warning("Union with no recognized types, defaulting to pointer");
    }
    else if (return_type == "list" || return_type == "List")
    {
      type.return_type() = type_handler_.get_list_type();
    }
    else if (return_type == "dict" || return_type == "Dict")
    {
      type.return_type() = dict_handler_->get_dict_struct_type();
    }
    else if (return_type == "str" || return_type == "string")
    {
      // String return types should be pointers, not arrays
      type.return_type() = gen_pointer_type(char_type());
    }
    else if (
      (return_type == "Tuple" || return_type == "tuple") &&
      return_node["_type"] == "Subscript")
    {
      type.return_type() =
        tuple_handler_->get_tuple_type_from_annotation(return_node);
    }
    else if (return_type == "Optional" && return_node["_type"] == "Subscript")
    {
      // Optional[T]: delegate to the annotation handler, which builds either
      // an Optional<T> struct (for primitive T) or a T* pointer (for str /
      // class T, where None is encoded as NULL). The previous fallthrough to
      // get_typet("Optional") returned a bare pointer_type() (unsignedbv),
      // which the later body_has_none_return pass then re-wrapped as
      // Optional<unsignedbv> — a struct unrelated to the annotated T.
      type.return_type() = get_type_from_annotation(return_node, function_node);
      annotation_is_optional = true;
    }
    else
    {
      type.return_type() =
        type_handler_.get_typet(return_type.get<std::string>());
    }
  }
  else if (return_node["_type"] == "BinOp")
  {
    // Handle PEP 604 union syntax: int | bool
    TypeFlags flags = type_utils::extract_binop_union_types(return_node);
    type.return_type() = type_utils::select_widest_type(flags, any_type());

    if (!flags.has_float && !flags.has_int && !flags.has_bool)
      log_warning("Union with no recognized types, defaulting to pointer");
  }
  else if (return_node["_type"] == "Tuple")
  {
    // Handle tuple return types such as (int, str)
    // TODO: we must still handle tuple types!
    type.return_type() = type_handler_.get_typet(std::string("tuple"));
  }
  else if (return_node["_type"] == "Constant" || return_node["_type"] == "Str")
  {
    std::string type_string =
      type_utils::remove_quotes(return_node["value"].get<std::string>());
    if (type_string == "str" || type_string == "string")
      type.return_type() = gen_pointer_type(char_type());
    else
      type.return_type() = type_handler_.get_typet(type_string);
  }
  else
    throw std::runtime_error("Return type undefined");

  // Setup function context
  const std::string caller_func_name = current_func_name_;

  locationt location = get_location_from_decl(function_node);

  current_element_type = type.return_type();
  std::string func_name = function_node["name"].get<std::string>();

  // __init__() is renamed to Classname()
  if (func_name == "__init__")
  {
    func_name = current_class_name_;
    type.return_type() = typet("constructor");
  }

  // If we are inside another function, create a nested name
  if (!caller_func_name.empty())
  {
    current_func_name_ = caller_func_name + "@F@" + func_name;
  }
  else
  {
    current_func_name_ = func_name;
  }

  scope_stack_.push_back("@F@" + func_name);

  symbol_id id = create_symbol_id();

  std::string module_name =
    current_python_file.substr(0, current_python_file.find_last_of("."));

  // Process function arguments
  process_function_arguments(function_node, type, id, location);

  // Create and register function symbol
  symbolt symbol = create_symbol(
    module_name, current_func_name_, id.to_string(), location, type);
  symbol.lvalue = true;
  symbol.is_extern = false;
  symbol.file_local = false;

  symbolt *added_symbol = symbol_table_.move_symbol_to_context(symbol);

  // Pre-scan: detect mixed value+None returns and upgrade return type to
  // Optional so None checks work correctly at runtime.
  // This applies even when the function has an explicit return annotation:
  // Python does not enforce annotations, so `-> int` with `return None` in
  // the body must be modelled as Optional[int].
  auto body_has_none_return = [](const nlohmann::json &body) -> bool {
    std::function<bool(const nlohmann::json &)> scan =
      [&](const nlohmann::json &stmts) -> bool {
      for (const auto &s : stmts)
      {
        if (s["_type"] == "Return")
        {
          if (s["value"].is_null())
            return true;
          // Bignum literals (issue #4642) carry `_bigint` with a null value;
          // they are int returns, not None.
          if (
            s["value"]["_type"] == "Constant" &&
            s["value"]["value"].is_null() && !s["value"].contains("_bigint"))
            return true;
        }
        if (s.contains("body") && s["body"].is_array() && scan(s["body"]))
          return true;
        if (s.contains("orelse") && s["orelse"].is_array() && scan(s["orelse"]))
          return true;
      }
      return false;
    };
    return scan(body);
  };

  bool already_optional =
    annotation_is_optional ||
    (type.return_type().is_struct() && to_struct_type(type.return_type())
                                         .tag()
                                         .as_string()
                                         .starts_with("tag-Optional_"));
  if (!already_optional && body_has_none_return(function_node["body"]))
  {
    if (type.return_type().is_empty())
    {
      // Unannotated function: need full type inference to pick value_type
      TypeFlags return_flags = infer_types_from_returns(function_node["body"]);
      bool has_value_return =
        return_flags.has_int || return_flags.has_float || return_flags.has_bool;
      if (has_value_return)
      {
        typet value_type =
          type_utils::select_widest_type(return_flags, long_long_int_type());
        typet optional_type = type_handler_.build_optional_type(value_type);
        type.return_type() = optional_type;
        current_element_type = optional_type;
        added_symbol->set_type(type);
      }
    }
    else
    {
      // Explicitly-annotated function (e.g., -> int) with return None paths:
      // upgrade the annotated type to Optional[annotated_type].
      typet optional_type =
        type_handler_.build_optional_type(type.return_type());
      type.return_type() = optional_type;
      current_element_type = optional_type;
      added_symbol->set_type(type);
    }
  }

  // For unannotated functions, attempt AST-based return inference before body
  // conversion so return expressions are typed in the right context.
  if (type.return_type().is_empty())
  {
    typet inferred_type = infer_return_type_from_body(function_node["body"]);
    if (!inferred_type.is_empty())
    {
      type.return_type() = inferred_type;
      current_element_type = inferred_type;
      added_symbol->set_type(type);
    }
  }

  // Save function return type for use in get_return_statements
  typet saved_func_return_type = current_func_return_type_;
  current_func_return_type_ = type.return_type();

  // Process function body. Mark it as a function body (not a conditional one)
  // so straight-line retyping (#4770/#4774) is permitted on the function's own
  // unconditional statements — see the retype gate in get_var_assign.
  exprt function_body =
    get_block(function_node["body"], /*is_function_body=*/true);

  // Restore saved function return type (for nested function defs)
  current_func_return_type_ = saved_func_return_type;

  // If return type is empty/unannotated, try to infer from return statements
  if (type.return_type().is_empty())
  {
    typet inferred_type = infer_return_type_from_body(function_node["body"]);
    if (!inferred_type.is_empty())
    {
      type.return_type() = inferred_type;
      added_symbol->set_type(type); // Update the symbol's type
    }
  }

  // If return type is still empty, scan the converted GOTO body for RETURN
  // instructions with typed values. This handles indirect calls through
  // function-pointer parameters (e.g., "return op(1,1)" where op defaults
  // to a typed function pointer).
  if (type.return_type().is_empty())
  {
    // First, the original top-level scan: a function with a fall-through
    // `return` at the body's top level (e.g. an early `return -1` sentinel
    // inside an `if`, followed by `return bin(...)` at the end) is typed from
    // that top-level return -- the dominant exit. Picking a nested branch's
    // type instead would narrow a heterogeneous function to the wrong branch
    // and collapse the call-site cross-type `==` fold to constant False
    // (GitHub #5157).
    auto top_level_return_type = [&]() -> std::optional<typet> {
      for (const auto &instr : function_body.operands())
      {
        if (!instr.is_code() || to_code(instr).get_statement() != "return")
          continue;
        const code_returnt &ret = to_code_return(to_code(instr));
        if (ret.has_return_value() && !ret.return_value().type().is_empty())
          return ret.return_value().type();
      }
      return std::nullopt;
    };

    // Fallback: when every `return` is nested inside a conditional (so the
    // top-level scan finds nothing), recurse to find a typed RETURN. Otherwise
    // an all-nested body (e.g. `if c: return s.split() else: return s.split()`)
    // leaves the return type empty -> void -> the value is stripped by
    // remove_returns and the call site reads nondet.
    std::function<std::optional<typet>(const exprt &)> nested_return_type =
      [&](const exprt &node) -> std::optional<typet> {
      if (node.is_code() && to_code(node).get_statement() == "return")
      {
        const code_returnt &ret = to_code_return(to_code(node));
        if (ret.has_return_value() && !ret.return_value().type().is_empty())
          return ret.return_value().type();
      }
      for (const auto &op : node.operands())
      {
        if (std::optional<typet> found = nested_return_type(op))
          return found;
      }
      return std::nullopt;
    };

    std::optional<typet> ret_type = top_level_return_type();
    if (!ret_type)
      ret_type = nested_return_type(function_body);
    if (ret_type)
    {
      type.return_type() = *ret_type;
      added_symbol->set_type(type);
    }
  }

  // Inject runtime checks for annotated parameters
  if (type_assertions_enabled())
    get_typechecker().inject_parameter_type_assertions(
      function_node, id, type, function_body);

  // Add ESBMC_Hide label for models/imports
  if (is_loading_models || is_importing_module)
  {
    code_labelt esbmc_hide;
    esbmc_hide.set_label("__ESBMC_HIDE");
    esbmc_hide.code() = code_skipt();
    function_body.operands().insert(
      function_body.operands().begin(), esbmc_hide);
  }

  // Validate return paths
  validate_return_paths(function_node, type, function_body);

  added_symbol->set_value(function_body);

  scope_stack_.pop_back();

  // Restore caller function name
  current_func_name_ = caller_func_name;
}
