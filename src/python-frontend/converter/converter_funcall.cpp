#include <python-frontend/function_call/builder.h>
#include <python-frontend/function_call/expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_consteval.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_lambda.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/python_types.h>
#include <util/std_code.h>
#include <util/string_constant.h>

using namespace json_utils;

// Resolve symbol values to constants
exprt python_converter::get_resolved_value(const exprt &expr)
{
  // Handle direct function call expressions
  if (expr.id() == "sideeffect")
  {
    const side_effect_exprt &side_effect = to_side_effect_expr(expr);
    if (
      side_effect.get_statement() == "function_call" &&
      side_effect.operands().size() >= 2)
      // Structure: operand 0 = function symbol, operand 1 = arguments
      return resolve_function_call(
        side_effect.operands()[0], side_effect.operands()[1]);
  }

  // Handle symbols that contain function calls or constants
  if (!expr.is_symbol())
    return nil_exprt();

  const symbol_exprt &sym = to_symbol_expr(expr);
  const symbolt *symbol = symbol_table_.find_symbol(sym.get_identifier());

  if (!symbol || symbol->get_value().is_nil())
    return nil_exprt();

  // Return constant values directly
  if (symbol->get_value().is_constant())
    return symbol->get_value();

  // Handle function calls stored as code
  if (symbol->get_value().is_code())
  {
    const codet &code = to_code(symbol->get_value());

    if (code.get_statement() == "function_call" && code.operands().size() >= 3)
    {
      // Structure: operand 1 = function symbol, operand 2 = arguments
      exprt result =
        resolve_function_call(code.operands()[1], code.operands()[2]);
      if (!result.is_nil())
        return result;
    }
  }

  return nil_exprt();
}

// Resolve function calls (both identity functions and constant-returning functions)
exprt python_converter::resolve_function_call(
  const exprt &func_expr,
  const exprt &args_expr)
{
  if (!func_expr.is_symbol())
    return nil_exprt();

  const symbol_exprt &func_sym = to_symbol_expr(func_expr);
  const symbolt *func_symbol =
    symbol_table_.find_symbol(func_sym.get_identifier());

  if (!func_symbol || func_symbol->get_value().is_nil())
    return nil_exprt();

  // First check if this function returns a constant value
  exprt constant_result =
    get_function_constant_return(func_symbol->get_value());
  if (!constant_result.is_nil())
    return constant_result;

  // Then check if this function is an identity function (returns its parameter)
  if (!is_identity_function(
        func_symbol->get_value(), func_sym.get_identifier().as_string()))
    return nil_exprt();

  // Extract the first argument for identity functions
  if (args_expr.id() != "arguments" || args_expr.operands().empty())
    return nil_exprt();

  exprt arg = args_expr.operands()[0];

  // Handle address_of wrapper
  if (arg.is_address_of() && arg.operands().size() > 0)
    arg = arg.operands()[0];

  // If the argument is itself a function call, recursively resolve it
  if (arg.id() == "sideeffect")
  {
    exprt nested_resolved = get_resolved_value(arg);
    if (!nested_resolved.is_nil())
      arg = nested_resolved;
  }

  // If the argument is a symbol, try to resolve it to its constant value
  if (arg.is_symbol())
  {
    const symbol_exprt &sym = to_symbol_expr(arg);
    const symbolt *symbol = symbol_table_.find_symbol(sym.get_identifier());
    if (symbol && symbol->get_value().is_constant())
      arg = symbol->get_value();
  }

  // Return string constants, array constants, and single character constants
  if (
    arg.id() == "string-constant" || (arg.is_constant() && arg.is_array()) ||
    (arg.is_constant() && arg.type().is_array()) ||
    (arg.is_constant() &&
     (arg.type().is_unsignedbv() || arg.type().is_signedbv())))
  {
    return arg;
  }

  return nil_exprt();
}

// Check if a function returns a constant value
exprt python_converter::get_function_constant_return(const exprt &func_value)
{
  if (!func_value.is_code())
    return nil_exprt();

  const codet &func_code = to_code(func_value);

  // Check if it's a simple return statement with a constant
  if (func_code.get_statement() == "return")
  {
    const code_returnt &ret = to_code_return(func_code);
    if (ret.has_return_value())
    {
      const exprt &return_val = ret.return_value();
      if (
        return_val.id() == "string-constant" ||
        (return_val.is_constant() && return_val.is_array()) ||
        (return_val.is_constant() && return_val.type().is_array()) ||
        (return_val.is_constant() && (return_val.type().is_unsignedbv() ||
                                      return_val.type().is_signedbv())))
      {
        return return_val;
      }
    }
  }

  // Check nested code structures
  for (const auto &operand : func_value.operands())
  {
    if (operand.is_code())
    {
      const codet &sub_code = to_code(operand);
      if (sub_code.get_statement() == "return")
      {
        const code_returnt &ret = to_code_return(sub_code);
        if (ret.has_return_value())
        {
          const exprt &return_val = ret.return_value();
          if (
            return_val.id() == "string-constant" ||
            (return_val.is_constant() && return_val.is_array()) ||
            (return_val.is_constant() && return_val.type().is_array()) ||
            (return_val.is_constant() && (return_val.type().is_unsignedbv() ||
                                          return_val.type().is_signedbv())))
          {
            return return_val;
          }
        }
      }
    }
  }

  return nil_exprt();
}

// Check if a function is an identity function (returns its parameter)
bool python_converter::is_identity_function(
  const exprt &func_value,
  const std::string &func_identifier)
{
  if (!func_value.is_code())
    return false;

  const codet &func_code = to_code(func_value);

  // Check if it's a simple return statement
  if (func_code.get_statement() == "return")
  {
    const code_returnt &ret = to_code_return(func_code);
    if (ret.has_return_value() && ret.return_value().is_symbol())
    {
      const symbol_exprt &return_sym = to_symbol_expr(ret.return_value());
      std::string return_identifier = return_sym.get_identifier().as_string();
      std::string parameter_prefix = func_identifier + "@";

      // Check if the returned symbol is a parameter of this function
      // Parameter pattern: func_identifier + "@" + parameter_name
      if (
        return_identifier.size() >= parameter_prefix.size() &&
        return_identifier.compare(
          0, parameter_prefix.size(), parameter_prefix) == 0)
        return true;
    }
  }

  // Check nested code structures
  for (const auto &operand : func_value.operands())
  {
    if (operand.is_code())
    {
      const codet &sub_code = to_code(operand);
      if (sub_code.get_statement() == "return")
      {
        const code_returnt &ret = to_code_return(sub_code);
        if (ret.has_return_value() && ret.return_value().is_symbol())
        {
          const symbol_exprt &return_sym = to_symbol_expr(ret.return_value());
          std::string return_identifier =
            return_sym.get_identifier().as_string();
          std::string parameter_prefix = func_identifier + "@";

          // Check if the returned symbol is a parameter of this function
          if (
            return_identifier.size() >= parameter_prefix.size() &&
            return_identifier.compare(
              0, parameter_prefix.size(), parameter_prefix) == 0)
            return true;
        }
      }
    }
  }

  return false;
}
exprt python_converter::get_function_call(const nlohmann::json &element)
{
  if (!element.contains("func") || element["_type"] != "Call")
    throw std::runtime_error("Invalid function call");

  // Handle direct range(...) calls by converting to list
  if (element["func"]["_type"] == "Name" && element["func"]["id"] == "range")
  {
    const auto &range_args = element["args"];
    return python_list::build_list_from_range(*this, range_args, element);
  }

  // Handle direct slice(...) calls by materialising a PySliceObject value.
  if (element["func"]["_type"] == "Name" && element["func"]["id"] == "slice")
    return build_slice_from_args(element["args"], element);

  // Handle set(iterable) and frozenset(iterable) calls. frozenset is
  // modelled as a regular set: the verifier doesn't reason about
  // immutability, so the only divergence (rejecting mutation methods)
  // is academic for the safety properties we check.
  if (
    element["func"]["_type"] == "Name" &&
    (element["func"]["id"] == "set" || element["func"]["id"] == "frozenset") &&
    element.contains("args") && element["args"].size() == 1)
  {
    exprt iterable_expr = get_expr(element["args"][0]);
    python_set set_handler(*this, element);
    return set_handler.get_from_iterable(iterable_expr, element);
  }

  // frozenset() with no args — empty frozenset.
  if (
    element["func"]["_type"] == "Name" &&
    element["func"]["id"] == "frozenset" &&
    (!element.contains("args") || element["args"].empty()))
  {
    python_set set_handler(*this, element);
    return set_handler.get_empty_set();
  }

  // list() with no args — empty list. Lower to a List literal so it routes
  // through the well-tested `[]` path (python_list::get()).
  if (
    element["func"]["_type"] == "Name" && element["func"]["id"] == "list" &&
    (!element.contains("args") || element["args"].empty()))
  {
    nlohmann::json list_node;
    list_node["_type"] = "List";
    list_node["elts"] = nlohmann::json::array();
    for (const char *k :
         {"lineno", "col_offset", "end_lineno", "end_col_offset"})
      if (element.contains(k))
        list_node[k] = element[k];
    return get_expr(list_node);
  }

  // Handle list(...) calls
  if (
    element["func"]["_type"] == "Name" && element["func"]["id"] == "list" &&
    element.contains("args") && element["args"].size() == 1)
  {
    const auto &list_arg = element["args"][0];

    // Handle list(range(...))
    if (
      list_arg["_type"] == "Call" && list_arg["func"]["_type"] == "Name" &&
      list_arg["func"]["id"] == "range")
    {
      return python_list::build_list_from_range(
        *this, list_arg["args"], element);
    }

    // Handle list(iterable) where iterable evaluates to a list — e.g.
    // list(d.items()), list(d.keys()), list(d.values()), list(some_list).
    // d.items() returns the keys member as a placeholder list (correct size).
    exprt arg_expr = get_expr(list_arg);
    if (arg_expr.type() == type_handler_.get_list_type())
      return arg_expr;
    // Fall through to the generic function-call builder below for non-list
    // iterables (e.g. list("abc") or list(42)).
  }

  // Handle dict(iterable) constructor:
  // lower to a Dict literal so it routes through the existing dict path.
  // Without this, the iterable is processed as a list
  // and later ".items()" / ".keys()" accesses crash BMC.
  if (
    element["func"]["_type"] == "Name" && element["func"]["id"] == "dict" &&
    element.contains("args") && element["args"].is_array() &&
    element["args"].size() == 1)
  {
    exprt result = dict_handler_->handle_dict_constructor(element);
    if (!result.is_nil())
      return result;
  }

  // Handle dict.keys(), dict.values(), and dict.items() methods
  if (element["func"]["_type"] == "Attribute")
  {
    const std::string &method_name = element["func"]["attr"].get<std::string>();

    if (method_name == "conjugate")
    {
      exprt result = complex_handler_.handle_attribute(element);
      if (!result.is_nil())
        return result;
    }

    if (
      method_name == "keys" || method_name == "values" ||
      method_name == "items")
    {
      exprt obj_expr = get_expr(element["func"]["value"]);

      // Check if this is a dict type
      if (dict_handler_->is_dict_type(obj_expr.type()))
      {
        typet list_type = type_handler_.get_list_type();
        // V.3: IREP2 member access (exact round-trip of member_exprt);
        // `obj_expr` is dict-typed (is_dict_type ⇒ struct), so the member2t
        // source precondition holds.
        expr2tc dict2;
        migrate_expr(obj_expr, dict2);
        if (method_name == "items")
        {
          // For-loop uses of items() are rewritten by the preprocessor into
          // separate keys()/values() accesses and never reach here.
          // For standalone/discarded calls (e.g. bare `d.items()` statement),
          // return the keys member as a placeholder — same size as the dict,
          // so size/emptiness comparisons (e.g. list(d.items()) == []) work.
          // Full (key, value) tuple semantics are not modelled.
          return migrate_expr_back(
            member2tc(migrate_type(list_type), dict2, "keys"));
        }
        // Return the keys or values member directly
        return migrate_expr_back(
          member2tc(migrate_type(list_type), dict2, method_name));
      }
    }
  }

  // Compile-time evaluation for parse_nested_parens on constant strings.
  if (
    element["func"]["_type"] == "Name" && element["func"].contains("id") &&
    element["func"]["id"] == "parse_nested_parens" &&
    element.contains("args") && element["args"].is_array() &&
    element["args"].size() == 1 &&
    (!element.contains("keywords") ||
     (element["keywords"].is_array() && element["keywords"].empty())))
  {
    bool can_fold = true;
    if (!ast_json || !ast_json->contains("body"))
      can_fold = false;

    nlohmann::json func_node;
    if (can_fold)
    {
      func_node = find_function((*ast_json)["body"], "parse_nested_parens");
      if (func_node.empty())
        can_fold = false;
    }

    auto returns_list = [](const nlohmann::json &ret) -> bool {
      if (ret.is_null())
        return false;
      if (
        ret.contains("_type") && ret["_type"] == "Subscript" &&
        ret.contains("value") && ret["value"].contains("id"))
      {
        const std::string &container = ret["value"]["id"];
        return container == "List" || container == "list";
      }
      if (ret.contains("id"))
      {
        const std::string &name = ret["id"];
        return name == "List" || name == "list";
      }
      return false;
    };

    if (
      !func_node.empty() &&
      (!func_node.contains("returns") || !returns_list(func_node["returns"])))
      can_fold = false;

    const auto &arg0 = element["args"][0];
    if (
      can_fold && arg0.contains("_type") && arg0["_type"] == "Constant" &&
      arg0.contains("value") && arg0["value"].is_string())
    {
      const std::string input = arg0["value"].get<std::string>();
      std::vector<long long> out;
      std::string token;
      auto flush_token = [&](const std::string &tok) {
        if (tok.empty())
          return;
        long long depth = 0;
        long long max_depth = 0;
        for (char c : tok)
        {
          if (c == '(')
          {
            ++depth;
            if (depth > max_depth)
              max_depth = depth;
          }
          else
          {
            --depth;
          }
        }
        out.push_back(max_depth);
      };
      for (char c : input)
      {
        if (c == ' ')
        {
          flush_token(token);
          token.clear();
        }
        else
        {
          token.push_back(c);
        }
      }
      flush_token(token);

      nlohmann::json list_node;
      list_node["_type"] = "List";
      list_node["elts"] = nlohmann::json::array();
      for (long long v : out)
      {
        nlohmann::json elt;
        elt["_type"] = "Constant";
        elt["value"] = v;
        copy_location_fields_from_decl(element, elt);
        list_node["elts"].push_back(elt);
      }
      copy_location_fields_from_decl(element, list_node);
      python_list list(*this, list_node);
      return list.get();
    }
  }

  // Check for forward-referenced constructor calls
  if (type_handler_.is_constructor_call(element))
  {
    code_blockt temp_block;
    process_forward_reference(element["func"], temp_block);
  }

  // Handle indirect calls through subscript (e.g., {'+': lambda: 1.0}[x]())
  if (element["func"]["_type"] == "Subscript")
  {
    const nlohmann::json &func_node = element["func"];
    exprt container = get_expr(func_node["value"]);

    exprt func_ptr;
    if (
      container.type().is_struct() &&
      dict_handler_->is_dict_type(container.type()))
      func_ptr = dict_handler_->handle_dict_subscript(
        container, func_node["slice"], gen_pointer_type(code_typet()));
    else
      func_ptr = get_expr(element["func"]);

    // Determine return type: prefer current function's declared return type,
    // then try to infer from any lambda body in the dict literal.
    typet ret_type = current_func_return_type_;
    if (ret_type.is_empty() && func_node["value"]["_type"] == "Dict")
    {
      for (const auto &val : func_node["value"]["values"])
      {
        if (val["_type"] == "Lambda" && val.contains("body"))
        {
          ret_type = lambda_handler_->infer_lambda_return_type(val["body"]);
          break;
        }
      }
    }
    if (ret_type.is_empty())
      ret_type = double_type();

    side_effect_expr_function_callt call;
    call.location() = get_location_from_decl(element);
    call.function() = func_ptr;
    call.type() = ret_type;
    if (element.contains("args"))
      for (const auto &arg : element["args"])
        call.arguments().push_back(get_expr(arg));

    return call;
  }

  // Handle indirect calls through function pointer variables
  if (element["func"]["_type"] == "Name")
  {
    std::string func_name = element["func"]["id"].get<std::string>();

    // Try to find as a variable first
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(func_name);
    symbolt *var_symbol = find_symbol(var_sid.to_string());

    if (var_symbol && var_symbol->get_type().is_pointer())
    {
      // This is an indirect call through function pointer
      side_effect_expr_function_callt call;
      call.location() = get_location_from_decl(element);

      // The function pointer itself, not dereferenced.
      // For Any-typed (void*) parameters, cast to a generic function pointer
      // so that the adjuster can dereference it to a code type (it calls
      // to_code_type on the dereferenced subtype, which would fail on void).
      exprt func_ptr_expr = symbol_expr(*var_symbol);
      if (var_symbol->get_type() == any_type())
        func_ptr_expr =
          typecast_exprt(func_ptr_expr, gen_pointer_type(code_typet()));
      call.function() = func_ptr_expr;

      // Resolve return type from the concrete target function stored in
      // the symbol's value (address_of(func)), because gen_pointer_type
      // does not preserve the full code_typet (return type + arguments).
      bool resolved = false;
      if (
        var_symbol->get_value().is_address_of() &&
        !var_symbol->get_value().operands().empty() &&
        var_symbol->get_value().operands()[0].is_symbol())
      {
        const symbolt *target_func = symbol_table_.find_symbol(
          var_symbol->get_value().operands()[0].identifier());
        if (target_func && target_func->get_type().is_code())
        {
          const code_typet &func_type = to_code_type(target_func->get_type());
          call.type() = func_type.return_type();
          resolved = true;
        }
      }

      // Try to get return type from the pointer's subtype
      if (!resolved && var_symbol->get_type().subtype().is_code())
      {
        const code_typet &func_type =
          to_code_type(var_symbol->get_type().subtype());
        call.type() = func_type.return_type();
        resolved = true;
      }

      // Fallback for Any-typed (void*) function parameters: use any_type so
      // the indirect call expression has a well-formed type.
      if (!resolved)
        call.type() = any_type();

      // Process arguments
      if (element.contains("args"))
      {
        for (const auto &arg_element : element["args"])
        {
          exprt arg_expr = get_expr(arg_element);
          // A function name used as an argument decays to a function pointer.
          if (arg_expr.type().is_code() && arg_expr.is_symbol())
            arg_expr = address_of_exprt(arg_expr);
          call.arguments().push_back(arg_expr);
        }
      }

      return call;
    }
  }

  // Handle empty set() creation
  if (
    element["func"]["_type"] == "Name" && element["func"]["id"] == "set" &&
    (!element.contains("args") || element["args"].empty()))
  {
    // Create an empty set (modeled as list)
    python_set set_handler(*this, element);
    return set_handler.get_empty_set();
  }

  // TypeVar(...) is only used to build typing aliases and has no runtime
  // effect in the frontend, so model it as an opaque placeholder value.
  if (element["func"]["_type"] == "Name" && element["func"]["id"] == "TypeVar")
  {
    return gen_zero(any_type());
  }

  const std::string function = config.options.get_option("function");
  // To verify a specific function, it is necessary to load the definitions of functions it calls.
  if (!function.empty() && !is_loading_models)
  {
    std::string func_name("");
    if (element["func"]["_type"] == "Name")
      func_name = element["func"]["id"];
    else if (element["func"]["_type"] == "Attribute")
      func_name = element["func"]["attr"];

    if (
      !type_utils::is_builtin_type(func_name) &&
      !type_utils::is_consensus_type(func_name) &&
      !type_utils::is_consensus_func(func_name) &&
      !type_utils::is_python_model_func(func_name) &&
      !is_class(func_name, *ast_json))
    {
      const auto &func_node = find_function((*ast_json)["body"], func_name);
      assert(!func_node.empty());
      get_function_definition(func_node);
    }
  }

  // Compile-time evaluation: if the function is user-defined and all
  // arguments are constants, try to evaluate the call entirely at
  // conversion time, eliminating loops from the GOTO program.
  if (
    element["func"]["_type"] == "Name" && ast_json &&
    element.contains("args") &&
    (!element.contains("keywords") || element["keywords"].empty()))
  {
    const std::string &callee = element["func"]["id"].get<std::string>();

    // Check if the callee is shadowed by a local FunctionDef inside
    // the current enclosing function.  Consteval only knows about
    // top-level definitions, so folding a shadowed name would resolve
    // to the wrong function.
    bool locally_shadowed = false;
    if (!current_func_name_.empty())
    {
      // Walk the function nesting path (split on "@F@").
      // Use const ref so the non-throwing find_function overload
      // (returns empty JSON on miss) is selected instead of the
      // mutable-ref overload that throws.
      const nlohmann::json &ast_body = (*ast_json)["body"];
      nlohmann::json cur_body = ast_body;
      std::string remaining = current_func_name_;
      while (!remaining.empty() && !locally_shadowed)
      {
        std::string part;
        auto sep = remaining.find("@F@");
        if (sep != std::string::npos)
        {
          part = remaining.substr(0, sep);
          remaining = remaining.substr(sep + 3);
        }
        else
        {
          part = remaining;
          remaining.clear();
        }
        auto fn =
          find_function(static_cast<const nlohmann::json &>(cur_body), part);
        if (fn.empty() || !fn.contains("body") || !fn["body"].is_array())
          break;
        for (const auto &stmt : fn["body"])
        {
          if (
            stmt.contains("_type") && stmt["_type"] == "FunctionDef" &&
            stmt.contains("name") && stmt["name"] == callee)
          {
            locally_shadowed = true;
            break;
          }
        }
        cur_body = fn["body"];
      }
    }

    // Skip builtins / models — only try user-defined functions
    if (
      !locally_shadowed && !type_utils::is_builtin_type(callee) &&
      !type_utils::is_python_model_func(callee) &&
      !find_function((*ast_json)["body"], callee).empty())
    {
      // Collect constant arguments
      bool all_const = true;
      std::vector<PyConstValue> const_args;
      for (const auto &arg_node : element["args"])
      {
        // Bignum literal (issue #4642): the constant carries `_bigint` and a
        // null `value`. Refuse to fold so the call falls back through to the
        // normal builder, whose recursion into get_literal raises the
        // overflow diagnostic. Without this guard the null `value` is taken
        // as Python None below and the bignum is silently consumed.
        const bool arg_is_bigint =
          arg_node["_type"] == "Constant" && arg_node.contains("_bigint");
        const bool usub_operand_is_bigint =
          arg_node["_type"] == "UnaryOp" && arg_node["op"]["_type"] == "USub" &&
          arg_node["operand"]["_type"] == "Constant" &&
          arg_node["operand"].contains("_bigint");
        if (arg_is_bigint || usub_operand_is_bigint)
        {
          all_const = false;
          break;
        }
        if (arg_node["_type"] == "Constant" && arg_node["value"].is_string())
        {
          const_args.push_back(
            PyConstValue::make_string(arg_node["value"].get<std::string>()));
        }
        else if (
          arg_node["_type"] == "Constant" &&
          arg_node["value"].is_number_integer())
        {
          const_args.push_back(
            PyConstValue::make_int(arg_node["value"].get<long long>()));
        }
        else if (
          arg_node["_type"] == "Constant" &&
          arg_node["value"].is_number_float())
        {
          const_args.push_back(
            PyConstValue::make_float(arg_node["value"].get<double>()));
        }
        else if (
          arg_node["_type"] == "Constant" && arg_node["value"].is_boolean())
        {
          const_args.push_back(
            PyConstValue::make_bool(arg_node["value"].get<bool>()));
        }
        else if (arg_node["_type"] == "Constant" && arg_node["value"].is_null())
        {
          const_args.push_back(PyConstValue::make_none());
        }
        else if (
          arg_node["_type"] == "UnaryOp" && arg_node["op"]["_type"] == "USub" &&
          arg_node["operand"]["_type"] == "Constant" &&
          arg_node["operand"]["value"].is_number_integer())
        {
          const_args.push_back(PyConstValue::make_int(
            -arg_node["operand"]["value"].get<long long>()));
        }
        else if (
          arg_node["_type"] == "UnaryOp" && arg_node["op"]["_type"] == "USub" &&
          arg_node["operand"]["_type"] == "Constant" &&
          arg_node["operand"]["value"].is_number_float())
        {
          const_args.push_back(PyConstValue::make_float(
            -arg_node["operand"]["value"].get<double>()));
        }
        else
        {
          all_const = false;
          break;
        }
      }

      if (all_const)
      {
        python_consteval evaluator(*ast_json);
        auto result = evaluator.try_eval_call(callee, const_args);
        if (result.has_value())
        {
          if (result->kind == PyConstValue::STRING)
            return string_builder_->build_string_literal(result->string_val);
          if (result->kind == PyConstValue::INT)
            return from_integer(result->int_val, long_long_int_type());
          if (result->kind == PyConstValue::BOOL)
            // V.3: build the folded bool constant in IREP2.
            return migrate_expr_back(
              result->bool_val ? gen_true_expr() : gen_false_expr());
          // NONE and FLOAT fall through to normal call
        }
      }
    }
  }

  function_call_builder call_builder(*this, element);
  exprt call_expr = call_builder.build();

  // Convert boolean-returning function calls to side-effect expressions when used
  // in expression contexts (e.g., logical operations). This prevents GOTO generation
  // failures where code statements appear in boolean expression operands.
  if (
    call_expr.is_code() && call_expr.statement() == "function_call" &&
    is_converting_rhs)
  {
    const code_function_callt &code_call =
      to_code_function_call(to_code(call_expr));
    const typet &return_type = code_call.type();

    if (return_type.is_bool())
    {
      side_effect_expr_function_callt side_effect_call;
      side_effect_call.function() = code_call.function();
      side_effect_call.arguments() = code_call.arguments();
      side_effect_call.type() = return_type;
      side_effect_call.location() = code_call.location();

      call_expr = side_effect_call;
    }
  }

  auto handle_keywords = [&](exprt &call_expr) {
    if (!element.contains("keywords") || element["keywords"].empty())
      return;

    const exprt &func =
      call_expr.operands().size() > 1 ? call_expr.operands()[1] : exprt();

    if (!func.is_symbol())
      return;

    const symbolt *func_symbol = symbol_table_.find_symbol(func.identifier());
    if (!func_symbol || !func_symbol->get_type().is_code())
      return;

    const code_typet &func_type = to_code_type(func_symbol->get_type());
    const code_typet::argumentst &params = func_type.arguments();

    code_function_callt &call = static_cast<code_function_callt &>(call_expr);
    auto &args = call.arguments();

    size_t positional_count =
      element.contains("args") && element["args"].is_array()
        ? element["args"].size()
        : 0;

    std::map<std::string, size_t> param_positions;
    for (size_t i = 0; i < params.size(); ++i)
    {
      std::string param_name = params[i].get_base_name().as_string();
      assert(!param_name.empty());
      param_positions[param_name] = i;
    }

    if (args.size() < params.size())
      args.resize(params.size(), exprt());

    for (const auto &kw : element["keywords"])
    {
      std::string arg_name = kw["arg"].get<std::string>();

      auto it = param_positions.find(arg_name);
      if (it == param_positions.end())
      {
        // For user-defined functions, unknown kwargs are a TypeError.
        // For builtins/models (e.g. sorted(key=...), max(key=...)), silently skip.
        if (search_function_in_ast(*ast_json, func_symbol->name.as_string()))
          throw std::runtime_error(
            "Unknown keyword argument: " + arg_name + " in function " +
            func_symbol->name.as_string());
        continue;
      }

      exprt arg_expr = get_expr(kw["value"]);

      // Convert array to pointer to match parameter type
      const typet &param_type = params[it->second].type();
      if (arg_expr.type().is_array() && param_type.is_pointer())
        arg_expr = string_handler_.get_array_base_address(arg_expr);

      args[it->second] = arg_expr;
    }

    // we need to check if the argument is provided despite being optional
    auto is_optional_type = [&](const typet &param_type) {
      if (!param_type.is_struct())
        return false;
      const struct_typet &struct_type = to_struct_type(param_type);
      const std::string &tag = struct_type.tag().as_string();
      return tag.starts_with("tag-Optional_");
    };

    std::vector<size_t> missing_required;
    std::vector<bool> provided(params.size(), false);

    size_t bound_params = 0;
    if (!params.empty())
    {
      const std::string &first_param_name =
        params[0].get_base_name().as_string();
      if (first_param_name == "self" || first_param_name == "cls")
        bound_params = 1;
    }

    for (size_t i = 0; i < bound_params && i < provided.size(); ++i)
      provided[i] = true;

    for (size_t i = 0; i < positional_count; ++i)
    {
      size_t param_idx = bound_params + i;
      if (param_idx < provided.size())
        provided[param_idx] = true;
    }

    for (const auto &entry : param_positions)
    {
      size_t index = entry.second;
      if (
        index < provided.size() &&
        !(args[index].is_nil() || args[index].id().empty()))
        provided[index] = true;
    }

    // check if any argument is missing
    for (size_t i = 0; i < params.size(); ++i)
    {
      if (provided[i])
        continue;

      bool has_default = params[i].has_default_value();
      bool optional_param = is_optional_type(params[i].type());

      if (!has_default && !optional_param)
      {
        missing_required.push_back(i); // add the index of the missing argument
      }
    }

    if (!missing_required.empty())
    {
      std::vector<std::string> missing_names;
      missing_names.reserve(missing_required.size());
      for (size_t idx : missing_required)
        missing_names.push_back(params[idx].get_base_name().as_string());

      std::ostringstream msg;
      if (missing_names.size() == 1)
      {
        msg << "TypeError: " << func_symbol->name.as_string()
            << "() missing 1 required positional argument: '"
            << missing_names.front() << "'";
      }
      else
      {
        msg << "TypeError: " << func_symbol->name.as_string() << "() missing "
            << missing_names.size() << " required positional arguments: ";
        for (size_t i = 0; i < missing_names.size(); ++i)
        {
          msg << "'" << missing_names[i] << "'";
          if (i + 2 < missing_names.size())
            msg << ", ";
          else if (i + 2 == missing_names.size())
            msg << " and ";
        }
      }

      throw std::runtime_error(msg.str());
    }

    // Fill empty arguments with proper Optional values or None for optional parameters
    for (size_t i = 0; i < args.size(); ++i)
    {
      if (args[i].is_nil() || args[i].id().empty())
      {
        const typet &param_type = params[i].type();

        // Check if this is an Optional type (struct with "is_none" field)
        if (is_optional_type(param_type))
        {
          // Create Optional value with is_none=true
          constant_exprt none_expr(none_type());
          none_expr.set_value("NULL");
          args[i] = wrap_in_optional(none_expr, param_type);
        }
        else
        {
          // Non-struct type - use NULL for None
          constant_exprt none_expr(none_type());
          none_expr.set_value("NULL");
          args[i] = none_expr;
        }
      }
    }
  };

  handle_keywords(call_expr);

  // Convert struct arguments to pointers for union-typed parameters
  // This handles both positional and keyword arguments
  if (call_expr.id() == "code" && call_expr.get("statement") == "function_call")
  {
    code_function_callt &call = static_cast<code_function_callt &>(call_expr);
    // Get function symbol to access parameter types
    const exprt &func = call.function();
    if (func.is_symbol())
    {
      const symbolt *func_symbol = symbol_table_.find_symbol(func.identifier());
      if (func_symbol && func_symbol->get_type().is_code())
      {
        const code_typet &func_type = to_code_type(func_symbol->get_type());
        const code_typet::argumentst &params = func_type.arguments();
        auto &args = call.arguments();
        for (size_t i = 0; i < args.size() && i < params.size(); ++i)
        {
          const typet &param_type = params[i].type();
          exprt &arg = args[i];

          // Get the actual type of the argument (resolve symbols)
          typet arg_actual_type = arg.type();
          if (arg.is_symbol())
          {
            const symbolt *arg_symbol =
              symbol_table_.find_symbol(arg.identifier());
            if (arg_symbol)
              arg_actual_type = arg_symbol->get_type();
          }
          // Follow a `symbol_typet` (`tag-Class`) to the struct it names. This
          // must run for ANY argument, not just symbol expressions: a by-value
          // class instance produced by a call (e.g. `use(make())` where
          // `make() -> C`) arrives as a side_effect whose *type* is the
          // unfollowed `tag-C` symbol — without following it, is_struct() below
          // is false and the struct-to-`Class*`-parameter coercion is skipped,
          // so a struct is passed to a pointer parameter (#4558/#4564).
          if (arg_actual_type.id() == "symbol")
            arg_actual_type = ns.follow(arg_actual_type);
          // Handle union types: if param is pointer and arg is struct (or symbol
          // to struct), take address. This is the post-processing pass for
          // general pointer-to-struct coercion.
          // NOTE: function_call_expr.cpp also has an earlier coercion pass that
          // specifically handles char[0]* union parameters (str | T pattern).
          // These two mechanisms are complementary: the pass here handles the
          // general case; the earlier pass handles the specific char[0]* union
          // representation and also materialises non-symbol struct temporaries.
          if (
            param_type.is_pointer() && arg_actual_type.is_struct() &&
            !arg.is_address_of() && !arg_actual_type.is_pointer())
          {
            // Rvalue structs (e.g. the constant `__ESBMC_PySliceObj` produced
            // by `a[i:j]`, or a call returning a by-value class instance)
            // cannot be the operand of `address_of` at the SMT level.
            // Materialise them in a temp symbol first so the address we hand to
            // the callee points at a real lvalue.
            if (!arg.is_symbol())
            {
              assert(
                current_block &&
                "rvalue struct argument requires a statement-level block to "
                "materialise into");
              locationt loc = get_location_from_decl(element);
              symbolt &tmp = create_tmp_symbol(
                element,
                "$struct_arg$",
                arg_actual_type,
                gen_zero(arg_actual_type));
              code_declt tmp_decl(symbol_expr(tmp));
              tmp_decl.location() = loc;
              current_block->copy_to_operands(tmp_decl);
              code_assignt tmp_assign(symbol_expr(tmp), arg);
              tmp_assign.location() = loc;
              current_block->copy_to_operands(tmp_assign);
              arg = symbol_expr(tmp);
            }
            arg = gen_address_of(arg);
          }

          // Propagate instance attributes set on this parameter back to the
          // caller's argument. This models Python's pass-by-object-reference:
          // if o.x = 5 is set inside f(a) via parameter o, then a.x should
          // reflect the instance attribute rather than the class attribute.
          // Note: this relies on the callee body being processed before the
          // call site (true for top-level sequential Python), so that
          // instance_attr_map[param] is already populated here.
          const exprt *arg_sym = &args[i];
          if (arg_sym->is_address_of())
            arg_sym = &arg_sym->op0();
          if (arg_sym->is_symbol())
            copy_instance_attributes(
              params[i].identifier().as_string(),
              arg_sym->identifier().as_string());
        }
      }
    }
  }

  return call_expr;
}
exprt python_converter::store_call_result(
  exprt call_expr,
  const locationt &location,
  const std::string &temp_prefix)
{
  if (!call_expr.is_function_call())
    return call_expr;

  symbolt temp_symbol =
    create_return_temp_variable(call_expr.type(), location, temp_prefix);
  symbol_table_.add(temp_symbol);
  exprt temp_var_expr = symbol_expr(temp_symbol);

  code_declt temp_decl(temp_var_expr);
  temp_decl.location() = location;
  if (!call_expr.type().is_empty())
    call_expr.op0() = temp_var_expr;
  if (current_block)
  {
    current_block->copy_to_operands(temp_decl);
    current_block->copy_to_operands(call_expr);
  }

  return temp_var_expr;
}
exprt python_converter::get_return_from_func(const char *func_symbol_id)
{
  symbolt *func_symbol = symbol_table_.find_symbol(func_symbol_id);
  assert(func_symbol);

  const auto &operands = func_symbol->get_value().operands();

  for (std::vector<exprt>::const_reverse_iterator it = operands.rbegin();
       it != operands.rend();
       ++it)
  {
    const codet &c = to_code(*it);
    if (c.statement() == "return")
    {
      return c;
    }
  }
  return nil_exprt();
}
exprt python_converter::materialize_list_function_call(
  const exprt &expr,
  const nlohmann::json &element,
  codet &target_block)
{
  // Check if this is a function call returning a list
  if (expr.id() != "code" || expr.get("statement") != "function_call")
    return expr;

  const code_function_callt &call = to_code_function_call(to_code(expr));

  // Only handle list-returning functions
  if (call.type() != type_handler_.get_list_type())
    return expr;

  locationt location = get_location_from_decl(element);

  // Create temporary variable for the list
  symbolt &tmp_var_symbol = create_tmp_symbol(
    element, "$iter_temp$", call.type(), gen_zero(call.type()));

  // Declare the temporary
  code_declt tmp_var_decl(symbol_expr(tmp_var_symbol));
  tmp_var_decl.location() = location;
  target_block.copy_to_operands(tmp_var_decl);

  // Create function call with temp as LHS
  code_function_callt new_call;
  new_call.function() = call.function();
  new_call.arguments() = call.arguments();
  new_call.lhs() = symbol_expr(tmp_var_symbol);
  new_call.type() = call.type();
  new_call.location() = location;

  target_block.copy_to_operands(new_call);

  // Return reference to the temp variable
  return symbol_expr(tmp_var_symbol);
}
