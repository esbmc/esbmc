/// \file solidity_convert_modifier.cpp
/// \brief Function and modifier definition conversion for the Solidity frontend.
///
/// Converts Solidity function definitions, modifier definitions, and fallback/
/// receive functions from the solc JSON AST into ESBMC's symbol table and code
/// representation. Handles parameter lists, return parameters, visibility,
/// mutability, this-pointer injection, and modifier invocation inlining.

#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_expr.h>
#include <util/message.h>
#include <fstream>

bool solidity_convertert::get_function_definition(
  const nlohmann::json &ast_node)
{
  // For Solidity rule function-definition:
  // Order matters! do not change!
  // 1. Check fd.isImplicit() --- skipped since it's not applicable to Solidity
  // 2. Check fd.isDefined() and fd.isThisDeclarationADefinition()

  // Check intrinsic functions
  if (check_intrinsic_function(ast_node))
    return false;

  const nlohmann::json *old_functionDecl = current_functionDecl;
  const std::string old_functionName = current_functionName;

  current_functionDecl = &ast_node;

  bool is_ctor = (*current_functionDecl)["name"].get<std::string>() == "" &&
                 (*current_functionDecl).contains("kind") &&
                 (*current_functionDecl)["kind"] == "constructor";

  bool is_receive_fallback =
    (*current_functionDecl)["name"].get<std::string>() == "" &&
    (*current_functionDecl).contains("kind") &&
    ((*current_functionDecl)["kind"] == "receive" ||
     (*current_functionDecl)["kind"] == "fallback");

  std::string c_name;
  get_current_contract_name(ast_node, c_name);

  if (is_ctor)
    // for construcotr
    current_functionName = c_name;
  else if (is_receive_fallback)
    current_functionName = (*current_functionDecl)["kind"].get<std::string>();
  else
    current_functionName = (*current_functionDecl)["name"].get<std::string>();
  assert(!current_functionName.empty());

  // 4. Return type
  code_typet type;
  if (is_ctor)
  {
    typet tmp_rtn_type("constructor");
    type.return_type() = tmp_rtn_type;
    type.set("#member_name", prefix + c_name);
    type.set("#inlined", true);
  }
  else if (ast_node.contains("returnParameters"))
  {
    if (get_type_description(ast_node["returnParameters"], type.return_type()))
      return true;
    //? set member name?
  }
  else
  {
    type.return_type() = empty_typet();
    type.return_type().set("cpp_type", "void");
    type.set("#member_name", prefix + c_name);
  }

  // special handling for tuple:
  // construct a tuple type and a tuple instance
  if (
    get_sol_type(type.return_type()) == SolidityGrammar::SolType::TUPLE_RETURNS)
  {
    exprt dump;
    if (get_tuple_definition(*current_functionDecl))
      return true;
    if (get_tuple_instance(*current_functionDecl, dump))
      return true;
    type.return_type().set("#sol_tuple_id", dump.identifier().as_string());
  }

  // 5. Check fd.isVariadic(), fd.isInlined()
  //  Skipped since Solidity does not support variadic (optional args) or inline function.
  //  Actually "inline" doesn not make sense in Solidity

  // 6. Populate "locationt location_begin"
  locationt location_begin;
  get_location_from_node(ast_node, location_begin);

  // 7. Populate "std::string id, name"
  std::string name, id;
  get_function_definition_name(ast_node, name, id);
  assert(!name.empty());
  log_debug(
    "solidity",
    "@@@ Parsing function {} in contract {}",
    id.c_str(),
    current_baseContractName);

  if (context.find_symbol(id) != nullptr)
  {
    current_functionDecl = old_functionDecl;
    current_functionName = old_functionName;
    log_debug(
      "solidity",
      "@@@ Already parsed function {} in contract {}",
      id.c_str(),
      current_baseContractName);
    return false;
  }

  // 8. populate "std::string debug_modulename"
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  // 9. Populate "symbol.static_lifetime", "symbol.is_extern" and "symbol.file_local"
  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, type, name, id, location_begin);

  symbol.lvalue = true;
  symbol.is_extern =
    false; // TODO: hard coded for now, may need to change later
  symbol.file_local = true;

  // 10. Add symbol into the context
  symbolt &added_symbol = *move_symbol_to_context(symbol);
  // 11. Convert parameters, if no parameter, assume ellipis
  //  - Convert params before body as they may get referred by the statement in the body

  // 11.1 add this pointer as the first param
  bool is_event_err_lib =
    ast_node.contains("nodeType") &&
    (ast_node["nodeType"] == "EventDefinition" ||
     ast_node["nodeType"] == "ErrorDefinition" ||
     SolidityGrammar::is_sol_library_function(ast_node["id"].get<int>()));
  bool is_free_function = ast_node.contains("kind") &&
                          ast_node["kind"].get<std::string>() == "freeFunction";
  if (!is_event_err_lib && !is_free_function)
    get_function_this_pointer_param(
      c_name, id, debug_modulename, location_begin, type);

  // 11.2 parse other params
  SolidityGrammar::ParameterListT params =
    SolidityGrammar::get_parameter_list_t(ast_node["parameters"]);
  if (params != SolidityGrammar::ParameterListT::EMPTY)
  {
    // convert parameters if the function has them
    // update the typet, since typet contains parameter annotations
    for (const auto &decl : ast_node["parameters"]["parameters"].items())
    {
      log_debug(
        "solidity",
        "  @@@ parsing function {}'s parameters",
        current_functionName);
      const nlohmann::json &func_param_decl = decl.value();

      code_typet::argumentt param;
      if (get_function_params(func_param_decl, c_name, param))
        return true;

      type.arguments().push_back(param);
    }
  }

  added_symbol.type = type;

  // 11.3 Declare named return parameters as local variables.
  // Solidity allows `returns (uint result) { result = 42; }` where `result`
  // is both a local variable and the implicit return value. We must emit
  // DECL + zero-init for each named return parameter so that assignments to
  // them work correctly in symex, and append an implicit return at the end.
  std::vector<exprt> named_ret_decls;
  std::vector<exprt> named_ret_syms;
  bool has_named_returns = false;
  if (
    !is_ctor && ast_node.contains("returnParameters") &&
    ast_node["returnParameters"].contains("parameters") &&
    get_sol_type(type.return_type()) != SolidityGrammar::SolType::TUPLE_RETURNS)
  {
    for (const auto &rparam : ast_node["returnParameters"]["parameters"])
    {
      std::string rname = rparam["name"].get<std::string>();
      if (rname.empty())
        continue; // unnamed return parameter — skip

      has_named_returns = true;
      exprt var_decl;
      if (get_var_decl(rparam, var_decl))
        return true;
      named_ret_decls.push_back(var_decl);

      // Retrieve the symbol we just created
      std::string rvar_name, rvar_id;
      if (get_var_decl_name(rparam, rvar_name, rvar_id))
        return true;
      const symbolt *sym = context.find_symbol(rvar_id);
      assert(sym != nullptr);
      named_ret_syms.push_back(symbol_expr(*sym));
    }
  }

  // 12. Convert body and embed the body into the same symbol
  // skip for 'unimplemented' functions which has no body,
  // e.g. asbstract/interface, the symbol value would be left as unset

  exprt body_exprt = code_blockt();
  if (
    ast_node.contains("body") ||
    (ast_node.contains("implemented") && ast_node["implemented"] == true))
  {
    log_debug(
      "solidity", "\t parsing function {}'s body", current_functionName);
    bool add_reentry = is_reentry_check && !is_event_err_lib && !is_ctor;

    if (has_modifier_invocation(ast_node))
    {
      // func() modf_1 modf_2
      // => func() => func_modf1() => func_modf2()
      if (get_func_modifier(
            ast_node, c_name, name, id, add_reentry, body_exprt))
        return true;
    }
    else
    {
      if (get_block(ast_node["body"], body_exprt))
        return true;
      if (add_reentry)
      {
        if (add_reentry_check(c_name, location_begin, body_exprt))
          return true;
      }
    }

    // Prepend named return parameter declarations at the start of the body
    if (has_named_returns && body_exprt.is_code())
    {
      code_blockt new_body;
      for (auto &decl : named_ret_decls)
        new_body.copy_to_operands(decl);
      for (auto &op : body_exprt.operands())
        new_body.copy_to_operands(op);

      // Append implicit return of the named return variable if the body
      // does not already end with an explicit return statement.
      bool has_explicit_return = false;
      if (!new_body.operands().empty())
      {
        const exprt &last = new_body.operands().back();
        if (last.is_code() && last.statement() == "return")
          has_explicit_return = true;
      }
      if (!has_explicit_return && named_ret_syms.size() == 1)
      {
        code_returnt implicit_ret;
        implicit_ret.return_value() = named_ret_syms[0];
        implicit_ret.location() = location_begin;
        new_body.copy_to_operands(implicit_ret);
      }

      body_exprt = new_body;
    }
  }

  // For library functions with storage parameters, append a copy-out
  // assignment to a global $out bridge at the end of the body. The
  // matching call-site code in solidity_convert_expr.cpp reads from this
  // bridge after the call to propagate modifications back to the caller.
  if (
    is_event_err_lib && ast_node.contains("parameters") &&
    ast_node["parameters"].contains("parameters"))
  {
    for (const auto &p : ast_node["parameters"]["parameters"])
    {
      if (!p.contains("storageLocation") || p["storageLocation"] != "storage")
        continue;

      std::string p_name = p["name"].get<std::string>();
      std::string p_sym_id =
        get_library_param_id(c_name, name, p_name, p["id"].get<int>());
      std::string out_id = p_sym_id + "$out";

      const symbolt *param_sym = context.find_symbol(p_sym_id);
      if (!param_sym)
      {
        log_error("storage-ref bridge: param symbol {} not found", p_sym_id);
        return true;
      }

      if (context.find_symbol(out_id) == nullptr)
      {
        symbolt out_sym;
        get_default_symbol(
          out_sym,
          debug_modulename,
          param_sym->type,
          p_name + "$out",
          out_id,
          location_begin);
        out_sym.static_lifetime = true;
        out_sym.lvalue = true;
        out_sym.value = gen_zero(get_complete_type(param_sym->type, ns), true);
        move_symbol_to_context(out_sym);
      }

      body_exprt.copy_to_operands(code_assignt(
        symbol_expr(*context.find_symbol(out_id)), symbol_expr(*param_sym)));
    }
  }

  added_symbol.value = body_exprt;

  // 13. Restore current_functionDecl
  log_debug("solidity", "@@@ Finish parsing function {}", current_functionName);
  current_functionDecl =
    old_functionDecl; // for __ESBMC_assume, old_functionDecl == null
  current_functionName = old_functionName;
  return false;
}

bool solidity_convertert::delete_modifier_json(
  const std::string &cname,
  const std::string &fname,
  nlohmann::json *&modifier_def)
{
  if (!src_ast_json.contains("nodes") || !src_ast_json["nodes"].is_array())
    return true;

  for (auto &node : src_ast_json["nodes"])
  {
    if (
      node.contains("name") && node["name"] == cname &&
      node.contains("nodeType") && node["nodeType"] == "ContractDefinition" &&
      node.contains("nodes") && node["nodes"].is_array())
    {
      nlohmann::json &contract_nodes = node["nodes"];

      for (auto it = contract_nodes.begin(); it != contract_nodes.end(); ++it)
      {
        if (
          it->contains("nodeType") &&
          (*it)["nodeType"] == "FunctionDefinition" && it->contains("name") &&
          (*it)["name"] == fname)
        {
          contract_nodes.erase(it);
          modifier_def = nullptr;
          return false;
        }
      }
    }
  }
  return true;
}

bool solidity_convertert::insert_modifier_json(
  const nlohmann::json &ast_node,
  const std::string &cname,
  const std::string &fname,
  nlohmann::json *&modifier_def)
{
  log_debug("solidity", "\tinsert modifier json {}", fname);
  nlohmann::json &nodes = src_ast_json["nodes"];
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if (
      (*itr).contains("name") && (*itr)["name"].get<std::string>() == cname &&
      (*itr)["nodeType"] == "ContractDefinition")
    {
      nlohmann::json &contract_nodes = (*itr)["nodes"];

      // check if we already inserted
      for (auto &func_node : contract_nodes)
      {
        if (
          func_node.contains("nodeType") &&
          func_node["nodeType"] == "FunctionDefinition" &&
          func_node.contains("name") && func_node["name"] == fname)
        {
          modifier_def = &func_node;
          return false;
        }
      }

      const nlohmann::json returnParameters = ast_node["returnParameters"];
      const nlohmann::json src = ast_node["src"];
      /*
        append below node to the contract_nodes
        {
          nodeType: FunctionDefinition
          id: 0
          name: fname
          returnParameters: returnParameters
          src: src
        }
      */
      nlohmann::json new_function = {
        {"nodeType", "FunctionDefinition"},
        {"id", 0},
        {"name", fname},
        {"kind", "function"},
        {"implemented", true},
        {"returnParameters", returnParameters},
        {"src", src}};

      contract_nodes.push_back(new_function);
      modifier_def = &contract_nodes.back();
      return false;
    }
  }

  return true; // unexpected error
}

void solidity_convertert::get_modifier_function_name(
  const std::string &cname,
  const std::string &mod_name,
  const std::string &func_name,
  std::string &name,
  std::string &id)
{
  name = func_name + "_" + mod_name;
  id = "sol:@C@" + cname + "@F@" + name + "#0";
}

bool solidity_convertert::has_modifier_invocation(
  const nlohmann::json &ast_node)
{
  // check if there is any modifier invocation (not constructor calls)
  if (!ast_node.contains("modifiers") || ast_node["modifiers"].empty())
  {
    return false;
  }

  nlohmann::json modifiers = nlohmann::json::array();
  for (const auto &m : ast_node["modifiers"])
  {
    assert(m.contains("kind"));
    if (m.contains("kind") && m["kind"] == "modifierInvocation")
    {
      modifiers.push_back(m);
    }
  }
  // if there is no modifier invocation, return
  if (modifiers.size() == 0)
    return false;

  return true;
}

bool solidity_convertert::add_reentry_check(
  const std::string &c_name,
  const locationt &loc,
  exprt &body_exprt)
{
  // we should only add this to the contract's functions
  // rather than interface and library's functions,
  // or contract's errors, events and ctor
  //TODO: detect is_library_function

  // add a global mutex checker _ESBMC_check_reentrancy() in the front
  side_effect_expr_function_callt call;
  get_library_function_call_no_args(
    "_ESBMC_check_reentrancy",
    "c:@F@_ESBMC_check_reentrancy",
    empty_typet(),
    loc,
    call);

  exprt this_expr;
  if (get_func_decl_this_ref(*current_functionDecl, this_expr))
    return true;

  exprt arg;
  get_contract_mutex_expr(c_name, this_expr, arg);
  call.arguments().push_back(arg);

  convert_expression_to_code(call);
  // Insert after the last front requirement (__ESBMC_assume) statement,
  // as the function may only be re-entered once the requirements are fulfilled.
  auto &ops = body_exprt.operands();
  for (auto it = ops.begin(); it != ops.end(); ++it)
  {
    if (
      it->op0().id() == "sideeffect" &&
      it->op0().op0().name() == "__ESBMC_assume")
      continue;

    ops.insert(it, call);
    break;
  }
  return false;
}

// parse the modifiers, this could be:
// 1. merge code into function body
// 2. construct an auxiliary function, move the body to it, and call it
bool solidity_convertert::get_func_modifier(
  const nlohmann::json &ast_node,
  const std::string &c_name,
  const std::string &f_name,
  const std::string &f_id,
  const bool add_reentry,
  exprt &body_exprt)
{
  log_debug("solidity", "parsing function modifiers");
  nlohmann::json modifiers = nlohmann::json::array();
  for (const auto &m : ast_node["modifiers"])
  {
    if (m.contains("kind") && m["kind"] == "modifierInvocation")
      modifiers.push_back(m);
  }
  // rebegin the function body
  for (auto it = modifiers.rbegin(); it != modifiers.rend(); ++it)
  {
    int modifier_id = (*it)["modifierName"]["referencedDeclaration"];
    // we cannot use reference here, as the src_ast_json got inserted/deleted later
    const nlohmann::json mod_def = find_decl_ref(modifier_id);
    assert(!mod_def.is_null());
    assert(!mod_def.empty());

    std::string func_name = f_name;
    std::string mod_name = mod_def["name"];
    std::string aux_func_name, aux_func_id;
    get_modifier_function_name(
      c_name, mod_name, func_name, aux_func_name, aux_func_id);

    nlohmann::json *modifier_func = nullptr;
    if (insert_modifier_json(ast_node, c_name, aux_func_name, modifier_func))
      return true;
    assert(modifier_func != nullptr);
    auto old_decl = current_functionDecl;
    auto old_name = current_functionName;
    current_functionDecl = modifier_func;
    current_functionName = aux_func_name;

    symbolt added_sym;
    locationt loc;
    get_location_from_node(ast_node, loc);
    std::string debug_mode = get_modulename_from_path(absolute_path);
    code_typet aux_type;

    bool has_return = ast_node.contains("returnParameters") &&
                      !ast_node["returnParameters"]["parameters"].empty();

    if (has_return)
    {
      // return func_modifier();
      if (get_type_description(
            ast_node["returnParameters"], aux_type.return_type()))
        return true;
    }
    else
    {
      aux_type.return_type() = empty_typet();
      aux_type.set("cpp_type", "void");
    }

    get_default_symbol(
      added_sym, debug_mode, aux_type, aux_func_name, aux_func_id, loc);
    added_sym.lvalue = true;
    added_sym.file_local = true;

    // move the symbol to the context
    symbolt &a_sym = *move_symbol_to_context(added_sym);
    get_function_this_pointer_param(
      c_name, aux_func_id, debug_mode, loc, aux_type);
    for (const auto &param : mod_def["parameters"]["parameters"])
    {
      code_typet::argumentt arg;
      if (get_function_params(param, c_name, arg))
        return true;
      aux_type.arguments().push_back(arg);
    }
    a_sym.type = aux_type;
    move_builtin_to_contract(c_name, symbol_expr(a_sym), "internal", true);

    // same as origin function body
    if (body_exprt.operands().empty())
    {
      // modify the src_ast_json: insert the func node
      // nodeType: esbmcModfunction
      if (get_block(ast_node["body"], body_exprt))
        return true;
      if (add_reentry)
      {
        if (add_reentry_check(c_name, loc, body_exprt))
          return true;
      }
    }
    else
    {
      // this can only be a function call
      assert(body_exprt.operands().size() == 1);
    }

    // get func body
    code_blockt mod_body;
    if (get_block(mod_def["body"], mod_body))
      return true;

    // merge the body
    auto stmt = mod_body.operands().begin();
    while (stmt != mod_body.operands().end())
    {
      if (stmt->get_bool("#is_modifier_placeholder"))
      {
        stmt = mod_body.operands().erase(stmt);
        stmt = mod_body.operands().insert(
          stmt, body_exprt.operands().begin(), body_exprt.operands().end());
      }
      else
        ++stmt;
    }

    if (has_return)
    {
      // int ret
      // ...
      // ret = func_modifier();
      // ...
      // return ret // insert in the end

      // 1. add the aux decl in the front
      std::string ret_name, ret_id;
      get_aux_var(ret_name, ret_id);
      symbolt ret_symbol;
      get_default_symbol(
        ret_symbol, debug_mode, aux_type.return_type(), ret_name, ret_id, loc);
      // move the symbol to the context
      symbolt &ret_sym = *move_symbol_to_context(ret_symbol);
      code_declt ret_decl(symbol_expr(ret_sym));
      //ret_sym.value = func_modifier;
      // ret_decl.operands().push_back(func_modifier);
      mod_body.operands().insert(mod_body.operands().begin(), ret_decl);

      // 2. replace every "return x" to "aux_var = x"
      for (auto op = mod_body.operands().begin();
           op != mod_body.operands().end();
           ++op)
      {
        if (op->is_code() && op->statement() == "return")
        {
          // make assignment
          exprt rhs = op->op0();
          code_assignt assign(symbol_expr(ret_sym), rhs);
          *op = assign;
        }
      }

      // 3. insert "return aux_var" in the end
      code_returnt return_expr = code_returnt();
      return_expr.return_value() = symbol_expr(ret_sym);
      mod_body.move_to_operands(return_expr);
    }

    a_sym.value = mod_body;

    // reset
    current_functionDecl = old_decl;
    current_functionName = old_name;
    if (delete_modifier_json(c_name, aux_func_name, modifier_func))
      return true;
    assert(modifier_func == nullptr);

    // construct the function call
    side_effect_expr_function_callt func_modifier;
    func_modifier.function() = symbol_expr(a_sym);

    exprt this_ptr;
    auto next_it = std::next(it);
    if (next_it != modifiers.rend())
    {
      int next_modifier_id =
        (*next_it)["modifierName"]["referencedDeclaration"];
      const nlohmann::json &next_mod_def = find_decl_ref(next_modifier_id);

      std::string next_mod_name = next_mod_def["name"];
      std::string next_aux_func_name, next_aux_func_id;
      get_modifier_function_name(
        c_name, next_mod_name, f_name, next_aux_func_name, next_aux_func_id);

      if (get_func_decl_this_ref(c_name, next_aux_func_id, this_ptr))
        return true;
      func_modifier.arguments().push_back(this_ptr);

      if (insert_modifier_json(
            ast_node, c_name, next_aux_func_name, modifier_func))
        return true;
      assert(modifier_func != nullptr);
      auto old_decl = current_functionDecl;
      auto old_name = current_functionName;
      current_functionDecl = modifier_func;
      current_functionName = next_aux_func_name;

      for (const auto &arg_json : (*it)["arguments"])
      {
        exprt arg_expr;
        if (get_expr(arg_json, arg_json["typeDescriptions"], arg_expr))
          return true;
        func_modifier.arguments().push_back(arg_expr);
      }

      // reset
      current_functionDecl = old_decl;
      current_functionName = old_name;
    }
    else
    {
      // original
      if (get_func_decl_this_ref(c_name, f_id, this_ptr))
        return true;
      func_modifier.arguments().push_back(this_ptr);

      for (const auto &arg_json : (*it)["arguments"])
      {
        exprt arg_expr;
        if (get_expr(arg_json, arg_json["typeDescriptions"], arg_expr))
          return true;
        func_modifier.arguments().push_back(arg_expr);
      }
    }

    code_blockt _block;
    if (has_return)
    {
      // return func_modifier();
      code_returnt return_expr = code_returnt();
      return_expr.return_value() = func_modifier;
      _block.move_to_operands(return_expr);
      body_exprt = _block;
    }
    else
    {
      convert_expression_to_code(func_modifier);
      _block.move_to_operands(func_modifier);
      body_exprt = _block;
    }
  }

  return false;
}
