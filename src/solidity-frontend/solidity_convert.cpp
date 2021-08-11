#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <iomanip>

solidity_convertert::solidity_convertert(contextt &_context,
  nlohmann::json &_ast_json, const std::string &_sol_func, const messaget &msg):
  context(_context),
  ns(context),
  ast_json(_ast_json),
  sol_func(_sol_func),
  msg(msg),
  current_scope_var_num(1),
  current_functionDecl(nullptr),
  current_functionName(""),
  global_scope_id(0)
{
}

bool solidity_convertert::convert()
{
  // This function consists of two parts:
  //  1. First, we perform pattern-based verificaiton
  //  2. Then we populate the context with symbols annotated based on the each AST node, and hence prepare for the GOTO conversion.

  if (!ast_json.contains("nodes")) // check json file contains AST nodes as Solidity might change
    assert(!"JSON file does not contain any AST nodes");

  if (!ast_json.contains("absolutePath")) // check json file contains AST nodes as Solidity might change
    assert(!"JSON file does not contain absolutePath");

  absolute_path = ast_json["absolutePath"].get<std::string>();

  // By now the context should have the symbols of all ESBMC's intrinsics and the dummy main
  // We need to convert Solidity AST nodes to the equivalent symbols and add them to the context
  nlohmann::json nodes = ast_json["nodes"];

  bool found_contract_def = false;
  unsigned index = 0;
  nlohmann::json::iterator itr = nodes.begin();
  for (; itr != nodes.end(); ++itr, ++index)
  {
    // ignore the meta information and locate nodes in ContractDefinition
    std::string node_type = (*itr)["nodeType"].get<std::string>();
    if (node_type == "ContractDefinition") // contains AST nodes we need
    {
      global_scope_id = (*itr)["id"];
      found_contract_def = true;
      // pattern-based verification
      assert(itr->contains("nodes"));
      auto pattern_check = std::make_unique<pattern_checker>((*itr)["nodes"], sol_func, msg);
      if(pattern_check->do_pattern_check())
        return true; // 'true' indicates something goes wrong.
    }
  }
  assert(found_contract_def);

  // reasoning-based verification
  //assert(!"Continue with symbol annotations");
  index = 0;
  itr = nodes.begin();
  for (; itr != nodes.end(); ++itr, ++index)
  {
    std::string node_type = (*itr)["nodeType"].get<std::string>();
    if (node_type == "ContractDefinition") // rule source-unit
    {
      if(convert_ast_nodes(*itr))
        return true; // 'true' indicates something goes wrong.
    }
  }

  return false; // 'false' indicates successful completion.
}

bool solidity_convertert::convert_ast_nodes(const nlohmann::json &contract_def)
{
  unsigned index = 0;
  nlohmann::json ast_nodes = contract_def["nodes"];
  nlohmann::json::iterator itr = ast_nodes.begin();
  for (; itr != ast_nodes.end(); ++itr, ++index)
  {
    nlohmann::json ast_node = *itr;
    std::string node_name = ast_node["name"].get<std::string>();
    std::string node_type = ast_node["nodeType"].get<std::string>();
    printf("@@ Converting node[%u]: name=%s, nodeType=%s ...\n",
        index, node_name.c_str(), node_type.c_str());
    //print_json_array_element(ast_node, node_type, index);
    exprt dummy_decl;
    if(get_decl(ast_node, dummy_decl))
      return true;
  }

  assert(current_functionDecl == nullptr);

  return false;
}

bool solidity_convertert::get_decl(const nlohmann::json &ast_node, exprt &new_expr)
{
  new_expr = code_skipt();

  if (!ast_node.contains("nodeType"))
    assert(!"Missing \'nodeType\' filed in ast_node");

  SolidityGrammar::ContractBodyElementT type = SolidityGrammar::get_contract_body_element_t(ast_node);

  // based on each element as in Solidty grammar "rule contract-body-element"
  switch(type)
  {
    case SolidityGrammar::ContractBodyElementT::StateVarDecl:
    {
      return get_var_decl(ast_node, new_expr); // rule state-variable-declaration
    }
    case SolidityGrammar::ContractBodyElementT::FunctionDef:
    {
      return get_function_definition(ast_node, new_expr); // rule function-definition
    }
    default:
    {
      assert(!"Unimplemented type in rule contract-body-element");
      return true;
    }
  }

  return false;
}

bool solidity_convertert::get_var_decl_stmt(const nlohmann::json &ast_node, exprt &new_expr)
{
  // For rule variable-declaration-statement
  new_expr = code_skipt();

  if (!ast_node.contains("nodeType"))
    assert(!"Missing \'nodeType\' filed in ast_node");

  SolidityGrammar::VarDeclStmtT type = SolidityGrammar::get_var_decl_stmt_t(ast_node);
  printf("	@@@ got Variable-declaration-statement: SolidityGrammar::VarDeclStmtT::%s\n",
      SolidityGrammar::var_decl_statement_to_str(type));

  switch(type)
  {
    case SolidityGrammar::VarDeclStmtT::VariableDecl:
    {
      return get_var_decl(ast_node, new_expr); // rule variable-declaration
    }
    default:
    {
      assert(!"Unimplemented type in rule variable-declaration-statement");
      return true;
    }
  }

  return false;
}

bool solidity_convertert::get_var_decl(const nlohmann::json &ast_node, exprt &new_expr)
{
  // For Solidity rule state-variable-declaration:
  // 1. populate typet
  typet t;
  // VariableDeclaration node contains both "typeName" and "typeDescriptions".
  // However, ExpressionStatement node just contains "typeDescriptions".
  // For consistensy, we use ["typeName"]["typeDescriptions"] as in state-variable-declaration
  // to improve the re-usability of get_type* function.
  if (get_type_description(ast_node["typeName"]["typeDescriptions"], t))
    return true;

  // 2. populate id and name
  std::string name, id;
  if (ast_node["stateVariable"] == true)
    get_state_var_decl_name(ast_node, name, id);
  else
  {
    if(!current_functionDecl)
    {
      assert(!"Error: ESBMC could not find the parent scope for this local variable");
    }
    else
    {
      current_function_name = (*current_functionDecl)["name"];
      get_var_decl_name(ast_node, name, id);
    }
  }

  // 3. populate location
  locationt location_begin;
  get_location_from_decl(ast_node, location_begin);

  // 4. populate debug module name
  std::string debug_modulename = get_modulename_from_path(location_begin.file().as_string());

  // 5. set symbol attributes
  symbolt symbol;
  get_default_symbol(
    symbol,
    debug_modulename,
    t,
    name,
    id,
    location_begin);

  symbol.lvalue = true;
  if (ast_node["stateVariable"] == true)
  {
    // for global variables
    symbol.static_lifetime = true;
    symbol.file_local = false;
  }
  else
  {
    // for local variables
    symbol.static_lifetime = false;
    symbol.file_local = true;
  }
  symbol.is_extern = false;

  // For state var decl, we look for "value".
  // For local var decl, we look for "initialValue"
  bool has_init = ast_node.contains("value") || ast_node.contains("initialValue");
  if(symbol.static_lifetime && !symbol.is_extern && !has_init)
  {
    symbol.value = gen_zero(t, true);
    symbol.value.zero_initializer(true);
  }

  // 6. add symbol into the context
  // just like clang-c-frontend, we have to add the symbol before converting the initial assignment
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // 7. populate init value if there is any
  code_declt decl(symbol_expr(added_symbol));

  if (has_init)
  {
    exprt val;
    if(get_expr(ast_node["initialValue"], val))
      return true;

    solidity_gen_typecast(ns, val, t);

    added_symbol.value = val;
    decl.operands().push_back(val);
  }

  decl.location() = location_begin;

  new_expr = decl;

  return false;
}

bool solidity_convertert::get_function_definition(const nlohmann::json &ast_node, exprt &new_expr)
{
  // For Solidity rule function-definition:
  // Order matters! do not change!
  // 1. Check fd.isImplicit() --- skipped since it's not applicable to Solidity
  // 2. Check fd.isDefined() and fd.isThisDeclarationADefinition()
  if (!ast_node["implemented"]) // TODO: for interface function, it's just a definition. Add something like "&& isInterface_JustDefinition()"
    return false;

  // Check intrinsic functions
  if (check_intrinsic_function(ast_node))
    return false;

  // 3. Set current_scope_var_num, current_functionDecl and old_functionDecl
  current_scope_var_num = 1;
  const nlohmann::json *old_functionDecl = current_functionDecl;
  current_functionDecl = &ast_node;
  current_functionName = ast_node["name"];

  // 4. Return type
  code_typet type;
  if (get_type_description(ast_node["returnParameters"], type.return_type()))
    return true;

  // 5. Check fd.isVariadic(), fd.isInlined()
  //  Skipped since Solidity does not support variadic (optional args) or inline function.
  //  Actually "inline" doesn not make sense in Solidity

  // 6. Populate "locationt location_begin"
  locationt location_begin;
  get_location_from_decl(ast_node, location_begin);

  // 7. Populate "std::string id, name"
  std::string name, id;
  get_function_definition_name(ast_node, name, id);

  // 8. populate "std::string debug_modulename"
  std::string debug_modulename = get_modulename_from_path(location_begin.file().as_string());

  // 9. Populate "symbol.static_lifetime", "symbol.is_extern" and "symbol.file_local"
  symbolt symbol;
  get_default_symbol(
    symbol,
    debug_modulename,
    type,
    name,
    id,
    location_begin);

  symbol.lvalue = true;
  symbol.is_extern = false; // TODO: hard coded for now, may need to change later
  symbol.file_local = false;

  // 10. Add symbol into the context
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // 11. Convert parameters, if no parameter, assume ellipis
  //  - Convert params before body as they may get referred by the statement in the body
  SolidityGrammar::ParameterListT params =
    SolidityGrammar::get_parameter_list_t(ast_node["parameters"]);
  if ( params == SolidityGrammar::ParameterListT::EMPTY )
    type.make_ellipsis();
  else
    assert(!"come back and continue - conversion of function arguments");

  added_symbol.type = type;

  // 12. Convert body and embed the body into the same symbol
  if (ast_node.contains("body"))
  {
    exprt body_exprt;
    get_block(ast_node["body"], body_exprt);

    added_symbol.value = body_exprt;
  }

  //assert(!"done - finished all expr stmt in function?");

  // 13. Restore current_functionDecl
  current_functionDecl = old_functionDecl; // for __ESBMC_assume, old_functionDecl == null

  return false;
}

bool solidity_convertert::get_block(const nlohmann::json &block, exprt &new_expr)
{
  // For rule block
  static int call_block_times = 0; // TODO: remove debug
  locationt location;
  get_start_location_from_stmt(block, location);

  SolidityGrammar::BlockT type = SolidityGrammar::get_block_t(block);
  printf("	@@@ got Block: SolidityGrammar::BlockT::%s, ", SolidityGrammar::block_to_str(type));
  printf("  call_block_times=%d\n", call_block_times++);

  switch(type)
  {
    // equivalent to clang::Stmt::CompoundStmtClass
    // deal with a block of statements
    case SolidityGrammar::BlockT::Statement:
    {
      const nlohmann::json &stmts = block["statements"];

      code_blockt _block;
      unsigned ctr = 0;
      // items() returns a key-value pair with key being the index
      for(auto const &stmt_kv : stmts.items())
      {
        exprt statement;
        //print_json_stmt_element(stmt_kv.value(), stmt_kv.value()["nodeType"], ctr);
        if(get_statement(stmt_kv.value(), statement))
          return true;

        convert_expression_to_code(statement);
        _block.operands().push_back(statement);
        ++ctr;
      }
      printf(" \t @@@ CompoundStmt has %u statements\n", ctr);

      // TODO: Figure out the source location manager of Solidity AST JSON
      // It's too encryptic. Currently we are using get_start_location_from_stmt.
      // However, it should be get_final_location_from_stmt.
      locationt location_end;
      get_start_location_from_stmt(block, location_end);

      //assert(!"done - all CompoundStmtClass?");

      _block.end_location(location_end);

      new_expr = _block;
      break;
    }
    default:
    {
      assert(!"Unimplemented type in rule block");
      return true;
    }
  }

  new_expr.location() = location;
  return false;
}

bool solidity_convertert::get_statement(const nlohmann::json &stmt, exprt &new_expr)
{
  // For rule statement
  // Since this is an additional layer of grammar rules compared to clang C, we do NOT set location here.
  // Just pass the new_expr reference to the next layer.
  static int call_stmt_times = 0; // TODO: remove debug

  SolidityGrammar::StatementT type = SolidityGrammar::get_statement_t(stmt);
  printf("	@@@ got Stmt: SolidityGrammar::StatementT::%s, ", SolidityGrammar::statement_to_str(type));
  printf("  call_stmt_times=%d\n", call_stmt_times++);

  switch(type)
  {
    case SolidityGrammar::StatementT::ExpressionStatement:
    {
      get_expr(stmt["expression"], new_expr);
      break;
    }
    case SolidityGrammar::StatementT::VariableDeclStatement:
    {
      const nlohmann::json &declgroup = stmt["declarations"];

      codet decls("decl-block");
      unsigned ctr = 0;
      // N.B. Although Solidity AST JSON uses "declarations": [],
      // there will ALWAYS be 1 declaration!
      // A second declaration will become another stmt in "statements"
      for(const auto &it : declgroup.items())
      {
        // deal with local var decl with init value
        nlohmann::json decl = it.value();
        if (stmt.contains("initialValue"))
        {
          // Need to combine init value with the decl JSON object
          decl["initialValue"] = stmt["initialValue"];
        }

        exprt single_decl;
        if(get_var_decl_stmt(decl, single_decl))
          return true;

        decls.operands().push_back(single_decl);
        ++ctr;
      }
      printf(" \t @@@ DeclStmt group has %u decls\n", ctr);

      new_expr = decls;
      break;
    }
    case SolidityGrammar::StatementT::ReturnStatement:
    {
      if(!current_functionDecl)
      {
        assert(!"Error: ESBMC could not find the parent scope for this ReturnStatement");
      }

      // 1. get return type
      // TODO: Fix me! Assumptions:
      //  a). It's "return <expr>;" not "return;"
      //  b). <expr> is pointing to a DeclRefExpr, we need to wrap it in an ImplicitCastExpr as a subexpr
      assert(stmt.contains("expression"));
      assert(stmt["expression"].contains("referencedDeclaration"));

      //print_json(stmt);
      typet return_type;
      if (get_type_description(stmt["expression"]["typeDescriptions"], return_type))
        return true;

      // 2. get return value
      code_returnt ret_expr;
      const nlohmann::json &rtn_expr = stmt["expression"];
      // wrap it in an ImplicitCastExpr to convert LValue to RValue
      nlohmann::json implicit_cast_expr = make_implicit_cast_expr(rtn_expr, "LValueToRValue");

      exprt val;
      if(get_expr(implicit_cast_expr, val))
        return true;

      solidity_gen_typecast(ns, val, return_type);
      ret_expr.return_value() = val;

      new_expr = ret_expr;
      break;
    }
    default:
    {
      assert(!"Unimplemented type in rule statement");
      return true;
    }
  }

  return false;
}

bool solidity_convertert::get_expr(const nlohmann::json &expr, exprt &new_expr)
{
  // For rule expression
  // We need to do location settings to match clang C's number of times to set the locations when recurring
  static int call_expr_times = 0; // TODO: remove debug
  locationt location;
  get_start_location_from_stmt(expr, location);

  SolidityGrammar::ExpressionT type = SolidityGrammar::get_expression_t(expr);
  printf("	@@@ got Expr: SolidityGrammar::ExpressionT::%s, ", SolidityGrammar::expression_to_str(type));
  printf("  call_expr_times=%d\n", call_expr_times++);

  switch(type)
  {
    case SolidityGrammar::ExpressionT::BinaryOperatorClass:
    {
      if (get_binary_operator_expr(expr, new_expr))
        return true;

      break;
    }
    case SolidityGrammar::ExpressionT::DeclRefExprClass:
    {
      if (expr["referencedDeclaration"] > 0)
      {
        // Soldity uses +ve odd numbers to refer to var or functions declared in the contract
        const nlohmann::json &decl = find_decl_ref(expr["referencedDeclaration"]);
        //printf("\t @@ Debug: this is the matching DeclRef JSON: \n");
        //print_json(decl);

        if (!check_intrinsic_function(decl))
        {
          if (decl["nodeType"] == "VariableDeclaration")
          {
            if (get_var_decl_ref(decl, new_expr))
              return true;
          }
          else if (decl["nodeType"] == "FunctionDefinition")
          {
            if (get_func_decl_ref(decl, new_expr))
              return true;
          }
          else
          {
            assert(!"Unsupported DeclRefExprClass type");
          }
        }
        else
        {
          // for special functions, we need to deal with it separately
          if (get_decl_ref_builtin(expr, new_expr))
            return true;
        }
      }
      else
      {
        // Soldity uses -ve odd numbers to refer to built-in var or functions that
        // are NOT declared in the contract
        if (get_decl_ref_builtin(expr, new_expr))
          return true;
      }

      break;
    }
    case SolidityGrammar::ExpressionT::Literal:
    {
      assert(current_BinOp_type.size());
      const nlohmann::json &binop_type = *(current_BinOp_type.top());
      // make a type-name json for integer literal conversion
      std::string the_value = expr["value"].get<std::string>();
      const nlohmann::json &integer_literal = expr["typeDescriptions"];

      if(convert_integer_literal(integer_literal, the_value, new_expr))
        return true;

      break;
    }
    case SolidityGrammar::ExpressionT::CallExprClass:
    {
      // 1. Get callee expr
      const nlohmann::json &callee_expr_json = expr["expression"];

      // wrap it in an ImplicitCastExpr to perform conversion of FunctionToPointerDecay
      nlohmann::json implicit_cast_expr = make_implicit_cast_expr(callee_expr_json, "FunctionToPointerDecay");
      exprt callee_expr;
      if(get_expr(implicit_cast_expr, callee_expr))
        return true;

      // 2. Get type
      // Need to "decrypt" the typeDescriptions and manually make a typeDescription
      nlohmann::json callee_rtn_type = make_callexpr_return_type(callee_expr_json["typeDescriptions"]);
      typet type;
      if (get_type_description(callee_rtn_type, type))
        return true;

      side_effect_expr_function_callt call;
      call.function() = callee_expr;
      call.type() = type;

      // 3. Set side_effect_expr_function_callt
      unsigned num_args = 0;
      for (const auto &arg : expr["arguments"].items())
      {
        exprt single_arg;
        if(get_expr(arg.value(), single_arg))
          return true;

        call.arguments().push_back(single_arg);
        ++num_args;
      }
      printf("  @@ num_args=%u\n", num_args);

      // 4. Convert call arguments
      new_expr = call;
      break;
    }
    case SolidityGrammar::ExpressionT::ImplicitCastExprClass:
    {
      if(get_cast_expr(expr, new_expr))
        return true;
      break;
    }
    default:
    {
      assert(!"Unimplemented type in rule expression");
      return true;
    }
  }

  new_expr.location() = location;
  return false;
}

bool solidity_convertert::get_binary_operator_expr(const nlohmann::json &expr, exprt &new_expr)
{
  // preliminary step:
  current_BinOp_type.push(&(expr["typeDescriptions"]));

  // 1. Convert LHS and RHS
  // For "Assignment" expression, it's called "leftHandSide" or "rightHandSide".
  // For "BinaryOperation" expression, it's called "leftExpression" or "leftExpression"
  exprt lhs, rhs;
  if (expr.contains("leftHandSide"))
  {
    if(get_expr(expr["leftHandSide"], lhs))
      return true;

    if(get_expr(expr["rightHandSide"], rhs))
      return true;
  }
  else if (expr.contains("leftExpression"))
  {
    if(get_expr(expr["leftExpression"], lhs))
      return true;

    if(get_expr(expr["rightExpression"], rhs))
      return true;
  }
  else
  {
    assert(!"should not be here - unrecognized LHS and RHS keywords in expression JSON");
  }

  // 2. Get type
  typet t;
  assert(current_BinOp_type.size());
  const nlohmann::json &binop_type = *(current_BinOp_type.top());
  if(get_type_description(binop_type, t))
    return true;

  // 3. Convert opcode
  SolidityGrammar::ExpressionT opcode = SolidityGrammar::get_expr_operator_t(expr);
  printf("  @@@ got binop.getOpcode: SolidityGrammar::%s\n", SolidityGrammar::expression_to_str(opcode));
  switch(opcode)
  {
    case SolidityGrammar::ExpressionT::BO_Assign:
    {
      new_expr = side_effect_exprt("assign", t);
      break;
    }
    case SolidityGrammar::ExpressionT::BO_Add:
    {
      if(t.is_floatbv())
        assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
      else
        new_expr = exprt("+", t);
      break;
    }
    case SolidityGrammar::ExpressionT::BO_Sub:
    {
      if(t.is_floatbv())
        assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
      else
        new_expr = exprt("-", t);
      break;
    }
    case SolidityGrammar::ExpressionT::BO_GT:
    {
      new_expr = exprt(">", t);
      break;
    }
    case SolidityGrammar::ExpressionT::BO_LT:
    {
      new_expr = exprt("<", t);
      break;
    }
    case SolidityGrammar::ExpressionT::BO_NE:
    {
      new_expr = exprt("notequal", t);
      break;
    }
    case SolidityGrammar::ExpressionT::BO_Rem:
    {
      new_expr = exprt("mod", t);
      break;
    }
    default:
    {
      assert(!"Unimplemented operator");
    }
  }

  // 4. Copy to operands
  new_expr.copy_to_operands(lhs, rhs);

  // Pop current_BinOp_type.push as we've finished this conversion
  current_BinOp_type.pop();

  return false;
}

bool solidity_convertert::get_cast_expr(const nlohmann::json &cast_expr, exprt &new_expr)
{
  // 1. convert subexpr
  exprt expr;
  if (get_expr(cast_expr["subExpr"], expr))
    return true;

  // 2. get type
  typet type;
  if (get_type_description(cast_expr["subExpr"]["typeDescriptions"], type))
    return true;

  // 3. get cast type and generate typecast
  SolidityGrammar::ImplicitCastTypeT cast_type =
    SolidityGrammar::get_implicit_cast_type_t(cast_expr["castType"].get<std::string>());
  switch(cast_type)
  {
    case SolidityGrammar::ImplicitCastTypeT::LValueToRValue:
    {
      solidity_gen_typecast(ns, expr, type);
      break;
    }
    case SolidityGrammar::ImplicitCastTypeT::FunctionToPointerDecay:
    {
      break;
    }
    default:
    {
      assert(!"Unimplemented implicit cast type");
    }
  }

  new_expr = expr;
  return false;
}

bool solidity_convertert::get_var_decl_ref(const nlohmann::json &decl, exprt &new_expr)
{
  // Function to configure new_expr that has a +ve referenced id, referring to a variable declaration
  assert(decl["nodeType"] == "VariableDeclaration");
  std::string name, id;
  if (decl["stateVariable"])
  {
    get_state_var_decl_name(decl, name, id);
  }
  else
  {
    get_var_decl_name(decl, name, id);
  }

  typet type;
  if (get_type_description(decl["typeName"]["typeDescriptions"], type)) // "type-name" as in state-variable-declaration
    return true;

  new_expr = exprt("symbol", type);
  new_expr.identifier(id);
  new_expr.cmt_lvalue(true);
  new_expr.name(name);
  return false;
}

bool solidity_convertert::get_func_decl_ref(const nlohmann::json &decl, exprt &new_expr)
{
  // Function to configure new_expr that has a +ve referenced id, referring to a function declaration
  assert(decl["nodeType"] == "FunctionDefinition");
  std::string name, id;
  get_function_definition_name(decl, name, id);

  typet type;
  if (get_func_decl_ref_type(decl, type)) // "type-name" as in state-variable-declaration
    return true;

  new_expr = exprt("symbol", type);
  new_expr.identifier(id);
  new_expr.cmt_lvalue(true);
  new_expr.name(name);
  return false;
}

bool solidity_convertert::get_decl_ref_builtin(const nlohmann::json &decl, exprt &new_expr)
{
  // Function to configure new_expr that has a -ve referenced id
  // -ve ref id means built-in functions or variables.
  // Add more special function names here
  assert(decl["name"] == "assert" ||
         decl["name"] == "__ESBMC_assume" ||
         decl["name"] == "__VERIFIER_assume");

  std::string name, id;
  name = decl["name"].get<std::string>();
  id = "c:@F@" + name;

  // manually unrolled recursion here
  // type config for Builtin && Int
  typet type;
  // Creat a new code_typet, parse the return_type and copy the code_typet to typet
  code_typet convert_type;
  typet return_type;
  if (decl["name"] == "assert" ||
      decl["name"] == "__ESBMC_assume" ||
      decl["name"] == "__VERIFIER_assume")
  {
    assert(decl["typeDescriptions"]["typeString"] == "function (bool) pure");
    // clang's assert(.) uses "signed_int" as assert(.) type (NOT the argument type),
    // while Solidity's assert uses "bool" as assert(.) type (NOT the argument type).
    return_type = bool_type();
    std::string c_type = "bool";
    return_type.set("#cpp_type", c_type);
    convert_type.return_type() = return_type;

    if(!convert_type.arguments().size())
      convert_type.make_ellipsis();
  }
  else
  {
    assert(!"Unsupported special functions");
  }


  type = convert_type;

  new_expr = exprt("symbol", type);
  new_expr.identifier(id);
  new_expr.cmt_lvalue(true);
  new_expr.name(name);
  return false;
}

bool solidity_convertert::get_type_description(const nlohmann::json &type_name, typet &new_type)
{
  // For Solidity rule type-name:
  SolidityGrammar::TypeNameT type = SolidityGrammar::get_type_name_t(type_name);

  switch(type)
  {
    case SolidityGrammar::TypeNameT::ElementaryTypeName:
    {
      // rule state-variable-declaration
      return get_elementary_type_name(type_name, new_type);
    }
    case SolidityGrammar::TypeNameT::ParameterList:
    {
      // rule parameter-list
      // Used for Solidity function parameter or return list
      return get_parameter_list(type_name, new_type);
    }
    case SolidityGrammar::TypeNameT::Pointer:
    {
      // auxiliary type: pointer
      // TODO: Fix me! Assuming it's a function call
      assert(type_name["typeString"].get<std::string>().find("function") != std::string::npos);

      // Since Solidity does not have this, first make a pointee
      nlohmann::json pointee = make_pointee_type(type_name);
      typet sub_type;
      if(get_func_decl_ref_type(pointee, sub_type))
        return true;

      // TODO: classes
      if(sub_type.is_struct() || sub_type.is_union()) // for "assert(sum > 100)", false || false
      {
        assert(!"struct or union is NOT supported");
      }

      new_type = gen_pointer_type(sub_type);
      break;
    }
    default:
    {
      assert(!"Unimplemented type in rule type-name");
      return true;
    }
  }

  // TODO: More var decl attributes checks:
  //    - Constant
  //    - Volatile
  //    - isRestrict

  return false;
}

bool solidity_convertert::get_func_decl_ref_type(const nlohmann::json &decl, typet &new_type)
{
  // Get type when we make a function call:
  //  - FunnctionNoProto: x = nondet()
  //  - FunctionProto:    z = add(x, y)
  // Similar to the function get_type_description()
  SolidityGrammar::FunctionDeclRefT type = SolidityGrammar::get_func_decl_ref_t(decl);

  switch(type)
  {
    case SolidityGrammar::FunctionDeclRefT::FunctionNoProto:
    {
      code_typet type;

      // Return type
      const nlohmann::json &rtn_type = decl["returnParameters"];

      typet return_type;
      if (get_type_description(rtn_type, return_type))
        return true;

      type.return_type() = return_type;

      if(!type.arguments().size())
        type.make_ellipsis();

      new_type = type;
      break;
    }
    default:
    {
      printf("Got type=%s ...\n", SolidityGrammar::func_decl_ref_to_str(type));
      assert(!"Unimplemented type in auxiliary type to convert function call");
      return true;
    }
  }

  // TODO: More var decl attributes checks:
  //    - Constant
  //    - Volatile
  //    - isRestrict
  return false;
}

bool solidity_convertert::get_elementary_type_name(const nlohmann::json &type_name, typet &new_type)
{
  // For Solidity rule elementary-type-name:
  // equivalent to clang's get_builtin_type()
  std::string c_type;
  SolidityGrammar::ElementaryTypeNameT type = SolidityGrammar::get_elementary_type_name_t(type_name);

  switch(type)
  {
    // rule unsigned-integer-type
    case SolidityGrammar::ElementaryTypeNameT::UINT8:
    {
      new_type = unsigned_char_type();
      c_type = "unsigned_char";
      new_type.set("#cpp_type", c_type);
      break;
    }
    case SolidityGrammar::ElementaryTypeNameT::BOOL:
    {
      new_type = bool_type();
      c_type = "bool";
      new_type.set("#cpp_type", c_type);
      break;
    }
    default:
    {
      printf(" Got elementary-type-name=%s ...\n", SolidityGrammar::elementary_type_name_to_str(type));
      assert(!"Unimplemented type in rule elementary-type-name");
      return true;
    }
  }

  return false;
}

bool solidity_convertert::get_parameter_list(const nlohmann::json &type_name, typet &new_type)
{
  // For Solidity rule parameter-list:
  //  - For non-empty param list, it may need to call get_elementary_type_name, since parameter-list is just a list of types
  std::string c_type;
  SolidityGrammar::ParameterListT type = SolidityGrammar::get_parameter_list_t(type_name);

  switch(type)
  {
    case SolidityGrammar::ParameterListT::EMPTY:
    {
      // equivalent to clang's "void"
      new_type = empty_typet();
      c_type = "void";
      new_type.set("#cpp_type", c_type);
      break;
    }
    case SolidityGrammar::ParameterListT::NONEMPTY:
    {
      //print_json(type_name);
      assert(type_name["parameters"].size() == 1); // TODO: Fix me! assuming one return parameter
      const nlohmann::json &rtn_type = type_name["parameters"].at(0)["typeName"]["typeDescriptions"];
      return get_elementary_type_name(rtn_type, new_type);

      break;
    }
    default:
    {
      assert(!"Unimplemented type in rule parameter-list");
      return true;
    }
  }

  return false;
}

void solidity_convertert::get_state_var_decl_name(
    const nlohmann::json &ast_node,
    std::string &name, std::string &id)
{
  // Follow the way in clang:
  //  - For state variable name, just use the ast_node["name"]
  //  - For state variable id, add prefix "c:@"
  name = ast_node["name"].get<std::string>(); // assume Solidity AST json object has "name" field, otherwise throws an exception in nlohmann::json
  id = "c:@" + name;
}

void solidity_convertert::get_var_decl_name(
    const nlohmann::json &ast_node,
    std::string &name, std::string &id)
{
  assert(current_functionDecl); // TODO: Fix me! assuming converting local variable inside a function
  // For non-state functions, we give it different id.
  // E.g. for local variable i in function nondet(), it's "c:overflow_2_nondet.c@55@F@nondet@i".
  name = ast_node["name"].get<std::string>(); // assume Solidity AST json object has "name" field, otherwise throws an exception in nlohmann::json
  id = "c:" + get_modulename_from_path(absolute_path) + ".solast" +
       //"@"  + std::to_string(ast_node["scope"].get<int>()) +
       "@"  + std::to_string(445) +
       "@"  + "F" +
       "@"  + current_function_name +
       "@"  + name;
}

void solidity_convertert::get_function_definition_name(
    const nlohmann::json &ast_node,
    std::string &name, std::string &id)
{
  // Follow the way in clang:
  //  - For function name, just use the ast_node["name"]
  //  - For function id, add prefix "c:@F@"
  name = ast_node["name"].get<std::string>(); // assume Solidity AST json object has "name" field, otherwise throws an exception in nlohmann::json
  id = "c:@F@" + name;
}

void solidity_convertert::get_location_from_decl(const nlohmann::json &ast_node, locationt &location)
{
  // The src manager of Solidity AST JSON is too encryptic.
  // For the time being we are setting it to "1".
  location.set_line(1);
  location.set_file(absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  // To annotate local declaration within a function
  if (ast_node["nodeType"] == "VariableDeclaration" &&
      ast_node["stateVariable"] == false &&
      ast_node["scope"] != global_scope_id)
  {
    assert(current_functionDecl); // must have a valid current function declaration
    location.set_function(current_functionName); // set the function where this local variable belongs to
  }
}

void solidity_convertert::get_start_location_from_stmt(const nlohmann::json &stmt_node, locationt &location)
{
  std::string function_name;

  if (current_functionDecl)
    function_name = current_functionName;

  // The src manager of Solidity AST JSON is too encryptic.
  // For the time being we are setting it to "1".
  location.set_line(1);
  location.set_file(absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  if (!function_name.empty())
    location.set_function(function_name);
}

std::string solidity_convertert::get_modulename_from_path(std::string path)
{
  std::string filename = get_filename_from_path(path);

  if(filename.find_last_of('.') != std::string::npos)
    return filename.substr(0, filename.find_last_of('.'));

  return filename;
}

std::string solidity_convertert::get_filename_from_path(std::string path)
{
  if(path.find_last_of('/') != std::string::npos)
    return path.substr(path.find_last_of('/') + 1);

  return path; // for _x, it just returns "overflow_2.c" because the test program is in the same dir as esbmc binary
}

const nlohmann::json& solidity_convertert::find_decl_ref(int ref_decl_id)
{
  // First, search state variable nodes
  nlohmann::json& nodes = ast_json["nodes"];
  nlohmann::json::iterator itr = nodes.begin();
  unsigned index = 0;
  for (; itr != nodes.end(); ++itr, ++index)
  {
    if ( (*itr)["nodeType"] == "ContractDefinition" ) // contains AST nodes we need
      break;
  }

  nlohmann::json& ast_nodes = nodes.at(index)["nodes"];
  nlohmann::json::iterator itrr = ast_nodes.begin();
  index = 0;
  for (; itrr != ast_nodes.end(); ++itrr, ++index)
  {
    if ( (*itrr)["id"] == ref_decl_id)
    {
      //print_json(ast_nodes.at(index));
      return ast_nodes.at(index);
    }
  }

  // Then search "declarations" in current function scope
  const nlohmann::json &current_func = *current_functionDecl;
  if (current_func.contains("body"))
  {
    if (current_func["body"].contains("statements"))
    {
      for (const auto &body_stmt : current_func["body"]["statements"].items())
      {
        const nlohmann::json &stmt = body_stmt.value();
        if (stmt["nodeType"] == "VariableDeclarationStatement")
        {
          for (const auto &local_decl : stmt["declarations"].items())
          {
            const nlohmann::json &the_decl = local_decl.value();
            if (the_decl["id"] == ref_decl_id)
            {
              return the_decl;
            }
          }
        }
      }
    }
    else
      assert(!"Unable to find the corresponding local variable decl. Function body  does not have statements.");
  }
  else
    assert(!"Unable to find the corresponding local variable decl. Current function does not have a function body.");

  assert(!"should not be here - no matching ref decl id found");
  return ast_json;
}

void solidity_convertert::get_default_symbol(
  symbolt &symbol,
  std::string module_name,
  typet type,
  std::string name,
  std::string id,
  locationt location)
{
  symbol.mode = "C";
  symbol.module = module_name;
  symbol.location = std::move(location);
  symbol.type = std::move(type);
  symbol.name = name;
  symbol.id = id;
}

symbolt *solidity_convertert::move_symbol_to_context(symbolt &symbol)
{
  symbolt *s = context.find_symbol(symbol.id);
  if(s == nullptr)
  {
    if(context.move(symbol, s))
    {
      std::cerr << "Couldn't add symbol " << symbol.name
                << " to symbol table\n";
      symbol.dump();
      abort();
    }
  }
  else
  {
    // types that are code means functions
    if(s->type.is_code())
    {
      if(symbol.value.is_not_nil() && !s->value.is_not_nil())
        s->swap(symbol);
    }
    else if(s->is_type)
    {
      if(symbol.type.is_not_nil() && !s->type.is_not_nil())
        s->swap(symbol);
    }
  }

  return s;
}

void solidity_convertert::convert_expression_to_code(exprt &expr)
{
  if(expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

bool solidity_convertert::check_intrinsic_function(const nlohmann::json &ast_node)
{
  // function to detect special intrinsic functions, e.g. __ESBMC_assume
  if (ast_node["name"] == "__ESBMC_assume" ||
     ast_node["name"] == "__VERIFIER_assume")
  {
    return true;
  }
  else
  {
    return false;
  }

  return false; // make old compiler happy, e.g. pre-GCC4.9
}

nlohmann::json solidity_convertert::make_implicit_cast_expr(const nlohmann::json& sub_expr, std::string cast_type)
{
  // Since Solidity AST does not have type cast information about return values,
  // we need to manually make a JSON object and wrap the return expression in it.
  std::map<std::string, std::string> m = {
    {"nodeType", "ImplicitCastExprClass"},
    {"castType", cast_type},
    {"subExpr", {}}
  };
  nlohmann::json implicit_cast_expr = m;
  implicit_cast_expr["subExpr"] = sub_expr;

  return implicit_cast_expr;
}

nlohmann::json solidity_convertert::make_pointee_type(const nlohmann::json& sub_expr)
{
  // Since Solidity function call node does not have enough information, we need to make a JSON object
  // manually create a JSON object to complete the conversions of function to pointer decay

  // make a mapping for JSON object creation latter
  // based on the usage of get_func_decl_ref_t() in get_func_decl_ref_type()
  nlohmann::json adjusted_expr;

  if (sub_expr["typeString"].get<std::string>().find("function") != std::string::npos)
  {
    // Add more special functions here
    if (sub_expr["typeString"].get<std::string>().find("function ()") != std::string::npos ||
        sub_expr["typeIdentifier"].get<std::string>().find("t_function_assert_pure$") != std::string::npos ||
        sub_expr["typeIdentifier"].get<std::string>().find("t_function_internal_pure$") != std::string::npos)
    {
      // e.g. FunctionNoProto: "typeString": "function () returns (uint8)" with () empty after keyword 'function'
      // "function ()" contains the function args in the parentheses.
      // make a type to behave like SolidityGrammar::FunctionDeclRefT::FunctionNoProto
      // Note that when calling "assert(.)", it's like "typeIdentifier": "t_function_assert_pure$......",
      //  it's also treated as "FunctionNoProto".
      auto j2 = R"(
        {
          "nodeType": "FunctionDefinition",
          "parameters":
            {
              "parameters" : []
            }
        }
      )"_json;
      adjusted_expr = j2;

      if (sub_expr["typeString"].get<std::string>().find("returns") != std::string::npos)
      {
        // e.g. for typeString like:
        // "typeString": "function () returns (uint8)"
        // TODO: currently we only assume one parameters
        if (sub_expr["typeString"].get<std::string>().find("returns (uint8)") != std::string::npos)
        {
          auto j2 = R"(
            {
              "typeIdentifier": "t_uint8",
              "typeString": "uint8"
            }
          )"_json;
          adjusted_expr["returnParameters"] = j2;
        }
        else
        {
          assert(!"Unsupported return types in pointee");
        }
      }
      else
      {
        // e.g. for typeString like:
        // "typeString": "function (bool) pure"
        auto j2 = R"(
          {
            "nodeType": "ParameterList",
            "parameters": []
          }
        )"_json;
        adjusted_expr["returnParameters"] = j2;
      }
    }
    else
    {
      assert(!"Unsupported - detected function call with parameters");
    }
  }
  else
  {
    assert(!"Unsupported pointee - currently we only support the semantics of function to pointer decay");
  }

  return adjusted_expr;
}

nlohmann::json solidity_convertert::make_callexpr_return_type(const nlohmann::json& type_descrpt)
{
  nlohmann::json adjusted_expr;
  if (type_descrpt["typeString"].get<std::string>().find("function") != std::string::npos)
  {
      if (type_descrpt["typeString"].get<std::string>().find("returns") != std::string::npos)
      {
        // e.g. for typeString like:
        // "typeString": "function () returns (uint8)"
        if (type_descrpt["typeString"].get<std::string>().find("returns (uint8)") != std::string::npos)
        {
          auto j2 = R"(
            {
              "typeIdentifier": "t_uint8",
              "typeString": "uint8"
            }
          )"_json;
          adjusted_expr = j2;
        }
        else
        {
          assert(!"Unsupported types in callee's return in CallExpr");
        }
      }
      else
      {
        // Since Solidity allows multiple parameters and multiple returns for functions,
        // we need to use "parameters" in conjunction with "returnParameters" to convert.
        // the following configuration will lead to "void".
        auto j2 = R"(
          {
            "nodeType": "ParameterList",
            "parameters": []
          }
        )"_json;
        adjusted_expr = j2;
      }
  }
  else
  {
    assert(!"Unsupported pointee - currently we only support the semantics of function to pointer decay");
  }

  return adjusted_expr;
}

// debug functions
void solidity_convertert::print_json(const nlohmann::json &json_in)
{
  printf("### JSON: ###\n");
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}

void solidity_convertert::print_json_element(const nlohmann::json &json_in, const unsigned index,
    const std::string &key, const std::string& json_name)
{
  printf("### %s element[%u] content: key=%s, size=%lu ###\n",
      json_name.c_str(), index, key.c_str(), json_in.size());
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}

void solidity_convertert::print_json_array_element(const nlohmann::json &json_in,
    const std::string& node_type, const unsigned index)
{
  printf("### node[%u]: nodeType=%s ###\n", index, node_type.c_str());
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}

void solidity_convertert::print_json_stmt_element(const nlohmann::json &json_in,
    const std::string& node_type, const unsigned index)
{
  printf("\t### stmt[%u]: nodeType=%s ###\n", index, node_type.c_str());
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}
