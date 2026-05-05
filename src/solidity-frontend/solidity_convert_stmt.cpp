/// \file solidity_convert_stmt.cpp
/// \brief Statement conversion for the Solidity frontend.
///
/// Converts Solidity statements (blocks, if/else, for, while, do-while,
/// return, break, continue, emit, revert, require/assert, variable
/// declaration statements, expression statements, and try-catch) from
/// the solc JSON AST into ESBMC's GOTO-level code representation.

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
#include <set>

void solidity_convertert::reset_auxiliary_vars()
{
  current_baseContractName = "";
  current_functionName = "";
  current_functionDecl = nullptr;
  current_forStmt = nullptr;
  initializers.clear();
}

bool solidity_convertert::get_function_params(
  const nlohmann::json &pd,
  const std::string &cname,
  exprt &param)
{
  // 1. get parameter type
  typet param_type;
  if (get_type_description(pd["typeDescriptions"], param_type))
    return true;

  // 2a. get id and name
  std::string id, name;
  assert(current_functionName != ""); // we are converting a function param now
  assert(current_functionDecl);
  get_local_var_decl_name(pd, cname, name, id);
  // 2b. handle Omitted Names in Function Definitions
  if (name == "")
  {
    // Items with omitted names will still be present on the stack, but they are inaccessible by name.
    // e.g. ~omitted1, ~omitted2. which is a invalid name for solidity.
    // Therefore it won't conflict with other arg names.
    //log_error("Omitted params are not supported");
    // return true;
    ;
  }

  param = code_typet::argumentt();
  param.type() = param_type;
  param.cmt_base_name(name);

  // 3. get location
  locationt location_begin;
  get_location_from_node(pd, location_begin);

  param.cmt_identifier(id);
  param.location() = location_begin;

  // 4. get symbol
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());
  symbolt param_symbol;
  get_default_symbol(
    param_symbol, debug_modulename, param_type, name, id, location_begin);

  // 5. set symbol's lvalue, is_parameter and file local
  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;

  // 6. add symbol to the context
  move_symbol_to_context(param_symbol);

  return false;
}

bool solidity_convertert::get_block(
  const nlohmann::json &block,
  exprt &new_expr)
{
  // For rule block
  locationt location;
  get_start_location_from_stmt(block, location);

  SolidityGrammar::BlockT type = SolidityGrammar::get_block_t(block);
  log_debug(
    "solidity",
    "	@@@ got Block: SolidityGrammar::BlockT::{}",
    SolidityGrammar::block_to_str(type));

  switch (type)
  {
  // equivalent to clang::Stmt::CompoundStmtClass
  // deal with a block of statements
  case SolidityGrammar::BlockT::Statement:
  {
    const nlohmann::json &stmts = block["statements"];

    // Track unchecked blocks: save/restore flag using RAII pattern
    const bool is_unchecked = (block["nodeType"] == "UncheckedBlock");
    const bool prev_unchecked = in_unchecked_block;
    if (is_unchecked)
      in_unchecked_block = true;

    code_blockt _block;
    unsigned ctr = 0;
    // items() returns a key-value pair with key being the index
    for (auto const &stmt_kv : stmts.items())
    {
      locationt cl;
      get_location_from_node(stmt_kv.value(), cl);
      if (in_unchecked_block)
        cl.set("#sol_unchecked", "1");

      exprt statement;
      if (get_statement(stmt_kv.value(), statement))
        return true;

      if (!expr_frontBlockDecl.operands().empty())
      {
        for (auto op : expr_frontBlockDecl.operands())
        {
          convert_expression_to_code(op);
          _block.operands().push_back(op);
        }
        expr_frontBlockDecl.clear();
      }
      statement.location() = cl;
      convert_expression_to_code(statement);
      _block.operands().push_back(statement);

      if (!expr_backBlockDecl.operands().empty())
      {
        for (auto op : expr_backBlockDecl.operands())
        {
          convert_expression_to_code(op);
          _block.operands().push_back(op);
        }
        expr_backBlockDecl.clear();
      }

      ++ctr;
    }
    log_debug("solidity", " \t@@@ CompoundStmt has {} statements", ctr);

    locationt location_end;
    get_final_location_from_stmt(block, location_end);

    _block.end_location(location_end);
    new_expr = _block;

    // Restore unchecked flag
    in_unchecked_block = prev_unchecked;
    break;
  }
  case SolidityGrammar::BlockT::BlockForStatement:
  case SolidityGrammar::BlockT::BlockIfStatement:
  case SolidityGrammar::BlockT::BlockWhileStatement:
  case SolidityGrammar::BlockT::BlockDoWhileStatement:
  {
    // this means only one statement in the block
    exprt statement;

    // pass directly to get_statement()
    if (get_statement(block, statement))
      return true;
    convert_expression_to_code(statement);
    new_expr = statement;
    break;
  }
  case SolidityGrammar::BlockT::BlockExpressionStatement:
  {
    get_expr(block["expression"], new_expr);
    break;
  }
  case SolidityGrammar::BlockT::BlockTError:
  default:
  {
    log_error("Unimplemented type in rule block");
    return true;
  }
  }

  new_expr.location() = location;
  return false;
}

bool solidity_convertert::get_statement(
  const nlohmann::json &stmt,
  exprt &new_expr)
{
  // For rule statement
  // Since this is an additional layer of grammar rules compared to clang C, we do NOT set location here.
  // Just pass the new_expr reference to the next layer.

  locationt loc;
  get_location_from_node(stmt, loc);

  SolidityGrammar::StatementT type = SolidityGrammar::get_statement_t(stmt);
  log_debug(
    "solidity",
    "	@@@ got Stmt: SolidityGrammar::StatementT::{}",
    SolidityGrammar::statement_to_str(type));

  switch (type)
  {
  case SolidityGrammar::StatementT::Block:
  {
    if (get_block(stmt, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::StatementT::ExpressionStatement:
  {
    if (get_expr(
          stmt["expression"], stmt["expression"]["typeDescriptions"], new_expr))
      return true;
    break;
  }
  case SolidityGrammar::StatementT::VariableDeclStatement:
  {
    const nlohmann::json &declgroup = stmt["declarations"];

    codet decls("decl-block");
    unsigned ctr = 0;
    // N.B. Although Solidity AST JSON uses "declarations": [],
    // the size of this array is alway 1!
    // A second declaration will become another stmt in "statements" array
    // e.g. "statements" : [
    //  {"declarations": [], "id": 1}
    //  {"declarations": [], "id": 2}
    //  {"declarations": [], "id": 3}
    // ]
    if (declgroup.size() == 1)
    {
      // deal with local var decl with init value
      const nlohmann::json &decl = declgroup[0];
      nlohmann::json initialValue = nlohmann::json::object();
      if (stmt.contains("initialValue"))
        initialValue = stmt["initialValue"];

      exprt single_decl;
      if (get_var_decl(decl, initialValue, single_decl))
        return true;

      decls.operands().push_back(single_decl);
      ++ctr;
    }
    else
    {
      // separate the decl and assignment
      for (const auto &it : declgroup.items())
      {
        if (it.value().is_null() || it.value().empty())
          continue;
        const nlohmann::json &decl = it.value();
        exprt single_decl;
        if (get_var_decl(decl, single_decl))
          return true;
        decls.operands().push_back(single_decl);
        ++ctr;
      }

      if (stmt.contains("initialValue"))
      {
        // this is a tuple expression
        const nlohmann::json &initialValue = stmt["initialValue"];
        exprt tuple_expr;
        if (get_expr(initialValue, tuple_expr))
          return true;

        // Build LHS block preserving original positions so that omitted
        // elements (null in declarations) map to nil_exprt at the correct
        // index.  construct_tuple_assigments uses positional "mem{i}" keys,
        // so the indices must match the RHS tuple struct layout.
        code_blockt lhs_block;
        unsigned decl_idx = 0;
        for (const auto &it : declgroup.items())
        {
          if (it.value().is_null() || it.value().empty())
          {
            lhs_block.copy_to_operands(nil_exprt());
          }
          else
          {
            assert(decl_idx < decls.operands().size());
            lhs_block.copy_to_operands(decls.operands()[decl_idx].op0());
            ++decl_idx;
          }
        }

        construct_tuple_assigments(stmt, lhs_block, tuple_expr);
      }
    }
    log_debug("solidity", " \t@@@ DeclStmt group has {} decls", ctr);

    new_expr = decls;
    break;
  }
  case SolidityGrammar::StatementT::ReturnStatement:
  {
    if (!current_functionDecl)
    {
      log_error(
        "Error: ESBMC could not find the parent scope for this "
        "ReturnStatement");
      return true;
    }

    // 1. get return type
    // TODO: Fix me! Assumptions:
    //  a). It's "return <expr>;" not "return;"
    //  b). <expr> is pointing to a DeclRefExpr, we need to wrap it in an ImplicitCastExpr as a subexpr
    //  c). For multiple return type, the return statement represented as a tuple expression using a components field.
    //      Besides, tuple can only be declared literally. https://docs.soliditylang.org/en/latest/control-structures.html#assignment
    //      e.g. return (false, 123)
    if (!stmt.contains("expression"))
    {
      // "return;"
      code_returnt ret_expr;
      new_expr = ret_expr;
      return false;
    }
    assert(stmt["expression"].contains("nodeType"));

    // get_type_description
    typet return_exrp_type;
    if (get_type_description(
          stmt["expression"]["typeDescriptions"], return_exrp_type))
      return true;

    if (
      get_sol_type(return_exrp_type) == SolidityGrammar::SolType::TUPLE_RETURNS)
    {
      if (
        stmt["expression"]["nodeType"].get<std::string>() !=
          "TupleExpression" &&
        stmt["expression"]["nodeType"].get<std::string>() != "FunctionCall")
      {
        log_error("Unexpected tuple");
        return true;
      }

      // get tuple instance
      std::string tname, tid;
      if (get_tuple_instance_name(*current_functionDecl, tname, tid))
        return true;
      if (context.find_symbol(tid) == nullptr)
      {
        log_error("cannot find tuple instance symbol: {}", tid);
        return true;
      }

      // get lhs
      exprt lhs = symbol_expr(*context.find_symbol(tid));

      if (
        stmt["expression"]["nodeType"].get<std::string>() == "TupleExpression")
      {
        // return (x,y) ==>
        // tuple.mem0 = x; tuple.mem1 = y; return ;

        // get rhs
        // hack: we need the expression block, not tuple instance
        current_lhsDecl = true;
        exprt rhs;
        if (get_expr(stmt["expression"], rhs))
          return true;
        current_lhsDecl = false;

        size_t ls = to_struct_type(lhs.type()).components().size();
        size_t rs = rhs.operands().size();
        if (ls != rs)
        {
          log_debug(
            "soldiity",
            "Handling return tuple.\nlhs = {}\nrhs = {}",
            lhs.to_string(),
            rhs.to_string());
          log_error("Internal tuple error.");
        }

        for (size_t i = 0; i < ls; i++)
        {
          // lop: struct member call (e.g. tuple.men0)
          exprt lop;
          if (get_tuple_member_call(
                lhs.identifier(),
                to_struct_type(lhs.type()).components().at(i),
                lop))
            return true;

          // rop: constant/symbol
          exprt rop = rhs.operands().at(i);

          // do assignment
          get_tuple_assignment(stmt, lop, rop);
        }
      }
      else
      {
        // return func(); ==>
        // tuple1.mem0 = tuple0.mem0; return;

        // get rhs
        exprt rhs;
        if (get_tuple_function_ref(stmt["expression"]["expression"], rhs))
          return true;

        // add function call
        exprt func_call;
        if (get_expr(
              stmt["expression"],
              stmt["expression"]["typeDescriptions"],
              func_call))
          return true;
        get_tuple_function_call(func_call);

        size_t ls = to_struct_type(lhs.type()).components().size();
        size_t rs = to_struct_type(rhs.type()).components().size();
        if (ls != rs)
        {
          log_error("Unexpected tuple structure");
          abort();
        }

        for (size_t i = 0; i < ls; i++)
        {
          // lop: struct member call (e.g. tupleA.men0)
          exprt lop;
          if (get_tuple_member_call(
                lhs.identifier(),
                to_struct_type(lhs.type()).components().at(i),
                lop))
            return true;

          // rop: struct member call (e.g. tupleB.men0)
          exprt rop;
          if (get_tuple_member_call(
                rhs.identifier(),
                to_struct_type(rhs.type()).components().at(i),
                rop))
            return true;

          // do assignment
          get_tuple_assignment(stmt, lop, rop);
        }
      }
      // do return in the end
      exprt return_expr = code_returnt();
      move_to_back_block(return_expr);

      new_expr = code_skipt();
      break;
    }

    typet return_type;
    // When inlining a delegate-shadow body, the return statement belongs to
    // the target function, not the caller. Use the override set by
    // try_get_delegate_shadow_call so we pick up the target's return type.
    const nlohmann::json *ret_params_src =
      delegate_shadow_target_return_params != nullptr
        ? delegate_shadow_target_return_params
        : ((*current_functionDecl).contains("returnParameters")
             ? &(*current_functionDecl)["returnParameters"]
             : nullptr);
    if (ret_params_src != nullptr)
    {
      // Skip the id-matching assertion when we're overriding — the caller
      // and the target have different ParameterList ids by construction.
      if (
        delegate_shadow_target_return_params == nullptr &&
        (*current_functionDecl).contains("returnParameters"))
      {
        assert(
          (*current_functionDecl)["returnParameters"]["id"]
            .get<std::uint16_t>() ==
          stmt["functionReturnParameters"].get<std::uint16_t>());
      }
      if (get_type_description(*ret_params_src, return_type))
        return true;
    }
    else
      return true;

    nlohmann::json literal_type = nullptr;

    auto expr_type = SolidityGrammar::get_expression_t(stmt["expression"]);
    bool expr_is_literal = expr_type == SolidityGrammar::Literal;
    if (expr_is_literal)
      literal_type = make_return_type_from_typet(return_type);

    // 2. get return value
    code_returnt ret_expr;
    const nlohmann::json &rtn_expr = stmt["expression"];
    // wrap it in an ImplicitCastExpr to convert LValue to RValue
    nlohmann::json implicit_cast_expr =
      make_implicit_cast_expr(rtn_expr, "LValueToRValue");

    /* There could be case like
      {
      "expression": {
          "kind": "number",
          "nodeType": "Literal",
          "typeDescriptions": {
              "typeIdentifier": "t_rational_11_by_1",
              "typeString": "int_const 11"
          },
          "value": "12345"
      },
      "nodeType": "Return",
      }
      Therefore, we need to pass the literal_type value.
      */

    exprt rhs;
    if (get_expr(implicit_cast_expr, literal_type, rhs))
      return true;

    solidity_gen_typecast(ns, rhs, return_type);
    ret_expr.return_value() = rhs;

    new_expr = ret_expr;

    break;
  }
  case SolidityGrammar::StatementT::ForStatement:
  {
    // Based on rule for-statement

    // For nested loop
    const nlohmann::json *old_forStmt = current_forStmt;
    current_forStmt = &stmt;

    // 1. annotate init
    codet init =
      code_skipt(); // code_skipt() means no init in for-stmt, e.g. for (; i< 10; ++i)
    if (stmt.contains("initializationExpression"))
      if (get_statement(stmt["initializationExpression"], init))
        return true;

    convert_expression_to_code(init);

    // 2. annotate condition
    exprt cond = true_exprt();
    if (stmt.contains("condition"))
      if (get_expr(stmt["condition"], cond))
        return true;

    // 3. annotate increment
    codet inc = code_skipt();
    if (stmt.contains("loopExpression"))
      if (get_statement(stmt["loopExpression"], inc))
        return true;

    convert_expression_to_code(inc);

    // 4. annotate body
    codet body = code_skipt();
    if (stmt.contains("body"))
      if (get_statement(stmt["body"], body))
        return true;

    convert_expression_to_code(body);

    code_fort code_for;
    code_for.init() = init;
    code_for.cond() = cond;
    code_for.iter() = inc;
    code_for.body() = body;

    new_expr = code_for;
    current_forStmt = old_forStmt;
    break;
  }
  case SolidityGrammar::StatementT::IfStatement:
  {
    // Based on rule if-statement
    // 1. Condition: make a exprt for condition
    exprt cond;
    if (get_expr(stmt["condition"], cond))
      return true;

    // 2. Then: make a exprt for trueBody
    exprt then;
    if (get_statement(stmt["trueBody"], then))
      return true;

    convert_expression_to_code(then);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(cond, then);

    // 3. Else: make a exprt for "falseBody" if the if-statement node contains an "else" block
    if (stmt.contains("falseBody"))
    {
      exprt else_expr;
      if (get_statement(stmt["falseBody"], else_expr))
        return true;

      convert_expression_to_code(else_expr);
      if_expr.copy_to_operands(else_expr);
    }

    new_expr = if_expr;
    break;
  }
  case SolidityGrammar::StatementT::WhileStatement:
  {
    exprt cond = true_exprt();
    if (get_expr(stmt["condition"], cond))
      return true;

    codet body = codet();
    if (get_block(stmt["body"], body))
      return true;

    convert_expression_to_code(body);

    code_whilet code_while;
    code_while.cond() = cond;
    code_while.body() = body;

    new_expr = code_while;
    break;
  }
  case SolidityGrammar::StatementT::DoWhileStatement:
  {
    exprt cond = true_exprt();
    if (get_expr(stmt["condition"], cond))
      return true;

    codet body = codet();
    if (get_block(stmt["body"], body))
      return true;

    convert_expression_to_code(body);

    code_dowhilet code_dowhile;
    code_dowhile.cond() = cond;
    code_dowhile.body() = body;

    new_expr = code_dowhile;
    break;
  }
  case SolidityGrammar::StatementT::ContinueStatement:
  {
    new_expr = code_continuet();
    break;
  }
  case SolidityGrammar::StatementT::BreakStatement:
  {
    new_expr = code_breakt();
    break;
  }
  case SolidityGrammar::StatementT::RevertStatement:
  {
    // e.g.
    // {
    //   "errorCall": {
    //     "nodeType": "FunctionCall",
    //   }
    //   "nodeType": "RevertStatement",
    // }
    if (!stmt.contains("errorCall") || get_expr(stmt["errorCall"], new_expr))
      return true;

    break;
  }
  case SolidityGrammar::StatementT::EmitStatement:
  {
    // treat emit as function call
    if (!stmt.contains("eventCall"))
    {
      log_error("Unexpected emit statement.");
      return true;
    }
    if (get_expr(stmt["eventCall"], new_expr))
      return true;

    break;
  }
  case SolidityGrammar::StatementT::PlaceholderStatement:
  {
    code_skipt placeholder;
    placeholder.set("#is_modifier_placeholder", true);
    new_expr = placeholder;
    break;
  }
  case SolidityGrammar::StatementT::TryStatement:
  {
    // Model try/catch as:
    //   if (nondet_bool()) { <success_block> } else { <catch_block(s)> }
    //
    // The external call result is nondeterministic since ESBMC verifies
    // one contract at a time and cannot resolve cross-contract calls.
    // Return variables in the success clause are assigned nondet values.

    if (
      !stmt.contains("clauses") || !stmt["clauses"].is_array() ||
      stmt["clauses"].size() < 2)
    {
      log_error(
        "TryStatement must have at least 2 clauses "
        "(success + catch)");
      return true;
    }

    const auto &clauses = stmt["clauses"];

    // --- success branch (first clause) ---
    const auto &success_clause = clauses[0];
    code_blockt success_block;

    // Declare return parameters with nondet initial values
    if (
      success_clause.contains("parameters") &&
      success_clause["parameters"].contains("parameters"))
    {
      for (const auto &param : success_clause["parameters"]["parameters"])
      {
        // Use get_var_decl to declare the variable in the symbol table
        exprt var_decl;
        if (get_var_decl(param, var_decl))
          return true;

        // The variable was declared; now assign it a nondet value
        // matching its type
        if (var_decl.is_code() && var_decl.statement() == "decl")
        {
          const symbolt &sym =
            *context.find_symbol(var_decl.op0().identifier());
          symbol_exprt sym_expr(sym.id, sym.type);

          exprt nondet_val;
          get_nondet_expr(sym.type, nondet_val);

          code_assignt assign(sym_expr, nondet_val);
          assign.location() = loc;

          success_block.copy_to_operands(var_decl);
          success_block.copy_to_operands(assign);
        }
        else
        {
          success_block.copy_to_operands(var_decl);
        }
      }
    }

    // Convert the success block body
    exprt success_body;
    if (get_block(success_clause["block"], success_body))
      return true;
    convert_expression_to_code(success_body);
    success_block.copy_to_operands(success_body);

    // --- catch branch(es) (remaining clauses) ---
    exprt catch_expr;
    if (clauses.size() == 2)
    {
      // Single catch clause
      const auto &cc = clauses[1];

      // Declare catch parameters if present (e.g. Error(string memory reason))
      code_blockt catch_block;
      if (cc.contains("parameters") && cc["parameters"].contains("parameters"))
      {
        for (const auto &param : cc["parameters"]["parameters"])
        {
          exprt var_decl;
          if (get_var_decl(param, var_decl))
            return true;

          if (var_decl.is_code() && var_decl.statement() == "decl")
          {
            const symbolt &sym =
              *context.find_symbol(var_decl.op0().identifier());
            symbol_exprt sym_expr(sym.id, sym.type);
            exprt nondet_val;
            get_nondet_expr(sym.type, nondet_val);
            code_assignt assign(sym_expr, nondet_val);
            assign.location() = loc;
            catch_block.copy_to_operands(var_decl);
            catch_block.copy_to_operands(assign);
          }
          else
          {
            catch_block.copy_to_operands(var_decl);
          }
        }
      }

      exprt catch_body;
      if (get_block(cc["block"], catch_body))
        return true;
      convert_expression_to_code(catch_body);
      catch_block.copy_to_operands(catch_body);
      catch_expr = catch_block;
    }
    else
    {
      // Multiple catch clauses: chain with nondet_bool
      // Build right-to-left: last clause is the final else
      const auto &last_cc = clauses[clauses.size() - 1];
      code_blockt last_block;
      if (
        last_cc.contains("parameters") &&
        last_cc["parameters"].contains("parameters"))
      {
        for (const auto &param : last_cc["parameters"]["parameters"])
        {
          exprt var_decl;
          if (get_var_decl(param, var_decl))
            return true;
          last_block.copy_to_operands(var_decl);
        }
      }
      exprt last_body;
      if (get_block(last_cc["block"], last_body))
        return true;
      convert_expression_to_code(last_body);
      last_block.copy_to_operands(last_body);

      catch_expr = last_block;

      // Build if-else chain from second-to-last back to first catch clause
      for (int i = static_cast<int>(clauses.size()) - 2; i >= 1; --i)
      {
        const auto &cc = clauses[i];
        code_blockt clause_block;
        if (
          cc.contains("parameters") && cc["parameters"].contains("parameters"))
        {
          for (const auto &param : cc["parameters"]["parameters"])
          {
            exprt var_decl;
            if (get_var_decl(param, var_decl))
              return true;
            clause_block.copy_to_operands(var_decl);
          }
        }
        exprt clause_body;
        if (get_block(cc["block"], clause_body))
          return true;
        convert_expression_to_code(clause_body);
        clause_block.copy_to_operands(clause_body);

        codet if_catch("ifthenelse");
        if_catch.copy_to_operands(nondet_bool_expr, clause_block, catch_expr);
        if_catch.location() = loc;
        catch_expr = if_catch;
      }
    }

    convert_expression_to_code(catch_expr);

    // Build top-level: if (nondet_bool()) { success } else { catch }
    codet try_if("ifthenelse");
    try_if.copy_to_operands(nondet_bool_expr, success_block, catch_expr);
    try_if.location() = loc;

    new_expr = try_if;
    break;
  }
  case SolidityGrammar::StatementT::InlineAssemblyStatement:
  {
    // Over-approximate inline assembly by havocing all externally referenced
    // variables. Assembly can read and write any referenced variable, so we
    // conservatively assign nondet values to each one.
    code_blockt havoc_block;

    if (
      stmt.contains("externalReferences") &&
      stmt["externalReferences"].is_array())
    {
      // Collect unique declaration IDs (a variable may appear multiple times)
      std::set<int> seen_decls;
      for (const auto &ref : stmt["externalReferences"])
      {
        if (!ref.contains("declaration"))
          continue;
        int decl_id = ref["declaration"].get<int>();
        if (!seen_decls.insert(decl_id).second)
          continue; // already processed

        // Skip .slot/.offset references — we'll havoc the variable itself
        if (ref.contains("isSlot") && ref["isSlot"].get<bool>())
          continue;
        if (ref.contains("isOffset") && ref["isOffset"].get<bool>())
          continue;

        const nlohmann::json &decl = find_decl_ref(decl_id);
        if (decl.empty() || decl["nodeType"] != "VariableDeclaration")
          continue;

        // Resolve the variable to a symbol expression
        bool is_state =
          decl.contains("stateVariable") && decl["stateVariable"].get<bool>();
        exprt var_expr;
        if (get_var_decl_ref(decl, is_state, var_expr))
          continue; // best-effort: skip if resolution fails

        // Assign nondet value
        exprt nondet_val;
        get_nondet_expr(var_expr.type(), nondet_val);
        code_assignt assign(var_expr, nondet_val);
        assign.location() = loc;
        havoc_block.copy_to_operands(assign);
      }

      // Also havoc variables referenced via .slot (state variables modified
      // through sstore). Find their declaration and havoc the variable.
      for (const auto &ref : stmt["externalReferences"])
      {
        if (!ref.contains("declaration"))
          continue;
        bool is_slot = ref.contains("isSlot") && ref["isSlot"].get<bool>();
        if (!is_slot)
          continue;

        int decl_id = ref["declaration"].get<int>();
        if (seen_decls.count(decl_id))
          continue; // already havoc'd above
        seen_decls.insert(decl_id);

        const nlohmann::json &decl = find_decl_ref(decl_id);
        if (decl.empty() || decl["nodeType"] != "VariableDeclaration")
          continue;

        exprt var_expr;
        if (get_var_decl_ref(decl, true, var_expr))
          continue;

        exprt nondet_val;
        get_nondet_expr(var_expr.type(), nondet_val);
        code_assignt assign(var_expr, nondet_val);
        assign.location() = loc;
        havoc_block.copy_to_operands(assign);
      }
    }

    if (havoc_block.operands().empty())
    {
      // No external references — assembly only touches internal EVM state.
      // Generate a skip (no-op).
      new_expr = code_skipt();
    }
    else
    {
      new_expr = havoc_block;
    }
    break;
  }
  case SolidityGrammar::StatementT::StatementTError:
  default:
  {
    log_error(
      "Unimplemented Statement type in rule statement. Got {}",
      SolidityGrammar::statement_to_str(type));
    return true;
  }
  }

  log_debug(
    "solidity", "finish statement {}", SolidityGrammar::statement_to_str(type));
  new_expr.location() = loc;
  return false;
}
