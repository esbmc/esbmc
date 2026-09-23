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

// Does `e` mention the symbol `id` anywhere in its tree?
static bool expr_mentions_symbol(const exprt &e, const irep_idt &id)
{
  if (e.id() == "symbol" && e.identifier() == id)
    return true;
  for (const auto &op : e.operands())
    if (expr_mentions_symbol(op, id))
      return true;
  return false;
}

void solidity_convertert::hoist_operands_read_by(
  const exprt &cond,
  std::size_t front_base,
  code_blockt &hoisted)
{
  // WHICH PENDING STATEMENTS MAY BE LIFTED ABOVE A BRANCH, AND WHICH MAY NOT.
  //
  // Converting a control statement's CONDITION queues statements on the shared
  // front block, and they are not all of one kind. Two live in the same queue
  // with OPPOSITE placement requirements:
  //
  //   (a) operand materialisation. `b == bytes32(uint256(1))` lowers to
  //       `bytes_static_equal(&b, &_ESBMC_aux18)` plus a queued
  //       `_ESBMC_aux18 = bytes_static_from_uint(1, 32)`. The condition READS
  //       the temporary, so leaving it behind emits a comparison against an
  //       unbuilt struct. MEASURED: it constrained `b` not at all, and ESBMC
  //       accepted one bytes32 equal to both bytes32(1) and bytes32(2).
  //
  //   (b) a guarded check. `k < 2 && b[k] == 0` queues the bounds assertion
  //       for `b[k]`, which the chain only evaluates when `k < 2`. Lifting it
  //       above the branch makes it unconditional and reports a bounds
  //       violation for k >= 2 that no execution performs -- a false positive,
  //       which in this pipeline is a RED generated test.
  //
  // A first attempt lifted the whole queue and traded (a) for (b): it fixed the
  // bytesN hole and broke `local_array_bounds_shortcircuit_guard_pass`. The two
  // cannot be told apart by statement kind without sniffing, and sniffing is
  // how a rule ends up right for the shapes that were thought of.
  //
  // THE PROPERTY THAT SEPARATES THEM IS IN THE DATA. (a) exists BECAUSE the
  // condition reads it -- the condition expression literally contains
  // `&_ESBMC_aux18`. (b) declares nothing the condition mentions: the bounds
  // assertion introduces no name, and the array-length temporary it does
  // introduce is read by the assertion, not by `k < 2 && b[k] == 0`. So the
  // test is "does the condition reference the symbol this statement declares",
  // asked of the converted expression rather than of the statement's shape.
  //
  // Everything not lifted stays queued and keeps its previous placement
  // exactly, so this narrows what moves rather than changing where the rest
  // goes. Note the pre-existing asymmetry this does NOT repair: a brace-less
  // body never reaches get_block, so its (b) statements were already being
  // flushed above the branch. That is the same defect in the other spelling and
  // is left alone here rather than fixed blind.
  //
  // CALLED FROM `if` AND `while` ONLY. The other two loop forms were MEASURED
  // rather than assumed, and the measurement contradicted the guess in both
  // directions:
  //
  //   for        STILL DEFECTIVE. `for (; b == bytes32(uint256(1)) &&
  //              b == bytes32(uint256(2)); )` runs its body -- the model admits
  //              one bytes32 equal to two constants. Reproduced at --unwind 4
  //              and 8 as well as the default, so it is not a bound artefact.
  //   do-while   NOT DEFECTIVE, and for a reason worth stating: a do-while's
  //              BODY executes before its condition, so `get_block` draining
  //              the pending queue into the body lands the temporary before the
  //              use rather than after it. The very mechanism that broke the
  //              braced `if` makes this form accidentally right.
  //
  // (The do-while probe was first read as vacuous -- a loop bound could make
  // its assertion hold without the fix -- and was re-run at two higher bounds
  // against the `for` case as a paired control, which stayed FAILED at each.
  // Without that control "SUCCESSFUL" would not have been a result.)
  //
  // `for` is left alone here rather than fixed blind, because it needs its own
  // regression and its own answer to where a LOOP condition's operands belong.
  // `while` above already had to choose "once, before the loop", and that
  // choice is wrong for a condition operand that depends on state the body
  // mutates. Extending the helper without settling that spreads one
  // half-answer to another place.
  code_blockt &fblk =
    current_functionDecl ? expr_frontBlockDecl : ctor_frontBlockDecl;
  auto &fops = fblk.operands();
  if (fops.size() <= front_base)
    return;

  exprt::operandst keep;
  std::vector<irep_idt> moved_syms;
  for (std::size_t i = front_base; i < fops.size(); ++i)
  {
    const exprt &op = fops[i];
    bool read_by_cond = false;
    if (op.is_code() && op.statement() == "decl" && !op.operands().empty())
      read_by_cond = expr_mentions_symbol(cond, op.op0().identifier());
    if (read_by_cond)
    {
      moved_syms.push_back(op.op0().identifier());
      exprt moved = op;
      convert_expression_to_code(moved);
      hoisted.operands().push_back(moved);
    }
    else
      keep.push_back(op);
  }

  // THE PARTITION IS SOUND ONLY IF NOTHING LEFT BEHIND READS SOMETHING MOVED.
  //
  // Order is preserved WITHIN the moved group and WITHIN the kept group, but
  // not BETWEEN them: the kept entries stay on the front block and are flushed
  // where they always were, while the moved ones are emitted in the wrapper
  // block that carries the branch, i.e. afterwards. So a pending statement that
  // originally came AFTER a moved decl and READS it would now run BEFORE it --
  // use before definition, which is precisely the defect this function exists
  // to remove, reintroduced one level up.
  //
  // No input is known to produce that shape, and searching for one would prove
  // nothing if the search came back empty. So the requirement is checked at run
  // time instead, and the whole regression suite becomes the search: if the
  // shape occurs anywhere it stops here and names both statements, rather than
  // emitting a program whose temporaries are out of order and letting the
  // result be read as a verdict about the contract.
  for (const auto &k : keep)
    for (const auto &sym : moved_syms)
      if (expr_mentions_symbol(k, sym))
      {
        log_error(
          "solidity frontend: hoisting the condition's operand temporaries "
          "would reorder them past a pending statement that reads one. "
          "Symbol `{}` is declared by a statement lifted above the branch, "
          "and the statement `{}`, which stays behind, mentions it. Emitting "
          "this would place a use before its definition.",
          sym.as_string(),
          k.pretty());
        abort();
      }

  fops.resize(front_base);
  for (auto &k : keep)
    fops.push_back(k);
}

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
  // Route through the decl-aware overload when the parameter AST carries a
  // typeName node. Mirrors get_parameter_list (used for return types) and
  // get_var_decl (used for locals); both reach get_array_pointer_type for
  // array shapes, which lowers fixed and dynamic arrays uniformly to
  // pointers. The plain `pd["typeDescriptions"]` form falls into the
  // typeIdentifier-string parser, which builds a real array_typet for
  // fixed-outer NestedArrayTypeName and disagrees with the local-var /
  // call-site representation — see `f(uint[3][2] calldata s)`, where the
  // caller passes `uint**` but the parameter symbol used to be `uint[3][2]`.
  typet param_type;
  if (pd.contains("typeName"))
  {
    if (get_type_description(
          pd, pd["typeName"]["typeDescriptions"], param_type))
      return true;
  }
  else
  {
    if (get_type_description(pd["typeDescriptions"], param_type))
      return true;
  }

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

  // Carry the fixed-bytes width on the argument itself (not just its type):
  // bytesN lowers to the shared `BytesStatic` struct, and the per-parameter
  // width `#sol_bytesn_size` set on the type is dropped when the parameter
  // type is later resolved/migrated to a plain struct. Stamping it on the
  // code_typet argument keeps it readable from the function signature (used by
  // the Foundry testcase generator to render an exact-width `bytesN(..)`).
  {
    const std::string bn = param_type.get("#sol_bytesn_size").as_string();
    if (!bn.empty())
      param.set("#sol_bytesn_size", bn);
    // A user-defined value type over bytesN loses its `#sol_udvt_name` in the
    // same migration; carry it on the argument so the generator still renders
    // `Name.wrap(bytesN(..))` rather than a bare (unassignable) bytesN literal.
    const std::string udvt = param_type.get("#sol_udvt_name").as_string();
    if (!udvt.empty())
      param.set("#sol_udvt_name", udvt);
  }

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
      // Carry the source-return marker across the location overwrite below.
      // `cl` is rebuilt from the AST node, so assigning it would erase the flag
      // that get_statement's ReturnStatement arm just stamped — silently, and
      // only for returns that are direct children of a block (a brace-less
      // `if (c) return x;` goes through a different arm and would keep it).
      // A marker that survives in one syntactic position and not the other is
      // worse than no marker at all.
      if (statement.location().get_bool("sol_source_return"))
        cl.set("sol_source_return", true);
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

      // Per-rollback granularity for B1's revert state-rollback:
      // OR `statement_is_mutation_top_level` of the AST statement we
      // just lowered into `current_function_seen_mutation` so the next
      // `build_revert_rollback_block` invocation knows whether to
      // emit the full `*this = save; return [nondet]` form (mutation
      // observed earlier) or the cheap `return [nondet]` early-return
      // form.  Done after the lowering so any require/revert *inside*
      // this statement reads the prior value of the flag (the
      // mutation in this statement only counts toward subsequent
      // statements).  The flag is itself reset on entry to each
      // function in get_function_definition.
      if (statement_is_mutation_top_level(stmt_kv.value()))
        current_function_seen_mutation = true;

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
    // Bare type-valued expressions such as `Arst.Foo;` or `C;` have no
    // runtime effect — their typeString is `type(...)`. Trying to lower
    // them through the normal ContractMemberCall / Identifier paths
    // trips on JSON fields that only exist for value-producing member
    // accesses (e.g. a struct-definition member looked up via
    // `contract_var.structName`). Skip them cleanly here.
    {
      const nlohmann::json &inner = stmt["expression"];
      if (
        inner.contains("typeDescriptions") &&
        inner["typeDescriptions"].contains("typeString") &&
        inner["typeDescriptions"]["typeString"].is_string())
      {
        const std::string ts =
          inner["typeDescriptions"]["typeString"].get<std::string>();
        if (ts.compare(0, 5, "type(") == 0)
        {
          new_expr = code_skipt();
          break;
        }
      }
    }
    if (get_expr(
          stmt["expression"], stmt["expression"]["typeDescriptions"], new_expr))
      return true;
    // Bare expression-statement forms that reduce to no-op exprt shapes
    // (e.g. `MyAddress.wrap;` — a MemberAccess naming a UDVT wrap/unwrap
    // function reference without invoking it) produce a typecast_exprt
    // whose op0 is never populated by the wrap/unwrap handler in
    // solidity_convert_ref.cpp (the comment there says "we will set the
    // op0 later"; "later" only happens inside the FunctionCall path).
    // typecast_exprt's constructor leaves op0 as a default-constructed
    // empty exprt, which has no meaningful side effect and trips
    // migrate_expr during GOTO generation. Replace with a skip.
    if (
      new_expr.id() == "typecast" &&
      (new_expr.operands().empty() || new_expr.op0().id().as_string().empty()))
      new_expr = code_skipt();
    // Bare value-expression statements (`this;`, `hax;`, `tuple_instance$0;`)
    // are used in Solidity to suppress unused-variable / state-mutability
    // warnings; they have no side effects. Without this, they reach symex
    // as raw symbol exprs in OTHER instructions and trip
    // `goto_symext: unexpected statement: symbol`.
    if (new_expr.id() == "symbol")
      new_expr = code_skipt();
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

    // FRONTEND CONTRACT: a normal exit must carry POSITIVE evidence.
    //
    // Downstream (complete-path coverage) has to tell an ordinary exit from a
    // reverting one. The only positive evidence available was "did this path
    // walk the function epilogue?", and the epilogue is emitted AFTER the
    // RETURN — which a returning path never reaches. So every value-returning
    // function's normal exit was classified `undetermined`, and since getters
    // and views all return values, that is most of a real contract.
    //
    // The alternative that was rejected is the negative inference "no revert
    // marker was seen, therefore normal". That is exactly the shape of
    // reasoning that produced the RETURN-exit bug, and the revert-observation
    // gate has declared scope gaps (constructors), so absence of a mark is not
    // evidence.
    //
    // This is the positive half: the frontend knows which RETURNs it lowered
    // from a source `return` and which it synthesised itself for a failing
    // `require` (`{ *this = _sol_save_this; return [nondet]; }`). Only the
    // former is marked. Downstream then has an affirmative fact rather than an
    // absence.
    //
    // Deliberately a LOCATION FLAG rather than a marker CALL symmetric to
    // `_ESBMC_sol_mark_revert`. A call is an extra instruction in a program
    // whose paths are being counted: it would have to be excluded from the
    // decision set, from the inliner's callee predicate and from the exit
    // census, and each exclusion is a place to get it wrong. A location flag
    // changes the goto program's SHAPE not at all, which for a pass that
    // exists to count paths is worth more than the symmetry.
    // Stamped in TWO places on purpose, because the location of a statement is
    // reassigned twice on its way out and either assignment silently drops it:
    // get_statement overwrites `new_expr.location()` with `loc` before
    // returning, and get_block then overwrites it again with its own `cl`.
    // MEASURED — the exprt-only stamp arrived as `flag=true` and was read back
    // `false` one call later.
    //
    // So: mark `loc` (which get_statement's own assignment carries), mark the
    // exprt (for the arms that `return false` early and never reach that
    // assignment), and carry the flag across `cl` in get_block. Each covers a
    // different exit route; a marker that survives only some of them is worse
    // than none, because the resulting classification would depend on where the
    // `return` happens to sit syntactically.
    loc.set("sol_source_return", true);
    auto mark_source_return = [](exprt &e) {
      e.location().set("sol_source_return", true);
    };

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
      mark_source_return(ret_expr);
      new_expr = ret_expr;
      return false;
    }
    assert(stmt["expression"].contains("nodeType"));

    // get_type_description
    typet return_exrp_type;
    if (get_type_description(
          stmt["expression"]["typeDescriptions"], return_exrp_type))
      return true;

    // `return f();` where f is itself void: solc gives the call expression
    // typeString "tuple()" (an empty tuple). The function being compiled
    // still has zero return parameters, so there is no tuple_instance to
    // assign into — emit the inner call as a statement and a bare `return;`.
    if (
      stmt["expression"].contains("typeDescriptions") &&
      stmt["expression"]["typeDescriptions"].value("typeString", "") ==
        "tuple()")
    {
      exprt inner;
      if (get_expr(stmt["expression"], inner))
        return true;
      code_blockt block;
      convert_expression_to_code(inner);
      block.copy_to_operands(inner);
      code_returnt void_ret;
      mark_source_return(void_ret);
      block.copy_to_operands(void_ret);
      new_expr = block;
      return false;
    }

    if (
      get_sol_type(return_exrp_type) == SolidityGrammar::SolType::TUPLE_RETURNS)
    {
      // `return c ? (a, b) : (x, y);` -- a Conditional whose branches are
      // tuples. Lower it to `if (c) return (a, b); else return (x, y);` by
      // re-entering get_statement on a synthesized IfStatement whose branches
      // are copies of this Return node with the branch expression swapped in,
      // so each branch takes the existing tuple-return path. Previously a
      // CONVERSION ERROR that aborted the whole file (perennial Product:
      // PositionLib.settled).
      if (
        stmt["expression"]["nodeType"].get<std::string>() == "Conditional" &&
        stmt["expression"].contains("trueExpression") &&
        stmt["expression"].contains("falseExpression"))
      {
        nlohmann::json ret_t = stmt, ret_f = stmt;
        ret_t["expression"] = stmt["expression"]["trueExpression"];
        ret_f["expression"] = stmt["expression"]["falseExpression"];
        nlohmann::json if_stmt = {
          {"nodeType", "IfStatement"},
          {"src", stmt["src"]},
          {"condition", stmt["expression"]["condition"]},
          {"trueBody", ret_t},
          {"falseBody", ret_f}};
        if (stmt.contains("id"))
          if_stmt["id"] = stmt["id"];
        return get_statement(if_stmt, new_expr);
      }
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
        bool rhs_is_nondet = false;
        if (get_tuple_function_ref(stmt["expression"]["expression"], rhs))
        {
          // Builtin tuple-producing callee (e.g. `addr.call(...)` returning
          // `(bool, bytes memory)`) has no referencedDeclaration and no
          // tuple_instance symbol. Fall back to nondet-per-member.
          rhs_is_nondet = true;
        }

        // add function call
        //
        // Only when we have a real tuple instance for the callee (i.e.
        // !rhs_is_nondet) do we need to materialise the function call
        // and emit it via get_tuple_function_call. For builtin tuple-
        // producing callees (`addr.call(...)`, `abi.decode(data, (T,U))`
        // etc.) the result is over-approximated by independent nondet
        // per member, so the call exprt itself is never used downstream.
        // Skipping get_expr in that case also avoids migrate failures
        // when the builtin's lowered return type (e.g. abi_decode's
        // single-uint256 identity) does not match the Solidity tuple
        // shape expected by the rest of the return path.
        exprt func_call;
        if (!rhs_is_nondet)
        {
          if (get_expr(
                stmt["expression"],
                stmt["expression"]["typeDescriptions"],
                func_call))
            return true;
          get_tuple_function_call(func_call);
        }

        size_t ls = to_struct_type(lhs.type()).components().size();
        size_t rs =
          rhs_is_nondet ? ls : to_struct_type(rhs.type()).components().size();
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

          // rop: struct member call (e.g. tupleB.men0), or nondet fallback
          exprt rop;
          if (rhs_is_nondet)
          {
            get_nondet_expr(lop.type(), rop);
          }
          else if (get_tuple_member_call(
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
      mark_source_return(return_expr);
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
      // Previously this block asserted that
      // current_functionDecl.returnParameters.id ==
      // stmt.functionReturnParameters, to catch cases where a Return node
      // refers to a different ParameterList than its enclosing function.
      // That invariant does not hold on real-world solc 0.6.x output for
      // contracts that inline base-contract function bodies via modifiers
      // or internal trampolines (observed on 1inch Mooniswap). Type
      // extraction below only uses ret_params_src, so the id mismatch is
      // harmless — rely on get_type_description and drop the assert.
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

    // If rhs is an inline array literal (id("array")) being returned
    // from a fixed-size Solidity array return type, the frontend models
    // the return type as a pointer and c_typecast's array→pointer decay
    // produces `&{...}[0]` — an address-of indexing into an inline
    // initializer that the dereference pass cannot simplify, recursing
    // unboundedly on member/index chains nested under the literal.
    //
    // Materialize the literal into a static auxiliary array first, then
    // let the regular implicit cast turn that aux symbol into a normal
    // pointer.
    if (
      rhs.id() == "array" && return_type.id() == typet::id_pointer &&
      rhs.type().is_array())
    {
      const typet &elem_type = return_type.subtype();
      for (auto &op : rhs.operands())
        solidity_gen_typecast(ns, op, elem_type);
      // Make sure the array's element type matches the destination
      // before stamping it into a static aux symbol.
      to_array_type(rhs.type()).subtype() = elem_type;
      exprt aux;
      get_aux_array(rhs, elem_type, aux);
      rhs = aux;
    }
    // `return "literal";` from a function returning `bytes memory` produces
    // a plain string_constantt (array of signed char). c_typecast cannot
    // convert that to a BytesDynamic struct, so the raw byte array would
    // leak into the GOTO `RETURN` and crash symex on the type mismatch.
    // Narrow the repair to cases where rhs is actually a string constant
    // (array of signed char) so that builtins already returning a pointer
    // or BytesDynamic keep their existing path.
    bool rhs_is_string_constant =
      rhs.id() == "string-constant" ||
      (rhs.type().is_array() && rhs.type().subtype().is_signedbv() &&
       to_signedbv_type(rhs.type().subtype()).get_width() == 8);
    // Same OVER-approximation for "scalar uint256 returned where the
    // function declares `bytes memory`": canonical case is
    //   function f() ... returns (bytes memory) {
    //     return abi.encodeCall(this.g, (...));
    //   }
    // where abi.encodeCall is the uint256 identity in solidity_abi.c.
    // c_typecast cannot promote a 256-bit scalar into a BytesDynamic
    // struct, and any naive cast leaks the raw int through `RETURN`
    // — symex then segfaults on the struct-vs-scalar shape mismatch
    // when the caller assigns it into a BytesDynamic temporary.
    bool rhs_is_scalar_to_bytes =
      get_sol_type(return_type) == SolidityGrammar::SolType::BYTES_DYN &&
      !rhs.type().is_pointer() && !rhs.type().is_array() &&
      !rhs.type().is_struct() && !rhs_is_string_constant;
    if (
      get_sol_type(return_type) == SolidityGrammar::SolType::BYTES_DYN &&
      (rhs_is_string_constant || rhs_is_scalar_to_bytes))
    {
      // convert_type_expr aborts when no containing contract supplies a
      // dynamic pool (e.g. free functions declared outside any contract
      // via `using { f } for T;`). In that case the precise string→bytes
      // conversion is not representable, so we over-approximate the
      // return value with llc_nondet_bytes() — a BytesDynamic with
      // fully-nondet length (post-T1.2) and initialized == 1.
      // [APPROX: OVER] Recorded in approximation ledger #21.
      exprt dummy_pool;
      if (get_dynamic_pool(stmt["expression"], dummy_pool))
      {
        locationt loc;
        get_start_location_from_stmt(stmt, loc);
        side_effect_expr_function_callt nondet_b;
        get_library_function_call_no_args(
          "llc_nondet_bytes",
          "c:@F@llc_nondet_bytes",
          return_type,
          loc,
          nondet_b);
        rhs = nondet_b;
      }
      else
        convert_type_expr(ns, rhs, return_type, stmt["expression"]);
    }
    else if (
      get_sol_type(return_type) == SolidityGrammar::SolType::BYTES_STATIC &&
      !rhs.type().is_struct())
    {
      // `return 0;` (or any scalar) from a function declared to return
      // bytesN. The bytesN return type is modeled as a BytesStatic struct,
      // but the literal/scalar rhs is an integer. solidity_gen_typecast
      // would emit a raw typecast(int -> struct); the value-set analysis
      // then walks make_member through that typecast onto a non-struct
      // operand and aborts (Release: SIGSEGV) in value_sett::make_member.
      // Route through convert_type_expr, which lowers int -> BytesStatic
      // via bytes_static_from_uint, producing a real struct value. Mirrors
      // the BYTES_DYN handling above.
      convert_type_expr(ns, rhs, return_type, stmt["expression"]);
    }
    else
      solidity_gen_typecast(ns, rhs, return_type);
    ret_expr.return_value() = rhs;
    mark_source_return(ret_expr);

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
    // Snapshot pending front/back-block sizes so any statement queued while
    // converting the (possibly brace-less) body is kept inside the loop scope
    // rather than leaking before the loop — see flush_pending_into_body.
    std::size_t for_front_base =
      (current_functionDecl ? expr_frontBlockDecl : ctor_frontBlockDecl)
        .operands()
        .size();
    std::size_t for_back_base =
      (current_functionDecl ? expr_backBlockDecl : ctor_backBlockDecl)
        .operands()
        .size();
    if (stmt.contains("body"))
      if (get_statement(stmt["body"], body))
        return true;

    flush_pending_into_body(body, for_front_base, for_back_base);

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
    const std::size_t cond_front_base =
      (current_functionDecl ? expr_frontBlockDecl : ctor_frontBlockDecl)
        .operands()
        .size();
    exprt cond;
    if (get_expr(stmt["condition"], cond))
      return true;
    // Operand temporaries the condition READS must be built before the branch
    // tests them; everything else the condition queued keeps its placement.
    code_blockt cond_hoisted;
    hoist_operands_read_by(cond, cond_front_base, cond_hoisted);

    // 2. Then: make a exprt for trueBody
    codet then = code_skipt();
    // Keep any bounds-check/decl queued by the (possibly brace-less) then-body
    // under the branch guard instead of leaking it before the `if`.
    std::size_t then_front_base =
      (current_functionDecl ? expr_frontBlockDecl : ctor_frontBlockDecl)
        .operands()
        .size();
    std::size_t then_back_base =
      (current_functionDecl ? expr_backBlockDecl : ctor_backBlockDecl)
        .operands()
        .size();
    if (get_statement(stmt["trueBody"], then))
      return true;

    flush_pending_into_body(then, then_front_base, then_back_base);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(cond, then);

    // 3. Else: make a exprt for "falseBody" if the if-statement node contains an "else" block.
    // solc 0.6.x always emits the field, with `null` when there is no else;
    // 0.8.x omits it. Treat both as "no else".
    if (stmt.contains("falseBody") && !stmt["falseBody"].is_null())
    {
      codet else_expr = code_skipt();
      std::size_t else_front_base =
        (current_functionDecl ? expr_frontBlockDecl : ctor_frontBlockDecl)
          .operands()
          .size();
      std::size_t else_back_base =
        (current_functionDecl ? expr_backBlockDecl : ctor_backBlockDecl)
          .operands()
          .size();
      if (get_statement(stmt["falseBody"], else_expr))
        return true;

      flush_pending_into_body(else_expr, else_front_base, else_back_base);
      if_expr.copy_to_operands(else_expr);
    }

    if (cond_hoisted.operands().empty())
      new_expr = if_expr;
    else
    {
      // THE BRANCH MUST CARRY ITS OWN LOCATION BEFORE IT IS WRAPPED.
      //
      // get_statement ends with `new_expr.location() = loc`, which was the only
      // thing giving this `ifthenelse` a source location. Wrapping it makes
      // `new_expr` the BLOCK, so the block gets `loc` and the branch inside
      // gets nothing -- the goto program then reads
      //
      //     // 3132 no location
      //     IF !(_Bool)return_value$_bytes_static_equal$2 THEN GOTO 1
      //
      // and branch coverage, which identifies a branch by its source location,
      // reports "No branch detected" and generates 0 VCCs. MEASURED: it silently
      // emptied `foundry_covgen_bytesN_fail`, whose whole subject is the branch
      // `if (x == bytes4(0x12345678))`. A wrapper that loses the location does
      // not break the branch, it makes it invisible to the pass that counts
      // branches -- which for this project is worse.
      if_expr.location() = loc;
      // The temporaries and then the branch, as one statement, so the enclosing
      // block cannot separate them.
      cond_hoisted.copy_to_operands(if_expr);
      new_expr = cond_hoisted;
    }
    break;
  }
  case SolidityGrammar::StatementT::WhileStatement:
  {
    const std::size_t wc_front_base =
      (current_functionDecl ? expr_frontBlockDecl : ctor_frontBlockDecl)
        .operands()
        .size();
    exprt cond = true_exprt();
    if (get_expr(stmt["condition"], cond))
      return true;
    code_blockt wc_hoisted;
    hoist_operands_read_by(cond, wc_front_base, wc_hoisted);

    codet body = codet();
    if (get_block(stmt["body"], body))
      return true;

    convert_expression_to_code(body);

    code_whilet code_while;
    code_while.cond() = cond;
    code_while.body() = body;

    if (wc_hoisted.operands().empty())
      new_expr = code_while;
    else
    {
      // ONE EVALUATION, BEFORE THE LOOP -- and that is a real restriction, not
      // a free choice. The temporaries are built once and the loop then tests
      // the condition against them on every iteration, which is right for the
      // constant operands this handles (`bytes32(uint256(1))`) and WRONG for a
      // condition operand that depends on state the body mutates. Nothing here
      // detects that case; it is stated so the limit is visible rather than
      // discovered. The previous behaviour built them inside the body, i.e.
      // after the first test, which is wrong for every shape including the
      // constant one.
      //
      // Location stamped before wrapping, for the same reason as the
      // IfStatement arm: get_statement's trailing `new_expr.location() = loc`
      // would otherwise land on the wrapper and leave the loop with none, which
      // is invisible to every pass that identifies a construct by location.
      code_while.location() = loc;
      wc_hoisted.copy_to_operands(code_while);
      new_expr = wc_hoisted;
    }
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
    if (!stmt.contains("errorCall"))
      return true;

    // Revert observation: route `revert CustomError(...)` through the same
    // rollback+mark lowering as `revert()` so __ESBMC_reverted() sees it (the
    // error arguments are pure and dropped).  Falls back to the legacy
    // errorCall lowering (custom-error body = __ESBMC_assume(false), which
    // prunes the path) when the feature is off or the scope is non-observable
    // (constructor / library).  See docs/claude/solidity/revert-observation.md.
    if (uses_revert_observation)
    {
      exprt rollback;
      if (!build_revert_rollback_block(nullptr, rollback))
      {
        new_expr = rollback;
        break;
      }
    }

    if (get_expr(stmt["errorCall"], new_expr))
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
    // Model try/catch.  Two lowerings, selected by the revert-observation gate:
    //
    //  - DEFAULT (no opt-in): a free nondet branch
    //      if (nondet_bool()) { <externalCall>; <success_body> }
    //      else               { <catch_block(s)> }
    //    The success arm executes the call; the catch arm is a nondet
    //    alternative NOT correlated with whether the call actually reverted.
    //    This is unsound for revert-rule properties but is the legacy
    //    behaviour, kept byte-for-byte for units that do not opt in
    //    (preserves k-induction stability).
    //
    //  - OPT-IN (source declares the `__ESBMC_reverted` stub, so
    //    `uses_revert_observation` is set): a revert-CORRELATED lowering.  The
    //    call is hoisted out of the arms and the branch tests the real revert
    //    flag with a save / clear / snapshot / restore discipline, so the catch
    //    arm is entered iff the call reverted within the captured scope, and the
    //    single global flag behaves like a per-`try` scoped observation (no
    //    contamination from nested calls).  See
    //    docs/claude/solidity/revert-observation.md and plan.md.
    //
    // Return values stay nondet in the success bindings in both lowerings
    // (cross-contract return resolution is out of scope for the AST frontend).

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
    const auto &success_clause = clauses[0];

    // Append the success return-param nondet decls + the user-written success
    // body to `out`.  When `include_call`, the external call is emitted first
    // (legacy shape: call lives inside the success arm).
    auto build_success_arm = [&](code_blockt &out, bool include_call) -> bool {
      if (include_call && stmt.contains("externalCall"))
      {
        exprt call_expr;
        if (get_expr(stmt["externalCall"], call_expr))
          return true;
        convert_expression_to_code(call_expr);
        out.copy_to_operands(call_expr);
      }
      if (
        success_clause.contains("parameters") &&
        success_clause["parameters"].contains("parameters"))
      {
        for (const auto &param : success_clause["parameters"]["parameters"])
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
            out.copy_to_operands(var_decl);
            out.copy_to_operands(assign);
          }
          else
            out.copy_to_operands(var_decl);
        }
      }
      exprt success_body;
      if (get_block(success_clause["block"], success_body))
        return true;
      convert_expression_to_code(success_body);
      out.copy_to_operands(success_body);
      return false;
    };

    // Build the catch arm: single clause, or multiple clauses chained with a
    // nondet selector (ESBMC cannot tell Error/Panic/low-level apart, so which
    // handler runs is over-approximated as nondet).
    auto build_catch_expr = [&](exprt &catch_expr) -> bool {
      if (clauses.size() == 2)
      {
        const auto &cc = clauses[1];
        code_blockt catch_block;
        if (
          cc.contains("parameters") && cc["parameters"].contains("parameters"))
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
              catch_block.copy_to_operands(var_decl);
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

        for (int i = static_cast<int>(clauses.size()) - 2; i >= 1; --i)
        {
          const auto &cc = clauses[i];
          code_blockt clause_block;
          if (
            cc.contains("parameters") &&
            cc["parameters"].contains("parameters"))
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
      return false;
    };

    if (!uses_revert_observation)
    {
      // ---- Legacy nondet lowering (byte-for-byte unchanged) ----
      // Same operations in the same order as before: success arm (with the
      // call) then catch arm, joined by a nondet branch.
      code_blockt success_block;
      if (build_success_arm(success_block, /*include_call=*/true))
        return true;
      exprt catch_expr;
      if (build_catch_expr(catch_expr))
        return true;

      codet try_if("ifthenelse");
      try_if.copy_to_operands(nondet_bool_expr, success_block, catch_expr);
      try_if.location() = loc;
      new_expr = try_if;
      break;
    }

    // ---- Opt-in revert-correlated lowering ----
    // The flag model is linked whenever uses_revert_observation; hard-error
    // (never fall back to the unsound nondet branch) if it is missing.
    const symbolt *flag_sym =
      context.find_symbol("c:@_ESBMC_sol_reverted_flag");
    if (
      flag_sym == nullptr ||
      context.find_symbol("c:@F@_ESBMC_sol_clear_revert") == nullptr)
    {
      log_error(
        "try/catch revert observation requires the _ESBMC_sol_reverted_flag "
        "model, but its symbols were not found");
      return true;
    }
    exprt flag_expr = symbol_expr(*flag_sym);
    const typet flag_t = flag_sym->type;
    const std::string dbg_mod = get_modulename_from_path(absolute_path);

    // Fresh bool temp matching the flag type.
    auto make_flag_temp = [&]() -> symbol_exprt {
      std::string nm, id;
      get_aux_var(nm, id);
      symbolt s;
      get_default_symbol(s, dbg_mod, flag_t, nm, id, loc);
      s.lvalue = true;
      s.file_local = true;
      const symbolt &added = *move_symbol_to_context(s);
      return symbol_exprt(added.id, added.type);
    };

    code_blockt try_block;

    // (A) save caller's prior revert status.
    symbol_exprt saved = make_flag_temp();
    try_block.copy_to_operands(code_declt(saved));
    {
      code_assignt a(saved, flag_expr);
      a.location() = loc;
      try_block.copy_to_operands(a);
    }

    // (B) clean baseline for THIS call.
    {
      exprt clear_stmt;
      build_revert_flag_call(
        "_ESBMC_sol_clear_revert",
        "c:@F@_ESBMC_sol_clear_revert",
        loc,
        clear_stmt);
      try_block.copy_to_operands(clear_stmt);
    }

    // (C) the external call, ALWAYS executed.  Drain the call's front/back
    // blocks here so its wrapper code cannot leak before (B) (where the outer
    // get_block would otherwise flush it, wiping the mark — see plan §2.2).
    if (stmt.contains("externalCall"))
    {
      exprt call_expr;
      if (get_expr(stmt["externalCall"], call_expr))
        return true;
      for (auto &op : expr_frontBlockDecl.operands())
      {
        convert_expression_to_code(op);
        try_block.copy_to_operands(op);
      }
      expr_frontBlockDecl.clear();
      convert_expression_to_code(call_expr);
      try_block.copy_to_operands(call_expr);
      for (auto &op : expr_backBlockDecl.operands())
      {
        convert_expression_to_code(op);
        try_block.copy_to_operands(op);
      }
      expr_backBlockDecl.clear();
    }

    // (D) snapshot THIS call's revert outcome before either body runs.
    symbol_exprt reverted = make_flag_temp();
    try_block.copy_to_operands(code_declt(reverted));
    {
      code_assignt a(reverted, flag_expr);
      a.location() = loc;
      try_block.copy_to_operands(a);
    }

    // (E) restore caller's prior status (scoped observation).
    {
      code_assignt a(flag_expr, saved);
      a.location() = loc;
      try_block.copy_to_operands(a);
    }

    // (F) branch on the real outcome: if (!reverted) success else catch.
    code_blockt success_block;
    if (build_success_arm(success_block, /*include_call=*/false))
      return true;
    exprt catch_expr;
    if (build_catch_expr(catch_expr))
      return true;

    codet try_if("ifthenelse");
    not_exprt not_reverted(reverted);
    try_if.copy_to_operands(not_reverted, success_block, catch_expr);
    try_if.location() = loc;
    try_block.copy_to_operands(try_if);

    new_expr = try_block;
    break;
  }
  case SolidityGrammar::StatementT::InlineAssemblyStatement:
  {
    // T2.4 — gradual Yul lowering for inline assembly.
    //
    // Strategy: try precise lowering of the YulBlock against a fixed
    // supported subset (let / := / pure-256-bit builtins / if / switch /
    // for / break / continue / sload(X.slot) / sstore(X.slot, v) on
    // scalar state vars / nested blocks / number+bool literals).
    // All-or-nothing per block: if any unsupported Yul construct
    // (mload/mstore/calldata*/keccak256/call/return/revert/
    // EVM-intrinsics/Yul-fn-defs/leave/.offset/computed-slot
    // sload/sstore/non-scalar-state-var sload/sstore/multi-LHS/
    // hex-or-string-literals) appears anywhere in the block, fall through
    // to the legacy havoc fallback below.  This guarantees the precise
    // portion can never observe a write the havoc'd portion would have made.
    {
      std::string unsupported_kind, unsupported_src;
      exprt precise;
      if (try_lower_yul_block_precise(
            stmt, loc, precise, unsupported_kind, unsupported_src))
      {
        new_expr = precise;
        break;
      }
      // [APPROX: OVER] Block contains an unsupported Yul construct; emit
      // a single warning naming the offending node + location so users can
      // see why an assertion that depends on this block became a havoc.
      log_warning(
        "[approx] inline assembly at {}:{}: over-approximating - "
        "unsupported Yul construct '{}' ({}); supported subset: "
        "let / := / arithmetic+bitwise+shift / lt/gt/eq/slt/sgt/iszero / "
        "and/or/xor/not / shl/shr / if / switch / for / break / continue / "
        "sload(X.slot) / sstore(X.slot, v) on scalar state vars",
        loc.get_file().c_str(),
        loc.get_line().c_str(),
        unsupported_kind,
        unsupported_src);
    }

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

// ============================================================================
// T2.4 — Yul precise lowering for inline assembly blocks.
//
// Translates a supported subset of Yul into ESBMC IR with full precision.
// Supported subset:
//   - Statements:  YulBlock, YulVariableDeclaration, YulAssignment, YulIf,
//                  YulSwitch, YulForLoop, YulBreak, YulContinue,
//                  YulExpressionStatement (only sstore at statement level)
//   - Expressions: YulIdentifier, YulLiteral (number/bool), YulFunctionCall
//                  with a whitelisted builtin name
//   - Builtins:    add sub mul div mod addmod mulmod
//                  lt gt slt sgt eq iszero
//                  and or xor not
//                  shl shr
//                  sload  (only sload(X.slot) on scalar state vars)
//                  sstore (only sstore(X.slot, v) on scalar state vars)
//
// Unsupported (block falls back to havoc with a single warning):
//   - Memory/calldata/hashing/calls/returns/reverts/EVM intrinsics
//   - Yul function definitions, leave
//   - Multi-LHS YulAssignment / YulVariableDeclaration
//   - YulLiteral kinds other than number / bool
//   - YulCase with a non-literal selector
//   - .offset external references (no intra-slot packing model)
//   - Computed-slot sload/sstore (sload(0), sload(add(X.slot, 1)),
//     sload(keccak256(...)), etc.)
//   - sload/sstore on non-scalar state vars (struct/mapping/array/bytes/
//     contract-typed)
//
// All-or-nothing rule: pre-flight scans the entire YulBlock; the precise
// lowerer either translates everything or refuses (returns false), and the
// caller emits one warning + falls through to the existing havoc fallback.
// This is the soundness lever — a partial precise translation could miss
// writes the unsupported portion would make.
// ============================================================================

namespace
{
bool is_supported_yul_builtin(const std::string &name)
{
  static const std::set<std::string> ok = {
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "addmod",
    "mulmod",
    "lt",
    "gt",
    "slt",
    "sgt",
    "eq",
    "iszero",
    "and",
    "or",
    "xor",
    "not",
    "shl",
    "shr",
    // sload/sstore are only valid with a YulIdentifier `X.slot` argument
    // resolving to a scalar state variable; the sload/sstore lowering
    // paths in convert_yul_expression / convert_yul_statement enforce
    // those constraints at lowering time (the pre-flight scan does not
    // have access to externalReferences).
    "sload",
    "sstore"};
  return ok.count(name) != 0;
}

// Storage byte size for the value types supported by single-slot struct
// packing: unsigned integers (address is uint160 -> 20 bytes) and bool.
// Signed ints (unpack would need `signextend`, unsupported), bytesN (modelled
// as a BytesStatic struct, not a bitvector), and every reference type are
// rejected so the caller aborts precise lowering and falls back to havoc.
bool yul_pack_value_byte_size(
  const namespacet &ns,
  const typet &t,
  unsigned &bytes)
{
  typet rt = t;
  while (rt.id() == "symbol")
    rt = ns.follow(rt);
  if (rt.id() == "bool")
  {
    bytes = 1;
    return true;
  }
  if (rt.id() == "unsignedbv")
  {
    unsigned w = to_unsignedbv_type(rt).get_width();
    if (w == 0 || (w % 8) != 0 || w > 256)
      return false;
    bytes = w / 8;
    return true;
  }
  return false;
}

struct yul_slot_field
{
  exprt member;
  unsigned bitoffset;
  unsigned bitwidth;
};

// Compute the slot-0 field layout of `struct_lval` (of struct type `st`).
// Returns false (abort precise lowering) unless EVERY field is a supported
// value type AND the whole struct fits in one 32-byte slot: Solidity never
// straddles a field across a slot boundary, so a spill past 32 bytes means the
// struct is multi-slot -> outside the single-slot subset -> abort.
bool yul_pack_slot0_fields(
  const namespacet &ns,
  const exprt &struct_lval,
  const struct_typet &st,
  std::vector<yul_slot_field> &out)
{
  unsigned off = 0; // running byte offset within slot 0
  for (const auto &comp : st.components())
  {
    unsigned sz = 0;
    if (!yul_pack_value_byte_size(ns, comp.type(), sz))
      return false;
    if (off + sz > 32)
      return false; // spills past slot 0 -> multi-slot -> abort
    yul_slot_field f;
    f.member = member_exprt(struct_lval, comp.get_name(), comp.type());
    f.member.location() = struct_lval.location();
    f.bitoffset = off * 8;
    f.bitwidth = sz * 8;
    out.push_back(f);
    off += sz;
  }
  return !out.empty();
}
} // namespace

bool solidity_convertert::yul_node_is_supported(
  const nlohmann::json &node,
  std::string &unsupported_kind,
  std::string &unsupported_src)
{
  if (!node.is_object())
    return true;
  const std::string nt = node.value("nodeType", "");

  if (nt == "YulBlock")
  {
    if (!node.contains("statements") || !node["statements"].is_array())
      return true;
    for (const auto &s : node["statements"])
      if (!yul_node_is_supported(s, unsupported_kind, unsupported_src))
        return false;
    return true;
  }
  if (nt == "YulVariableDeclaration")
  {
    if (
      !node.contains("variables") || !node["variables"].is_array() ||
      node["variables"].size() != 1)
    {
      unsupported_kind = "YulVariableDeclaration:multi-LHS";
      unsupported_src = node.value("src", "");
      return false;
    }
    if (node.contains("value") && node["value"].is_object())
      return yul_node_is_supported(
        node["value"], unsupported_kind, unsupported_src);
    return true;
  }
  if (nt == "YulAssignment")
  {
    if (
      !node.contains("variableNames") || !node["variableNames"].is_array() ||
      node["variableNames"].size() != 1)
    {
      unsupported_kind = "YulAssignment:multi-LHS";
      unsupported_src = node.value("src", "");
      return false;
    }
    if (!node.contains("value") || !node["value"].is_object())
    {
      unsupported_kind = "YulAssignment:no-value";
      unsupported_src = node.value("src", "");
      return false;
    }
    return yul_node_is_supported(
      node["value"], unsupported_kind, unsupported_src);
  }
  if (nt == "YulIf")
  {
    if (
      node.contains("condition") &&
      !yul_node_is_supported(
        node["condition"], unsupported_kind, unsupported_src))
      return false;
    if (
      node.contains("body") &&
      !yul_node_is_supported(node["body"], unsupported_kind, unsupported_src))
      return false;
    return true;
  }
  if (nt == "YulSwitch")
  {
    if (
      node.contains("expression") &&
      !yul_node_is_supported(
        node["expression"], unsupported_kind, unsupported_src))
      return false;
    if (!node.contains("cases") || !node["cases"].is_array())
    {
      unsupported_kind = "YulSwitch:no-cases";
      unsupported_src = node.value("src", "");
      return false;
    }
    for (const auto &c : node["cases"])
    {
      // Default-case selector is either null (older solc) or the string
      // "default" (newer solc); non-default must be a YulLiteral.
      bool is_default = false;
      if (c.contains("value"))
      {
        const auto &v = c["value"];
        if (v.is_null())
          is_default = true;
        else if (v.is_string() && v.get<std::string>() == "default")
          is_default = true;
        else if (v.is_object())
        {
          if (v.value("nodeType", "") != "YulLiteral")
          {
            unsupported_kind = "YulCase:non-literal-selector";
            unsupported_src = c.value("src", "");
            return false;
          }
          if (!yul_node_is_supported(v, unsupported_kind, unsupported_src))
            return false;
        }
        else
        {
          unsupported_kind = "YulCase:unknown-selector";
          unsupported_src = c.value("src", "");
          return false;
        }
      }
      (void)is_default;
      if (
        c.contains("body") &&
        !yul_node_is_supported(c["body"], unsupported_kind, unsupported_src))
        return false;
    }
    return true;
  }
  if (nt == "YulForLoop" || nt == "YulFor")
  {
    for (const char *k : {"pre", "condition", "post", "body"})
    {
      if (
        node.contains(k) &&
        !yul_node_is_supported(node[k], unsupported_kind, unsupported_src))
        return false;
    }
    return true;
  }
  if (nt == "YulFunctionCall")
  {
    const std::string fname =
      node.value("functionName", nlohmann::json::object()).value("name", "");
    if (!is_supported_yul_builtin(fname))
    {
      unsupported_kind = "YulFunctionCall:" + fname;
      unsupported_src = node.value("src", "");
      return false;
    }
    if (node.contains("arguments") && node["arguments"].is_array())
      for (const auto &a : node["arguments"])
        if (!yul_node_is_supported(a, unsupported_kind, unsupported_src))
          return false;
    return true;
  }
  if (nt == "YulIdentifier")
    return true;
  if (nt == "YulLiteral")
  {
    const std::string kind = node.value("kind", "");
    if (kind == "number" || kind == "bool")
      return true;
    unsupported_kind = "YulLiteral:" + kind;
    unsupported_src = node.value("src", "");
    return false;
  }
  if (nt == "YulBreak" || nt == "YulContinue")
    return true;

  if (nt == "YulExpressionStatement")
  {
    // Wraps a side-effecting Yul expression at statement position.  In the
    // supported subset this is exclusively `sstore(X.slot, v)`; the
    // statement-level lowering path enforces the sstore-only restriction
    // and slot-arg validation.  Recurse into the expression so the scan
    // catches an unsupported builtin (e.g. `mstore`) up-front.
    if (node.contains("expression") && node["expression"].is_object())
      return yul_node_is_supported(
        node["expression"], unsupported_kind, unsupported_src);
    unsupported_kind = "YulExpressionStatement:no-expression";
    unsupported_src = node.value("src", "");
    return false;
  }

  // Unknown / unsupported nodeType — typically YulFunctionDefinition,
  // YulLeave, YulTypedName appearing where a statement is expected, etc.
  unsupported_kind = nt.empty() ? "YulNode:unknown" : nt;
  unsupported_src = node.value("src", "");
  return false;
}

bool solidity_convertert::make_yul_local(
  const std::string &asm_id,
  int seq,
  const std::string &yul_name,
  const locationt &loc,
  exprt &out_sym)
{
  std::string name =
    "_yul_" + asm_id + "_" + std::to_string(seq) + "_" + yul_name;
  std::string id;
  if (current_baseContractName.empty() || current_functionName.empty())
  {
    // Free-function context: scope to the function only.
    id = "sol:@F@" +
         (current_functionName.empty() ? std::string("_anon_")
                                       : current_functionName) +
         "@" + name;
  }
  else
  {
    id = "sol:@C@" + current_baseContractName + "@F@" + current_functionName +
         "@" + name;
  }
  symbolt s;
  std::string mod = get_modulename_from_path(absolute_path);
  unsignedbv_typet u256(256);
  get_default_symbol(s, mod, u256, name, id, loc);
  s.lvalue = true;
  s.file_local = true;
  s.static_lifetime = false;
  if (move_symbol_to_context(s) == nullptr)
    return true;
  out_sym = symbol_exprt(id, u256);
  out_sym.location() = loc;
  return false;
}

bool solidity_convertert::convert_yul_expression(
  const nlohmann::json &yul_expr,
  const std::map<std::string, int> &src_to_decl,
  const std::map<std::string, int> &slot_refs,
  const std::map<std::string, exprt> &locals,
  const locationt &loc,
  exprt &out)
{
  unsignedbv_typet u256(256);
  const std::string nt = yul_expr.value("nodeType", "");

  if (nt == "YulLiteral")
  {
    const std::string kind = yul_expr.value("kind", "");
    const std::string val = yul_expr.value("value", "0");
    if (kind == "bool")
      out = from_integer(BigInt(val == "true" ? 1 : 0), u256);
    else if (kind == "number")
    {
      BigInt v;
      if (val.size() > 2 && val[0] == '0' && (val[1] == 'x' || val[1] == 'X'))
        v = string2integer(val.substr(2), 16);
      else
        v = string2integer(val, 10);
      out = from_integer(v, u256);
    }
    else
      return true;
    out.location() = loc;
    return false;
  }

  if (nt == "YulIdentifier")
  {
    const std::string name = yul_expr.value("name", "");
    auto lit = locals.find(name);
    if (lit != locals.end())
    {
      out = lit->second;
      return false;
    }
    auto sit = src_to_decl.find(yul_expr.value("src", ""));
    if (sit == src_to_decl.end())
      return true;
    const nlohmann::json &decl = find_decl_ref(sit->second);
    if (decl.empty() || decl.value("nodeType", "") != "VariableDeclaration")
      return true;
    bool is_state =
      decl.contains("stateVariable") && decl["stateVariable"].get<bool>();
    if (get_var_decl_ref(decl, is_state, out))
      return true;
    // Yul reads of internal-fn-ptr-typed locals see 0 (uninit fn-ptr semantics).
    if (out.type().get_bool("#sol_func_ptr"))
      out = from_integer(BigInt(0), u256);
    else
      solidity_gen_typecast(ns, out, u256);
    return false;
  }

  if (nt == "YulFunctionCall")
  {
    const std::string fname =
      yul_expr.value("functionName", nlohmann::json::object())
        .value("name", "");
    const auto &args = yul_expr["arguments"];

    auto eval_arg = [&](size_t i, exprt &dst) -> bool {
      return convert_yul_expression(
        args[i], src_to_decl, slot_refs, locals, loc, dst);
    };
    auto u256_const = [&](const BigInt &v) { return from_integer(v, u256); };
    auto bool_to_u256 = [&](const exprt &cond) {
      return if_exprt(cond, u256_const(1), u256_const(0));
    };

    // Pure 2-operand uint256 ops: + - *
    if (fname == "add" || fname == "sub" || fname == "mul")
    {
      if (args.size() != 2)
        return true;
      exprt a, b;
      if (eval_arg(0, a) || eval_arg(1, b))
        return true;
      if (fname == "add")
        out = plus_exprt(a, b);
      else if (fname == "sub")
        out = minus_exprt(a, b);
      else
        out = mult_exprt(a, b);
      out.type() = u256;
      out.location() = loc;
      return false;
    }

    // div / mod: lower to plain div_exprt / mod_exprt so goto_check's
    // div-by-zero check fires when the divisor is symbolically reachable
    // as 0. EVM/Yul defines `div(_, 0) == 0` (no panic, no revert), but
    // this is almost always a bug pattern in user code (programmer
    // expected Solidity 0.8+ panic semantics). With default checks
    // enabled, ESBMC reports "division by zero" violations on reachable
    // zero divisors — which is the verification-useful behavior. With
    // `--no-div-by-zero-check`, the check is suppressed and the model
    // returns SMT-LIB's bvudiv-zero result (all-1s), NOT Yul's 0 — see
    // approximation-ledger.md row #1 for this soundness gap.
    if (fname == "div" || fname == "mod")
    {
      if (args.size() != 2)
        return true;
      exprt a, b;
      if (eval_arg(0, a) || eval_arg(1, b))
        return true;
      if (fname == "div")
        out = div_exprt(a, b);
      else
        out = mod_exprt(a, b);
      out.type() = u256;
      out.location() = loc;
      return false;
    }

    // addmod / mulmod: same div-by-zero exposure for the modulus arg.
    // (Note: the inner add/mul still uses 2^256-wrap arithmetic, not the
    // EVM-spec arbitrary-precision intermediate — that's a separate T2.4
    // gap, not addressed here.)
    if (fname == "addmod" || fname == "mulmod")
    {
      if (args.size() != 3)
        return true;
      exprt a, b, m;
      if (eval_arg(0, a) || eval_arg(1, b) || eval_arg(2, m))
        return true;
      exprt inner;
      if (fname == "addmod")
        inner = plus_exprt(a, b);
      else
        inner = mult_exprt(a, b);
      inner.type() = u256;
      out = mod_exprt(inner, m);
      out.type() = u256;
      out.location() = loc;
      return false;
    }

    // Unsigned comparisons → uint256 (1 / 0)
    if (fname == "lt" || fname == "gt" || fname == "eq")
    {
      if (args.size() != 2)
        return true;
      exprt a, b;
      if (eval_arg(0, a) || eval_arg(1, b))
        return true;
      exprt cond;
      if (fname == "lt")
        cond = binary_relation_exprt(a, "<", b);
      else if (fname == "gt")
        cond = binary_relation_exprt(a, ">", b);
      else
        cond = equality_exprt(a, b);
      out = bool_to_u256(cond);
      out.type() = u256;
      out.location() = loc;
      return false;
    }

    // Signed comparisons: cast operands to int256 first.
    if (fname == "slt" || fname == "sgt")
    {
      if (args.size() != 2)
        return true;
      exprt a, b;
      if (eval_arg(0, a) || eval_arg(1, b))
        return true;
      signedbv_typet s256(256);
      solidity_gen_typecast(ns, a, s256);
      solidity_gen_typecast(ns, b, s256);
      exprt cond;
      if (fname == "slt")
        cond = binary_relation_exprt(a, "<", b);
      else
        cond = binary_relation_exprt(a, ">", b);
      out = bool_to_u256(cond);
      out.type() = u256;
      out.location() = loc;
      return false;
    }

    if (fname == "iszero")
    {
      if (args.size() != 1)
        return true;
      exprt a;
      if (eval_arg(0, a))
        return true;
      equality_exprt cond(a, u256_const(0));
      out = bool_to_u256(cond);
      out.type() = u256;
      out.location() = loc;
      return false;
    }

    // Bitwise binary
    if (fname == "and" || fname == "or" || fname == "xor")
    {
      if (args.size() != 2)
        return true;
      exprt a, b;
      if (eval_arg(0, a) || eval_arg(1, b))
        return true;
      const char *id = (fname == "and")  ? "bitand"
                       : (fname == "or") ? "bitor"
                                         : "bitxor";
      out = exprt(id, u256);
      out.copy_to_operands(a, b);
      out.location() = loc;
      return false;
    }

    if (fname == "not")
    {
      if (args.size() != 1)
        return true;
      exprt a;
      if (eval_arg(0, a))
        return true;
      out = exprt("bitnot", u256);
      out.copy_to_operands(a);
      out.location() = loc;
      return false;
    }

    // Shifts. Yul argument order: shift amount FIRST, value SECOND.
    // EVM clamps `shift >= 256` to 0 for shl/shr.
    if (fname == "shl" || fname == "shr")
    {
      if (args.size() != 2)
        return true;
      exprt s, v;
      if (eval_arg(0, s) || eval_arg(1, v))
        return true;
      const char *id = (fname == "shl") ? "shl" : "lshr";
      exprt shifted(id, u256);
      shifted.copy_to_operands(v, s);
      binary_relation_exprt in_range(s, "<", u256_const(256));
      out = if_exprt(in_range, shifted, u256_const(0));
      out.type() = u256;
      out.location() = loc;
      return false;
    }

    // sload(X.slot) — read scalar state variable X (widened to uint256).
    // The argument must be a YulIdentifier whose src matches a `.slot`
    // external reference resolving to a scalar state variable.  Any
    // failed gate (computed slot, non-state-var, non-scalar type) returns
    // true to abort precise lowering, so the caller falls through to
    // havoc.  The havoc fallback's existing `.slot`-ref handling
    // (lines 1162-1191 of this file) re-nondets every state var hit via
    // `.slot`, preserving soundness for the rejected block.
    if (fname == "sload")
    {
      if (args.size() != 1)
        return true;
      const nlohmann::json &a0 = args[0];
      if (a0.value("nodeType", "") != "YulIdentifier")
        return true;
      auto sit = slot_refs.find(a0.value("src", ""));
      if (sit == slot_refs.end())
        return true;
      const nlohmann::json &decl = find_decl_ref(sit->second);
      if (decl.empty() || decl.value("nodeType", "") != "VariableDeclaration")
        return true;
      const bool is_state = decl.value("stateVariable", false);
      // A scalar state var OR a `storage` reference (state-var struct /
      // library `T storage` param) is eligible; anything else aborts.
      const bool is_storage =
        is_state || decl.value("storageLocation", std::string()) == "storage";
      if (!is_storage)
        return true;
      exprt base;
      if (get_var_decl_ref(decl, is_state, base))
        return true;
      exprt lval = base;
      if (lval.type().id() == "pointer")
        lval = dereference_exprt(base, base.type().subtype());
      typet rt = lval.type();
      while (rt.id() == "symbol")
        rt = ns.follow(rt);

      if (rt.id() == "struct")
      {
        // Single-slot struct: sload(X.slot) reads slot 0 as a 256-bit word.
        // Reconstruct it by packing the slot-0 fields:
        //   word = OR_i ( zext256(field_i) << bitoffset_i ).
        std::vector<yul_slot_field> fields;
        if (!yul_pack_slot0_fields(ns, lval, to_struct_type(rt), fields))
          return true;
        exprt word = u256_const(BigInt(0));
        for (const auto &f : fields)
        {
          exprt v = f.member;
          // bool -> 0/1, unsigned -> zero-extend, to the full slot width.
          solidity_gen_typecast(ns, v, u256);
          if (f.bitoffset != 0)
          {
            exprt sh("shl", u256);
            sh.copy_to_operands(v, u256_const(BigInt(f.bitoffset)));
            v = sh;
          }
          exprt orr("bitor", u256);
          orr.copy_to_operands(word, v);
          orr.location() = loc;
          word = orr;
        }
        out = word;
        out.location() = loc;
        return false;
      }

      // scalar state variable
      const irep_idt tid = lval.type().id();
      if (tid != "unsignedbv" && tid != "signedbv" && tid != "bool")
        return true;
      out = lval;
      solidity_gen_typecast(ns, out, u256);
      out.location() = loc;
      return false;
    }

    return true; // unknown builtin (pre-flight should have rejected)
  }

  return true; // unknown nodeType
}

bool solidity_convertert::convert_yul_statement(
  const nlohmann::json &yul_stmt,
  const std::string &asm_id,
  const std::map<std::string, int> &src_to_decl,
  const std::map<std::string, int> &slot_refs,
  std::map<std::string, exprt> &locals,
  int &local_seq,
  const locationt &loc,
  exprt &out)
{
  unsignedbv_typet u256(256);
  const std::string nt = yul_stmt.value("nodeType", "");

  if (nt == "YulBlock")
    return convert_yul_block(
      yul_stmt, asm_id, src_to_decl, slot_refs, locals, local_seq, loc, out);

  if (nt == "YulVariableDeclaration")
  {
    const std::string yname = yul_stmt["variables"][0].value("name", "");
    if (yname.empty())
      return true;

    exprt sym;
    if (make_yul_local(asm_id, local_seq++, yname, loc, sym))
      return true;
    locals[yname] = sym;

    code_blockt blk;
    code_declt decl(sym);
    decl.location() = loc;
    blk.copy_to_operands(decl);

    exprt rhs;
    if (yul_stmt.contains("value") && yul_stmt["value"].is_object())
    {
      if (convert_yul_expression(
            yul_stmt["value"], src_to_decl, slot_refs, locals, loc, rhs))
        return true;
    }
    else
      rhs = from_integer(BigInt(0), u256);
    solidity_gen_typecast(ns, rhs, u256);
    code_assignt assign(sym, rhs);
    assign.location() = loc;
    blk.copy_to_operands(assign);
    out = blk;
    return false;
  }

  if (nt == "YulAssignment")
  {
    const nlohmann::json &lhs_id = yul_stmt["variableNames"][0];
    const nlohmann::json &rhs_node = yul_stmt["value"];
    const std::string lname = lhs_id.value("name", "");

    // Resolve LHS
    exprt lhs;
    auto lit = locals.find(lname);
    if (lit != locals.end())
      lhs = lit->second;
    else
    {
      auto sit = src_to_decl.find(lhs_id.value("src", ""));
      if (sit == src_to_decl.end())
        return true;
      const nlohmann::json &decl = find_decl_ref(sit->second);
      if (decl.empty() || decl.value("nodeType", "") != "VariableDeclaration")
        return true;
      bool is_state =
        decl.contains("stateVariable") && decl["stateVariable"].get<bool>();
      if (get_var_decl_ref(decl, is_state, lhs))
        return true;
    }

    // Special case: `dst := src` where rhs is a YulIdentifier — preserve the
    // legacy fast-path semantics so bytes-struct / fn-ptr destinations work
    // (they don't roundtrip through uint256).
    if (rhs_node.value("nodeType", "") == "YulIdentifier")
    {
      const std::string rname = rhs_node.value("name", "");
      exprt src_expr;
      auto rit = locals.find(rname);
      if (rit != locals.end())
        src_expr = rit->second;
      else
      {
        auto sit = src_to_decl.find(rhs_node.value("src", ""));
        if (sit == src_to_decl.end())
          return true;
        const nlohmann::json &decl = find_decl_ref(sit->second);
        if (decl.empty() || decl.value("nodeType", "") != "VariableDeclaration")
          return true;
        bool is_state =
          decl.contains("stateVariable") && decl["stateVariable"].get<bool>();
        if (get_var_decl_ref(decl, is_state, src_expr))
          return true;
      }
      // Resolve a type through pointer/symbol indirection down to the
      // underlying struct; returns the resolved symbol tag iff it is a
      // struct, else empty (mirrors struct_type_has_component's idiom;
      // struct_union_typet exposes no plain tag accessor).
      auto struct_tag = [this](const typet &t) -> irep_idt {
        typet rt = t;
        if (rt.id() == "pointer")
          rt = rt.subtype();
        irep_idt tag;
        while (rt.id() == "symbol")
        {
          tag = to_symbol_type(rt).get_identifier();
          const symbolt *s = context.find_symbol(tag);
          if (s == nullptr)
            return irep_idt();
          rt = s->type;
        }
        return rt.id() == "struct" ? tag : irep_idt();
      };

      exprt rhs;
      if (src_expr.type().get_bool("#sol_func_ptr"))
        rhs = from_integer(BigInt(0), lhs.type());
      else
      {
        const irep_idt lt = struct_tag(lhs.type());
        const irep_idt rt = struct_tag(src_expr.type());
        if (!lt.empty() && !rt.empty() && lt != rt)
        {
          // [APPROX: OVER] EVM inline-assembly pointer/value reinterpret
          // between two distinct struct layouts (e.g. bytes32 <-> a memory
          // struct) that c_typecastt cannot bridge. A faithful memory-
          // pointer-aliasing model is out of scope; over-approximate by
          // havoc'ing the destination to a nondet of its own declared
          // type so the dependent member access stays well-typed.
          log_warning(
            "[approx] inline assembly at {}:{}: over-approximating "
            "struct/value reinterpret '{} := {}' (struct '{}' := struct "
            "'{}') - '{}' havoc'd to nondet of its declared type; faithful "
            "EVM memory-pointer aliasing is out of scope",
            loc.get_file().c_str(),
            loc.get_line().c_str(),
            lname,
            rname,
            lt,
            rt,
            lname);
          get_nondet_expr(lhs.type(), rhs);
        }
        else
        {
          rhs = src_expr;
          solidity_gen_typecast(ns, rhs, lhs.type());
        }
      }
      code_assignt assign(lhs, rhs);
      assign.location() = loc;
      out = assign;
      return false;
    }

    exprt rhs;
    if (convert_yul_expression(
          rhs_node, src_to_decl, slot_refs, locals, loc, rhs))
      return true;
    solidity_gen_typecast(ns, rhs, lhs.type());
    code_assignt assign(lhs, rhs);
    assign.location() = loc;
    out = assign;
    return false;
  }

  if (nt == "YulIf")
  {
    exprt cond_val;
    if (convert_yul_expression(
          yul_stmt["condition"], src_to_decl, slot_refs, locals, loc, cond_val))
      return true;
    binary_relation_exprt cond_ne(
      cond_val, "notequal", from_integer(BigInt(0), u256));

    exprt body;
    if (convert_yul_block(
          yul_stmt["body"],
          asm_id,
          src_to_decl,
          slot_refs,
          locals,
          local_seq,
          loc,
          body))
      return true;

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(cond_ne, body);
    if_expr.location() = loc;
    out = if_expr;
    return false;
  }

  if (nt == "YulSwitch")
  {
    exprt e;
    if (convert_yul_expression(
          yul_stmt["expression"], src_to_decl, slot_refs, locals, loc, e))
      return true;

    const auto &cases = yul_stmt["cases"];

    // Build the default branch first; non-default cases chain right-to-left
    // around it.  solc represents default-case selector as either JSON null
    // (older) or the literal string "default" (newer).
    auto is_default = [](const nlohmann::json &c) -> bool {
      if (!c.contains("value"))
        return true;
      const auto &v = c["value"];
      return v.is_null() ||
             (v.is_string() && v.get<std::string>() == "default");
    };

    exprt tail = code_skipt();
    for (const auto &c : cases)
    {
      if (is_default(c))
      {
        if (convert_yul_block(
              c["body"],
              asm_id,
              src_to_decl,
              slot_refs,
              locals,
              local_seq,
              loc,
              tail))
          return true;
        break;
      }
    }

    std::vector<nlohmann::json> non_default;
    for (const auto &c : cases)
      if (!is_default(c))
        non_default.push_back(c);

    for (auto it = non_default.rbegin(); it != non_default.rend(); ++it)
    {
      exprt key;
      if (convert_yul_expression(
            (*it)["value"], src_to_decl, slot_refs, locals, loc, key))
        return true;
      equality_exprt cond(e, key);
      exprt body;
      if (convert_yul_block(
            (*it)["body"],
            asm_id,
            src_to_decl,
            slot_refs,
            locals,
            local_seq,
            loc,
            body))
        return true;
      codet if_expr("ifthenelse");
      if_expr.copy_to_operands(cond, body, tail);
      if_expr.location() = loc;
      tail = if_expr;
    }

    out = tail;
    return false;
  }

  if (nt == "YulForLoop" || nt == "YulFor")
  {
    // Snapshot locals so `let` declarations in `pre` fall out of scope after
    // the for-loop completes.
    auto snapshot = locals;

    code_blockt outer;

    // Walk pre's statements directly (NOT via convert_yul_block) — Yul's `pre`
    // is not a scoped block; its `let`-bindings must remain in scope for
    // cond/post/body.  The outer for-loop snapshot drops them after the loop.
    if (
      yul_stmt.contains("pre") && yul_stmt["pre"].is_object() &&
      yul_stmt["pre"].value("nodeType", "") == "YulBlock" &&
      yul_stmt["pre"].contains("statements") &&
      yul_stmt["pre"]["statements"].is_array())
    {
      for (const auto &s : yul_stmt["pre"]["statements"])
      {
        exprt s_expr;
        if (convert_yul_statement(
              s,
              asm_id,
              src_to_decl,
              slot_refs,
              locals,
              local_seq,
              loc,
              s_expr))
          return true;
        outer.copy_to_operands(s_expr);
      }
    }

    exprt cond_val;
    if (convert_yul_expression(
          yul_stmt["condition"], src_to_decl, slot_refs, locals, loc, cond_val))
      return true;
    binary_relation_exprt cond_ne(
      cond_val, "notequal", from_integer(BigInt(0), u256));

    exprt post_expr;
    if (convert_yul_block(
          yul_stmt["post"],
          asm_id,
          src_to_decl,
          slot_refs,
          locals,
          local_seq,
          loc,
          post_expr))
      return true;

    exprt body_expr;
    if (convert_yul_block(
          yul_stmt["body"],
          asm_id,
          src_to_decl,
          slot_refs,
          locals,
          local_seq,
          loc,
          body_expr))
      return true;

    code_fort code_for;
    code_for.init() = code_skipt();
    code_for.cond() = cond_ne;
    code_for.iter() = static_cast<const codet &>(post_expr);
    code_for.body() = static_cast<const codet &>(body_expr);
    code_for.location() = loc;
    outer.copy_to_operands(code_for);
    out = outer;

    locals = snapshot;
    return false;
  }

  if (nt == "YulBreak")
  {
    code_breakt brk;
    brk.location() = loc;
    out = brk;
    return false;
  }

  if (nt == "YulContinue")
  {
    code_continuet cont;
    cont.location() = loc;
    out = cont;
    return false;
  }

  if (nt == "YulExpressionStatement")
  {
    // sstore(X.slot, v) is the only side-effecting expression statement
    // in the supported subset.  Everything else (a bare sload at
    // statement position, an unknown builtin call) returns true to
    // abort precise lowering — caller falls through to havoc.
    if (!yul_stmt.contains("expression") || !yul_stmt["expression"].is_object())
      return true;
    const nlohmann::json &expr = yul_stmt["expression"];
    if (expr.value("nodeType", "") != "YulFunctionCall")
      return true;
    const std::string fname =
      expr.value("functionName", nlohmann::json::object()).value("name", "");
    if (fname != "sstore")
      return true;
    const auto &args = expr["arguments"];
    if (!args.is_array() || args.size() != 2)
      return true;

    // Resolve slot arg → state-variable symbol.  Must be a YulIdentifier
    // whose src matches a `.slot` ext-ref pointing at a scalar state var.
    const nlohmann::json &a0 = args[0];
    if (a0.value("nodeType", "") != "YulIdentifier")
      return true;
    auto sit = slot_refs.find(a0.value("src", ""));
    if (sit == slot_refs.end())
      return true;
    const nlohmann::json &decl = find_decl_ref(sit->second);
    if (decl.empty() || decl.value("nodeType", "") != "VariableDeclaration")
      return true;
    const bool is_state = decl.value("stateVariable", false);
    const bool is_storage =
      is_state || decl.value("storageLocation", std::string()) == "storage";
    if (!is_storage)
      return true;
    exprt base;
    if (get_var_decl_ref(decl, is_state, base))
      return true;
    exprt lval = base;
    if (lval.type().id() == "pointer")
      lval = dereference_exprt(base, base.type().subtype());
    typet rt = lval.type();
    while (rt.id() == "symbol")
      rt = ns.follow(rt);

    if (rt.id() == "struct")
    {
      // Single-slot struct: sstore(X.slot, w) distributes the 256-bit word w
      // back into the slot-0 fields:
      //   field_i = truncate_to_type_i( (w >> bitoffset_i) & mask_i ).
      // Fields in slots >=1 do not exist here (single-slot subset), so no
      // write is dropped. Build all field writes in a local block and only
      // publish `out` once every field is validated (all-or-nothing).
      std::vector<yul_slot_field> fields;
      if (!yul_pack_slot0_fields(ns, lval, to_struct_type(rt), fields))
        return true;
      exprt w;
      if (convert_yul_expression(
            args[1], src_to_decl, slot_refs, locals, loc, w))
        return true;
      code_blockt blk;
      for (const auto &f : fields)
      {
        exprt shifted = w;
        if (f.bitoffset != 0)
        {
          exprt sh("lshr", u256);
          sh.copy_to_operands(w, from_integer(BigInt(f.bitoffset), u256));
          shifted = sh;
        }
        // Mask to the field width BEFORE the cast: a bool cast keys on
        // `!= 0`, so higher-field bits must be cleared first; for integer
        // fields the cast truncates anyway, so the mask is belt-and-braces.
        exprt masked = shifted;
        if (f.bitwidth < 256)
        {
          const BigInt m = (BigInt(1) << f.bitwidth) - 1;
          exprt band("bitand", u256);
          band.copy_to_operands(shifted, from_integer(m, u256));
          masked = band;
        }
        exprt val = masked;
        solidity_gen_typecast(ns, val, f.member.type());
        code_assignt a(f.member, val);
        a.location() = loc;
        blk.copy_to_operands(a);
      }
      out = blk;
      return false;
    }

    // scalar state variable
    const irep_idt tid = lval.type().id();
    if (tid != "unsignedbv" && tid != "signedbv" && tid != "bool")
      return true;

    // Lower the value expression and narrow to the state var's native type.
    exprt rhs;
    if (convert_yul_expression(
          args[1], src_to_decl, slot_refs, locals, loc, rhs))
      return true;
    solidity_gen_typecast(ns, rhs, lval.type());

    code_assignt assign(lval, rhs);
    assign.location() = loc;
    out = assign;
    return false;
  }

  return true; // unknown nodeType (pre-flight should have caught it)
}

bool solidity_convertert::convert_yul_block(
  const nlohmann::json &yul_block,
  const std::string &asm_id,
  const std::map<std::string, int> &src_to_decl,
  const std::map<std::string, int> &slot_refs,
  std::map<std::string, exprt> &locals,
  int &local_seq,
  const locationt &loc,
  exprt &out)
{
  if (!yul_block.is_object() || yul_block.value("nodeType", "") != "YulBlock")
    return true;

  // Snapshot for nested-scope shadowing.
  auto snapshot = locals;

  code_blockt blk;
  if (yul_block.contains("statements") && yul_block["statements"].is_array())
  {
    for (const auto &s : yul_block["statements"])
    {
      exprt s_expr;
      if (convert_yul_statement(
            s, asm_id, src_to_decl, slot_refs, locals, local_seq, loc, s_expr))
        return true;
      blk.copy_to_operands(s_expr);
    }
  }

  out = blk;
  out.location() = loc;
  locals = snapshot;
  return false;
}

bool solidity_convertert::try_lower_yul_block_precise(
  const nlohmann::json &asm_stmt,
  const locationt &loc,
  exprt &out,
  std::string &unsupported_kind,
  std::string &unsupported_src)
{
  if (!asm_stmt.contains("AST") || !asm_stmt["AST"].is_object())
  {
    unsupported_kind = "InlineAssembly:no-AST";
    unsupported_src = asm_stmt.value("src", "");
    return false;
  }
  const nlohmann::json &yul_root = asm_stmt["AST"];

  // Pre-flight scan — all-or-nothing rule.
  if (!yul_node_is_supported(yul_root, unsupported_kind, unsupported_src))
    return false;

  // Build two src-range -> declaration-id maps for outer-scope identifier
  // lookup: `src_to_decl` for plain (non-suffixed) external references,
  // `slot_refs` for `.slot` references consumed by sload/sstore.
  // `.offset` references imply intra-slot packing — we have no model for
  // that, so reject the entire block on `.offset`.
  std::map<std::string, int> src_to_decl;
  std::map<std::string, int> slot_refs;
  if (
    asm_stmt.contains("externalReferences") &&
    asm_stmt["externalReferences"].is_array())
  {
    for (const auto &ref : asm_stmt["externalReferences"])
    {
      const bool is_slot = ref.contains("isSlot") && ref["isSlot"].get<bool>();
      const bool is_offset =
        ref.contains("isOffset") && ref["isOffset"].get<bool>();
      if (is_offset)
      {
        unsupported_kind = "ExternalReference:offset";
        unsupported_src = ref.value("src", "");
        return false;
      }
      if (
        !ref.contains("declaration") || !ref["declaration"].is_number() ||
        !ref.contains("src") || !ref["src"].is_string())
        continue;
      const std::string src_key = ref["src"].get<std::string>();
      const int decl_id = ref["declaration"].get<int>();
      if (is_slot)
        slot_refs[src_key] = decl_id;
      else
        src_to_decl[src_key] = decl_id;
    }
  }

  std::string asm_id = "asm" + std::to_string(asm_stmt.value("id", 0));
  std::map<std::string, exprt> locals;
  int local_seq = 0;

  if (convert_yul_block(
        yul_root, asm_id, src_to_decl, slot_refs, locals, local_seq, loc, out))
  {
    unsupported_kind = "convert_failure";
    unsupported_src = asm_stmt.value("src", "");
    return false;
  }

  return true;
}
