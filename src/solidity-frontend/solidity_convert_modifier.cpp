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
#include <functional>

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
  const std::string old_functionId = current_functionId;
  const bool old_function_used_snapshot = current_function_used_snapshot;
  const bool old_function_seen_mutation = current_function_seen_mutation;
  const bool old_function_revert_observable =
    current_function_revert_observable;
  std::vector<std::pair<std::string, std::string>>
    old_function_restored_globals;
  old_function_restored_globals.swap(current_function_restored_globals);
  current_function_used_snapshot = false;
  current_function_seen_mutation = false;
  current_function_revert_observable = false;

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

  // Stamp payability on the function type so the Foundry coverage-test
  // generator can emit `{value: N}` ONLY for payable methods — sending value to
  // a non-payable function reverts in the EVM. view / pure / nonpayable carry
  // no flag.
  if (
    ast_node.contains("stateMutability") &&
    ast_node["stateMutability"] == "payable")
    type.set("#sol_payable", true);

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
  current_functionId = id;
  log_debug(
    "solidity",
    "@@@ Parsing function {} in contract {}",
    id.c_str(),
    current_baseContractName);

  if (context.find_symbol(id) != nullptr)
  {
    current_functionDecl = old_functionDecl;
    current_functionName = old_functionName;
    current_functionId = old_functionId;
    current_function_used_snapshot = old_function_used_snapshot;
    current_function_seen_mutation = old_function_seen_mutation;
    current_function_revert_observable = old_function_revert_observable;
    current_function_restored_globals.swap(old_function_restored_globals);
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
  // Revert observation: an *observable* scope lowers a no-snapshot revert to
  // `mark + return` (a real branch) instead of legacy path-pruning.  See
  // build_revert_rollback_block.
  //
  // Libraries and free functions used to be excluded, and that exclusion was
  // measured to be the worst defect in the path-coverage pipeline. Their
  // `require(c)` was lowered to a bare `assume(c)` with NO control flow, so the
  // `!c` execution did not exist in the model at all while on-chain it reverts.
  // The damage does not stop at an incomplete decision set: the `!c` inputs
  // belong to no enumerated path, so the downstream subtraction can never
  // remove them, the interval bound is a syntactic product that cannot carry
  // `c`, and the certifying query itself runs under the same `assume(c)` — so a
  // candidate gets certified over inputs the verifier never saw, and the
  // emitted test reverts on the UNMODIFIED contract while claiming to be
  // certified. `require` inside a library is the standard shape (OpenZeppelin
  // SafeERC20, every SafeMath variant), so this was not a corner case.
  //
  // Still excluded, for reasons that are about semantics rather than scope:
  //   * EventDefinition / ErrorDefinition — not real functions; their bodies
  //     are synthesised (an error body is exactly `ASSUME(false)`, which IS the
  //     revert, not a guard inside one).
  //   * constructors — a revert anywhere in `constructor -> _mint -> require`
  //     must prune, because the EVM aborts contract creation. Making it
  //     observable would let construction continue with a half-initialised
  //     contract.
  const bool is_event_or_err = ast_node.contains("nodeType") &&
                               (ast_node["nodeType"] == "EventDefinition" ||
                                ast_node["nodeType"] == "ErrorDefinition");
  current_function_revert_observable = !is_event_or_err && !is_ctor;
  if (!is_event_err_lib && !is_free_function)
    get_function_this_pointer_param(
      c_name, id, debug_modulename, location_begin, type);

  // 11.1b Pre-create the per-function `_sol_save_this` snapshot symbol so
  // that revert/require lowering during body conversion can reference it
  // (see build_revert_rollback_block).  At function entry we copy `*this`
  // into this local; on revert we restore from it.  Without this rollback,
  // a `revert`/`require(false)` would simply prune the path and rule out
  // the real-EVM execution where the call returns to its caller with
  // state unchanged.
  //
  // Scope (B1 v1): only public/external entry points (those reachable
  // from the dispatcher loop or from another contract) get a snapshot.
  // Internal/private helpers keep the legacy `__ESBMC_assume(...)`
  // lowering — they correspond to subroutines that on the EVM revert
  // by *propagating* up to their caller's frame, which our v1 snapshot
  // (per-function, no propagation) would not faithfully model.  Most
  // importantly this preserves the constructor's call-chain semantics:
  // a revert anywhere inside `constructor → _mint → require(...)` still
  // prunes the construction path (matches real EVM, which would abort
  // contract creation), instead of restoring `_mint`'s snapshot and
  // letting construction continue with a half-initialised contract.
  // Skipped also for ctor/event/error/library/free-function (no
  // dispatcher-reachable `*this` to snapshot).
  bool is_external_entry = false;
  if (!is_event_err_lib && !is_free_function && !is_ctor)
  {
    if (is_receive_fallback)
    {
      is_external_entry = true;
    }
    else if (ast_node.contains("visibility"))
    {
      const std::string vis = ast_node["visibility"].get<std::string>();
      is_external_entry = (vis == "public" || vis == "external");
    }
  }
  if (is_external_entry)
  {
    std::string save_id = id + "#_sol_save_this";
    if (context.find_symbol(save_id) == nullptr)
    {
      typet save_type = symbol_typet(prefix + c_name);
      symbolt save_sym;
      get_default_symbol(
        save_sym,
        debug_modulename,
        save_type,
        "_sol_save_this",
        save_id,
        location_begin);
      save_sym.lvalue = true;
      save_sym.file_local = true;
      move_symbol_to_context(save_sym);
    }
    // No function-level pre-pass needed: the per-rollback flag
    // `current_function_seen_mutation` is updated forward-incrementally
    // by get_block as each top-level statement is converted, and read
    // by build_revert_rollback_block at every require/revert site.
  }

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
  std::vector<size_t> named_ret_positions;
  bool has_named_returns = false;
  const bool is_tuple_return =
    get_sol_type(type.return_type()) == SolidityGrammar::SolType::TUPLE_RETURNS;
  // Emit DECL + zero-init for named return parameters in both the
  // single-return and tuple-return cases. For tuples, this ensures that
  // body code (including inline assembly, which havocs via symbol
  // lookup) can refer to the named outputs as regular local variables.
  if (
    !is_ctor && ast_node.contains("returnParameters") &&
    ast_node["returnParameters"].contains("parameters"))
  {
    size_t pos = 0;
    for (const auto &rparam : ast_node["returnParameters"]["parameters"])
    {
      std::string rname = rparam["name"].get<std::string>();
      if (rname.empty())
      {
        ++pos;    // keep the tuple-member index aligned with declaration order
        continue; // unnamed return parameter — skip
      }

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
      named_ret_positions.push_back(pos);
      ++pos;
    }
  }

  // 12. Convert body and embed the body into the same symbol
  // skip for 'unimplemented' functions which has no body,
  // e.g. asbstract/interface, the symbol value would be left as unset

  // solc 0.6.x always emits the `body` field with JSON null for unimplemented
  // (interface / abstract) functions; 0.8.x omits it. Treat both as "no body".
  bool has_body = ast_node.contains("body") && !ast_node["body"].is_null();
  exprt body_exprt = code_blockt();
  if (has_body)
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
      if (
        !has_explicit_return && !is_tuple_return && named_ret_syms.size() == 1)
      {
        code_returnt implicit_ret;
        implicit_ret.return_value() = named_ret_syms[0];
        implicit_ret.location() = location_begin;
        new_body.copy_to_operands(implicit_ret);
      }
      else if (
        !has_explicit_return && is_tuple_return &&
        !has_modifier_invocation(ast_node))
      {
        // Multi-return function that assigns its NAMED return variables
        // (e.g. via inline assembly `x := ...`, or plain `x = ...`) without
        // an explicit `return (x, y)`: bind each named return into the
        // callee's tuple_instance (mem<pos>) at the fall-through exit, then a
        // bare `return;`. Without this the named returns are computed and
        // discarded and the caller reads an uninitialised tuple_instance
        // (this is the 1inch/aqua BalanceLib `(a, b) = balance.load()` bug).
        // Reached only on the fall-through path, so an early explicit return
        // still wins on its own path.
        //
        // Restricted to the non-modifier path: under a modifier the body is
        // re-scoped into an aux wrapper (get_func_modifier) so the named
        // returns the body actually writes are DIFFERENT symbols from the
        // outer-scope `named_ret_syms` collected here; binding those would
        // silently propagate zero/stale outer locals. Modifier + tuple +
        // named-return-without-explicit-return is a separate pre-existing
        // gap (pinned KNOWNBUG); the aqua/BalanceLib targets are modifier-free.
        std::string tname, tid;
        if (
          !get_tuple_instance_name(*current_functionDecl, tname, tid) &&
          context.find_symbol(tid) != nullptr)
        {
          const symbolt &tsym = *context.find_symbol(tid);
          const struct_typet &tst = to_struct_type(tsym.type);
          for (size_t k = 0; k < named_ret_syms.size(); ++k)
          {
            const size_t pos = named_ret_positions[k];
            if (pos >= tst.components().size())
              continue;
            exprt lop;
            if (get_tuple_member_call(tid, tst.components().at(pos), lop))
              continue;
            code_assignt bind(lop, named_ret_syms[k]);
            bind.location() = location_begin;
            new_body.copy_to_operands(bind);
          }
          code_returnt bare_ret;
          bare_ret.location() = location_begin;
          new_body.copy_to_operands(bare_ret);
        }
      }

      body_exprt = new_body;
    }

    // Wrap contract method bodies with an enclosing-contract save /
    // set / restore so library bodies called from within can recover
    // the currently-executing contract's identity (msg.sender for
    // library external calls, caller-side $balance debit for library
    // transfer/send/call{value:v}).  Applied to every method with a
    // real `this`: contracts, ctors, receive/fallback.  Skipped for
    // library internal functions, free functions, events, errors —
    // none of which have a meaningful `this` to record.
    //
    // Early returns inside `body_exprt` skip the trailing restore,
    // leaving the ambient value at this method's this after exit.
    // That stale value is harmless:
    //   - If the caller is another contract method, its own call-site
    //     `get_high_level_call_wrapper` saved the pre-call ambient and
    //     restores on return, overwriting the stale value.
    //   - If the caller is the auto-dispatch loop (`_ESBMC_Main_X`),
    //     no code runs after the callee returns, so the stale value
    //     is never observed.
    if (
      !is_event_err_lib && !is_free_function && has_body &&
      body_exprt.is_code())
    {
      std::string this_id = id + "#this";
      const symbolt *this_sym = context.find_symbol(this_id);
      if (this_sym != nullptr)
      {
        exprt this_expr = symbol_expr(*this_sym);
        typet addr_t_sym = unsignedbv_typet(160);
        typet void_ptr_t = pointer_typet(empty_typet());

        exprt encl_addr_g = symbol_expr(
          *context.find_symbol("c:@_ESBMC_enclosing_contract_address"));
        exprt encl_this_g = symbol_expr(
          *context.find_symbol("c:@_ESBMC_enclosing_contract_this"));

        symbolt sa;
        get_default_symbol(
          sa,
          debug_modulename,
          addr_t_sym,
          "_saved_encl_addr",
          "sol:@C@" + c_name + "@F@_saved_encl_addr#" +
            std::to_string(aux_counter++),
          location_begin);
        symbolt &added_sa = *move_symbol_to_context(sa);
        code_declt sa_decl(symbol_expr(added_sa));
        added_sa.value = encl_addr_g;
        sa_decl.operands().push_back(encl_addr_g);

        symbolt st;
        get_default_symbol(
          st,
          debug_modulename,
          void_ptr_t,
          "_saved_encl_this",
          "sol:@C@" + c_name + "@F@_saved_encl_this#" +
            std::to_string(aux_counter++),
          location_begin);
        symbolt &added_st = *move_symbol_to_context(st);
        code_declt st_decl(symbol_expr(added_st));
        added_st.value = encl_this_g;
        st_decl.operands().push_back(encl_this_g);

        exprt this_addr_mem = member_exprt(this_expr, "$address", addr_t_sym);
        exprt assign_set_addr = side_effect_exprt("assign", addr_t_sym);
        assign_set_addr.copy_to_operands(encl_addr_g, this_addr_mem);
        convert_expression_to_code(assign_set_addr);

        exprt this_cast = this_expr;
        solidity_gen_typecast(ns, this_cast, void_ptr_t);
        exprt assign_set_this = side_effect_exprt("assign", void_ptr_t);
        assign_set_this.copy_to_operands(encl_this_g, this_cast);
        convert_expression_to_code(assign_set_this);

        exprt assign_rst_addr = side_effect_exprt("assign", addr_t_sym);
        assign_rst_addr.copy_to_operands(encl_addr_g, symbol_expr(added_sa));
        convert_expression_to_code(assign_rst_addr);

        exprt assign_rst_this = side_effect_exprt("assign", void_ptr_t);
        assign_rst_this.copy_to_operands(encl_this_g, symbol_expr(added_st));
        convert_expression_to_code(assign_rst_this);

        code_blockt wrapped;

        // Revert observation: clear the global revert flag at the
        // public/external call boundary, so __ESBMC_reverted() reflects only
        // THIS call's subtree (and no state leaks across --bound dispatcher
        // iterations).  Gated + hidden + coverage-skipped.  Internal helpers
        // and constructors deliberately do NOT clear, so their reverts
        // accumulate and propagate up to the external boundary.
        // See docs/claude/solidity/revert-observation.md.
        // The low-level-call dispatcher clears the flag locally right before
        // each callee call (emit_call_revert_clear), so a global entry-clear
        // is unnecessary for `--bound` failure modeling — and emitting it at
        // every external entry bloated the SSA (transfer-only paths). Keep the
        // clear scoped to the explicit `__ESBMC_reverted` opt-in.
        if (uses_revert_observation && is_external_entry)
        {
          exprt clear_stmt;
          build_revert_flag_call(
            "_ESBMC_sol_clear_revert",
            "c:@F@_ESBMC_sol_clear_revert",
            location_begin,
            clear_stmt);
          wrapped.copy_to_operands(clear_stmt);
        }

        // EVM revert state-rollback snapshot.  Emits at function entry:
        //   <Contract> _sol_save_this;
        //   _sol_save_this = *this;
        // build_revert_rollback_block() restores `*this` from this local
        // when it lowers a revert/require failure, so state writes that
        // happen before the revert do not leak past the failure point.
        //
        // Lazy emission: only if the body actually referenced the
        // snapshot (i.e. lowered at least one require/revert that hit
        // build_revert_rollback_block).  Functions without
        // require/revert keep their pre-B1 SSA shape — important for
        // k-induction's inductive step, which struggles with the extra
        // whole-struct copy in functions where it carries no semantic
        // value (e.g. pure read-only `assert(...)` invariants like
        // erc20_1's `test_supply`).  Also skipped for any function
        // that didn't receive an `_sol_save_this` symbol in step 11.1b
        // (constructor / event / error / library / free / internal +
        // private).  Decl and init are emitted as two separate
        // statements rather than a decl-with-initialiser because the
        // right-hand side is a whole-struct dereference, and goto-symex's
        // deref+struct simplification handles the assignment-form more
        // reliably than a decl-init for struct types.
        if (current_function_used_snapshot)
        {
          std::string save_id = id + "#_sol_save_this";
          const symbolt *save_sym_ptr = context.find_symbol(save_id);
          if (save_sym_ptr != nullptr)
          {
            exprt this_deref = dereference_exprt(this_expr, save_sym_ptr->type);
            code_declt save_decl(symbol_expr(*save_sym_ptr));
            code_assignt save_init(symbol_expr(*save_sym_ptr), this_deref);
            wrapped.copy_to_operands(save_decl);
            wrapped.copy_to_operands(save_init);
          }

          // Snapshot the out-of-struct global stores (mappings, dynarray
          // data/length) that the body's revert rollback restores.  The save
          // symbols are per-frame locals (like _sol_save_this), so emit a decl
          // + init to capture the entry value before the body runs.
          for (const auto &gpair : current_function_restored_globals)
          {
            const symbolt *store_sym = context.find_symbol(gpair.first);
            const symbolt *g_save_sym = context.find_symbol(gpair.second);
            if (store_sym == nullptr || g_save_sym == nullptr)
              continue;
            code_declt g_decl(symbol_expr(*g_save_sym));
            code_assignt g_init(
              symbol_expr(*g_save_sym), symbol_expr(*store_sym));
            wrapped.copy_to_operands(g_decl);
            wrapped.copy_to_operands(g_init);
          }
        }

        wrapped.copy_to_operands(sa_decl);
        wrapped.copy_to_operands(st_decl);
        wrapped.copy_to_operands(assign_set_addr);
        wrapped.copy_to_operands(assign_set_this);
        for (auto &op : body_exprt.operands())
          wrapped.copy_to_operands(op);
        wrapped.copy_to_operands(assign_rst_addr);
        wrapped.copy_to_operands(assign_rst_this);

        body_exprt = wrapped;
      }
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
  current_functionId = old_functionId;
  current_function_used_snapshot = old_function_used_snapshot;
  current_function_seen_mutation = old_function_seen_mutation;
  current_function_revert_observable = old_function_revert_observable;
  current_function_restored_globals.swap(old_function_restored_globals);
  return false;
}

bool solidity_convertert::statement_is_mutation_top_level(
  const nlohmann::json &stmt)
{
  // Per-statement AST classifier.  Returns true when the statement
  // is conservatively *state-mutating* — the caller (the get_block
  // body walker) ORs the result into `current_function_seen_mutation`
  // after each statement so that the next `build_revert_rollback_block`
  // emission knows whether the rollback at that point needs to
  // restore `*this` (full form) or can early-return (cheap form).
  //
  // Recognised non-mutating statements:
  //   * `VariableDeclarationStatement` with no init or pure init
  //   * `ExpressionStatement` whose inner expression is a call to
  //     `require` / `revert` / `assert`, or a typeConversion
  //   * `RevertStatement` (custom-error revert)
  //   * `Return` / `Break` / `Continue` / `EmitStatement` /
  //     `PlaceholderStatement`
  //   * Bare value expressions (e.g. `x;` for warning suppression)
  //
  // Conservative-mutation fallback for:
  //   * `VariableDeclarationStatement` whose `initialValue` contains
  //     a non-typeConversion `FunctionCall` (e.g. tuple-destructure
  //     of an external call: `(bool s,) = addr.call{value:v}("");`)
  //   * `ExpressionStatement` with `Assignment` / `UnaryOperation` /
  //     non-pure `FunctionCall`
  //   * Any `Block` / `IfStatement` / `ForStatement` / `WhileStatement`
  //     / `DoWhileStatement` / `TryStatement` / `InlineAssembly` /
  //     `UncheckedBlock` — too complex to scan cheaply, assume
  //     mutating
  //   * Any unrecognised shape (safety default)
  //
  // The classifier is intentionally TOP-LEVEL only: it does not
  // recurse into the body of an `IfStatement` / `ForStatement` etc.,
  // because those are already conservatively classified as
  // mutating.  It also does not recurse into `require(impure_expr())`
  // — a require whose argument expression mutates state would set
  // the flag *after* this require is lowered, which would emit the
  // cheap form despite the argument's mutation.  This blind spot
  // is shared with the v2 function-level analysis and accepted as
  // a v3 limitation; revisit if real Solidity code hits it.
  if (!stmt.is_object() || !stmt.contains("nodeType"))
    return true;
  const std::string ntype = stmt["nodeType"].get<std::string>();

  if (ntype == "VariableDeclarationStatement")
  {
    if (
      stmt.contains("initialValue") &&
      initializer_has_side_effect(stmt["initialValue"]))
      return true;
    return false;
  }

  if (ntype == "ExpressionStatement" && stmt.contains("expression"))
  {
    const auto &e = stmt["expression"];
    const std::string ent = e.value("nodeType", "");
    if (ent == "FunctionCall")
    {
      if (e.value("kind", "") == "typeConversion")
        return false;
      std::string callee_name;
      if (e.contains("expression"))
      {
        const auto &callee = e["expression"];
        callee_name = callee.value("name", "");
        if (callee_name.empty() && callee.contains("memberName"))
          callee_name = callee["memberName"].get<std::string>();
      }
      if (
        callee_name == "require" || callee_name == "revert" ||
        callee_name == "assert")
        return false;
      return true;
    }
    if (ent == "Assignment" || ent == "UnaryOperation")
      return true;
    return false;
  }

  if (
    ntype == "Return" || ntype == "Break" || ntype == "Continue" ||
    ntype == "EmitStatement" || ntype == "PlaceholderStatement" ||
    ntype == "RevertStatement")
    return false;

  return true;
}

bool solidity_convertert::initializer_has_side_effect(
  const nlohmann::json &node)
{
  // Recursively scan a JSON subtree for a non-typeConversion
  // FunctionCall.  Used by detect_function_needs_snapshot to flag
  // VariableDeclarationStatements whose RHS performs an external
  // call (e.g. `(bool s,) = addr.call{value:v}("");` — the call
  // mutates global balances even though the LHS is a local).
  if (!node.is_object() && !node.is_array())
    return false;
  if (node.is_object() && node.contains("nodeType"))
  {
    const std::string nt = node["nodeType"].get<std::string>();
    if (nt == "FunctionCall" && node.value("kind", "") != "typeConversion")
      return true;
  }
  if (node.is_object())
  {
    for (auto it = node.begin(); it != node.end(); ++it)
      if (initializer_has_side_effect(it.value()))
        return true;
  }
  else
  {
    for (const auto &child : node)
      if (initializer_has_side_effect(child))
        return true;
  }
  return false;
}

void solidity_convertert::build_revert_flag_call(
  const std::string &name,
  const std::string &id,
  const locationt &loc,
  exprt &out_stmt)
{
  // A no-arg void call to a solidity_misc.c helper, lowered to a statement
  // and tagged `skipped` so condition/branch coverage ignores it (the helper
  // body also lives in the library file, filtered out by location_pool).
  side_effect_expr_function_callt call;
  get_library_function_call_no_args(name, id, empty_typet(), loc, call);
  convert_expression_to_code(call);
  locationt skip_loc = loc;
  skip_loc.property("skipped");
  call.location() = skip_loc;
  out_stmt = call;
}

void solidity_convertert::collect_contract_global_stores(
  const std::string &store_prefix,
  std::vector<std::pair<std::string, typet>> &out)
{
  // No caching: the symbol table keeps growing while functions are converted
  // (e.g. lazily-created mapping globals on first member access), so a cached
  // first scan could miss a store a later function writes and reverts.  The
  // scan is conversion-time and only runs at revert/require sites.
  out.clear();
  context.foreach_operand([&](const symbolt &s) {
    // Mappings and state-variable dynamic-array companions are lowered to
    // file-local static infinite-array globals that live outside *this.
    if (!s.static_lifetime || !s.file_local)
      return;
    if (s.type.id() != "array")
      return;
    const std::string sid = id2string(s.id);
    if (sid.compare(0, store_prefix.size(), store_prefix) != 0)
      return;
    // Exclude nested function symbols / their locals and our own snapshots.
    if (sid.find("@F@") != std::string::npos)
      return;
    if (sid.find("_sol_save_") != std::string::npos)
      return;
    out.emplace_back(sid, s.type);
  });
}

bool solidity_convertert::build_revert_rollback_block(
  const exprt *cond,
  exprt &out)
{
  // Replaces the legacy `__ESBMC_assume(false)` / `__ESBMC_assume(cond)`
  // lowering for `revert` / `require`.  The legacy lowering pruned the
  // path *with* the pre-revert state writes still in the SSA — so a
  // dispatcher iteration that wrote to `*this` then reverted would
  // contribute its writes to the surviving (non-revert) paths' state
  // even though, on the EVM, the revert would have rolled them back.
  //
  // The replacement makes the revert path feasible (no assume(false))
  // and restores `*this` to its function-entry snapshot, then returns
  // a nondet of the function's return type.  The caller (in real EVM
  // semantics) would also revert when their callee reverts without a
  // try/catch wrapper; we do not propagate the revert up the stack
  // here — that is an over-approximation that lets the caller continue
  // with its own pre-call writes still intact.  Sound for safety
  // verification (admits more paths, never rules real EVM paths out).
  //
  // Returns true (does not set `out`) when the rollback shape is not
  // applicable: outside any function context, no `_sol_save_this`
  // symbol (constructors / events / errors / libraries / free
  // functions), or a tuple-returning function (the multi-component
  // return shape needs the existing tuple_instance plumbing — out of
  // scope for the headline rollback fix).  In those cases the caller
  // falls back to the legacy `__ESBMC_assume(...)` lowering.
  if (current_functionId.empty() || current_functionDecl == nullptr)
    return true;

  std::string save_id = current_functionId + "#_sol_save_this";
  const symbolt *save_sym = context.find_symbol(save_id);
  std::string this_id = current_functionId + "#this";
  const symbolt *this_sym = context.find_symbol(this_id);

  // Public/external entry points carry a `_sol_save_this` snapshot (+ `#this`)
  // so the revert can roll `*this` back — emit the full rollback form.
  const bool have_snapshot = (save_sym != nullptr && this_sym != nullptr);

  // Without a snapshot (constructor / library / free / event-error /
  // internal-private helper), the legacy lowering prunes the path via
  // `__ESBMC_assume`.  Under the revert-observation gate, an *observable*
  // scope (contract fn / internal helper — current_function_revert_observable)
  // instead lowers to `mark + return` so the revert is visible to
  // `__ESBMC_reverted()`; non-observable scopes (ctor / library / free /
  // event-error) keep pruning.  See docs/claude/solidity/revert-observation.md.
  if (
    !have_snapshot &&
    !(uses_revert_observation && current_function_revert_observable))
    return true;

  // return [nondet of return type]; — or bare `return;` when void.
  code_returnt return_stmt;
  if (current_functionDecl->contains("returnParameters"))
  {
    typet ret_type;
    if (get_type_description(
          (*current_functionDecl)["returnParameters"], ret_type))
      return true;
    if (ret_type.is_not_nil() && ret_type.id() != "empty")
    {
      // Tuple-returning functions (`returns (T1 a, T2 b)`) used to bail
      // out here, dropping back to legacy __ESBMC_assume(cond) — which
      // loses the branch coverage decision because there is no GOTO IF
      // for the require.  Instead, build a nondet of the (possibly
      // tuple) return type so the if-then-else rollback shape is still
      // emitted.  get_nondet_expr handles both scalar and tuple types
      // (the tuple case yields a struct of nondet components, matching
      // the tuple-return ABI used by the caller).
      exprt nondet_val;
      get_nondet_expr(ret_type, nondet_val);
      return_stmt.return_value() = nondet_val;
    }
  }

  // Choose between the full restore form and the early-return-only
  // form based on `current_function_seen_mutation`, which the body
  // walker (get_block) updates forward-incrementally.  The
  // early-return form avoids the whole-struct `*this = save` copy
  // when no state mutation has been observed earlier in the body
  // (semantic no-op when seen_mutation == false).  When the cheap
  // form is selected we deliberately do NOT set
  // current_function_used_snapshot, so the body wrap can skip the
  // matching snapshot decl + init at function entry whenever every
  // rollback site lowered to the cheap form — keeping the SSA
  // shape identical to the pre-B1 baseline for guard-only patterns
  // (the common Solidity case).
  code_blockt block;
  // Revert observation: set the global flag before restoring/returning so the
  // caller's `__ESBMC_reverted()` sees this revert.  Hidden + coverage-skipped.
  // Also marked under `--bound` (without the explicit opt-in) so the
  // low-level-call dispatcher can model `.call`/`.send` failure as
  // `ok = !reverted`.  Invariant: the only reader of the flag under plain
  // `--bound` is that dispatcher, which clears the flag immediately before
  // each callee call (emit_call_revert_clear) and restores it afterwards, so
  // a stale mark from an earlier revert-and-return is never misread.  Any
  // future `--bound` reader MUST likewise clear before reading.
  if (uses_revert_observation || is_bound)
  {
    locationt mloc;
    get_location_from_node(*current_functionDecl, mloc);
    exprt mark_stmt;
    build_revert_flag_call(
      "_ESBMC_sol_mark_revert", "c:@F@_ESBMC_sol_mark_revert", mloc, mark_stmt);
    block.copy_to_operands(mark_stmt);
  }
  if (have_snapshot && current_function_seen_mutation)
  {
    // *this = _sol_save_this;
    exprt this_expr = symbol_expr(*this_sym);
    exprt this_deref = dereference_exprt(this_expr, save_sym->type);
    exprt save_ref = symbol_expr(*save_sym);
    code_assignt restore(this_deref, save_ref);
    block.copy_to_operands(restore);

    // EVM revert rolls back ALL state, but `*this = save` only restores the
    // contract struct.  Mappings and state-variable dynamic-array data/length
    // companions are file-local static infinite-array globals outside *this,
    // so also restore each from a per-function snapshot.  The matching
    // `_sol_save_g_<base> = <store>` snapshot is emitted at function entry by
    // get_function_definition for every pair recorded here.
    const std::string::size_type fpos = current_functionId.find("@F@");
    if (fpos != std::string::npos)
    {
      const std::string store_prefix = current_functionId.substr(0, fpos) + "@";
      std::vector<std::pair<std::string, typet>> stores;
      collect_contract_global_stores(store_prefix, stores);
      for (const auto &st : stores)
      {
        const symbolt *store_sym = context.find_symbol(st.first);
        if (store_sym == nullptr)
          continue;
        const std::string base = id2string(store_sym->name);
        const std::string g_save_id =
          current_functionId + "#_sol_save_g_" + base;
        if (context.find_symbol(g_save_id) == nullptr)
        {
          symbolt g_save;
          get_default_symbol(
            g_save,
            store_sym->module.as_string(),
            store_sym->type,
            "_sol_save_g_" + base,
            g_save_id,
            store_sym->location);
          // Per-frame local (NOT static), mirroring _sol_save_this: symex
          // renames it per call frame, so a recursive / re-entrant call gets
          // its own snapshot and the outer frame's revert restores to the
          // outer entry state.  A shared static slot would let an inner entry
          // overwrite the outer snapshot and leak the reverted writes.
          g_save.lvalue = true;
          g_save.file_local = true;
          move_symbol_to_context(g_save);
          current_function_restored_globals.emplace_back(st.first, g_save_id);
        }
        const symbolt *g_save_sym = context.find_symbol(g_save_id);
        // <store> = _sol_save_g_<base>;
        code_assignt g_restore(
          symbol_expr(*store_sym), symbol_expr(*g_save_sym));
        block.copy_to_operands(g_restore);
      }
    }
  }
  block.copy_to_operands(return_stmt);

  if (cond == nullptr)
  {
    // Unconditional rollback (revert).
    out = block;
  }
  else
  {
    // Conditional rollback (require): if (!cond) { [restore;] return; }
    not_exprt neg_cond(*cond);
    codet ifstmt("ifthenelse");
    ifstmt.copy_to_operands(neg_cond, block);
    out = ifstmt;
  }
  // Tag the emitted statement so `add_reentry_check` can recognise
  // it as a leading guard and place the reentrancy assertion *after*
  // it, mirroring the legacy `__ESBMC_assume(cond)` skip behaviour.
  // Without this tag the reentry check fires before the require has
  // a chance to bail out, making any `require(!lock)` mutex pattern
  // appear to admit a reentrancy on its first call.
  out.set("#sol_revert_rollback", true);
  // Mark the current function as actually needing the per-frame
  // `_sol_save_this` decl + init only when the full restore form
  // was emitted.  Lowering to early-return-only does not reference
  // the snapshot, so the decl can be skipped — keeping the SSA
  // shape identical to the pre-B1 baseline for guard-only
  // functions (important for k-induction's inductive step).
  if (have_snapshot && current_function_seen_mutation)
    current_function_used_snapshot = true;
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
    // solc 0.6.x ModifierInvocation nodes may omit "kind" for base-constructor
    // calls; the guarded check below already handles the missing field.
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
  // Two leading-guard shapes are recognised:
  //   (1) Legacy `__ESBMC_assume(cond)` — a function-call expression statement
  //       whose op0 is the sideeffect call to __ESBMC_assume.
  //   (2) B1 require/revert rollback — an `ifthenelse` codet that
  //       build_revert_rollback_block tagged with `#sol_revert_rollback`.
  // Skip past either form before placing the reentrancy assertion.
  auto &ops = body_exprt.operands();
  for (auto it = ops.begin(); it != ops.end(); ++it)
  {
    if (
      it->op0().id() == "sideeffect" &&
      it->op0().op0().name() == "__ESBMC_assume")
      continue;
    if (it->get_bool("#sol_revert_rollback"))
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
    nlohmann::json mod_def = find_decl_ref(modifier_id);
    assert(!mod_def.is_null());
    assert(!mod_def.empty());

    // `modifier mod virtual;` declares the modifier without a body — the
    // actual definition lives in a derived contract that overrides it.
    // Walk the linearizedBaseContracts of the current contract and pick
    // the first concrete override with the same name. If none is found
    // (e.g. all overrides are also virtual), skip this modifier entirely
    // — better than dereferencing the missing `body` key downstream and
    // crashing get_block with a json type_error.
    if (!mod_def.contains("body") || mod_def["body"].is_null())
    {
      const std::string mod_name = mod_def["name"].get<std::string>();
      bool resolved = false;
      // Find current contract node and walk linearizedBaseContracts.
      for (const auto &top : src_ast_json["nodes"])
      {
        if (
          top.value("nodeType", "") != "ContractDefinition" ||
          top.value("name", "") != c_name)
          continue;
        if (!top.contains("linearizedBaseContracts"))
          break;
        for (const auto &base_id : top["linearizedBaseContracts"])
        {
          const nlohmann::json base_node = find_decl_ref(base_id.get<int>());
          if (base_node.is_null() || !base_node.contains("nodes"))
            continue;
          for (const auto &member : base_node["nodes"])
          {
            if (
              member.value("nodeType", "") == "ModifierDefinition" &&
              member.value("name", "") == mod_name && member.contains("body") &&
              !member["body"].is_null())
            {
              mod_def = member;
              resolved = true;
              break;
            }
          }
          if (resolved)
            break;
        }
        break;
      }
      if (!resolved)
        continue;
    }

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
      // Multi-return functions lower their return type to empty_typet
      // (TUPLE_RETURNS) — get_parameter_list returns void for tuples.
      // The aux-return-variable rewrite below would create a void-typed
      // symbol and assign to it, which symex resolves through a
      // symbol_typet that nothing fills in (symbolic_type_excp). Treat
      // tuple-returning modifier-wrapped functions as void here: the
      // tuple instance struct already carries the named outputs.
      if (
        aux_type.return_type().is_empty() ||
        get_sol_type(aux_type.return_type()) ==
          SolidityGrammar::SolType::TUPLE_RETURNS)
      {
        has_return = false;
        aux_type.return_type() = empty_typet();
        aux_type.set("cpp_type", "void");
      }
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
    // Pass through the wrapped function's own parameters so the inlined
    // body (which references them by id with the aux function's scope
    // prefix) can resolve to real parameter symbols. Without this, symex
    // hits `value_set: unknown symbol` on e.g.
    // preInteraction_onlyLimitOrderProtocol@extraData#64.
    // current_functionName/Decl have already been switched to the aux
    // above, so get_function_params will register the param symbols
    // under the aux function's scope (sol:@C@<c>@F@<aux>@<name>#id).
    if (
      ast_node.contains("parameters") &&
      ast_node["parameters"].contains("parameters"))
    {
      for (const auto &param : ast_node["parameters"]["parameters"])
      {
        code_typet::argumentt arg;
        if (get_function_params(param, c_name, arg))
          return true;
        aux_type.arguments().push_back(arg);
      }
    }
    for (const auto &param : mod_def["parameters"]["parameters"])
    {
      code_typet::argumentt arg;
      if (get_function_params(param, c_name, arg))
        return true;
      aux_type.arguments().push_back(arg);
    }
    a_sym.type = aux_type;
    // Stamp the wrapper with the base method it wraps. The wrapper is named
    // `<f_name>_<mod_name>` (get_modifier_function_name), but `_` is a legal
    // identifier char and the delimiter is unescaped, so the name alone cannot
    // be split back to the real method reliably (`a_b_mod` is ambiguous between
    // `a`+`b_mod` and `a_b`+`mod`). The Foundry coverage-test generator reads
    // this authoritative marker to attribute a covered branch inside the wrapper
    // to its externally-callable method, rather than guessing by name prefix.
    a_sym.type.set("#sol_modifier_wrapper_for", f_name);
    move_builtin_to_contract(c_name, symbol_expr(a_sym), "internal", true);

    // If the wrapped function returns a tuple, register a tuple instance
    // for the synthetic aux as well. get_tuple_instance_name builds
    // `tuple_instance$<current_functionDecl.id>` so a modifier-wrapped
    // tuple-returning function (id=0 in insert_modifier_json) otherwise
    // looks up `tuple_instance$0` which was never created, aborting with
    // "cannot find tuple instance symbol". The aux function's own return
    // type was already coerced to void above (see comment there); check
    // the ORIGINAL function's return parameters instead.
    {
      typet orig_ret_type;
      bool orig_is_tuple = false;
      if (ast_node.contains("returnParameters"))
      {
        if (
          !get_type_description(ast_node["returnParameters"], orig_ret_type) &&
          get_sol_type(orig_ret_type) ==
            SolidityGrammar::SolType::TUPLE_RETURNS)
          orig_is_tuple = true;
      }
      if (orig_is_tuple)
      {
        exprt dump;
        if (get_tuple_definition(*current_functionDecl))
          return true;
        if (get_tuple_instance(*current_functionDecl, dump))
          return true;
      }
    }

    // If the wrapped function declares NAMED return parameters (e.g.
    // `returns (string memory _doc, address[] memory _signers)`), the
    // original body references those names as ordinary local variables.
    // The caller of `get_func_modifier` has already created DECLs for
    // them under the OUTER function's scope (`sol:@C@<c>@F@<orig>@<p>#id`),
    // but when we now convert the body under the AUX function's scope
    // the lookup rebuilds the identifier as
    // `sol:@C@<c>@F@<aux>@<p>#id` — which does not exist in the symbol
    // table, so symex later crashes with
    //   value_set: unknown symbol `sol:@C@<c>@F@<aux>@<p>#id`
    // (Dataset buggy_10: getDetail with a validDoc modifier + named
    // tuple returns reproduced this.)  Re-register each named return
    // parameter under the aux scope and prepend a DECL at the start of
    // the body so the inlined references resolve locally.
    std::vector<exprt> aux_named_ret_decls;
    if (
      ast_node.contains("returnParameters") &&
      ast_node["returnParameters"].contains("parameters"))
    {
      for (const auto &rparam : ast_node["returnParameters"]["parameters"])
      {
        const std::string rname = rparam.value("name", "");
        if (rname.empty())
          continue;
        exprt aux_rdecl;
        if (get_var_decl(rparam, aux_rdecl))
          return true;
        aux_named_ret_decls.push_back(aux_rdecl);
      }
    }

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
    // The `_;` placeholder may appear at any nesting depth inside the
    // modifier body (e.g. inside an if/else, for/while, or nested block:
    //   modifier onlyOwner { if (msg.sender == owner) { _; } else revert(); }
    // A flat scan over mod_body.operands() misses such placeholders and the
    // wrapped function's body is silently dropped, so assertions inside
    // e.g. `mintToken(...) onlyOwner` are never reached by symex — they are
    // absent from the goto, which makes ESBMC wrongly conclude
    // `VERIFICATION SUCCESSFUL` on overflows it should catch.
    // Recurse into every container operand so placeholders at any depth are
    // replaced by a copy of body_exprt's statements.
    // Splice the wrapped function's body into every `_;` placeholder.
    // Substitution is parent-aware: a flat splice (erase placeholder,
    // insert N body operands as siblings) is only safe when the parent
    // is itself a code_blockt — a fixed-arity parent like
    // code_ifthenelset / code_whilet / code_fort cannot absorb N != 1
    // siblings without corrupting its operand count. In particular, a
    // bare `if (cond) _;` modifier with an empty function body (N = 0)
    // would drop the parent ifthenelse from 2 operands to 1 and crash
    // goto_convert.cpp:1647-1651; a `if (cond) _;` modifier with a
    // 2-statement body (N = 2) would silently inflate the parent to a
    // phantom 3-operand ifthenelse, splitting the body across the
    // conditional. When the parent is fixed-arity, wrap body_exprt's
    // operands in a single code_blockt so exactly one operand replaces
    // the placeholder. The block-parent path keeps the legacy flatten
    // behaviour so the top-level rewriter at line 1265-1283 (which
    // walks mod_body's outer operands looking for `return` to lift to
    // `aux_var = x`) continues to see naked statements.
    std::function<void(exprt &)> splice_placeholders = [&](exprt &node) {
      // Skip leaves so the non-const operands() accessor below does
      // not lazy-allocate an empty operands sub-irep on bare symbol
      // exprts. exprt::operands() (util/expr.h:57-60) calls
      // add(o_operands).get_sub() which materialises the field even
      // when no operands will be appended; once present, has_operands()
      // returns true based on `!find(o_operands).is_nil()` regardless
      // of operand count, poisoning the symbol for downstream
      // is_symbol-based casts. The trigger that surfaced this was a
      // modifier of shape `{ _; assert(...); }` whose unintercepted
      // c:@F@assert function-symbol got walked into and silently
      // poisoned, then crashed clang_c_adjust::do_special_functions
      // (clang_c_adjust_expr.cpp:892) on `to_symbol_expr`'s
      // `id == symbol && !has_operands()` precondition.
      if (!node.has_operands())
        return;
      const bool parent_is_block =
        node.is_code() && node.statement() == "block";
      auto &ops = node.operands();
      for (auto it = ops.begin(); it != ops.end();)
      {
        if (it->get_bool("#is_modifier_placeholder"))
        {
          if (parent_is_block)
          {
            it = ops.erase(it);
            it = ops.insert(
              it, body_exprt.operands().begin(), body_exprt.operands().end());
            std::advance(it, body_exprt.operands().size());
          }
          else
          {
            code_blockt wrapped;
            wrapped.operands() = body_exprt.operands();
            *it = wrapped;
            ++it;
          }
        }
        else
        {
          splice_placeholders(*it);
          ++it;
        }
      }
    };
    splice_placeholders(mod_body);

    // Prepend DECLs for named return parameters re-registered under
    // the aux scope (see rationale above).  Must come before the first
    // body statement that references them.
    if (!aux_named_ret_decls.empty())
    {
      auto insert_at = mod_body.operands().begin();
      for (auto &decl : aux_named_ret_decls)
        insert_at = std::next(mod_body.operands().insert(insert_at, decl));
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

      // 2. replace every "return x" to "aux_var = x". Bare `return;` inside
      // a value-returning modifier path (e.g. `if (a) return; else _;`) has
      // no operand; emitting `aux_var = <empty>` produces a symbol-typed
      // assignment that crashes symex with symbolic_type_excp. Replace it
      // with a no-op so the modifier just falls through to the implicit
      // `return aux_var` we add below — leaving aux_var at its zero
      // initialisation, which is the value-returning equivalent of "skip
      // the body".
      for (auto op = mod_body.operands().begin();
           op != mod_body.operands().end();
           ++op)
      {
        if (op->is_code() && op->statement() == "return")
        {
          if (
            op->operands().empty() || op->op0().type().id().as_string().empty())
          {
            *op = code_skipt();
          }
          else
          {
            exprt rhs = op->op0();
            code_assignt assign(symbol_expr(ret_sym), rhs);
            *op = assign;
          }
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

      // Pass through the wrapped function's own parameters so the next
      // wrapper's signature [this, wrapped_params, mod_params] is
      // satisfied. The wrapper signature was built at lines 1077-1106
      // and DOES include the wrapped params, but without this loop the
      // call only pushes [this, mod_args] and clang_c_adjust later
      // aborts with `function call: not enough arguments`. Mirrors the
      // final-call branch loop below. current_functionDecl/Name is still
      // the OUTER wrapper here (the switch to next_aux happens after),
      // and the wrapped params were registered under the outer wrapper's
      // scope by the signature loop at 1087-1098, so find_symbol
      // resolves against that aux scope.
      if (
        ast_node.contains("parameters") &&
        ast_node["parameters"].contains("parameters"))
      {
        for (const auto &param : ast_node["parameters"]["parameters"])
        {
          std::string pname, pid;
          get_local_var_decl_name(param, c_name, pname, pid);
          const symbolt *psym = context.find_symbol(pid);
          if (psym == nullptr)
          {
            log_error(
              "modifier wrapper: missing parameter symbol `{}` for function "
              "`{}` (intermediate-modifier-call)",
              pid,
              f_name);
            return true;
          }
          func_modifier.arguments().push_back(symbol_expr(*psym));
        }
      }

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

      // Pass through the wrapped function's own parameters. Order must
      // match aux_type.arguments(): [this, wrapped_params..., mod_params...].
      // At this point current_functionDecl/Name are back to the original
      // function (reset above), so get_local_var_decl_name produces the
      // original scope's ids and the symbols exist from the main param
      // loop in get_function_definition.
      if (
        ast_node.contains("parameters") &&
        ast_node["parameters"].contains("parameters"))
      {
        for (const auto &param : ast_node["parameters"]["parameters"])
        {
          std::string pname, pid;
          get_local_var_decl_name(param, c_name, pname, pid);
          const symbolt *psym = context.find_symbol(pid);
          if (psym == nullptr)
          {
            log_error(
              "modifier wrapper: missing parameter symbol `{}` for function "
              "`{}`",
              pid,
              f_name);
            return true;
          }
          func_modifier.arguments().push_back(symbol_expr(*psym));
        }
      }

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
      // Set the call's type so the chained-modifier substitution loop
      // (lines 1316-1334 above) recognises this as a value-returning
      // return.  Without a type set, op->op0().type().id().as_string()
      // is empty and the substitution mistakenly replaces this `return
      // call(next_modifier)` with `code_skipt()`, dropping the chain
      // entirely.  The aux wrapper's return type is the same as the
      // wrapped function's, captured in aux_type.return_type().
      func_modifier.type() = aux_type.return_type();
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
