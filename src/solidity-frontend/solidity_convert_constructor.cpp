/// \file solidity_convert_constructor.cpp
/// \brief Constructor conversion for the Solidity frontend.
///
/// Handles parsing of explicit Solidity constructors and generation of
/// implicit default constructors for contracts that lack one. Manages
/// constructor parameter conversion, state variable initialization
/// ordering, and base contract constructor chaining for inheritance.

#include <solidity-frontend/solidity_convert.h>
#include <set>
#include <solidity-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/focus_function.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_expr.h>
#include <util/message.h>
#include <fstream>
#include <functional>

// parse the explicit ctor, or add the implicit ctor
bool solidity_convertert::get_constructor(
  const nlohmann::json &ast_node,
  const std::string &contract_name)
{
  log_debug("solidity", "Parsing Constructor {}...", contract_name);

  // check if we could find a explicit constructor
  nlohmann::json ast_nodes = ast_node["nodes"];
  for (nlohmann::json::iterator itr = ast_nodes.begin(); itr != ast_nodes.end();
       ++itr)
  {
    nlohmann::json ast_node = *itr;
    SolidityGrammar::ContractBodyElementT type =
      SolidityGrammar::get_contract_body_element_t(ast_node);
    switch (type)
    {
    case SolidityGrammar::ContractBodyElementT::FunctionDef:
    {
      if (
        ast_node.contains("kind") && !ast_node["kind"].empty() &&
        ast_node["kind"].get<std::string>() == "constructor")
        return get_function_definition(ast_node);
      continue;
    }
    default:
    {
      continue;
    }
    }
  }

  // reset
  assert(current_functionDecl == nullptr);

  // check if we need to add implicit constructor
  if (add_implicit_constructor(contract_name))
  {
    log_error("Failed to add implicit constructor");
    return true;
  }

  return false;
}

// Tag a contract's constructor symbol as non-instantiable when the contract is
// abstract / an interface / a library (i.e. present in nonContractNamesList).
// Solidity forbids `new` on such contracts, so the Foundry coverage-test
// generator reads this flag to degrade the instantiation to UNSUPPORTED rather
// than emit an uncompilable `new <AbstractContract>(...)`. Verification-inert:
// no symex/solver reader consults it.
void solidity_convertert::mark_ctor_instantiability(
  const std::string &contract_name)
{
  std::string ctor_id;
  get_ctor_call_id(contract_name, ctor_id);
  symbolt *ctor = context.find_symbol(ctor_id);
  if (!ctor)
    return;

  // Stamp the linearized base contracts (excluding self) on the constructor.
  // The Foundry generator uses this to instantiate ONLY the most-derived
  // contract (`new Leaf(...)`): a base contract is constructed transitively by
  // its derived contract and must never be `new`'d on its own. Inert for
  // symex/solver (a `#sol_*` attribute only).
  {
    std::string bases;
    auto it = linearizedBaseList.find(contract_name);
    if (it != linearizedBaseList.end())
      for (int id : it->second)
      {
        auto nm = contractNamesMap.find(id);
        if (nm != contractNamesMap.end() && nm->second != contract_name)
          bases += (bases.empty() ? "" : " ") + nm->second;
      }
    if (!bases.empty())
      ctor->type.set("#sol_bases", bases);
  }

  if (nonContractNamesList.count(contract_name) == 0)
    return;
  ctor->type.set("#sol_no_new", true);
  // Distinguish a library (called statically as `Lib.fn(args)`) from an
  // abstract contract / interface (which cannot be called at all). The
  // Foundry generator emits a static call for the former.
  if (libraryNamesList.count(contract_name))
    ctor->type.set("#sol_library", true);
  // Mark a true interface. The Foundry generator synthesizes an
  // `ESBMCMock_<I> is <I>` only for an interface (guaranteed no constructor
  // args and no abstract receive/fallback, so the mock is always fully
  // implementable); an abstract contract is not mockable.
  if (interfaceNamesList.count(contract_name))
    ctor->type.set("#sol_interface", true);
}

// add a empty constructor to the contract
bool solidity_convertert::add_implicit_constructor(
  const std::string &contract_name)
{
  log_debug("solidity", "\t@@@ Adding implicit constructor");
  std::string name, id;
  name = contract_name;

  // do nothing if there is already an explicit or implicit ctor
  get_ctor_call_id(contract_name, id);
  if (context.find_symbol(id) != nullptr)
    return false;

  // if we reach here, the id must be equal to get_implicit_ctor_id()
  // an implicit constructor is an void empty function
  code_typet type;
  typet tmp_rtn_type("constructor");
  type.return_type() = tmp_rtn_type;
  type.set("#member_name", prefix + contract_name);
  type.set("#inlined", true);

  locationt location_begin;

  std::string debug_modulename = get_modulename_from_path(absolute_path);

  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, type, name, id, location_begin);

  symbol.lvalue = true;
  symbol.is_extern = false;
  symbol.file_local = false;

  auto &sym = *move_symbol_to_context(symbol);

  code_blockt body_exprt = code_blockt();
  sym.value = body_exprt;

  // add this pointer
  get_function_this_pointer_param(
    contract_name, id, debug_modulename, location_begin, type);

  sym.type = type;
  return false;
}

void solidity_convertert::get_temporary_object(exprt &call, exprt &new_expr)
{
  side_effect_exprt tmp_obj("temporary_object", call.type());
  codet code_expr("expression");
  code_expr.operands().push_back(call);
  tmp_obj.initializer(code_expr);
  tmp_obj.location() = call.location();
  call.swap(tmp_obj);
  new_expr = call;
}

void solidity_convertert::convert_unboundcall_nondet(
  exprt &new_expr,
  const typet common_type,
  const locationt &l)
{
  if (
    new_expr.is_code() && new_expr.statement() == "function_call" &&
    new_expr.operands().size() >= 1 &&
    new_expr.op1().name() == "_ESBMC_Nondet_Extcall")
  {
    move_to_front_block(new_expr);
    get_nondet_expr(common_type, new_expr);
    new_expr.location() = l;
  }
}

/* the member access is something like:
  class Base
  {
  public:
    static Base instance;
    void test()
    {
      instance.doSomething(); // 
    }
    void doSomething()
    {
      assert(0);
    }
  };
  Base Base::instance;

  int main()
  {
    Base::instance.test();
  }
*/
bool solidity_convertert::get_unbound_expr(
  const nlohmann::json expr,
  const std::string &c_name,
  exprt &new_expr)
{
  log_debug("solidity", "get_unbound_expr");
  if (c_name.empty())
  {
    log_error("got empty contract name");
    return true;
  }
  // it's not a member access, as it can only jump within current contract
  assert(!c_name.empty());
  code_function_callt func_call;
  if (get_unbound_funccall(c_name, func_call))
    return true;

  locationt l;
  get_location_from_node(expr, l);
  func_call.location() = l;

  // reentry check
  if (is_reentry_check)
  {
    exprt this_expr;
    assert(current_functionDecl);
    if (get_func_decl_this_ref(*current_functionDecl, this_expr))
    {
      log_error("cannot get internal this pointer reference");
      return true;
    }

    exprt _mutex;
    get_contract_mutex_expr(c_name, this_expr, _mutex);

    exprt assign_lock = side_effect_exprt("assign", bool_t);
    assign_lock.copy_to_operands(_mutex, true_exprt());
    convert_expression_to_code(assign_lock);

    exprt assign_unlock = side_effect_exprt("assign", bool_t);
    assign_unlock.copy_to_operands(_mutex, false_exprt());
    convert_expression_to_code(assign_unlock);

    // this should before the unbound_func_call
    move_to_front_block(assign_lock);
    move_to_back_block(assign_unlock);
  }

  move_to_front_block(func_call);
  new_expr = func_call;
  return false;
}

// construct the unbound verification harness
/* [APPROX: OVER] Unbound nondet entry-harness.
 *
 * For each public/external method of a contract, the harness emits a
 * nondet-guarded call with nondet parameters. msg.sender / msg.value /
 * tx.origin / block.* are all nondet. This is the default `--contract`
 * mode and is sound for safety (every reachable state of every external
 * call is explored in one transaction).
 *
 * False positives: the harness can call every public method in any order,
 *   so protocol invariants that assume a specific call ordering (e.g.
 *   "must call init() before transfer()") report spurious violations
 *   unless the contract enforces ordering internally.
 * False positives: each method is called with arbitrary nondet msg.value,
 *   so a payable branch `assert(msg.value > 0)` at the top of a function
 *   will fire even if the original test only invoked it with a positive
 *   value (see stress_libsol_fntype_inline_array_value_call).
 * False negatives: bugs reachable only through a multi-transaction
 *   sequence with state carried across transactions are NOT explored —
 *   the harness is single-transaction per loop iteration. Use
 *   --multi-transaction for multi-tx modes (where applicable).
 */
bool solidity_convertert::get_unbound_function(
  const std::string &c_name,
  symbolt &sym)
{
  std::string h_name = "_ESBMC_Nondet_Extcall_" + c_name;
  std::string h_id = "sol:@C@" + c_name + "@" + h_name + "#";
  log_debug("solidity", "\tget_unbound_function {}", h_name);

  symbolt h_sym;

  if (context.find_symbol(h_id) != nullptr)
    h_sym = *context.find_symbol(h_id);
  else
  {
    // construct unbound_function

    // 1.0 func body
    code_blockt func_body;
    func_body.make_block();

    // add __ESBMC_HIDE
    code_labelt label;
    label.set_label("__ESBMC_HIDE");
    label.code() = code_skipt();
    func_body.operands().push_back(label);

    // 1.1 get static contract instance
    exprt contract_var;
    get_static_contract_instance_ref(c_name, contract_var);

    // construct return; to avoid fall-through
    exprt return_expr = code_returnt();

    // 2.0 check visibility setting
    bool skip_vis =
      config.options.get_option("no-visibility").empty() ? false : true;
    if (skip_vis)
    {
      log_warning(
        "force to verify every function, even it's an unreachable "
        "internal/private function. This might lead to false positives.");
    }

    // 2.1 construct if-then-else statement
    const auto methods = funcSignatures[c_name];

    // --focus-function: when the caller is the target contract and a focus is
    // set, restrict the dispatch loop to the functions it names. Other contracts
    // (e.g. cross-contract targets reached from inside a focused function) keep
    // their full nondet dispatch.
    //
    // The value names a SET, not one function: `--focus-function a,b` keeps both
    // entries. Membership goes through focus_function_selects() -- the same
    // parser the path-coverage pass narrows INSTRUMENTATION with -- so the two
    // cannot disagree about which entries exist. A disagreement would be silent
    // in the worst direction: an entry this loop kept but the pass did not
    // instrument is a unit the harness can enter and no claim ever measures.
    const bool focus_applies = !focus_func.empty() && tgt_cnt_set.size() == 1 &&
                               c_name == *tgt_cnt_set.begin();

    for (const auto &method : methods)
    {
      // we only handle public (and external) function
      // as the private and internal function cannot be directly called
      if (
        !skip_vis && method.visibility != "public" &&
        method.visibility != "external")
        // skip internal and private
        continue;
      const std::string func_name = method.name;
      if (func_name == c_name)
        // skip constructor
        continue;
      // Dispatch fallback() and receive() as ordinary harness branches.
      // Skipping them used to hide assertion violations inside their bodies
      // (e.g. `assert(msg.sender == address(0))` would be unreachable because
      // the body was never invoked). The generic dispatch below is sound: the
      // harness entry already seeds msg_sender/msg_value to nondet values,
      // so the body is exercised under arbitrary caller state, which is the
      // correct over-approximation for both low-level entry points.
      if (focus_applies && !focus_function_selects(focus_func, func_name))
        // focus-function mode: skip every function on the target contract that
        // the focus does not name.
        continue;

      // then: function_call
      // do member access
      exprt mem_access = member_exprt(contract_var, method.id, method.type);

      // Overload-aware lookup: method.id is of the form
      // "sol:@C@<cname>@F@<fname>#<ast_node_id>". Match by AST node id so
      // that overloaded functions (same name, different params) resolve to
      // the exact declaration recorded in funcSignatures. The name-only
      // fallback `get_func_decl_ref(c_name, func_name)` returns the first
      // FunctionDefinition with a matching name, which for an overloaded
      // external function could silently bind to a different overload
      // (e.g. an internal one) and call it with nondet arguments whose
      // shape does not match the callee — resulting in spurious OOB or
      // type errors at symex time.
      nlohmann::json decl_ref = empty_json;
      int target_node_id = -1;
      {
        auto hash_pos = method.id.rfind('#');
        if (hash_pos != std::string::npos)
        {
          try
          {
            target_node_id = std::stoi(method.id.substr(hash_pos + 1));
          }
          catch (...)
          {
            target_node_id = -1;
          }
        }
      }
      if (target_node_id >= 0)
      {
        for (auto &top_node : src_ast_json["nodes"])
        {
          if (
            top_node.contains("nodeType") &&
            top_node["nodeType"] == "ContractDefinition" &&
            top_node.contains("name") && top_node["name"] == c_name)
          {
            for (auto &inner : top_node["nodes"])
            {
              if (
                inner.contains("nodeType") &&
                inner["nodeType"] == "FunctionDefinition" &&
                inner.contains("id") &&
                inner["id"].get<int>() == target_node_id)
              {
                decl_ref = inner;
                break;
              }
            }
            if (!decl_ref.empty())
              break;
          }
        }
      }
      if (decl_ref.empty())
        decl_ref = get_func_decl_ref(c_name, func_name);
      if (decl_ref.empty())
      {
        log_error(
          "Internal error: fail to find the definition of function {}",
          method.name);
        abort();
      }

      side_effect_expr_function_callt then_expr;
      if (get_non_library_function_call(decl_ref, empty_json, then_expr))
        return true;

      // set &_ESBMC_tmp as the first argument
      then_expr.arguments().at(0) = contract_var;
      convert_expression_to_code(then_expr);

      code_blockt then;
      then.copy_to_operands(then_expr, return_expr);

      // ifthenelse-statement:
      codet if_expr("ifthenelse");
      if_expr.copy_to_operands(nondet_bool_expr, then);

      func_body.copy_to_operands(if_expr);
    }

    // 3. construct harness
    symbolt new_symbol;
    code_typet h_type;
    typet e_type = empty_typet();
    e_type.set("cpp_type", "void");
    h_type.return_type() = e_type;
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    get_default_symbol(
      new_symbol, debug_modulename, h_type, h_name, h_id, locationt());

    new_symbol.lvalue = true;
    new_symbol.is_extern = false;
    new_symbol.file_local = false;

    symbolt &added_sym = *context.move_symbol_to_context(new_symbol);

    // no params
    h_type.make_ellipsis();

    added_sym.type = h_type;
    added_sym.value = func_body;
    h_sym = added_sym;
  }

  sym = h_sym;
  return false;
}

/* --bound mode helper: synthesise `_ESBMC_nondet_new_<c_name>()`.
 *
 * Returns a freshly heap-allocated `C *` whose state has been driven
 * through a bounded nondet-dispatch loop, i.e. the instance represents
 * some reachable Updated State (not just the Initial State left by the
 * constructor alone).
 *
 * Body shape:
 *   C *_ESBMC_nondet_new_C() {
 *       __ESBMC_HIDE:
 *       C *x = new C();                // cpp_new + ctor
 *       while (nondet_bool()) {
 *           if (nondet_bool()) (*x).f1(nondet args...);
 *           if (nondet_bool()) (*x).f2(...);
 *           ...
 *       }
 *       return x;
 *   }
 *
 * Used from `assign_param_nondet` CONTRACT branch when `--bound` is
 * active so a parameter `C c` passed into a harness function sees its
 * state vars as SMT-symbolic reachable values instead of ctor defaults.
 */
bool solidity_convertert::build_bound_drive_helper(
  const std::string &c_name,
  symbolt &sym)
{
  const std::string h_name = "_ESBMC_nondet_new_" + c_name;
  const std::string h_id = "sol:@C@" + c_name + "@F@" + h_name + "#";
  log_debug("solidity", "\tbuild_bound_drive_helper {}", h_name);

  // memoise: if already built, return it
  if (context.find_symbol(h_id) != nullptr)
  {
    sym = *context.find_symbol(h_id);
    return false;
  }

  const typet contract_struct_t = symbol_typet(prefix + c_name);
  const pointer_typet contract_ptr_t(contract_struct_t);

  // RE-ENTRY GUARD. The symbol above is registered only after the body is
  // built, and building the body drives every public method, whose
  // contract-typed params come back here (assign_param_nondet ->
  // get_non_library_function_call -> build_bound_drive_helper). Mutually
  // referencing contracts (perennial IProduct <-> IController ...) recursed
  // until the stack overflowed: SIGSEGV. A contract whose driver is already
  // being built gets a PLAIN helper instead -- `new C()` only, no nondet
  // dispatch loop -- so that nested instance keeps its constructor state
  // (an under-approximation of that one level; the outer instance is still
  // driven).
  static std::set<std::string> building;
  if (building.count(c_name))
  {
    const std::string p_name = "_ESBMC_nondet_new_plain_" + c_name;
    const std::string p_id = "sol:@C@" + c_name + "@F@" + p_name + "#";
    if (context.find_symbol(p_id) == nullptr)
    {
      code_blockt p_body;
      p_body.make_block();
      code_labelt p_label;
      p_label.set_label("__ESBMC_HIDE");
      p_label.code() = code_skipt();
      p_body.operands().push_back(p_label);
      exprt p_new;
      if (get_new_object_ctor_call(c_name, empty_json, false, p_new))
        return true;
      code_returnt p_ret;
      p_ret.return_value() = p_new;
      p_body.move_to_operands(p_ret);
      code_typet p_type;
      p_type.return_type() = contract_ptr_t;
      symbolt p_sym;
      locationt p_loc;
      p_loc.file(absolute_path);
      get_default_symbol(
        p_sym,
        get_modulename_from_path(absolute_path),
        p_type,
        p_name,
        p_id,
        p_loc);
      p_sym.lvalue = true;
      p_sym.is_extern = false;
      p_sym.file_local = false;
      symbolt &p_added = *context.move_symbol_to_context(p_sym);
      p_added.type = p_type;
      p_added.value = p_body;
    }
    sym = *context.find_symbol(p_id);
    return false;
  }
  building.insert(c_name);
  struct erase_on_exit
  {
    std::set<std::string> &s;
    std::string k;
    ~erase_on_exit()
    {
      s.erase(k);
    }
  } _guard{building, c_name};

  // 1. Build function body
  code_blockt func_body;
  func_body.make_block();

  // __ESBMC_HIDE
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  // 1.1 local pointer `x`
  const std::string x_name = "_ESBMC_nondet_new_target_" + c_name;
  const std::string x_id = h_id + "@" + x_name;
  symbolt x_sym;
  locationt x_loc;
  x_loc.file(absolute_path);
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(
    x_sym, debug_modulename, contract_ptr_t, x_name, x_id, x_loc);
  x_sym.lvalue = true;
  x_sym.file_local = true;
  x_sym.static_lifetime = false;
  symbolt &added_x = *context.move_symbol_to_context(x_sym);

  // x := new C();
  exprt new_call;
  if (get_new_object_ctor_call(c_name, empty_json, false, new_call))
    return true;

  code_declt decl_x(symbol_expr(added_x));
  decl_x.operands().resize(2);
  decl_x.op0() = symbol_expr(added_x);
  decl_x.op1() = new_call;
  func_body.move_to_operands(decl_x);

  // 1.2 while-loop body: nondet-dispatch over every public/external method
  code_blockt while_body;
  while_body.make_block();

  // Per-tx ambient reseed at the top of each dispatcher iter.
  emit_per_tx_reseed_call(while_body);

  bool skip_vis =
    config.options.get_option("no-visibility").empty() ? false : true;

  // contract_var = *x (dereference of the local pointer) — used as the
  // method-access receiver and as the implicit `this` argument.
  exprt contract_var =
    dereference_exprt(symbol_expr(added_x), contract_struct_t);

  const auto methods = funcSignatures[c_name];
  for (const auto &method : methods)
  {
    if (
      !skip_vis && method.visibility != "public" &&
      method.visibility != "external")
      continue;
    if (method.name == c_name)
      // skip constructor
      continue;

    // resolve the exact FunctionDefinition AST node from method.id
    nlohmann::json decl_ref = empty_json;
    int target_node_id = -1;
    {
      auto hash_pos = method.id.rfind('#');
      if (hash_pos != std::string::npos)
      {
        try
        {
          target_node_id = std::stoi(method.id.substr(hash_pos + 1));
        }
        catch (...)
        {
          target_node_id = -1;
        }
      }
    }
    if (target_node_id >= 0)
    {
      for (auto &top_node : src_ast_json["nodes"])
      {
        if (
          top_node.contains("nodeType") &&
          top_node["nodeType"] == "ContractDefinition" &&
          top_node.contains("name") && top_node["name"] == c_name)
        {
          for (auto &inner : top_node["nodes"])
          {
            if (
              inner.contains("nodeType") &&
              inner["nodeType"] == "FunctionDefinition" &&
              inner.contains("id") && inner["id"].get<int>() == target_node_id)
            {
              decl_ref = inner;
              break;
            }
          }
          if (!decl_ref.empty())
            break;
        }
      }
    }
    if (decl_ref.empty())
      decl_ref = get_func_decl_ref(c_name, method.name);
    if (decl_ref.empty())
    {
      log_error(
        "Internal error: fail to find the definition of function {}",
        method.name);
      abort();
    }

    side_effect_expr_function_callt then_expr;
    if (get_non_library_function_call(decl_ref, empty_json, then_expr))
      return true;
    // implicit this = *x
    then_expr.arguments().at(0) = contract_var;
    convert_expression_to_code(then_expr);

    code_blockt then;
    then.copy_to_operands(then_expr);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(nondet_bool_expr, then);
    while_body.copy_to_operands(if_expr);
  }

  // Transaction-sequence driver for this --bound peer/external instance:
  // bounded-by-default (deterministic unroll) so k-induction/BMC converges,
  // unbounded under --solidity-max-tx 0 / --solidity-precise. Same bound as
  // the top-level harness (multi_transaction_verification) for consistency.
  emit_tx_driver(func_body, while_body);

  // return x;
  code_returnt ret;
  ret.return_value() = symbol_expr(added_x);
  func_body.move_to_operands(ret);

  // 2. Build function symbol with (no args, returns C *)
  code_typet h_type;
  h_type.return_type() = contract_ptr_t;

  symbolt new_symbol;
  locationt h_loc;
  h_loc.file(absolute_path);
  get_default_symbol(new_symbol, debug_modulename, h_type, h_name, h_id, h_loc);
  new_symbol.lvalue = true;
  new_symbol.is_extern = false;
  new_symbol.file_local = false;

  symbolt &added_sym = *context.move_symbol_to_context(new_symbol);
  added_sym.type = h_type;
  added_sym.value = func_body;

  sym = added_sym;
  return false;
}

/* __ESOL_nondet_state_forward intrinsic helper:
 *   `void _ESBMC_state_forward_<c_name>(C *c)`.
 *
 * Drives *c in place through a nondet-dispatch loop over every
 * public/external method.  After the call, `*c` represents some
 * reachable state (not just the Initial State from the ctor).
 * Unlike build_bound_drive_helper this version does NOT allocate —
 * it takes a caller-supplied pointer so the user can compose it
 * (e.g. in a TOD harness, drive `c1`, then deep-copy into `c2`).
 *
 * Body shape:
 *   void _ESBMC_state_forward_C(C *c) {
 *       __ESBMC_HIDE:
 *       while (nondet_bool()) {
 *           if (nondet_bool()) (*c).f1(nondet args...);
 *           if (nondet_bool()) (*c).f2(...);
 *           ...
 *       }
 *   }
 */
bool solidity_convertert::build_esol_state_forward_helper(
  const std::string &c_name,
  symbolt &sym)
{
  const std::string h_name = "_ESBMC_state_forward_" + c_name;
  const std::string h_id = "sol:@C@" + c_name + "@F@" + h_name + "#";
  log_debug("solidity", "\tbuild_esol_state_forward_helper {}", h_name);

  // memoise
  if (context.find_symbol(h_id) != nullptr)
  {
    sym = *context.find_symbol(h_id);
    return false;
  }

  const typet contract_struct_t = symbol_typet(prefix + c_name);
  const pointer_typet contract_ptr_t(contract_struct_t);

  // 1. Function body
  code_blockt func_body;
  func_body.make_block();

  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  // 2. `c` formal parameter (C *c)
  const std::string c_param_name = "_ESBMC_state_forward_c_" + c_name;
  const std::string c_param_id = h_id + "@" + c_param_name;
  symbolt c_param_sym;
  locationt c_param_loc;
  c_param_loc.file(absolute_path);
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(
    c_param_sym,
    debug_modulename,
    contract_ptr_t,
    c_param_name,
    c_param_id,
    c_param_loc);
  c_param_sym.lvalue = true;
  c_param_sym.file_local = true;
  c_param_sym.is_parameter = true;
  symbolt &added_param = *context.move_symbol_to_context(c_param_sym);

  // 3. while-loop body: nondet-dispatch over every public/external method
  code_blockt while_body;
  while_body.make_block();

  // Per-tx ambient reseed at the top of each dispatcher iter.
  emit_per_tx_reseed_call(while_body);

  bool skip_vis =
    config.options.get_option("no-visibility").empty() ? false : true;

  // contract_var = *c — method-access receiver / implicit `this`.
  exprt contract_var =
    dereference_exprt(symbol_expr(added_param), contract_struct_t);

  const auto methods = funcSignatures[c_name];
  for (const auto &method : methods)
  {
    if (
      !skip_vis && method.visibility != "public" &&
      method.visibility != "external")
      continue;
    if (method.name == c_name)
      // skip constructor
      continue;

    // resolve the exact FunctionDefinition AST node from method.id
    nlohmann::json decl_ref = empty_json;
    int target_node_id = -1;
    {
      auto hash_pos = method.id.rfind('#');
      if (hash_pos != std::string::npos)
      {
        try
        {
          target_node_id = std::stoi(method.id.substr(hash_pos + 1));
        }
        catch (...)
        {
          target_node_id = -1;
        }
      }
    }
    if (target_node_id >= 0)
    {
      for (auto &top_node : src_ast_json["nodes"])
      {
        if (
          top_node.contains("nodeType") &&
          top_node["nodeType"] == "ContractDefinition" &&
          top_node.contains("name") && top_node["name"] == c_name)
        {
          for (auto &inner : top_node["nodes"])
          {
            if (
              inner.contains("nodeType") &&
              inner["nodeType"] == "FunctionDefinition" &&
              inner.contains("id") && inner["id"].get<int>() == target_node_id)
            {
              decl_ref = inner;
              break;
            }
          }
          if (!decl_ref.empty())
            break;
        }
      }
    }
    if (decl_ref.empty())
      decl_ref = get_func_decl_ref(c_name, method.name);
    if (decl_ref.empty())
    {
      log_error(
        "Internal error: fail to find the definition of function {}",
        method.name);
      abort();
    }

    side_effect_expr_function_callt then_expr;
    if (get_non_library_function_call(decl_ref, empty_json, then_expr))
      return true;
    // implicit this = *c
    then_expr.arguments().at(0) = contract_var;
    convert_expression_to_code(then_expr);

    code_blockt then;
    then.copy_to_operands(then_expr);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(nondet_bool_expr, then);
    while_body.copy_to_operands(if_expr);
  }

  code_whilet code_while;
  code_while.cond() = nondet_bool_expr;
  code_while.body() = while_body;
  func_body.move_to_operands(code_while);

  // 4. Function symbol: void (_ESBMC_state_forward_<C>)(C *c)
  code_typet h_type;
  h_type.return_type() = empty_typet();
  code_typet::argumentt c_arg;
  c_arg.type() = contract_ptr_t;
  c_arg.cmt_base_name(c_param_name);
  c_arg.cmt_identifier(c_param_id);
  h_type.arguments().push_back(c_arg);

  symbolt new_symbol;
  locationt h_loc;
  h_loc.file(absolute_path);
  get_default_symbol(new_symbol, debug_modulename, h_type, h_name, h_id, h_loc);
  new_symbol.lvalue = true;
  new_symbol.is_extern = false;
  new_symbol.file_local = false;

  symbolt &added_sym = *context.move_symbol_to_context(new_symbol);
  added_sym.type = h_type;
  added_sym.value = func_body;

  sym = added_sym;
  return false;
}

/* TOD clone helper: synthesise `_ESBMC_clone_<c_name>(C *base)`.
 *
 * Produces a freshly-allocated `C *` instance whose non-mapping fields
 * are a bit-level copy of `*base`, while mapping fields are retargeted
 * to the clone's fresh `$address` so subsequent reads/writes via the
 * clone do not alias `base`'s mapping keyspace (they start empty under
 * the clone's addr — a known limitation; pre-race mapping contents are
 * not mirrored).
 *
 * Used from `assign_param_nondet` to give TOD harness parameters
 * identical pre-race state: the first contract parameter of a given
 * cname is driven via `_ESBMC_nondet_new_<C>()`, and every subsequent
 * same-cname parameter is cloned from that base.  This models
 * "two orderings starting from the same reachable state S" without
 * the harness emitter having to snapshot/restore state explicitly.
 *
 * Body shape:
 *   C *_ESBMC_clone_C(C *base) {
 *       __ESBMC_HIDE:
 *       C *c = new C();                  // fresh allocation + ctor
 *       *c = *base;                      // struct-level copy
 *       c->$address = nondet_uint();     // fresh identity
 *       __ESBMC_assume(c->$address != base->$address);
 *       c->m1.addr = c->$address;        // per-mapping redirect
 *       c->m2.addr = c->$address;
 *       ...
 *       return c;
 *   }
 */
bool solidity_convertert::build_tod_clone_helper(
  const std::string &c_name,
  symbolt &sym)
{
  const std::string h_name = "_ESBMC_clone_" + c_name;
  const std::string h_id = "sol:@C@" + c_name + "@F@" + h_name + "#";
  log_debug("solidity", "\tbuild_tod_clone_helper {}", h_name);

  if (context.find_symbol(h_id) != nullptr)
  {
    sym = *context.find_symbol(h_id);
    return false;
  }

  const typet contract_struct_t = symbol_typet(prefix + c_name);
  const pointer_typet contract_ptr_t(contract_struct_t);

  // 1. Function body
  code_blockt func_body;
  func_body.make_block();

  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  // 2. `base` formal parameter
  const std::string base_name = "_ESBMC_clone_base_" + c_name;
  const std::string base_id = h_id + "@" + base_name;
  symbolt base_sym;
  locationt base_loc;
  base_loc.file(absolute_path);
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(
    base_sym, debug_modulename, contract_ptr_t, base_name, base_id, base_loc);
  base_sym.lvalue = true;
  base_sym.file_local = true;
  base_sym.is_parameter = true;
  symbolt &added_base = *context.move_symbol_to_context(base_sym);

  // 3. Local `c` pointer, initialised via cpp_new (runs the contract's
  //    default constructor).
  const std::string c_var_name = "_ESBMC_clone_c_" + c_name;
  const std::string c_var_id = h_id + "@" + c_var_name;
  symbolt c_sym;
  locationt c_loc;
  c_loc.file(absolute_path);
  get_default_symbol(
    c_sym, debug_modulename, contract_ptr_t, c_var_name, c_var_id, c_loc);
  c_sym.lvalue = true;
  c_sym.file_local = true;
  symbolt &added_c = *context.move_symbol_to_context(c_sym);

  exprt new_call;
  if (get_new_object_ctor_call(c_name, empty_json, false, new_call))
    return true;

  code_declt decl_c(symbol_expr(added_c));
  decl_c.operands().resize(2);
  decl_c.op0() = symbol_expr(added_c);
  decl_c.op1() = new_call;
  func_body.move_to_operands(decl_c);

  // 4. Whole-struct copy *c = *base.
  //    GOTO/SSA expands this into per-field assignments automatically —
  //    no need for the frontend to enumerate components (which previously
  //    silently no-op'd when context.find_symbol(tag-<C>) missed the
  //    merged struct for inherited contracts).  The $address field is
  //    overwritten in step 5 to give the clone a fresh identity.
  exprt c_deref = dereference_exprt(symbol_expr(added_c), contract_struct_t);
  exprt base_deref =
    dereference_exprt(symbol_expr(added_base), contract_struct_t);
  {
    code_assignt struct_copy(c_deref, base_deref);
    func_body.move_to_operands(struct_copy);
  }

  // 5. c->$address = nondet_uint();
  //    Require c's addr to differ from base's addr so they do not alias.
  const symbolt *addr_sym =
    context.find_symbol(prefix + c_name); // just to sanity-check struct exists
  (void)addr_sym;
  // c->$address = _ESBMC_get_unique_address(c, cname).
  //
  // T1.1 Stage S2: route through the unique-address helper so the clone's
  // $address is REGISTERED in `sol_addr_array`.  Previously this used a
  // bare `nondet_uint()` cast to `addr_t`, which gave the clone a fresh
  // address but left it unregistered — subsequent `new C()` calls in a
  // dispatcher loop could produce a NEW instance with the same nondet
  // value as a prior unregistered clone, silently aliasing length and
  // element keyspaces under the new addr-keyed dyn-array model.  Using
  // the unique-address helper keeps the address narrow (the loose
  // variant still casts a uint32 nondet to address_t, preserving the
  // mapping-fold collision invariant) AND enforces distinct-from-prior
  // via the registered slot list.
  side_effect_expr_function_callt unique_addr;
  get_library_function_call_no_args(
    "_ESBMC_get_unique_address",
    "c:@F@_ESBMC_get_unique_address",
    addr_t,
    c_loc,
    unique_addr);
  unique_addr.arguments().push_back(symbol_expr(added_c));
  exprt cname_str;
  get_cname_expr(c_name, cname_str);
  unique_addr.arguments().push_back(cname_str);

  // c->$address = unique_addr
  exprt c_addr_member = member_exprt(c_deref, "$address", addr_t);
  code_assignt addr_assign(c_addr_member, unique_addr);
  func_body.move_to_operands(addr_assign);

  // __ESBMC_assume(c->$address != base->$address)
  exprt base_addr_member = member_exprt(base_deref, "$address", addr_t);
  exprt neq_expr = exprt("notequal", bool_type());
  neq_expr.copy_to_operands(c_addr_member, base_addr_member);
  side_effect_expr_function_callt assume_call;
  get_library_function_call_no_args(
    "__ESBMC_assume", "c:@F@__ESBMC_assume", empty_typet(), c_loc, assume_call);
  assume_call.arguments().push_back(neq_expr);
  convert_expression_to_code(assume_call);
  func_body.move_to_operands(assume_call);

  // 5b. T1.1 Stage S3: per-state-var dyn-array length+element copy.
  //
  //     State-var dyn-arrays live OUTSIDE the contract struct (they
  //     are global SMT arrays, see is_dynarray_state branch in
  //     get_var_decl), so the `*c = *base` struct copy above doesn't
  //     touch them — the walker at step 6 also explicitly skips them.
  //     Instead, iterate `dynarray_state_vars[c_name]` collected at
  //     decl-time and emit, per dyn-array:
  //       1. `<arr>_dynarray_len[clone.addr] = <arr>_dynarray_len[base.addr]`
  //       2. `for (i = 0; i < clone.len; i++)`
  //          `  <arr>[fold(clone.addr, i)] = <arr>[fold(base.addr, i)]`
  //     The for-loop is bounded by `--unwind`; long arrays need a
  //     larger unwind to fully copy (analogous to `_ESBMC_arrcpy`'s
  //     memcpy-fallback bound).
  {
    auto it = dynarray_state_vars.find(c_name);
    if (it != dynarray_state_vars.end())
    {
      const std::string debug_modulename =
        get_modulename_from_path(absolute_path);
      const typet uint256_t_typet = unsignedbv_typet(256);
      const typet uint64_t_typet = unsignedbv_typet(64);
      const typet bool_t = bool_typet();

      for (const auto &[var_id, elem_type] : it->second)
      {
        const std::string len_id_str = var_id + "_dynarray_len";
        const symbolt *arr_sym = context.find_symbol(var_id);
        const symbolt *len_sym = context.find_symbol(len_id_str);
        if (!arr_sym || !len_sym)
          continue;

        // base.$address and clone.$address as exprs
        exprt base_addr_e = member_exprt(base_deref, "$address", addr_t);
        exprt clone_addr_e = member_exprt(c_deref, "$address", addr_t);

        // 1. Copy length:
        //      <arr>_dynarray_len[clone.addr] = <arr>_dynarray_len[base.addr]
        exprt clone_len_ref =
          index_exprt(symbol_expr(*len_sym), clone_addr_e, uint256_t_typet);
        exprt base_len_ref =
          index_exprt(symbol_expr(*len_sym), base_addr_e, uint256_t_typet);
        code_assignt len_copy(clone_len_ref, base_len_ref);
        func_body.move_to_operands(len_copy);

        // 2. Per-element copy loop.
        //    Counter symbol _i.
        std::string ctr_name, ctr_id;
        get_aux_var(ctr_name, ctr_id);
        symbolt ctr_sym;
        get_default_symbol(
          ctr_sym, debug_modulename, uint256_t_typet, ctr_name, ctr_id, c_loc);
        ctr_sym.lvalue = true;
        ctr_sym.file_local = true;
        ctr_sym.value = gen_zero(uint256_t_typet);
        auto &added_ctr = *move_symbol_to_context(ctr_sym);
        exprt ctr_ref = symbol_expr(added_ctr);

        code_declt ctr_decl(ctr_ref);
        ctr_decl.operands().push_back(gen_zero(uint256_t_typet));
        func_body.move_to_operands(ctr_decl);

        code_assignt init_assign(ctr_ref, gen_zero(uint256_t_typet));
        // Re-read clone length each iteration for the cond — cheap and
        // matches the pattern used by other state-var copy loops.
        exprt cond = gen_binary("<", bool_t, ctr_ref, clone_len_ref);
        exprt one = constant_exprt(
          integer2binary(1, bv_width(uint256_t_typet)), "1", uint256_t_typet);
        code_assignt iter_assign(
          ctr_ref, gen_binary("+", uint256_t_typet, ctr_ref, one));

        // Body: arr[fold(clone.addr, i)] = arr[fold(base.addr, i)].
        side_effect_expr_function_callt clone_fold;
        get_library_function_call_no_args(
          "_ESBMC_dynarr_idx",
          "c:@F@_ESBMC_dynarr_idx",
          uint64_t_typet,
          c_loc,
          clone_fold);
        clone_fold.arguments().push_back(clone_addr_e);
        clone_fold.arguments().push_back(ctr_ref);

        side_effect_expr_function_callt base_fold;
        get_library_function_call_no_args(
          "_ESBMC_dynarr_idx",
          "c:@F@_ESBMC_dynarr_idx",
          uint64_t_typet,
          c_loc,
          base_fold);
        base_fold.arguments().push_back(base_addr_e);
        base_fold.arguments().push_back(ctr_ref);

        exprt clone_elem =
          index_exprt(symbol_expr(*arr_sym), clone_fold, elem_type);
        exprt base_elem =
          index_exprt(symbol_expr(*arr_sym), base_fold, elem_type);

        // T1.1 Stage S5.5: when the SMT element type is a heap pointer
        // (the inner-row pointer for a 2D state-var dyn-array such as
        // `uint256[][]`), a bit-level copy `clone_elem = base_elem`
        // leaves clone aliasing base's inner heap row.  Real Solidity
        // deep-copies the inner row.  Emit
        //   clone_elem = _ESBMC_arrcpy(base_elem,
        //                              _ESBMC_array_length(base_elem),
        //                              sizeof(leaf_t))
        // which allocates a fresh inner buffer and element-copies via
        // _ESBMC_arrcpy's typed-loop branch (uint256_t/int256_t).
        // Scalar elem_type (1D state-var dyn-array) keeps the existing
        // bit-copy path.
        code_assignt body_assign(clone_elem, base_elem);
        if (elem_type.is_pointer())
        {
          const typet leaf_t = elem_type.subtype();
          exprt leaf_size_of;
          get_size_of_expr(leaf_t, leaf_size_of);

          side_effect_expr_function_callt len_call;
          get_library_function_call_no_args(
            "_ESBMC_array_length",
            "c:@F@_ESBMC_array_length",
            uint_type(),
            c_loc,
            len_call);
          len_call.arguments().push_back(base_elem);

          side_effect_expr_function_callt acpy_call;
          get_arrcpy_function_call(c_loc, acpy_call);
          acpy_call.arguments().push_back(base_elem);
          acpy_call.arguments().push_back(len_call);
          acpy_call.arguments().push_back(leaf_size_of);
          solidity_gen_typecast(ns, acpy_call, elem_type);
          body_assign = code_assignt(clone_elem, acpy_call);
        }

        code_fort copy_loop;
        copy_loop.init() = init_assign;
        copy_loop.cond() = cond;
        copy_loop.iter() = iter_assign;
        copy_loop.body() = body_assign;
        func_body.move_to_operands(copy_loop);
      }
    }
  }

  // 6. Per-field deep-copy fixup pass.
  //
  //    `*c = *base` above is a bit-level struct copy.  For fields that
  //    are stored as pointers into heap-allocated backing buffers
  //    (top-level `uint256[N]`; fixed arrays nested inside user
  //    structs) it leaves base and clone pointing at the same buffer —
  //    writes through base post-clone would be visible via clone.
  //    For mapping-t struct fields it leaves `c->m.addr == base.addr`,
  //    making clone share base's mapping keyspace.
  //
  //    The walker recurses through inline user structs so every
  //    pointer-backed array gets its own fresh `_ESBMC_arrcpy` and
  //    every mapping (at any nesting depth) has its `.addr` retargeted
  //    to the clone's fresh $address.  Scalars, bytes structs,
  //    dyn-array state vars (globals, not struct members), and
  //    contract-handle fields are unaffected — the outer `*c = *base`
  //    already produces the right value for them.
  const symbolt *struct_sym = context.find_symbol(prefix + c_name);
  if (struct_sym && struct_sym->type.id() == "struct")
  {
    const struct_typet &st = to_struct_type(struct_sym->type);
    for (const auto &comp : st.components())
    {
      // Skip nested type declarations (e.g. a nested `struct Box {...}`
      // def appears as a component of the outer contract struct with a
      // raw `struct` type — whereas a real FIELD of struct type always
      // goes through a symbol_type indirection since Solidity structs
      // are named types).  Those components are type declarations, not
      // fields — iterating into them would produce malformed
      // `this->.sub` member accesses because they have no storage slot.
      if (comp.type().id() == "struct")
        continue;
      if (comp.is_type())
        continue;
      const irep_idt &comp_name = comp.get_name();
      if (comp_name.empty())
        continue;
      const typet &comp_type = comp.type();
      exprt dst_field = member_exprt(c_deref, comp_name, comp_type);
      exprt src_field = member_exprt(base_deref, comp_name, comp_type);
      if (emit_clone_deep_copy_fixup(
            dst_field, src_field, comp_type, c_addr_member, func_body))
        return true;
    }
  }

  // 7. return c;
  code_returnt ret;
  ret.return_value() = symbol_expr(added_c);
  func_body.move_to_operands(ret);

  // 8. Build the function symbol (takes 1 param: C *base).
  code_typet h_type;
  h_type.return_type() = contract_ptr_t;
  code_typet::argumentt base_arg;
  base_arg.type() = contract_ptr_t;
  base_arg.cmt_base_name(base_name);
  base_arg.cmt_identifier(base_id);
  h_type.arguments().push_back(base_arg);

  symbolt new_symbol;
  locationt h_loc;
  h_loc.file(absolute_path);
  get_default_symbol(new_symbol, debug_modulename, h_type, h_name, h_id, h_loc);
  new_symbol.lvalue = true;
  new_symbol.is_extern = false;
  new_symbol.file_local = false;

  symbolt &added_sym = *context.move_symbol_to_context(new_symbol);
  added_sym.type = h_type;
  added_sym.value = func_body;

  sym = added_sym;
  return false;
}

bool solidity_convertert::needs_clone_deep_fixup(const typet &t)
{
  // mapping_t struct: always needs retarget.
  const std::string tid = t.is_symbol()
                            ? to_symbol_type(t).get_identifier().as_string()
                            : t.get("identifier").as_string();
  if (tid.find("mapping_t") != std::string::npos)
    return true;

  // Pointer-backed fixed-size array: always needs arrcpy (outer) +
  // potential element-level recurse.
  if (!t.get("#sol_array_size").empty() && t.is_pointer())
    return true;

  // Resolve symbol_type to its struct.
  typet resolved = t;
  if (resolved.is_symbol())
  {
    const symbolt *rs =
      ns.lookup(to_symbol_type(resolved).get_identifier().as_string());
    if (rs)
      resolved = rs->type;
  }
  // Bytes struct / contract struct: do NOT recurse.  Bytes structs
  // are value-semantics carriers whose scalars ride on the outer
  // struct copy; contract structs represent external handles that
  // are not owned by the cloner.
  if (
    resolved.is_struct() && !is_byte_type(resolved) &&
    get_sol_type(resolved) != SolidityGrammar::SolType::CONTRACT)
  {
    const struct_typet &st = to_struct_type(resolved);
    for (const auto &comp : st.components())
    {
      if (
        comp.type().id() == "struct" || comp.is_type() ||
        comp.get_name().empty())
        continue;
      if (needs_clone_deep_fixup(comp.type()))
        return true;
    }
  }
  return false;
}

bool solidity_convertert::emit_clone_deep_copy_fixup(
  const exprt &dst_lvalue,
  const exprt &src_lvalue,
  const typet &field_type,
  const exprt &clone_addr_expr,
  code_blockt &func_body)
{
  // The two shapes that actually need a fixup are:
  //   - pointer-backed fixed-size arrays (`#sol_array_size` set on a
  //     pointer type): *c = *base has copied only the pointer, so we
  //     reallocate + element-copy into a fresh buffer (via single
  //     _ESBMC_arrcpy for trivial element types, or compile-time-
  //     unrolled per-element recurse when the element needs its own
  //     fixup);
  //   - mapping_t struct instances: retarget `.addr` to the clone's
  //     fresh $address so writes via the clone use a disjoint keyspace.
  //
  // Everything else either rides on the outer `*c = *base` (scalars,
  // bytes structs, contract-handle pointers, dyn-array state vars —
  // which are global infinite arrays keyed elsewhere, not struct
  // components) or triggers recursion (user structs containing the
  // above).

  // Resolve symbol_type → the pointed-to struct_typet, since the
  // "is this a mapping / is this a user struct" decision is driven by
  // the resolved shape.
  typet resolved = field_type;
  std::string type_id;
  if (resolved.is_symbol())
  {
    type_id = to_symbol_type(resolved).get_identifier().as_string();
    const symbolt *rs = ns.lookup(type_id);
    if (rs)
      resolved = rs->type;
  }
  else
  {
    type_id = resolved.get("identifier").as_string();
  }

  const bool is_mapping_field = type_id.find("mapping_t") != std::string::npos;

  // -- Case 1: mapping field.  Just retarget addr.  The struct-copy
  //    already carried the mapping's `$inf_storage*` (the global pool
  //    backing it), so only the addr needs flipping for isolation.
  if (is_mapping_field)
  {
    exprt addr_member = member_exprt(dst_lvalue, "addr", addr_t);
    code_assignt assign(addr_member, clone_addr_expr);
    func_body.move_to_operands(assign);
    return false;
  }

  // -- Case 2: fixed-size array pointer.  `#sol_array_size=N` is set
  //    on the pointer type itself at get_array_pointer_type() time.
  const std::string sz_str = field_type.get("#sol_array_size").as_string();
  if (!sz_str.empty() && field_type.is_pointer())
  {
    const typet &elem_t = field_type.subtype();
    exprt size_expr = constant_exprt(
      integer2binary(string2integer(sz_str), bv_width(uint_type())),
      sz_str,
      uint_type());
    exprt size_of_expr;
    get_size_of_expr(elem_t, size_of_expr);

    if (!needs_clone_deep_fixup(elem_t))
    {
      // Trivial element type (scalar, bytes struct, contract handle,
      // or struct of same).  Single _ESBMC_arrcpy call.
      side_effect_expr_function_callt acpy_call;
      get_arrcpy_function_call(dst_lvalue.location(), acpy_call);
      acpy_call.arguments().push_back(src_lvalue);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      solidity_gen_typecast(ns, acpy_call, field_type);
      acpy_call.type().set("#sol_array_size", sz_str);
      code_assignt assign(dst_lvalue, acpy_call);
      func_body.move_to_operands(assign);
      return false;
    }

    // Multi-dim fixed array of scalar leaves (`uint[M][N]` and friends):
    // use `_ESBMC_arrcpy_2d` for a single function-call emission.
    // Emitting `c->grid = _ESBMC_alloc_array(N, 8)` followed by
    // per-slot `c->grid[i] = _ESBMC_arrcpy(base->grid[i], M, elem_sz)`
    // at the Solidity-frontend level broke symex value-set tracking:
    // successive index writes to a freshly-reassigned pointer field
    // didn't flow through to subsequent reads (see
    // `esol_clone_multi_dim_knownbug` repro).  Wrapping the whole
    // allocate+fill dance inside one C helper keeps every write inside
    // a single function frame where symex handles it cleanly.
    {
      const typet &elem_t_outer = field_type.subtype();
      const bool inner_is_ptr_backed =
        !elem_t_outer.get("#sol_array_size").empty() &&
        elem_t_outer.is_pointer();
      if (
        inner_is_ptr_backed && !needs_clone_deep_fixup(elem_t_outer.subtype()))
      {
        const std::string inner_sz_str =
          elem_t_outer.get("#sol_array_size").as_string();
        unsigned long long inner_N = 0;
        try
        {
          inner_N = std::stoull(inner_sz_str);
        }
        catch (const std::exception &)
        {
          inner_N = 0;
        }
        if (inner_N != 0)
        {
          exprt inner_size = from_integer(inner_N, size_type());
          exprt leaf_size_of_expr;
          get_size_of_expr(elem_t_outer.subtype(), leaf_size_of_expr);

          side_effect_expr_function_callt acpy2d_call;
          get_arrcpy_2d_function_call(dst_lvalue.location(), acpy2d_call);
          acpy2d_call.arguments().push_back(src_lvalue);
          acpy2d_call.arguments().push_back(size_expr);  // outer count
          acpy2d_call.arguments().push_back(inner_size); // inner count
          acpy2d_call.arguments().push_back(leaf_size_of_expr);
          solidity_gen_typecast(ns, acpy2d_call, field_type);
          acpy2d_call.type().set("#sol_array_size", sz_str);
          code_assignt assign(dst_lvalue, acpy2d_call);
          func_body.move_to_operands(assign);
          return false;
        }
      }
    }

    // Non-trivial element type: calloc a fresh outer buffer, then
    // compile-time-unrolled per-element copy + recurse.  This handles
    // array-of-struct-with-mapping (where arrcpy's memcpy fallback
    // would bit-copy mapping.addr and leave the clone aliasing base's
    // keyspace) and nested fixed arrays (multi-dim).
    side_effect_expr_function_callt calc_call;
    get_calloc_function_call(dst_lvalue.location(), calc_call);
    calc_call.arguments().push_back(size_expr);
    calc_call.arguments().push_back(size_of_expr);
    solidity_gen_typecast(ns, calc_call, field_type);
    calc_call.type().set("#sol_array_size", sz_str);
    code_assignt alloc(dst_lvalue, calc_call);
    func_body.move_to_operands(alloc);

    unsigned long long N = 0;
    try
    {
      N = std::stoull(sz_str);
    }
    catch (const std::exception &)
    {
      // Length unresolvable at compile time — fall back to no-op
      // (no worse than pre-walker behaviour).
      return false;
    }

    // Decide whether to emit an element-level bit-copy BEFORE the
    // recurse.  The bit-copy is useful for carrying scalar sub-fields
    // of a struct-typed element (the recurse only touches the
    // mapping/array sub-fields), but it is REDUNDANT when the element
    // is itself a bare pointer-backed fixed array: the recurse's
    // single `_ESBMC_arrcpy` overwrites the slot with a fresh deep
    // copy, so the intermediate pointer alias is pure noise.
    const bool elem_is_ptr_backed_array =
      !elem_t.get("#sol_array_size").empty() && elem_t.is_pointer();
    for (unsigned long long i = 0; i < N; i++)
    {
      exprt idx = from_integer(i, uint_type());
      exprt dst_elem = index_exprt(dst_lvalue, idx, elem_t);
      exprt src_elem = index_exprt(src_lvalue, idx, elem_t);
      if (!elem_is_ptr_backed_array)
      {
        // Struct / mapping-containing element: bit-copy first so
        // scalar sub-fields travel across; the recurse below only
        // fixes up pointer/mapping sub-fields.
        code_assignt elem_copy(dst_elem, src_elem);
        func_body.move_to_operands(elem_copy);
      }
      if (emit_clone_deep_copy_fixup(
            dst_elem, src_elem, elem_t, clone_addr_expr, func_body))
        return true;
    }
    return false;
  }

  // -- Case 3: inline user struct.  Recurse into each component.
  //    Skip bytes structs (BytesStatic / BytesDynamic — value-
  //    semantics carriers) and contract structs (external handles
  //    that are not part of the cloner's state).
  if (
    resolved.is_struct() && !is_byte_type(resolved) &&
    get_sol_type(resolved) != SolidityGrammar::SolType::CONTRACT)
  {
    const struct_typet &st = to_struct_type(resolved);
    for (const auto &comp : st.components())
    {
      if (comp.type().id() == "struct" || comp.is_type())
        continue; // nested type-decl, not a real field.
      const irep_idt &comp_name = comp.get_name();
      if (comp_name.empty())
        continue; // anonymous padding.
      const typet &ct = comp.type();
      exprt dst_sub = member_exprt(dst_lvalue, comp_name, ct);
      exprt src_sub = member_exprt(src_lvalue, comp_name, ct);
      if (emit_clone_deep_copy_fixup(
            dst_sub, src_sub, ct, clone_addr_expr, func_body))
        return true;
    }
    return false;
  }

  // -- Default: scalar / bytes struct / contract handle / dyn-array
  //    state-var pointer (not a struct member).  Outer struct copy
  //    already did the right thing.
  return false;
}

// ---------------------------------------------------------------------------
// Phase 2: ctor-side recursive init walker
// ---------------------------------------------------------------------------
//
// Mirror of emit_clone_deep_copy_fixup, but for the *initial* state-var
// assignment in a contract constructor.  Rationale:
//
// The state-var-decl path (solidity_convert_decl.cpp) emits a single
// `_ESBMC_calloc(N, sizeof(elem))` for a top-level pointer-backed
// fixed-size-array state var.  That is the OUTER layer.  When `elem` is
// itself pointer-backed (multi-dim `uint[M][N]`) the inner M-row buffers
// stay NULL — writes through them only "succeed" under
// `--no-standard-checks` by aliasing nondet memory, and any second
// instance (clone) sees uninitialised inner pointers.
//
// Analogously, a struct state var `B bx` where `B` has pointer-backed
// fields gets zero-initialised; `bx.cells == NULL` is the same failure
// mode one nesting level down.
//
// This walker posts inner-level `_ESBMC_calloc`s into the ctor body so
// every pointer-backed field (no matter how deep, including inside
// fixed-array elements) starts out with its own fresh backing buffer.
// Combined with the Phase 1 clone walker, this produces correct deep
// isolation for:
//   - struct field `uint256[K] cells`           (was KNOWNBUG #1)
//   - multi-dim state var `uint256[M][N] grid`  (was KNOWNBUG #2)
//
// Not handled (out of scope for Phase 2):
//   - Nested mappings inside user structs (they need
//     _ESBMC_Mapping/mapping_t rigging that the current frontend only
//     builds for top-level mapping state vars).
//   - User-provided nested array literals initialisers
//     (`uint[2][3] g = [[1,2],[3,4],[5,6]]`) — but the frontend does
//     not currently support that init form at all.
bool solidity_convertert::needs_ctor_deep_init(const typet &t)
{
  // mapping_t struct field: per-instance {base, mid, addr} init must
  // run in the ctor.  Top-level state-var mappings are handled by Block
  // B in solidity_convert_decl.cpp (which queues the init through
  // move_to_initializer); struct-internal mappings have no decl path,
  // so the walker's Case 2 must emit the same shape.  Mirrors the
  // identifier-substring test in needs_clone_deep_fixup.
  const std::string tid = t.is_symbol()
                            ? to_symbol_type(t).get_identifier().as_string()
                            : t.get("identifier").as_string();
  if (tid.find("mapping_t") != std::string::npos)
    return true;

  // Pointer-backed fixed-size array: always "needs init" from the
  // predicate's perspective.  The caller's responsibility is separated
  // into two cases — both are live:
  //   - Top-level state var: the state-var-decl path already emitted
  //     the OUTER calloc; our walker runs per-slot inner init
  //     (no-op for scalar element — Case 1 handles the early exit).
  //   - Struct-inline field: the parent struct's zero-init leaves this
  //     field NULL; our walker's Case 2 emits an OUTER calloc for it
  //     (then recurses on the slot to handle deeper nesting).
  // Either way, returning true here is safe — the walker itself decides
  // what to emit; this predicate only gates whether the walker is
  // entered for a given sub-field.
  if (!t.get("#sol_array_size").empty() && t.is_pointer())
    return true;

  // Inline user struct: recurse into components.
  typet resolved = t;
  if (resolved.is_symbol())
  {
    const symbolt *rs =
      ns.lookup(to_symbol_type(resolved).get_identifier().as_string());
    if (rs)
      resolved = rs->type;
  }
  if (
    resolved.is_struct() && !is_byte_type(resolved) &&
    get_sol_type(resolved) != SolidityGrammar::SolType::CONTRACT)
  {
    const struct_typet &st = to_struct_type(resolved);
    for (const auto &comp : st.components())
    {
      if (
        comp.type().id() == "struct" || comp.is_type() ||
        comp.get_name().empty())
        continue;
      if (needs_ctor_deep_init(comp.type()))
        return true;
    }
  }
  return false;
}

bool solidity_convertert::emit_ctor_deep_init_fixup(
  const exprt &lvalue,
  const typet &field_type,
  code_blockt &out_block,
  const std::string &path_name)
{
  // Case 0: mapping_t field at the leaf.  Emit the canonical
  // `{ base, mid, addr=this->$address }` init via the shared helper.
  // Top-level state-var mappings reach here only when the user nests a
  // `Box bx; struct Box { mapping m; }` — a top-level mapping decl is
  // initialized through Block B in get_var_decl, not the walker.
  {
    const std::string tid =
      field_type.is_symbol()
        ? to_symbol_type(field_type).get_identifier().as_string()
        : field_type.get("identifier").as_string();
    if (tid.find("mapping_t") != std::string::npos)
    {
      // Use the ctor's `this` for addr — current_functionDecl is null
      // here because the walker is invoked from move_initializer_to_ctor.
      // Even if it were non-null, we always want the ctor's $address.
      exprt ctor_this_expr;
      if (
        current_baseContractName.empty() ||
        get_ctor_decl_this_ref(current_baseContractName, ctor_this_expr))
      {
        log_error(
          "ctor walker: cannot resolve ctor `this` for nested mapping init");
        return true;
      }

      // The path must be non-empty so the global gets a unique name; if
      // the caller forgot to thread one, fall back to a counter-suffixed
      // anonymous path (still distinct, just less readable).
      std::string disambig = path_name;
      if (disambig.empty())
        disambig = "anon_" + std::to_string(next_mapping_mid);

      exprt inits;
      if (build_mapping_t_init_value(
            current_baseContractName,
            disambig,
            ctor_this_expr,
            lvalue.location(),
            inits))
        return true;

      code_assignt assign(lvalue, inits);
      out_block.move_to_operands(assign);
      return false;
    }
  }

  // Case 1: the current lvalue is already a pointer-backed fixed-size
  // array.  The outer N-slot buffer exists (allocated by the state-var
  // decl or by a parent iteration of this walker).  For each slot i in
  // [0, N), if the element type is itself pointer-backed, emit an inner
  // calloc; then recurse on the slot regardless, to handle deeper
  // nesting or struct elements with pointer sub-fields.
  const std::string sz_str = field_type.get("#sol_array_size").as_string();
  if (!sz_str.empty() && field_type.is_pointer())
  {
    unsigned long long N = 0;
    try
    {
      N = std::stoull(sz_str);
    }
    catch (const std::exception &)
    {
      return false; // non-integer length — no-op (pre-walker behaviour).
    }

    const typet &elem_t = field_type.subtype();
    const bool elem_is_ptr_array =
      !elem_t.get("#sol_array_size").empty() && elem_t.is_pointer();

    // If the element type neither itself needs an inner calloc nor
    // contains any nested pointer-backed field, we are done — the outer
    // calloc is sufficient and avoiding per-slot emission keeps the
    // SSA compact.
    if (!elem_is_ptr_array && !needs_ctor_deep_init(elem_t))
      return false;

    for (unsigned long long i = 0; i < N; i++)
    {
      exprt idx = from_integer(i, uint_type());
      exprt elem_slot = index_exprt(lvalue, idx, elem_t);

      if (elem_is_ptr_array)
      {
        // lvalue[i] = _ESBMC_calloc(inner_N, sizeof(inner_elem)).
        const std::string inner_sz_str =
          elem_t.get("#sol_array_size").as_string();
        exprt inner_size = constant_exprt(
          integer2binary(string2integer(inner_sz_str), bv_width(uint_type())),
          inner_sz_str,
          uint_type());
        exprt inner_sizeof;
        get_size_of_expr(elem_t.subtype(), inner_sizeof);

        side_effect_expr_function_callt calc_call;
        get_calloc_function_call(lvalue.location(), calc_call);
        calc_call.arguments().push_back(inner_size);
        calc_call.arguments().push_back(inner_sizeof);
        solidity_gen_typecast(ns, calc_call, elem_t);
        calc_call.type().set("#sol_array_size", inner_sz_str);
        code_assignt alloc(elem_slot, calc_call);
        out_block.move_to_operands(alloc);
      }

      // Recurse into the slot for deeper nesting (3D+) or struct elements.
      // Append "_<i>" so the global naming for any nested mapping inside
      // an element disambiguates per-slot.
      const std::string elem_path = path_name + "_" + std::to_string(i);
      if (emit_ctor_deep_init_fixup(elem_slot, elem_t, out_block, elem_path))
        return true;
    }
    return false;
  }

  // Case 2: inline user struct — recurse into each component.  For a
  // component that is itself a pointer-backed fixed-size array, emit
  // the OUTER calloc here (the struct-inline path never gets one from
  // the state-var-decl layer) and then recurse into it.  For nested
  // struct components, just recurse.
  typet resolved = field_type;
  if (resolved.is_symbol())
  {
    const symbolt *rs =
      ns.lookup(to_symbol_type(resolved).get_identifier().as_string());
    if (rs)
      resolved = rs->type;
  }
  if (
    resolved.is_struct() && !is_byte_type(resolved) &&
    get_sol_type(resolved) != SolidityGrammar::SolType::CONTRACT)
  {
    const struct_typet &st = to_struct_type(resolved);
    for (const auto &comp : st.components())
    {
      if (comp.type().id() == "struct" || comp.is_type())
        continue; // nested type-decl, not a real field.
      const typet &ct = comp.type();
      if (comp.get_name().empty())
        continue; // anonymous padding — nothing to init.
      if (!needs_ctor_deep_init(ct))
        continue;

      exprt field_lvalue = member_exprt(lvalue, comp.get_name(), ct);

      if (!ct.get("#sol_array_size").empty() && ct.is_pointer())
      {
        // Struct-inline pointer-backed array: allocate the outer buffer.
        const std::string csz_str = ct.get("#sol_array_size").as_string();
        exprt csz = constant_exprt(
          integer2binary(string2integer(csz_str), bv_width(uint_type())),
          csz_str,
          uint_type());
        exprt cso;
        get_size_of_expr(ct.subtype(), cso);

        side_effect_expr_function_callt calc_call;
        get_calloc_function_call(lvalue.location(), calc_call);
        calc_call.arguments().push_back(csz);
        calc_call.arguments().push_back(cso);
        solidity_gen_typecast(ns, calc_call, ct);
        calc_call.type().set("#sol_array_size", csz_str);
        code_assignt alloc(field_lvalue, calc_call);
        out_block.move_to_operands(alloc);
      }

      const std::string sub_path =
        path_name.empty() ? comp.get_name().as_string()
                          : path_name + "_" + comp.get_name().as_string();
      if (emit_ctor_deep_init_fixup(field_lvalue, ct, out_block, sub_path))
        return true;
    }
    return false;
  }

  // Default: scalar / bytes struct / contract handle / dyn-array state
  // var / mapping / (etc.) — the outer assignment already produced the
  // right shape.
  return false;
}

// Normally, we would expect expr to be a code_declt expression
void solidity_convertert::move_to_initializer(const exprt &expr)
{
  // the initializer will clear its elements, so we populate the copy instead of origins
  if (!ctor_frontBlockDecl.operands().empty())
  {
    // reverse order
    for (auto &op : ctor_frontBlockDecl.operands())
    {
      convert_expression_to_code(op);
      initializers.copy_to_operands(op);
    }
    ctor_frontBlockDecl.clear();
  }

  initializers.copy_to_operands(expr);

  if (!ctor_backBlockDecl.operands().empty())
  {
    // reverse order
    for (auto &op : ctor_backBlockDecl.operands())
    {
      convert_expression_to_code(op);
      initializers.copy_to_operands(op);
    }
    ctor_backBlockDecl.clear();
  }
}

bool solidity_convertert::move_initializer_to_ctor(
  const nlohmann::json *based_contracts,
  const nlohmann::json &current_contract,
  const std::string contract_name)
{
  return move_initializer_to_ctor(
    based_contracts, current_contract, contract_name, false);
}

// move library initializer and bind-name initializer to main
bool solidity_convertert::move_initializer_to_main(codet &func_body)
{
  side_effect_expr_function_callt call;
  get_library_function_call_no_args(
    "initialize", "c:@F@initialize", empty_typet(), locationt(), call);
  convert_expression_to_code(call);
  func_body.move_to_operands(call);

  for (auto str : contractNamesList)
  {
    std::string fname, fid;
    get_bind_cname_func_name(str, fname, fid);
    if (context.find_symbol(fid) == nullptr)
      return true;
    exprt func = symbol_expr(*context.find_symbol(fid));
    side_effect_expr_function_callt _call;
    _call.function() = func;
    convert_expression_to_code(_call);
    func_body.move_to_operands(_call);
  }
  return false;
}

// convert the initialization of the state variable
// into the equivalent assignmment in the ctor
// for inheritance_ctor, we skip the builtin assignment
bool solidity_convertert::move_initializer_to_ctor(
  const nlohmann::json *based_contracts,
  const nlohmann::json &current_contract,
  const std::string contract_name,
  bool is_aux_ctor)
{
  log_debug(
    "solidity",
    "@@@ Moving initialization of the state variable to the constructor {}().",
    contract_name);

  std::string ctor_id;
  if (is_aux_ctor)
  {
    exprt dump;
    get_inherit_ctor_definition(contract_name, dump);
    ctor_id = dump.identifier().as_string();
  }
  else
  {
    if (get_ctor_call_id(contract_name, ctor_id))
    {
      log_error("cannot find the construcor");
      return true;
    }
  }

  if (context.find_symbol(ctor_id) == nullptr)
  {
    log_error("cannot find the ctor ref of {}", ctor_id);
    return true;
  }
  symbolt &sym = *context.find_symbol(ctor_id);

  // get this pointer
  exprt base;
  if (get_func_decl_this_ref(contract_name, ctor_id, base))
  {
    log_error("cannot find function's this pointer");
    return true;
  }

  // queue insert initialization of the state
  for (auto it = initializers.operands().rbegin();
       it != initializers.operands().rend();
       ++it)
  {
    if (
      it->type().is_code() &&
      to_code(*it).get_statement().as_string() == "decl")
    {
      exprt comp = to_code_decl(to_code(*it)).op0();
      log_debug(
        "solidity",
        "\t@@@ initializing symbol {} in the constructor",
        comp.name().as_string());

      bool is_state = comp.type().get("#sol_state_var") == "1";
      if (!is_state)
      {
        // auxiliary local variable we created
        exprt tmp = *it;
        sym.value.operands().insert(sym.value.operands().begin(), tmp);
        continue;
      }
      if (is_aux_ctor)
      {
        if (
          comp.name().empty() ||
          is_sol_builin_symbol(contract_name, comp.name().as_string()))
          continue;
      }

      exprt lhs = member_exprt(base, comp.name(), comp.type());
      if (context.find_symbol(comp.identifier()) == nullptr)
      {
        log_error("Interal Error: cannot find symbol");
        abort();
      }
      symbolt *symbol = context.find_symbol(comp.identifier());
      exprt rhs = symbol->value;

      // B3: native nested multi-dim fixed arrays (`array_typet(array_typet
      // (T, N), M)` from option B). A single `this->grid = { { 0 } }`
      // assignment triggers a dereference-of-array-rvalue crash in
      // `src/pointer-analysis/dereference.cpp` when symex tries to
      // materialise the array-valued RHS through `*this`. Unroll the
      // zero-init into per-leaf scalar assignments instead — each
      // `this->grid[i0][i1]...[iN-1] = 0` has a scalar LHS, which the
      // dereference pipeline handles correctly. Without this unroll the
      // struct-level default leaves nested cells nondet (observed on
      // `uint256[2][2]`: `assert(g[0][0] == 0)` fails post-ctor).
      if (
        comp.type().is_array() && comp.type().has_subtype() &&
        comp.type().subtype().is_array() && rhs.get("#zero_initializer") == "1")
      {
        std::function<void(const exprt &, const typet &)> unroll_zero =
          [&](const exprt &cur_lhs, const typet &cur_type) {
            if (cur_type.is_array())
            {
              const exprt &sz = to_array_type(cur_type).size();
              BigInt n = string2integer(sz.value().as_string(), 2);
              for (uint64_t i = 0; i < n.to_uint64(); ++i)
              {
                exprt idx = constant_exprt(
                  integer2binary(BigInt(i), bv_width(int_type())),
                  integer2string(BigInt(i)),
                  int_type());
                exprt elem = index_exprt(cur_lhs, idx, cur_type.subtype());
                unroll_zero(elem, cur_type.subtype());
              }
              return;
            }
            // Follow symbol types (e.g. the `BytesStatic` tag used for
            // `bytes32`) to their concrete definition before deciding
            // whether to recurse into struct fields or emit a scalar
            // assign. `gen_zero` on an unresolved symbol_typet returns
            // nil, which later crashes `replace_nondet` when symex
            // walks the ctor body.
            const typet &resolved = ns.follow(cur_type);
            if (resolved.is_struct())
            {
              for (const auto &field : to_struct_type(resolved).components())
              {
                exprt member =
                  member_exprt(cur_lhs, field.name(), field.type());
                unroll_zero(member, field.type());
              }
              return;
            }
            exprt zero = gen_zero(resolved);
            // If gen_zero still couldn't build a value (union, opaque
            // type), bail on this leaf rather than emit a nil RHS.
            if (zero.is_nil())
              return;
            exprt assign = side_effect_exprt("assign", cur_type);
            assign.copy_to_operands(cur_lhs, zero);
            convert_expression_to_code(assign);
            sym.value.operands().insert(sym.value.operands().begin(), assign);
          };
        unroll_zero(lhs, comp.type());
        continue;
      }

      exprt _assign;
      if (
        get_sol_type(lhs.type()) == SolidityGrammar::SolType::STRING &&
        rhs.get("#zero_initializer") != "1" && rhs.id() != "string-constant")
      {
        // p = NULL;
        // _str_assign(&p, "hello");
        // since it's in the intializer, there should be no memory leak
        get_string_assignment(lhs, rhs, _assign);
        convert_expression_to_code(_assign);
      }
      else
      {
        _assign = side_effect_exprt("assign", comp.type());
        _assign.location() = sym.location;
        assert(current_contract != nullptr);
        if (rhs.get("#zero_initializer") != "1")
          convert_type_expr(ns, rhs, comp, current_contract);
        _assign.copy_to_operands(lhs, rhs);
      }
      convert_expression_to_code(_assign);

      // Phase 2 recursive init for nested pointer-backed storage.
      // Emit inner calloc's that cover what the top-level decl + outer
      // calloc (encoded in `_assign`) leaves NULL: struct-inline fixed
      // arrays and the inner rows of multi-dim arrays.  Builds a block
      // first, then splices it in right AFTER `_assign` in the ctor
      // body — i.e. the final order is [_assign, fix1, fix2, ...].
      //
      // Works by iterating the block in reverse and inserting each at
      // `begin()` (reversing twice nets a no-op on order), then the
      // final `_assign` insert at `begin()` places `_assign` in front
      // of all the fix statements, so forward-execution is:
      //     _assign  (this->field = calloc(outer) or zero_struct)
      //     fix1     (this->field[0] = calloc(inner)  / this->field.sub = calloc(...))
      //     fix2
      //     ...
      code_blockt fix_block;
      // Seed the path with the outer state-var's name so any nested
      // mapping field gets a globally-unique `_ESBMC_inf_<C>_<path>[]`
      // backing array name.
      if (emit_ctor_deep_init_fixup(
            lhs, comp.type(), fix_block, comp.name().as_string()))
      {
        log_error("Phase 2 ctor deep-init fixup failed");
        return true;
      }
      for (auto rit = fix_block.operands().rbegin();
           rit != fix_block.operands().rend();
           ++rit)
      {
        exprt op = *rit;
        convert_expression_to_code(op);
        sym.value.operands().insert(sym.value.operands().begin(), op);
      }

      // insert before the sym.value.operands
      sym.value.operands().insert(sym.value.operands().begin(), _assign);

      // we might need to insert some expression
      // due to the convert_type_expr and get_string_assignment
      if (ctor_frontBlockDecl.operands().size() != 0)
      {
        for (auto &op : ctor_frontBlockDecl.operands())
        {
          convert_expression_to_code(op);
          sym.value.operands().insert(sym.value.operands().begin(), op);
        }
        ctor_frontBlockDecl.clear();
      }
    }
    else
    {
      exprt tmp = *it;
      convert_expression_to_code(tmp);
      sym.value.operands().insert(sym.value.operands().begin(), tmp);
    }
  }

  // insert parent ctor call in the front
  if (move_inheritance_to_ctor(based_contracts, contract_name, ctor_id, sym))
    return true;

  if (is_aux_ctor)
  {
    // hide it
    code_labelt label;
    label.set_label("__ESBMC_HIDE");
    label.code() = code_skipt();
    sym.value.operands().insert(sym.value.operands().begin(), label);
  }

  // _sol_init_
  side_effect_expr_function_callt init_call;
  get_library_function_call_no_args(
    "_sol_init_", "sol:@F@_sol_init_", empty_typet(), locationt(), init_call);
  convert_expression_to_code(init_call);
  sym.value.operands().insert(sym.value.operands().begin(), init_call);

  return false;
}

void solidity_convertert::move_to_front_block(const exprt &expr)
{
  if (current_functionDecl)
    expr_frontBlockDecl.copy_to_operands(expr);
  else
    ctor_frontBlockDecl.copy_to_operands(expr);
}

void solidity_convertert::move_to_back_block(const exprt &expr)
{
  if (current_functionDecl)
    expr_backBlockDecl.copy_to_operands(expr);
  else
    ctor_backBlockDecl.copy_to_operands(expr);
}

void solidity_convertert::flush_pending_into_body(
  codet &body,
  std::size_t front_base,
  std::size_t back_base)
{
  // Operate on whichever pending-block pair move_to_front/back_block feeds in
  // the current context (function vs. constructor body).
  code_blockt &fblk =
    current_functionDecl ? expr_frontBlockDecl : ctor_frontBlockDecl;
  code_blockt &bblk =
    current_functionDecl ? expr_backBlockDecl : ctor_backBlockDecl;

  auto &fops = fblk.operands();
  auto &bops = bblk.operands();

  // Nothing new was queued while converting the body: keep prior behaviour.
  if (fops.size() <= front_base && bops.size() <= back_base)
  {
    convert_expression_to_code(body);
    return;
  }

  code_blockt wrapper;
  // Front-block statements added by the body precede it, inside the body scope.
  for (std::size_t i = front_base; i < fops.size(); ++i)
  {
    exprt op = fops[i];
    convert_expression_to_code(op);
    wrapper.operands().push_back(op);
  }
  // Drop the moved tail; statements queued before the body (e.g. by a for/if
  // condition) stay pending so the enclosing block flushes them unconditionally.
  fops.resize(front_base);

  convert_expression_to_code(body);
  wrapper.operands().push_back(body);

  // Back-block statements added by the body follow it, still inside the scope.
  for (std::size_t i = back_base; i < bops.size(); ++i)
  {
    exprt op = bops[i];
    convert_expression_to_code(op);
    wrapper.operands().push_back(op);
  }
  bops.resize(back_base);

  body = wrapper;
}

bool solidity_convertert::move_inheritance_to_ctor(
  const nlohmann::json *based_contracts,
  const std::string contract_name,
  std::string ctor_id,
  symbolt &sym)
{
  log_debug(
    "solidity",
    "@@@ Moving parents' constructor calls to the current constructor");

  std::string this_id = ctor_id + "#this";
  if (context.find_symbol(this_id) == nullptr)
  {
    log_error("Failed to find ctor this pointer {}", this_id);
    return true;
  }
  exprt this_expr = symbol_expr(*context.find_symbol(this_id));

  // As we are handling the constructor
  const std::string old_fname = current_functionName;
  current_functionName = contract_name;

  // queue insert the ctor initializaiton based on the linearizedBaseList
  if (based_contracts != nullptr && context.find_symbol(this_id) != nullptr)
  {
    /*
      Constructors are executed in the following order:
      1 - Base2
      2 - Base1
      3 - Derived3
      contract Derived3 is Base2, Base1 {
          constructor() Base1() Base2() {}
        }

      E.g. 
        contract DD is BB(3)
      Result ctor symbol table:
        Symbol......: c:@S@DD@F@DD#
        Module......: 1
        Base name...: DD
        Mode........: C++
        Type........: constructor  (struct DD *)
        Value.......: 
        {
          BB((struct BB *)this, 3);
        }
      However, since the c++ frontend is broken(esbmc/issues/1866),
      we convert it as 
        function ctor()
        {
          // create temporary object
          Base2 _ESBMC_ctor_Base2_tmp = new Base();
          // copy value
          this.x =  _ESBMC_ctor_Base2_tmp.x ;
          ...
        }
    */

    const std::vector<int> &id_list = linearizedBaseList[contract_name];
    for (auto it = id_list.begin() + 1; it != id_list.end(); ++it)
    {
      // handling inheritance
      // skip the first one as it is the contract itself
      std::string target_c_name = contractNamesMap[*it];

      for (const auto &c_node : (*based_contracts))
      {
        assert(c_node.contains("baseName"));
        std::string c_name = c_node["baseName"]["name"].get<std::string>();
        if (c_name != target_c_name)
          continue;

        typet c_type(irept::id_symbol);
        c_type.identifier(prefix + c_name);

        // get value
        // search for the parameter list for the constructor
        // they could be in two places:
        // - contract DD is BB(3)
        // or
        // - constructor() BB(3)
        nlohmann::json c_args_list_node = empty_json;
        const nlohmann::json &ctor_node = find_constructor_ref(contract_name);

        if (c_node.contains("arguments"))
          c_args_list_node = c_node;
        else if (!ctor_node.empty())
        {
          auto _ctor = ctor_node["modifiers"];
          for (const auto &c_mdf : _ctor)
          {
            // solc 0.8.x tags base-ctor modifier invocations with
            // kind=="baseConstructorSpecifier"; solc 0.6.x omits the field
            // entirely on the same node. Only apply the kind filter when it
            // is present — otherwise fall through to the name check below,
            // which matches against c_name from linearizedBaseList and so
            // only fires on actual base contracts. Requiring kind here
            // silently drops all 0.6.x base-ctor invocations, leaving
            // get_inherit_static_contract_instance with an empty args_list
            // and producing calls with fewer actuals than the callee's
            // formal params (→ symex "not enough arguments").
            if (!c_mdf.contains("modifierName"))
              continue;
            if (
              c_mdf.contains("kind") &&
              c_mdf["kind"] != "baseConstructorSpecifier")
              continue;

            if (c_mdf["modifierName"]["name"].get<std::string>() == c_name)
            {
              c_args_list_node = c_mdf;
              break;
            }
          }
        }

        // BB _ESBMC_aux_BB = BB(&this, 3, true);
        symbolt added_ctor_symbol;
        get_inherit_static_contract_instance(
          contract_name, c_name, c_args_list_node, added_ctor_symbol);

        // copy value e.g.  this.data = X.data
        struct_typet type_complete =
          to_struct_type(context.find_symbol(prefix + contract_name)->type);
        struct_typet c_type_complete =
          to_struct_type(context.find_symbol(prefix + c_name)->type);

        exprt lhs;
        exprt rhs;
        exprt _assign;
        for (const auto &c_comp : c_type_complete.components())
        {
          for (const auto &comp : type_complete.components())
          {
            if (c_comp.name() == comp.name())
            {
              assert(!comp.name().empty());
              assert(!c_comp.name().empty());

              if (is_sol_builin_symbol(c_name, c_comp.name().as_string()))
                // skip builtin symbol.
                //e.g. this->$address = _ESBMC_ctor_A_tmp.$address;
                continue;

              lhs = member_exprt(this_expr, comp.name(), comp.type());
              rhs = member_exprt(
                symbol_expr(added_ctor_symbol), c_comp.name(), c_comp.type());
              if (get_sol_type(comp.type()) == SolidityGrammar::SolType::STRING)
                // it have been initialized so should have no dereference failure
                get_string_assignment(lhs, rhs, _assign);
              else
              {
                _assign = side_effect_exprt("assign", comp.type());
                //? convert_type_expr(ns, rhs, comp.type(), current_contract);
                _assign.copy_to_operands(lhs, rhs);
              }

              convert_expression_to_code(_assign);
              sym.value.operands().insert(
                sym.value.operands().begin(), _assign);
              break;
            }
          }
        }

        // insert ctor call
        code_declt dl(symbol_expr(added_ctor_symbol));
        dl.operands().push_back(added_ctor_symbol.value);
        sym.value.operands().insert(sym.value.operands().begin(), dl);
      }
    }
  }

  current_functionName = old_fname;
  return false;
}

bool solidity_convertert::get_ctor_decl_this_ref(
  const std::string &c_name,
  exprt &this_object)
{
  std::string ctor_id;
  if (get_ctor_call_id(c_name, ctor_id))
  {
    log_error("failed to get the ctor id");
    return true;
  }

  if (get_func_decl_this_ref(c_name, ctor_id, this_object))
  {
    log_error("failed to get this ref of function {}", ctor_id);
    return true;
  }
  return false;
}

bool solidity_convertert::get_ctor_decl_this_ref(
  const nlohmann::json &caller,
  exprt &this_object)
{
  log_debug("solidity", "get_ctor_decl_this_ref");

  // ctor
  std::string current_cname;
  get_current_contract_name(caller, current_cname);
  if (current_cname.empty())
  {
    log_error("Failed to get caller's contract name");
    return true;
  }

  return get_ctor_decl_this_ref(current_cname, this_object);
}

// get the constructor symbol id
// noted that the ctor might not have been parsed yet
bool solidity_convertert::get_ctor_call_id(
  const std::string &contract_name,
  std::string &ctor_id)
{
  // we first try to find the explicit constructor defined in the source file.
  ctor_id = get_explicit_ctor_call_id(contract_name);
  if (ctor_id.empty())
    // then we try to find the implicit constructor we manually added
    ctor_id = get_implict_ctor_call_id(contract_name);

  if (ctor_id.empty())
  {
    // this means the neither explicit nor implicit constructor is found
    return true;
  }
  return false;
}

// get the explicit constructor symbol id
// retrun empty string if no explicit ctor
std::string
solidity_convertert::get_explicit_ctor_call_id(const std::string &contract_name)
{
  // get the constructor
  const nlohmann::json &ctor_ref = find_constructor_ref(contract_name);
  if (!ctor_ref.empty())
  {
    int id = ctor_ref["id"].get<int>();
    return "sol:@C@" + contract_name + "@F@" + contract_name + "#" +
           std::to_string(id);
  }

  // not found
  return "";
}

// get the implicit constructor symbol id
std::string
solidity_convertert::get_implict_ctor_call_id(const std::string &contract_name)
{
  // for implicit ctor, the id is manually set as 0
  return "sol:@C@" + contract_name + "@F@" + contract_name + "#";
}

// return construcor node based on the *contract* id
const nlohmann::json &solidity_convertert::find_constructor_ref(int contract_id)
{
  nlohmann::json &nodes = src_ast_json["nodes"];
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if (
      (*itr)["id"].get<int>() == contract_id &&
      (*itr)["nodeType"] == "ContractDefinition")
    {
      nlohmann::json &ast_nodes = (*itr)["nodes"];
      for (nlohmann::json::iterator ittr = ast_nodes.begin();
           ittr != ast_nodes.end();
           ++ittr)
      {
        if ((*ittr).contains("kind") && (*ittr)["kind"] == "constructor")
          return *ittr;
      }
    }
  }

  // implicit constructor call
  return empty_json;
}

const nlohmann::json &
solidity_convertert::find_constructor_ref(const std::string &contract_name)
{
  log_debug(
    "solidity", "\t@@@ finding reference of constructor {}", contract_name);
  nlohmann::json &nodes = src_ast_json["nodes"];
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if (
      (*itr).contains("name") &&
      (*itr)["name"].get<std::string>() == contract_name &&
      (*itr)["nodeType"] == "ContractDefinition")
    {
      nlohmann::json &ast_nodes = (*itr)["nodes"];
      for (nlohmann::json::iterator ittr = ast_nodes.begin();
           ittr != ast_nodes.end();
           ++ittr)
      {
        if ((*ittr).contains("kind") && (*ittr)["kind"] == "constructor")
          return *ittr;
      }
    }
  }

  log_debug("solidity", "\t@@@ Failed to find explicit constructor");
  // implicit constructor call
  return empty_json;
}

/**
 * @param decl_ref: the declaration of the ctor. Can be empty for implicit ctor.
 * @caller: the caller node that might contain the arguments
*/
bool solidity_convertert::get_ctor_call(
  const nlohmann::json &decl_ref,
  const nlohmann::json &caller,
  side_effect_expr_function_callt &call)
{
  /*
  we need to convert
    call(1)
  to
    call(&Base, 1)
  */
  log_debug("solidity", "\t\t@@@ get_ctor_call");
  locationt l;
  get_location_from_node(caller, l);

  if (!decl_ref.empty())
  {
    if (get_non_library_function_call(decl_ref, caller, call))
      return true;

    // reset the type. due to the empty returnParameters, the type of the call
    // is wrongly set as void.
    const auto &_contract =
      find_parent_contract(src_ast_json["nodes"], decl_ref);
    std::string c_name = _contract["name"].get<std::string>();
    call.type() = symbol_typet(prefix + c_name);

    // reset the this object
    exprt this_object;
    get_new_object(symbol_typet(prefix + c_name), this_object);
    this_object = address_of_exprt(this_object);
    call.arguments().at(0) = this_object;
    // set constructor
    call.set("constructor", 1);
  }
  else
  {
    log_error("unexpected implicit ctor");
    return true;
  }

  return false;
}

bool solidity_convertert::get_new_object_ctor_call(
  const nlohmann::json &caller,
  const bool is_object,
  exprt &new_expr)
{
  log_debug("solidity", "generating new contract object");
  // 1. get the ctor call expr
  nlohmann::json callee_expr_json;
  // if the caller's nextnode is a NewExpression, we can use it's expression directly
  // else, we need to use the expression's expression
  if (caller["expression"]["nodeType"] == "NewExpression")
    callee_expr_json = caller["expression"];
  else
    callee_expr_json = caller["expression"]["expression"];
  int ref_decl_id = callee_expr_json["typeName"]["referencedDeclaration"];
  // get contract name
  const std::string contract_name = contractNamesMap[ref_decl_id];
  if (contract_name.empty())
  {
    log_error("cannot find the contract name");
    abort();
  }
  if (get_new_object_ctor_call(contract_name, caller, is_object, new_expr))
    return true;

  return false;
}

// return a new expression: new Base(2);
bool solidity_convertert::get_new_object_ctor_call(
  const std::string &contract_name,
  const nlohmann::json caller,
  const bool is_object,
  exprt &new_expr)
{
  log_debug("solidity", "get_new_object_ctor_call");
  assert(linearizedBaseList.count(contract_name) && !contract_name.empty());

  // setup initializer, i.e. call the constructor
  side_effect_expr_function_callt call;
  const nlohmann::json &constructor_ref = find_constructor_ref(contract_name);
  if (constructor_ref.empty())
    return get_implicit_ctor_ref(contract_name, is_object, new_expr);

  if (get_ctor_call(constructor_ref, caller, call))
    return true;

  // construct temporary object
  if (is_object)
  {
    // Base x = &sideefect(..)
    get_temporary_object(call, new_expr);
  }
  else
  {
    // Base *x = new Base();
    // Wrap the constructor call directly in a code_expression and feed it as
    // the cpp_new initializer. This lets cpp_new_initializer's
    // replace_new_object rewrite the call's `&new_object` first arg into
    // `new_ptr` so the constructor runs on the heap allocation. A prior
    // get_temporary_object()+convert_expression_to_code() wrapping caused
    // remove_temporary_object to materialise a stack-local tmp$N and
    // replace `new_object` with `&tmp$N` first, leaving the registry in
    // _ESBMC_get_unique_address pointing at a dead local — which broke
    // address(new_C).balance reads.
    codet code_expr("expression");
    code_expr.copy_to_operands(call);
    new_expr = side_effect_exprt(
      "cpp_new", pointer_typet(symbol_typet(prefix + contract_name)));
    new_expr.initializer(code_expr);
  }
  return false;
}

bool solidity_convertert::get_implicit_ctor_ref(
  const std::string &contract_name,
  const bool is_object,
  exprt &new_expr)
{
  log_debug("solidity", "\t\tgetting implicit ctor call");

  // to obtain the type info
  std::string id;
  id = get_implict_ctor_call_id(contract_name);
  if (context.find_symbol(id) == nullptr)
  {
    if (add_implicit_constructor(contract_name))
      return true;
  }

  exprt ctor = symbol_expr(*context.find_symbol(id));
  code_typet type;
  type.return_type() = symbol_typet(prefix + contract_name);

  side_effect_expr_function_callt call;
  call.function() = ctor;
  call.set("constructor", 1);
  call.type() = type.return_type();
  call.location().file(absolute_path);

  exprt this_object;
  get_new_object(symbol_typet(prefix + contract_name), this_object);
  this_object = address_of_exprt(this_object);
  call.arguments().push_back(this_object);

  if (is_object)
  {
    // Base x = &sideefect(..)
    get_temporary_object(call, new_expr);
  }
  else
  {
    // Base *x = new Base(); — see comment in get_new_object_ctor_call
    // above for why we skip the temporary_object wrapping.
    codet code_expr("expression");
    code_expr.copy_to_operands(call);
    new_expr = side_effect_exprt(
      "cpp_new", pointer_typet(symbol_typet(prefix + contract_name)));
    new_expr.initializer(code_expr);
  }
  return false;
}
