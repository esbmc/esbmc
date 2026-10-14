/// \file solidity_convert_builtin.cpp
/// \brief Built-in function and low-level call handling for the Solidity
/// frontend.
///
/// Implements recognition and conversion of Solidity built-in operations:
/// low-level calls (call, delegatecall, staticcall, transfer, send),
/// built-in properties (msg.sender, msg.value, block.number, address.balance),
/// type conversion functions, abi.encode/decode, keccak256/sha256, and the
/// move_builtin_to_contract() helper for contract-scoped symbol registration.

#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith/arith_tools.h>
#include <util/arith/bitvector.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/base/i2string.h>
#include <util/arith/mp_arith.h>
#include <util/irep/std_expr.h>
#include <util/message/message.h>
#include <fstream>

bool solidity_convertert::is_low_level_call(const std::string &name)
{
  std::set<std::string> llc_set = {
    "call", "delegatecall", "staticcall", "callcode", "transfer", "send"};
  if (llc_set.count(name) != 0)
    return true;

  return false;
}

bool solidity_convertert::is_low_level_property(const std::string &name)
{
  std::set<std::string> llc_set = {"code", "codehash", "balance"};
  if (llc_set.count(name) != 0)
    return true;

  return false;
}

// e.g. address(x).balance => x->balance
// address(x).transfer() => x->transfer();

// everytime we call a ctor, we will assign an unique random address
// constructor(address _addr)
// {
//    A x = A(_addr);
// }
// =>
//  A tmp = new A();
//  if(_ESBMC_get_addr_array_idx(_addr) == -1)
//     tmp = &(struct A*)get_address_object_ptr(_addr);
// A& x = tmp;

bool solidity_convertert::add_auxiliary_members(
  const nlohmann::json &json,
  const std::string contract_name)
{
  log_debug("solidity", "@@@ adding esbmc auxiliary members");

  // name prefix:
  std::string sol_prefix = "sol:@C@" + contract_name + "@";

  // value — use 256-bit nondet for $balance/$codehash/$code so the
  // implicit zero-extend from a 32-bit nondet doesn't silently constrain
  // these uint256 fields to [0, 2^32).
  side_effect_expr_function_callt _ndt_uint = nondet_uint256_expr;

  // _ESBMC_get_unique_address(this, cname)
  // Default: 16-slot unrolled if-chain (loose; documented in README.md
  // section "Address uniqueness modelling").  --solidity-precise opts
  // into the quantifier-based encoding (sound at any slot count) — see
  // _ESBMC_get_unique_address_precise in solidity_address.c for solver
  // caveats.  Future under-approximations in the Solidity frontend
  // bind to the same flag so users get one knob, not many.
  //
  // Audit 2026-04-30: switching the default to precise was attempted
  // (S1.2) but reverted. The precise variant's for-loop is bounded by
  // --unwind, and many existing tests use --unwind 2-4 which is below
  // their actual allocation count. Under those tests, precise becomes
  // WORSE than the 16-slot loose default (the loop truncates silently
  // with --no-unwinding-assertions). Until tests are audited for
  // adequate --unwind values, keeping loose default + opt-in precise
  // is the lesser evil. KNOWNBUG-locked by
  // `address_allocator_17_distinct_pass_knownbug`.
  const bool precise = !config.options.get_option("solidity-precise").empty();
  const std::string addr_helper_name =
    precise ? "_ESBMC_get_unique_address_precise" : "_ESBMC_get_unique_address";
  const std::string addr_helper_id = "c:@F@" + addr_helper_name;

  side_effect_expr_function_callt _addr;
  locationt l;
  l.function(contract_name);

  typet t = addr_t;

  get_library_function_call_no_args(
    addr_helper_name, addr_helper_id, t, l, _addr);

  exprt this_ptr;
  std::string ctor_id;
  get_ctor_call_id(contract_name, ctor_id);

  if (get_func_decl_this_ref(contract_name, ctor_id, this_ptr))
    return true;
  _addr.arguments().push_back(this_ptr);
  exprt cname_str;
  get_cname_expr(contract_name, cname_str);
  _addr.arguments().push_back(cname_str);

  // address
  get_builtin_symbol(
    "$address", sol_prefix + "$address", t, l, _addr, contract_name);

  // codehash
  get_builtin_symbol(
    "$codehash",
    sol_prefix + "$codehash",
    unsignedbv_typet(256),
    l,
    _ndt_uint,
    contract_name);
  // balance
  // For payable constructors, initialize $balance to msg.value so that
  // ether sent via new D{value: amount}() is available during the constructor.
  // For non-payable constructors, use nondet_uint as before.
  //
  // NOTE (SMTChecker parity): SMTChecker models address(this).balance as
  // nondet with assume(balance >= msg.value), accounting for pre-existing
  // funds from selfdestruct/coinbase. ESBMC does NOT yet model this because:
  // 1. address(this).balance in user code resolves through
  //    get_aux_property_function (returning a fresh nondet) rather than
  //    reading this->$balance — so an assume on $balance doesn't constrain
  //    the user-visible expression.
  // 2. Changing $balance to nondet breaks inter-contract balance tracking
  //    (e.g., new D{value: amount}() expects D.$balance == amount).
  // Prerequisite: fix address(this).balance to read this->$balance.
  //
  // Audit 2026-04-30: tightening this to gen_zero (matching real EVM
  // non-payable-ctor semantics) was attempted but reverted — it broke
  // existing tests that rely on the nondet over-approximation to set
  // up `require(address(this).balance >= V)` paths (e.g. library_
  // caller_balance_debit_fail, new_value_empty_ctor_args_fail). The
  // nondet over-approximation is load-bearing because the harness has
  // no mechanism for "between-call external transfers in" — without
  // that simulation, gen_zero makes most balance-positive paths
  // infeasible. The proper fix is a nondet-bump at method-call entry
  // (or per-iter dispatcher reseed) so $balance can grow externally
  // between calls; until then, keeping this as `_ndt_uint` (audit
  // finding F6, ledger entry #6) is the lesser evil. KNOWNBUG-locked
  // by `new_contract_initial_balance_zero_pass_knownbug` and
  // `transfer_standalone_balance_invariant_pass`.
  {
    // Non-payable ctors: initialise $balance to a BOUNDED nondet ([0, 2^128))
    // via the `_ESBMC_nondet_init_balance()` model helper rather than a full
    // uint256 nondet. An unbounded nondet lets the solver pick a value near
    // 2^256 so that a later deposit (`$balance += value`) overflows and the
    // `.call{value:}` funding check spuriously fails, silently dropping
    // reentrant callbacks (a completeness gap that hides reentrancy bugs in
    // differential harnesses). The bound lives inside the C helper, NOT as a
    // compound initializer here: the ctor-init machinery
    // (move_initializer_to_ctor) only handles a plain side-effect-call RHS,
    // so a wrapped expression such as `nondet & mask` is silently dropped.
    side_effect_expr_function_callt _bal_ndt;
    get_library_function_call_no_args(
      "_ESBMC_nondet_init_balance",
      "c:@F@_ESBMC_nondet_init_balance",
      unsignedbv_typet(256),
      l,
      _bal_ndt);
    exprt balance_init = _bal_ndt;
    auto str_field = [](const nlohmann::json &n, const char *k) {
      if (!n.contains(k) || !n[k].is_string())
        return std::string();
      return n[k].get<std::string>();
    };
    if (json.contains("nodes"))
    {
      for (const auto &node : json["nodes"])
      {
        if (
          str_field(node, "nodeType") == "FunctionDefinition" &&
          str_field(node, "kind") == "constructor" &&
          str_field(node, "stateMutability") == "payable")
        {
          balance_init = symbol_expr(*context.find_symbol("c:@msg_value"));
          break;
        }
      }
    }
    get_builtin_symbol(
      "$balance",
      sol_prefix + "$balance",
      unsignedbv_typet(256),
      l,
      balance_init,
      contract_name);
  }
  // code
  get_builtin_symbol(
    "$code",
    sol_prefix + "$code",
    unsignedbv_typet(256),
    l,
    _ndt_uint,
    contract_name);

  // for dynamic bytes
  if (add_dynamic_pool_member(json, contract_name))
    return true;

  if (is_reentry_check)
  {
    // populate reentry mutex flag
    std::string tx_name, tx_id;
    get_contract_mutex_name(contract_name, tx_name, tx_id);
    typet _t = bool_t;

    get_builtin_symbol(tx_name, tx_id, _t, l, gen_zero(_t), contract_name);
  }

  // binding
  exprt bind_expr;
  if (!is_bound)
    get_cname_expr(contract_name, bind_expr);
  else
  {
    exprt call;
    if (assign_nondet_contract_name(contract_name, call))
      return true;
    bind_expr = call;
  }

  t = string_t;
  get_builtin_symbol(
    "_ESBMC_bind_cname",
    sol_prefix + "_ESBMC_bind_cname",
    t,
    l,
    bind_expr,
    contract_name);

  get_builtin_symbol(
    "_ESBMC_bind_cname",
    sol_prefix + "_ESBMC_bind_cname",
    t,
    l,
    bind_expr,
    contract_name);

  if (populate_low_level_functions(contract_name))
    return true;

  return false;
}

// For each contract/library whose AST contains bytes/string usage, add a
// static infinity-sized backing array ($<entity>_pool) and a $dynamic_pool
// BytesPool struct member that wraps it. Without this member, lowering of
// `_b[i]` / string-literal-to-bytes conversions inside the entity's
// functions emits `member_exprt(this, "$dynamic_pool")` and crashes goto
// migration with "Looking up index of nonexistant member".
//
// Called from add_auxiliary_members for contracts; called directly from
// the library handler (see solidity_convert_decl.cpp) since libraries
// skip add_auxiliary_members but still need the pool when their bodies
// touch bytes.
bool solidity_convertert::add_dynamic_pool_member(
  const nlohmann::json &json,
  const std::string &contract_name)
{
  if (!has_contract_bytes(json))
    return false;

  std::string sol_prefix = "sol:@C@" + contract_name + "@";
  locationt l;
  l.function(contract_name);

  symbolt pool_sym;
  typet pool_t = array_typet(unsigned_char_type(), exprt("infinity"));
  std::string pool_name = "$" + contract_name + "_pool#";
  std::string pool_id = "sol:@C@" + contract_name + "@" + pool_name;

  get_default_symbol(pool_sym, "C++", pool_t, pool_name, pool_id, l);
  pool_sym.file_local = true;
  pool_sym.lvalue = true;
  pool_sym.static_lifetime = true;
  auto &added_pool_sym = *move_symbol_to_context(pool_sym);

  side_effect_expr_function_callt init_call;
  get_library_function_call_no_args(
    "bytes_pool_init",
    "c:@F@bytes_pool_init",
    symbol_typet(lib_prefix + "BytesPool"),
    l,
    init_call);

  init_call.arguments().push_back(symbol_expr(added_pool_sym));
  get_builtin_symbol(
    "$dynamic_pool",
    sol_prefix + "$dynamic_pool#",
    symbol_typet(lib_prefix + "BytesPool"),
    l,
    init_call,
    contract_name);

  return false;
}

void solidity_convertert::move_builtin_to_contract(
  const std::string cname,
  const exprt &sym,
  bool is_method)
{
  move_builtin_to_contract(cname, sym, "private", is_method);
}

void solidity_convertert::move_builtin_to_contract(
  const std::string cname,
  const exprt &sym,
  const std::string &access,
  bool is_method)
{
  std::string c_id = prefix + cname;
  if (context.find_symbol(c_id) == nullptr)
  {
    log_error("parsing order error for struct {}", c_id);
    abort();
  }
  symbolt &c_sym = *context.find_symbol(c_id);
  assert(c_sym.get_type().is_struct());

  if (!is_method)
  {
    // check if it's already inserted
    for (auto i : to_struct_type(c_sym.get_type()).components())
    {
      if (i.identifier() == sym.identifier())
        return;
    }

    struct_typet::componentt comp(sym.name(), sym.name(), sym.type());
    comp.set_access(access);
    comp.type().member_name(c_sym.get_type().tag());
    {
      typet t = c_sym.get_type();
      to_struct_type(t).components().push_back(comp);
      c_sym.set_type(std::move(t));
    }
  }
  else
  {
    // check if it's already inserted
    for (auto i : to_struct_type(c_sym.get_type()).methods())
    {
      if (i.identifier() == sym.identifier())
        return;
    }

    struct_typet::componentt comp;
    // construct comp
    comp.type() = sym.type();
    comp.identifier(sym.identifier());
    comp.name(sym.name());
    comp.pretty_name(sym.name());
    comp.set_access(access);
    comp.id("symbol");
    {
      typet t = c_sym.get_type();
      to_struct_type(t).methods().push_back(comp);
      c_sym.set_type(std::move(t));
    }
  }
}

// this function:
// - move the created auxiliary variables to the constructor
// - append the symbol as the component to the struct class
void solidity_convertert::get_builtin_symbol(
  const std::string name,
  const std::string id,
  const typet t,
  const locationt &l,
  const exprt &val,
  const std::string c_name)
{
  // skip if it's already in the symbol table
  if (context.find_symbol(id) != nullptr)
    return;

  symbolt sym;
  get_default_symbol(sym, "C++", t, name, id, l);
  set_sol_state_var(id, true);
  sym.file_local = true;
  sym.lvalue = true;
  auto &added_sym = *move_symbol_to_context(sym);
  code_declt decl(symbol_expr(added_sym));
  added_sym.set_value(migrate_expr(val));
  decl.operands().push_back(val);
  move_to_initializer(decl);

  if (!c_name.empty())
    // we need to update the fields of the contract struct symbol
    move_builtin_to_contract(c_name, symbol_expr(added_sym), false);
}

// create a function: get_{property_name}(addr)
// this function is universal for every contract
void solidity_convertert::get_aux_property_function(
  const std::string &cname,
  const exprt &base,
  const typet &return_t,
  const locationt &loc,
  const std::string &property_name,
  exprt &new_expr)
{
  std::string fname = "get_" + property_name;
  std::string fid = "sol:@C@" + cname + "@F@" + fname + "#";

  exprt cur_this_expr;
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, cur_this_expr))
      abort();
  }
  else
  {
    if (get_ctor_decl_this_ref(cname, cur_this_expr))
      abort();
  }

  assert(base.is_constant() || base.is_member() || base.is_symbol());

  if (context.find_symbol(fid) != nullptr)
  {
    side_effect_expr_function_callt _call;
    get_library_function_call_no_args(fname, fid, return_t, loc, _call);
    _call.arguments().push_back(cur_this_expr);
    _call.arguments().push_back(base);
    new_expr = _call;
    return;
  }

  // poplate function definition
  // e.g. get_balance(this, this->addr);
  symbolt sym;
  code_typet type;
  type.return_type() = return_t;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(sym, debug_modulename, type, fname, fid, loc);
  auto &added_symbol = *move_symbol_to_context(sym);

  get_function_this_pointer_param(cname, fid, debug_modulename, loc, type);

  // param: arg
  std::string aname = "_addr";
  std::string aid = "sol:@F@" + fname + "@" + aname + "#";
  addr_t.cmt_constant(true);
  symbolt addr_s;
  get_default_symbol(addr_s, debug_modulename, addr_t, aname, aid, loc);
  move_symbol_to_context(addr_s);

  auto param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(aname);
  param.cmt_identifier(aid);
  param.location() = loc;
  type.arguments().push_back(param);

  // populate param
  added_symbol.set_type(type);
  // move to struct symbol
  move_builtin_to_contract(cname, symbol_expr(added_symbol), true);

  exprt addr_expr = symbol_expr(*context.find_symbol(aid));
  /* body:
      address(_addr).code
    =>
      if(get_object(_addr, "A") != NULL)
        return  (A *)get_object(_addr, "A")->code;
      if(get_object(_addr, "B") != NULL)
        return  (B *)get_object(_addr, "B")->code;
      return nondet_uint();
  */

  code_blockt _block;

  // hide it
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  _block.move_to_operands(label);

  for (auto str : contractNamesList)
  {
    if (context.find_symbol("c:@F@_ESBMC_get_obj") == nullptr)
    {
      log_error("cannot find builtin library");
      abort();
    }
    // skip interface/abstract contract/library
    if (nonContractNamesList.count(str) != 0 && str != cname)
      continue;

    // param
    exprt _cname;
    get_cname_expr(str, _cname);

    // get_object(_addr, A)
    side_effect_expr_function_callt get_obj;
    get_library_function_call_no_args(
      "_ESBMC_get_obj",
      "c:@F@_ESBMC_get_obj",
      pointer_typet(empty_typet()),
      loc,
      get_obj);

    get_obj.arguments().push_back(addr_expr);
    get_obj.arguments().push_back(_cname);

    // typecast
    typet _struct = symbol_typet(prefix + str);
    exprt tc = typecast_exprt(get_obj, pointer_typet(_struct));

    // member access
    std::string comp_name = "$" + property_name;
    exprt mem = member_exprt(tc, comp_name, return_t);

    // return
    code_returnt ret_call;
    ret_call.return_value() = mem;

    // if(get_object(_addr, "A") != NULL)
    exprt _null = gen_zero(pointer_typet(empty_typet()));
    exprt _equal = exprt("notequal", bool_t);
    _equal.operands().push_back(get_obj);
    _equal.operands().push_back(_null);
    _equal.location() = loc;

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, ret_call);
    if_expr.location() = loc;
    _block.move_to_operands(if_expr);
  }

  // For `balance`, fall through to the EOA balance map: addresses that
  // do not match any tracked _ESBMC_Object_<C> are EOAs and their ETH
  // balance lives in `sol_eoa_balance_array` (credited by the
  // EOA-fallback in get_transfer_definition / get_send_definition).
  // The map auto-allocates a slot with nondet initial balance on first
  // sight, so unsighted addresses still over-approximate.
  //
  // For `code` / `codehash`, route through parallel per-address summary
  // maps (`sol_eoa_code_array` / `sol_eoa_codehash_array`) so that two
  // reads of the same address return the same value within a path —
  // without this branch, the fall-through emitted a fresh
  // `nondet_uint256_expr` on every read and `addr.codehash ==
  // addr.codehash` could fail.  The summary map shares the EOA address
  // pool, so each address gets exactly one slot regardless of which
  // property is read first.  --bound only; unbound mode short-circuits
  // earlier in solidity_convert_expr.cpp before reaching this aux fn.
  auto emit_helper_call =
    [&](const std::string &fname, const std::string &fid) {
      side_effect_expr_function_callt _hcall;
      get_library_function_call_no_args(fname, fid, return_t, loc, _hcall);
      _hcall.arguments().push_back(addr_expr);
      code_returnt _ret;
      _ret.return_value() = _hcall;
      _block.move_to_operands(_ret);
    };

  if (property_name == "balance")
    emit_helper_call("_ESBMC_eoa_balance_of", "c:@F@_ESBMC_eoa_balance_of");
  else if (property_name == "codehash")
    emit_helper_call("_ESBMC_codehash_of", "c:@F@_ESBMC_codehash_of");
  else if (property_name == "code")
    emit_helper_call("_ESBMC_code_of", "c:@F@_ESBMC_code_of");
  else
  {
    // catch-all: any future builtin property keeps the over-approximate
    // nondet_uint256 fallback until it gets its own per-address map.
    code_returnt ret_uint;
    ret_uint.return_value() = nondet_uint256_expr;
    _block.move_to_operands(ret_uint);
  }

  // populate body
  added_symbol.set_value(_block);

  // do function call. Set the call's own type to the helper's return
  // type — without this, the first invocation produces an untyped
  // sideeffect expr and downstream consumers (e.g. the `.length`
  // member-on-bytes fallback in solidity_convert_ref.cpp) fail their
  // is_unsignedbv()/is_signedbv() guards and emit ill-formed
  // member_exprt over a typeless source. The cached path at line
  // ~412 uses get_library_function_call_no_args which already sets
  // the type; this branch was the missing match.
  side_effect_expr_function_callt _call;
  _call.function() = symbol_expr(added_symbol);
  _call.type() = return_t;
  _call.location() = loc;
  _call.arguments().push_back(cur_this_expr);
  _call.arguments().push_back(base);
  new_expr = _call;
}

bool solidity_convertert::struct_type_has_component(
  const typet &type,
  const std::string &comp)
{
  typet rt = type;
  if (rt.id() == "pointer")
    rt = rt.subtype();
  while (rt.id() == "symbol")
  {
    const symbolt *s = context.find_symbol(to_symbol_type(rt).get_identifier());
    if (s == nullptr)
      return false;
    rt = s->get_type();
  }
  if (rt.id() != "struct")
    return false;
  return to_struct_type(rt).has_component(comp);
}

// get member access of built-in property.
// e.g. x.$balance, x.$code ...
void solidity_convertert::get_builtin_property_expr(
  const std::string &cname,
  const std::string &name,
  const exprt &base,
  const locationt &loc,
  exprt &new_expr)
{
  log_debug("solidity", "Getting built-in property");

  typet t;
  std::string comp_name = "$" + name;

  if (name == "address")
    t = addr_t;
  else if (name == "code" || name == "codehash" || name == "balance")
  {
    t = unsignedbv_typet(256);
    set_sol_type(t, SolidityGrammar::SolType::UINT256);
  }
  else
  {
    log_error("got unexpected builtin property {}", name);
    abort();
  }

  // Decide whether the handle denotes a model object whose own field is the
  // account we must read.
  //
  // `this` always does — it is the executing instance.
  //
  // A concrete contract-typed handle does too: both `new C()` and the cast
  // `C(_addr)` produce `&_ESBMC_Object_C` (convert_type_expr, the
  // address->contract branch), and `x.f()` dispatches to that same singleton
  // when `x->_ESBMC_bind_cname == C`, so reads and dispatched writes meet on
  // one object.
  //
  // An interface- or abstract-typed handle does NOT.  `IBank(_addr)` also
  // lowers to `&_ESBMC_Object_IBank`, but that singleton is never a dispatch
  // target — `get_high_level_member_access` only ever selects implementation
  // singletons (interfaces/abstracts are skipped everywhere the cname ladder
  // is built).  A direct `handle->$balance` therefore reads a field no call
  // can write: an attacker's `while (address(target).balance > 0)` guard stays
  // constant across the nested call, re-entrancy never terminates cleanly and
  // the violation is missed (regression/esbmc-solidity/reentrance_2).  Route
  // these through the address-resolution ladder, which resolves the handle's
  // `$address` against every tracked object and falls back to the EOA balance
  // map — the account identity EVM itself uses for `address(x).balance`.
  bool own_field = false;
  if (base.is_member())
  {
    if (base.op0().name() == "this")
      own_field = true;
    else if (
      get_sol_type(base.op0().type()) == SolidityGrammar::SolType::CONTRACT)
    {
      const std::string base_cname =
        base.op0().type().get("#sol_contract").as_string();
      // unclassifiable handle: keep the historical direct read
      own_field =
        base_cname.empty() || nonContractNamesList.count(base_cname) == 0;
    }
  }

  exprt mem;
  if (own_field)
    // e.g. address(_ins_).balance => _ins_.balance
    //      address(this) => this->address
    mem = member_exprt(base.op0(), comp_name, t);
  else
  {
    // e.g. address(msg.sender).balance, address(IBank(_addr)).balance
    // we do not know which instance the address points to, so over-approximate
    get_aux_property_function(cname, base, t, loc, name, mem);
  }

  mem.location() = loc;
  new_expr = mem;
}
