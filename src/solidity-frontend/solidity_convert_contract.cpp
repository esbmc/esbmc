/// \file solidity_convert_contract.cpp
/// \brief Contract-level conversion for the Solidity frontend.
///
/// Handles the top-level conversion of Solidity contracts: iterating over
/// contract body elements (state variables, functions, structs, enums,
/// events, errors, modifiers), registering the contract struct type in the
/// symbol table, generating static lifetime initialization, and managing
/// the contract-scoped conversion context.

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
#include <util/irep/std_types.h>
#include <fstream>

// ---- --path-cov-fixture: the deployment is REPLACED by a recorded state ----
//
// The constructor and the unit under test are forced to share one --unwind:
// the path enumeration's loop bound and the symex unwind bound must agree, so
// the pass installs one number for both. MEASURED on notes/coverage/poc/
// D03_StructWithDynArray.sol and D04_AddressSetShape.sol, whose constructors
// push to a dynamic array inside a struct:
//
//   default bound (4)  rc=-6, `Generated 0 VCC(s)`, 0 of 3 instrumented path
//                      claim(s) reached the solver, process aborts
//   --unwind 64        rc=1, F=3 U=0, 0.4s -- the SAME query, every path
//                      witnessed
//
// Raising the shared bound is not the fix: it is paid for in the UNIT's
// enumeration, which is exponential in loop iterations, and the deployment is
// not what the unit query is about. Removing the deployment from the query
// separates the two numbers.
//
// The mechanism this hangs off already exists and is already used: passing
// run_ctor=false to add_static_contract_instance leaves the singleton's `value`
// nil, and clang_c_maint::static_lifetime_init then emits no
// `<C>(&_ESBMC_Object_<C>)` call into __ESBMC_main at all. So "the constructor
// must not appear in the unit query" is not a new capability -- what is new is
// putting a REPRODUCIBLE state where the deployment used to be.
//
// FAIL CLOSED, every time. A named file that cannot be read, cannot be parsed,
// or names a state variable the contract does not have ABORTS. An ignored
// fixture would run the ordinary symbolic deployment while every downstream
// report said a concrete one had been used, and the two are indistinguishable
// from the outside -- the exact shape of a scope that gets silently rewritten
// by a missing input.
namespace
{
struct path_cov_fixturet
{
  bool present = false;          // a --path-cov-fixture file was given
  bool skip_constructor = false; // ... and it asks for the deployment to go
  std::string contract;          // which contract it describes
  // name -> value, in file order. A vector rather than a map so the emitted
  // assignments are in the order the fixture was written, which is the order a
  // reader of the goto dump will look for them in.
  std::vector<std::pair<std::string, BigInt>> state;
};

// Parsed once. The option is read from config.options because that is where
// every other Solidity-frontend option is read from (see get_tx_bound).
const path_cov_fixturet &path_cov_fixture()
{
  static path_cov_fixturet f;
  static bool loaded = false;
  if (loaded)
    return f;
  loaded = true;

  const std::string path = config.options.get_option("path-cov-fixture");
  if (path.empty())
    return f;

  std::ifstream in(path);
  if (!in)
  {
    log_error("--path-cov-fixture: cannot open '{}'", path);
    abort();
  }
  nlohmann::json j;
  try
  {
    in >> j;
  }
  catch (const std::exception &e)
  {
    log_error("--path-cov-fixture: '{}' is not valid JSON: {}", path, e.what());
    abort();
  }
  if (!j.is_object())
  {
    log_error("--path-cov-fixture: '{}' must hold a JSON object", path);
    abort();
  }

  f.present = true;
  f.skip_constructor = j.value("skip_constructor", false);
  f.contract = j.value("contract", std::string());
  if (f.contract.empty())
  {
    log_error(
      "--path-cov-fixture: '{}' has no \"contract\" field. A fixture that does "
      "not name the contract it describes would be applied to whichever "
      "contract happens to be the target",
      path);
    abort();
  }

  if (j.contains("state"))
  {
    if (!j["state"].is_object())
    {
      log_error(
        "--path-cov-fixture: \"state\" must be an object of name:value");
      abort();
    }
    for (auto it = j["state"].begin(); it != j["state"].end(); ++it)
    {
      std::string raw;
      if (it.value().is_string())
        raw = it.value().get<std::string>();
      else if (it.value().is_number_integer())
        raw = std::to_string(it.value().get<long long>());
      else if (it.value().is_boolean())
        raw = it.value().get<bool>() ? "1" : "0";
      else
      {
        log_error(
          "--path-cov-fixture: state['{}'] must be a decimal/0x-hex string, an "
          "integer or a bool",
          it.key());
        abort();
      }
      BigInt v;
      try
      {
        if (raw.rfind("0x", 0) == 0 || raw.rfind("0X", 0) == 0)
          v = string2integer(raw.substr(2), 16);
        else
          v = string2integer(raw, 10);
      }
      catch (const std::exception &e)
      {
        log_error(
          "--path-cov-fixture: state['{}'] = '{}' is not an integer literal",
          it.key(),
          raw);
        abort();
      }
      f.state.emplace_back(it.key(), v);
    }
  }

  log_status(
    "--path-cov-fixture: '{}' describes contract {}: skip_constructor={}, {} "
    "concrete state value(s)",
    path,
    f.contract,
    f.skip_constructor ? "true" : "false",
    f.state.size());
  return f;
}

// Does the fixture apply to THIS contract, and does it ask for the deployment
// to be dropped?
bool fixture_drops_deployment(const std::string &c_name)
{
  const path_cov_fixturet &f = path_cov_fixture();
  return f.present && f.skip_constructor && f.contract == c_name;
}
} // namespace

/*
  e.g. x = func();
  ==>  x = nondet();
       unbound();
*/
bool solidity_convertert::get_unbound_funccall(
  const std::string contractName,
  code_function_callt &call)
{
  log_debug("solidity", "\tget_unbound_funccall");
  symbolt f_sym;
  if (get_unbound_function(contractName, f_sym))
    return true;

  code_function_callt func_call;
  func_call.location() = f_sym.location;
  func_call.function() = symbol_expr(f_sym);

  call = func_call;
  return false;
}

void solidity_convertert::get_static_contract_instance_name(
  const std::string c_name,
  std::string &name,
  std::string &id)
{
  name = "_ESBMC_Object_" + c_name;
  id = "sol:@" + name + "#";
}

void solidity_convertert::get_static_contract_instance_ref(
  const std::string &c_name,
  exprt &new_expr)
{
  std::string name, id;
  get_static_contract_instance_name(c_name, name, id);
  typet t = symbol_typet(prefix + c_name);
  new_expr = exprt("symbol", t);
  new_expr.identifier(id);
  new_expr.name(name);
  new_expr.pretty_name(name);
}

void solidity_convertert::add_static_contract_instance(
  const std::string c_name,
  bool run_ctor)
{
  log_debug(
    "solidity",
    "\tAdd static instance of contract {}{}",
    c_name,
    run_ctor ? "" : " (not deployed: base of the --contract target)");

  std::string ctor_ins_name, ctor_ins_id;
  get_static_contract_instance_name(c_name, ctor_ins_name, ctor_ins_id);

  // inheritance instance: make a copy of the static instance
  // this is to make sure we insert it before the _ESBMC_Object_cname
  locationt ctor_ins_loc;
  ctor_ins_loc.file(absolute_path);
  ctor_ins_loc.line(1);
  std::string ctor_ins_debug_modulename =
    get_modulename_from_path(absolute_path);
  typet ctor_ins_typet = symbol_typet(prefix + c_name);

  symbolt ctor_ins_symbol;
  get_default_symbol(
    ctor_ins_symbol,
    ctor_ins_debug_modulename,
    ctor_ins_typet,
    ctor_ins_name,
    ctor_ins_id,
    ctor_ins_loc);
  ctor_ins_symbol.lvalue = true;
  ctor_ins_symbol.is_extern = false;
  // the instance should be set as static
  ctor_ins_symbol.static_lifetime = true;
  ctor_ins_symbol.file_local = true;

  auto &added_sym = *move_symbol_to_context(ctor_ins_symbol);

  // get value
  std::string ctor_id;
  // we do not check the return value as we might have not parsed the symbol yet
  if (get_ctor_call_id(c_name, ctor_id))
  {
    // probably the contract is not parsed yet
    log_error("cannot find ctor id for contract {}", c_name);
    abort();
  }
  if (context.find_symbol(ctor_id) == nullptr)
  {
    // this means that contract, including ctor, has not been parse yet
    // this will lead to issue in the following process, particuarly
    // this_pointer
    const auto &json = find_constructor_ref(c_name);
    if (json.empty())
    {
      if (add_implicit_constructor(c_name))
      {
        log_error("Failed to add the implicit constructor");
        abort();
      }
    }
    else
    {
      // parsing the constructor in another contract
      std::string old = current_baseContractName;
      current_baseContractName = c_name;
      if (get_function_definition(json))
      {
        auto name = json["name"].get<std::string>();
        if (name.empty())
          name = json["kind"].get<std::string>();
        log_error(
          "Failed to parse the function {}", json["name"].get<std::string>());
        abort();
      }
      current_baseContractName = old;
    }
  }

  exprt ctor;
  if (get_new_object_ctor_call(c_name, empty_json, true, ctor))
  {
    log_error("failed to construct a temporary object");
    abort();
  }

  // The symbol's `value` is what clang_c_maint::static_lifetime_init turns
  // into the `<C>(&_ESBMC_Object_<C>)` call at the top of __ESBMC_main.
  // Leaving it nil registers the singleton (every `$call` ladder and the
  // enclosing-debit helper still resolve it) without deploying it.
  //
  // --path-cov-fixture reuses exactly that: a fixture that asks for the
  // deployment to be dropped takes the same branch a non-deployed base
  // contract already takes, so the "no constructor in the unit query" property
  // is the existing mechanism's property rather than a second implementation
  // of it. The concrete state that replaces the deployment is emitted by
  // multi_transaction_verification, which is where the entry function's body
  // is built.
  if (run_ctor && fixture_drops_deployment(c_name))
  {
    log_status(
      "--path-cov-fixture: contract {} is NOT deployed -- no constructor call "
      "is emitted into the entry function; the fixture's recorded state is "
      "assigned instead",
      c_name);
    return;
  }

  if (run_ctor)
    added_sym.set_value(ctor);
}

void solidity_convertert::get_inherit_static_contract_instance_name(
  const std::string bs_c_name,
  const std::string c_name,
  std::string &name,
  std::string &id)
{
  name = "_ESBMC_aux_" + c_name;
  id = "sol:@C@" + bs_c_name + "@" + name + "#";
}

void solidity_convertert::get_inherit_ctor_definition_name(
  const std::string c_name,
  std::string &name,
  std::string &id)
{
  name = c_name;
  id = "sol:@C@" + c_name + "@F@" + name + "#0";
}

void solidity_convertert::get_inherit_ctor_definition(
  const std::string c_name,
  exprt &new_expr)
{
  std::string fname, fid;
  get_inherit_ctor_definition_name(c_name, fname, fid);

  if (context.find_symbol(fid) != nullptr)
  {
    new_expr = symbol_expr(*context.find_symbol(fid));
    return;
  }
  code_typet ft;
  typet tmp_rtn_type("constructor");
  ft.return_type() = tmp_rtn_type;
  ft.member_name(prefix + c_name);
  ft.set("#inlined", true);
  symbolt fs;
  locationt l;
  l.file(absolute_path);
  l.line(1);
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(fs, debug_modulename, ft, fname, fid, l);
  auto &add_sym = *move_symbol_to_context(fs);

  get_function_this_pointer_param(c_name, fid, debug_modulename, l, ft);

  // copy the param list
  // e.g. the original ctor: Base(int x)
  //      the inherit ctor:  Base(int x, bool aux)
  auto ctor_json = find_constructor_ref(c_name);
  bool is_implicit_ctor = ctor_json.empty();

  auto old_current_functionName = current_functionName;
  auto old_current_functionDecl = current_functionDecl;
  ctor_json["id"] = 0;
  ctor_json["is_inherited"] = true;
  current_functionDecl = &ctor_json;
  current_functionName = c_name;

  if (!is_implicit_ctor)
  {
    SolidityGrammar::ParameterListT params =
      SolidityGrammar::get_parameter_list_t(ctor_json["parameters"]);
    if (params != SolidityGrammar::ParameterListT::EMPTY)
    {
      for (const auto &decl : ctor_json["parameters"]["parameters"].items())
      {
        const nlohmann::json &func_param_decl = decl.value();

        code_typet::argumentt param;
        if (get_function_params(func_param_decl, c_name, param))
        {
          log_error("internal error in parsing params");
          abort();
        }
        ft.arguments().push_back(param);
      }
    }
  }

  // param: bool
  std::string aname = "aux";
  std::string aid = "sol:@C@" + c_name + "@F@" + c_name + "@" + aname + "#";
  typet addr_t = bool_t;
  addr_t.cmt_constant(true);
  symbolt addr_s;
  get_default_symbol(addr_s, debug_modulename, addr_t, aname, aid, l);
  move_symbol_to_context(addr_s);

  auto param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(aname);
  param.cmt_identifier(aid);
  param.location() = l;
  ft.arguments().push_back(param);
  add_sym.set_type(migrate_type(ft));

  // body
  exprt func_body = code_blockt();
  if (!is_implicit_ctor)
  {
    // insert
    for (auto &node : src_ast_json["nodes"])
    {
      if (node["nodeType"] == "ContractDefinition" && node["name"] == c_name)
        node["nodes"].push_back(ctor_json);
    }

    // 0.6.x emits `body: null` for unimplemented functions; treat as missing.
    if (
      (*current_functionDecl).contains("body") &&
      !(*current_functionDecl)["body"].is_null())
    {
      log_debug(
        "solidity", "@@@ parsing function {}'s body", current_functionName);
      if (get_block((*current_functionDecl)["body"], func_body))
      {
        log_error("internal error in constructing inherit_ctor body");
        abort();
      }
    }

    // remove insertion
    for (nlohmann::json &node : src_ast_json["nodes"])
    {
      if (node["nodeType"] == "ContractDefinition" && node["name"] == c_name)
      {
        // Reverse iterate to find the last inserted object with id == 0
        for (auto it = node.rbegin(); it != node.rend(); ++it)
        {
          if (
            it.value().is_object() && it.value().contains("id") &&
            it.value()["id"].get<int>() == 0)
          {
            // Erase requires a forward iterator, so convert it
            node.erase(std::next(it).base());
            break; // Only one needs to be removed
          }
        }
      }
    }
  }

  // reset
  current_functionDecl = old_current_functionDecl;
  current_functionName = old_current_functionName;

  add_sym.set_value(func_body);

  new_expr = symbol_expr(add_sym);
  move_builtin_to_contract(c_name, new_expr, "public", true);
}

// create an const instance only used for inheritance assignment
// e.g. this->x = _ESBMC_ctor_A_tmp.x;
// @bs_c_name: caller's contract name
// @c_name: callee ctor name
void solidity_convertert::get_inherit_static_contract_instance(
  const std::string bs_c_name,
  const std::string c_name,
  const nlohmann::json &args_list,
  symbolt &sym)
{
  log_debug("solidity", "get inherit_static_contract_instance");
  std::string ctor_ins_name, ctor_ins_id;
  get_inherit_static_contract_instance_name(
    bs_c_name, c_name, ctor_ins_name, ctor_ins_id);

  if (context.find_symbol(ctor_ins_id) != nullptr)
  {
    sym = *context.find_symbol(ctor_ins_id);
    return;
  }

  std::string ctor_ins_debug_modulename =
    get_modulename_from_path(absolute_path);
  typet ctor_Ins_typet = symbol_typet(prefix + c_name);

  symbolt ctor_ins_symbol;
  get_default_symbol(
    ctor_ins_symbol,
    ctor_ins_debug_modulename,
    ctor_Ins_typet,
    ctor_ins_name,
    ctor_ins_id,
    locationt());
  ctor_ins_symbol.lvalue = true;
  ctor_ins_symbol.file_local = true;

  symbolt &added_ctor_symbol = *move_symbol_to_context(ctor_ins_symbol);

  // create aux constructor
  // Base(&this, float y) // since solidity has no floating point
  side_effect_expr_function_callt call;
  exprt new_expr;
  get_inherit_ctor_definition(c_name, new_expr);

  exprt this_object;
  get_new_object(symbol_typet(prefix + c_name), this_object);
  call.arguments().push_back(this_object);

  // 0.6.x emits `arguments: null` for parent specifiers with no constructor
  // arguments (e.g. `contract C is IFoo`); 0.8.x omits the field. Treat
  // both as "no args" so we don't try to look up an interface's missing
  // constructor.
  if (
    args_list.contains("arguments") && !args_list["arguments"].is_null() &&
    !args_list["arguments"].empty())
  {
    // get param type
    auto &decl_ref = find_constructor_ref(c_name);
    if (decl_ref.empty() || decl_ref.is_null())
    {
      log_error("cannot find ctor ref");
      abort();
    }

    assert(decl_ref.contains("parameters"));
    nlohmann::json param_nodes = decl_ref["parameters"]["parameters"];
    nlohmann::json param = nullptr;
    nlohmann::json::iterator itr = param_nodes.begin();

    for (const auto &arg : args_list["arguments"].items())
    {
      if (itr != param_nodes.end())
      {
        if ((*itr).contains("typeDescriptions"))
        {
          param = (*itr)["typeDescriptions"];
        }
        ++itr;
      }

      exprt single_arg;
      if (get_expr(arg.value(), param, single_arg))
      {
        log_error("parsing arguments error");
        abort();
      }

      call.arguments().push_back(single_arg);
      param = nullptr;
    }
  }
  else
  {
    // Abstract/intermediate contract that inherits from a ctor-bearing base
    // without supplying base-ctor args in its own header or modifier list.
    // Example: `abstract contract G is ERC20, Ownable { constructor(X x) {} }`
    // — ERC20's (name, symbol) are supplied by the concrete descendant, not
    // by G. We still synthesise an `_ESBMC_aux_ERC20` copy helper for G so
    // that field copy works, and that helper's call must match the callee's
    // formal arity (this + N user params + aux bool). Pad with nondet
    // placeholders via assign_param_nondet so symex's
    // argument_assignments doesn't trip "function call: not enough
    // arguments". The helper's value is only used to copy fields over, so
    // exact argument values don't matter — only the shape does.
    auto &decl_ref = find_constructor_ref(c_name);
    if (!decl_ref.empty() && !decl_ref.is_null())
    {
      if (assign_param_nondet(decl_ref, call))
        log_error("failed to pad nondet args for inherited ctor of {}", c_name);
    }
  }

  call.arguments().push_back(true_exprt());

  call.function() = new_expr;
  call.set("constructor", 1);
  call.type() = symbol_typet(prefix + c_name);
  call.location().file(absolute_path);
  call.location().line(1);
  exprt val;
  get_temporary_object(call, val);
  added_ctor_symbol.set_value(val);
  sym = added_ctor_symbol;

  log_debug("solidity", "finish get_inherit_static_contract_instance");
}

void solidity_convertert::get_contract_mutex_name(
  const std::string c_name,
  std::string &name,
  std::string &id)
{
  name = "$mutex_" + c_name;
  id = "sol:@C@" + c_name + "@" + name + "#";
}

// for reentry check
void solidity_convertert::get_contract_mutex_expr(
  const std::string c_name,
  const exprt &this_expr,
  exprt &expr)
{
  std::string name, id;
  exprt _mutex;
  get_contract_mutex_name(c_name, name, id);
  if (context.find_symbol(id) == nullptr)
  {
    log_error("cannot find auxiliary var {}", id);
    abort();
  }
  _mutex = symbol_expr(*context.find_symbol(id));

  expr = member_exprt(this_expr, _mutex.name(), _mutex.type());
}

bool solidity_convertert::is_sol_builin_symbol(
  const std::string &cname,
  const std::string &name)
{
  std::string tx_name, tx_id;
  get_contract_mutex_name(cname, tx_name, tx_id);
  std::set<std::string> list = {
    "$address", "$codehash", "$balance", "$code", tx_name, "_ESBMC_bind_cname"};
  if (list.count(name) != 0)
    return true;

  return false;
}

/*
  old_sender = msg.sender
  msg.sender = instance.$address
  (_ESBMC_mutext_Base = true)
  (call to payable func)
  (_ESBMC_mutext_Base = false)
  msg.sender = old_sender;
  */
bool solidity_convertert::get_high_level_call_wrapper(
  const std::string cname,
  const exprt &this_expr,
  exprt &front_block,
  exprt &back_block,
  bool is_library)
{
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));

  symbolt old_sender;
  get_default_symbol(
    old_sender,
    debug_modulename,
    addr_t,
    "old_sender",
    "sol:@C@" + cname + "@F@old_sender#" + std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_sender = *move_symbol_to_context(old_sender);
  code_declt old_sender_decl(symbol_expr(added_old_sender));
  added_old_sender.set_value(migrate_expr(msg_sender));
  old_sender_decl.operands().push_back(msg_sender);
  front_block.move_to_operands(old_sender_decl);

  // msg_sender = this.address;
  //
  // Libraries are inlined into the caller's context in real Solidity:
  // when a library body performs an external call, msg.sender for
  // the callee should be the ENCLOSING CONTRACT's address.  We can't
  // reach the enclosing contract's `this` through the library's own
  // `this` (Lib is a dummy singleton with no real $address), so we
  // read it from the global `_ESBMC_enclosing_contract_address`
  // ambient that every contract-method entry sets (see
  // `get_function_definition` in solidity_convert_modifier.cpp).
  // Contracts still use `this_expr->$address` directly — their own
  // `this` IS the caller.
  exprt new_sender_value;
  if (is_library)
  {
    new_sender_value =
      symbol_expr(*context.find_symbol("c:@_ESBMC_enclosing_contract_address"));
  }
  else
  {
    new_sender_value = member_exprt(this_expr, "$address", addr_t);
  }
  exprt assign_sender = side_effect_exprt("assign", addr_t);
  assign_sender.copy_to_operands(msg_sender, new_sender_value);
  convert_expression_to_code(assign_sender);
  front_block.move_to_operands(assign_sender);

  if (is_reentry_check)
  {
    exprt _mutex;
    get_contract_mutex_expr(cname, this_expr, _mutex);

    // _ESBMC_mutex = true;
    typet _t = bool_t;
    exprt _true = true_exprt();
    exprt _false = false_exprt();
    _true.location() = this_expr.location();
    _false.location() = this_expr.location();
    exprt assign_lock = side_effect_exprt("assign", _t);
    assign_lock.copy_to_operands(_mutex, _true);
    assign_lock.location() = this_expr.location();
    convert_expression_to_code(assign_lock);
    front_block.move_to_operands(assign_lock);

    // _ESBMC_mutex = false;
    exprt assign_unlock = side_effect_exprt("assign", _t);
    assign_unlock.copy_to_operands(_mutex, _false);
    assign_unlock.location() = this_expr.location();
    convert_expression_to_code(assign_unlock);
    back_block.move_to_operands(assign_unlock);
  }

  // msg_sender = old_sender;
  exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
  assign_sender_restore.copy_to_operands(
    msg_sender, symbol_expr(added_old_sender));
  convert_expression_to_code(assign_sender_restore);
  back_block.move_to_operands(assign_sender_restore);
  return false;
}

nlohmann::json solidity_convertert::reorder_arguments(
  const nlohmann::json &expr,
  const nlohmann::json &callee_expr_json)
{
  // build a map from name to argument
  std::unordered_map<std::string, nlohmann::json> name_to_arg;
  for (size_t i = 0; i < expr["names"].size(); ++i)
    name_to_arg[expr["names"][i]] = expr["arguments"][i];

  std::vector<std::string> param_order;
  const auto &decl_ref =
    find_decl_ref(callee_expr_json["referencedDeclaration"]);
  // check if the function has parameters and store the order
  if (
    decl_ref.contains("parameters") &&
    decl_ref["parameters"].contains("parameters"))
  {
    for (const auto &param : decl_ref["parameters"]["parameters"])
    {
      if (param.contains("name"))
        param_order.push_back(param["name"]);
    }
  }
  // reorder the arguments based on the parameter order
  nlohmann::json ordered_args = nlohmann::json::array();
  for (const auto &param : param_order)
  {
    ordered_args.push_back(name_to_arg[param]);
  }
  // use the reordered arguments
  nlohmann::json clean_expr = expr;
  clean_expr["arguments"] = ordered_args;
  clean_expr.erase("names");
  return clean_expr;
}

// Emit `_sol_per_tx_reseed()` at the top of a dispatcher while-loop
// iteration. Each iter == one EVM transaction with its own
// msg.sender / msg.value / tx.* / block.*. Always fires — frozen-
// sender behaviour was itself unsound (made inverted access-control
// bugs unreachable by construction), so there is no opt-out.
// The C symbol comes from src/c2goto/library/solidity/solidity_misc.c.
long solidity_convertert::get_tx_bound()
{
  // Coverage and TOD modes manage their own exploration (k-step ladders,
  // transaction-ordering enumeration) and need the full unbounded harness:
  // a fixed tx bound distorts coverage counts and hides ordering races.
  // --solidity-precise likewise opts back into the unbounded (sound) harness.
  static const char *const unbounded_modes[] = {
    "solidity-precise",
    "tod-race-check",
    "tod-balance-check",
    "assertion-coverage",
    "assertion-coverage-claims",
    "branch-coverage",
    "branch-coverage-claims",
    "branch-function-coverage",
    "branch-function-coverage-claims",
    "condition-coverage",
    "condition-coverage-claims",
    "condition-coverage-rm",
    "condition-coverage-claims-rm",
    "k-path-coverage",
    "k-path-coverage-claims"};

  const std::string max_tx_opt = config.options.get_option("solidity-max-tx");
  bool mode_forces_unbounded = false;
  for (const char *opt : unbounded_modes)
    if (!config.options.get_option(opt).empty())
    {
      mode_forces_unbounded = true;
      break;
    }

  long bound;
  if (!max_tx_opt.empty())
    bound = std::stol(max_tx_opt); // explicit user choice always wins
  else if (mode_forces_unbounded)
    bound = 0; // unbounded
  else
    bound = 2; // bounded-by-default

  if (bound > 0 && !tx_bound_warned)
  {
    log_warning(
      "Solidity harness: transaction sequence bounded to {} tx (default). A "
      "VERIFICATION SUCCESSFUL result is bounded -- it means no violation "
      "within {} transaction(s), NOT an unbounded proof; bugs requiring more "
      "transactions are not explored. Use --solidity-precise (or "
      "--solidity-max-tx 0) for an unbounded proof.",
      bound,
      bound);
    tx_bound_warned = true;
  }
  return bound;
}

void solidity_convertert::emit_tx_driver(codet &func_body, const codet &tx_body)
{
  const long bound = get_tx_bound();
  if (bound <= 0)
  {
    // unbounded: while(nondet_bool) { tx_body }
    side_effect_expr_function_callt cond_expr = nondet_bool_expr;
    code_whilet code_while;
    code_while.cond() = cond_expr;
    code_while.body() = tx_body;
    func_body.move_to_operands(code_while);
  }
  else
    // bounded: deterministically emit the transaction body `bound` times.
    for (long i = 0; i < bound; ++i)
      func_body.copy_to_operands(tx_body);
}

void solidity_convertert::emit_per_tx_reseed_call(codet &out)
{
  const symbolt *reseed_sym = context.find_symbol("c:@F@_sol_per_tx_reseed");
  if (reseed_sym == nullptr)
  {
    log_warning(
      "_sol_per_tx_reseed not found in symbol table; per-tx ambient "
      "reseed disabled. The Solidity C library may not be linked.");
    return;
  }
  code_function_callt reseed_call;
  reseed_call.location() = reseed_sym->location;
  reseed_call.function() = symbol_expr(*reseed_sym);
  out.move_to_operands(reseed_call);
}

/*
  perform multi-transaction verification
  the idea is to verify the assertions that must be held
  in any function calling order.
  convert the verifying contract to a "sol_main" function, e.g.

  Contract Base
  {
      constrcutor(){}
      function A(){}
      function B(){}
  }

  will be converted to

  void sol_main()
  {
    Base()  // constructor_call
    while(nondet_bool)
    {
      if(nondet_bool) A();
      if(nondet_bool) B();
    }
  }

  Additionally, we need to handle the inheritance. Theoretically, we need to
  merge (i.e. create a copy) the public and internal state variables and
  functions inside Base contracts into the Derive contract. However, in practice
  we do not need to do so. Instead, we
    - call the constructors based on the linearizedBaseList
    - add the inherited public function call to the if-body

*/
bool solidity_convertert::multi_transaction_verification(
  const std::string &c_name)
{
  log_debug(
    "Solidity", "@@@ performs transaction verification on contract {}", c_name);

  // 0. initialize "sol_main" body and per-transaction body
  codet func_body, tx_body;
  static_lifetime_init(context, func_body);
  func_body.make_block();
  static_lifetime_init(context, tx_body);
  tx_body.make_block();

  // add __ESBMC_HIDE
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  // initialize

  // 1. get constructor call
  if (
    std::find(contractNamesList.begin(), contractNamesList.end(), c_name) ==
    contractNamesList.end())
  {
    log_error("Cannot find the contract name {}", c_name);
    return true;
  }

  // 1b. --path-cov-fixture: the recorded deployment state.
  //
  // Emitted into func_body, i.e. OUTSIDE the transaction driver and before it,
  // which is exactly where the constructor call used to run. Placing it inside
  // tx_body would re-apply it at the head of every transaction and silently
  // undo whatever the previous transaction wrote -- the ladder's second
  // transaction would then never see the first one's effect, which is the one
  // thing a multi-transaction harness exists for.
  {
    const path_cov_fixturet &fx = path_cov_fixture();
    if (fx.present && fx.contract == c_name && !fx.state.empty())
    {
      const symbolt *c_sym = context.find_symbol(prefix + c_name);
      if (c_sym == nullptr)
      {
        log_error(
          "--path-cov-fixture: contract {} has no struct symbol {}",
          c_name,
          prefix + c_name);
        abort();
      }
      const struct_typet &c_type = to_struct_type(c_sym->get_type());

      exprt instance;
      get_static_contract_instance_ref(c_name, instance);

      for (const auto &kv : fx.state)
      {
        // Iterate the EXPECTED name and hard-fail when the contract does not
        // have it, printing every component it does have. A fixture entry
        // silently skipped is a state variable the report believes was pinned
        // and which is in fact whatever the default initialiser left.
        const typet *comp_type = nullptr;
        for (const auto &comp : c_type.components())
          if (comp.name() == kv.first)
          {
            comp_type = &comp.type();
            break;
          }
        if (comp_type == nullptr)
        {
          std::string have;
          for (const auto &comp : c_type.components())
            have += (have.empty() ? "" : ", ") + id2string(comp.name());
          log_error(
            "--path-cov-fixture: contract {} has no state variable '{}'. It "
            "has: {}",
            c_name,
            kv.first,
            have);
          abort();
        }

        const unsigned w = bv_width(*comp_type);
        if (w == 0)
        {
          log_error(
            "--path-cov-fixture: state variable '{}' of contract {} is not a "
            "scalar this stage can assign (its type has no bit width). Only "
            "integer / address / bool state is supported",
            kv.first,
            c_name);
          abort();
        }

        exprt value = constant_exprt(
          integer2binary(kv.second, w), integer2string(kv.second), *comp_type);
        exprt lhs = member_exprt(instance, kv.first, *comp_type);
        exprt assign = side_effect_exprt("assign", *comp_type);
        assign.copy_to_operands(lhs, value);
        convert_expression_to_code(assign);
        func_body.move_to_operands(assign);

        log_status(
          "--path-cov-fixture: {}.{} = {} (assigned before the transaction "
          "driver, in place of the deployment)",
          c_name,
          kv.first,
          integer2string(kv.second));
      }
    }
  }

  // Per-tx ambient reseed: each dispatcher iter is a fresh EVM tx
  // with its own msg.sender / msg.value / tx.* / block.* context.
  // The constructor's `owner = msg.sender` already ran outside the
  // while-loop (deployer-binding), so reseeding here properly
  // distinguishes deployer identity from per-iter sender identities.
  // Without this, contracts that compare msg.sender to a stored
  // owner-like address see a frozen invariant `msg.sender == owner`
  // and access-control bugs become unreachable (false negatives).
  emit_per_tx_reseed_call(tx_body);

  // get sol harness function and move into the per-transaction body
  code_function_callt func_call;
  if (get_unbound_funccall(c_name, func_call))
    return true;

  tx_body.move_to_operands(func_call);

  // Transaction-sequence driver (see --solidity-max-tx / --solidity-precise).
  // Default: deterministically unroll to N=2 transactions so k-induction/BMC
  // converges on the otherwise-unbounded harness loop; --solidity-max-tx 0 or
  // --solidity-precise restores the unbounded while(nondet_bool) loop. A
  // bounded harness is an under-approximation: SUCCESSFUL means "no violation
  // within N tx", NOT an unbounded proof; FAILED stays sound. Each unrolled tx
  // re-calls _sol_per_tx_reseed, so msg.sender/value/block.* are freshly
  // havoc'd per transaction.
  emit_tx_driver(func_body, tx_body);

  // construct _ESBMC_Main_Base
  symbolt new_symbol;
  code_typet main_type;
  const std::string main_name = "_ESBMC_Main_" + c_name;
  const std::string main_id = "sol:@C@" + c_name + "@F@" + main_name + "#";
  typet e_type = empty_typet();
  e_type.set("cpp_type", "void");
  main_type.return_type() = e_type;
  const symbolt &_contract = *context.find_symbol(prefix + c_name);
  new_symbol.location = _contract.location;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(
    new_symbol,
    debug_modulename,
    main_type,
    main_name,
    main_id,
    _contract.location);

  new_symbol.lvalue = true;
  new_symbol.is_extern = false;
  new_symbol.file_local = false;

  symbolt &main_sym = *context.move_symbol_to_context(new_symbol);

  // no params
  main_type.make_ellipsis();

  main_sym.set_type(migrate_type(main_type));
  main_sym.set_value(func_body);

  // set "_ESBMC_Main_X" as the main function
  // this will be overwrite in multi-contract mode.
  config.main = main_name;

  return false;
}

/*
  This function perform multi-transaction verification on each contract in
  isolation. To do so, we construct nondetered switch_case;
*/
// Shared: call multi_transaction_verification for each contract and collect
// the resulting _ESBMC_Main_X entry function symbols.
bool solidity_convertert::prepare_harness_entry_functions(
  const std::set<std::string> &cname_set,
  std::vector<const symbolt *> &entry_syms)
{
  for (const auto &c_name : cname_set)
  {
    if (multi_transaction_verification(c_name))
    {
      log_error(
        "Failed to construct multi-transaction verification for contract {}",
        c_name);
      return true;
    }

    const std::string sub_sol_id =
      "sol:@C@" + c_name + "@F@_ESBMC_Main_" + c_name + "#";
    if (context.find_symbol(sub_sol_id) == nullptr)
    {
      log_error("Cannot find the {}'s main function {}", c_name, sub_sol_id);
      return true;
    }
    entry_syms.push_back(context.find_symbol(sub_sol_id));
  }
  return false;
}

// Shared: create the _ESBMC_Main symbol and register it in the context.
bool solidity_convertert::register_harness_main(
  const std::string &sol_id,
  const codet &func_body)
{
  const std::string sol_name = "_ESBMC_Main";
  if (context.find_symbol(prefix + *contractNamesList.begin()) == nullptr)
  {
    log_error("Cannot find the main function");
    return true;
  }

  const symbolt &contract =
    *context.find_symbol(prefix + *contractNamesList.begin());

  symbolt new_symbol;
  code_typet main_type;
  typet e_type = empty_typet();
  e_type.set("cpp_type", "void");
  main_type.return_type() = e_type;

  new_symbol.location = contract.location;
  std::string debug_modulename =
    get_modulename_from_path(contract.location.file().as_string());
  get_default_symbol(
    new_symbol,
    debug_modulename,
    main_type,
    sol_name,
    sol_id,
    new_symbol.location);

  new_symbol.lvalue = true;
  new_symbol.is_extern = false;
  new_symbol.file_local = false;

  symbolt &added_symbol = *context.move_symbol_to_context(new_symbol);
  main_type.make_ellipsis();
  added_symbol.set_type(migrate_type(main_type));
  added_symbol.set_value(func_body);
  config.main = sol_name;
  return false;
}

bool solidity_convertert::multi_contract_verification_bound(
  std::set<std::string> &tgt_set)
{
  log_debug("solidity", "multi_contract_verification_bound");

  codet func_body;
  static_lifetime_init(context, func_body);
  func_body.make_block();

  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  std::set<std::string> cname_set;
  if (!tgt_set.empty())
    cname_set = tgt_set;
  else
    cname_set =
      std::set<std::string>(contractNamesList.begin(), contractNamesList.end());

  // Build entry functions for each contract
  std::vector<const symbolt *> entry_syms;
  if (prepare_harness_entry_functions(cname_set, entry_syms))
    return true;

  // Assemble switch(nondet_uint()) { case 0: Main_A(); ... }
  codet switch_body;
  static_lifetime_init(context, switch_body);
  switch_body.make_block();

  int cnt = 0;
  for (const auto *sym : entry_syms)
  {
    exprt case_cond = constant_exprt(
      integer2binary(cnt, bv_width(int_type())),
      integer2string(cnt),
      int_type());

    codet case_body;
    static_lifetime_init(context, case_body);
    case_body.make_block();

    code_function_callt func_expr;
    func_expr.location() = sym->location;
    func_expr.function() = symbol_expr(*sym);
    case_body.move_to_operands(func_expr);
    exprt break_expr = code_breakt();
    case_body.move_to_operands(break_expr);

    code_switch_caset switch_case;
    switch_case.case_op() = case_cond;
    convert_expression_to_code(case_body);
    switch_case.code() = to_code(case_body);
    switch_body.move_to_operands(switch_case);
    ++cnt;
  }

  code_switcht code_switch;
  code_switch.value() = nondet_uint_expr;
  code_switch.body() = switch_body;
  func_body.move_to_operands(code_switch);

  std::string sol_id;
  if (!tgt_set.empty())
    sol_id = "sol:@C@" + *tgt_set.begin() + "@F@_ESBMC_Main#";
  else
    sol_id = "sol:@F@_ESBMC_Main#";

  return register_harness_main(sol_id, func_body);
}

// For unbound, we verify each contract sequentially in isolation
bool solidity_convertert::multi_contract_verification_unbound(
  std::set<std::string> &tgt_set)
{
  log_debug("solidity", "multi_contract_verification_unbound");

  codet func_body;
  static_lifetime_init(context, func_body);
  func_body.make_block();

  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  std::set<std::string> cname_set;
  if (!tgt_set.empty())
    cname_set = tgt_set;
  else
    cname_set =
      std::set<std::string>(contractNamesList.begin(), contractNamesList.end());

  // Build entry functions for each contract
  std::vector<const symbolt *> entry_syms;
  if (prepare_harness_entry_functions(cname_set, entry_syms))
    return true;

  // Call each entry function sequentially
  for (const auto *sym : entry_syms)
  {
    code_function_callt func_expr;
    func_expr.location() = sym->location;
    func_expr.function() = symbol_expr(*sym);
    func_body.move_to_operands(func_expr);
  }

  return register_harness_main("sol:@F@_ESBMC_Main#", func_body);
}
