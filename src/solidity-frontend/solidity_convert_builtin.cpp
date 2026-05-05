/// \file solidity_convert_builtin.cpp
/// \brief Built-in function and low-level call handling for the Solidity frontend.
///
/// Implements recognition and conversion of Solidity built-in operations:
/// low-level calls (call, delegatecall, staticcall, transfer, send),
/// built-in properties (msg.sender, msg.value, block.number, address.balance),
/// type conversion functions, abi.encode/decode, keccak256/sha256, and the
/// move_builtin_to_contract() helper for contract-scoped symbol registration.

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

  // value
  side_effect_expr_function_callt _ndt_uint = nondet_uint_expr;

  // _ESBMC_get_unique_address(this, cname)
  side_effect_expr_function_callt _addr;
  locationt l;
  l.function(contract_name);

  typet t = addr_t;

  get_library_function_call_no_args(
    "_ESBMC_get_unique_address", "c:@F@_ESBMC_get_unique_address", t, l, _addr);

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
  {
    exprt balance_init = _ndt_uint;
    if (json.contains("nodes"))
    {
      for (const auto &node : json["nodes"])
      {
        if (
          node["nodeType"] == "FunctionDefinition" && node.contains("kind") &&
          node["kind"] == "constructor" && node.contains("stateMutability") &&
          node["stateMutability"] == "payable")
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
  // for each contract, we add a static infinity array {$cname}_pool
  // e.g. __attribute__((annotate("__ESBMC_inf_size"))) unsigned char base_pool[1];
  // then we add symbol call $dynamic_pool
  // e.g. BytesPool pool = bytes_pool_init(base_pool);
  // 1. declare static base pool

  // however, this will affect the performance,
  // so first check if we need to add this
  if (has_contract_bytes(json))
  {
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
  }

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
  //set_sol_type(t, SolidityGrammar::SolType::STRING);
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
  assert(c_sym.type.is_struct());

  if (!is_method)
  {
    // check if it's already inserted
    for (auto i : to_struct_type(c_sym.type).components())
    {
      if (i.identifier() == sym.identifier())
        return;
    }

    struct_typet::componentt comp(sym.name(), sym.name(), sym.type());
    comp.set_access(access);
    comp.type().set("#member_name", c_sym.type.tag());
    to_struct_type(c_sym.type).components().push_back(comp);
  }
  else
  {
    // check if it's already inserted
    for (auto i : to_struct_type(c_sym.type).methods())
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
    to_struct_type(c_sym.type).methods().push_back(comp);
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
  sym.type.set("#sol_state_var", "1");
  sym.file_local = true;
  sym.lvalue = true;
  auto &added_sym = *move_symbol_to_context(sym);
  code_declt decl(symbol_expr(added_sym));
  added_sym.value = val;
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
  added_symbol.type = type;
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

  // return nondet_uint
  code_returnt ret_uint;
  ret_uint.return_value() = nondet_uint_expr;
  _block.move_to_operands(ret_uint);

  // populate body
  added_symbol.value = _block;

  // do function call
  side_effect_expr_function_callt _call;
  _call.function() = symbol_expr(added_symbol);
  _call.location() = loc;
  _call.arguments().push_back(cur_this_expr);
  _call.arguments().push_back(base);
  new_expr = _call;
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

  exprt mem;
  if (
    base.is_member() &&
    (base.op0().name() == "this" ||
     get_sol_type(base.op0().type()) == SolidityGrammar::SolType::CONTRACT))
    // e.g. address(_ins_).balance => _ins_.balance
    //      address(this) => this->address
    //TODO: fixme! this pattern match is weak
    mem = member_exprt(base.op0(), comp_name, t);
  else
  {
    // e.g. address(msg.sender).balance
    // we do not know what instance is msg.sender pointed to, so over-approximate
    get_aux_property_function(cname, base, t, loc, name, mem);
  }

  mem.location() = loc;
  new_expr = mem;
}
