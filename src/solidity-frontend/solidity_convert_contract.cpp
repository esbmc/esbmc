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
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_expr.h>
#include <util/message.h>
#include <fstream>

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

void solidity_convertert::add_static_contract_instance(const std::string c_name)
{
  log_debug("solidity", "\tAdd static instance of contract {}", c_name);

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
    // this will lead to issue in the following process, particuarly this_pointer
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

  added_sym.value = ctor;
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
  ft.set("#member_name", prefix + c_name);
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
  add_sym.type = ft;

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

    if (
      (*current_functionDecl).contains("body") ||
      ((*current_functionDecl).contains("implemented") &&
       (*current_functionDecl)["implemented"] == true))
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

  add_sym.value = func_body;

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

  if (args_list.contains("arguments"))
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

  call.arguments().push_back(true_exprt());

  call.function() = new_expr;
  call.set("constructor", 1);
  call.type() = symbol_typet(prefix + c_name);
  call.location().file(absolute_path);
  call.location().line(1);
  exprt val;
  get_temporary_object(call, val);
  added_ctor_symbol.value = val;
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
  exprt &back_block)
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
  added_old_sender.value = msg_sender;
  old_sender_decl.operands().push_back(msg_sender);
  front_block.move_to_operands(old_sender_decl);

  // msg_sender = this.address;
  exprt this_address = member_exprt(this_expr, "$address", addr_t);
  exprt assign_sender = side_effect_exprt("assign", addr_t);
  assign_sender.copy_to_operands(msg_sender, this_address);
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
  const nlohmann::json &src_ast_json,
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

  Additionally, we need to handle the inheritance. Theoretically, we need to merge (i.e. create a copy) the public and internal state variables and functions inside Base contracts into the Derive contract. However, in practice we do not need to do so. Instead, we 
    - call the constructors based on the linearizedBaseList 
    - add the inherited public function call to the if-body 

*/
bool solidity_convertert::multi_transaction_verification(
  const std::string &c_name)
{
  log_debug(
    "Solidity", "@@@ performs transaction verification on contract {}", c_name);

  // 0. initialize "sol_main" body and while-loop body
  codet func_body, while_body;
  static_lifetime_init(context, func_body);
  func_body.make_block();
  static_lifetime_init(context, while_body);
  while_body.make_block();

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

  // get sol harness function and move into the while body
  code_function_callt func_call;
  if (get_unbound_funccall(c_name, func_call))
    return true;
  while_body.move_to_operands(func_call);

  // while-cond:
  side_effect_expr_function_callt cond_expr = nondet_bool_expr;

  // while-loop statement:
  code_whilet code_while;
  code_while.cond() = cond_expr;
  code_while.body() = while_body;
  func_body.move_to_operands(code_while);

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

  main_sym.type = main_type;
  main_sym.value = func_body;

  // set "_ESBMC_Main_X" as the main function
  // this will be overwrite in multi-contract mode.
  config.main = main_name;

  return false;
}

/*
  This function perform multi-transaction verification on each contract in isolation.
  To do so, we construct nondetered switch_case;
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
  added_symbol.type = main_type;
  added_symbol.value = func_body;
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
