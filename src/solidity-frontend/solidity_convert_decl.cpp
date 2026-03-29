#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/solidity_template.h>
#include <solidity-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_expr.h>
#include <util/message.h>
#include <regex>
#include <optional>

#include <fstream>
#include <iostream>

bool solidity_convertert::get_non_function_decl(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  new_expr = code_skipt();

  if (!ast_node.contains("nodeType"))
  {
    log_error("Missing \'nodeType\' filed in ast_node");
    abort();
  }

  SolidityGrammar::ContractBodyElementT type =
    SolidityGrammar::get_contract_body_element_t(ast_node);

  log_debug(
    "solidity",
    "\t@@@ Expecting non-function definition, Got {}",
    SolidityGrammar::contract_body_element_to_str(type));

  // based on each element as in Solidty grammar "rule contract-body-element"
  switch (type)
  {
  case SolidityGrammar::ContractBodyElementT::VarDecl:
  {
    return get_var_decl(ast_node, new_expr); // rule state-variable-declaration
  }
  case SolidityGrammar::ContractBodyElementT::StructDef:
  {
    return get_struct_class(ast_node); // rule enum-definition
  }
  case SolidityGrammar::ContractBodyElementT::ModifierDef:
  case SolidityGrammar::ContractBodyElementT::FunctionDef:
  case SolidityGrammar::ContractBodyElementT::EnumDef:
  case SolidityGrammar::ContractBodyElementT::ErrorDef:
  case SolidityGrammar::ContractBodyElementT::EventDef:
  case SolidityGrammar::ContractBodyElementT::UsingForDef:
  {
    break;
  }
  default:
  {
    log_error("Unimplemented type in rule contract-body-element");
    return true;
  }
  }
  return false;
}

bool solidity_convertert::get_function_decl(const nlohmann::json &ast_node)
{
  if (!ast_node.contains("nodeType"))
  {
    log_error("Missing \'nodeType\' filed in ast_node");
    abort();
  }

  SolidityGrammar::ContractBodyElementT type =
    SolidityGrammar::get_contract_body_element_t(ast_node);

  log_debug(
    "solidity",
    "\t@@@ Expecting function definition, Got {}",
    SolidityGrammar::contract_body_element_to_str(type));

  // based on each element as in Solidty grammar "rule contract-body-element"
  switch (type)
  {
  case SolidityGrammar::ContractBodyElementT::FunctionDef:
  {
    return get_function_definition(ast_node); // rule function-definition
  }
  case SolidityGrammar::ContractBodyElementT::VarDecl:
  case SolidityGrammar::ContractBodyElementT::StructDef:
  case SolidityGrammar::ContractBodyElementT::EnumDef:
  case SolidityGrammar::ContractBodyElementT::ErrorDef:
  case SolidityGrammar::ContractBodyElementT::EventDef:
  case SolidityGrammar::ContractBodyElementT::UsingForDef:
  case SolidityGrammar::ContractBodyElementT::ModifierDef:
  {
    break;
  }
  default:
  {
    log_error("Unimplemented type in rule contract-body-element");
    return true;
  }
  }
  return false;
}

// push back a this pointer to the type
void solidity_convertert::get_function_this_pointer_param(
  const std::string &contract_name,
  const std::string &func_id,
  const std::string &debug_modulename,
  const locationt &location_begin,
  code_typet &type)
{
  log_debug("solidity", "\t@@@ getting function this pointer param");
  assert(!contract_name.empty());
  code_typet::argumentt this_param;
  std::string this_name = "this";
  //? do we need to drop the '#n' tail in func_id?
  std::string this_id = func_id + "#" + this_name;

  this_param.cmt_base_name(this_name);
  this_param.cmt_identifier(this_id);

  this_param.type() = gen_pointer_type(symbol_typet(prefix + contract_name));
  symbolt param_symbol;
  get_default_symbol(
    param_symbol,
    debug_modulename,
    this_param.type(),
    this_name,
    this_id,
    location_begin);
  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;

  if (context.find_symbol(this_id) == nullptr)
  {
    context.move_symbol_to_context(param_symbol);
  }

  type.arguments().push_back(this_param);
}

bool solidity_convertert::get_var_decl(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  return get_var_decl(ast_node, empty_json, new_expr);
}

// rule state-variable-declaration
// rule variable-declaration-statement
// @initialValue: for declaration block
bool solidity_convertert::get_var_decl(
  const nlohmann::json &ast_node,
  const nlohmann::json &initialValue,
  exprt &new_expr)
{
  if (ast_node.is_null() || ast_node.empty())
  {
    new_expr = nil_exprt();
    return false;
  }

  std::string current_contractName;
  get_current_contract_name(ast_node, current_contractName);
  bool is_library = !current_contractName.empty() &&
                    std::find(
                      contractNamesList.begin(),
                      contractNamesList.end(),
                      current_contractName) == contractNamesList.end();

  // For Solidity rule state-variable-declaration:
  // 1. populate typet
  typet t;
  // VariableDeclaration node contains both "typeName" and "typeDescriptions".
  // However, ExpressionStatement node just contains "typeDescriptions".
  // For consistensy, we use ["typeName"]["typeDescriptions"] as in state-variable-declaration
  // to improve the re-usability of get_type* function, when dealing with non-array var decls.
  // For array, do NOT use ["typeName"]. Otherwise, it will cause problem
  // when populating typet in get_cast

  if (get_type_description(
        ast_node, ast_node["typeName"]["typeDescriptions"], t))
    return true;

  bool is_contract =
    t.get("#sol_type").as_string() == "CONTRACT" ? true : false;
  bool is_mapping = t.get("#sol_type").as_string() == "MAPPING" ? true : false;
  bool is_new_expr = newContractSet.count(current_contractName);
  bool is_byte_static = is_bytesN_type(t);
  if (is_new_expr)
  {
    // hack: check if it's unbound and the only verifying targets
    if (
      !is_bound && tgt_cnt_set.count(current_contractName) > 0 &&
      tgt_cnt_set.size() == 1)
      is_new_expr = false;
  }

  // for mapping: populate the element type
  if (is_mapping && !is_new_expr)
  {
    assert(t.is_array());
    const auto &val_type = ast_node["typeName"]["valueType"];
    typet val_t;
    if (get_type_description(val_type["typeDescriptions"], val_t))
      return true;
    t.subtype() = val_t;
  }

  // set const qualifier
  bool is_constant =
    ast_node.contains("mutability") && ast_node["mutability"] == "constant";
  if (is_constant)
    t.cmt_constant(true);

  // record the state info
  // this will be used to decide if the var will be converted to this->var
  // when parsing function body.
  bool is_state_var = ast_node["stateVariable"].get<bool>();
  t.set("#sol_state_var", std::to_string(is_state_var));

  bool is_inherited = ast_node.contains("is_inherited");

  // 2. populate id and name
  std::string name, id;
  //TODO: Omitted variable
  if (ast_node["name"].get<std::string>().empty())
    // Omitted variable
    get_aux_var(name, id);
  else
  {
    if (get_var_decl_name(ast_node, name, id))
      return true;
  }

  // if we have already populated the var symbol, we do not need to re-parse
  // however, we need to return the symbol info
  if (context.find_symbol(id) != nullptr)
  {
    log_debug("solidity", "Found parsed symbol, skip parsing");
    new_expr = symbol_expr(*context.find_symbol(id));
    return false;
  }

  // 3. populate location
  locationt location_begin;
  get_location_from_node(ast_node, location_begin);

  // 4. populate debug module name
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  // 5. set symbol attributes
  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, t, name, id, location_begin);

  symbol.lvalue = true;
  // static_lifetime: this means it's defined in the file level, not inside contract
  // special case for mapping, even if it's inside a contract
  symbol.static_lifetime = current_contractName.empty() ||
                           (is_mapping && !is_new_expr) ||
                           (is_library && is_constant);
  symbol.file_local = true;
  symbol.is_extern = false;

  // For state var decl, we look for "value".
  // For local var decl, we look for "initialValue"
  bool has_init = (ast_node.contains("value") || !initialValue.empty());
  // For inherited ones, the initial value will be set in "move_inheritance_to_ctor()"
  // e.g. D.x = B.x
  // Therefore, even if the copied json nodes contain init_value (has_init = true), we still skip such settings.
  bool set_init = has_init && !is_inherited;
  const nlohmann::json init_value =
    ast_node.contains("value") ? ast_node["value"] : initialValue;
  const nlohmann::json literal_type = ast_node["typeDescriptions"];
  if (!set_init && !(is_mapping && is_new_expr && is_byte_static))
  {
    // for both state and non-state variables, set default value as zero
    symbol.value = gen_zero(get_complete_type(t, ns), true);
    symbol.value.zero_initializer(true);
  }

  // 6. add symbol into the context
  // just like clang-c-frontend, we have to add the symbol before converting the initial assignment
  symbolt &added_symbol = *move_symbol_to_context(symbol);
  code_declt decl(symbol_expr(added_symbol));

  // 7. populate init value if there is any
  // special handling for array/dynarray
  std::string t_sol_type = t.get("#sol_type").as_string();

  // this pointer
  exprt this_expr;
  if (!current_contractName.empty())
  {
    if (current_functionDecl)
    {
      if (get_func_decl_this_ref(*current_functionDecl, this_expr))
        return true;
    }
    else
    {
      if (get_ctor_decl_this_ref(ast_node, this_expr))
        return true;
    }
  }

  exprt val;
  if (t_sol_type == "ARRAY" || t_sol_type == "ARRAY_LITERAL")
  {
    /** 
      uint[2] z;            // uint *z = (uint *)calloc(2, sizeof(uint));
      
                            // uint tmp1[2] = {1,2}; // populated into sym tab, not a real statement
      uint[2] zz = [1,2];   // uint *zz = (uint *)_ESBMC_arrcpy(tmp1, 2, 2, sizeof(uint));

      uint[2] y = x;        // uint *zz = (uint *)_ESBMC_arrcpy(x, 2, 2, sizeof(uint));

      TODO: suport disorder:
      uint[2] y = x;
      uint[2] x = [1,2];
    **/

    // get size
    std::string arr_size = "0";
    if (!t.get("#sol_array_size").empty())
      arr_size = t.get("#sol_array_size").as_string();
    else if (t.has_subtype() && !t.subtype().get("#sol_array_size").empty())
      arr_size = t.subtype().get("#sol_array_size").as_string();
    else
    {
      log_error("cannot get the size of fixed array");
      return true;
    }
    exprt size_expr = constant_exprt(
      integer2binary(string2integer(arr_size), bv_width(uint_type())),
      arr_size,
      uint_type());

    // get sizeof
    exprt size_of_expr;
    get_size_of_expr(t.subtype(), size_of_expr);

    if (set_init)
    {
      if (get_init_expr(init_value, literal_type, t, val))
        return true;

      side_effect_expr_function_callt acpy_call;
      get_arrcpy_function_call(location_begin, acpy_call);
      acpy_call.arguments().push_back(val);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      // typecast
      solidity_gen_typecast(ns, acpy_call, t);
      acpy_call.type().set("#sol_array_size", arr_size);
      // set as rvalue
      added_symbol.value = acpy_call;
      decl.operands().push_back(acpy_call);
    }
    else
    {
      // do calloc
      side_effect_expr_function_callt calc_call;
      get_calloc_function_call(location_begin, calc_call);
      calc_call.arguments().push_back(size_expr);
      calc_call.arguments().push_back(size_of_expr);
      // typecast
      solidity_gen_typecast(ns, calc_call, t);
      // set as rvalue
      added_symbol.value = calc_call;
      decl.operands().push_back(calc_call);
    }
  }
  else if (t_sol_type == "DYNARRAY" && set_init)
  {
    // Note for inherited dynamic array, they will be registered in
    // D.dyn_arr = _ESBMC_arrcpy(B.dyn_ar)
    exprt val;
    if (get_init_expr(init_value, literal_type, t, val))
      return true;

    if (val.is_typecast() || val.type().get("#sol_type") == "ARRAY_CALLOC")
    {
      // uint[] zz = new uint(10);
      // uint[] zz = new uint(len);
      //=> uint* zz = (uint *)calloc(10, sizeof(uint));
      //=> uint* zz = (uint *)calloc(len, sizeof(uint));
      solidity_gen_typecast(ns, val, t);
      added_symbol.value = val;
      decl.operands().push_back(val);

      // get rhs size, e.g. 10
      nlohmann::json callee_arg_json = init_value["arguments"][0];
      exprt size_expr;
      const nlohmann::json literal_type = callee_arg_json["typeDescriptions"];
      if (get_expr(callee_arg_json, literal_type, size_expr))
        return true;

      // construct statement _ESBMC_store_array(zz, 10);
      exprt func_call;
      if (is_state_var)
        store_update_dyn_array(
          member_exprt(this_expr, added_symbol.name, added_symbol.type),
          size_expr,
          func_call);
      else
        store_update_dyn_array(symbol_expr(added_symbol), size_expr, func_call);
      move_to_back_block(func_call);
    }
    else if (val.is_symbol())
    {
      /** 
      uint[] zzz;           // uint* zzz; // will not reach here actually
                            // 
      uint[] zzzz = [1,2];  // memcpy(zzzz, tmp2, 2*sizeof(uint));
                            // uint* zzzzz = 0;
      uint[] zzzzzz = zzz;  // memcpy(zzzzzz, zzz, zzz.size * sizeof(uint));

      Theoretically we can convert it to something like int *z = new int[2]{0,1};
      However, this feature seems to be not fully supported in current esbmc-cpp (v7.6.1)
    */
      // get size
      exprt size_expr;
      get_size_expr(val, size_expr);

      // get sizeof
      exprt size_of_expr;
      get_size_of_expr(t.subtype(), size_of_expr);

      side_effect_expr_function_callt acpy_call;
      // we will store the array inside the copy function
      get_arrcpy_function_call(location_begin, acpy_call);
      acpy_call.arguments().push_back(val);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      // typecast
      solidity_gen_typecast(ns, acpy_call, t);
      // set as rvalue
      added_symbol.value = acpy_call;
      decl.operands().push_back(acpy_call);

      // store length
      exprt func_call;
      if (is_state_var)
        store_update_dyn_array(
          member_exprt(this_expr, added_symbol.name, added_symbol.type),
          size_expr,
          func_call);
      else
        store_update_dyn_array(symbol_expr(added_symbol), size_expr, func_call);
      move_to_back_block(func_call);
    }
    else
    {
      log_error("Unexpect initialization for dynamic array");
      log_debug("solidity", "{}", val);
      return true;
    }
  }
  // special handling for mapping
  else if (is_mapping && is_new_expr)
  {
    // mapping(string => uint) test;
    // 1. the contract that contains this mapping is also used in a new expression
    // => __attribute__((annotate("__ESBMC_inf_size"))) struct _ESBMC_Mapping _ESBMC_inf_test[];
    // => struct mapping_t test = {_ESBMC_inf_test, this.address};
    // 2.
    // => struct mapping_t_fast test = {_ESBMC_inf_test};
    // 1. construct static infinite array
    std::string arr_name, arr_id;
    get_mapping_inf_arr_name(current_contractName, name, arr_name, arr_id);
    symbolt arr_s;
    std::string mapping_struct_name = "_ESBMC_Mapping";

    if (context.find_symbol(prefix + mapping_struct_name) == nullptr)
    {
      log_error("failed to find _ESBMC_Mapping reference");
      return true;
    }

    typet arr_t = array_typet(
      symbol_typet(prefix + mapping_struct_name), exprt("infinity"));
    get_default_symbol(
      arr_s, debug_modulename, arr_t, arr_name, arr_id, location_begin);
    arr_s.static_lifetime = true;
    arr_s.file_local = true;
    arr_s.lvalue = true;
    auto &add_added_s = *move_symbol_to_context(arr_s);
    add_added_s.value = gen_zero(get_complete_type(arr_t, ns), true);

    // 2. construct mapping_t struct instance's value
    typet map_t;
    map_t = context.find_symbol(prefix + "mapping_t")->type;

    assert(map_t.is_struct());
    exprt inits = gen_zero(map_t);

    exprt op0 = symbol_expr(add_added_s);
    // array => &array[0]
    solidity_gen_typecast(
      ns, op0, to_struct_type(map_t).components().at(0).type());
    inits.op0() = op0;

    // address => this->
    exprt addr_expr = member_exprt(this_expr, "$address", addr_t);
    solidity_gen_typecast(
      ns, addr_expr, to_struct_type(map_t).components().at(1).type());
    inits.op1() = addr_expr;

    added_symbol.value = inits;
    decl.operands().push_back(inits);
  }
  else if (!set_init && is_byte_static)
  {
    side_effect_expr_function_callt call;
    get_library_function_call_no_args(
      "bytes_static_init_zero",
      "c:@F@bytes_static_init_zero",
      t,
      location_begin,
      call);
    assert(!t.get("#sol_bytesn_size").empty());
    exprt len = from_integer(
      std::stoul(t.get("#sol_bytesn_size").as_string()), uint_type());
    call.arguments().push_back(len);
    added_symbol.value = call;
    decl.operands().push_back(call);
  }
  // now we have rule out other special cases
  else if (set_init)
  {
    if (get_init_expr(init_value, literal_type, t, val))
      return true;
    added_symbol.value = val;
    decl.operands().push_back(val);
  }

  // store state variable, which will be initialized in the constructor
  // note that for the state variables that do not have initializer
  // we have already set it as zero value
  // For unintialized contract type, no need to move to the initializer
  if (
    is_state_var && !is_inherited && !(is_contract && !has_init) &&
    !(is_mapping && !is_new_expr))
    move_to_initializer(decl);

  decl.location() = location_begin;
  new_expr = decl;

  log_debug(
    "solidity", "@@@ Finish parsing symbol {}", added_symbol.name.as_string());
  return false;
}

// This function handles both contract and struct
// The contract can be regarded as the class in C++, converting to a struct
bool solidity_convertert::get_struct_class(const nlohmann::json &struct_def)
{
  // 1. populate name, id
  std::string id, name;
  struct_typet t = struct_typet();
  std::string cname;

  if (struct_def["nodeType"].get<std::string>() == "ContractDefinition")
  {
    name = struct_def["name"].get<std::string>();
    id = prefix + name;
    t.tag(name);
    cname = name;
  }
  else if (struct_def["nodeType"].get<std::string>() == "StructDefinition")
  {
    // ""tag-struct Struct_Name"
    name = struct_def["name"].get<std::string>();
    id = prefix + "struct " + struct_def["canonicalName"].get<std::string>();
    t.tag("struct " + name);

    // populate the member_entity_scope
    // this map is used to find reference when there is no decl_ref_id provided in the nodes
    // or replace the find_decl_ref in order to speed up
    int scp = struct_def["id"].get<int>();
    member_entity_scope.insert(std::pair<int, std::string>(scp, name));
  }
  else
  {
    log_error(
      "Got nodeType={}. Unsupported struct type",
      struct_def["nodeType"].get<std::string>());
    return true;
  }

  log_debug("solidity", "Parsing struct/contract class {}", name);

  // 2. Check if the symbol is already added to the context, do nothing if it is
  // already in the context.
  if (context.find_symbol(id) != nullptr)
    return false;

  // 3. populate location
  locationt location_begin;
  get_location_from_node(struct_def, location_begin);

  // 4. populate debug module name
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, t, name, id, location_begin);

  symbol.is_type = true;
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // 5. populate fields(state var) and method(function)
  // We have to add fields before methods as the fields are likely to be used
  // in the methods
  nlohmann::json ast_nodes;
  if (struct_def.contains("nodes"))
    ast_nodes = struct_def["nodes"];
  else if (struct_def.contains("members"))
    ast_nodes = struct_def["members"];
  else
  {
    // Defining empty structs is disallowed.
    // Contracts can be empty
    log_warning("Empty contract.");
  }

  for (nlohmann::json::iterator itr = ast_nodes.begin(); itr != ast_nodes.end();
       ++itr)
  {
    SolidityGrammar::ContractBodyElementT type =
      SolidityGrammar::get_contract_body_element_t(*itr);

    log_debug(
      "solidity",
      "@@@ got ContractBodyElementT = {}",
      SolidityGrammar::contract_body_element_to_str(type));

    switch (type)
    {
    case SolidityGrammar::ContractBodyElementT::VarDecl:
    {
      // this can be both state and non-state variable
      if (get_struct_class_fields(*itr, t))
        return true;
      break;
    }
    case SolidityGrammar::ContractBodyElementT::FunctionDef:
    {
      if (get_struct_class_method(*itr, t))
        return true;
      break;
    }
    case SolidityGrammar::ContractBodyElementT::StructDef:
    {
      exprt tmp_expr;
      if (get_noncontract_decl_ref(*itr, tmp_expr))
        return true;

      struct_typet::componentt comp;
      comp.swap(tmp_expr);
      comp.id("component");
      comp.type().set("#member_name", t.tag());

      if (get_access_from_decl(*itr, comp))
        return true;
      t.components().push_back(comp);
      break;
    }
    case SolidityGrammar::ContractBodyElementT::EnumDef:
    case SolidityGrammar::ContractBodyElementT::UsingForDef:
    case SolidityGrammar::ContractBodyElementT::ModifierDef:
    {
      // skip as it do not need to be populated to the value of the struct
      break;
    }
    case SolidityGrammar::ContractBodyElementT::ErrorDef:
    case SolidityGrammar::ContractBodyElementT::EventDef:
    {
      exprt tmp_expr;
      if (get_noncontract_decl_ref(*itr, tmp_expr))
        return true;
      struct_typet::componentt comp;
      comp.swap(tmp_expr);

      if (comp.is_code() && to_code(comp).statement() == "skip")
        break;

      // set virtual / override
      if ((*itr).contains("virtual") && (*itr)["virtual"] == true)
        comp.set("#is_sol_virtual", true);
      else if ((*itr).contains("overrides"))
        comp.set("#is_sol_override", true);

      t.methods().push_back(comp);
      break;
    }
    default:
    {
      log_error("Unimplemented type in rule contract-body-element");
      return true;
    }
    }
  }

  t.location() = location_begin;
  added_symbol.type = t;

  return false;
}

// parse a contract definition
bool solidity_convertert::get_contract_definition(const std::string &c_name)
{
  // cache
  // this is due to that we might call this funciton to parse another contract B
  // when we are parsing contract A
  auto old_current_baseContractName = current_baseContractName;
  auto old_current_functionName = current_functionName;
  auto old_current_functionDecl = current_functionDecl;
  auto old_current_forStmt = current_forStmt;
  auto old_initializers = initializers;

  // reset
  reset_auxiliary_vars();

  nlohmann::json &nodes = src_ast_json["nodes"];
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    std::string node_type = (*itr)["nodeType"].get<std::string>();
    if (
      node_type == "ContractDefinition" &&
      (*itr)["name"] == c_name) // rule source-unit
    {
      if (
        (*itr).contains("contractKind") && (*itr)["contractKind"] == "library")
        // we paerse library in the get_noncontract_defition
        continue;

      log_debug("solidity", "Parsing Contract {}", c_name);

      // set based contract name
      current_baseContractName = c_name;

      // set baseContracts
      // this will be used in ctor initialization
      nlohmann::json *based_contracts = nullptr;
      if ((*itr).contains("baseContracts") && !(*itr)["baseContracts"].empty())
        based_contracts = &((*itr)["baseContracts"]);

      nlohmann::json &ast_nodes = (*itr)["nodes"];
      for (nlohmann::json::iterator ittr = ast_nodes.begin();
           ittr != ast_nodes.end();
           ++ittr)
      {
        // struct/error/event....
        if (get_noncontract_defition(*ittr))
          return true;
      }

      for (nlohmann::json::iterator ittr = ast_nodes.begin();
           ittr != ast_nodes.end();
           ++ittr)
      {
        if (get_noncontract_defition(*ittr))
          return true;
      }

      // add a struct symbol for each contract
      // e.g. contract Base => struct Base
      if (get_struct_class(*itr))
        return true;

      // add solidity built-in property like balance, codehash
      if (add_auxiliary_members(*itr, c_name))
        return true;

      // parse contract body
      if (convert_ast_nodes(*itr, c_name))
        return true;
      log_debug("solidity", "@@@ Finish parsing contract {}'s body", c_name);

      // for inheritance
      bool has_inherit_from = inheritanceMap[c_name].size() > 1;
      if (
        has_inherit_from &&
        move_initializer_to_ctor(based_contracts, *itr, c_name, true))
        return true;

      // initialize state variable
      if (move_initializer_to_ctor(based_contracts, *itr, c_name))
        return true;

      symbolt s = *context.find_symbol(prefix + c_name);
    }
  }

  // restore
  current_baseContractName = old_current_baseContractName;
  current_functionName = old_current_functionName;
  current_functionDecl = old_current_functionDecl;
  current_forStmt = old_current_forStmt;
  initializers = old_initializers;

  return false;
}

bool solidity_convertert::get_struct_class_fields(
  const nlohmann::json &ast_node,
  struct_typet &type)
{
  struct_typet::componentt comp;

  if (get_var_decl_ref(ast_node, false, comp))
    return true;

  if (comp.type().get("#sol_type") == "MAPPING" && comp.type().is_array())
  {
    //! hack: for the (non-nested) mapping from contract that is not used in a new expression
    // we convert it to a global static infinity array
    // thuse we do not populate it into the struct symbol
    return false;
  }

  comp.id("component");
  // TODO: add bitfield
  // if (comp.type().get_bool("#extint"))
  // {
  //   typet t;
  //   if (get_type_description(ast_node["typeName"]["typeDescriptions"], t))
  //     return true;

  //   comp.type().set("#bitfield", true);
  //   comp.type().subtype() = t;
  //   comp.set_is_unnamed_bitfield(false);
  // }
  comp.type().set("#member_name", type.tag());

  if (get_access_from_decl(ast_node, comp))
    return true;
  type.components().push_back(comp);

  return false;
}

bool solidity_convertert::get_struct_class_method(
  const nlohmann::json &ast_node,
  struct_typet &type)
{
  struct_typet::componentt comp;
  if (get_func_decl_ref(ast_node, comp))
    return true;

  log_debug(
    "solidity", "\t\t@@@ populating method {}", comp.identifier().as_string());

  if (comp.is_code() && to_code(comp).statement() == "skip")
    return false;

  if (get_access_from_decl(ast_node, comp))
    return true;

  // set virtual / override
  if (ast_node.contains("virtual") && ast_node["virtual"] == true)
    comp.set("#is_sol_virtual", true);
  else if (ast_node.contains("overrides"))
    comp.set("#is_sol_override", true);

  type.methods().push_back(comp);
  return false;
}

bool solidity_convertert::get_noncontract_decl_ref(
  const nlohmann::json &decl,
  exprt &new_expr)
{
  log_debug(
    "solidity",
    "\tget_noncontract_decl_ref, got nodeType={}",
    decl["nodeType"].get<std::string>());
  if (decl["nodeType"] == "StructDefinition")
  {
    std::string id;
    id = prefix + "struct " + decl["canonicalName"].get<std::string>();

    if (context.find_symbol(id) == nullptr)
    {
      if (get_struct_class(decl))
        return true;
    }

    new_expr = symbol_expr(*context.find_symbol(id));
  }
  else if (decl["nodeType"] == "ErrorDefinition")
  {
    std::string name, id;
    get_error_definition_name(decl, name, id);

    if (context.find_symbol(id) == nullptr)
      return true;
    new_expr = symbol_expr(*context.find_symbol(id));
  }
  else if (decl["nodeType"] == "EventDefinition")
  {
    // treat event as a function definition
    if (get_func_decl_ref(decl, new_expr))
      return true;
  }
  else if (
    decl["nodeType"] == "ContractDefinition" &&
    decl["contractKind"] == "library")
  {
    new_expr = code_skipt();
    new_expr.type().set("#sol_type", "LIBRARY");
  }
  else
  {
    log_error("Internal parsing error");
    abort();
  }

  return false;
}

// definition of event/error/interface/struct/library/...
bool solidity_convertert::get_noncontract_defition(nlohmann::json &ast_node)
{
  std::string node_type = (ast_node)["nodeType"].get<std::string>();
  log_debug(
    "solidity", "@@@ Expecting non-contract definition, got {}", node_type);

  if (node_type == "StructDefinition")
  {
    if (get_struct_class(ast_node))
      return true;
  }
  else if (node_type == "EnumDefinition")
    // set the ["Value"] for each member inside enum
    add_enum_member_val(ast_node);
  else if (node_type == "ErrorDefinition")
  {
    add_empty_body_node(ast_node);
    if (get_error_definition(ast_node))
      return true;
  }
  else if (node_type == "EventDefinition")
  {
    add_empty_body_node(ast_node);
    if (get_function_definition(ast_node))
      return true;
  }
  else if (node_type == "ContractDefinition" && ast_node["abstract"] == true)
  {
    // for abstract contract
    add_empty_body_node(ast_node);
  }
  else if (
    node_type == "ContractDefinition" && ast_node["contractKind"] == "library")
  {
    // for library entity
    // a library is equivalent to a static class
    std::string lib_name = ast_node["name"].get<std::string>();

    // we treat library as a contract, but we do not populate it as struct/contract symbol
    // instead, we only populate the entity and functions
    std::string old = current_baseContractName;
    current_baseContractName = lib_name;
    if (get_struct_class(ast_node))
      return true;

    if (convert_ast_nodes(ast_node, lib_name))
      return true;

    current_baseContractName = old;
  }

  return false;
}

// add a "body" node to funcitons within interfacae && abstract && event
// the idea is to utilize the function-handling APIs.
void solidity_convertert::add_empty_body_node(nlohmann::json &ast_node)
{
  //? will this affect find_decl_ref?
  if (ast_node["nodeType"] == "EventDefinition")
  {
    // for event-definition
    if (!ast_node.contains("body"))

      ast_node["body"] = {
        {"nodeType", "Block"},
        {"statements", nlohmann::json::array()},
        {"src", ast_node["src"]}};
  }
  else if (ast_node["contractKind"] == "interface")
  {
    // For interface: functions have no body
    for (auto &subNode : ast_node["nodes"])
    {
      if (
        (subNode["nodeType"] == "FunctionDefinition") &&
        !subNode.contains("body"))
        subNode["body"] = {
          {"nodeType", "Block"},
          {"statements", nlohmann::json::array()},
          {"src", ast_node["src"]}};
    }
  }
  else if (ast_node["abstract"] == true)
  {
    // For abstract: functions may or may not have body
    for (auto &subNode : ast_node["nodes"])
    {
      if (
        (subNode["nodeType"] == "FunctionDefinition") &&
        !subNode.contains("body"))
        subNode["body"] = {
          {"nodeType", "Block"},
          {"statements", nlohmann::json::array()},
          {"src", ast_node["src"]}};
    }
  }
}

void solidity_convertert::add_enum_member_val(nlohmann::json &ast_node)
{
  /*
  "nodeType": "EnumDefinition",
  "members": 
    [
      {
          "id": 2,
          "name": "SMALL",
          "nameLocation": "66:5:0",
          "nodeType": "EnumValue",
          "src": "66:5:0",
          "Value": 0 => new added object
      },
      {
          "id": 3,
          "name": "MEDIUM",
          "nameLocation": "73:6:0",
          "nodeType": "EnumValue",
          "src": "73:6:0",
          "Value": 1  => new added object
      },
    ] */

  assert(ast_node["nodeType"] == "EnumDefinition");
  int idx = 0;
  nlohmann::json &members = ast_node["members"];
  for (nlohmann::json::iterator itr = members.begin(); itr != members.end();
       ++itr, ++idx)
  {
    if (!(*itr).contains("Value"))
      (*itr).push_back(
        nlohmann::json::object_t::value_type("Value", std::to_string(idx)));
  }
}

// covert the error_definition to a function
bool solidity_convertert::get_error_definition(const nlohmann::json &ast_node)
{
  // e.g.
  // error errmsg(int num1, uint num2, uint[2] addrs);
  //   to
  // function 'tag-erro errmsg@12'() { __ESBMC_assume(false);}

  const nlohmann::json *old_functionDecl = current_functionDecl;
  const std::string old_functionName = current_functionName;

  std::string cname;
  get_current_contract_name(ast_node, cname);

  // e.g. name: errmsg; id: sol:@errmsg#12
  std::string name, id;
  get_error_definition_name(ast_node, name, id);
  const int id_num = ast_node["id"].get<int>();

  if (context.find_symbol(id) != nullptr)
  {
    current_functionDecl = old_functionDecl;
    current_functionName = old_functionName;
    return false;
  }
  // update scope map
  member_entity_scope.insert(std::pair<int, std::string>(id_num, name));

  // just to pass the internal assertions
  current_functionName = name;
  current_functionDecl = &ast_node;

  // no return value
  code_typet type;
  typet e_type = empty_typet();
  e_type.set("cpp_type", "void");
  type.return_type() = e_type;

  locationt location_begin;
  get_location_from_node(ast_node, location_begin);
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, type, name, id, location_begin);
  symbol.lvalue = true;

  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // populate the params
  SolidityGrammar::ParameterListT params =
    SolidityGrammar::get_parameter_list_t(ast_node["parameters"]);
  if (params == SolidityGrammar::ParameterListT::EMPTY)
    type.make_ellipsis();
  else
  {
    for (const auto &decl : ast_node["parameters"]["parameters"].items())
    {
      const nlohmann::json &func_param_decl = decl.value();

      code_typet::argumentt param;
      if (get_function_params(func_param_decl, cname, param))
        return true;

      type.arguments().push_back(param);
    }
  }
  added_symbol.type = type;

  // construct a "__ESBMC_assume(false)" statement
  typet return_type = empty_typet();
  locationt loc;
  get_location_from_node(ast_node, loc);
  side_effect_expr_function_callt call;
  get_library_function_call_no_args(
    "__ESBMC_assume", "c:@F@__ESBMC_assume", return_type, loc, call);

  exprt arg = false_exprt();
  call.arguments().push_back(arg);
  convert_expression_to_code(call);

  // insert it to the body
  code_blockt body;
  body.operands().push_back(call);
  added_symbol.value = body;

  // restore
  current_functionDecl = old_functionDecl;
  current_functionName = old_functionName;

  return false;
}

// add ["is_inherited"] = true to node and all sub_node that contains an "id"
bool solidity_convertert::get_access_from_decl(
  const nlohmann::json &ast_node,
  struct_typet::componentt &comp)
{
  if (
    SolidityGrammar::get_access_t(ast_node) ==
    SolidityGrammar::VisibilityT::UnknownT)
    return true;

  std::string access = ast_node["visibility"].get<std::string>();
  comp.set_access(access);

  return false;
}

void solidity_convertert::get_state_var_decl_name(
  const nlohmann::json &ast_node,
  const std::string &cname,
  std::string &name,
  std::string &id)
{
  // Follow the way in clang:
  //  - For state variable name, just use the ast_node["name"], e.g. sol:@C@Base@x#11
  //  - For state variable id, add prefix "sol:@"
  name = ast_node["name"].get<std::string>();
  if (!cname.empty())
    id = "sol:@C@" + cname + "@" + name + "#" +
         i2string(ast_node["id"].get<std::int16_t>());
  else
    id = "sol:@" + name + "#" + i2string(ast_node["id"].get<std::int16_t>());
}

bool solidity_convertert::get_var_decl_name(
  const nlohmann::json &decl,
  std::string &name,
  std::string &id)
{
  std::string cname;
  get_current_contract_name(decl, cname);

  if (decl["stateVariable"])
    get_state_var_decl_name(decl, cname, name, id);
  else
  {
    if (cname.empty() && decl["mutability"] == "constant")
      // global variable
      get_state_var_decl_name(decl, "", name, id);
    else
      get_local_var_decl_name(decl, cname, name, id);
  }

  return false;
}

// parse the non-state variable
void solidity_convertert::get_local_var_decl_name(
  const nlohmann::json &ast_node,
  const std::string &cname,
  std::string &name,
  std::string &id)
{
  assert(ast_node.contains("id"));
  assert(ast_node.contains("name"));

  name = ast_node["name"].get<std::string>();
  if ((current_functionDecl || !current_functionName.empty()) && !cname.empty())
  {
    // converting local variable inside a function
    // For non-state functions, we give it different id.
    // E.g. for local variable i in function nondet(), it's "sol:@C@Base@F@nondet@i#55".
    if (current_functionName.empty())
      current_functionName = (*current_functionDecl)["name"];
    assert(!current_functionName.empty());
    // As the local variable inside the function will not be inherited, we can use current_functionName
    id = "sol:@C@" + cname + "@F@" + current_functionName + "@" + name + "#" +
         i2string(ast_node["id"].get<std::int16_t>());
  }
  else if (ast_node.contains("scope"))
  {
    // This means we are handling a local variable which is not inside a function body.
    //! Assume it is a variable inside struct/error which can be declared outside the contract
    int scp = ast_node["scope"].get<int>();
    if (member_entity_scope.count(scp) == 0)
    {
      log_error("cannot find struct/error name");
      abort();
    }
    std::string struct_name = member_entity_scope.at(scp);
    if (cname.empty())
      id = "sol:@" + struct_name + "@" + name + "#" +
           i2string(ast_node["id"].get<std::int16_t>());
    else
      id = "sol:@C@" + cname + "@" + struct_name + "@" + name + "#" +
           i2string(ast_node["id"].get<std::int16_t>());
  }
  else
  {
    log_error("Unsupported local variable");
    abort();
  }
}

void solidity_convertert::get_error_definition_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  std::string cname;
  get_current_contract_name(ast_node, cname);
  const int id_num = ast_node["id"].get<int>();
  name = ast_node["name"].get<std::string>();
  if (cname.empty())
    id = "sol:@" + name + "#" + std::to_string(id_num);
  else
    // e.g. sol:@C@Base@F@error@1
    id = "sol:@C@" + cname + "@F@" + name + "#" + std::to_string(id_num);
}

void solidity_convertert::get_function_definition_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  // Follow the way in clang:
  //  - For function name, just use the ast_node["name"]
  // assume Solidity AST json object has "name" field, otherwise throws an exception in nlohmann::json
  std::string contract_name;
  get_current_contract_name(ast_node, contract_name);
  if (contract_name.empty())
  {
    name = ast_node["name"].get<std::string>();
    id = "sol:@F@" + name + "#" + i2string(ast_node["id"].get<std::int16_t>());
    return;
  }

  //! for event/... who have added an body node. It seems that a ["kind"] is automatically added.?
  if (
    ast_node.contains("kind") && !ast_node["kind"].is_null() &&
    ast_node["kind"].get<std::string>() == "constructor")
    // In solidity
    // - constructor does not have a name
    // - there can be only one constructor in each contract
    // we, however, mimic the C++ grammar to manually assign it with a name
    // whichi is identical to the contract name
    // we also allows multiple constructor where the added ctor has no  `id`
    name = contract_name;
  else
    name = ast_node["name"] == "" ? ast_node["kind"] : ast_node["name"];

  id = "sol:@C@" + contract_name + "@F@" + name + "#" +
       i2string(ast_node["id"].get<std::int16_t>());

  log_debug("solidity", "\t\t@@@ got function name {}", name);
}

