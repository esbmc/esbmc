/// \file solidity_convert_mapping.cpp
/// \brief Mapping type conversion for the Solidity frontend.
///
/// Converts Solidity mapping types into ESBMC's representation using
/// infinite arrays (modeled as arrays with nondet size). Handles nested
/// mappings, mapping access expressions, and the generation of helper
/// symbols for mapping state variables.

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

void solidity_convertert::get_mapping_inf_arr_name(
  const std::string &cname,
  const std::string &name,
  std::string &arr_name,
  std::string &arr_id)
{
  arr_name = "_ESBMC_inf_" + name;
  // we cannot define a mapping inside a function body
  arr_id = "sol:@C@" + cname + "@" + arr_name + "#";
}

/**
	@target: target index access child json
	return true if it's a mapping_set, including assign, assign+, tuple assign...
	otherwise return false, representing mapping_get
*/
bool solidity_convertert::is_mapping_set_lvalue(const nlohmann::json &target)
{
  assert(target.value("nodeType", "") == "IndexAccess");
  assert(target.contains("lValueRequested"));
  return target["lValueRequested"].get<bool>();
}

bool solidity_convertert::get_mapping_key_value_type(
  const nlohmann::json &map_node,
  typet &key_t,
  typet &value_t,
  SolidityGrammar::SolType &key_sol_type,
  SolidityGrammar::SolType &val_sol_type)
{
  assert(map_node.contains("typeName"));
  if (get_type_description(
        map_node["typeName"]["keyType"]["typeDescriptions"], key_t))
  {
    log_error("cannot get mapping key type");
    return true;
  }
  if (get_type_description(
        map_node["typeName"]["valueType"]["typeDescriptions"], value_t))
  {
    log_error("cannot get mapping value type");
    return true;
  }

  // set type flag
  key_sol_type = get_sol_type(key_t);
  val_sol_type = get_sol_type(value_t);
  if (val_sol_type == SolidityGrammar::SolType::UNSET)
    return true;
  return false;
}

// e.g. bytes3 = bytes3(x) ==> length == 3
void solidity_convertert::get_bytesN_size(
  const exprt &src_expr,
  exprt &len_expr)
{
  std::string byte_size = src_expr.type().get("#sol_bytesn_size").as_string();
  if (!byte_size.empty())
    len_expr = from_integer(std::stoul(byte_size), size_type());
  else
  {
    assert(!src_expr.is_nil());
    len_expr = member_exprt(src_expr, "length", size_type());
  }
}

bool solidity_convertert::get_dynamic_pool(
  const std::string &c_name,
  exprt &pool)
{
  exprt cur_this_expr;
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, cur_this_expr))
      return true;
  }
  else
  {
    if (get_ctor_decl_this_ref(c_name, cur_this_expr))
      return true;
  }

  pool = member_exprt(
    cur_this_expr, "$dynamic_pool", symbol_typet(lib_prefix + "BytesPool"));

  return false;
}

bool solidity_convertert::get_dynamic_pool(
  const nlohmann::json &expr,
  exprt &pool)
{
  std::string c_name;
  get_current_contract_name(expr, c_name);
  if (c_name.empty())
    return true;
  return get_dynamic_pool(c_name, pool);
}

/**
 * @brief Recursively checks whether a Solidity AST node contains
 *        any usage of array-specific operations: push, pop, or length,
 *        excluding operations on the `bytes` type.
 *
 * This function traverses the AST (produced by solc in JSON format)
 * and looks for `MemberAccess` nodes with member names "push", "pop",
 * or "length". It ensures that these accesses are on array types
 * (e.g., `int[]`, `uint256[]`) and not on the dynamic `bytes` type.
 *
 * @param node A JSON node from the Solidity AST.
 * @return true if any array push/pop/length usage is found (excluding bytes).
 * @return false otherwise.
 */
bool solidity_convertert::check_array_push_pop_length(
  const nlohmann::json &node)
{
  auto is_array_type_not_bytes = [](const nlohmann::json &type_desc) -> bool {
    if (!type_desc.is_object())
      return false;
    if (!type_desc.contains("typeString"))
      return false;

    std::string type_str = type_desc["typeString"];
    if (type_str == "bytes")
      return false;
    if (type_str.find("[]") != std::string::npos)
      return true;
    return false;
  };

  if (node.is_object())
  {
    if (node.contains("nodeType") && node["nodeType"] == "MemberAccess")
    {
      if (
        node.contains("memberName") &&
        (node["memberName"] == "push" || node["memberName"] == "pop" ||
         node["memberName"] == "length"))
      {
        if (
          node.contains("expression") &&
          node["expression"].contains("typeDescriptions") &&
          is_array_type_not_bytes(node["expression"]["typeDescriptions"]))
        {
          return true;
        }
      }
    }

    for (const auto &kv : node.items())
    {
      if (check_array_push_pop_length(kv.value()))
        return true;
    }
  }
  else if (node.is_array())
  {
    for (const auto &element : node)
    {
      if (check_array_push_pop_length(element))
        return true;
    }
  }

  return false;
}

// check if current contract have bytes (not bytesN) type
bool solidity_convertert::has_contract_bytes(const nlohmann::json &node)
{
  if (node.is_object())
  {
    if (
      node.contains("typeDescriptions") &&
      node["typeDescriptions"].contains("typeString"))
    {
      const std::string &ts = node["typeDescriptions"]["typeString"];
      // Match "bytes", "bytes storage pointer", "bytes memory", etc.
      // Also match "string" variants since string uses BytesDynamic internally.
      if (
        ts == "bytes" || ts.substr(0, 6) == "bytes " || ts == "string" ||
        ts.substr(0, 7) == "string ")
        return true;
    }

    for (const auto &kv : node.items())
    {
      if (has_contract_bytes(kv.value()))
        return true;
    }
  }
  else if (node.is_array())
  {
    for (const auto &element : node)
    {
      if (has_contract_bytes(element))
        return true;
    }
  }

  return false;
}

void solidity_convertert::gen_mapping_key_typecast(
  const std::string &c_name,
  exprt &pos,
  const locationt &location,
  const typet &key_type)
{
  SolidityGrammar::SolType key_sol_type = get_sol_type(key_type);
  if (
    key_sol_type == SolidityGrammar::SolType::STRING ||
    key_sol_type == SolidityGrammar::SolType::STRING_LITERAL)
  {
    side_effect_expr_function_callt str2uint_call;
    assert(context.find_symbol("c:@F@str2uint") != nullptr);
    get_library_function_call_no_args(
      "str2uint",
      "c:@F@str2uint",
      unsignedbv_typet(256),
      location,
      str2uint_call);
    str2uint_call.arguments().push_back(pos);
    pos = str2uint_call;
    solidity_gen_typecast(ns, pos, unsignedbv_typet(256));
    return;
  }
  // bytesN: use bytes_static_to_mapping_key
  else if (is_bytesN_type(key_type))
  {
    // bytes_static_to_mapping_key(pos)
    side_effect_expr_function_callt call;
    get_library_function_call_no_args(
      "bytes_static_to_mapping_key",
      "c:@F@bytes_static_to_mapping_key",
      unsignedbv_typet(256),
      location,
      call);
    call.arguments().push_back(pos);
    pos = call;
    return;
  }
  else if (is_bytes_type(key_type))
  {
    side_effect_expr_function_callt bytes_dynamic_call;
    assert(context.find_symbol("c:@F@bytes_dynamic_to_mapping_key") != nullptr);
    get_library_function_call_no_args(
      "bytes_dynamic_to_mapping_key",
      "c:@F@bytes_dynamic_to_mapping_key",
      unsignedbv_typet(256),
      location,
      bytes_dynamic_call);
    bytes_dynamic_call.arguments().push_back(pos);

    // get dynamic_pool from current contract instance
    // get this
    exprt dynamic_pool_member;
    if (get_dynamic_pool(c_name, dynamic_pool_member))
      abort();

    bytes_dynamic_call.arguments().push_back(dynamic_pool_member);

    pos = bytes_dynamic_call;
    return;
  }
  // fallback for all others: keep old logic
  solidity_gen_typecast(ns, pos, unsignedbv_typet(256));
}

void solidity_convertert::xor_fold_key_to_64bit(exprt &key)
{
  // Fold a 256-bit mapping key to 64-bit to avoid SMT performance issues
  // with 256-bit array domains, while preserving collision resistance at 2^-64.
  //
  // Routed through a single library call _ESBMC_str_key_fold64 so the side
  // effect of `key` (e.g. a str2uint or bytes_static_to_mapping_key
  // function call) is evaluated exactly once. An inline 4-way XOR over `key`
  // would duplicate the side effect and produce four independent symbolic
  // results that fail to alias on equal inputs (causing false negatives on
  // mapping reads that should hit a previously-stored entry).

  const locationt loc = static_cast<const locationt &>(key.find("#location"));

  side_effect_expr_function_callt fold_call;
  assert(context.find_symbol("c:@F@_ESBMC_str_key_fold64") != nullptr);
  get_library_function_call_no_args(
    "_ESBMC_str_key_fold64",
    "c:@F@_ESBMC_str_key_fold64",
    unsignedbv_typet(64),
    loc,
    fold_call);
  fold_call.arguments().push_back(key);

  key = fold_call;
}

/**
  index accesss could either be set or get:
  x[1]      => map_uint_get(&m, 1)
  x[1] = 2  => map_uint_set(&x, 1, 2)
  @array: x
  @pos: 1
  @is_mapping_set: true if it's a setValue, otherwise getValue
*/
bool solidity_convertert::get_new_mapping_index_access(
  const typet &value_t,
  SolidityGrammar::SolType val_sol_type,
  bool is_mapping_set,
  const exprt &array,
  const exprt &pos,
  const locationt &location,
  exprt &new_expr)
{
  std::string val_flg;
  typet func_type;
  if (
    SolidityGrammar::is_uint_type(val_sol_type) ||
    SolidityGrammar::is_bytes_type(val_sol_type) ||
    SolidityGrammar::is_address_type(val_sol_type) ||
    val_sol_type == SolidityGrammar::SolType::ENUM)
  {
    val_flg = "uint";
    func_type = unsignedbv_typet(256);
  }
  else if (SolidityGrammar::is_int_type(val_sol_type))
  {
    val_flg = "int";
    func_type = signedbv_typet(256);
  }
  else if (val_sol_type == SolidityGrammar::SolType::BOOL)
  {
    val_flg = "bool";
    func_type = bool_typet();
  }
  else if (
    val_sol_type == SolidityGrammar::SolType::STRING ||
    val_sol_type == SolidityGrammar::SolType::STRING_LITERAL)
  {
    val_flg = "string";
    func_type = value_t;
  }
  else
  {
    val_flg = "generic";
    // void *
    func_type = pointer_typet(empty_typet());
  }

  // construct func call
  std::string func_name;
  if (is_mapping_set)
  {
    func_name = "map_" + val_flg + "_set";
    func_type = empty_typet();
    // overwrite func_type
    func_type.set("cpp_type", "void");
  }
  else
    func_name = "map_" + val_flg + "_get";

  if (context.find_symbol("c:@F@" + func_name) == nullptr)
  {
    log_error(
      "cannot find mapping ref {}. Got val_sol_type={}",
      func_name,
      SolidityGrammar::sol_type_to_str(val_sol_type));
    return true;
  }
  side_effect_expr_function_callt call;
  get_library_function_call_no_args(
    func_name, "c:@F@" + func_name, func_type, location, call);

  // &array
  call.arguments().push_back(address_of_exprt(array));

  // index
  call.arguments().push_back(pos);

  if (is_mapping_set)
  {
    /*
        case 1: x[1] += 2 =>
          DECL temp = map_uint_get(&array, pos);  <-- move to front block
          temp += 2;
          map_uint_set(&array, pos, temp);  <-- move to back block
          (map_generic_set(&array, pos, temp, sizeof(temp));)
    */
    std::string aux_name, aux_id;
    get_aux_var(aux_name, aux_id);
    symbolt aux_sym;
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    typet aux_type = value_t;
    get_default_symbol(
      aux_sym, debug_modulename, aux_type, aux_name, aux_id, location);
    aux_sym.file_local = true;
    aux_sym.lvalue = true;
    auto &added_sym = *move_symbol_to_context(aux_sym);
    code_declt decl(symbol_expr(added_sym));

    // populate initial value
    side_effect_expr_function_callt get_call;
    std::string f_get_name = "map_" + val_flg + "_get";

    get_library_function_call_no_args(
      f_get_name, "c:@F@" + f_get_name, value_t, location, get_call);
    get_call.arguments().push_back(address_of_exprt(array));
    get_call.arguments().push_back(pos);
    solidity_gen_typecast(ns, get_call, aux_type);
    added_sym.value = get_call;
    decl.operands().push_back(get_call);
    move_to_front_block(decl);

    // value
    call.arguments().push_back(symbol_expr(added_sym));
    if (val_flg == "generic")
    {
      // sizeof
      exprt size_of_expr;
      get_size_of_expr(value_t, size_of_expr);
      call.arguments().push_back(symbol_expr(added_sym));
    }

    convert_expression_to_code(call);
    move_to_back_block(call);

    new_expr = symbol_expr(added_sym);
  }
  else if (val_flg == "generic")
  {
    /* generic_get:
          case 2: users[msg.sender].age; =>
            DECL struct temp = map_users_get(&array, pos);
            temp.age;
        */
    std::string aux_name, aux_id;
    get_aux_var(aux_name, aux_id);
    symbolt aux_sym;
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    typet aux_type = value_t; // struct *
    get_default_symbol(
      aux_sym, debug_modulename, aux_type, aux_name, aux_id, location);
    aux_sym.file_local = true;
    aux_sym.lvalue = true;
    auto &added_sym = *move_symbol_to_context(aux_sym);
    code_declt decl(symbol_expr(added_sym));

    // construct map_{struct_name}_get() function
    // e.g. map_Base_User_get();
    exprt map_struct_get;
    std::string struct_contract_name = value_t.identifier().as_string();
    assert(!struct_contract_name.empty());
    assert(val_sol_type == SolidityGrammar::SolType::STRUCT); // t_symbol
    get_mapping_struct_function(
      value_t, struct_contract_name, call, map_struct_get);

    // struct temp = map_users_get(&array, pos);
    added_sym.value = map_struct_get;
    decl.operands().push_back(map_struct_get);
    move_to_front_block(decl);

    new_expr = symbol_expr(added_sym);
  }
  else
  {
    // e.g. (int8)map_int_get(&arr, 1);
    solidity_gen_typecast(ns, call, value_t);
    new_expr = call;
  }

  return false;
}

void solidity_convertert::get_mapping_struct_function(
  const typet &struct_t,
  std::string &struct_contract_name,
  const side_effect_expr_function_callt &gen_call,
  exprt &new_expr)
{
  /*
  e.g.
  struct A map_get_A_default_val(struct mapping_t *m, uint256_t k)
  {
  __ESBMC_HIDE:;
    struct A *ap = (struct A *)map_get_generic(m, k);
    return ap ? *ap : (struct A){0};
  }
  */
  side_effect_expr_function_callt call;

  // split contract struct name
  // drop prefix
  struct_contract_name = struct_contract_name.substr(prefix.length());
  // replace "." to "_"
  std::replace(
    struct_contract_name.begin(), struct_contract_name.end(), '.', '_');
  std::string func_name = "map_" + struct_contract_name + "_get";
  std::string func_id = "c:@F@" + func_name;
  if (context.find_symbol(func_id) != nullptr)
  {
    call.function() = symbol_expr(*context.find_symbol(func_id));
    call.type() = struct_t;
    for (auto &arg : gen_call.arguments())
      call.arguments().push_back(arg); // same arugments as map_get_generic
    new_expr = call;
    return;
  }

  std::string debug_modulename = get_modulename_from_path(absolute_path);
  code_typet func_t;
  func_t.return_type() = struct_t;
  symbolt sym;
  get_default_symbol(
    sym, debug_modulename, func_t, func_name, func_id, gen_call.location());
  sym.file_local = true;
  auto &func_sym = *move_symbol_to_context(sym);

  code_blockt func_body;
  // hide it
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.move_to_operands(label);

  // struct A *ap = (struct A *)map_get_generic(m, k);
  std::string aux_name, aux_id;
  get_aux_var(aux_name, aux_id);
  symbolt aux_sym;
  typet aux_type = gen_pointer_type(struct_t); // struct *
  get_default_symbol(
    aux_sym, debug_modulename, aux_type, aux_name, aux_id, gen_call.location());
  aux_sym.file_local = true;
  aux_sym.lvalue = true;
  auto &added_sym = *move_symbol_to_context(aux_sym);
  code_declt decl(symbol_expr(added_sym));
  // for typcast
  side_effect_expr_function_callt temp_call = gen_call;
  solidity_gen_typecast(ns, temp_call, aux_type);
  added_sym.value = temp_call;
  decl.operands().push_back(temp_call);
  // move to func body
  func_body.operands().push_back(decl);

  // ternary: return ap ? *ap : (struct A){0};
  // we split it into
  // - struct A aux = {0};
  // - return ap ? *ap : aux;

  // construct empty struct instance
  std::string aux_name2, aux_id2;
  get_aux_var(aux_name2, aux_id2);
  symbolt aux_sym2;
  typet aux_type2 = struct_t; // struct *
  get_default_symbol(
    aux_sym2,
    debug_modulename,
    aux_type2,
    aux_name2,
    aux_id2,
    gen_call.location());
  aux_sym2.file_local = true;
  aux_sym2.lvalue = true;
  auto &added_sym2 = *move_symbol_to_context(aux_sym2);
  code_declt decl2(symbol_expr(added_sym2));
  // zero value
  exprt inits = gen_zero(get_complete_type(aux_type2, ns), true);
  added_sym2.value = inits;
  decl2.operands().push_back(inits);
  // move to func body
  func_body.operands().push_back(decl2);

  // ap ? *ap : aux;
  exprt if_expr("if", struct_t);
  if_expr.operands().push_back(symbol_expr(added_sym));
  if_expr.operands().push_back(
    dereference_exprt(symbol_expr(added_sym), struct_t));
  if_expr.operands().push_back(symbol_expr(added_sym2));
  if_expr.location() = gen_call.location();

  // return ap ? *ap : aux;
  code_returnt ret;
  ret.return_value() = if_expr;
  func_body.operands().push_back(ret);

  func_sym.value = func_body;

  // func call
  call.function() = symbol_expr(func_sym);
  call.type() = struct_t;
  for (auto &arg : gen_call.arguments())
    call.arguments().push_back(arg); // same arugments as map_get_generic
  new_expr = call;
}

// invoking a function in the library
// note that the function symbol might not be inside the symbol table at the moment
void solidity_convertert::get_library_function_call_no_args(
  const std::string &func_name,
  const std::string &func_id,
  const typet &t,
  const locationt &l,
  exprt &new_expr)
{
  side_effect_expr_function_callt call_expr;

  exprt type_expr("symbol");
  type_expr.name(func_name);
  type_expr.pretty_name(func_name);
  type_expr.identifier(func_id);

  typet type;
  if (t.is_code())
    // this means it's a func symbol read from the symbol_table
    type = to_code_type(t).return_type();
  else
    type = t;

  call_expr.function() = type_expr;
  call_expr.type() = type;

  call_expr.location() = l;
  new_expr = call_expr;
}

void solidity_convertert::get_malloc_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &malc_call)
{
  const std::string malc_name = "malloc";
  const std::string malc_id = "c:@F@malloc";
  const symbolt &malc_sym = *context.find_symbol(malc_id);
  get_library_function_call_no_args(
    malc_name, malc_id, symbol_expr(malc_sym).type(), loc, malc_call);
}

void solidity_convertert::get_calloc_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &calc_call)
{
  const std::string calc_name = "calloc";
  const std::string calc_id = "c:@F@calloc";
  const symbolt &calc_sym = *context.find_symbol(calc_id);
  get_library_function_call_no_args(
    calc_name, calc_id, symbol_expr(calc_sym).type(), loc, calc_call);
}

void solidity_convertert::get_arrcpy_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &calc_call)
{
  const std::string calc_name = "_ESBMC_arrcpy";
  const std::string calc_id = "c:@F@_ESBMC_arrcpy";
  const symbolt &calc_sym = *context.find_symbol(calc_id);
  get_library_function_call_no_args(
    calc_name, calc_id, symbol_expr(calc_sym).type(), loc, calc_call);
}

void solidity_convertert::get_str_assign_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &_call)
{
  const std::string func_name = "_str_assign";
  const std::string func_id = "c:@F@_str_assign";
  const symbolt &func_sym = *context.find_symbol(func_id);
  get_library_function_call_no_args(
    func_name, func_id, symbol_expr(func_sym).type(), loc, _call);
}

void solidity_convertert::get_memcpy_function_call(
  const locationt &loc,
  side_effect_expr_function_callt &memc_call)
{
  const std::string memc_name = "memcpy";
  const std::string memc_id = "c:@F@memcpy";
  const symbolt &memc_sym = *context.find_symbol(memc_id);
  get_library_function_call_no_args(
    memc_name, memc_id, symbol_expr(memc_sym).type(), loc, memc_call);
}

// check if the function is a library function (defined in solidity.h)
bool solidity_convertert::is_esbmc_library_function(const std::string &id)
{
  if (context.find_symbol(id) == nullptr)
    return false;
  if (id.compare(0, 3, "c:@") == 0)
    return true;
  return false;
}
