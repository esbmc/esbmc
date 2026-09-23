/// \file solidity_convert_tuple.cpp
/// \brief Tuple expression conversion for the Solidity frontend.
///
/// Converts Solidity tuple expressions and multi-return-value function calls
/// from the solc JSON AST. Handles tuple declarations, tuple assignments
/// (destructuring), and the creation of temporary variables for multi-valued
/// returns.

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

bool solidity_convertert::get_tuple_definition(const nlohmann::json &ast_node)
{
  log_debug("solidity", "\t@@@ Parsing tuple...");

  // Free functions (outside any contract) also produce tuple types
  // (e.g. `function unpackTokenId(...) returns (uint256, uint32)`).
  // The tuple struct type is keyed only by ast node id, so the
  // current-contract check is not actually needed here.
  std::string current_contractName;
  get_current_contract_name(ast_node, current_contractName);

  struct_typet t = struct_typet();

  // get name/id:
  std::string name, id;
  get_tuple_name(ast_node, name, id);

  // get type:
  t.tag("struct " + name);

  // get location
  locationt location_begin;
  get_location_from_node(ast_node, location_begin);

  // get debug module name
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  // populate struct type symbol
  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, t, name, id, location_begin);
  symbol.static_lifetime = true;
  symbol.file_local = true;
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  auto &args = ast_node.contains("components")
                 ? ast_node["components"]
                 : ast_node["returnParameters"]["parameters"];

  // populate params
  //TODO: flatten the nested tuple (e.g. ((x,y),z) = (func(),1); )
  size_t counter = 0;
  for (const auto &arg : args.items())
  {
    if (arg.value().is_null())
    {
      ++counter;
      continue;
    }

    struct_typet::componentt comp;

    // manually create a member_name
    // follow the naming rule defined in get_local_var_decl_name
    assert(!current_contractName.empty());
    const std::string mem_name = "mem" + std::to_string(counter);
    const std::string mem_id = "sol:@C@" + current_contractName + "@" + name +
                               "@" + mem_name + "#" +
                               i2string(ast_node["id"].get<int>());

    // get type — route through the decl-aware overload when the tuple
    // component's AST carries a typeName, so array shapes flow through
    // get_array_pointer_type and end up as pointers (consistent with
    // get_function_params and get_var_decl). The plain typeDescriptions
    // path lowers fixed-outer NestedArrayTypeName to a real array_typet,
    // which then fights the rest of the call/assignment plumbing.
    typet mem_type;
    if (arg.value().contains("typeName"))
    {
      if (get_type_description(
            arg.value(),
            arg.value()["typeName"]["typeDescriptions"],
            mem_type))
        return true;
    }
    else
    {
      if (get_type_description(arg.value()["typeDescriptions"], mem_type))
        return true;
    }

    // construct comp
    comp.type() = mem_type;
    comp.type().set("#member_name", t.tag());
    comp.identifier(mem_id);
    comp.name(mem_name);
    comp.pretty_name(mem_name);
    comp.set_access("internal");

    // update struct type component
    t.components().push_back(comp);

    // update cnt
    ++counter;
  }

  t.location() = location_begin;
  added_symbol.type = t;

  return false;
}

bool solidity_convertert::get_tuple_instance(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  std::string name, id;
  get_tuple_name(ast_node, name, id);

  if (context.find_symbol(id) == nullptr)
    return true;

  // get type
  typet t = context.find_symbol(id)->type;
  set_sol_type(t, SolidityGrammar::SolType::TUPLE_INSTANCE);
  assert(t.id() == typet::id_struct);

  // get instance name,id
  if (get_tuple_instance_name(ast_node, name, id))
    return true;

  // get location
  locationt location_begin;
  get_location_from_node(ast_node, location_begin);

  // get debug module name
  std::string debug_modulename =
    get_modulename_from_path(location_begin.file().as_string());

  // populate struct type symbol
  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, t, name, id, location_begin);
  symbol.static_lifetime = true;
  symbol.file_local = true;

  symbol.value = gen_zero(get_complete_type(t, ns), true);
  symbol.value.zero_initializer(true);
  symbolt &added_symbol = *move_symbol_to_context(symbol);
  new_expr = symbol_expr(added_symbol);
  new_expr.identifier(id);

  if (!ast_node.contains("components"))
  {
    // assume it's function return parameter list
    // therefore no initial value
    return false;
  }

  // do assignment
  auto &args = ast_node["components"];

  size_t i = 0;
  size_t j = 0;
  unsigned is = to_struct_type(t).components().size();
  unsigned as = args.size();
  assert(is <= as);

  exprt comp;
  exprt member_access;
  while (i < is && j < as)
  {
    if (args.at(j).is_null())
    {
      ++j;
      continue;
    }

    comp = to_struct_type(t).components().at(i);
    if (get_tuple_member_call(id, comp, member_access))
      return true;

    exprt init;
    const nlohmann::json &litera_type = args.at(j)["typeDescriptions"];

    if (get_expr(args.at(j), litera_type, init))
      return true;

    get_tuple_assignment(ast_node, member_access, init);

    // update
    ++i;
    ++j;
  }

  return false;
}

void solidity_convertert::get_tuple_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  name = "tuple" + std::to_string(ast_node["id"].get<int>());
  id = prefix + "struct " + name;
}

bool solidity_convertert::get_tuple_instance_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  std::string c_name;
  get_current_contract_name(ast_node, c_name);
  // Free functions have no owning contract; use a synthetic scope
  // so multi-return free functions still get a distinct tuple instance.
  if (c_name.empty())
    c_name = "__free__";

  name = "tuple_instance$" + std::to_string(ast_node["id"].get<int>());
  id = "sol:@C@" + c_name + "@" + name;
  return false;
}

/*
  obtain the corresponding tuple struct instance from the symbol table
  based on the function definition json
*/
bool solidity_convertert::get_tuple_function_ref(
  const nlohmann::json &ast_node,
  exprt &new_expr)
{
  assert(ast_node.contains("nodeType"));

  // Resolve the function declaration ID depending on the node type:
  // - Identifier: direct function reference (e.g., func())
  // - MemberAccess: cross-contract call (e.g., p.getPair())
  int ref_decl_id;
  if (
    ast_node["nodeType"] == "Identifier" ||
    ast_node["nodeType"] == "MemberAccess")
  {
    // Builtin low-level calls (`addr.call`, `addr.delegatecall`,
    // `addr.staticcall`) and other builtin tuple-producing members
    // are MemberAccess nodes with no referencedDeclaration. The
    // tuple instance for their (bool, bytes) return is synthesised
    // by the low-level call path, not by the per-function tuple
    // registry — fail softly so the caller falls back.
    if (
      !ast_node.contains("referencedDeclaration") ||
      ast_node["referencedDeclaration"].is_null())
    {
      log_debug(
        "solidity",
        "get_tuple_function_ref: callee has no referencedDeclaration "
        "(nodeType={}), cannot resolve tuple instance",
        ast_node["nodeType"].get<std::string>());
      return true;
    }
    ref_decl_id = ast_node["referencedDeclaration"].get<int>();

    // Call through a function-pointer variable: the referencedDeclaration
    // points to a VariableDeclaration of FunctionTypeName, not to any
    // FunctionDefinition. No per-callee tuple_instance symbol exists for
    // these (we do not inline the target), so fail softly and let the
    // caller fall back to nondet-per-member over-approximation.
    const nlohmann::json &ref_node =
      find_node_by_id(src_ast_json["nodes"], ref_decl_id);
    if (
      ref_node.is_object() &&
      ref_node.value("nodeType", "") == "VariableDeclaration")
    {
      log_debug(
        "solidity",
        "get_tuple_function_ref: callee is a function-pointer variable "
        "(refDecl={}), falling back to nondet tuple",
        ref_decl_id);
      return true;
    }
  }
  else
  {
    log_error(
      "get_tuple_function_ref: unexpected nodeType '{}'",
      ast_node["nodeType"].get<std::string>());
    return true;
  }

  // The tuple instance was created when the callee function was processed.
  // Look it up using the callee's contract name as scope.
  // First try the current contract, then search all contracts.
  std::string c_name;
  get_current_contract_name(ast_node, c_name);

  std::string name = "tuple_instance$" + std::to_string(ref_decl_id);
  std::string id = "sol:@C@" + c_name + "@" + name;

  if (context.find_symbol(id) != nullptr)
  {
    new_expr = symbol_expr(*context.find_symbol(id));
    return false;
  }

  // For cross-contract calls, the tuple instance is in the callee's contract scope.
  // Search all contracts for the tuple instance.
  for (const auto &contract_name : contractNamesList)
  {
    std::string alt_id = "sol:@C@" + contract_name + "@" + name;
    if (context.find_symbol(alt_id) != nullptr)
    {
      new_expr = symbol_expr(*context.find_symbol(alt_id));
      return false;
    }
  }

  // Libraries / interfaces / abstract contracts are not in contractNamesList
  // but their tuple instances are still emitted under their own scope.
  for (const auto &lib_name : nonContractNamesList)
  {
    std::string alt_id = "sol:@C@" + lib_name + "@" + name;
    if (context.find_symbol(alt_id) != nullptr)
    {
      new_expr = symbol_expr(*context.find_symbol(alt_id));
      return false;
    }
  }

  // Free functions (outside any contract) have their tuple instance
  // stored under the synthetic "__free__" scope by
  // get_tuple_instance_name().
  {
    std::string free_id = "sol:@C@__free__@" + name;
    if (context.find_symbol(free_id) != nullptr)
    {
      new_expr = symbol_expr(*context.find_symbol(free_id));
      return false;
    }
  }

  // The callee's FunctionDefinition has not been processed yet (e.g. the
  // call site is reached during inline-method-population for an inheriting
  // contract before we walk the callee's owning contract). Materialise the
  // tuple struct + instance on demand so the lookup can proceed.
  const nlohmann::json &fn_def =
    find_node_by_id(src_ast_json["nodes"], ref_decl_id);
  if (
    fn_def.is_object() &&
    fn_def.value("nodeType", "") == "FunctionDefinition" &&
    fn_def.contains("returnParameters"))
  {
    typet rt;
    if (!get_type_description(fn_def["returnParameters"], rt) &&
        get_sol_type(rt) == SolidityGrammar::SolType::TUPLE_RETURNS)
    {
      exprt dump;
      if (!get_tuple_definition(fn_def) && !get_tuple_instance(fn_def, dump))
      {
        new_expr = dump;
        return false;
      }
    }
  }

  log_error("cannot find tuple instance for declaration id {}", ref_decl_id);
  return true;
}

// Knowing that there is a component x in the struct_tuple_instance A, we construct A.x
bool solidity_convertert::get_tuple_member_call(
  const irep_idt instance_id,
  const exprt &comp,
  exprt &new_expr)
{
  // tuple_instance
  assert(!instance_id.empty());
  exprt base;
  if (context.find_symbol(instance_id) == nullptr)
    return true;

  base = symbol_expr(*context.find_symbol(instance_id));
  new_expr = member_exprt(base, comp.name(), comp.type());
  return false;
}

void solidity_convertert::get_tuple_function_call(const exprt &op)
{
  assert(op.id() == "sideeffect");
  exprt func_call = op;
  convert_expression_to_code(func_call);
  if (current_functionDecl)
    move_to_back_block(func_call);
  else
    move_to_initializer(func_call);
}

void solidity_convertert::get_llc_ret_tuple(symbolt &s)
{
  log_debug("solidity", "\tconvert return value to tuple");
  std::string _id = lib_prefix + "sol_llc_ret";
  if (context.find_symbol(_id) == nullptr)
  {
    log_error("cannot find library symbol {}", _id);
    abort();
  }
  const symbolt &struct_sym = *context.find_symbol(_id);

  typet sym_t = struct_sym.type;
  set_sol_type(sym_t, SolidityGrammar::SolType::TUPLE_INSTANCE);

  std::string name, id;
  name = "tuple_instance$" + std::to_string(aux_counter);
  id = "sol:@" + name;
  locationt l;
  symbolt symbol;
  get_default_symbol(
    symbol, get_modulename_from_path(absolute_path), sym_t, name, id, l);
  symbol.static_lifetime = true;
  symbol.file_local = true;
  auto &added_sym = *move_symbol_to_context(symbol);

  // value
  typet t = struct_sym.type;
  exprt inits = gen_zero(t);
  // Cast nondet_bool to match the struct field type (C frontend compiles
  // _Bool/bool as unsigned int in struct layout due to padding)
  exprt bool_val = nondet_bool_expr;
  if (inits.op0().type() != nondet_bool_expr.type())
  {
    typecast_exprt cast(nondet_bool_expr, inits.op0().type());
    bool_val = cast;
  }
  inits.op0() = bool_val;
  inits.op1() = nondet_uint_expr;
  added_sym.value = inits;
  s = added_sym;
}

void solidity_convertert::get_string_assignment(
  const exprt &lhs,
  const exprt &rhs,
  exprt &new_expr)
{
  if (
    rhs.id() == "string-constant" ||
    (lhs.id() == "member" && lhs.component_name() == "_ESBMC_bind_cname"))
  {
    // todo: for immutable var, we can just use assign
    // char * this->bind_name = (const char *)_ESBMC_get_nondet_cont_name"
    // since we do not change the value of bind_name so we should be fine
    side_effect_exprt _assign("assign", lhs.type());
    exprt new_rhs = rhs;
    // pass a dump json as we will not reach bytes part anyway
    convert_type_expr(ns, new_rhs, lhs, empty_json);
    _assign.copy_to_operands(lhs, new_rhs);
    new_expr = _assign;
  }
  else
  {
    //? always assign it to null first?
    exprt null_str = gen_zero(pointer_typet(signed_char_type()));
    side_effect_exprt _assign("assign", lhs.type());
    _assign.location() = lhs.location();
    _assign.copy_to_operands(lhs, null_str);
    move_to_front_block(_assign);

    side_effect_expr_function_callt call;
    get_str_assign_function_call(lhs.location(), call);
    call.arguments().push_back(address_of_exprt(lhs));
    call.arguments().push_back(rhs);
    new_expr = call;
  }
}

/*
  lhs: code_blockt — each operand is a target expr or nil (omitted slot)
  rhs: tuple_return / tuple_instance — a struct symbol with mem0, mem1, ... components

  Uses explicit position-based matching: LHS position i maps to RHS component "mem{i}".
  This is robust regardless of which positions are omitted on either side.
*/
bool solidity_convertert::construct_tuple_assigments(
  const nlohmann::json &expr,
  const exprt &lhs,
  const exprt &rhs)
{
  log_debug("solidity", "Handling tuple assignment.");

  typet rt = rhs.type();
  SolidityGrammar::SolType rt_sol = get_sol_type(rt);

  assert(lhs.type().is_code() && to_code(lhs).statement() == "block");
  exprt new_rhs = rhs;
  bool rhs_is_nondet = false;

  // If the RHS is already a non-struct sideeffect (e.g. abi.decode(data,
  // (T,U,...)) lowered by the builtin path to a single-uint256 identity
  // call), we cannot split it into tuple members the normal way. Treat
  // it as a fully nondet tuple RHS — matches the over-approximation
  // documented in src/c2goto/library/solidity/solidity_abi.c and the
  // sibling fallback in the return-statement tuple path.
  if (
    rt_sol != SolidityGrammar::SolType::TUPLE_RETURNS &&
    !new_rhs.type().is_struct() && new_rhs.id() == "sideeffect")
  {
    rhs_is_nondet = true;
  }
  else if (rt_sol == SolidityGrammar::SolType::TUPLE_RETURNS)
  {
    // (x,y) = func();
    // => func() populates tuple instance; then extract members
    // The function call JSON is in "rightHandSide" (Assignment) or "initialValue" (VarDeclStmt)
    const nlohmann::json *rhs_call_json = nullptr;
    if (expr.contains("rightHandSide"))
      rhs_call_json = &expr["rightHandSide"];
    else if (expr.contains("initialValue"))
      rhs_call_json = &expr["initialValue"];

    // Conditional RHS: (x, y) = cond ? (a, b) : (c, d);
    // The rest of this block only understands FunctionCall-shaped RHS
    // (looking up the callee via `.expression`). For a Conditional whose
    // branches are both TupleExpressions we decompose element-wise into
    // per-slot ternaries before the function-call path can drop the
    // statement. Other Conditional shapes (e.g. function-call branches)
    // fall back to nondet-per-slot — sound over-approximation.
    if (
      rhs_call_json && rhs_call_json->is_object() &&
      rhs_call_json->value("nodeType", "") == "Conditional")
    {
      const nlohmann::json &cond_j = (*rhs_call_json)["condition"];
      const nlohmann::json &true_j = (*rhs_call_json)["trueExpression"];
      const nlohmann::json &false_j = (*rhs_call_json)["falseExpression"];

      const bool both_tuple_literals =
        true_j.is_object() && false_j.is_object() &&
        true_j.value("nodeType", "") == "TupleExpression" &&
        false_j.value("nodeType", "") == "TupleExpression" &&
        true_j.contains("components") && false_j.contains("components");

      if (both_tuple_literals)
      {
        exprt cond_expr;
        if (get_expr(cond_j, cond_expr))
          return true;

        const auto &t_comps = true_j["components"];
        const auto &f_comps = false_j["components"];

        for (size_t i = 0; i < lhs.operands().size(); ++i)
        {
          exprt lop = lhs.operands().at(i);
          if (lop.is_nil())
            continue;
          if (
            i >= t_comps.size() || i >= f_comps.size() ||
            t_comps[i].is_null() || f_comps[i].is_null())
          {
            log_error(
              "tuple conditional assignment: branch arity mismatch at slot {}",
              i);
            return true;
          }
          exprt t_val, f_val;
          if (get_expr(t_comps[i], t_val))
            return true;
          if (get_expr(f_comps[i], f_val))
            return true;

          // Align both branches to the LHS slot type before building the
          // ternary so the if_expr has a single well-defined type.
          convert_type_expr(ns, t_val, lop, empty_json);
          convert_type_expr(ns, f_val, lop, empty_json);

          exprt ternary("if", lop.type());
          ternary.copy_to_operands(cond_expr, t_val, f_val);

          get_tuple_assignment(expr, lop, ternary);
        }
        return false;
      }

      // Unhandled Conditional shape: fall through to nondet fallback.
      rhs_is_nondet = true;
    }
    else if (rhs_call_json && rhs_call_json->contains("expression"))
    {
      if (get_tuple_function_ref((*rhs_call_json)["expression"], new_rhs))
      {
        // Builtin tuple-producing callee (abi.decode(data, (T,U,...)),
        // addr.call(...), ...) has no per-function tuple instance.
        // Over-approximate each LHS slot as nondet — matches the
        // corresponding fallback in the return-statement tuple path.
        rhs_is_nondet = true;
      }
    }
    else
    {
      log_error("tuple assignment: cannot locate function call in RHS");
      return true;
    }

    if (!rhs_is_nondet)
      get_tuple_function_call(rhs);
  }

  if (rhs_is_nondet)
  {
    // No RHS struct to pull components from — synthesise an independent
    // nondet value for every non-nil LHS slot and return.
    for (const auto &lop : lhs.operands())
    {
      if (lop.is_nil())
        continue;
      exprt rop;
      get_nondet_expr(lop.type(), rop);
      get_tuple_assignment(expr, lop, rop);
    }
    return false;
  }

  // A struct-typed tuple RHS can arrive as an UNRESOLVED symbol type (`tag-struct C.Holder`)
  // rather than an inlined struct_typet -- this is what solc's synthesised public getter for a
  // `mapping(K => Struct)` state variable produces, e.g. `(uint a, uint b) = c.Acc(k);`.
  // `is_struct()` is false on the symbol type itself, so follow it through the namespace before
  // deciding, and keep the resolved type so the component lookup below sees the real struct.
  {
    const typet &followed = ns.follow(new_rhs.type());
    if (followed.is_struct())
      new_rhs.type() = followed;
    else if (
      new_rhs.id() == "dereference" && new_rhs.operands().size() == 1 &&
      new_rhs.op0().type().id() == "pointer")
    {
      // A `mapping(K => Struct)` slot now arrives as
      // `*(struct S *)map_fixed_arr_get(...)`; the dereference can reach here
      // carrying a nil type, so recover the struct from the pointer's subtype.
      const typet sub = ns.follow(new_rhs.op0().type().subtype());
      if (sub.is_struct())
        new_rhs.type() = sub;
    }
  }
  if (!new_rhs.type().is_struct())
  {
    log_error(
      "expecting struct type for tuple RHS, got id={} type_id={} followed_id={}",
      new_rhs.id().as_string(),
      new_rhs.type().id().as_string(),
      ns.follow(new_rhs.type()).id().as_string());
    return true;
  }

  // Build component lookup for the RHS struct.
  // For generated tuple structs, components are named "mem0", "mem1", etc.
  // For library structs (sol_llc_ret), components have their own names (x, y, ...).
  // We build both a name map and a positional list of non-padding components.
  const struct_typet &rhs_struct = to_struct_type(new_rhs.type());
  std::map<std::string, exprt> rhs_by_name;
  std::vector<exprt> rhs_by_pos; // non-padding components in order
  for (const auto &comp : rhs_struct.components())
  {
    std::string cname = comp.get_name().as_string();
    rhs_by_name[cname] = comp;
    // Skip padding components (anon_pad$N)
    if (cname.find("anon_pad") == std::string::npos)
      rhs_by_pos.push_back(comp);
  }

  // Match LHS targets to RHS components by position
  std::set<exprt> assigned_symbol;
  for (size_t i = 0; i < lhs.operands().size(); i++)
  {
    exprt lop = lhs.operands().at(i);
    if (lop.is_nil() || assigned_symbol.count(lop))
      continue;

    // Try positional name "mem{i}" first (generated tuple structs),
    // then fall back to positional index (library structs like sol_llc_ret).
    std::string mem_name = "mem" + std::to_string(i);
    exprt comp;
    auto it = rhs_by_name.find(mem_name);
    if (it != rhs_by_name.end())
      comp = it->second;
    else if (i < rhs_by_pos.size())
      comp = rhs_by_pos[i];
    else
    {
      log_error(
        "tuple assignment: cannot find RHS component for position {}", i);
      return true;
    }

    exprt rop;
    if (new_rhs.id() == "symbol")
    {
      if (get_tuple_member_call(new_rhs.identifier(), comp, rop))
        return true;
    }
    else
    {
      // The RHS is not a named tuple instance but an lvalue EXPRESSION -- this is
      // what solc's synthesised getter for `mapping(K => Struct)` lowers to now
      // that such a mapping is reached BY REFERENCE
      // (`*(struct S *)map_fixed_arr_get(&m, k, sizeof(S))`, see
      // solidity_convert_mapping.cpp). Take the component straight off it.
      rop = member_exprt(new_rhs, comp.get("name"), comp.type());
      rop.location() = new_rhs.location();
    }

    // Nested tuple: LHS operand is a code_blockt (inner tuple),
    // RHS component is a tuple struct — recursively unpack.
    if (
      lop.type().is_code() && lop.is_code() &&
      to_code(lop).statement() == "block" &&
      get_sol_type(rop.type()) == SolidityGrammar::SolType::TUPLE_INSTANCE)
    {
      // rop is a tuple instance symbol — recursively assign inner members
      if (construct_tuple_assigments(expr, lop, rop))
        return true;
      continue;
    }

    assigned_symbol.insert(lop);
    get_tuple_assignment(expr, lop, rop);
  }
  return false;
}

bool solidity_convertert::flatten_nested_tuple_assignment(
  const nlohmann::json &expr,
  const nlohmann::json &lhs_json,
  const nlohmann::json &rhs_json)
{
  // Flatten nested tuple assignments by walking LHS and RHS in parallel.
  // For each leaf LHS target, resolve the corresponding RHS value and assign.
  //
  // Example: ((a, b), c) = (getPair(), 30)
  //   LHS components: [TupleExpression(a,b), Identifier(c)]
  //   RHS components: [FunctionCall(getPair), Literal(30)]
  //   Result: call getPair() → a = tuple.mem0, b = tuple.mem1, c = 30

  assert(lhs_json.value("nodeType", "") == "TupleExpression");
  assert(lhs_json.contains("components"));

  // Strip redundant outer parens. `((a, b)) = (2, true)` parses as a
  // 1-component TupleExpression wrapping the real 2-tuple. Without unwrapping
  // we'd recurse into the leaf branch with a non-tuple RHS and crash on
  // rhs["components"]. Unwrap whenever a single-element tuple contains a
  // tuple (or until LHS/RHS arity matches).
  auto unwrap_paren_tuple = [](const nlohmann::json &n) -> const nlohmann::json & {
    const nlohmann::json *cur = &n;
    while (cur->is_object() && cur->value("nodeType", "") == "TupleExpression" &&
           cur->contains("components") && (*cur)["components"].is_array() &&
           (*cur)["components"].size() == 1 && !(*cur)["components"][0].is_null() &&
           (*cur)["components"][0].is_object() &&
           (*cur)["components"][0].value("nodeType", "") == "TupleExpression")
      cur = &(*cur)["components"][0];
    return *cur;
  };
  const nlohmann::json &lhs_unwrapped = unwrap_paren_tuple(lhs_json);
  const nlohmann::json &rhs_unwrapped =
    rhs_json.is_object() ? unwrap_paren_tuple(rhs_json) : rhs_json;

  const auto &lhs_comps = lhs_unwrapped["components"];
  if (
    !rhs_unwrapped.is_object() ||
    rhs_unwrapped.value("nodeType", "") != "TupleExpression" ||
    !rhs_unwrapped.contains("components"))
  {
    log_error("nested tuple: RHS is not a TupleExpression after unwrapping");
    return true;
  }
  const auto &rhs_comps = rhs_unwrapped["components"];

  for (size_t i = 0; i < lhs_comps.size(); i++)
  {
    if (lhs_comps[i].is_null())
      continue; // omitted slot

    if (i >= rhs_comps.size())
    {
      log_error("nested tuple: LHS has more components than RHS");
      return true;
    }

    if (lhs_comps[i].value("nodeType", "") == "TupleExpression")
    {
      // Nested LHS: ((a, b), ...) — RHS must be a tuple-returning function call
      const auto &rhs_val = rhs_comps[i];
      typet rhs_t;
      if (get_type_description(rhs_val["typeDescriptions"], rhs_t))
        return true;

      // Only treat the RHS as a tuple-returning function call when it
      // actually *is* a FunctionCall node. A tuple LITERAL (e.g.
      // `((1, 2), 3)`) has typeString `tuple(...)` too, which would
      // otherwise misclassify as TUPLE_RETURNS and trip on the absent
      // `.expression` field inside get_tuple_function_ref.
      const bool rhs_is_fn_call =
        rhs_val.is_object() &&
        rhs_val.value("nodeType", "") == "FunctionCall";
      if (
        rhs_is_fn_call &&
        get_sol_type(rhs_t) == SolidityGrammar::SolType::TUPLE_RETURNS)
      {
        // RHS is a function call returning tuple.
        // 1. Call the function (populates its tuple instance)
        exprt func_call;
        if (get_expr(rhs_val, rhs_val["typeDescriptions"], func_call))
          return true;
        get_tuple_function_call(func_call);

        // 2. Find the tuple instance for this function
        assert(rhs_val.contains("expression"));
        exprt tuple_inst;
        if (get_tuple_function_ref(rhs_val["expression"], tuple_inst))
          return true;

        // 3. Assign inner LHS targets from the tuple instance members
        const struct_typet &inner_struct = to_struct_type(tuple_inst.type());
        const auto &inner_lhs_comps = lhs_comps[i]["components"];
        size_t mem_idx = 0;
        for (size_t j = 0; j < inner_lhs_comps.size(); j++)
        {
          if (inner_lhs_comps[j].is_null())
          {
            mem_idx++;
            continue;
          }

          // Find component by name "mem{j}"
          std::string mem_name = "mem" + std::to_string(j);
          bool found = false;
          for (const auto &comp : inner_struct.components())
          {
            if (comp.get_name().as_string() == mem_name)
            {
              exprt target;
              if (get_expr(
                    inner_lhs_comps[j],
                    inner_lhs_comps[j]["typeDescriptions"],
                    target))
                return true;

              exprt member;
              if (get_tuple_member_call(
                    tuple_inst.identifier(), comp, member))
                return true;

              get_tuple_assignment(expr, target, member);
              found = true;
              break;
            }
          }
          if (!found)
          {
            log_error(
              "nested tuple: cannot find inner component '{}'", mem_name);
            return true;
          }
          mem_idx++;
        }
      }
      else
      {
        // Nested LHS but RHS is a tuple literal — recurse
        if (flatten_nested_tuple_assignment(expr, lhs_comps[i], rhs_comps[i]))
          return true;
      }
    }
    else
    {
      // Leaf LHS target — direct assignment
      exprt target;
      if (get_expr(
            lhs_comps[i], lhs_comps[i]["typeDescriptions"], target))
        return true;

      exprt value;
      if (get_expr(
            rhs_comps[i], rhs_comps[i]["typeDescriptions"], value))
        return true;

      get_tuple_assignment(expr, target, value);
    }
  }

  return false;
}

void solidity_convertert::get_tuple_assignment(
  const nlohmann::json &expr,
  const exprt &lop,
  exprt rop)
{
  // When the LHS is `bytes memory` but the RHS is not bytes (e.g. the `y`
  // component of sol_llc_ret, which is a plain uint placeholder), substitute a
  // fresh nondet BytesDynamic so that `data.length` has the correct type and a
  // fully unconstrained symbolic length value.
  // We use get_nondet_expr with lop.type() to avoid any sort mismatch that
  // would arise from casting the uint placeholder to BytesDynamic.
  if (is_bytes_type(lop.type()) && !is_bytes_type(rop.type()))
  {
    get_nondet_expr(lop.type(), rop);
  }

  exprt assign_expr;
  if (get_sol_type(lop.type()) == SolidityGrammar::SolType::STRING)
    get_string_assignment(lop, rop, assign_expr);
  else
  {
    assign_expr = side_effect_exprt("assign", lop.type());
    convert_type_expr(ns, rop, lop, expr);
    assign_expr.copy_to_operands(lop, rop);
  }
  convert_expression_to_code(assign_expr);
  if (current_functionDecl)
    move_to_back_block(assign_expr);
  else
    move_to_initializer(assign_expr);
}
