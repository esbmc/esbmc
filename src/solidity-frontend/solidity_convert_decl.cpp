/// \file solidity_convert_decl.cpp
/// \brief Declaration conversion for the Solidity frontend.
///
/// Converts Solidity variable declarations (state variables, local variables,
/// parameters), non-contract top-level definitions (enums, structs, free
/// functions, using-for directives), and struct/enum type definitions into
/// ESBMC's symbol table and irep2 representation.

#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith/arith_tools.h>
#include <util/arith/bitvector.h>
#include <util/lang/c_types.h>
#include <irep2/irep2_utils.h>
#include <util/expr/expr_util.h>
#include <util/base/i2string.h>
#include <util/arith/mp_arith.h>
#include <util/irep/std_expr.h>
#include <util/message/message.h>
#include <fstream>

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
    return get_struct_class(ast_node); // rule struct-definition
  }
  case SolidityGrammar::ContractBodyElementT::ModifierDef:
  case SolidityGrammar::ContractBodyElementT::FunctionDef:
  case SolidityGrammar::ContractBodyElementT::EnumDef:
  case SolidityGrammar::ContractBodyElementT::ErrorDef:
  case SolidityGrammar::ContractBodyElementT::EventDef:
  case SolidityGrammar::ContractBodyElementT::UsingForDef:
  case SolidityGrammar::ContractBodyElementT::UserDefinedValueTypeDef:
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

// ---------------------------------------------------------------------------
// AST-level facts that only the AST can supply, recorded on the function symbol
// for later passes. Both come from ONE walk of the function body.
//
// (1) `this.f(...)` call sites.  On a real EVM this is an EXTERNAL call: it
//     re-enters through the ABI, so inside `f` the value of `msg.sender` is the
//     CONTRACT'S OWN ADDRESS, not the original caller. The frontend lowers it
//     to the same direct call as a plain `f(a)` (measured), which keeps the
//     caller's msg.sender — so the model is not merely missing a branch, it
//     disagrees with the EVM. A `require(msg.sender == owner)` inside `f` fails
//     on-chain when reached via `this.f()` and can pass in the model. Any test
//     generated from such a path is red when run, while being labelled
//     certified. Recording the site here lets the test emitter declare a NAMED
//     OBSTACLE and refuse to emit, instead of emitting something wrong.
//     Detection is syntactically trivial and must not wait for the emitter:
//     otherwise the requirement lives only in prose and is lost at the next
//     handover.
//
// (2) The number of VALUE-RETURNING `return` statements written in the source.
//     The path enumeration works on the goto program; a source-level exit that
//     the frontend dropped before goto conversion is invisible there, and this
//     count is the only independent witness that such an exit existed.
//
//     Only value-returning returns are counted, and that restriction is load
//     bearing: a bare `return;` lowers to a plain jump to END_FUNCTION and
//     produces no RETURN instruction at all, so counting it made the consumer
//     hard-fail on a correct program (measured on a function whose only exit
//     statement is `return;`). Undercounting is the safe direction here — the
//     check only ever asserts "if the source returns values, some enumerated
//     path must end at a RETURN", which is what the value-returning case
//     guarantees.
static void collect_path_cov_ast_facts(
  const nlohmann::json &node,
  unsigned &return_sites,
  std::vector<std::string> &this_call_names)
{
  if (node.is_array())
  {
    for (const auto &e : node)
      collect_path_cov_ast_facts(e, return_sites, this_call_names);
    return;
  }
  if (!node.is_object())
    return;

  if (node.contains("nodeType") && node["nodeType"].is_string())
  {
    const std::string nt = node["nodeType"].get<std::string>();
    if (nt == "Return")
    {
      if (node.contains("expression") && !node["expression"].is_null())
        ++return_sites;
    }
    // this.f(...) == FunctionCall( MemberAccess( Identifier "this", "f" ) )
    else if (nt == "FunctionCall" && node.contains("expression"))
    {
      const nlohmann::json &callee = node["expression"];
      if (
        callee.is_object() && callee.value("nodeType", "") == "MemberAccess" &&
        callee.contains("expression"))
      {
        const nlohmann::json &base = callee["expression"];
        if (
          base.is_object() && base.value("nodeType", "") == "Identifier" &&
          base.value("name", "") == "this")
          this_call_names.push_back(callee.value("memberName", "?"));
      }
    }
  }

  for (auto it = node.begin(); it != node.end(); ++it)
    if (it.value().is_object() || it.value().is_array())
      collect_path_cov_ast_facts(it.value(), return_sites, this_call_names);
}

void solidity_convertert::stamp_path_cov_ast_facts(
  const nlohmann::json &ast_node)
{
  if (!ast_node.contains("body") || ast_node["body"].is_null())
    return;

  std::string name, id;
  get_function_definition_name(ast_node, name, id);
  symbolt *sym = context.find_symbol(id);
  if (sym == nullptr)
    return;

  unsigned return_sites = 0;
  std::vector<std::string> this_calls;
  collect_path_cov_ast_facts(ast_node["body"], return_sites, this_calls);

  // Only a function with EXACTLY ONE return parameter is guaranteed to lower a
  // value-returning `return` to a RETURN instruction. With two or more, the
  // frontend routes the values into the function's tuple_instance and emits a
  // BARE `return;` — which lowers to a plain jump to END_FUNCTION, exactly like
  // a valueless source return. The exits are still enumerated (at
  // END_FUNCTION); only the instruction kind differs.
  //
  // Measured: this misfired on the real benchmark `Aqua::rawBalances` and was
  // reproduced on a two-line contract (a `returns (uint256, uint256)` function
  // trips it, the same body with one return parameter does not). Reporting zero
  // for the multi-return case keeps the count on the safe side — the consumer's
  // rule is "if the source returns values then some path must end at a RETURN",
  // and undercounting only ever loses a check, never invents one.
  unsigned ret_params = 0;
  if (
    ast_node.contains("returnParameters") &&
    ast_node["returnParameters"].contains("parameters"))
    ret_params = (unsigned)ast_node["returnParameters"]["parameters"].size();
  if (ret_params != 1)
    return_sites = 0;

  {
    typet t = sym->get_type();
    t.set("#sol_ast_return_sites", std::to_string(return_sites));
    sym->set_type(std::move(t));
  }
  if (!this_calls.empty())
  {
    std::string names;
    for (const auto &n : this_calls)
      names += (names.empty() ? "" : ";") + n;
    {
      typet t = sym->get_type();
      t.set("#sol_this_call_count", std::to_string(this_calls.size()));
      sym->set_type(std::move(t));
    }
    {
      typet t = sym->get_type();
      t.set("#sol_this_call_names", names);
      sym->set_type(std::move(t));
    }
  }
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
    if (get_function_definition(ast_node)) // rule function-definition
      return true;
    // Record the AST-only facts on the symbol just created. Done here rather
    // than inside get_function_definition so the walk sees the ORIGINAL body
    // AST, before any lowering can drop a source-level exit.
    stamp_path_cov_ast_facts(ast_node);
    return false;
  }
  case SolidityGrammar::ContractBodyElementT::VarDecl:
  case SolidityGrammar::ContractBodyElementT::StructDef:
  case SolidityGrammar::ContractBodyElementT::EnumDef:
  case SolidityGrammar::ContractBodyElementT::ErrorDef:
  case SolidityGrammar::ContractBodyElementT::EventDef:
  case SolidityGrammar::ContractBodyElementT::UsingForDef:
  case SolidityGrammar::ContractBodyElementT::ModifierDef:
  case SolidityGrammar::ContractBodyElementT::UserDefinedValueTypeDef:
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
  // For consistensy, we use ["typeName"]["typeDescriptions"] as in
  // state-variable-declaration to improve the re-usability of get_type*
  // function, when dealing with non-array var decls. For array, do NOT use
  // ["typeName"]. Otherwise, it will cause problem when populating typet in
  // get_cast

  if (
    get_type_description(ast_node, ast_node["typeName"]["typeDescriptions"], t))
    return true;

  bool is_contract =
    get_sol_type(t) == SolidityGrammar::SolType::CONTRACT ? true : false;
  bool is_mapping =
    get_sol_type(t) == SolidityGrammar::SolType::MAPPING ? true : false;
  bool is_mapping_array = get_sol_mapping_array(t);
  bool is_new_expr = should_treat_as_new(current_contractName);
  bool is_byte_static = is_bytesN_type(t);
  // Detect state-var dynamic arrays: model as infinite SMT array + length var.
  //
  // T1.1 Stage S1: dropped the previous `!is_new_expr` gate.  Before, only
  // singleton/unbound contracts used the SMT-array model; `new`'d contracts
  // fell through to a heap-malloc path whose `_ESBMC_array_push` memcpy
  // truncated under low --unwind, killing every post-push verification path.
  // Unifying on the SMT-array model removes that hazard.  Per-instance
  // length isolation is added in the same stage by addr-keying the length
  // companion (see the `_dynarray_len` symbol creation block below).
  bool is_state_var_check =
    ast_node.contains("stateVariable") && ast_node["stateVariable"].get<bool>();
  bool is_dynarray_state =
    get_sol_type(t) == SolidityGrammar::SolType::DYNARRAY &&
    is_state_var_check && !get_sol_mapping_array(t);

  // for mapping: populate the element type (recursively for nested mappings)
  bool is_mapping_of_dynarr = false;
  if (is_mapping && !is_new_expr)
  {
    assert(t.is_array());
    // Walk the valueType chain to handle mapping(K1 => mapping(K2 => ... => V))
    typet *cur_type = &t;
    const nlohmann::json *cur_node = &ast_node["typeName"];
    while (true)
    {
      const auto &val_json = (*cur_node)["valueType"];
      typet val_t;
      if (get_type_description(val_json["typeDescriptions"], val_t))
        return true;
      cur_type->subtype() = val_t;

      // If inner value is also a mapping, continue recursion
      if (
        get_sol_type(val_t) == SolidityGrammar::SolType::MAPPING &&
        val_t.is_array())
      {
        cur_type = &cur_type->subtype();
        cur_node = &val_json;
      }
      else
        break;
    }

    // mapping(K => T[N]) and mapping(K => T[M][N]) with fixed-array
    // leaf value: force `mapping_t` even under the !is_new_expr
    // "static singleton" optimisation. The chained-subtype lowering
    // `array<T[N], inf>` / `array<T[M][N], inf>` produces an
    // unbounded array whose element sort is itself an array sort —
    // array_convt cannot represent that (see
    // src/solvers/smt/array_conv.cpp:92-95, bitwuzla reports "terms with
    // mismatching sort at indices 0 and 1"). Routing this mapping through the
    // mapping_t + map_fixed_arr_get helper sidesteps the nested-array sort
    // entirely. Restrict to single-level mappings: nested
    // `mapping(K1=>mapping(K2=>T[N]))` would also need access-layer
    // combine_mapping_keys_256 gating on the rewritten type, which is not in
    // scope here. 1D fixed array leaves carry SolType::ARRAY_LITERAL; 2D+
    // nested fixed arrays (e.g. T[M][N]) carry SolType::ARRAY — accept both.
    SolidityGrammar::SolType leaf_sol = get_sol_type(cur_type->subtype());
    if (
      cur_type == &t && cur_type->is_array() &&
      (leaf_sol == SolidityGrammar::SolType::ARRAY_LITERAL ||
       leaf_sol == SolidityGrammar::SolType::ARRAY))
    {
      // Try the per-mapping flat-array encoder first.  Replaces the slow
      // mapping_t + map_fixed_arr_get helper path with a leaf-typed
      // `array<T, infinity>` indexed by `(key << inner_bits) | inner_offset`.
      // Sound under three AST-detectable conditions (see header):
      //   1. !is_new_expr (already gated by enclosing `if`)
      //   2. no `T[N] storage ref = m[k]` storage-pointer alias
      //   3. all mapping accesses reach the scalar leaf
      // Failure to satisfy any condition falls back to the slow path so
      // existing behavior is preserved bit-for-bit for unsupported shapes.
      bool flat_encoded = false;
      unsigned long inner_extent = 0;
      typet leaf_t;
      if (
        ast_node.contains("id") &&
        compute_flat_extent(cur_type->subtype(), inner_extent, leaf_t))
      {
        const int map_decl_id = ast_node["id"].get<int>();
        const unsigned expected_depth =
          1 + array_nesting_depth(cur_type->subtype());
        if (
          !has_mapping_storage_ref(map_decl_id, current_contractName) &&
          !has_partial_mapping_access(
            map_decl_id, expected_depth, current_contractName))
        {
          // ceil_log2(inner_extent) — bits for the inner offset slot.
          unsigned inner_bits = 0;
          for (unsigned long v = inner_extent - 1; v != 0; v >>= 1)
            ++inner_bits;
          if (inner_extent == 1)
            inner_bits =
              1; // single-element fixed array — still allow a 1-bit slot
          // Per the existing scalar fast path (solidity_convert_type.cpp:611)
          // the key portion is widened to 256 bits.
          const unsigned index_width = 256 + inner_bits;

          t = array_typet(leaf_t, exprt("infinity"));
          set_sol_type(t, SolidityGrammar::SolType::MAPPING);
          t.set("#esbmc_index_width", std::to_string(index_width));
          t.set("#sol_mapping_flat_encoded", true);
          // Intentionally do NOT set #sol_mapping_fixed_arr_value: the
          // mapping-init block at line 956 reads that tag to dispatch to
          // build_mapping_t_init_value (which constructs the slow
          // {base, mid, addr} struct).  The flat encoder's type is a
          // plain array_typet — the existing scalar-mapping init flow
          // (default zero-init via __ESBMC_inf_size) is what we want.
          t.set("#sol_flat_inner_extent", std::to_string(inner_extent));
          t.set("#sol_flat_inner_bits", std::to_string(inner_bits));
          flat_encoded = true;
        }
      }
      if (!flat_encoded)
      {
        t = symbol_typet(lib_prefix + "mapping_t");
        set_sol_type(t, SolidityGrammar::SolType::MAPPING);
        t.set("#sol_mapping_fixed_arr_value", true);
      }
    }

    /* mapping(K => V[]) state-var: promote the leaf DYNARRAY value from
     * the default pointer model to a nested infinite SMT array, keyed
     * by the folded 64-bit key. With the pointer model, push(x) called
     * `_ESBMC_array_push_uint256` which fresh-mallocs a new slab and
     * can't carry prior pushes.
     *
     * Shape: `array_typet(array_typet(elem, inf), inf)`. Per-key length
     * is tracked by a sibling aux `<name>_mapdynarr_len` of type
     * `array_typet(uint256, inf)` (created below alongside the symbol).
     *
     * Only promote when the leaf element is a scalar (uint/int/address/
     * bool/bytesN). Non-scalar elements (struct, string, nested array)
     * stay on the pointer model — they need a different element copy
     * protocol that's out of scope for this pass. */
    if (
      cur_type->is_array() && cur_type->subtype().is_pointer() &&
      get_sol_type(cur_type->subtype()) == SolidityGrammar::SolType::DYNARRAY)
    {
      const typet &elem_t = cur_type->subtype().subtype();
      SolidityGrammar::SolType elem_sol = get_sol_type(elem_t);
      bool elem_is_scalar = SolidityGrammar::is_uint_type(elem_sol) ||
                            SolidityGrammar::is_int_type(elem_sol) ||
                            SolidityGrammar::is_address_type(elem_sol) ||
                            elem_sol == SolidityGrammar::SolType::BOOL ||
                            elem_sol == SolidityGrammar::SolType::ENUM ||
                            SolidityGrammar::is_bytes_type(elem_sol);
      if (elem_is_scalar)
      {
        typet inner_inf = array_typet(elem_t, exprt("infinity"));
        set_sol_type(inner_inf, SolidityGrammar::SolType::DYNARRAY);
        inner_inf.set("#sol_dynarr_inner", true);
        cur_type->subtype() = inner_inf;
        t.set("#sol_mapping_of_dynarr", true);
        is_mapping_of_dynarr = true;
      }
    }
  }

  // for mapping arrays: populate the inner mapping's subtype chain
  // mapping(K=>V)[] is modeled as array[inf] of (array[inf] of V)
  if (is_mapping_array && !is_new_expr)
  {
    assert(t.is_array() && t.subtype().is_array());
    // The AST has typeName.baseType pointing to the mapping
    const nlohmann::json &map_node = ast_node["typeName"]["baseType"];
    typet *cur_type = &t.subtype();
    const nlohmann::json *cur_node = &map_node;
    while (true)
    {
      const auto &val_json = (*cur_node)["valueType"];
      typet val_t;
      if (get_type_description(val_json["typeDescriptions"], val_t))
        return true;
      cur_type->subtype() = val_t;

      if (
        get_sol_type(val_t) == SolidityGrammar::SolType::MAPPING &&
        val_t.is_array())
      {
        cur_type = &cur_type->subtype();
        cur_node = &val_json;
      }
      else
        break;
    }
  }

  // For dynarray state vars: change type from pointer to infinite array
  if (is_dynarray_state)
  {
    assert(t.is_pointer());
    typet elem_type = t.subtype();
    // If the element is itself a pointer-backed fixed-size array (T[N]),
    // promote it to a native SMT array_typet(T, N). Without this, the
    // outer `array<pointer<T>, inf>` slot is pointer-sorted while the
    // push emitter's zero-initializer and the read path use ARRAY sort
    // for the literal `{0,...,0}` — the sort divergence blows up
    // cvc5_convt::mk_ite inside array_convt::execute_array_ite. Promoting
    // the inner slot to native array aligns both sides with a single
    // ARRAY sort. Standalone T[N] state-vars (not nested under a dynamic
    // outer) still use the pointer model; unifying those is a broader
    // refactor tracked separately.
    if (elem_type.is_pointer() && !elem_type.get("#sol_array_size").empty())
    {
      const std::string sz_str = elem_type.get("#sol_array_size").as_string();
      BigInt sz = string2integer(sz_str);
      typet leaf = elem_type.subtype();
      typet inner_arr = array_typet(
        leaf,
        constant_exprt(
          integer2binary(sz, bv_width(int_type())),
          integer2string(sz),
          int_type()));
      inner_arr.set("#sol_array_size", sz_str);
      set_sol_type(inner_arr, SolidityGrammar::SolType::ARRAY);
      elem_type = inner_arr;
    }
    t = array_typet(elem_type, exprt("infinity"));
    set_sol_type(t, SolidityGrammar::SolType::DYNARRAY);
    set_sol_dynarray_state(t, true);
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

  // For local storage reference variables (e.g. Wrapper storage ref = param),
  // register an alias so that uses of 'ref' resolve to the source symbol.
  bool is_storage_ref_alias = false;
  if (
    !is_state_var && ast_node.contains("storageLocation") &&
    ast_node["storageLocation"] == "storage" && !initialValue.empty())
  {
    int this_id = ast_node["id"].get<int>();
    // The simple int-id alias path (`storage_ref_aliases`) only works when
    // the RHS is itself an Identifier — then we can resolve the alias by
    // chasing referencedDeclaration to another VariableDeclaration. For a
    // compound expression like `self.data` (MemberAccess), the RHS *also*
    // carries a referencedDeclaration (pointing at the struct field), but
    // following that id loses the `self.` base and yields a bare symbol
    // referencing the struct field as if it were a free variable — which
    // then crashes goto-symex with `phi_function: no symbol`. Always take
    // the expression-alias path for non-Identifier RHSes.
    const bool init_is_identifier = initialValue.contains("nodeType") &&
                                    initialValue["nodeType"] == "Identifier";
    if (
      init_is_identifier && initialValue.contains("referencedDeclaration") &&
      get_sol_type(t) == SolidityGrammar::SolType::STRUCT)
    {
      int src_id = initialValue["referencedDeclaration"].get<int>();
      storage_ref_aliases[this_id] = src_id;
      is_storage_ref_alias = true;
      log_debug(
        "solidity",
        "@@@ storage alias: {} -> {} (local storage ref)",
        this_id,
        src_id);
    }
    else
    {
      storage_ref_expr_aliases[this_id] = initialValue;
      is_storage_ref_alias = true;
      log_debug(
        "solidity",
        "@@@ storage expr alias: {} (local storage ref to expression)",
        this_id);
    }
  }

  bool is_inherited = ast_node.contains("is_inherited");

  // 2. populate id and name
  std::string name, id;
  // TODO: Omitted variable
  if (ast_node["name"].get<std::string>().empty())
    // Omitted variable
    get_aux_var(name, id);
  else
  {
    if (get_var_decl_name(ast_node, name, id))
      return true;
  }

  // The state flag is keyed by symbol id, so it is recorded here rather than
  // beside the type: the id is not known until now (§10).
  set_sol_state_var(id, is_state_var);

  // if we have already populated the var symbol, we do not need to re-parse
  // however, we still need to return a code_declt so callers in statement
  // contexts (VariableDeclStatement, parameter lists) end up with a proper
  // declaration node — returning a bare symbol_exprt leaks an OTHER
  // instruction with a raw symbol payload into goto-symex, which then trips
  // `goto_symext: unexpected statement: symbol`.
  if (context.find_symbol(id) != nullptr)
  {
    log_debug("solidity", "Found parsed symbol, skip parsing");
    new_expr = code_declt(symbol_expr(*context.find_symbol(id)));
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
  // static_lifetime: this means it's defined in the file level, not inside
  // contract special case for mapping, even if it's inside a contract
  symbol.static_lifetime = current_contractName.empty() ||
                           (is_mapping && !is_new_expr) ||
                           (is_mapping_array && !is_new_expr) ||
                           is_dynarray_state || (is_library && is_constant);
  symbol.file_local = true;
  symbol.is_extern = false;

  // For state var decl, we look for "value".
  // For local var decl, we look for "initialValue"
  bool has_init = (ast_node.contains("value") || !initialValue.empty());
  // For inherited STATE variables the initial value is set in
  // "move_inheritance_to_ctor()" (D.x = B.x), so the copied json node's own
  // init is deliberately skipped here.
  //
  // `&& is_state_var` IS LOAD BEARING, and its absence was a silent
  // miscompile of every inherited function body. This same routine serves
  // "rule variable-declaration-statement" (see the header comment above), i.e.
  // LOCALS — and `add_inherit_label` stamps `is_inherited` on EVERY sub-node
  // carrying an `id`, which includes every local declaration inside an
  // inherited function. Without the state-var guard, `uint256 y = x + 1;` in a
  // base function became `y = 0` in the derived contract's copy, and the
  // derived copy is the one the dispatcher calls. There is no
  // move_inheritance_to_ctor step for a local to be deferred to, so the value
  // was not moved — it was lost.
  //
  // MEASURED, 20 lines of Solidity (notes/repro/immutable_clamp_*.sol and the
  // st1inch benchmark): `sol:@C@B4@F@f#17` has `ASSIGN y = x + 1` while
  // `sol:@C@D4@F@f#17` had `ASSIGN y = 0`. On st1inch this zeroed
  // `t = timestamp - ORIGIN` inside the inherited `_votingPowerAt`, which made
  // all thirty `if (t & bit)` guards constant-false, made the constructor's own
  // sanity check revert unconditionally, and left the whole contract
  // unreachable: `Generated 0 VCC(s)` with every path reported undecided.
  bool set_init =
    has_init && !(is_inherited && is_state_var) && !is_storage_ref_alias;
  const nlohmann::json init_value =
    ast_node.contains("value") ? ast_node["value"] : initialValue;
  const nlohmann::json literal_type = ast_node["typeDescriptions"];
  if (!set_init && !(is_mapping && is_new_expr && is_byte_static))
  {
    // for both state and non-state variables, set default value as zero
    {
      exprt _v = gen_zero(get_complete_type(t, ns), true);
      _v.zero_initializer(true);
      symbol.set_value(std::move(_v));
    }
  }

  // 6. add symbol into the context
  // just like clang-c-frontend, we have to add the symbol before converting the
  // initial assignment
  symbolt &added_symbol = *move_symbol_to_context(symbol);
  code_declt decl(symbol_expr(added_symbol));

  // 6b. for mapping arrays, create auxiliary _length variable
  if (is_mapping_array && !is_new_expr)
  {
    std::string len_name = name + "_mapping_arr_len";
    std::string len_id = id + "_mapping_arr_len";
    symbolt len_sym;
    get_default_symbol(
      len_sym,
      debug_modulename,
      unsignedbv_typet(256),
      len_name,
      len_id,
      location_begin);
    len_sym.lvalue = true;
    len_sym.static_lifetime = true;
    len_sym.file_local = true;
    len_sym.is_extern = false;
    {
      exprt _v = gen_zero(unsignedbv_typet(256));
      _v.zero_initializer(true);
      len_sym.set_value(std::move(_v));
    }
    move_symbol_to_context(len_sym);
  }

  // 6c. for dynarray state vars, create auxiliary _dynarray_len variable.
  //
  // T1.1 Stage S1: the length companion is now an addr-keyed infinite array
  // (`array_typet(uint256, infinity)`) instead of a global scalar.  Per-
  // instance length is read/written as `<arr>_dynarray_len[this->$address]`
  // via the get_dynarr_len_ref helper.  Two `new C()` instances no longer
  // alias on length.
  if (is_dynarray_state)
  {
    std::string len_name = name + "_dynarray_len";
    std::string len_id = id + "_dynarray_len";
    typet len_arr_t = array_typet(unsignedbv_typet(256), exprt("infinity"));
    symbolt len_sym;
    get_default_symbol(
      len_sym, debug_modulename, len_arr_t, len_name, len_id, location_begin);
    len_sym.lvalue = true;
    len_sym.static_lifetime = true;
    len_sym.file_local = true;
    len_sym.is_extern = false;
    {
      exprt _v = gen_zero(get_complete_type(len_arr_t, ns), true);
      _v.zero_initializer(true);
      len_sym.set_value(std::move(_v));
    }
    move_symbol_to_context(len_sym);

    // T1.1 Stage S3: register this state-var dyn-array so the clone
    // helper can emit per-instance length+element copy.  Keyed by the
    // current contract; inherited dyn-arrays land under the derived
    // contract on its decl pass (merge_inheritance_ast already
    // duplicates the AST node).
    if (!current_contractName.empty())
      dynarray_state_vars[current_contractName].emplace_back(id, t.subtype());
  }

  // 6d. for mapping(K => V[]) state-var: create auxiliary _mapdynarr_len
  // infinite array keyed by the folded 64-bit mapping key. Mirrors
  // 6c, but the length is per-key rather than a single counter.
  if (is_mapping_of_dynarr)
  {
    std::string len_name = name + "_mapdynarr_len";
    std::string len_id = id + "_mapdynarr_len";
    typet len_arr_t = array_typet(unsignedbv_typet(256), exprt("infinity"));
    symbolt len_sym;
    get_default_symbol(
      len_sym, debug_modulename, len_arr_t, len_name, len_id, location_begin);
    len_sym.lvalue = true;
    len_sym.static_lifetime = true;
    len_sym.file_local = true;
    len_sym.is_extern = false;
    {
      exprt _v = gen_zero(get_complete_type(len_arr_t, ns), true);
      _v.zero_initializer(true);
      len_sym.set_value(std::move(_v));
    }
    move_symbol_to_context(len_sym);
  }

  // 7. populate init value if there is any
  // special handling for array/dynarray
  SolidityGrammar::SolType t_sol_type = get_sol_type(t);

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
  // Native nested multi-dim fixed arrays (B2): the type is an
  // embedded `array_typet(array_typet(..))` on the contract struct,
  // zero-initialised by the struct's default construction — no calloc
  // / arrcpy needed. Literal initialisers on such fields are
  // uncommon in Solidity; if/when we support them, they go here.
  bool is_native_nested_array =
    (t_sol_type == SolidityGrammar::SolType::ARRAY) && t.is_array() &&
    t.has_subtype() && t.subtype().is_array();

  if (is_native_nested_array)
  {
    // Nothing to emit at decl time. Fall through to the post-decl
    // block below (which handles other state-var housekeeping).
  }
  else if (
    t_sol_type == SolidityGrammar::SolType::ARRAY ||
    t_sol_type == SolidityGrammar::SolType::ARRAY_LITERAL)
  {
    /**
      uint[2] z;            // uint *z = (uint *)calloc(2, sizeof(uint));

                            // uint tmp1[2] = {1,2}; // populated into sym tab,
    not a real statement uint[2] zz = [1,2];   // uint *zz = (uint
    *)_ESBMC_arrcpy(tmp1, 2, 2, sizeof(uint));

      uint[2] y = x;        // uint *zz = (uint *)_ESBMC_arrcpy(x, 2, 2,
    sizeof(uint));

      TODO: suport disorder:
      uint[2] y = x;
      uint[2] x = [1,2];
    **/

    // get size
    std::string arr_size = "0";
    if (has_sol_array_size(t))
      arr_size = get_sol_array_size(t);
    else if (t.has_subtype() && has_sol_array_size(t.subtype()))
      arr_size = get_sol_array_size(t.subtype());
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
      set_sol_array_size(acpy_call.type(), arr_size);
      // set as rvalue
      added_symbol.set_value(migrate_expr(acpy_call));
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
      added_symbol.set_value(migrate_expr(calc_call));
      decl.operands().push_back(calc_call);
    }
  }
  else if (is_dynarray_state && set_init)
  {
    // Dynarray state var with init: just set the length variable.
    // Elements are zero by default in the infinite SMT array.
    // For `new uint[](n)`: set length = n
    // For literal init like `= [1,2,3]`: handled in assignment expression
    if (
      init_value.contains("nodeType") &&
      init_value["nodeType"] == "FunctionCall" &&
      init_value.contains("arguments") && init_value["arguments"].size() > 0)
    {
      nlohmann::json callee_arg_json = init_value["arguments"][0];
      exprt size_expr;
      const nlohmann::json lit_type = callee_arg_json["typeDescriptions"];
      if (get_expr(callee_arg_json, lit_type, size_expr))
        return true;
      solidity_gen_typecast(ns, size_expr, unsignedbv_typet(256));

      // T1.1 Stage S1: the length companion is now addr-keyed.  A
      // decl-time set of the symbol's static value (`len_mut.value =
      // size_expr`) cannot carry a per-instance length any more — there
      // is no `this` at decl-parse time.  For state-var dyn-arrays
      // declared with an explicit `= new uint[](N)` initializer, the
      // matching ctor-time runtime assign is emitted via the
      // `arr = new uint[](n)` ARRAY_CALLOC branch in
      // solidity_convert_expr.cpp:4839+ when the initializer is
      // evaluated as a state-var assignment.  Here we leave the
      // symbol's static value at gen_zero (the default for the array
      // type), so each instance starts with length 0 until a runtime
      // write occurs.
      (void)id; // suppress unused-variable warning if id is otherwise unused
    }
    // Zero-initialize the infinite array so elements read as 0
    {
      exprt _v = gen_zero(get_complete_type(t, ns), true);
      _v.zero_initializer(true);
      added_symbol.set_value(std::move(_v));
    }
  }
  else if (t_sol_type == SolidityGrammar::SolType::DYNARRAY && set_init)
  {
    // Note for inherited dynamic array, they will be registered in
    // D.dyn_arr = _ESBMC_arrcpy(B.dyn_ar)
    exprt val;
    if (get_init_expr(init_value, literal_type, t, val))
      return true;

    // Detect the `uint[] x = new T[](N)` / `new bytes(N)` shape: the AST
    // init value is a FunctionCall whose callee is a NewExpression. This
    // is the only shape where reading `init_value["arguments"][0]` as the
    // length is correct. Other typecast-wrapped function-call returns
    // (e.g. `abi.decode(corrupt, (uint[][]))`) accidentally matched the
    // old `val.is_typecast() || ARRAY_CALLOC` gate and ended up handing
    // the bytes payload to _ESBMC_store_array as a "length", which then
    // tripped a struct-vs-unsignedbv type mismatch in goto-symex.
    const bool init_is_new_array =
      init_value.contains("nodeType") &&
      init_value["nodeType"] == "FunctionCall" &&
      init_value.contains("expression") &&
      init_value["expression"].contains("nodeType") &&
      init_value["expression"]["nodeType"] == "NewExpression" &&
      init_value.contains("arguments") && init_value["arguments"].is_array() &&
      !init_value["arguments"].empty();

    if (
      init_is_new_array &&
      (val.is_typecast() ||
       get_sol_type(val.type()) == SolidityGrammar::SolType::ARRAY_CALLOC))
    {
      // uint[] zz = new uint(10);
      // uint[] zz = new uint(len);
      //=> uint* zz = (uint *)calloc(10, sizeof(uint));
      //=> uint* zz = (uint *)calloc(len, sizeof(uint));
      solidity_gen_typecast(ns, val, t);
      added_symbol.set_value(migrate_expr(val));
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
          member_exprt(this_expr, added_symbol.name, added_symbol.get_type()),
          size_expr,
          func_call);
      else
        store_update_dyn_array(symbol_expr(added_symbol), size_expr, func_call);
      move_to_back_block(func_call);
    }
    else if (
      val.is_typecast() && val.operands().size() > 0 &&
      !val.operands()[0].type().is_pointer() &&
      !val.operands()[0].type().is_array())
    {
      // Scalar→pointer cast: the RHS is a scalar value masquerading as
      // a dynamic-array pointer. The canonical case is
      //   uint[][] m = abi.decode(corrupt, (uint[][]));
      // where abi.decode is lowered to the uint256 identity in
      // solidity_abi.c. We cannot copy from a scalar through arrcpy and
      // we cannot read its array length — both would deref garbage. Over-
      // approximate the same way assign_param_nondet does for harness
      // entry points: allocate a fresh small backing of the destination
      // shape (zero contents, declared length recorded in the header).
      // 1D vs 2D is dispatched on whether the leaf element is itself a
      // pointer; 3D+ falls through to nil so the existing
      // "Unexpect initialization for dynamic array" error fires (no
      // silent miscompile).
      constexpr unsigned long kInitNondetLen = 4;

      exprt alloc_expr;
      bool built = false;

      if (t.is_pointer() && !t.subtype().is_pointer())
      {
        // 1D dyn: alloc(kInitNondetLen, sizeof(elem))
        exprt size_expr = constant_exprt(
          integer2binary(kInitNondetLen, bv_width(uint_type())),
          std::to_string(kInitNondetLen),
          uint_type());

        exprt sizeof_expr;
        get_size_of_expr(t.subtype(), sizeof_expr);

        side_effect_expr_function_callt alloc;
        get_calloc_function_call(location_begin, alloc);
        alloc.arguments().push_back(size_expr);
        alloc.arguments().push_back(sizeof_expr);

        alloc_expr = typecast_exprt(alloc, t);
        built = true;
      }
      else if (
        t.is_pointer() && t.subtype().is_pointer() &&
        !t.subtype().subtype().is_pointer())
      {
        // 2D dyn: alloc_nested_2d(kInitNondetLen, kInitNondetLen, sizeof(leaf))
        exprt outer_expr = constant_exprt(
          integer2binary(kInitNondetLen, bv_width(uint_type())),
          std::to_string(kInitNondetLen),
          uint_type());
        exprt inner_expr = outer_expr;

        exprt elem_sizeof;
        get_size_of_expr(t.subtype().subtype(), elem_sizeof);

        side_effect_expr_function_callt alloc;
        get_library_function_call_no_args(
          "_ESBMC_alloc_nested_2d",
          "c:@F@_ESBMC_alloc_nested_2d",
          pointer_typet(empty_typet()),
          location_begin,
          alloc);
        alloc.arguments().push_back(outer_expr);
        alloc.arguments().push_back(inner_expr);
        alloc.arguments().push_back(elem_sizeof);

        alloc_expr = typecast_exprt(alloc, t);
        built = true;
      }

      if (built)
      {
        added_symbol.set_value(alloc_expr);
        decl.operands().push_back(alloc_expr);

        exprt size_expr_for_hdr = constant_exprt(
          integer2binary(kInitNondetLen, bv_width(uint_type())),
          std::to_string(kInitNondetLen),
          uint_type());
        exprt func_call;
        if (is_state_var)
          store_update_dyn_array(
            member_exprt(this_expr, added_symbol.name, added_symbol.get_type()),
            size_expr_for_hdr,
            func_call);
        else
          store_update_dyn_array(
            symbol_expr(added_symbol), size_expr_for_hdr, func_call);
        move_to_back_block(func_call);
      }
      else
      {
        log_error("Unexpect initialization for dynamic array");
        log_debug("solidity", "{}", val);
        return true;
      }
    }
    else if (val.id() == "sideeffect")
    {
      // e.g. `uint[] memory xs = someFunc();` where someFunc
      // returns `uint[] memory`. The side-effect expression is
      // the function call; assign it directly and compute the
      // length via _ESBMC_array_length on the result.
      solidity_gen_typecast(ns, val, t);
      added_symbol.set_value(val);
      decl.operands().push_back(val);

      // store length: len = _ESBMC_array_length(val)
      locationt loc;
      get_location_from_node(init_value, loc);
      side_effect_expr_function_callt length_call;
      get_library_function_call_no_args(
        "_ESBMC_array_length",
        "c:@F@_ESBMC_array_length",
        uint_type(),
        loc,
        length_call);
      length_call.arguments().push_back(val);

      exprt func_call;
      if (is_state_var)
        store_update_dyn_array(
          member_exprt(this_expr, added_symbol.name, added_symbol.get_type()),
          length_call,
          func_call);
      else
        store_update_dyn_array(
          symbol_expr(added_symbol), length_call, func_call);
      move_to_back_block(func_call);
    }
    else if (val.type().is_pointer() || val.type().is_array())
    {
      // Any other dynamic-array-shaped initializer: bare symbol, member
      // access, index access on an outer dynamic array, etc. Examples:
      //   uint[] zzzzzz = zzz;       // existing variable
      //   uint[] m = a[i];           // inner array of uint[][]
      //   uint[] m = s.field;        // struct field
      // All collapse to "copy a pointer + propagate the runtime length
      // tracked by _ESBMC_array_length" — _ESBMC_arrcpy does both.
      exprt size_expr;
      get_size_expr(val, size_expr);

      exprt size_of_expr;
      get_size_of_expr(t.subtype(), size_of_expr);

      side_effect_expr_function_callt acpy_call;
      get_arrcpy_function_call(location_begin, acpy_call);
      acpy_call.arguments().push_back(val);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      solidity_gen_typecast(ns, acpy_call, t);
      added_symbol.set_value(acpy_call);
      decl.operands().push_back(acpy_call);

      exprt func_call;
      if (is_state_var)
        store_update_dyn_array(
          member_exprt(this_expr, added_symbol.name, added_symbol.get_type()),
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
  // Extended to cover mapping(K => T[N]) under !is_new_expr: the chain
  // walk above rewrote `t` to mapping_t so this decl must also run the
  // `{base=_ESBMC_inf_*, addr=this->$address}` init block; otherwise
  // `map_fixed_arr_get(&m, k, sz)` sees zero-init fields and every
  // lookup returns a fresh nondet slab.
  else if (
    is_mapping && (is_new_expr || t.get_bool("#sol_mapping_fixed_arr_value")))
  {
    // mapping(string => uint) test;
    // 1. the contract that contains this mapping is also used in a new
    // expression
    // => __attribute__((annotate("__ESBMC_inf_size"))) struct _ESBMC_Mapping
    // _ESBMC_inf_test[];
    // => struct mapping_t test = {_ESBMC_inf_test, this.address};
    //
    // The construction (inf-array global + {base, mid, addr=this->$address}
    // struct value) is shared with the Phase-2 ctor walker for nested
    // mappings inside user-struct fields, so it lives in
    // build_mapping_t_init_value (solidity_convert_mapping.cpp).  The
    // owner-of-`addr` is always the ctor's `this`: when this state-var
    // decl is being parsed lazily from inside a function body (e.g. an
    // inherited mapping first referenced from within `set`), `this_expr`
    // resolved earlier in this function would point at that function's
    // `this`, not the ctor's — baking a wrong addr into the symbol's
    // static initializer.
    exprt ctor_this_expr;
    bool have_ctor_this =
      is_state_var && !current_contractName.empty() &&
      !get_ctor_decl_this_ref(current_contractName, ctor_this_expr);
    const exprt &addr_owner_this = have_ctor_this ? ctor_this_expr : this_expr;

    exprt inits;
    if (
      build_mapping_t_init_value(
        current_contractName, name, addr_owner_this, location_begin, inits))
      return true;

    added_symbol.set_value(migrate_expr(inits));
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
    assert(has_sol_bytesn_size(t));
    exprt len = from_integer(std::stoul(get_sol_bytesn_size(t)), uint_type());
    call.arguments().push_back(len);
    added_symbol.set_value(migrate_expr(call));
    decl.operands().push_back(call);
  }
  // now we have rule out other special cases
  else if (set_init)
  {
    if (get_init_expr(init_value, literal_type, t, val))
      return true;
    added_symbol.set_value(migrate_expr(val));
    decl.operands().push_back(val);
  }

  // For local variables without explicit initializer, Solidity guarantees
  // zero-initialization.  Emit the zero value so the GOTO program gets
  // an assignment (DECL alone leaves the variable uninitialised).
  // Only add if no init operand was already pushed by a special-case handler
  // above (arrays, dynarray, mapping, etc. handle their own initialization).
  if (
    !is_state_var && decl.operands().size() == 1 && !is_contract && !is_mapping)
    decl.operands().push_back(gen_zero(get_complete_type(t, ns), true));

  // store state variable, which will be initialized in the constructor
  // note that for the state variables that do not have initializer
  // we have already set it as zero value
  // For unintialized contract type, no need to move to the initializer
  //
  // B8 fix: inherited mappings still need to be initialized in the derived
  // contract's ctor when the derived contract is `new`'d.  The aux-base ctor
  // copy in move_inheritance_to_ctor cannot carry mapping fields (Base's
  // own struct never gets the mapping_t component because Base.m is filtered
  // out by the "mapping && is_array" guard at get_struct_class_fields when
  // Base itself is not in newContractSet).  Without a per-instance init,
  // every derived instance shares m.addr=0 and writes alias across instances.
  const bool inherited_mapping_needs_per_instance_init =
    is_inherited && is_mapping && is_new_expr;
  if (
    is_state_var &&
    (!is_inherited || inherited_mapping_needs_per_instance_init) &&
    !(is_contract && !has_init) && !(is_mapping && !is_new_expr) &&
    !(is_mapping_array && !is_new_expr) && !is_dynarray_state)
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

  if (struct_def["nodeType"].get<std::string>() == "ContractDefinition")
  {
    name = struct_def["name"].get<std::string>();
    id = prefix + name;
    t.tag(name);
  }
  else if (struct_def["nodeType"].get<std::string>() == "StructDefinition")
  {
    // ""tag-struct Struct_Name"
    name = struct_def["name"].get<std::string>();
    id = prefix + "struct " + struct_def["canonicalName"].get<std::string>();
    // THE TAG MUST BE AS QUALIFIED AS THE ID, and it was not.
    //
    // The id above uses `canonicalName` -- `L1.Data` -- while the tag used the
    // bare `name`, `Data`. Two names for one thing, and they disagreed on
    // whether the declaring scope counts. The tag is what survives into
    // `struct_type2t::name`, and z3_conv.cpp:1030 builds the tuple SORT NAME
    // out of exactly that:
    //
    //     std::string("struct_type_" + strct.name.as_string())
    //
    // So two structs called `Data` in different libraries (or contracts) asked
    // z3 for two datatypes under ONE name. When one of them holds the other --
    // `library AddressSet { struct Data { AddressArray.Data items; } }`, the
    // ordinary Solidity idiom -- the sort ends up containing itself, and z3
    // refuses at ENCODING time with `datatype is not well-founded` and aborts.
    //
    // MEASURED: notes/coverage/poc/D13_Z3TupleNotWellFounded.sol, reduced from
    // st1inch (4874 lines -> 17). The same file with the structs renamed
    // `AlphaData` / `BetaData` and nothing else changed completes normally;
    // that one-variable pair is the whole diagnosis. st1inch hits it through
    // AddressArray.Data + AddressSet.Data, which is why the path-coverage
    // collector has to carry `--z3 --tuple-node-flattener` as a workaround.
    //
    // WHY HERE AND NOT IN z3_conv: a sort name has to be STABLE across the
    // repeated mk_struct_sort calls for one type, so "make it unique" cannot
    // mean a counter, and synthesising one from the members would leave two
    // distinct Solidity types sharing a name everywhere else. The name is
    // supposed to identify the type; giving two types one name is the defect.
    t.tag("struct " + struct_def["canonicalName"].get<std::string>());

    // populate the member_entity_scope
    // this map is used to find reference when there is no decl_ref_id provided
    // in the nodes or replace the find_decl_ref in order to speed up
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
      comp.type().member_name(t.tag());

      if (get_access_from_decl(*itr, comp))
        return true;
      t.components().push_back(comp);
      break;
    }
    case SolidityGrammar::ContractBodyElementT::EnumDef:
    case SolidityGrammar::ContractBodyElementT::UsingForDef:
    case SolidityGrammar::ContractBodyElementT::ModifierDef:
    case SolidityGrammar::ContractBodyElementT::UserDefinedValueTypeDef:
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
  added_symbol.set_type(t);

  return false;
}

// parse a contract definition
bool solidity_convertert::get_contract_definition(const std::string &c_name)
{
  // cache
  // this is due to that we might call this function to parse another contract B
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

  if (
    get_sol_type(comp.type()) == SolidityGrammar::SolType::MAPPING &&
    comp.type().is_array())
  {
    // Mappings (including nested) in contracts not used in `new` expressions
    // are converted to global static infinite arrays.
    // Cannot be struct members due to infinite size (breaks padding/gen_zero).
    return false;
  }

  // mapping(K=>V)[] is also modeled as a 2D infinite array (not a pointer)
  if (get_sol_mapping_array(comp.type()))
    return false;

  // dynarray state vars are modeled as global infinite arrays (not struct
  // members)
  if (get_sol_dynarray_state(comp.type()))
    return false;

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
  comp.type().member_name(type.tag());

  // A fixed `bytesN` field lowers to the generic `BytesStatic` struct, which
  // (unlike a top-level bytesN parameter) loses its source width. Re-stamp
  // `#sol_bytesn_size` from the AST type string so the Foundry coverage-test
  // generator can render a struct-literal field as `bytesN(0x..)`. The dynamic
  // `bytes` type string is exactly "bytes" (length 5) and is excluded; an array
  // like "bytes32[]" is excluded by the trailing-char check.
  if (
    ast_node.contains("typeName") &&
    ast_node["typeName"].contains("typeDescriptions"))
  {
    const std::string ts =
      ast_node["typeName"]["typeDescriptions"].value("typeString", "");
    if (ts.compare(0, 5, "bytes") == 0 && ts.size() > 5)
    {
      char *end = nullptr;
      unsigned long n = std::strtoul(ts.c_str() + 5, &end, 10);
      if (end && *end == '\0' && n >= 1 && n <= 32)
        // Stamp on the COMPONENT irep (not its type): a later type-follow
        // resolves the `BytesStatic` symbol type to its struct body and would
        // drop a stamp on the type, but the component's own attributes survive.
        comp.set("#sol_bytesn_size", static_cast<unsigned>(n));
    }
  }

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
    set_sol_type(new_expr.type(), SolidityGrammar::SolType::LIBRARY);
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
    // R0's event rung: mark the SYMBOL as an event, here, where the AST still
    // says so. Everything downstream sees an ordinary function symbol with an
    // empty body -- indistinguishable from an interface stub or an abstract
    // override -- so this is the last point at which the fact exists.
    //
    // ON THE SYMBOL rather than on a statement location: MEASURED, an extra
    // irep field on a Solidity statement's location does NOT survive goto
    // conversion, which rebuilds the instruction location from file/line/
    // function. Symbols are carried in the symbol table and are not rebuilt,
    // so a consumer can look the callee up by name.
    std::string ev_name, ev_id;
    get_function_definition_name(ast_node, ev_name, ev_id);
    if (symbolt *ev_sym = context.find_symbol(ev_id))
    {
      typet t = ev_sym->get_type();
      t.set("sol_event", true);
      ev_sym->set_type(std::move(t));
    }
  }
  else if (node_type == "ContractDefinition" && ast_node["abstract"] == true)
  {
    // for abstract contract
    add_empty_body_node(ast_node);
  }
  else if (
    node_type == "ContractDefinition" &&
    ast_node["contractKind"] == "interface")
  {
    // Round-1 interface handling: the interface itself is later processed
    // in round 2 (get_contract_definition), but its nested type children —
    // struct / enum / error / event — must be registered *now* so that
    // round-1 libraries whose function signatures reference `IFoo.Bar` can
    // resolve the symbol. Without this pre-pass, a library returning an
    // interface-nested struct would crash on named-return declaration.
    add_empty_body_node(ast_node);

    if (ast_node.contains("nodes"))
    {
      std::string if_name = ast_node["name"].get<std::string>();
      std::string old = current_baseContractName;
      current_baseContractName = if_name;
      for (auto &sub : ast_node["nodes"])
      {
        const std::string nt = sub["nodeType"].get<std::string>();
        if (
          nt == "ErrorDefinition" || nt == "EventDefinition" ||
          nt == "StructDefinition" || nt == "EnumDefinition")
        {
          if (get_noncontract_defition(sub))
            return true;
        }
      }
      current_baseContractName = old;
    }
  }
  else if (
    node_type == "FunctionDefinition" && current_baseContractName.empty())
  {
    // __ESOL_* intrinsic stubs: the user declares these free functions
    // purely so solc accepts the syntax of intrinsic calls; the frontend
    // intercepts every call site (see get_call_expr) and rewrites it to
    // the corresponding built-in.  Skip parsing the stub body entirely —
    // it typically returns a contract value, which is not a meaningful
    // IR shape and crashes downstream expression conversion.
    if (
      ast_node.contains("name") && ast_node["name"].is_string() &&
      ast_node["name"].get<std::string>().rfind("__ESOL_", 0) == 0)
    {
      log_debug(
        "solidity",
        "skipping __ESOL_ intrinsic stub {}",
        ast_node["name"].get<std::string>());
      return false;
    }

    // Free function (outside any contract) — only handle at top-level scope.
    // Contract-internal functions are handled by convert_ast_nodes after
    // get_struct_class has registered the contract struct symbol.
    if (get_function_definition(ast_node))
      return true;
  }
  else if (
    node_type == "ContractDefinition" && ast_node["contractKind"] == "library")
  {
    // for library entity
    // a library is equivalent to a static class
    std::string lib_name = ast_node["name"].get<std::string>();

    // we treat library as a contract, but we do not populate it as
    // struct/contract symbol instead, we only populate the entity and functions
    std::string old = current_baseContractName;
    current_baseContractName = lib_name;

    // Register inner error/event/struct/enum symbols first, so that
    // get_struct_class's body iteration can resolve them as references.
    // Regular contracts do this in get_contract_definition; libraries
    // previously skipped the pre-pass and crashed on `error Foo();`.
    if (ast_node.contains("nodes"))
    {
      for (auto &sub : ast_node["nodes"])
      {
        const std::string nt = sub["nodeType"].get<std::string>();
        if (
          nt == "ErrorDefinition" || nt == "EventDefinition" ||
          nt == "StructDefinition" || nt == "EnumDefinition")
        {
          if (get_noncontract_defition(sub))
            return true;
        }
      }
    }

    if (get_struct_class(ast_node))
      return true;

    // Library bodies that touch bytes/string need a `$dynamic_pool`
    // member on the library struct — bytes operations lower to
    // `member_exprt(this, "$dynamic_pool")` regardless of whether `this`
    // points at a contract or a library, and the lookup crashes goto
    // migration if the member is missing. The pool data is per-library
    // (not shared with the calling contract) which makes the read an
    // over-approximation: `b[i]` returns 0 rather than the caller's
    // actual byte. Acceptable trade-off: calldata is read-only and
    // libraries have no other channel to read non-trivial data, so the
    // verification semantics stay sound.
    if (add_dynamic_pool_member(ast_node, lib_name))
      return true;

    // Register per-library low-level call helpers (`$call#0`, `$call#1`,
    // `$transfer#0`, `$send#0`, `$staticcall#0`, `$delegatecall#0`) so
    // that `.call(data)` / `.transfer(v)` / ... emitted from inside a
    // library body resolve to a callable symbol at symex time.  Without
    // this, the caller at `get_low_level_member_accsss` would look up
    // `sol:@C@<Lib>@F@$call#0` and ESBMC would abort with
    // "Function type mismatch: expected code".  Library-mode bodies are
    // over-approximated (msg.sender/mutex/balance state skipped) — see
    // the is_library branches inside each populate helper.
    if (populate_low_level_functions(lib_name, /*is_library=*/true))
      return true;

    if (convert_ast_nodes(ast_node, lib_name))
      return true;

    current_baseContractName = old;
  }

  return false;
}

// add a "body" node to functions within interface && abstract && event
// the idea is to utilize the function-handling APIs.
void solidity_convertert::add_empty_body_node(nlohmann::json &ast_node)
{
  //? will this affect find_decl_ref?
  // 0.6.x emits `body: null` for unimplemented functions (interface /
  // abstract); 0.8.x omits the field entirely. Either form means "no body".
  auto missing_body = [](const nlohmann::json &n) {
    return !n.contains("body") || n["body"].is_null();
  };

  if (ast_node["nodeType"] == "EventDefinition")
  {
    // for event-definition
    if (missing_body(ast_node))
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
        (subNode["nodeType"] == "FunctionDefinition") && missing_body(subNode))
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
        (subNode["nodeType"] == "FunctionDefinition") && missing_body(subNode))
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
  added_symbol.set_type(type);

  // Mark the compiled error function so the Foundry coverage-test generator can
  // recognise a `revert CustomError(...)` (lowered to a call of this symbol) as
  // a revert terminator and emit `vm.expectRevert()`. Verification-inert: read
  // only by goto_coverage's branch-edge classifier and the generator, never by
  // symex/solver/k-induction.
  {
    typet t = added_symbol.get_type();
    t.set("#sol_error", name);
    added_symbol.set_type(std::move(t));
  }

  // Custom error body: `__ESBMC_assume(false)` to prune the revert path
  // at the SMT level (matches real-EVM revert-aborts-construction
  // semantics). Coverage-mode exception (--branch-coverage-claims):
  // a custom error inside a constructor invariant pruning the
  // construction path means EVERY post-constructor path is proven
  // unreachable when SAT cannot satisfy the invariant's PASS direction
  // (e.g. st1inch's `_votingPowerAt` nonlinear-bitshift conjunction).
  // Native lcov reaches those post-constructor methods because the
  // deployment satisfies the invariant by construction; for the
  // coverage methodology to be comparable, ESBMC's coverage-mode lower
  // must NOT prune those paths.  Emit `code_skipt()` so the revert
  // "continues" — sound for the coverage-measurement methodology
  // (every reach reported is a real reach in some execution model);
  // unsound for safety verification on the same run (so this flip is
  // gated on `--branch-coverage-claims`, which already converts
  // assertions to guards via goto_coverage.cpp:438-441 for the same
  // reason — the mode is for coverage, not safety).
  const bool coverage_mode =
    !config.options.get_option("branch-coverage-claims").empty();
  code_blockt body;
  if (coverage_mode)
  {
    body.operands().push_back(code_skipt());
  }
  else
  {
    typet return_type = empty_typet();
    locationt loc;
    get_location_from_node(ast_node, loc);
    side_effect_expr_function_callt call;
    get_library_function_call_no_args(
      "__ESBMC_assume", "c:@F@__ESBMC_assume", return_type, loc, call);
    exprt arg = false_exprt();
    call.arguments().push_back(arg);
    convert_expression_to_code(call);
    body.operands().push_back(call);
  }
  added_symbol.set_value(body);

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
  //  - For state variable name, just use the ast_node["name"], e.g.
  //  sol:@C@Base@x#11
  //  - For state variable id, add prefix "sol:@"
  name = ast_node["name"].get<std::string>();
  if (!cname.empty())
    id = "sol:@C@" + cname + "@" + name + "#" +
         i2string(ast_node["id"].get<int>());
  else
    id = "sol:@" + name + "#" + i2string(ast_node["id"].get<int>());
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
  // Struct/error fields carry a `scope` pointing to the StructDefinition
  // (or ErrorDefinition) AST node id, which we registered in
  // member_entity_scope when walking the struct.  Detect that *before*
  // the function-local branch, otherwise a field VariableDeclaration that
  // happens to be re-walked from inside a function body (e.g. via
  // get_var_decl_ref while resolving `self.field` for a storage-ref
  // alias) gets a `sol:@C@<cname>@F@<func>@<field>#<id>` naming and
  // collides with the local-variable namespace — the resulting symbol
  // never lands in the symbol table (the struct field's symbol is
  // recorded under its struct-qualified id) and goto-symex later trips
  // `phi_function: no symbol for ...` when merging branches that
  // assigned through the alias.
  if (
    ast_node.contains("scope") &&
    member_entity_scope.count(ast_node["scope"].get<int>()) > 0)
  {
    int scp = ast_node["scope"].get<int>();
    std::string struct_name = member_entity_scope.at(scp);
    if (cname.empty())
      id = "sol:@" + struct_name + "@" + name + "#" +
           i2string(ast_node["id"].get<int>());
    else
      id = "sol:@C@" + cname + "@" + struct_name + "@" + name + "#" +
           i2string(ast_node["id"].get<int>());
  }
  else if (
    (current_functionDecl || !current_functionName.empty()) && !cname.empty())
  {
    // converting local variable inside a function
    // For non-state functions, we give it different id.
    // E.g. for local variable i in function nondet(), it's
    // "sol:@C@Base@F@nondet@i#55".
    if (current_functionName.empty())
      current_functionName = (*current_functionDecl)["name"];
    assert(!current_functionName.empty());
    // As the local variable inside the function will not be inherited, we can
    // use current_functionName
    id = "sol:@C@" + cname + "@F@" + current_functionName + "@" + name + "#" +
         i2string(ast_node["id"].get<int>());
  }
  else if (
    (current_functionDecl || !current_functionName.empty()) && cname.empty())
  {
    // Free function (outside any contract): use sol:@F@funcName@varName#id
    if (current_functionName.empty())
      current_functionName = (*current_functionDecl)["name"];
    assert(!current_functionName.empty());
    id = "sol:@F@" + current_functionName + "@" + name + "#" +
         i2string(ast_node["id"].get<int>());
  }
  else if (ast_node.contains("scope"))
  {
    // This means we are handling a local variable which is not inside a
    // function body.
    //! Assume it is a variable inside struct/error which can be declared
    //! outside the contract
    int scp = ast_node["scope"].get<int>();
    if (member_entity_scope.count(scp) == 0)
    {
      log_error("cannot find struct/error name");
      abort();
    }
    std::string struct_name = member_entity_scope.at(scp);
    if (cname.empty())
      id = "sol:@" + struct_name + "@" + name + "#" +
           i2string(ast_node["id"].get<int>());
    else
      id = "sol:@C@" + cname + "@" + struct_name + "@" + name + "#" +
           i2string(ast_node["id"].get<int>());
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
    // e.g. sol:@C@Base@F@errmsg#12
    id = "sol:@C@" + cname + "@F@" + name + "#" + std::to_string(id_num);
}

void solidity_convertert::get_function_definition_name(
  const nlohmann::json &ast_node,
  std::string &name,
  std::string &id)
{
  // Follow the way in clang:
  //  - For function name, just use the ast_node["name"]
  // assume Solidity AST json object has "name" field, otherwise throws an
  // exception in nlohmann::json
  std::string contract_name;
  get_current_contract_name(ast_node, contract_name);
  if (contract_name.empty())
  {
    name = ast_node["name"].get<std::string>();
    id = "sol:@F@" + name + "#" + i2string(ast_node["id"].get<int>());
    return;
  }

  //! for event/... who have added an body node. It seems that a ["kind"] is
  //! automatically added.?
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
       i2string(ast_node["id"].get<int>());

  log_debug("solidity", "\t\t@@@ got function name {}", name);
}
