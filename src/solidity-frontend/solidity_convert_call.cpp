/// \file solidity_convert_call.cpp
/// \brief Function call conversion for the Solidity frontend.
///
/// Converts Solidity function calls from the solc JSON AST into ESBMC's
/// side_effect_expr_function_callt representation. Handles library function
/// calls, using-for directive calls, and regular internal/external function
/// calls with argument conversion and this-pointer injection.

#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith/arith_tools.h>
#include <util/arith/bitvector.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/base/i2string.h>
#include <util/arith/mp_arith.h>
#include <util/irep/std_expr.h>
#include <irep2/irep2_utils.h>
#include <util/irep/migrate.h>
#include <util/message/message.h>
#include <fstream>
#include <limits>

bool solidity_convertert::get_library_function_call(
  const nlohmann::json &decl_ref,
  const nlohmann::json &caller,
  side_effect_expr_function_callt &call,
  bool skip_first_param)
{
  assert(!decl_ref.empty());
  assert(decl_ref.contains("returnParameters"));

  exprt func;
  if (get_func_decl_ref(decl_ref, func))
    return true;

  code_typet t;
  if (get_type_description(decl_ref["returnParameters"], t.return_type()))
    return true;

  return get_library_function_call(
    func, t, decl_ref, caller, call, skip_first_param);
}

// library/error/event functions have no definition node
// the key difference comparing to the `get_non_library_function_call` is that
// we do not need a this-object as the first argument for the function call
bool solidity_convertert::get_library_function_call(
  const exprt &func,
  const typet &t,
  const nlohmann::json &decl_ref,
  const nlohmann::json &caller,
  side_effect_expr_function_callt &call,
  bool skip_first_param)
{
  call.function() = func;
  if (t.is_code())
  {
    call.type() = to_code_type(t).return_type();
    assert(!call.type().is_code());
  }
  else
    call.type() = t;
  locationt l;
  get_location_from_node(caller, l);
  call.location() = l;

  if (caller.contains("arguments"))
  {
    nlohmann::json param = nullptr;
    const nlohmann::json empty_array = nlohmann::json::array();
    auto itr = empty_array.end();
    auto itr_end = empty_array.end();
    if (!decl_ref.empty() && decl_ref.contains("parameters"))
    {
      assert(decl_ref["parameters"].contains("parameters"));
      const nlohmann::json &param_nodes = decl_ref["parameters"]["parameters"];
      itr = param_nodes.begin();
      itr_end = param_nodes.end();
    }

    // For "using for" calls (e.g. z.limb(1) => limb(z, 1)), the base object
    // will be prepended as the first argument by the caller. Skip the first
    // parameter so that remaining arguments match the correct parameter types.
    if (skip_first_param && itr != itr_end)
      ++itr;

    // Determine the maximum number of parameters the C model function accepts.
    // Only apply this limit when the function type has explicitly declared
    // parameters (size > 0). Builtins like assert/require have code_type with
    // no declared params but still accept arguments from the caller.
    size_t max_params = std::numeric_limits<size_t>::max();
    if (decl_ref.empty() && t.is_code())
    {
      size_t declared = to_code_type(t).arguments().size();
      if (declared > 0)
        max_params = declared;
    }

    //  builtin functions do not need the this object as the first arguments
    const nlohmann::json *param_decl = nullptr;
    for (const auto &arg : caller["arguments"].items())
    {
      // Stop collecting arguments once we have enough for the C model function.
      if (call.arguments().size() >= max_params)
        break;

      // Skip non-value arguments that cannot be evaluated as expressions:
      //  - type expressions (t_type$...): e.g. (uint256) in abi.decode
      //  - function declarations (t_function_declaration_...): e.g.
      //    ITarget.transfer in abi.encodeCall
      if (arg.value().contains("typeDescriptions"))
      {
        std::string tid =
          arg.value()["typeDescriptions"].value("typeIdentifier", "");
        if (
          tid.compare(0, 7, "t_type$") == 0 ||
          tid.compare(0, 23, "t_function_declaration_") == 0)
          continue;
        // [APPROX: UNDER] Function references passed by name as r-values
        // (internal or external function types), e.g. Utils.sum in
        // Utils.reduce(x, Utils.sum, 0). Substitute an opaque nondet
        // pointer — indirect calls through the parameter inside the callee
        // are modelled as nondet returns by the function-pointer indirect-
        // call path in get_call_expr. This loses the precise callee
        // identity (UNDER-approximation of control flow): bugs inside
        // functions reachable only via such fn-ptr arguments are not
        // detected. No false positives — the nondet return covers every
        // possible value.
        if (
          tid.compare(0, 20, "t_function_internal_") == 0 ||
          tid.compare(0, 20, "t_function_external_") == 0)
        {
          exprt nondet_fp =
            exprt("sideeffect", gen_pointer_type(empty_typet()));
          nondet_fp.type().set("#sol_func_ptr", true);
          nondet_fp.statement("nondet");
          call.arguments().push_back(nondet_fp);
          if (itr != itr_end)
            ++itr;
          continue;
        }
      }

      exprt single_arg;
      if (itr != itr_end && (*itr).contains("typeDescriptions"))
      {
        param = (*itr)["typeDescriptions"];
        param_decl = &(*itr);
        ++itr;
      }
      else if (arg.value().contains("commonType"))
        param = arg.value()["commonType"];
      else if (arg.value().contains("typeDescriptions"))
        param = arg.value()["typeDescriptions"];

      if (get_expr(arg.value(), param, single_arg))
        return true;

      // Coerce the actual argument to the formal parameter's type, mirroring
      // the same fixup in get_non_library_function_call. Without this,
      // built-in lowerings whose result type differs from the declared
      // Solidity type — e.g. abi.encodeWithSelector → uint256 identity (see
      // solidity_abi.c) being passed to a `bytes memory` parameter
      // (BytesDynamic struct) — leak a scalar through the call boundary
      // and goto-symex aborts at the function-arg type check with
      // `type mismatch: got unsignedbv, expected struct`. SafeERC20's
      // _callOptionalReturn(IERC20, bytes memory) is the canonical case.
      if (param_decl != nullptr)
      {
        typet formal_t;
        bool got_formal = false;
        if (param_decl->contains("typeName"))
        {
          if (!get_type_description(
                *param_decl,
                (*param_decl)["typeName"]["typeDescriptions"],
                formal_t))
            got_formal = true;
        }
        else if (!param.is_null())
        {
          if (!get_type_description(param, formal_t))
            got_formal = true;
        }

        if (got_formal && single_arg.type() != formal_t)
          convert_type_expr(ns, single_arg, formal_t, arg.value());
      }

      call.arguments().push_back(single_arg);
      param = nullptr;
      param_decl = nullptr;
    }
  }

  return false;
}

/**
    * call to a non-library function:
    *   this.func(); // func(&this)
    * @param decl_ref: the function declaration node
    * @param caller: the function caller node which contains the arguments
    TODO: if the paramenter is a 'memory' type, we need to create
    a copy. E.g. string memory x => char *x => char * x_cpy
    this could be done by memcpy. However, for dyn_array, we do not have
    the size info. Thus in the future we need to convert the dyn array to
    a struct which record both array and size. This will also help us to support
    array.length, .push and .pop
**/
bool solidity_convertert::get_non_library_function_call(
  const nlohmann::json &decl_ref,
  const nlohmann::json &caller,
  side_effect_expr_function_callt &call)
{
  if (decl_ref.empty() || decl_ref.is_null())
  {
    log_error("unexpect empty or null declaration json");
    return true;
  }

  log_debug(
    "solidity",
    "\tget_non_library_function_call {}",
    decl_ref["name"] != "" ? decl_ref["name"].get<std::string>()
                           : decl_ref["kind"].get<std::string>());

  locationt loc = locationt();
  if (!caller.empty())
    get_location_from_node(caller, loc);
  else if (current_functionDecl)
    get_location_from_node(*current_functionDecl, loc);

  exprt func;
  if (get_func_decl_ref(decl_ref, func))
    return true;

  assert(decl_ref.contains("returnParameters"));
  code_typet t;
  if (get_type_description(decl_ref["returnParameters"], t.return_type()))
    return true;

  call.location() = loc;
  call.function() = func;
  call.function().location() = loc;
  call.type() = t.return_type();

  // Populating arguments

  // this object — skip for free functions (no this pointer)
  bool is_free_func = decl_ref.contains("kind") &&
                      decl_ref["kind"].get<std::string>() == "freeFunction";
  if (!is_free_func)
  {
    exprt this_object = nil_exprt();
    if (current_functionDecl)
    {
      if (get_func_decl_this_ref(*current_functionDecl, this_object))
        return true;
    }
    else if (!caller.empty())
    {
      if (get_ctor_decl_this_ref(caller, this_object))
        return true;
    }
    // otherwise, it's the auxiliary function we defined //e.g. call,
    // delegatecall...

    call.arguments().push_back(this_object);
  }

  if (decl_ref.contains("parameters") && caller.contains("arguments"))
  {
    // * Assume it is a normal function call, including ctor call with params
    // set caller object as the first argument

    nlohmann::json param_nodes = decl_ref["parameters"]["parameters"];
    nlohmann::json param = nullptr;
    const nlohmann::json *param_decl = nullptr;
    nlohmann::json::iterator itr = param_nodes.begin();

    for (const auto &arg : caller["arguments"].items())
    {
      if (itr != param_nodes.end())
      {
        if ((*itr).contains("typeDescriptions"))
        {
          param = (*itr)["typeDescriptions"];
        }
        param_decl = &(*itr);
        ++itr;
      }

      exprt single_arg;
      if (get_expr(arg.value(), param, single_arg))
        return true;

      // Coerce the argument to the formal parameter's type. Built-in
      // model functions (e.g. abi.encode / abi.encodeCall / keccak256
      // → uint256 identity in solidity_abi.c, and address.call result
      // → tuple struct) return a different concrete shape than the
      // declared Solidity argument type. Without an explicit cast the
      // raw uint256 / scalar leaks into the call and goto-symex's
      // type check aborts at the boundary, e.g.
      //   abi.encodeCall(this.g, (...)) → uint256, then
      //   removeSignature(bytes calldata x) expects BytesDynamic.
      // convert_type_expr already knows how to wrap uint256 → bytes
      // via bytes_dynamic_from_uint and the various other shape
      // conversions, so just dispatch through it once we have the
      // formal parameter type.
      if (param_decl != nullptr)
      {
        typet formal_t;
        bool got_formal = false;
        if (param_decl->contains("typeName"))
        {
          if (!get_type_description(
                *param_decl,
                (*param_decl)["typeName"]["typeDescriptions"],
                formal_t))
            got_formal = true;
        }
        else if (!param.is_null())
        {
          if (!get_type_description(param, formal_t))
            got_formal = true;
        }

        if (got_formal && single_arg.type() != formal_t)
          convert_type_expr(ns, single_arg, formal_t, arg.value());
      }

      call.arguments().push_back(single_arg);
      param = nullptr;
      param_decl = nullptr;
    }
  }
  else
  {
    // we know we are calling a function within the source code
    // however, the definition json or the calling argument json is not provided
    // it could be the function call in the multi-transaction-verification
    // populate nil arguements
    if (assign_param_nondet(decl_ref, call))
    {
      log_error("Failed to populate nil parameters");
      return true;
    }
  }

  return false;
}

// extract new contract instance expression
// we insert that contract name into the newContractSet if there is a new
// expresssion related to this contract e.g. Base x = new Base(); then we insert
// "Base" into newContractSet the idea is that if the contract is not used in
// 'new', then we can simply create a global static infinity array to play as a
// mapping structure
void solidity_convertert::extract_new_contracts()
{
  if (!src_ast_json.contains("nodes"))
    return;

  std::function<void(const nlohmann::json &)> process_node;
  process_node = [&](const nlohmann::json &node) {
    if (node.is_object())
    {
      if (node.contains("nodeType") && node["nodeType"] == "NewExpression")
      {
        if (node.contains("typeName"))
        {
          typet new_type;
          if (
            get_type_description(
              node["typeName"]["typeDescriptions"], new_type))
          {
            log_error("failed to obtain typeDescriptions");
            abort();
          }
          if (get_sol_type(new_type) == SolidityGrammar::SolType::CONTRACT)
          {
            std::string contract_name = get_sol_contract(new_type);
            newContractSet.insert(contract_name);
          }
        }
      }

      for (const auto &item : node.items())
      {
        process_node(item.value());
      }
    }
    else if (node.is_array())
    {
      for (const auto &child : node)
      {
        process_node(child);
      }
    }
  };

  for (const auto &top_level_node : src_ast_json["nodes"])
  {
    if (
      top_level_node.contains("nodeType") &&
      top_level_node["nodeType"] == "ContractDefinition")
    {
      // for get_type_descriptions
      std::string old = current_baseContractName;
      current_baseContractName = top_level_node["name"].get<std::string>();
      process_node(top_level_node);
      current_baseContractName = old;
    }
  }

  // TOD harness mode: every contract-typed function parameter is
  // allocated via cpp_new at assign_param_nondet time, so every contract
  // effectively has a `new` call-site.  Force newContractSet to cover
  // all declared contracts so the mapping dispatch picks the pointer-
  // based store shape (not the global infinity-array fallback).  Kept
  // gated on an active TOD harness — non-TOD verification still relies
  // on the legacy singleton + infinity-array mapping model for
  // pre-existing tests.
  const bool tod_active =
    !config.options.get_option("tod-race-check").empty() ||
    !config.options.get_option("tod-balance-check").empty();
  if (tod_active)
  {
    for (const auto &top_level_node : src_ast_json["nodes"])
    {
      if (
        top_level_node.contains("nodeType") &&
        top_level_node["nodeType"] == "ContractDefinition" &&
        top_level_node.contains("name"))
        newContractSet.insert(top_level_node["name"].get<std::string>());
    }
  }
}

bool solidity_convertert::get_base_contract_name(
  const exprt &base,
  std::string &cname)
{
  log_debug("solidity", "\t\t@@@ get_base_contract_name");

  if (!has_sol_contract(base.type()))
  {
    log_error("cannot find base contract name");
    return true;
  }

  cname = get_sol_contract(base.type());
  return false;
}

void solidity_convertert::get_nondet_expr(const typet &t, exprt &new_expr)
{
  new_expr = exprt("sideeffect", t);
  new_expr.statement("nondet");
}

// x._ESBMC_bind_cname = _ESBMC_get_nondet_cont_name();
//                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// for high-level call, we bind the external calls with cname
// e.g.
// if(x.cname == Base)
//   _ESBMC_Object_Base.func()
// for low-level call, we bind the external calls with address
bool solidity_convertert::assign_nondet_contract_name(
  const std::string &_cname,
  exprt &new_expr)
{
  std::unordered_set<std::string> cname_set;
  unsigned int length = 0;

  cname_set = structureTypingMap[_cname];
  assert(!cname_set.empty());
  length = cname_set.size();
  if (length > 1)
  {
    // remove non-contract
    for (auto non_cname : nonContractNamesList)
    {
      if (non_cname == _cname)
        // we don't remove itself
        continue;
      if (cname_set.count(non_cname) != 0)
        cname_set.erase(non_cname);
    }
    // update length
    length = cname_set.size();
  }
  if (length == 1)
  {
    get_cname_expr(_cname, new_expr);
    return false;
  }

  locationt l;
  l.function(_cname);

  side_effect_expr_function_callt _call;
  get_library_function_call_no_args(
    "_ESBMC_get_nondet_cont_name",
    "c:@F@_ESBMC_get_nondet_cont_name",
    string_t,
    l,
    _call);

  exprt size_expr;
  size_expr = constant_exprt(
    integer2binary(length, bv_width(uint_type())),
    integer2string(length),
    uint_type());

  // convert this string array (e.g. {"base", "derive"}) to a symbol
  std::string aux_name, aux_id;
  aux_name = "$" + _cname + "_bind_cname_list";
  aux_id = "sol:@C@" + _cname + "@" + aux_name;

  if (context.find_symbol(aux_id) == nullptr)
  {
    log_error("cannot find contract cname list");
    return true;
  }
  exprt sym = symbol_expr(*context.find_symbol(aux_id));

  // _ESBMC_get_nondet_cont_name
  _call.arguments().push_back(sym);
  _call.arguments().push_back(size_expr);

  new_expr = _call;
  return false;
}

// special handle for the contract type parameter.
// we need to bind them to the contract instance _ESBMC_Object
bool solidity_convertert::assign_param_nondet(
  const nlohmann::json &decl_ref,
  side_effect_expr_function_callt &call)
{
  log_debug("solidity", "\t\tpopulating nil arguments");
  assert(decl_ref.contains("parameters"));

  nlohmann::json param_nodes = decl_ref["parameters"]["parameters"];
  unsigned int cnt = 1;
  for (const auto &p_node : param_nodes)
  {
    if (p_node.contains("typeDescriptions"))
    {
      // Mirror get_function_params: when the parameter AST carries a
      // typeName, route through the decl-aware overload so the resulting
      // type matches what the callee's parameter symbol uses (pointer for
      // arrays, not a real array_typet). Without this the harness builds
      // a different shape than the callee expects, and downstream checks
      // either silently miscompile or trip type-equality assertions.
      typet t;
      if (p_node.contains("typeName"))
      {
        if (get_type_description(
              p_node, p_node["typeName"]["typeDescriptions"], t))
          return true;
      }
      else
      {
        if (get_type_description(p_node["typeDescriptions"], t))
          return true;
      }
      if (get_sol_type(t) == SolidityGrammar::SolType::CONTRACT)
      {
        std::string base_cname = get_sol_contract(t);
        assert(!base_cname.empty());
        // Contract-typed harness parameter: emit a nondet pointer of the
        // declared contract/interface type. The SAT solver then picks any
        // value: aliasing a tracked `_ESBMC_Object_<C>` singleton (or any
        // state-var pointer of compatible type), or staying distinct from
        // them. This is the only sound model of "an external caller can
        // pass any address the EVM lets it construct" — including the
        // contract's own state-var pointers, which is the case behind the
        // pin `dispatcher_contract_param_eq_state_var_unsound_fail`.
        //
        // The previous default was `get_new_object_ctor_call` (fresh
        // `cpp_new` allocation). That was sound for the c1/c2 distinctness
        // scenario the original commit fixed, but unsound for the dual
        // direction: a real caller invoking `c.f(c.T())` makes the param
        // alias `T`, yet the fresh allocation gives a pointer statically
        // != T's pointer, so `if (param == T) {body}` was vacuously
        // unreachable and any bug in `body` was hidden from ESBMC's
        // verification (false negative, soundness violation).
        //
        // Independent nondet pointers preserve c1/c2 distinctness when the
        // solver picks different values for each call site, and witness
        // aliasing when it picks the same — both regimes reachable.
        //
        // --bound mode still routes through `build_bound_drive_helper` so
        // the dispatcher can drive state transitions on a valid instance;
        // the bound-mode soundness gap on this pattern is a separate
        // follow-up (the bound helper also fresh-allocates, but it needs
        // a real allocation to host the dispatcher's state mutations).
        if (is_bound)
        {
          symbolt drive_sym;
          if (build_bound_drive_helper(base_cname, drive_sym))
            return true;
          side_effect_expr_function_callt drive_call;
          drive_call.function() = symbol_expr(drive_sym);
          drive_call.type() = to_code_type(drive_sym.get_type()).return_type();
          drive_call.location() = drive_sym.location;
          call.arguments().push_back(drive_call);
        }
        else
        {
          exprt nondet_ptr;
          typet ptr_t = pointer_typet(symbol_typet(prefix + base_cname));
          get_nondet_expr(ptr_t, nondet_ptr);
          call.arguments().push_back(nondet_ptr);
        }
      }
      else if (
        get_sol_type(t) == SolidityGrammar::SolType::STRING && is_pointer_check)
      {
        //! specific for string, we need to explicitly assign it as
        //! nondet_string()
        // otherwise we will get invalid_object
        side_effect_expr_function_callt nondet_str;
        get_library_function_call_no_args(
          "nondet_string",
          "c:@F@nondet_string",
          string_t,
          locationt(),
          nondet_str);
        call.arguments().push_back(nondet_str);
      }
      else if (get_sol_type(t) == SolidityGrammar::SolType::BYTES_DYN)
      {
        // For `bytes calldata` / `bytes memory` entry-harness parameters,
        // build a nondet BytesDynamic via llc_nondet_bytes() so that
        // init-checks inside the callee do not fire spuriously. The helper
        // sets initialized==1 and capacity==length; length is fully nondet
        // size_t (post-T1.2). Direct user declarations (e.g. `bytes memory
        // x;`) still flow through the normal struct-init path. Contracts
        // that need a tighter length range must require() it explicitly.
        side_effect_expr_function_callt nondet_b;
        get_library_function_call_no_args(
          "llc_nondet_bytes",
          "c:@F@llc_nondet_bytes",
          t,
          locationt(),
          nondet_b);
        call.arguments().push_back(nondet_b);
      }
      else if (
        (get_sol_type(t) == SolidityGrammar::SolType::ARRAY ||
         get_sol_type(t) == SolidityGrammar::SolType::DYNARRAY) &&
        t.is_pointer() && !t.subtype().is_pointer())
      {
        // 1D array parameter (fixed or dynamic, e.g. `uint8[3] memory`,
        // `uint256[] calldata`): synthesize a backing allocation so the
        // callee body can read/write through the pointer without
        // dereferencing a nil arg. Without this the harness would call
        // `f(nil)` on every external entry that takes an array, and any
        // in-body element read or _ESBMC_arrcpy would trip a null-pointer
        // assertion against a value the user never authored. The contents
        // stay zero-initialised, which is a sound under-approximation of
        // "an external caller passed some array of the declared length".
        // Fixed-length: take N from #sol_array_size.
        // Dynamic-length: pick a small constant (kHarnessDynLen). The
        // length is recorded in the array header, so `a.length` reads
        // the same value.
        constexpr unsigned long kHarnessDynLen = 4;
        std::string sz_str = t.get("#sol_array_size").as_string();
        unsigned long sz_val =
          sz_str.empty() ? kHarnessDynLen : std::stoul(sz_str);
        std::string sz_repr = sz_str.empty() ? std::to_string(sz_val) : sz_str;
        exprt size_expr = constant_exprt(
          integer2binary(sz_val, bv_width(uint_type())), sz_repr, uint_type());

        exprt sizeof_expr;
        get_size_of_expr(t.subtype(), sizeof_expr);

        side_effect_expr_function_callt alloc;
        get_calloc_function_call(locationt(), alloc);
        alloc.arguments().push_back(size_expr);
        alloc.arguments().push_back(sizeof_expr);

        exprt cast_alloc = typecast_exprt(alloc, t);
        call.arguments().push_back(cast_alloc);
      }
      else if (
        (get_sol_type(t) == SolidityGrammar::SolType::ARRAY ||
         get_sol_type(t) == SolidityGrammar::SolType::DYNARRAY) &&
        t.is_pointer() && t.subtype().is_pointer() &&
        !t.subtype().subtype().is_pointer())
      {
        // 2D array parameter (any combination of fixed/dynamic outer and
        // inner). Routes through the _ESBMC_alloc_nested_2d helper so the
        // outer pointer's slots actually point at allocated inner buffers
        // — passing nil here would null-deref on the first `a[i][j]`
        // read, and a single-level alloc would null-deref on the second
        // index. Same bounded-content over-approximation as the 1D case.
        constexpr unsigned long kHarnessDynLen = 4;
        std::string outer_sz = t.get("#sol_array_size").as_string();
        std::string inner_sz = t.subtype().get("#sol_array_size").as_string();
        unsigned long outer_n =
          outer_sz.empty() ? kHarnessDynLen : std::stoul(outer_sz);
        unsigned long inner_n =
          inner_sz.empty() ? kHarnessDynLen : std::stoul(inner_sz);
        std::string outer_repr =
          outer_sz.empty() ? std::to_string(outer_n) : outer_sz;
        std::string inner_repr =
          inner_sz.empty() ? std::to_string(inner_n) : inner_sz;

        exprt outer_expr = constant_exprt(
          integer2binary(outer_n, bv_width(uint_type())),
          outer_repr,
          uint_type());
        exprt inner_expr = constant_exprt(
          integer2binary(inner_n, bv_width(uint_type())),
          inner_repr,
          uint_type());

        exprt elem_sizeof;
        get_size_of_expr(t.subtype().subtype(), elem_sizeof);

        side_effect_expr_function_callt alloc;
        get_library_function_call_no_args(
          "_ESBMC_alloc_nested_2d",
          "c:@F@_ESBMC_alloc_nested_2d",
          pointer_typet(empty_typet()),
          locationt(),
          alloc);
        alloc.arguments().push_back(outer_expr);
        alloc.arguments().push_back(inner_expr);
        alloc.arguments().push_back(elem_sizeof);

        exprt cast_alloc = typecast_exprt(alloc, t);
        call.arguments().push_back(cast_alloc);
      }
      else
      {
        // Scalar harness parameter (uint/int/bool/address/bytesN/enum):
        //
        // bytesN SITS HERE AND IS NOT A SCALAR, AND THAT IS A KNOWN, MEASURED
        // GAP -- recorded here rather than fixed, because the obvious fix was
        // tried and made the pipeline worse.
        //
        // bytesN lowers to `BytesStatic { unsigned char data[32]; size_t
        // length; }`, so the whole-struct nondet below leaves `.length` as free
        // as `.data`. `bytes_static_equal` returns FALSE OUTRIGHT on a length
        // mismatch (solidity_bytes.c:354), and an ABI bytesN argument has
        // length N by construction, so the model admits states the chain
        // cannot produce. MEASURED -- both of these are Solidity tautologies
        // and ESBMC reports VERIFICATION FAILED for them:
        //
        //     b == bytes32(uint256(b))
        //     !(b == bytes32(uint256(1))) && uint256(b) == 1
        //
        // PoC notes/coverage/poc/D12_Bytes32LengthFree.sol; pinned by
        // solidity_bytesn_param_length_free_knownbug.
        //
        // WHAT WAS TRIED AND REVERTED: passing
        // `bytes_static_from_uint(nondet_uint256(), N)`, which pins `.length`
        // and made both tautologies SUCCESSFUL. It also routes around
        // `get_nondet_expr` -- and the paragraph below is exactly why that is
        // not free. The parameter's value stopped being recoverable, so the
        // emitter reported it DEFAULTED, and on D11_Bytes32Equality the funnel
        // went from 3 paths / 3 cases / 0 merges to 3 paths / 1 case / 1 merge.
        // A sound model whose counterexample cannot be read back is not an
        // improvement for a pipeline whose deliverable is the test. Any real
        // fix has to pin the length AND keep the value recoverable.
        //
        // Pass an explicit nondet side-effect rather than a nil arg. The
        // sibling branches above already pass real nondet values for
        // string/bytes/array params; a nil scalar left the callee param as
        // an unassigned free symbol (symex_function.cpp:168 "is this valid?"
        // no-ops on nil), which is semantically nondet but invisible to the
        // testcase-generation machinery (collect_nondet_values only sees
        // `nondet$symex::` symbols). get_nondet_expr lowers to
        // sideeffect(nondet) -> replace_nondet mints `nondet$symex::nondetN`,
        // so the value is both sound (unchanged reachability) and recoverable
        // for witness/testcase emission.
        exprt nondet_scalar;
        get_nondet_expr(t, nondet_scalar);
        call.arguments().push_back(nondet_scalar);
      }
    }
    ++cnt;
  }

  return false;
}

// check if the target contract have at least one non-ctor external or public
// function
bool solidity_convertert::has_callable_func(const std::string &cname)
{
  return std::any_of(
    funcSignatures[cname].begin(),
    funcSignatures[cname].end(),
    [&cname](const solidity_convertert::func_sig &sig) {
      // must be public or external, even if the address is itself
      return sig.name != cname &&
             (sig.visibility == "public" || sig.visibility == "external");
    });
}

// check if there is a function with `func_name` in the contract `cname`
bool solidity_convertert::has_target_function(
  const std::string &cname,
  const std::string func_name)
{
  auto it = funcSignatures.find(cname);
  if (it == funcSignatures.end())
    return false;

  return std::any_of(
    it->second.begin(), it->second.end(), [&](const func_sig &sig) {
      return sig.name == func_name;
    });
}

solidity_convertert::func_sig solidity_convertert::get_target_function(
  const std::string &cname,
  const std::string &func_name)
{
  // Check if the contract exists in funcSignatures
  auto it = funcSignatures.find(cname);
  if (it == funcSignatures.end())
  {
    // If contract not found, return an empty func_sig
    return solidity_convertert::func_sig(
      "", "", "", code_typet(), false, false, false);
  }

  // Search for the function with the matching name
  auto &functions = it->second;
  auto func_it = std::find_if(
    functions.begin(),
    functions.end(),
    [&func_name](const solidity_convertert::func_sig &sig) {
      return sig.name == func_name;
    });

  // If function is found, return it; otherwise, return an empty func_sig
  if (func_it != functions.end())
  {
    return *func_it;
  }
  else
  {
    return solidity_convertert::func_sig(
      "",
      "",
      "",
      code_typet(),
      false,
      false,
      false); // Return an empty func_sig if not found
  }
}

bool solidity_convertert::get_high_level_member_access(
  const nlohmann::json &expr,
  const exprt &base,
  const exprt &member,
  const exprt &_mem_call,
  const bool is_func_call,
  exprt &new_expr)
{
  return get_high_level_member_access(
    expr, empty_json, base, member, _mem_call, is_func_call, new_expr);
}

bool solidity_convertert::handle_forge_std_assert(
  const std::string &name,
  const nlohmann::json &expr,
  const locationt &l,
  exprt &new_expr,
  bool &handled)
{
  handled = false;

  // Arity-1 boolean assertions.
  if (name == "assertTrue" || name == "assertFalse")
  {
    if (!expr.contains("arguments") || expr["arguments"].size() < 1)
      return false;
    nlohmann::json blit = {
      {"typeIdentifier", "t_bool"}, {"typeString", "bool"}};
    exprt x;
    if (get_expr(expr["arguments"][0], blit, x))
      return true;
    exprt cond = (name == "assertFalse") ? (exprt)not_exprt(x) : x;
    code_assertt a(cond);
    a.location() = l;
    new_expr = a;
    handled = true;
    log_warning("[foundry] lowered forge-std {} to assert", name);
    return false;
  }

  // Arity-3 approximate-equality assertion: assertApproxEqAbs(a, b, maxDelta)
  // passes iff |a - b| <= maxDelta (boundary inclusive, order-independent —
  // verified vs real forge). The `*Decimal` variant appends a `decimals` arg
  // that only affects failure-message formatting (ignored, like the other
  // *Decimal asserts). We encode the absolute difference as a ternary
  //   (a >= b ? a - b : b - a)
  // which is underflow-free for unsigned operands (the dominant real usage) and
  // matches forge-std's stdMath.delta exactly. The delta is cast to maxDelta's
  // type before the comparison. (Extreme int256 spans whose difference overflows
  // the signed operand type are a residual — real forge-std widens internally;
  // typical test magnitudes are unaffected.)
  if (name == "assertApproxEqAbs" || name == "assertApproxEqAbsDecimal")
  {
    if (!expr.contains("arguments") || expr["arguments"].size() < 3)
      return false;
    const nlohmann::json &a0j = expr["arguments"][0];
    const nlohmann::json &a1j = expr["arguments"][1];
    const nlohmann::json &a2j = expr["arguments"][2];
    nlohmann::json t0 =
      a0j.contains("typeDescriptions")
        ? a0j["typeDescriptions"]
        : nlohmann::json{
            {"typeIdentifier", "t_uint256"}, {"typeString", "uint256"}};
    exprt a0, a1, maxd;
    if (get_expr(a0j, t0, a0))
      return true;
    if (get_expr(a1j, t0, a1))
      return true;
    if (get_expr(a2j, t0, maxd))
      return true;
    solidity_gen_typecast(ns, a1, a0.type());
    // |a - b| via ternary (underflow-free for unsigned).
    exprt ge = binary_relation_exprt(a0, ">=", a1);
    exprt fwd = exprt("-", a0.type());
    fwd.copy_to_operands(a0, a1);
    exprt bwd = exprt("-", a0.type());
    bwd.copy_to_operands(a1, a0);
    exprt absdiff = if_exprt(ge, fwd, bwd);
    solidity_gen_typecast(ns, absdiff, maxd.type());
    exprt cond = binary_relation_exprt(absdiff, "<=", maxd);
    code_assertt a(cond);
    a.location() = l;
    new_expr = a;
    handled = true;
    log_warning("[foundry] lowered forge-std {} to assert", name);
    return false;
  }

  // Arity-2 comparison assertions.
  // The `*Decimal` variants have an extra trailing `decimals` argument that only
  // affects failure-message formatting, NOT the pass/fail verdict (verified vs
  // real forge), so they map to the same comparison on args[0]/args[1] and the
  // 3rd arg is simply ignored. `assertEqUint` is an alias of `assertEq`.
  bool is_eq = false, is_neq = false;
  const char *rel = nullptr;
  if (name == "assertEq" || name == "assertEqUint" || name == "assertEqDecimal")
    is_eq = true;
  else if (
    name == "assertNotEq" || name == "assertNeq" ||
    name == "assertNotEqDecimal")
    is_neq = true;
  else if (name == "assertGt" || name == "assertGtDecimal")
    rel = ">";
  else if (name == "assertGe" || name == "assertGeDecimal")
    rel = ">=";
  else if (name == "assertLt" || name == "assertLtDecimal")
    rel = "<";
  else if (name == "assertLe" || name == "assertLeDecimal")
    rel = "<=";
  else
    return false; // not a recognized forge-std assertion

  if (!expr.contains("arguments") || expr["arguments"].size() < 2)
    return false;

  // Convert arg0 with its own type; hint arg1 with arg0's type so a literal
  // second operand (the idiomatic `assertEq(actual, expectedLiteral)`) is
  // typed to match. Reconcile with a typecast for the comparison.
  const nlohmann::json &a0j = expr["arguments"][0];
  const nlohmann::json &a1j = expr["arguments"][1];
  nlohmann::json t0 =
    a0j.contains("typeDescriptions")
      ? a0j["typeDescriptions"]
      : nlohmann::json{
          {"typeIdentifier", "t_uint256"}, {"typeString", "uint256"}};
  exprt a0, a1;
  if (get_expr(a0j, t0, a0))
    return true;
  if (get_expr(a1j, t0, a1))
    return true;
  solidity_gen_typecast(ns, a1, a0.type());

  // Reference-type equality (assertEq / assertNotEq only). The generic
  // equality_exprt below compares heap references: for dynamic `bytes` and for
  // arrays it would treat two content-equal-but-distinct operands as UNEQUAL —
  // a false WRONG (reports a correct test as FAILED), violating the never-
  // false-WRONG contract. Handle these operand types explicitly:
  //   - dynamic `bytes`: lower to the precise content comparison via the
  //     C-model helper bytes_dynamic_equal(a, b, $dynamic_pool);
  //   - arrays (`T[]` / `T[N]`): no cheap content-equality lowering exists, so
  //     conservatively PRUNE (ASSUME(false)) rather than emit a false-WRONG
  //     reference assert. This may render the assertion vacuous (an array
  //     mismatch bug can be missed) but never reports a correct test as WRONG.
  if (is_eq || is_neq)
  {
    if (is_bytes_type(a0.type()))
    {
      // Mirror the existing `bytes == bytes` lowering (solidity_convert_expr.cpp
      // BO_EQ/BO_NE): operands are passed by address via aux vars, the pool by
      // member, and the return type is the C bool (`bool_t`), not bool_typet().
      side_effect_expr_function_callt beq;
      get_library_function_call_no_args(
        "bytes_dynamic_equal", "c:@F@bytes_dynamic_equal", bool_t, l, beq);
      exprt a0_tmp = make_aux_var(a0, l);
      exprt a1_tmp = make_aux_var(a1, l);
      beq.arguments().push_back(address_of_exprt(a0_tmp));
      beq.arguments().push_back(address_of_exprt(a1_tmp));
      exprt pool;
      if (get_dynamic_pool(expr, pool))
        return true;
      beq.arguments().push_back(pool);
      // The helper returns a C `_Bool` (a bitvector); coerce it to the SMT
      // boolean sort so it is a valid assert/negation operand.
      exprt pred = typecast_exprt(beq, bool_typet());
      exprt cond = is_neq ? (exprt)not_exprt(pred) : pred;
      code_assertt a(cond);
      a.location() = l;
      new_expr = a;
      handled = true;
      log_warning(
        "[foundry] lowered forge-std {} on bytes to bytes_dynamic_equal", name);
      return false;
    }
    const std::string tid = t0.value("typeIdentifier", "");
    if (tid.rfind("t_array", 0) == 0)
    {
      exprt f = false_exprt();
      code_assumet ap(f);
      ap.location() = l;
      new_expr = ap;
      handled = true;
      log_warning(
        "[foundry] {} on array operands -> path pruned (no content-equality "
        "lowering; conservative, no false WRONG)",
        name);
      return false;
    }
  }

  exprt cond;
  if (is_eq)
    cond = equality_exprt(a0, a1);
  else if (is_neq)
    cond = binary_relation_exprt(a0, "notequal", a1);
  else
    cond = binary_relation_exprt(a0, rel, a1);

  code_assertt a(cond);
  a.location() = l;
  new_expr = a;
  handled = true;
  log_warning("[foundry] lowered forge-std {} to assert", name);
  return false;
}

bool solidity_convertert::handle_foundry_cheatcode(
  const nlohmann::json &expr,
  const locationt &l,
  exprt &new_expr,
  bool &handled)
{
  handled = false;

  // Cheatcode name = the MemberAccess memberName on the FunctionCall callee.
  if (
    !expr.contains("expression") || !expr["expression"].is_object() ||
    !expr["expression"].contains("memberName"))
    return false;
  const std::string m = expr["expression"]["memberName"].get<std::string>();

  // vm.expectRevert([selector|data]) — arm the expectation that the NEXT
  // external call reverts. Handled before the one-arg guard because it may take
  // zero args. The selector/return-data payload is ignored (conservative: a
  // call that reverts with a *different* error still satisfies this, so we may
  // miss a wrong test, but never emit a false WRONG). Consumed at the next
  // external call site (get_high_level_member_access) which injects
  // `assert(_ESBMC_sol_reverted_flag)` after the call.
  if (m == "expectRevert")
  {
    pending_expect_revert = true;
    new_expr = code_skipt();
    handled = true;
    log_warning(
      "[foundry] armed vm.expectRevert (next external call must revert)");
    return false;
  }

  // All modeled cheatcodes here take exactly one argument.
  if (
    !expr.contains("arguments") || !expr["arguments"].is_array() ||
    expr["arguments"].empty())
    return false;

  // vm.assume(cond) -> path-pruning ASSUME (sound: assume only prunes, so it
  // can never cause a false WRONG). Enables bounded property/fuzz tests.
  if (m == "assume")
  {
    nlohmann::json blit = {
      {"typeIdentifier", "t_bool"}, {"typeString", "bool"}};
    exprt cond;
    if (get_expr(expr["arguments"][0], blit, cond))
      return true;
    code_assumet a(cond);
    a.location() = l;
    new_expr = a;
    handled = true;
    log_warning("[foundry] modeled cheatcode vm.assume");
    return false;
  }

  // Block/tx-environment setters — each is a single deterministic global
  // assignment (the simplest, least-error-prone cheatcodes). Each verified
  // against real `forge test`. All take one argument; coinbase is address-typed,
  // the rest uint256 — handled generically by converting with the arg's own type.
  //   vm.warp(t)       -> block_timestamp = t
  //   vm.roll(n)       -> block_number    = n
  //   vm.fee(x)        -> block_basefee   = x
  //   vm.chainId(x)    -> block_chainid   = x
  //   vm.prevrandao(x) -> block_prevrandao  = x
  //   vm.txGasPrice(x) -> tx_gasprice     = x
  //   vm.coinbase(a)   -> block_coinbase  = a
  // NOTE: vm.difficulty is deliberately NOT modeled — real forge reverts on it
  // after the Paris hard fork ("use prevrandao instead", EIP-4399), so modeling
  // it as a set would DISAGREE with `forge test`. It falls through to prune.
  const char *global = nullptr;
  if (m == "warp")
    global = "c:@block_timestamp";
  else if (m == "roll")
    global = "c:@block_number";
  else if (m == "fee")
    global = "c:@block_basefee";
  else if (m == "chainId")
    global = "c:@block_chainid";
  else if (m == "prevrandao")
    global = "c:@block_prevrandao";
  else if (m == "txGasPrice")
    global = "c:@tx_gasprice";
  else if (m == "coinbase")
    global = "c:@block_coinbase";
  else
  {
    // Conservative hard-taint gate (design-plan F1.0). We cannot model this
    // cheatcode's effect, so the continuation is unknown; prune it with
    // ASSUME(false). Assertions BEFORE this reached point are still checked
    // (reachability-sensitive), so a real pre-cheatcode bug is still found;
    // only the un-modelable suffix is suppressed. This can never introduce a
    // false WRONG (the never-false-WRONG invariant), at the cost of missing
    // bugs downstream of an unmodeled cheatcode (completeness, acceptable for
    // the conservative "report CORRECT unless definitely WRONG" contract).
    exprt f = false_exprt();
    code_assumet a(f);
    a.location() = l;
    new_expr = a;
    handled = true;
    log_warning(
      "[foundry] UNMODELED cheatcode vm.{} -> path pruned "
      "(conservative; no false WRONG, downstream bugs not explored)",
      m);
    return false;
  }

  const symbolt *g = context.find_symbol(global);
  if (g == nullptr)
    return true;
  exprt lhs = symbol_expr(*g);

  exprt arg0;
  const nlohmann::json &a0j = expr["arguments"][0];
  nlohmann::json lit =
    a0j.contains("typeDescriptions")
      ? a0j["typeDescriptions"]
      : nlohmann::json{
          {"typeIdentifier", "t_uint256"}, {"typeString", "uint256"}};
  if (get_expr(a0j, lit, arg0))
    return true;
  solidity_gen_typecast(ns, arg0, lhs.type());

  // Hand back an unconverted `assign` side-effect (mirrors the high-level call
  // result at the msg_sender-prank path); the caller converts it to code.
  exprt assign = side_effect_exprt("assign", lhs.type());
  assign.location() = l;
  assign.copy_to_operands(lhs, arg0);
  new_expr = assign;
  handled = true;
  log_warning("[foundry] modeled cheatcode vm.{} (block env setter)", m);
  return false;
}

/** 
 * Conversion: 
  constructor()
  {
    this->_ESBMC_bind_cname = get_nondet_cname(); // unless we have a new
 Base(), then = Base;
  }

  function test1(Base x, address _addr) public
  {
      x = new Base();     // x._ESBMC_bind_cname = Base;
      x.test();           // if x._ESBMC_bind_cname == base
                          //   _ESBMC_Object_base.test();
      x = Base(_addr);    // x = ESBMC_Object_base
                          // if _addr == _ESBMC_Object_base.$address
                          //   x._ESBMC_bind_cname = base
                          // if _addr == _ESBMC_Object_y.$address
                          //   x._ESBMC_bind_cname = y;
  }

  the auxilidary tmp var will not be created if the member_type is void
  @expr: the whole member access expression json
  @options: call with options
  @is_func_call: true if it's a function member access; false state variable
 access
  @_mem_call: function call statement, with arguments populated
  return true: we fail to generate the high_level_member_access bound harness
               however, this should not be treated as an erorr.
               E.g. x.access() where x is a state variable
*/
bool solidity_convertert::get_high_level_member_access(
  const nlohmann::json &expr,
  const nlohmann::json &options,
  const exprt &base,
  const exprt &member,
  const exprt &_mem_call,
  const bool is_func_call,
  exprt &new_expr)
{
  log_debug("solidity", "Getting high-level member access");
  new_expr = _mem_call;

  // get 'Base'
  std::string _cname;
  if (get_sol_type(base.type()) != SolidityGrammar::SolType::CONTRACT)
  {
    log_error("Expecting contract type");
    base.type().dump();
    return true;
  }
  if (get_base_contract_name(base, _cname))
    return true;

  // current contract name
  std::string cname;
  get_current_contract_name(expr, cname);

  // current this pointer reference
  exprt cur_this_expr;
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, cur_this_expr))
      return true;
  }
  else
  {
    if (get_ctor_decl_this_ref(expr, cur_this_expr))
      return true;
  }

  locationt l;
  get_location_from_node(expr, l);

  // Foundry cheatcode interception: `vm.<name>(...)` on the forge-std cheatcode
  // handle is not a real external call. The handle may be typed `Vm` or its
  // parent `VmSafe` (view/pure cheatcodes) — both must be gated so no cheatcode
  // escapes to the external-call harness (see solidity_convert_expr.cpp).
  if (is_func_call && (_cname == "Vm" || _cname == "VmSafe"))
  {
    bool handled = false;
    if (handle_foundry_cheatcode(expr, l, new_expr, handled))
      return true;
    if (handled)
      return false;
  }

  std::unordered_set<std::string> cname_set = structureTypingMap[_cname];
  assert(!cname_set.empty());
  if (cname_set.size() > 1)
  {
    // remove non-contract
    for (auto non_cname : nonContractNamesList)
    {
      if (non_cname == _cname)
        // we don't remove itself
        continue;
      if (cname_set.count(non_cname) != 0)
        cname_set.erase(non_cname);
    }
  }

  exprt balance;
  bool is_call_w_options = is_func_call && options.is_array();
  if (is_call_w_options)
  {
    // this can be value, gas ...
    // For now, we only consider value
    nlohmann::json literal_type = {
      {"typeIdentifier", "t_uint256"}, {"typeString", "uint256"}};
    if (get_expr(options[0], literal_type, balance))
      return true;
  }

  if (cname_set.size() == 1)
  {
    // skip the "if(..)"
    if (is_func_call)
    {
      // wrap it
      exprt front_block = code_blockt();
      exprt back_block = code_blockt();
      if (is_call_w_options)
      {
        if (
          model_transaction(
            expr, cur_this_expr, base, balance, l, front_block, back_block))
        {
          log_error("failed to model the transaction property changes");
          return true;
        }
      }
      else if (
        get_high_level_call_wrapper(
                 cname,
                 cur_this_expr,
                 front_block,
                 back_block,
                 nonContractNamesList.count(cname) != 0))
        return true;

      for (auto op : front_block.operands())
        move_to_front_block(op);
      for (auto op : back_block.operands())
        move_to_back_block(op);
    }

    return false; // since it has only one possible option, no need to futher
                  // binding
  }

  // now we need to consider the binding

  if (get_sol_type(member.type()) == SolidityGrammar::SolType::TUPLE_RETURNS)
  {
    log_error("Unsupported return tuple");
    return true;
  }

  bool is_return_void = member.type().is_empty() ||
                        (member.type().is_code() &&
                         to_code_type(member.type()).return_type().is_empty());

  // construct auxiliary function
  // e.g.
  //  Bank target;
  //  target.withdraw()
  // => Bank_withdraw(this, this->target)
  assert(!_cname.empty());
  assert(!member.name().empty());
  std::string fname = _cname + "_" + member.name().as_string();
  std::string fid = "sol:@C@" + cname + "@F@" + fname + "#";
  code_typet ft;
  if (!is_return_void)
  {
    if (is_func_call)
      ft.return_type() = to_code_type(member.type()).return_type();
    else
      ft.return_type() = member.type();
  }
  else
    ft.return_type() = empty_typet();
  symbolt fs;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(fs, debug_modulename, ft, fname, fid, locationt());
  fs.lvalue = true;
  fs.is_extern = false;
  fs.file_local = true;
  auto &added_fsymbol = *move_symbol_to_context(fs);

  // add this pointer to arguments
  get_function_this_pointer_param(
    cname, fid, debug_modulename, locationt(), ft);
  // add base to arguments
  code_typet::argumentt base_param;
  std::string base_name = "base";
  std::string base_id =
    "sol:@C@" + cname + "@F@" + fname + "@" + base_name + "#";
  base_param.cmt_base_name(base_name);
  base_param.cmt_identifier(base_id);

  base_param.type() = base.type();
  symbolt param_symbol;
  get_default_symbol(
    param_symbol,
    debug_modulename,
    base_param.type(),
    base_name,
    base_id,
    locationt());
  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;
  if (context.find_symbol(base_id) == nullptr)
  {
    context.move_symbol_to_context(param_symbol);
  }

  ft.arguments().push_back(base_param);
  exprt new_base = symbol_expr(*context.find_symbol(base_id));

  added_fsymbol.set_type(migrate_type(ft));
  //! we need to move it to the struct symbol
  // this is because we use the member from the contract
  move_builtin_to_contract(cname, symbol_expr(added_fsymbol), true);

  exprt this_expr;
  if (get_func_decl_this_ref(cname, fid, this_expr))
    return true;

  // function body

  // add esbmc_hide label
  exprt func_body = code_blockt();
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.move_to_operands(label);

  // get 'x._ESBMC_bind_cname'
  exprt bind_expr = member_exprt(new_base, "_ESBMC_bind_cname", string_t);

  // get memebr type
  exprt tmp;
  if (!is_return_void)
  {
    std::string aux_name, aux_id;
    aux_name =
      "$return_" + base.name().as_string() + "_" + member.name().as_string();
    aux_id = "sol:@" + aux_name + std::to_string(aux_counter++);
    symbolt s;
    typet t = ft.return_type();
    if (t.id() == irept::id_code)
      t = to_code_type(t).return_type();
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    get_default_symbol(
      s, debug_modulename, t, aux_name, aux_id, member.location());
    auto &added_symbol = *move_symbol_to_context(s);
    s.lvalue = true;
    s.file_local = true;
    code_declt decl(symbol_expr(added_symbol));

    tmp = symbol_expr(added_symbol);
    func_body.move_to_operands(decl);
  }

  // rhs
  // @str: contract name
  for (auto str : cname_set)
  {
    // strcmp（_ESBMC_NODET_cont_name, Base)
    exprt cname_string;
    typet ct = string_t;
    ct.cmt_constant(true);
    get_symbol_decl_ref(str, "sol:@" + str, ct, cname_string);

    // since we do not modify the string, and it always point to the known
    // object
    exprt _cmp_cname = exprt("=", string_t);
    _cmp_cname.operands().push_back(bind_expr);
    _cmp_cname.operands().push_back(cname_string);

    // member access
    exprt memcall;
    exprt rhs;

    exprt _base;
    get_static_contract_instance_ref(str, _base);

    // ?fix address?. e.g.
    // B target = B(_addr); // previously
    // base->$address =  _ESBMC_Object_B.$address // note that pointer
    // this->target == base

    bool is_revert = false;
    if (is_func_call)
    {
      // e.g. x.call() y.call(). we need to find the definition of the call
      // beyond the contract x/y separately get call
      std::string func_name = member.name().as_string();
      assert(!func_name.empty());
      const nlohmann::json &member_decl_ref = get_func_decl_ref(str, func_name);
      if (member_decl_ref == empty_json)
        continue;

      exprt comp;
      if (get_func_decl_ref(member_decl_ref, comp))
        return true;

      side_effect_expr_function_callt call;
      if (get_non_library_function_call(member_decl_ref, expr, call))
        return true;

      // func(&this) => func(&_ESBMC_Object_str)
      call.arguments().at(0) = _base;
      memcall = call;
    }
    else
    {
      assert(!member.name().empty());

      // Verify the target struct actually has this component. solc 0.6.x
      // using-for attaches library methods to contract-typed variables,
      // producing a high-level member access whose cname_set can include
      // the library itself (UniERC20). Libraries have no state, so e.g.
      // an accidental `$address` access on them would abort in
      // get_component_number. Treat a missing component as revert.
      bool struct_has_member = false;
      const symbolt *struct_sym = context.find_symbol(prefix + str);
      if (struct_sym != nullptr && struct_sym->get_type().id() == "struct")
        struct_has_member =
          to_struct_type(struct_sym->get_type()).has_component(member.name());

      if (struct_has_member && inheritanceMap[_cname].count(str))
        memcall = member_exprt(_base, member.name(), member.type());
      else if (
        struct_has_member &&
        is_var_getter_matched(str, member.name().as_string(), member.type()))
      {
        memcall = member_exprt(_base, member.name(), member.type());
      }
      else
      {
        // this should be a revert
        // however, esbmc-kind havs trouble in __ESBMC_asusme(false) (v7.8)
        side_effect_expr_function_callt call;
        get_library_function_call_no_args(
          "__ESBMC_assume", "c:@F@__ESBMC_assume", empty_typet(), l, call);

        exprt arg = false_exprt();
        call.arguments().push_back(arg);
        memcall = call;
        is_revert = true;
      }
    }
    rhs = memcall;
    if (!is_return_void && !is_revert)
    {
      exprt _assign = side_effect_exprt("assign", tmp.type());
      convert_type_expr(ns, memcall, tmp, expr);
      _assign.copy_to_operands(tmp, memcall);
      rhs = _assign;
    }
    convert_expression_to_code(rhs);

    // wrap it
    if (is_func_call)
    {
      exprt front_block = code_blockt();
      exprt back_block = code_blockt();
      if (is_call_w_options)
      {
        // Credit the SAME object the method body executes on. The multi-type
        // dispatcher runs the call on the static singleton `_base`
        // (`call.arguments().at(0) = _base` above), so for the value transfer
        // to be visible to the body's own `this->$balance` the credit must
        // land on `_base->$balance`, not the dynamic-instance pointer
        // `new_base`. Crediting `new_base` left the body's balance at 0, so a
        // subsequent `.call{value:a}` read 0 and reverted — silently dropping
        // every `{value:}` transfer to a `new`-created instance whenever >=2
        // contract types shared the method name (which is exactly the shape of
        // a differential harness: P and M are two such types).
        if (
          model_transaction(
              expr, this_expr, _base, balance, l, front_block, back_block))
        {
          log_error("failed to model the transaction property changes");
          return true;
        }
      }
      else if (get_high_level_call_wrapper(
                 cname,
                 this_expr,
                 front_block,
                 back_block,
                 nonContractNamesList.count(cname) != 0))
        return true;

      // Record the calling instance's $address in the global
      // `_ESBMC_caller_inst_addr` for the DURATION of this call (save before,
      // restore after), so the low-level `.call`/`$call` builders can present
      // it as the reentrant msg.sender — making a callback see
      // msg.sender == address(p). Gated on a polymorphic cluster
      // (cname_set.size() > 1): single-type calls run the body on the instance
      // already, so this->$address is correct without it. Only this dedicated
      // global is touched (no registered $address changes), so address-keyed
      // transfer/EOA dispatch and the sol_addr_array lookups are unaffected.
      if (cname_set.size() > 1 && str == _cname)
      {
        const symbolt *cia_sym_p =
          context.find_symbol("c:@_ESBMC_caller_inst_addr");
        if (cia_sym_p != nullptr)
        {
          exprt cia = symbol_expr(*cia_sym_p);
          exprt inst_addr = member_exprt(new_base, "$address", addr_t);

          std::string sv_name = "_saved_caller_inst_addr";
          std::string sv_id = "sol:@" + sv_name + std::to_string(aux_counter++);
          symbolt sv;
          get_default_symbol(
            sv,
            get_modulename_from_path(absolute_path),
            addr_t,
            sv_name,
            sv_id,
            member.location());
          sv.lvalue = true;
          sv.file_local = true;
          exprt sv_sym = symbol_expr(*move_symbol_to_context(sv));
          code_declt sv_decl(sv_sym);

          exprt save = side_effect_exprt("assign", addr_t);
          save.copy_to_operands(sv_sym, cia);
          exprt set = side_effect_exprt("assign", addr_t);
          set.copy_to_operands(cia, inst_addr);
          exprt restore = side_effect_exprt("assign", addr_t);
          restore.copy_to_operands(cia, sv_sym);
          convert_expression_to_code(save);
          convert_expression_to_code(set);
          convert_expression_to_code(restore);

          back_block.operands().push_back(restore);
          front_block.operands().insert(front_block.operands().begin(), set);
          front_block.operands().insert(front_block.operands().begin(), save);
          front_block.operands().insert(
            front_block.operands().begin(), sv_decl);
        }
      }

      // if-body
      code_blockt block;
      for (auto &op : front_block.operands())
        block.move_to_operands(op);
      block.move_to_operands(rhs);
      for (auto &op : back_block.operands())
        block.move_to_operands(op);
      rhs = block;
    }
    else
    {
      code_blockt block;
      block.move_to_operands(rhs);
      rhs = block;
    }

    codet if_expr("ifthenelse");
    if_expr.move_to_operands(_cmp_cname, rhs);
    if_expr.location() = l;
    //? empty file?
    if_expr.location().file("");
    func_body.move_to_operands(if_expr);
  }

  // return
  if (!is_return_void)
  {
    code_returnt _ret;
    _ret.return_value() = tmp;
    func_body.move_to_operands(_ret);
  }

  added_fsymbol.set_value(migrate_expr(func_body));

  // construct function call
  side_effect_expr_function_callt _call;
  _call.function() = symbol_expr(added_fsymbol);
  _call.type() = ft.return_type();
  _call.location() = l;
  // bank_withdraw(this, this->target)
  _call.arguments().push_back(cur_this_expr);
  _call.arguments().push_back(base);

  new_expr = _call;

  log_debug("solidity", "\tSuccessfully modelled member access.");
  return false;
}

/**
 * Resolve a low-level call (.call/.send/.transfer/.delegatecall/.staticcall)
 * in bound mode by finding the enclosing FunctionCall AST node, extracting
 * arguments, and dispatching to get_low_level_member_accsss.
 *
 * e.g.   x.call{value: val}("")
 * @base:     (this->)x
 * @mem_name: call
 */
bool solidity_convertert::get_bound_low_level_call(
  const nlohmann::json &expr,
  const nlohmann::json &literal_type,
  const std::string &mem_name,
  const exprt &base,
  exprt &new_expr)
{
  // Walk up the AST to find the enclosing FunctionCall node.
  // May need to skip an intermediate FunctionCallOptions node (for {value: X}).
  const nlohmann::json &initial_func_call =
    find_last_parent(src_ast_json["nodes"], expr);
  const nlohmann::json *func_call = &initial_func_call;

  if ((*func_call)["nodeType"] == "FunctionCallOptions")
  {
    const nlohmann::json &second_call =
      find_last_parent(src_ast_json["nodes"], initial_func_call);
    func_call = &second_call;
  }

  assert((*func_call)["nodeType"] == "FunctionCall");

  if ((*func_call).empty() || (*func_call).is_null())
  {
    log_error("failed to resolve function call in member access");
    return true;
  }

  // Fast path: if this is a plain .call with a literal
  // abi.encodeWithSignature(...) payload, we can dispatch by signature
  // directly and bypass the generic $call#0 helper. Falls through on any
  // shape we don't recognise.
  if (mem_name == "call")
  {
    if (!try_get_signature_dispatched_call(expr, *func_call, base, new_expr))
      return false;
  }

  // Delegate-shadow fast path: .delegatecall(abi.encodeWithSignature(...))
  // with a literal signature and caller/target state-var compatibility.
  // Falls through to the generic $delegatecall#0 helper on failure.
  if (mem_name == "delegatecall")
  {
    if (!try_get_delegate_shadow_call(expr, *func_call, base, new_expr))
      return false;
  }

  exprt arg = nil_exprt();
  assert((*func_call).contains("arguments"));

  if ((*func_call)["arguments"].size() > 0)
  {
    auto &arguments = (*func_call)["arguments"][0];
    if (get_expr(arguments, expr["argumentTypes"][0], arg))
      return true;
  }

  return get_low_level_member_accsss(
    expr, literal_type, mem_name, base, arg, new_expr);
}

/**
 * @options: val
 * @arg: ""
 */
bool solidity_convertert::get_low_level_member_accsss(
  const nlohmann::json &expr,
  const nlohmann::json &options,
  const std::string mem_name,
  const exprt &base,
  const exprt &arg,
  exprt &new_expr)
{
  log_debug("solidity", "Getting low-level member access");

  locationt loc;
  get_location_from_node(expr, loc);
  side_effect_expr_function_callt call;

  std::string cname;
  get_current_contract_name(expr, cname);
  if (cname.empty())
    return true;

  // get this
  exprt this_object;
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, this_object))
      return true;
  }
  else if (!expr.empty())
  {
    if (get_ctor_decl_this_ref(expr, this_object))
      return true;
  }
  else
  {
    log_error("cannot get this object ref");
    return true;
  }

  if (mem_name == "call")
  {
    std::string func_name = "call";
    exprt addr = base;
    if (options != nullptr)
    {
      // do call#1(this, addr, value) (call with ether)
      set_sol_type(addr.type(), SolidityGrammar::SolType::ADDRESS_PAYABLE);
      exprt value;
      // type should be uint256
      nlohmann::json literal_type = {
        {"typeIdentifier", "t_uint256"}, {"typeString", "uint256"}};

      if (get_expr(options[0], literal_type, value))
        return true;

      // --reentry-balance-drain-check: emit pre-snapshots and post-assert
      // around the call.  `value_for_call` is the symbol the caller MUST
      // pass into call.arguments() so the user-side value expression is
      // evaluated exactly once (the assert references the same snapshot).
      exprt value_for_call;
      if (emit_balance_drain_wrapper(
            cname, this_object, value, loc, value_for_call))
        return true;

      std::string func_id = "sol:@C@" + cname + "@F@$call#1";

      get_library_function_call_no_args(func_name, func_id, bool_t, loc, call);
      call.arguments().push_back(this_object);
      call.arguments().push_back(addr);
      call.arguments().push_back(value_for_call);
    }
    else
    {
      // To call#0(this, addr)
      set_sol_type(addr.type(), SolidityGrammar::SolType::ADDRESS);

      std::string func_id = "sol:@C@" + cname + "@F@$call#0";
      get_library_function_call_no_args(func_name, func_id, bool_t, loc, call);
      call.arguments().push_back(this_object);
      call.arguments().push_back(addr);
    }

    // Wire the dispatch's own boolean return value into the tuple's
    // `success` slot.  $call#0/#1 bodies return `true` on the
    // tracked-target success path (the target's receive/fallback was
    // invoked and our model completed it) and `false` / `nondet_bool`
    // on the EOA fallthrough depending on variant.  Previously we
    // discarded the return and the tuple's `success` was a fresh
    // `nondet_bool` initialised by `get_llc_ret_tuple`, so user code
    // doing `(bool ok, ) = addr.call(...)` observed an arbitrary
    // value — making it impossible to prove `ok` true on a tracked-
    // target success path.  Now the dispatch value is propagated
    // deterministically and the over-approx lives only on the paths
    // where the $call body itself emits `nondet_bool` (EOA, library
    // unknown).
    symbolt dump;
    get_llc_ret_tuple(dump);
    exprt dump_expr = symbol_expr(dump);

    // The tuple's `success` field is laid out as `unsigned int` (C
    // struct padding of _Bool), not raw bool — cast the bool-typed
    // dispatch return to match the member's type before assigning.
    const struct_typet &tup_stype = to_struct_type(ns.follow(dump_expr.type()));
    assert(!tup_stype.components().empty());
    const typet &x_type = tup_stype.components().front().type();
    exprt dump_x = member_exprt(dump_expr, "x", x_type);
    exprt call_casted = call;
    if (call_casted.type() != x_type)
      solidity_gen_typecast(ns, call_casted, x_type);
    exprt assign_succ = side_effect_exprt("assign", x_type);
    assign_succ.copy_to_operands(dump_x, call_casted);
    convert_expression_to_code(assign_succ);
    move_to_front_block(assign_succ);

    new_expr = dump_expr;
  }
  else if (mem_name == "transfer")
  {
    // transfer(this, to_addr, balance_value)
    exprt addr = base;
    assert(!arg.is_nil());

    // --reentry-balance-drain-check: see emit_balance_drain_wrapper
    exprt value_for_call;
    if (emit_balance_drain_wrapper(
          cname, this_object, arg, loc, value_for_call))
      return true;

    std::string func_name = "transfer";
    std::string func_id = "sol:@C@" + cname + "@F@$transfer#0";
    get_library_function_call_no_args(func_name, func_id, bool_t, loc, call);
    call.arguments().push_back(this_object);
    call.arguments().push_back(addr);
    call.arguments().push_back(value_for_call);

    new_expr = call;
  }
  else if (mem_name == "send")
  {
    // send(this, to_addr, balance_value)
    exprt addr = base;
    assert(!arg.is_nil());

    // --reentry-balance-drain-check: see emit_balance_drain_wrapper
    exprt value_for_call;
    if (emit_balance_drain_wrapper(
          cname, this_object, arg, loc, value_for_call))
      return true;

    std::string func_name = "send";
    std::string func_id = "sol:@C@" + cname + "@F@$send#0";
    get_library_function_call_no_args(func_name, func_id, bool_t, loc, call);
    call.arguments().push_back(this_object);
    call.arguments().push_back(addr);
    call.arguments().push_back(value_for_call);

    new_expr = call;
  }
  else if (mem_name == "staticcall")
  {
    // staticcall(this, addr) — read-only call, same dispatch as call#0
    exprt addr = base;
    set_sol_type(addr.type(), SolidityGrammar::SolType::ADDRESS);

    std::string func_name = "staticcall";
    std::string func_id = "sol:@C@" + cname + "@F@$staticcall#0";
    get_library_function_call_no_args(func_name, func_id, bool_t, loc, call);
    call.arguments().push_back(this_object);
    call.arguments().push_back(addr);

    // Wire the dispatch return into the tuple's success slot (see
    // call case above for rationale).
    symbolt dump;
    get_llc_ret_tuple(dump);
    exprt dump_expr = symbol_expr(dump);

    // The tuple's `success` field is laid out as `unsigned int` (C
    // struct padding of _Bool), not raw bool — cast the bool-typed
    // dispatch return to match the member's type before assigning.
    const struct_typet &tup_stype = to_struct_type(ns.follow(dump_expr.type()));
    assert(!tup_stype.components().empty());
    const typet &x_type = tup_stype.components().front().type();
    exprt dump_x = member_exprt(dump_expr, "x", x_type);
    exprt call_casted = call;
    if (call_casted.type() != x_type)
      solidity_gen_typecast(ns, call_casted, x_type);
    exprt assign_succ = side_effect_exprt("assign", x_type);
    assign_succ.copy_to_operands(dump_x, call_casted);
    convert_expression_to_code(assign_succ);
    move_to_front_block(assign_succ);

    new_expr = dump_expr;
  }
  else if (mem_name == "delegatecall")
  {
    // delegatecall(this, addr) — runs in caller's context,
    // msg.sender and msg.value are preserved
    exprt addr = base;
    set_sol_type(addr.type(), SolidityGrammar::SolType::ADDRESS);

    std::string func_name = "delegatecall";
    std::string func_id = "sol:@C@" + cname + "@F@$delegatecall#0";
    get_library_function_call_no_args(func_name, func_id, bool_t, loc, call);
    call.arguments().push_back(this_object);
    call.arguments().push_back(addr);

    // Wire the dispatch return into the tuple's success slot (see
    // call case above for rationale).
    symbolt dump;
    get_llc_ret_tuple(dump);
    exprt dump_expr = symbol_expr(dump);

    // The tuple's `success` field is laid out as `unsigned int` (C
    // struct padding of _Bool), not raw bool — cast the bool-typed
    // dispatch return to match the member's type before assigning.
    const struct_typet &tup_stype = to_struct_type(ns.follow(dump_expr.type()));
    assert(!tup_stype.components().empty());
    const typet &x_type = tup_stype.components().front().type();
    exprt dump_x = member_exprt(dump_expr, "x", x_type);
    exprt call_casted = call;
    if (call_casted.type() != x_type)
      solidity_gen_typecast(ns, call_casted, x_type);
    exprt assign_succ = side_effect_exprt("assign", x_type);
    assign_succ.copy_to_operands(dump_x, call_casted);
    convert_expression_to_code(assign_succ);
    move_to_front_block(assign_succ);

    new_expr = dump_expr;
  }
  else
  {
    log_error("unsupported low-level call type {}", mem_name);
    return true;
  }

  return false;
}

void solidity_convertert::get_bind_cname_func_name(
  const std::string &cname,
  std::string &fname,
  std::string &fid)
{
  fname = "initialize_" + cname + +"_bind_cname";
  fid = "sol:@F@" + fname + "#";
}

// return expr: contract_instance._ESBMC_bind_cname
bool solidity_convertert::get_bind_cname_expr(
  const nlohmann::json &json,
  exprt &bind_cname_expr)
{
  const nlohmann::json &parent = find_last_parent(src_ast_json, json);
  locationt l;
  get_location_from_node(json, l);
  exprt lvar;

  if (!parent.contains("nodeType"))
  {
    log_error("{}", parent.dump());
    abort();
  }
  if (parent["nodeType"] == "ExpressionStatement")
    return true; // e.g. new Base(); Base(_addr); with no lvalue
  else if (parent["nodeType"] == "VariableDeclarationStatement")
  {
    assert(parent.contains("declarations"));
    if (get_var_decl_ref(parent["declarations"][0], true, lvar))
      return true;
  }
  else if (parent["nodeType"] == "VariableDeclaration")
  {
    if (get_var_decl_ref(parent, true, lvar))
      return true;
  }
  else if (parent["nodeType"] == "Assignment")
  {
    if (get_expr(parent["leftHandSide"], lvar))
      return true;
  }
  else
  {
    log_warning(
      "got Unexpected nodeType: {}", parent["nodeType"].get<std::string>());
    return true;
  }

  bind_cname_expr = member_exprt(lvar, "_ESBMC_bind_cname", string_t);
  bind_cname_expr.location() = l;
  return false;
}

bool solidity_convertert::get_or_create_bind_shadow(
  const exprt &lvar,
  const std::string &declared_cname,
  exprt &shadow_out)
{
  if (lvar.id() != "symbol")
    return true;
  if (declared_cname.empty())
    return true;

  const std::string lvar_id = lvar.identifier().as_string();
  const std::string lvar_name = lvar.name().as_string();
  const std::string shadow_id = lvar_id + "$bind";
  const std::string shadow_name = lvar_name + "$bind";

  const symbolt *existing = context.find_symbol(shadow_id);
  if (existing == nullptr)
  {
    symbolt s;
    const std::string debug_mod = get_modulename_from_path(absolute_path);
    get_default_symbol(
      s, debug_mod, string_t, shadow_name, shadow_id, lvar.location());
    s.lvalue = true;
    s.file_local = true;

    exprt init_val;
    get_cname_expr(declared_cname, init_val);
    solidity_gen_typecast(ns, init_val, string_t);
    s.set_value(init_val);

    move_symbol_to_context(s);
    existing = context.find_symbol(shadow_id);
    if (existing == nullptr)
      return true;
  }

  shadow_out = symbol_expr(*existing);
  return false;
}

bool solidity_convertert::get_bind_shadow_read(
  const exprt &base,
  exprt &shadow_out)
{
  if (base.id() != "symbol")
    return true;
  const std::string shadow_id = base.identifier().as_string() + "$bind";
  const symbolt *existing = context.find_symbol(shadow_id);
  if (existing == nullptr)
    return true;
  shadow_out = symbol_expr(*existing);
  return false;
}

/**
 * symbol
 *   * identifier: tag-Bank
 */
void solidity_convertert::get_new_object(const typet &t, exprt &this_object)
{
  log_debug("solidity", "\t\tget this object ref");
  assert(t.is_symbol());

  exprt temporary = exprt("new_object");
  temporary.type() = t;
  this_object = temporary;
}

// ======================================================================
// Signature-based dispatch for .call(abi.encodeWithSignature(...))
// ----------------------------------------------------------------------
// When a low-level .call payload is a literal abi.encodeWithSignature
// invocation, we know the exact target function signature and arguments
// at translation time. In that case we bypass the generic $call#0 helper
// (which does nondet dispatch and discards the args) and instead emit a
// per-caller helper that walks every contract whose address might match,
// invokes the function with the exact signature on that contract, and
// returns true on success / false when no contract matches.
// ======================================================================

// Strip whitespace characters from a string in place.
static std::string strip_spaces(const std::string &s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
      out.push_back(c);
  return out;
}

std::string
solidity_convertert::build_canonical_signature(const nlohmann::json &func_def)
{
  if (
    !func_def.is_object() || !func_def.contains("name") ||
    !func_def.contains("parameters"))
    return "";
  std::string name = func_def["name"].get<std::string>();
  if (name.empty())
    return "";
  std::string sig = name + "(";
  const auto &params = func_def["parameters"]["parameters"];
  bool first = true;
  for (const auto &p : params)
  {
    if (!first)
      sig += ",";
    first = false;
    if (!p.contains("typeDescriptions"))
      return "";
    const auto &td = p["typeDescriptions"];
    if (!td.contains("typeString"))
      return "";
    sig += strip_spaces(td["typeString"].get<std::string>());
  }
  sig += ")";
  return sig;
}

const nlohmann::json &solidity_convertert::find_function_by_signature(
  const std::string &cname,
  const std::string &target_sig)
{
  nlohmann::json &nodes = src_ast_json["nodes"];
  for (auto &cn : nodes)
  {
    if (!cn.is_object())
      continue;
    if (cn.value("nodeType", "") != "ContractDefinition")
      continue;
    if (cn.value("name", "") != cname)
      continue;
    for (auto &sub : cn["nodes"])
    {
      if (!sub.is_object())
        continue;
      if (sub.value("nodeType", "") != "FunctionDefinition")
        continue;
      // Only externally callable functions are reachable via a low-level
      // call (private/internal are never exposed in the ABI).
      std::string vis = sub.value("visibility", "");
      if (vis != "public" && vis != "external")
        continue;
      std::string sig = build_canonical_signature(sub);
      if (sig.empty())
        continue;
      if (sig == target_sig)
        return sub;
    }
  }
  return empty_json;
}

// Resolve a JSON node to a FunctionDefinition referenced by it.
// Accepts the two forms emitted by solc when a function is used as a value:
//   Logic.f           -> MemberAccess { memberName: "f", referencedDeclaration:
//   <fn id> } freeFunction f    -> Identifier   { referencedDeclaration: <fn
//   id> }
// Uses the static `find_node_by_id` helper against the full AST so that a
// cross-contract reference like `Logic.setX` resolves from a delegatecall
// in Proxy even though current_baseContractName is still Proxy (the scoped
// find_decl_ref would miss it).
static const nlohmann::json *resolve_function_reference(
  const nlohmann::json &ast_root,
  const nlohmann::json &node)
{
  if (!node.is_object())
    return nullptr;
  const std::string nt = node.value("nodeType", "");
  if (nt != "MemberAccess" && nt != "Identifier")
    return nullptr;
  if (
    !node.contains("referencedDeclaration") ||
    !node["referencedDeclaration"].is_number_integer())
    return nullptr;
  int ref = node["referencedDeclaration"].get<int>();
  if (ref <= 0)
    return nullptr;
  const nlohmann::json &fdecl =
    solidity_convertert::find_node_by_id(ast_root, ref);
  if (fdecl.empty() || fdecl.is_null())
    return nullptr;
  if (fdecl.value("nodeType", "") != "FunctionDefinition")
    return nullptr;
  return &fdecl;
}

bool solidity_convertert::extract_abi_encode_signature(
  const nlohmann::json &payload,
  std::string &sig_literal,
  std::vector<const nlohmann::json *> &args_out)
{
  if (!payload.is_object())
    return true;
  if (payload.value("nodeType", "") != "FunctionCall")
    return true;
  if (!payload.contains("expression"))
    return true;
  const auto &callee = payload["expression"];
  if (!callee.is_object() || callee.value("nodeType", "") != "MemberAccess")
    return true;
  if (!callee.contains("expression"))
    return true;
  const auto &base_expr = callee["expression"];
  if (
    !base_expr.is_object() || base_expr.value("nodeType", "") != "Identifier" ||
    base_expr.value("name", "") != "abi")
    return true;
  const std::string encoder = callee.value("memberName", "");
  if (
    encoder != "encodeWithSignature" && encoder != "encodeWithSelector" &&
    encoder != "encodeCall")
    return true;
  if (!payload.contains("arguments"))
    return true;
  const auto &args = payload["arguments"];
  if (!args.is_array() || args.empty())
    return true;

  args_out.clear();

  // Case 1: encodeWithSignature("sig(T,...)", user_args...)
  // The first argument must be a string literal. Remaining args are user
  // arguments passed through as-is.
  if (encoder == "encodeWithSignature")
  {
    const auto &first = args[0];
    if (
      !first.is_object() || first.value("nodeType", "") != "Literal" ||
      first.value("kind", "") != "string")
      return true;
    sig_literal = strip_spaces(first.value("value", ""));
    for (size_t i = 1; i < args.size(); ++i)
      args_out.push_back(&args[i]);
    return false;
  }

  // Case 2: encodeWithSelector(Logic.f.selector, user_args...)
  // The first argument must be a MemberAccess whose memberName is "selector"
  // and whose base resolves to a FunctionDefinition. Recover the canonical
  // signature from that definition. Remaining args are user arguments.
  if (encoder == "encodeWithSelector")
  {
    const auto &first = args[0];
    if (
      !first.is_object() || first.value("nodeType", "") != "MemberAccess" ||
      first.value("memberName", "") != "selector" ||
      !first.contains("expression"))
      return true;
    const nlohmann::json *fdecl =
      resolve_function_reference(src_ast_json, first["expression"]);
    if (fdecl == nullptr)
      return true;
    std::string canonical = build_canonical_signature(*fdecl);
    if (canonical.empty())
      return true;
    sig_literal = strip_spaces(canonical);
    for (size_t i = 1; i < args.size(); ++i)
      args_out.push_back(&args[i]);
    return false;
  }

  // Case 3: encodeCall(Logic.f, (user_args...))
  // args[0] is a function reference (MemberAccess/Identifier), args[1] is a
  // TupleExpression whose components are the user arguments.
  if (encoder == "encodeCall")
  {
    if (args.size() < 2)
      return true;
    const nlohmann::json *fdecl =
      resolve_function_reference(src_ast_json, args[0]);
    if (fdecl == nullptr)
      return true;
    std::string canonical = build_canonical_signature(*fdecl);
    if (canonical.empty())
      return true;
    sig_literal = strip_spaces(canonical);
    const auto &tup = args[1];
    if (!tup.is_object())
      return true;
    if (
      tup.value("nodeType", "") == "TupleExpression" &&
      tup.contains("components") && tup["components"].is_array())
    {
      for (const auto &c : tup["components"])
      {
        if (c.is_null())
          return true; // holes in the tuple are unsupported
        args_out.push_back(&c);
      }
      return false;
    }
    // Solidity collapses single-element parens into the inner expression,
    // so `(singleArg)` may show up as the expression itself rather than
    // a TupleExpression of length 1. Treat that as a one-element tuple.
    args_out.push_back(&args[1]);
    return false;
  }

  return true;
}

bool solidity_convertert::get_typed_call_definition(
  const std::string &caller_cname,
  const std::string &target_sig,
  const std::vector<exprt> &arg_exprs,
  symbolt *&out_sym)
{
  // Unique helper name per call site (relies on aux_counter).
  std::string helper_name = "$typed_call$" + std::to_string(aux_counter++);
  std::string helper_id = "sol:@C@" + caller_cname + "@F@" + helper_name + "#0";

  code_typet t;
  t.return_type() = bool_t;
  std::string debug_modulename = get_modulename_from_path(absolute_path);

  symbolt s;
  get_default_symbol(
    s, debug_modulename, t, helper_name, helper_id, locationt());
  auto &added_symbol = *move_symbol_to_context(s);

  // `this` param (mirrors get_call_definition setup).
  get_function_this_pointer_param(
    caller_cname, helper_id, debug_modulename, locationt(), t);

  // `address _addr` param.
  std::string addr_name = "_addr";
  std::string addr_param_id = helper_id + "@" + addr_name;
  {
    symbolt addr_s;
    get_default_symbol(
      addr_s, debug_modulename, addr_t, addr_name, addr_param_id, locationt());
    addr_s.lvalue = true;
    addr_s.is_parameter = true;
    addr_s.file_local = true;
    move_symbol_to_context(addr_s);

    code_typet::argumentt param;
    param.type() = addr_t;
    param.cmt_base_name(addr_name);
    param.cmt_identifier(addr_param_id);
    t.arguments().push_back(param);
  }

  // Per-arg params; keep ids so we can later reference them in the body.
  std::vector<std::string> arg_param_ids;
  arg_param_ids.reserve(arg_exprs.size());
  for (size_t i = 0; i < arg_exprs.size(); ++i)
  {
    std::string pname = "_arg" + std::to_string(i);
    std::string pid = helper_id + "@" + pname;
    symbolt ps;
    get_default_symbol(
      ps, debug_modulename, arg_exprs[i].type(), pname, pid, locationt());
    ps.lvalue = true;
    ps.is_parameter = true;
    ps.file_local = true;
    move_symbol_to_context(ps);

    code_typet::argumentt param;
    param.type() = arg_exprs[i].type();
    param.cmt_base_name(pname);
    param.cmt_identifier(pid);
    t.arguments().push_back(param);

    arg_param_ids.push_back(pid);
  }
  added_symbol.set_type(migrate_type(t));

  // Body construction.
  code_blockt func_body;
  {
    code_labelt label;
    label.set_label("__ESBMC_HIDE");
    label.code() = code_skipt();
    func_body.move_to_operands(label);
  }

  const symbolt &addr_sym_ref = *context.find_symbol(addr_param_id);
  exprt addr_expr = symbol_expr(addr_sym_ref);

  // For each candidate contract, if it has a function matching the target
  // signature, emit a dispatch arm.
  for (const auto &str : contractNamesList)
  {
    if (nonContractNamesList.count(str) != 0 && str != caller_cname)
      continue;
    const nlohmann::json &decl_ref =
      find_function_by_signature(str, target_sig);
    if (decl_ref.empty() || decl_ref.is_null())
      continue;

    side_effect_expr_function_callt callx;
    if (get_non_library_function_call(decl_ref, empty_json, callx))
      return true;

    // arg 0 is the implicit `this`, replace with the static contract instance.
    exprt static_ins;
    get_static_contract_instance_ref(str, static_ins);
    if (callx.arguments().empty())
      return true;
    callx.arguments().at(0) = static_ins;

    // Replace remaining formals with our helper's parameter symbols. If the
    // arity mismatches (shouldn't, since signature matched), bail out and
    // fall back to the generic path.
    if (callx.arguments().size() != arg_param_ids.size() + 1)
      return true;
    for (size_t i = 0; i < arg_param_ids.size(); ++i)
    {
      const symbolt &p = *context.find_symbol(arg_param_ids[i]);
      callx.arguments().at(i + 1) = symbol_expr(p);
    }

    code_blockt then;
    convert_expression_to_code(callx);
    then.move_to_operands(callx);

    code_returnt ret_true;
    ret_true.return_value() = true_exprt();
    then.move_to_operands(ret_true);

    // condition: _addr == ins.$address
    exprt mem_addr = member_exprt(static_ins, "$address", addr_t);
    exprt cond = exprt("=", bool_t);
    cond.operands().push_back(addr_expr);
    cond.operands().push_back(mem_addr);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(cond, then);
    func_body.move_to_operands(if_expr);
  }

  // Fallthrough: no contract matched => call fails.
  {
    code_returnt ret_false;
    ret_false.return_value() = false_exprt();
    func_body.move_to_operands(ret_false);
  }

  added_symbol.set_value(migrate_expr(func_body));
  out_sym = &added_symbol;
  return false;
}

// ======================================================================
// Delegate-shadow dispatch for
// .delegatecall(abi.encodeWithSignature(...))
// ----------------------------------------------------------------------
// Real delegatecall executes the target function's code in the CALLER's
// storage context. The generic $delegatecall#0 helper runs the target
// against its own static instance, which is wrong when the caller and
// target share storage layout (library/proxy patterns).
//
// When the payload is a literal abi.encodeWithSignature, we can clone
// the target function's body into the caller's scope and bind state var
// references to the caller's own fields by name. The cloned body runs
// inside the caller's method context, so `this` naturally resolves to
// the caller.
//
// v1 restrictions (falls back to the generic helper on any mismatch):
//   - Literal abi.encodeWithSignature payload only.
//   - Every state var referenced in the target body must have a same
//     name and same type counterpart in the caller.
//   - Target parameters are remapped to local variables declared in the
//     caller's scope right before the inlined body.
// ======================================================================

// Recursively walk `body` looking for every Identifier that points to a
// state variable in some contract other than caller_cname. Require each
// such reference to have a same-name, same-type counterpart on caller_cname.
// Returns false on compatibility, true on mismatch.
bool solidity_convertert::validate_delegate_shadow_compatible(
  const std::string &caller_cname,
  const nlohmann::json &body)
{
  // funcSignatures is a name-keyed view; for state vars we need the full
  // VariableDeclaration AST. Walk the caller contract AST once and build
  // a name -> typeString map for its state vars.
  std::unordered_map<std::string, std::string> caller_state_vars;
  for (const auto &cn : src_ast_json["nodes"])
  {
    if (!cn.is_object())
      continue;
    if (cn.value("nodeType", "") != "ContractDefinition")
      continue;
    if (cn.value("name", "") != caller_cname)
      continue;
    for (const auto &sub : cn["nodes"])
    {
      if (!sub.is_object())
        continue;
      if (sub.value("nodeType", "") != "VariableDeclaration")
        continue;
      if (!sub.value("stateVariable", false))
        continue;
      if (!sub.contains("name") || !sub.contains("typeDescriptions"))
        continue;
      caller_state_vars[sub["name"].get<std::string>()] =
        sub["typeDescriptions"].value("typeString", "");
    }
    break;
  }

  // Depth-first walk via an explicit stack to avoid recursion overhead.
  std::vector<const nlohmann::json *> worklist;
  worklist.push_back(&body);
  while (!worklist.empty())
  {
    const nlohmann::json &node = *worklist.back();
    worklist.pop_back();
    if (node.is_array())
    {
      for (const auto &el : node)
        worklist.push_back(&el);
      continue;
    }
    if (!node.is_object())
      continue;

    // Only Identifier nodes actually reference declarations. Other nodes
    // may contain child JSON we still need to traverse.
    if (
      node.value("nodeType", "") == "Identifier" &&
      node.contains("referencedDeclaration") &&
      node["referencedDeclaration"].is_number_integer() &&
      node["referencedDeclaration"].get<int>() > 0)
    {
      int ref_id = node["referencedDeclaration"].get<int>();
      // The body being validated is from a non-caller contract whose
      // own state vars are NOT visible to find_decl_ref (which scopes
      // by current_baseContractName + libs/interfaces). Searching the
      // top-level src_ast_json directly finds the decl regardless of
      // which contract owns it — required for the cross-contract
      // delegate-shadow validation to actually inspect target-side
      // state-var refs.
      auto find_decl_anywhere = [&](int id) -> const nlohmann::json * {
        for (const auto &cn : src_ast_json["nodes"])
        {
          if (!cn.is_object())
            continue;
          if (cn.value("nodeType", "") != "ContractDefinition")
            continue;
          if (!cn.contains("nodes"))
            continue;
          for (const auto &sub : cn["nodes"])
          {
            if (
              sub.is_object() && sub.contains("id") &&
              sub["id"].is_number_integer() && sub["id"].get<int>() == id)
              return &sub;
          }
        }
        return nullptr;
      };
      const nlohmann::json *decl_p = find_decl_anywhere(ref_id);
      if (
        decl_p != nullptr &&
        decl_p->value("nodeType", "") == "VariableDeclaration" &&
        decl_p->value("stateVariable", false))
      {
        const nlohmann::json &decl = *decl_p;
        std::string name = decl.value("name", "");
        std::string ty = decl.contains("typeDescriptions")
                           ? decl["typeDescriptions"].value("typeString", "")
                           : "";
        auto it = caller_state_vars.find(name);
        if (it == caller_state_vars.end())
        {
          log_debug(
            "solidity",
            "delegate shadow: caller {} has no state var named {}",
            caller_cname,
            name);
          return true;
        }
        if (it->second != ty)
        {
          log_debug(
            "solidity",
            "delegate shadow: type mismatch on {}.{}: target {} vs caller {}",
            caller_cname,
            name,
            ty,
            it->second);
          return true;
        }
      }
    }

    // Recurse into all child JSON values.
    for (auto it = node.begin(); it != node.end(); ++it)
      worklist.push_back(&it.value());
  }
  return false;
}

void solidity_convertert::rewrite_returns_for_delegate_shadow(
  exprt &node,
  const exprt &ret_lvalue,
  const std::string &end_label)
{
  // A return statement lives as a codet with statement() == "return".
  // Replace it in-place with a compound block that assigns its value (if any)
  // to the caller-side $dl_ret local and jumps to the end-of-arm label.
  if (node.id() == "code" && node.statement() == "return")
  {
    code_blockt blk;
    // If the return carries a value and we have a target lvalue of matching
    // shape, emit $dl_ret = value as a code-expression assignment.
    if (
      !node.operands().empty() && !node.op0().is_nil() &&
      ret_lvalue.is_not_nil())
    {
      exprt rv = node.op0();
      if (rv.type() != ret_lvalue.type())
        solidity_gen_typecast(ns, rv, ret_lvalue.type());
      exprt assign = side_effect_exprt("assign", ret_lvalue.type());
      assign.copy_to_operands(ret_lvalue, rv);
      convert_expression_to_code(assign);
      blk.move_to_operands(assign);
    }
    code_gotot g(end_label);
    blk.copy_to_operands(g);
    node = blk;
    return;
  }

  // Only recurse into codet children. Non-code sub-expressions (conditions,
  // rhs values, argument lists, etc.) cannot contain statements.
  if (node.id() != "code")
    return;
  for (auto &op : node.operands())
    rewrite_returns_for_delegate_shadow(op, ret_lvalue, end_label);
}

bool solidity_convertert::try_inline_delegate_shadow_helper_call(
  const nlohmann::json &call_expr,
  const nlohmann::json &fdecl,
  exprt &new_expr)
{
  // Must have a body to inline. Abstract/virtual functions fall through.
  // 0.6.x emits `body: null` for unimplemented functions; treat as missing.
  if (!fdecl.contains("body") || fdecl["body"].is_null())
    return true;
  if (
    !fdecl.contains("parameters") ||
    !fdecl["parameters"].contains("parameters"))
    return true;

  // Convert the caller-side argument expressions first, under the CURRENT
  // param_remap (which maps the outer function's formal params to the
  // outer $dl_arg_i locals).  This happens before we swap to the helper's
  // remap, so argument expressions that reference the outer parameters
  // still resolve correctly.
  const auto &arg_json = call_expr.contains("arguments")
                           ? call_expr["arguments"]
                           : nlohmann::json::array();
  std::vector<exprt> arg_exprs;
  arg_exprs.reserve(arg_json.size());
  for (const auto &aj : arg_json)
  {
    exprt ae;
    nlohmann::json literal_type;
    if (aj.contains("typeDescriptions"))
      literal_type = aj["typeDescriptions"];
    if (get_expr(aj, literal_type, ae))
      return true;
    arg_exprs.push_back(ae);
  }

  const auto &params = fdecl["parameters"]["parameters"];
  if (params.size() != arg_exprs.size())
    return true;

  // Allocate new $dl_arg_i locals for the helper's parameters and stage
  // their decls into a local wrapper block. Doing everything in a local
  // wrapper (rather than pushing individual decls to front_block) avoids
  // the nested-flush ordering problem that scrambled arg decls at the
  // top level before we switched to the wrapper pattern there.
  locationt loc;
  get_location_from_node(call_expr, loc);
  std::string debug_modulename = get_modulename_from_path(absolute_path);

  unsigned slot = aux_counter++;
  code_blockt wrapper;

  std::vector<std::string> helper_arg_ids;
  helper_arg_ids.reserve(params.size());
  for (size_t i = 0; i < params.size(); ++i)
  {
    std::string local_name =
      "$dl_harg" + std::to_string(i) + "$" + std::to_string(slot);
    std::string local_id =
      "sol:@C@" + delegate_shadow_target_cname + "@F@" + local_name + "#0";

    symbolt ls;
    get_default_symbol(
      ls, debug_modulename, arg_exprs[i].type(), local_name, local_id, loc);
    ls.lvalue = true;
    ls.file_local = true;
    auto &added_local = *move_symbol_to_context(ls);
    added_local.set_value(migrate_expr(arg_exprs[i]));

    code_declt decl(symbol_expr(added_local));
    decl.operands().push_back(arg_exprs[i]);
    wrapper.copy_to_operands(decl);

    helper_arg_ids.push_back(local_id);
  }

  // Optional $dl_ret$N$slot for single-return helpers.
  exprt ret_lvalue = nil_exprt();
  bool helper_has_ret = false;
  {
    const auto &rp_node =
      fdecl.value("returnParameters", nlohmann::json::object());
    const auto &rp = rp_node.contains("parameters") ? rp_node["parameters"]
                                                    : nlohmann::json::array();
    if (rp.is_array() && rp.size() == 1)
    {
      typet rt;
      if (get_type_description(rp[0]["typeDescriptions"], rt))
        return true;
      std::string rname = "$dl_hret$" + std::to_string(slot);
      std::string rid =
        "sol:@C@" + delegate_shadow_target_cname + "@F@" + rname + "#0";
      symbolt rs;
      get_default_symbol(rs, debug_modulename, rt, rname, rid, loc);
      rs.lvalue = true;
      rs.file_local = true;
      // expr2tc return type, so this cannot silently fall back to the legacy
      // gen_zero the decl operand below still wants.
      const expr2tc rzero =
        gen_zero(migrate_type(get_complete_type(rt, ns)), true);
      rs.set_value(rzero);
      auto &added_ret = *move_symbol_to_context(rs);
      code_declt rdecl(symbol_expr(added_ret));
      rdecl.operands().push_back(gen_zero(get_complete_type(rt, ns), true));
      wrapper.copy_to_operands(rdecl);
      ret_lvalue = symbol_expr(added_ret);
      helper_has_ret = true;
    }
  }
  std::string end_label = "$dl_hend$" + std::to_string(slot);

  // Swap param_remap / return_params to the helper's for the nested body
  // conversion, then restore afterwards.  We keep current_baseContractName
  // pointing at the target contract (the helper lives there).
  auto saved_remap = delegate_shadow_param_remap;
  delegate_shadow_param_remap.clear();
  for (size_t i = 0; i < params.size(); ++i)
    delegate_shadow_param_remap[params[i]["id"].get<int>()] = helper_arg_ids[i];

  const nlohmann::json *saved_ret_params = delegate_shadow_target_return_params;
  if (fdecl.contains("returnParameters"))
    delegate_shadow_target_return_params = &fdecl["returnParameters"];

  exprt converted_helper_body;
  bool body_err = get_block(fdecl["body"], converted_helper_body);

  delegate_shadow_target_return_params = saved_ret_params;
  delegate_shadow_param_remap = saved_remap;

  if (body_err)
    return true;

  rewrite_returns_for_delegate_shadow(
    converted_helper_body, ret_lvalue, end_label);

  wrapper.move_to_operands(converted_helper_body);

  // Emit end label as landing site for the rewritten returns.
  code_labelt lbl;
  lbl.set_label(end_label);
  lbl.code() = code_skipt();
  wrapper.move_to_operands(lbl);

  move_to_front_block(wrapper);

  // Set new_expr. For non-void helpers, the call expression evaluates to
  // $dl_hret. For void helpers, any caller context is an ExpressionStatement
  // where the result is discarded; use a skip-shaped value.
  if (helper_has_ret)
    new_expr = ret_lvalue;
  else
    new_expr = code_skipt();

  return false;
}

bool solidity_convertert::try_get_delegate_shadow_call(
  const nlohmann::json &expr,
  const nlohmann::json &func_call,
  const exprt &base,
  exprt &new_expr)
{
  if (!func_call.contains("arguments") || func_call["arguments"].empty())
    return true;

  std::string target_sig;
  std::vector<const nlohmann::json *> raw_args;
  if (
    extract_abi_encode_signature(
      func_call["arguments"][0], target_sig, raw_args))
    return true;

  std::string caller_cname;
  get_current_contract_name(expr, caller_cname);
  if (caller_cname.empty())
    return true;
  log_debug(
    "solidity",
    "try_get_delegate_shadow_call: sig={} caller={}",
    target_sig,
    caller_cname);

  // Collect candidate target contracts whose function body we can shadow.
  struct shadow_candidate
  {
    std::string cname;
    const nlohmann::json *func_decl;
  };
  std::vector<shadow_candidate> candidates;
  for (const auto &str : contractNamesList)
  {
    // Skip interface/abstract unless it's the caller itself.
    if (nonContractNamesList.count(str) != 0 && str != caller_cname)
      continue;
    const nlohmann::json &decl_ref =
      find_function_by_signature(str, target_sig);
    if (decl_ref.empty() || decl_ref.is_null())
      continue;
    if (!decl_ref.contains("body") || decl_ref["body"].is_null())
      continue;
    if (validate_delegate_shadow_compatible(caller_cname, decl_ref["body"]))
      continue;
    candidates.push_back({str, &decl_ref});
  }
  if (candidates.empty())
    return true;

  // Convert each encoded arg to an exprt before emitting any code.
  std::vector<exprt> arg_exprs;
  arg_exprs.reserve(raw_args.size());
  for (const auto *aj : raw_args)
  {
    exprt ae;
    nlohmann::json literal_type;
    if (aj->contains("typeDescriptions"))
      literal_type = (*aj)["typeDescriptions"];
    if (get_expr(*aj, literal_type, ae))
      return true;
    arg_exprs.push_back(ae);
  }

  locationt loc;
  get_location_from_node(expr, loc);
  std::string debug_modulename = get_modulename_from_path(absolute_path);

  // Build a single wrapper block that holds all decls + per-candidate arms,
  // and push ONLY the wrapper to front_block.  Pushing each decl to
  // front_block individually is unsafe: get_block() inside this function
  // recursively flushes front_block the first time it processes a nested
  // block statement (e.g. an `if` in the target body), which would scramble
  // the decl order.  The wrapper keeps them all outside of get_block's
  // reach.
  code_blockt wrapper_block;

  // Declare one local per arg. These mirror the target function's formal
  // parameters and carry the caller-supplied values into the inlined body.
  // Name them with a fresh slot so multiple delegatecalls in the same
  // function don't collide.
  unsigned slot = aux_counter++;
  std::vector<std::string> arg_local_ids;
  arg_local_ids.reserve(arg_exprs.size());
  for (size_t i = 0; i < arg_exprs.size(); ++i)
  {
    std::string local_name =
      "$dl_arg" + std::to_string(i) + "$" + std::to_string(slot);
    std::string local_id = "sol:@C@" + caller_cname + "@F@" + local_name + "#0";

    symbolt ls;
    get_default_symbol(
      ls, debug_modulename, arg_exprs[i].type(), local_name, local_id, loc);
    ls.lvalue = true;
    ls.file_local = true;
    auto &added_local = *move_symbol_to_context(ls);
    added_local.set_value(migrate_expr(arg_exprs[i]));

    code_declt decl(symbol_expr(added_local));
    decl.operands().push_back(arg_exprs[i]);
    wrapper_block.copy_to_operands(decl);

    arg_local_ids.push_back(local_id);
  }

  // Declare the bool success local. Initialized to false; each matched arm
  // sets it to true. Ends up in the (bool, bytes) tuple.
  std::string succ_name = "$dl_success$" + std::to_string(slot);
  std::string succ_id = "sol:@C@" + caller_cname + "@F@" + succ_name + "#0";
  symbolt ss;
  get_default_symbol(ss, debug_modulename, bool_t, succ_name, succ_id, loc);
  ss.lvalue = true;
  ss.file_local = true;
  auto &added_succ = *move_symbol_to_context(ss);
  added_succ.set_value(gen_false_expr());
  {
    code_declt decl(symbol_expr(added_succ));
    decl.operands().push_back(false_exprt());
    wrapper_block.copy_to_operands(decl);
  }

  // Build an if-else arm per candidate.
  for (const auto &cand : candidates)
  {
    // Populate the parameter remap: each Logic.f formal parameter's AST id
    // points at its corresponding $dl_arg_i local. get_decl_ref_expr picks
    // this up ahead of its normal AST resolution path.
    const auto &params = (*cand.func_decl)["parameters"]["parameters"];
    if (params.size() != arg_local_ids.size())
      continue; // shape mismatch, skip this arm
    delegate_shadow_param_remap.clear();
    for (size_t i = 0; i < params.size(); ++i)
    {
      int pid = params[i]["id"].get<int>();
      delegate_shadow_param_remap[pid] = arg_local_ids[i];
    }

    // Convert the target body in the caller's function context. `this`
    // resolves to the caller's this pointer (via current_functionDecl),
    // and state var references resolve to the caller's same-named fields
    // because get_var_decl_ref uses the current function's this pointer.
    //
    // However, find_decl_ref is scoped to current_baseContractName, so we
    // must temporarily switch it to the target contract so Logic's state
    // var / parameter AST ids can still be resolved during the walk.
    exprt converted_body;
    std::string saved_base = current_baseContractName;
    current_baseContractName = cand.cname;
    std::string saved_target_cname = delegate_shadow_target_cname;
    delegate_shadow_target_cname = cand.cname;
    const nlohmann::json *saved_ret_params =
      delegate_shadow_target_return_params;
    if ((*cand.func_decl).contains("returnParameters"))
      delegate_shadow_target_return_params =
        &(*cand.func_decl)["returnParameters"];
    bool body_err = get_block((*cand.func_decl)["body"], converted_body);
    delegate_shadow_target_return_params = saved_ret_params;
    delegate_shadow_target_cname = saved_target_cname;
    current_baseContractName = saved_base;
    if (body_err)
    {
      delegate_shadow_param_remap.clear();
      return true;
    }
    delegate_shadow_param_remap.clear();

    // If the target function has a single return parameter, allocate a
    // caller-side $dl_ret$N local for it and rewrite any `return X;` in the
    // converted body to `$dl_ret = X; goto $dl_end$N;`. The label is emitted
    // at the very end of the inlined body so that returns exit the arm
    // without escaping the enclosing caller function. Multi-return tuples
    // are left to the fallback path for now.
    exprt ret_lvalue = nil_exprt();
    std::string end_label = "$dl_end$" + std::to_string(slot) + "$" +
                            std::to_string(&cand - &candidates[0]);
    {
      const auto &ret_params_node =
        (*cand.func_decl).value("returnParameters", nlohmann::json::object());
      const auto &ret_params = ret_params_node.contains("parameters")
                                 ? ret_params_node["parameters"]
                                 : nlohmann::json::array();
      if (ret_params.is_array() && ret_params.size() == 1)
      {
        typet rt;
        if (get_type_description(ret_params[0]["typeDescriptions"], rt))
        {
          return true;
        }
        std::string rname = "$dl_ret$" + std::to_string(slot) + "$" +
                            std::to_string(&cand - &candidates[0]);
        std::string rid = "sol:@C@" + caller_cname + "@F@" + rname + "#0";
        symbolt rs;
        get_default_symbol(rs, debug_modulename, rt, rname, rid, loc);
        rs.lvalue = true;
        rs.file_local = true;
        // expr2tc return type; see the sibling site.
        const expr2tc rzero =
          gen_zero(migrate_type(get_complete_type(rt, ns)), true);
        rs.set_value(rzero);
        auto &added_ret = *move_symbol_to_context(rs);
        code_declt rdecl(symbol_expr(added_ret));
        rdecl.operands().push_back(gen_zero(get_complete_type(rt, ns), true));
        wrapper_block.copy_to_operands(rdecl);
        ret_lvalue = symbol_expr(added_ret);
      }
      // Even for void returns we still need to neutralise bare `return;`.
      rewrite_returns_for_delegate_shadow(
        converted_body, ret_lvalue, end_label);
    }

    // Assemble the then-arm: inlined body + end label + success = true.
    // The label lands inside the arm so rewritten returns exit the arm
    // without escaping the enclosing caller function.
    code_blockt then;
    then.move_to_operands(converted_body);
    {
      code_labelt lbl;
      lbl.set_label(end_label);
      lbl.code() = code_skipt();
      then.move_to_operands(lbl);
    }

    exprt assign_succ = side_effect_exprt("assign", bool_t);
    assign_succ.copy_to_operands(symbol_expr(added_succ), true_exprt());
    convert_expression_to_code(assign_succ);
    then.move_to_operands(assign_succ);

    // Delegatecall dispatches by function signature, not by address.
    // The candidate was already matched by name in try_get_delegate_shadow_call,
    // so emit the inlined body unconditionally. With only one matching candidate
    // per signature, there is no ambiguity.
    wrapper_block.move_to_operands(then);
  }

  // Push the whole wrapper as one unit. Doing it here (after all get_block
  // calls) means nothing inside the wrapper can be scrambled by nested
  // front_block flushes.
  move_to_front_block(wrapper_block);

  // Wrap into the (bool, bytes) tuple. Like the generic $delegatecall#0
  // path, we leave `success` as the nondet_bool initial value produced by
  // get_llc_ret_tuple(). The shadow dispatch's own `added_succ` flag is
  // still meaningful as a guard inside the wrapper block (it controls
  // whether state writes happen), but the user-visible `success` return
  // is nondet — modelling that a low-level call may fail regardless of
  // whether the target was reached.
  symbolt dump;
  get_llc_ret_tuple(dump);
  new_expr = symbol_expr(dump);
  return false;
}

bool solidity_convertert::try_get_signature_dispatched_call(
  const nlohmann::json &expr,
  const nlohmann::json &func_call,
  const exprt &base,
  exprt &new_expr)
{
  if (!func_call.contains("arguments") || func_call["arguments"].empty())
    return true;

  std::string target_sig;
  std::vector<const nlohmann::json *> raw_args;
  if (
    extract_abi_encode_signature(
      func_call["arguments"][0], target_sig, raw_args))
    return true;
  log_debug(
    "solidity",
    "try_get_signature_dispatched_call: sig={} args={}",
    target_sig,
    raw_args.size());

  // Convert each non-signature arg into an irep exprt using the arg's own
  // typeDescriptions (not the outer .call argumentTypes, which only has
  // bytes entries).
  std::vector<exprt> arg_exprs;
  arg_exprs.reserve(raw_args.size());
  for (const auto *aj : raw_args)
  {
    exprt ae;
    nlohmann::json literal_type;
    if (aj->contains("typeDescriptions"))
      literal_type = (*aj)["typeDescriptions"];
    if (get_expr(*aj, literal_type, ae))
      return true;
    arg_exprs.push_back(ae);
  }

  std::string cname;
  get_current_contract_name(expr, cname);
  if (cname.empty())
    return true;

  symbolt *helper_sym = nullptr;
  if (get_typed_call_definition(cname, target_sig, arg_exprs, helper_sym))
    return true;
  if (helper_sym == nullptr)
    return true;
  // Register as a private method of the caller contract, mirroring the
  // other generated low-level helpers.
  move_builtin_to_contract(cname, symbol_expr(*helper_sym), true);

  // Build the call expression: helper(this, addr, args...).
  locationt loc;
  get_location_from_node(expr, loc);

  side_effect_expr_function_callt call;
  call.function() = symbol_expr(*helper_sym);
  call.type() = to_code_type(helper_sym->get_type()).return_type();
  call.location() = loc;

  exprt this_object;
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, this_object))
      return true;
  }
  else
  {
    if (get_ctor_decl_this_ref(expr, this_object))
      return true;
  }
  call.arguments().push_back(this_object);

  exprt addr = base;
  set_sol_type(addr.type(), SolidityGrammar::SolType::ADDRESS);
  call.arguments().push_back(addr);

  for (const auto &ae : arg_exprs)
    call.arguments().push_back(ae);

  // Wrap into the (bool success, bytes data) tuple in the same shape as
  // get_low_level_member_accsss does for the generic call path.
  // Success stays nondet (see note in get_low_level_member_accsss).
  convert_expression_to_code(call);
  move_to_front_block(call);

  symbolt dump;
  get_llc_ret_tuple(dump);
  new_expr = symbol_expr(dump);
  return false;
}

// add `call(address _addr)` to the contract
// If it contains the function signature, it should be directly converted to the
// function calls rather than invoke this `call` e.g.
// addr.call(abi.encodeWithSignature("doSomething(uint256)", 123))
// => _ESBMC_Object_Base.doSomething(123);
//
// `is_library` = true when this $call#0 is being populated for a
// library.  In that case the `this` pointer has no meaningful content
// (libraries don't own a singleton with a `$address` field), so we
// skip the msg.sender swap and reentry-mutex updates.  The
// address-dispatch ladder itself still fires: reads of _ESBMC_Object_X
// via `get_static_contract_instance_ref(str, ...)` are preserved so
// external state updates in X remain visible to the caller.
exprt solidity_convertert::reentrant_msg_sender(
  const std::string &cname,
  const exprt &fallback_addr)
{
  // Only a contract in a multi-type structural cluster is invoked via the
  // bind_cname dispatcher that records _ESBMC_caller_inst_addr; a single-type
  // contract runs its body on the instance already, so this->$address is
  // correct. Skip the conditional there to avoid bloating call-heavy formulas
  // with an always-false branch on the global.
  if (structureTypingMap[cname].size() <= 1)
    return fallback_addr;

  const symbolt *cia = context.find_symbol("c:@_ESBMC_caller_inst_addr");
  if (cia == nullptr)
    return fallback_addr;

  exprt cia_e = symbol_expr(*cia); // addr_t (uint160)
  exprt cond("notequal", bool_t);
  cond.copy_to_operands(cia_e, from_integer(0, cia_e.type()));

  exprt chosen = cia_e;
  if (chosen.type() != fallback_addr.type())
    chosen = typecast_exprt(chosen, fallback_addr.type());

  return if_exprt(cond, chosen, fallback_addr);
}

void solidity_convertert::emit_call_revert_clear(
  code_blockt &blk,
  symbol_exprt &saved_out,
  const locationt &loc)
{
  // The revert-flag model (solidity_misc.c) is always linked into the GOTO
  // library before frontend conversion runs.
  const symbolt *flag_sym = context.find_symbol("c:@_ESBMC_sol_reverted_flag");
  assert(flag_sym && "revert-flag model must be linked");
  exprt flag_expr = symbol_expr(*flag_sym);

  // bool _saved = _ESBMC_sol_reverted_flag;  (remember caller's status)
  std::string nm, id;
  get_aux_var(nm, id);
  symbolt s;
  get_default_symbol(
    s, get_modulename_from_path(absolute_path), flag_sym->get_type(), nm, id, loc);
  s.lvalue = true;
  s.file_local = true;
  saved_out = symbol_exprt(move_symbol_to_context(s)->id, flag_sym->get_type());
  blk.copy_to_operands(code_declt(saved_out));
  blk.copy_to_operands(code_assignt(saved_out, flag_expr));

  // _ESBMC_sol_clear_revert();  (clean baseline for THIS call)
  exprt clear_stmt;
  build_revert_flag_call(
    "_ESBMC_sol_clear_revert", "c:@F@_ESBMC_sol_clear_revert", loc, clear_stmt);
  blk.copy_to_operands(clear_stmt);
}

void solidity_convertert::emit_call_revert_return(
  code_blockt &blk,
  const symbol_exprt &saved,
  const exprt *value_rollback,
  const locationt &loc)
{
  const symbolt *flag_sym = context.find_symbol("c:@_ESBMC_sol_reverted_flag");
  assert(flag_sym && "revert-flag model must be linked");
  exprt flag_expr = symbol_expr(*flag_sym);

  // bool _rev = _ESBMC_sol_reverted_flag;  (snapshot THIS call's outcome)
  std::string nm, id;
  get_aux_var(nm, id);
  symbolt s;
  get_default_symbol(
    s, get_modulename_from_path(absolute_path), flag_sym->get_type(), nm, id, loc);
  s.lvalue = true;
  s.file_local = true;
  symbol_exprt reverted(move_symbol_to_context(s)->id, flag_sym->get_type());
  blk.copy_to_operands(code_declt(reverted));
  blk.copy_to_operands(code_assignt(reverted, flag_expr));

  // _ESBMC_sol_reverted_flag = _saved;  (scoped observation: restore caller)
  blk.copy_to_operands(code_assignt(flag_expr, saved));

  // if (_rev) { <value_rollback> }  — undo the direct value transfer the EVM
  // rolls back on failure.  This is transfer-only (1 level): deep state a
  // reentrant callee subtree mutated is not unwound here, matching the
  // pre-existing revert-model granularity (`*this = save` per frame).
  if (value_rollback != nullptr)
  {
    codet rb_if("ifthenelse");
    rb_if.copy_to_operands(reverted, *value_rollback);
    blk.copy_to_operands(rb_if);
  }

  // return !_rev;  — ok is false iff the callee reverted.
  code_returnt ret;
  ret.return_value() = not_exprt(reverted);
  blk.copy_to_operands(ret);
}

bool solidity_convertert::get_call_definition(
  const std::string &cname,
  exprt &new_expr,
  bool is_library)
{
  std::string call_name = "call";
  std::string call_id = "sol:@C@" + cname + "@F@$call#0";
  symbolt s;
  // The real return type is (bool success, bytes memory data).
  // The inner function returns bool; the bytes component is added as a
  // nondet BytesDynamic by get_llc_ret_tuple() at the call site.
  code_typet t;
  t.return_type() = bool_t;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(s, debug_modulename, t, call_name, call_id, locationt());
  auto &added_symbol = *move_symbol_to_context(s);
  get_function_this_pointer_param(
    cname, call_id, debug_modulename, locationt(), t);

  // param: address _addr;
  std::string addr_name = "_addr";
  std::string addr_id = "sol:@C@" + cname + "@F@call@" + addr_name + "#" +
                        std::to_string(aux_counter++);
  symbolt addr_s;
  get_default_symbol(
    addr_s, debug_modulename, addr_t, addr_name, addr_id, locationt());
  auto addr_added_symbol = *move_symbol_to_context(addr_s);

  code_typet::argumentt param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(addr_name);
  param.cmt_identifier(addr_id);
  t.arguments().push_back(param);

  added_symbol.set_type(migrate_type(t));

  // body:
  /*
  if(_addr == _ESBMC_Object_x)
  {
    *Also check if it has public or external non-ctor function
    old_sender = msg_sender
    meg_sender = this.address
    _ESBMC_Nondet_Extcall_x();
    msg_sender = old_sender
    return true;
  }
  if(...) {...}

  return false;
  */
  code_blockt func_body;

  // add __ESBMC_HIDE label
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.move_to_operands(label);

  exprt addr_expr = symbol_expr(addr_added_symbol);
  const symbolt &msg_sender_sym = *context.find_symbol("c:@msg_sender");
  exprt msg_sender = symbol_expr(msg_sender_sym);
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  // Libraries don't own a singleton with a `$address`/`$mutex` field, so
  // the this-derived expressions below are only meaningful for contracts.
  exprt this_address;
  if (!is_library)
    this_address = member_exprt(this_expr, "$address", addr_t);

  // uint160_t old_sender =  msg_sender;  (contracts only; libraries skip
  // the msg.sender swap since the inlined library's caller's msg.sender
  // is already the active value.)
  symbolt *added_old_sender_ptr = nullptr;
  if (!is_library)
  {
    symbolt old_sender;
    get_default_symbol(
      old_sender,
      debug_modulename,
      addr_t,
      "old_sender",
      "sol:@C@" + cname + "@F@old_sender#" + std::to_string(aux_counter++),
      locationt());
    added_old_sender_ptr = move_symbol_to_context(old_sender);
    code_declt old_sender_decl(symbol_expr(*added_old_sender_ptr));
    added_old_sender_ptr->set_value(msg_sender);
    old_sender_decl.operands().push_back(msg_sender);
    func_body.move_to_operands(old_sender_decl);
  }

  for (auto str : contractNamesList)
  {
    // skip interface/abstract contract/library
    if (nonContractNamesList.count(str) != 0 && str != cname)
      continue;

    if (!has_callable_func(str))
      continue;

    code_blockt then;

    if (!is_library)
    {
      // msg_sender = this.address;
      exprt assign_sender = side_effect_exprt("assign", addr_t);
      assign_sender.copy_to_operands(
        msg_sender, reentrant_msg_sender(cname, this_address));
      convert_expression_to_code(assign_sender);
      then.move_to_operands(assign_sender);
    }

    if (is_reentry_check && !is_library)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = true;
      exprt assign_lock = side_effect_exprt("assign", bool_t);
      assign_lock.copy_to_operands(_mutex, true_exprt());
      convert_expression_to_code(assign_lock);
      then.move_to_operands(assign_lock);
    }

    // Low-level-call failure modeling: clear the revert flag, run the
    // callee, return `ok = !reverted` (below) — a reverting callee makes
    // `(bool ok, ) = addr.call(...)` observe `ok == false`.
    symbol_exprt saved_revert;
    emit_call_revert_clear(then, saved_revert, locationt());

    // _ESBMC_Nondet_Extcall_x();
    code_function_callt call;
    if (get_unbound_funccall(str, call))
      return true;
    then.move_to_operands(call);

    if (is_reentry_check && !is_library)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = false;
      exprt assign_unlock = side_effect_exprt("assign", bool_t);
      assign_unlock.copy_to_operands(_mutex, false_exprt());
      convert_expression_to_code(assign_unlock);
      then.move_to_operands(assign_unlock);
    }

    if (!is_library)
    {
      // msg_sender = old_sender;
      exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
      assign_sender_restore.copy_to_operands(
        msg_sender, symbol_expr(*added_old_sender_ptr));
      convert_expression_to_code(assign_sender_restore);
      then.move_to_operands(assign_sender_restore);
    }

    // return !reverted;  ($call#0 moves no value -> no rollback)
    emit_call_revert_return(then, saved_revert, nullptr, locationt());

    // _addr == _ESBMC_Object_str.$address
    exprt static_ins;
    get_static_contract_instance_ref(str, static_ins);
    exprt mem_addr = member_exprt(static_ins, "$address", addr_t);
    exprt _equal = exprt("=", bool_t);
    _equal.operands().push_back(addr_expr);
    _equal.operands().push_back(mem_addr);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }

  // add "Return false;" in the end
  code_returnt return_expr;
  return_expr.return_value() = false_exprt();
  func_body.move_to_operands(return_expr);

  added_symbol.set_value(migrate_expr(func_body));
  new_expr = symbol_expr(added_symbol);
  return false;
}

/** e.g. x = target.deposit{value: msg.value}()
 * @expr: member_call json
 * @this_expr: this->(target)
 * @base: target
 * @value: msg.value
 * @block: returns
 */
bool solidity_convertert::model_transaction(
  const nlohmann::json &expr,
  const exprt &this_expr,
  const exprt &base,
  const exprt &value,
  const locationt &loc,
  exprt &front_block,
  exprt &back_block)
{
  log_debug("solidity", "modelling the transaction property changes");
  /*
  old_sender = msg.sender
  old_value = msg.value
  msg.sender = instance.$address
  msg.value = instance.$balance
  instance.$balance -= value
  base.$balance += value
  (_ESBMC_mutext_Base = true)
  (call to payable func)
  (_ESBMC_mutext_Base = false)
  msg.sender = old_sender;
  msg.value = old_value
  */
  front_block = code_blockt();
  back_block = code_blockt();
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  std::string cname;
  get_current_contract_name(expr, cname);
  if (cname.empty())
    return true;

  typet val_t = unsignedbv_typet(256);
  const symbolt &msg_value_sym = *context.find_symbol("c:@msg_value");
  exprt msg_value = symbol_expr(msg_value_sym);

  if (get_high_level_call_wrapper(
        cname,
        this_expr,
        front_block,
        back_block,
        nonContractNamesList.count(cname) != 0))
    return true;

  exprt this_balance = member_exprt(this_expr, "$balance", val_t);

  symbolt old_value;
  get_default_symbol(
    old_value,
    debug_modulename,
    unsignedbv_typet(256),
    "old_value",
    "sol:@old_value#" + std::to_string(aux_counter++),
    loc);
  symbolt &added_old_value = *move_symbol_to_context(old_value);
  code_declt old_val_decl(symbol_expr(added_old_value));
  added_old_value.set_value(symbol_expr2tc(msg_value_sym));
  old_val_decl.operands().push_back(msg_value);
  front_block.move_to_operands(old_val_decl);

  // Solidity int literals default to signedbv at the source level, but
  // every balance / msg_value field is unsignedbv_256. Align the value
  // type once so the downstream comparison and arithmetic don't feed
  // mismatched-signedness operands into smt_conv's `<` / `+=` / `-=`
  // (which assert is_signedbv == is_signedbv at encoding time).
  exprt value_typed = value;
  if (value_typed.type() != val_t)
    solidity_gen_typecast(ns, value_typed, val_t);

  // msg_value = _val;
  exprt assign_val = side_effect_exprt("assign", val_t);
  assign_val.copy_to_operands(msg_value, value_typed);
  convert_expression_to_code(assign_val);
  front_block.move_to_operands(assign_val);

  // Solidity 0.8 `new C{value:V}()` and `target.method{value:V}(...)`
  // both REVERT on insufficient sender balance — they are not
  // early-returns. A `return false` here would let callers keep
  // executing past the revert and observe a partially-initialised
  // state (e.g. constructor exits before assigning state-variable
  // pointers, leaving them at their initial NONDET value, which
  // breaks subsequent `address(stateVar).balance` reads). Model the
  // revert as __ESBMC_assume(false) so the infeasible path is pruned
  // at the SMT level — matches the pattern in get_transfer_definition.
  // if(this.balance < val) __ESBMC_assume(false);
  {
    exprt less_than = exprt("<", bool_t);
    less_than.copy_to_operands(this_balance, value_typed);
    codet cmp_less_than("ifthenelse");

    side_effect_expr_function_callt assume_call;
    get_library_function_call_no_args(
      "__ESBMC_assume", "c:@F@__ESBMC_assume", empty_typet(), loc, assume_call);
    assume_call.arguments().push_back(false_exprt());
    convert_expression_to_code(assume_call);

    cmp_less_than.copy_to_operands(less_than, assume_call);
    front_block.move_to_operands(cmp_less_than);
  }

  // this.balance -= _val;
  exprt sub_assign = side_effect_exprt("assign-", val_t);
  sub_assign.copy_to_operands(this_balance, value_typed);
  convert_expression_to_code(sub_assign);
  front_block.move_to_operands(sub_assign);

  // base.balance += _val;
  exprt target_balance = member_exprt(base, "$balance", val_t);
  exprt add_assign = side_effect_exprt("assign+", val_t);
  add_assign.copy_to_operands(target_balance, value_typed);
  convert_expression_to_code(add_assign);
  front_block.move_to_operands(add_assign);

  // msg_value = old_value;
  exprt assign_val_restore = side_effect_exprt("assign", value.type());
  assign_val_restore.copy_to_operands(msg_value, symbol_expr(added_old_value));
  convert_expression_to_code(assign_val_restore);
  back_block.move_to_operands(assign_val_restore);

  convert_expression_to_code(front_block);
  convert_expression_to_code(back_block);
  return false;
}

// --reentry-balance-drain-check: at every value-transfer call site,
// emit
//   uint256 __re_drain_val = V;
//   uint256 __re_drain_pre = this->$balance;
// to the front block, and
//   __ESBMC_assert(
//     __re_drain_val > __re_drain_pre ||
//     this->$balance >= __re_drain_pre - __re_drain_val,
//     "reentrancy balance drain")
// to the back block.  Returns the symbol exprt to substitute for
// `value` in the call's argument list — guarantees single evaluation
// of an arbitrary value expression (e.g. `computeAmount()`) across
// the call argument and the post-assert.
//
// When the check is disabled or `this_expr` is nil (library context),
// `replacement_value` is set to `value` and the helper returns false
// without touching front/back blocks.
//
// Postcondition: on a successful injection,
// `outbound_drain_site_count[cname]` is incremented by 1.
bool solidity_convertert::emit_balance_drain_wrapper(
  const std::string &cname,
  const exprt &this_expr,
  const exprt &value,
  const locationt &loc,
  exprt &replacement_value)
{
  // Default no-op fallthrough: caller's call argument stays as-is.
  replacement_value = value;

  if (!is_reentry_balance_drain_check)
    return false;
  if (this_expr.is_nil())
    return false;
  if (cname.empty())
    return false;

  typet u256 = unsignedbv_typet(256);
  std::string mname = get_modulename_from_path(absolute_path);

  // 1. uint256 __re_drain_val = V;
  symbolt val_sym;
  std::string val_id =
    "sol:@C@" + cname + "@F@__re_drain_val#" + std::to_string(aux_counter++);
  get_default_symbol(val_sym, mname, u256, "__re_drain_val", val_id, loc);
  symbolt &added_val = *move_symbol_to_context(val_sym);
  added_val.set_value(value);
  code_declt val_decl(symbol_expr(added_val));
  val_decl.operands().push_back(value);
  move_to_front_block(val_decl);

  // 2. uint256 __re_drain_pre = this->$balance;
  exprt this_balance = member_exprt(this_expr, "$balance", u256);
  symbolt pre_sym;
  std::string pre_id =
    "sol:@C@" + cname + "@F@__re_drain_pre#" + std::to_string(aux_counter++);
  get_default_symbol(pre_sym, mname, u256, "__re_drain_pre", pre_id, loc);
  symbolt &added_pre = *move_symbol_to_context(pre_sym);
  added_pre.set_value(this_balance);
  code_declt pre_decl(symbol_expr(added_pre));
  pre_decl.operands().push_back(this_balance);
  move_to_front_block(pre_decl);

  // 3. __ESBMC_assert(
  //      __re_drain_val > __re_drain_pre ||
  //      this->$balance >= __re_drain_pre - __re_drain_val,
  //      "reentrancy balance drain");
  //
  // The disjunction guards against unsigned underflow when
  // `__re_drain_val > __re_drain_pre` (the transfer would have
  // failed in real EVM — call returns false / transfer reverts —
  // so balance is unchanged on that path).  Without the guard,
  // `pre - val` wraps to a giant unsigned value and the >= check
  // almost certainly fails, producing a spurious counterexample.
  exprt val_sym_expr = symbol_expr(added_val);
  exprt pre_sym_expr = symbol_expr(added_pre);

  exprt under_guard(">", bool_t);
  under_guard.copy_to_operands(val_sym_expr, pre_sym_expr);

  exprt sub("-", u256);
  sub.copy_to_operands(pre_sym_expr, val_sym_expr);
  exprt geq(">=", bool_t);
  geq.copy_to_operands(this_balance, sub);

  exprt or_expr("or", bool_t);
  or_expr.copy_to_operands(under_guard, geq);

  side_effect_expr_function_callt assert_call;
  get_library_function_call_no_args(
    "__ESBMC_assert", "c:@F@__ESBMC_assert", empty_typet(), loc, assert_call);
  assert_call.arguments().push_back(or_expr);
  string_constantt msg("reentrancy balance drain");
  assert_call.arguments().push_back(msg);
  convert_expression_to_code(assert_call);
  move_to_back_block(assert_call);

  // Replace V with the snapshot symbol in the call's argument list —
  // single-evaluation guarantee.
  replacement_value = val_sym_expr;
  outbound_drain_site_count[cname]++;
  return false;
}

// `call(address _addr, uint _val)`.  `is_library=true` adapts the body
// for library scope: libraries have no `this.$balance` / `this.$mutex`
// / real `this->$address`, so we skip the sender swap to
// `this->$address` (emit nondet msg.sender instead — matches D's
// over-approximation of the enclosing-contract address), skip the
// debit + underflow-revert side (the caller contract's $balance isn't
// reachable from inside the library's `$call#1` body), and skip the
// reentry-mutex toggles.  The CREDIT side stays: target contract's
// `$balance += _val` still fires, and the receive/fallback dispatch
// is preserved so external state updates propagate.  The boolean
// return is NONDET (real EVM can revert for gas / target-revert /
// cold-access reasons even when a tracked receive would have
// succeeded — a sound over-approximation).  For addresses that match
// no tracked contract, the library falls through to `_ESBMC_eoa_credit`
// so EOA balance reads still see the credit under `--bound`.
bool solidity_convertert::get_call_value_definition(
  const std::string &cname,
  exprt &new_expr,
  bool is_library)
{
  std::string call_name = "call";
  std::string call_id = "sol:@C@" + cname + "@F@$call#1";
  symbolt s;
  // The real return type is (bool success, bytes memory data).
  // The inner function returns bool; the bytes component is added as a
  // nondet BytesDynamic by get_llc_ret_tuple() at the call site.
  code_typet t;
  t.return_type() = bool_t;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(s, debug_modulename, t, call_name, call_id, locationt());
  auto &added_symbol = *move_symbol_to_context(s);
  get_function_this_pointer_param(
    cname, call_id, debug_modulename, locationt(), t);

  // param: address _addr;
  std::string addr_name = "_addr";
  std::string addr_id = "sol:@C@" + cname + "@F@call@" + addr_name + "#1";
  symbolt addr_s;
  get_default_symbol(
    addr_s, debug_modulename, addrp_t, addr_name, addr_id, locationt());
  auto addr_added_symbol = *move_symbol_to_context(addr_s);

  code_typet::argumentt param = code_typet::argumentt();
  param.type() = addrp_t;
  param.cmt_base_name(addr_name);
  param.cmt_identifier(addr_id);
  t.arguments().push_back(param);

  // param: uint _val;
  std::string val_name = "_val";
  std::string val_id = "sol:@C@" + cname + "@F@call@" + val_name + "#1";
  typet val_t = unsignedbv_typet(256);
  symbolt val_s;
  get_default_symbol(
    val_s, debug_modulename, val_t, val_name, val_id, locationt());
  auto val_added_symbol = *move_symbol_to_context(val_s);

  param = code_typet::argumentt();
  param.type() = val_t;
  param.cmt_base_name(val_name);
  param.cmt_identifier(val_id);
  t.arguments().push_back(param);

  added_symbol.set_type(migrate_type(t));

  // body:
  /*
  __ESBMC_Hide;
  uint256_t old_value = msg_value;
  uint160_t old_sender =  msg_sender;
  if(_addr == _ESBMC_Object_x.$address)
  {
    *! we do not consider gas consumption

    msg_value = value
    msg_sender = this.address;
    if(this.balance < x)      <-- simulate EVM rollback
      return false;
    this.balance -= x;
    _ESBMC_Object_x.balance += x;

    _ESBMC_Object_x.receive() * or fallback

    msg_value = old_value;
    msg_sender = old_sender
    return true;
  }
  if(...) {...}

  return false;
  */
  code_blockt func_body;
  exprt addr_expr = symbol_expr(addr_added_symbol);
  exprt val_expr = symbol_expr(val_added_symbol);

  // add __ESBMC_HIDE label
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  const symbolt &msg_sender_sym = *context.find_symbol("c:@msg_sender");
  exprt msg_sender = symbol_expr(msg_sender_sym);
  const symbolt &msg_value_sym = *context.find_symbol("c:@msg_value");
  exprt msg_value = symbol_expr(msg_value_sym);
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  // Library scope has no meaningful `this->$address` / `this->$balance`
  // — Lib is a dummy singleton with no real address or balance slot.
  // Only construct these member accesses for contracts.
  exprt this_address;
  exprt this_balance;
  if (!is_library)
  {
    this_address = member_exprt(this_expr, "$address", addrp_t);
    this_balance = member_exprt(this_expr, "$balance", val_t);
  }

  // uint256_t old_value = msg_value;
  symbolt old_value;
  get_default_symbol(
    old_value,
    debug_modulename,
    unsignedbv_typet(256),
    "old_value",
    "sol:@C@" + cname + "@F@call@old_value#" + std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_value = *move_symbol_to_context(old_value);
  code_declt old_val_decl(symbol_expr(added_old_value));
  added_old_value.set_value(symbol_expr2tc(msg_value_sym));
  old_val_decl.operands().push_back(msg_value);
  func_body.move_to_operands(old_val_decl);

  // uint160_t old_sender =  msg_sender;
  symbolt old_sender;
  get_default_symbol(
    old_sender,
    debug_modulename,
    addrp_t,
    "old_sender",
    "sol:@C@" + cname + "@F@call@old_sender#" + std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_sender = *move_symbol_to_context(old_sender);
  code_declt old_sender_decl(symbol_expr(added_old_sender));
  added_old_sender.set_value(symbol_expr2tc(msg_sender_sym));
  old_sender_decl.operands().push_back(msg_sender);
  func_body.move_to_operands(old_sender_decl);

  for (auto str : contractNamesList)
  {
    // skip interface/abstract contract/library
    if (nonContractNamesList.count(str) != 0 && str != cname)
      continue;
    // Here, we only consider if there is receive and fallback function
    // as the call with signature should be directly modelled.
    // order:
    // 1. match payable receive
    // 2. match payable fallback
    // 3. return false (revert)
    nlohmann::json decl_ref;
    if (has_target_function(str, "receive"))
      decl_ref = get_func_decl_ref(str, "receive");
    else if (has_target_function(str, "fallback"))
      decl_ref = get_func_decl_ref(str, "fallback");
    else
      // other payable function
      continue;
    if (decl_ref["stateMutability"] != "payable")
      continue;

    code_blockt then;

    // _addr == _ESBMC_Object_str.$address
    exprt static_ins;
    get_static_contract_instance_ref(str, static_ins);
    exprt mem_addr = member_exprt(static_ins, "$address", addrp_t);

    exprt _equal = exprt("=", bool_t);
    _equal.operands().push_back(addr_expr);
    _equal.operands().push_back(mem_addr);

    // msg_value = _val;
    exprt assign_val = side_effect_exprt("assign", val_expr.type());
    assign_val.copy_to_operands(msg_value, val_expr);
    convert_expression_to_code(assign_val);
    then.move_to_operands(assign_val);

    // msg_sender swap: for contracts use the caller's this.$address;
    // for libraries use the ambient _ESBMC_enclosing_contract_address
    // — the currently-executing contract's address, written at every
    // contract-method entry by `get_function_definition`'s wrapper.
    // This is the deterministic replacement for the earlier NONDET
    // over-approximation.
    exprt new_sender;
    if (is_library)
    {
      new_sender = symbol_expr(
        *context.find_symbol("c:@_ESBMC_enclosing_contract_address"));
    }
    else
    {
      new_sender = reentrant_msg_sender(cname, this_address);
    }
    exprt assign_sender = side_effect_exprt("assign", addrp_t);
    assign_sender.copy_to_operands(msg_sender, new_sender);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

    if (!is_library)
    {
      // if(this.balance < val) return false;
      exprt less_than = exprt("<", bool_t);
      less_than.copy_to_operands(this_balance, val_expr);
      codet cmp_less_than("ifthenelse");
      code_returnt ret_false;
      ret_false.return_value() = false_exprt();
      cmp_less_than.copy_to_operands(less_than, ret_false);
      then.move_to_operands(cmp_less_than);

      // this.balance -= _val;
      exprt sub_assign = side_effect_exprt("assign-", val_t);
      sub_assign.copy_to_operands(this_balance, val_expr);
      convert_expression_to_code(sub_assign);
      then.move_to_operands(sub_assign);
    }
    else
    {
      // Library caller: debit the ENCLOSING CONTRACT's $balance via
      // the pointer-dispatch helper.  Skip the underflow revert
      // check — the helper is best-effort: it may silently
      // under-approx if the enclosing contract isn't tracked (e.g.
      // library-from-Harness before any contract method entered).
      side_effect_expr_function_callt debit_call;
      get_library_function_call_no_args(
        "_ESBMC_enclosing_debit",
        "c:@F@_ESBMC_enclosing_debit",
        empty_typet(),
        locationt(),
        debit_call);
      debit_call.arguments().push_back(val_expr);
      convert_expression_to_code(debit_call);
      then.move_to_operands(debit_call);
    }

    // _ESBMC_Object_str.balance += _val;
    exprt target_balance = member_exprt(static_ins, "$balance", val_t);
    exprt add_assign = side_effect_exprt("assign+", val_t);
    add_assign.copy_to_operands(target_balance, val_expr);
    convert_expression_to_code(add_assign);
    then.move_to_operands(add_assign);

    if (is_reentry_check && !is_library)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = true;
      exprt assign_lock = side_effect_exprt("assign", bool_t);
      assign_lock.copy_to_operands(_mutex, true_exprt());
      convert_expression_to_code(assign_lock);
      then.move_to_operands(assign_lock);
    }

    // Low-level-call failure modeling: clear the revert flag, run the
    // callee, then return `ok = !reverted` (below) instead of a constant
    // `true`, so a reverting receive/fallback surfaces as `ok == false`.
    symbol_exprt saved_revert;
    emit_call_revert_clear(then, saved_revert, locationt());

    // func_call, e.g. receive(&_ESBMC_Object_str)
    side_effect_expr_function_callt call;
    if (get_non_library_function_call(decl_ref, empty_json, call))
      return true;
    call.arguments().at(0) = static_ins;
    convert_expression_to_code(call);
    then.move_to_operands(call);

    if (is_reentry_check && !is_library)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = false;
      exprt assign_unlock = side_effect_exprt("assign", bool_t);
      assign_unlock.copy_to_operands(_mutex, false_exprt());
      convert_expression_to_code(assign_unlock);
      then.move_to_operands(assign_unlock);
    }

    // msg_value = old_value;
    exprt assign_val_restore = side_effect_exprt("assign", val_expr.type());
    assign_val_restore.copy_to_operands(
      msg_value, symbol_expr(added_old_value));
    convert_expression_to_code(assign_val_restore);
    then.move_to_operands(assign_val_restore);

    // msg_sender = old_sender;
    exprt assign_sender_restore = side_effect_exprt("assign", addrp_t);
    assign_sender_restore.copy_to_operands(
      msg_sender, symbol_expr(added_old_sender));
    convert_expression_to_code(assign_sender_restore);
    then.move_to_operands(assign_sender_restore);

    // Tracked-target match: the target's receive/fallback was invoked.
    // Return `ok = !reverted` so a reverting callee is observable as a
    // failed call (gas exhaustion is still not modeled).  On revert, undo
    // the value transfer the EVM rolls back: the callee's `*this = save`
    // only restores the target's post-credit snapshot, so the caller's
    // debit and the target's credit both survive a naive return — leaving
    // `(ok==false ∧ balance moved)`, a state real EVM never reaches.
    code_blockt value_rollback;
    const exprt *vr = nullptr;
    if (!is_library)
    {
      // this.$balance += _val;  (undo the debit)
      exprt add_back = side_effect_exprt("assign+", val_t);
      add_back.copy_to_operands(this_balance, val_expr);
      convert_expression_to_code(add_back);
      value_rollback.move_to_operands(add_back);

      // _ESBMC_Object_str.$balance -= _val;  (undo the credit)
      exprt tgt_bal = member_exprt(static_ins, "$balance", val_t);
      exprt sub_back = side_effect_exprt("assign-", val_t);
      sub_back.copy_to_operands(tgt_bal, val_expr);
      convert_expression_to_code(sub_back);
      value_rollback.move_to_operands(sub_back);
      vr = &value_rollback;
    }
    emit_call_revert_return(then, saved_revert, vr, locationt());

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }

  if (is_library)
  {
    // Library fallthrough: address matched no tracked contract.  Credit
    // the EOA-balance map so a subsequent `recipient.balance` read under
    // `--bound` still observes the transfer (matches the EOA tracking
    // from transfer/send documented in CLAUDE.md "EOA Balance Modeling").
    side_effect_expr_function_callt eoa_credit;
    get_library_function_call_no_args(
      "_ESBMC_eoa_credit",
      "c:@F@_ESBMC_eoa_credit",
      empty_typet(),
      locationt(),
      eoa_credit);
    eoa_credit.arguments().push_back(addr_expr);
    eoa_credit.arguments().push_back(val_expr);
    convert_expression_to_code(eoa_credit);
    func_body.move_to_operands(eoa_credit);

    code_returnt return_nondet;
    side_effect_expr_function_callt nondet_call;
    get_library_function_call_no_args(
      "nondet_bool", "c:@F@nondet_bool", bool_t, locationt(), nondet_call);
    return_nondet.return_value() = nondet_call;
    func_body.move_to_operands(return_nondet);
  }
  else
  {
    // add "Return false;" in the end
    code_returnt return_expr;
    return_expr.return_value() = false_exprt();
    func_body.move_to_operands(return_expr);
  }

  added_symbol.set_value(migrate_expr(func_body));
  new_expr = symbol_expr(added_symbol);
  return false;
}

bool solidity_convertert::get_transfer_definition(
  const std::string &cname,
  exprt &new_expr,
  bool is_library)
{
  std::string call_name = "transfer";
  std::string call_id = "sol:@C@" + cname + "@F@$transfer#0";
  code_typet t;
  t.return_type() = bool_t;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  symbolt s;
  get_default_symbol(s, debug_modulename, t, call_name, call_id, locationt());
  auto &added_symbol = *move_symbol_to_context(s);
  get_function_this_pointer_param(
    cname, call_id, debug_modulename, locationt(), t);

  // param: address _addr;
  std::string addr_name = "_addr";
  std::string addr_id = "sol:@C@" + cname + "@F@transfer@" + addr_name + "#0";
  symbolt addr_s;
  get_default_symbol(
    addr_s, debug_modulename, addrp_t, addr_name, addr_id, locationt());
  auto addr_added_symbol = *move_symbol_to_context(addr_s);

  code_typet::argumentt param = code_typet::argumentt();
  param.type() = addrp_t;
  param.cmt_base_name(addr_name);
  param.cmt_identifier(addr_id);
  t.arguments().push_back(param);

  // param: uint _val;
  std::string val_name = "_val";
  std::string val_id = "sol:@C@" + cname + "@F@transfer@" + val_name + "#0";
  typet val_t = unsignedbv_typet(256);
  symbolt val_s;
  get_default_symbol(
    val_s, debug_modulename, val_t, val_name, val_id, locationt());
  auto val_added_symbol = *move_symbol_to_context(val_s);

  param = code_typet::argumentt();
  param.type() = val_t;
  param.cmt_base_name(val_name);
  param.cmt_identifier(val_id);
  t.arguments().push_back(param);

  added_symbol.set_type(migrate_type(t));

  // Library mode adapts the body in-place — see the companion comment
  // in `get_call_value_definition`.  Summary: the per-target dispatch
  // + target-balance credit + receive/fallback stay; the caller-side
  // debit (this.$balance -=), the underflow revert check, the reentry
  // mutex, and the msg_sender swap to this.$address are omitted (no
  // `this` balance slot in library scope; msg_sender swaps to NONDET
  // instead).  The fallback EOA path keeps `_ESBMC_eoa_credit` and
  // omits the sender-side debit.

  code_blockt func_body;
  exprt addr_expr = symbol_expr(addr_added_symbol);
  exprt val_expr = symbol_expr(val_added_symbol);

  // add __ESBMC_HIDE label
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  const symbolt &msg_sender_sym = *context.find_symbol("c:@msg_sender");
  exprt msg_sender = symbol_expr(msg_sender_sym);
  const symbolt &msg_value_sym = *context.find_symbol("c:@msg_value");
  exprt msg_value = symbol_expr(msg_value_sym);
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  exprt this_address;
  exprt this_balance;
  if (!is_library)
  {
    this_address = member_exprt(this_expr, "$address", addrp_t);
    this_balance = member_exprt(this_expr, "$balance", val_t);
  }

  // uint256_t old_value = msg_value;
  symbolt old_value;
  get_default_symbol(
    old_value,
    debug_modulename,
    unsignedbv_typet(256),
    "old_value",
    "sol:@C@" + cname + "@F@transfer@old_value#" +
      std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_value = *move_symbol_to_context(old_value);
  code_declt old_val_decl(symbol_expr(added_old_value));
  added_old_value.set_value(symbol_expr2tc(msg_value_sym));
  old_val_decl.operands().push_back(msg_value);
  func_body.move_to_operands(old_val_decl);

  // uint160_t old_sender =  msg_sender;
  symbolt old_sender;
  get_default_symbol(
    old_sender,
    debug_modulename,
    addrp_t,
    "old_sender",
    "sol:@C@" + cname + "@F@transfer@old_sender#" +
      std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_sender = *move_symbol_to_context(old_sender);
  code_declt old_sender_decl(symbol_expr(added_old_sender));
  added_old_sender.set_value(symbol_expr2tc(msg_sender_sym));
  old_sender_decl.operands().push_back(msg_sender);
  func_body.move_to_operands(old_sender_decl);

  for (auto str : contractNamesList)
  {
    // skip interface/abstract contract/library
    if (nonContractNamesList.count(str) != 0 && str != cname)
      continue;
    // Check if this contract has a payable receive or fallback function.
    // Balance updates always happen; the receive/fallback call is optional.
    nlohmann::json decl_ref;
    bool has_payable_callback = false;
    if (has_target_function(str, "receive"))
      decl_ref = get_func_decl_ref(str, "receive");
    else if (has_target_function(str, "fallback"))
      decl_ref = get_func_decl_ref(str, "fallback");
    if (
      !decl_ref.empty() && !decl_ref.is_null() &&
      decl_ref["stateMutability"] == "payable")
      has_payable_callback = true;

    code_blockt then;

    // _addr == _ESBMC_Object_str.$address
    exprt static_ins;
    get_static_contract_instance_ref(str, static_ins);
    exprt mem_addr = member_exprt(static_ins, "$address", addrp_t);

    exprt _equal = exprt("=", bool_t);
    _equal.operands().push_back(addr_expr);
    _equal.operands().push_back(mem_addr);

    // msg_value = _val;
    exprt assign_val = side_effect_exprt("assign", val_expr.type());
    assign_val.copy_to_operands(msg_value, val_expr);
    convert_expression_to_code(assign_val);
    then.move_to_operands(assign_val);

    // msg_sender swap: contracts use this.$address; libraries use
    // the ambient _ESBMC_enclosing_contract_address (see
    // get_call_value_definition for rationale).
    exprt new_sender;
    if (is_library)
    {
      new_sender = symbol_expr(
        *context.find_symbol("c:@_ESBMC_enclosing_contract_address"));
    }
    else
    {
      new_sender = reentrant_msg_sender(cname, this_address);
    }
    exprt assign_sender = side_effect_exprt("assign", addrp_t);
    assign_sender.copy_to_operands(msg_sender, new_sender);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

    if (!is_library)
    {
      // Real Solidity transfer() reverts on insufficient balance (it is a
      // void-returning function in the language; returning `false` here
      // would let callers keep executing past a revert and observe a
      // partially updated state). Model the revert as __ESBMC_assume(false)
      // so the infeasible path is pruned at the SMT level.
      // if(this.balance < val) __ESBMC_assume(false);
      {
        exprt less_than = exprt("<", bool_t);
        less_than.copy_to_operands(this_balance, val_expr);
        codet cmp_less_than("ifthenelse");

        side_effect_expr_function_callt assume_call;
        get_library_function_call_no_args(
          "__ESBMC_assume",
          "c:@F@__ESBMC_assume",
          empty_typet(),
          locationt(),
          assume_call);
        assume_call.arguments().push_back(false_exprt());
        convert_expression_to_code(assume_call);

        cmp_less_than.copy_to_operands(less_than, assume_call);
        then.move_to_operands(cmp_less_than);
      }

      // this.balance -= _val;
      exprt sub_assign = side_effect_exprt("assign-", val_t);
      sub_assign.copy_to_operands(this_balance, val_expr);
      convert_expression_to_code(sub_assign);
      then.move_to_operands(sub_assign);
    }
    else
    {
      // Library caller: debit the enclosing contract's $balance.
      side_effect_expr_function_callt debit_call;
      get_library_function_call_no_args(
        "_ESBMC_enclosing_debit",
        "c:@F@_ESBMC_enclosing_debit",
        empty_typet(),
        locationt(),
        debit_call);
      debit_call.arguments().push_back(val_expr);
      convert_expression_to_code(debit_call);
      then.move_to_operands(debit_call);
    }

    // _ESBMC_Object_str.balance += _val;
    exprt target_balance = member_exprt(static_ins, "$balance", val_t);
    exprt add_assign = side_effect_exprt("assign+", val_t);
    add_assign.copy_to_operands(target_balance, val_expr);
    convert_expression_to_code(add_assign);
    then.move_to_operands(add_assign);

    // Only call receive/fallback if the contract has one
    if (has_payable_callback)
    {
      if (is_reentry_check && !is_library)
      {
        exprt _mutex;
        get_contract_mutex_expr(cname, this_expr, _mutex);

        // _ESBMC_mutex = true;
        exprt assign_lock = side_effect_exprt("assign", bool_t);
        //! Do not use gen_one(bool_type()) to replace true_exprt()
        //! it will make the verification process stuck somehow
        assign_lock.copy_to_operands(_mutex, true_exprt());
        convert_expression_to_code(assign_lock);
        then.move_to_operands(assign_lock);
      }

      // func_call, e.g. receive(&_ESBMC_Object_str)
      side_effect_expr_function_callt call;
      if (get_non_library_function_call(decl_ref, empty_json, call))
        return true;
      call.arguments().at(0) = static_ins;
      convert_expression_to_code(call);
      then.move_to_operands(call);

      if (is_reentry_check && !is_library)
      {
        exprt _mutex;
        get_contract_mutex_expr(cname, this_expr, _mutex);

        // _ESBMC_mutex = false;
        exprt assign_unlock = side_effect_exprt("assign", bool_t);
        assign_unlock.copy_to_operands(_mutex, false_exprt());
        convert_expression_to_code(assign_unlock);
        then.move_to_operands(assign_unlock);
      }
    }

    // msg_value = old_value;
    exprt assign_val_restore = side_effect_exprt("assign", val_expr.type());
    assign_val_restore.copy_to_operands(
      msg_value, symbol_expr(added_old_value));
    convert_expression_to_code(assign_val_restore);
    then.move_to_operands(assign_val_restore);

    // msg_sender = old_sender;
    exprt assign_sender_restore = side_effect_exprt("assign", addrp_t);
    assign_sender_restore.copy_to_operands(
      msg_sender, symbol_expr(added_old_sender));
    convert_expression_to_code(assign_sender_restore);
    then.move_to_operands(assign_sender_restore);

    // return true;
    code_returnt ret_true;
    ret_true.return_value() = true_exprt();
    then.move_to_operands(ret_true);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }

  // EOA / unknown-recipient fallback.  Real EVM `transfer(addr, val)`
  // moves `val` out of the sender regardless of whether the recipient
  // is a tracked contract, an EOA, or an unknown address.  Deduct from
  // sender (contracts only; libraries have no owned balance), then
  // credit the recipient's slot in the global EOA balance map
  // (`_ESBMC_eoa_credit`) so a subsequent `addr.balance` read sees
  // the new value.  This is what unlocks order-sensitive properties on
  // recipient balances (e.g. SolidiFI TOD-Balance pattern A).
  //
  // Guard against uint256 underflow on `this.$balance -= _val`: real
  // EVM transfers REVERT when the sender's balance is insufficient.
  // Without this guard the deduct wraps to a near-2^256 value, letting
  // downstream `address(this).balance` reads observe an enormous balance
  // that real EVM never produces — breaks plain debit invariants like
  // `balance_after <= balance_before` and two-Vault conservation
  // (regression-locked by `transfer_standalone_partial_state_unobservable_pass`
  // and `transfer_standalone_balance_invariant_pass`, 2026-04-30).
  // Note that the multi-instance dispatch above matches only the static
  // `_ESBMC_Object_<C>` per contract type; new-allocated instances of the
  // same type fall through to this EOA branch even when the recipient is
  // technically a tracked contract, making this guard load-bearing for
  // ANY transfer involving non-static instances.
  {
    if (!is_library)
    {
      // __ESBMC_assume(this->$balance >= _val);
      side_effect_expr_function_callt assume_call;
      get_library_function_call_no_args(
        "__ESBMC_assume",
        "c:@F@__ESBMC_assume",
        empty_typet(),
        locationt(),
        assume_call);
      exprt geq = exprt(">=", bool_t);
      geq.copy_to_operands(this_balance, val_expr);
      assume_call.arguments().push_back(geq);
      convert_expression_to_code(assume_call);
      func_body.move_to_operands(assume_call);

      // this->$balance -= _val;
      exprt sub_assign = side_effect_exprt("assign-", val_t);
      sub_assign.copy_to_operands(this_balance, val_expr);
      convert_expression_to_code(sub_assign);
      func_body.move_to_operands(sub_assign);
    }
    else
    {
      // Library caller: debit enclosing contract's $balance.
      side_effect_expr_function_callt debit_call;
      get_library_function_call_no_args(
        "_ESBMC_enclosing_debit",
        "c:@F@_ESBMC_enclosing_debit",
        empty_typet(),
        locationt(),
        debit_call);
      debit_call.arguments().push_back(val_expr);
      convert_expression_to_code(debit_call);
      func_body.move_to_operands(debit_call);
    }

    // _ESBMC_eoa_credit(_addr, _val);
    side_effect_expr_function_callt eoa_credit;
    get_library_function_call_no_args(
      "_ESBMC_eoa_credit",
      "c:@F@_ESBMC_eoa_credit",
      empty_typet(),
      locationt(),
      eoa_credit);
    eoa_credit.arguments().push_back(addr_expr);
    eoa_credit.arguments().push_back(val_expr);
    convert_expression_to_code(eoa_credit);
    func_body.move_to_operands(eoa_credit);

    // return true;
    code_returnt return_expr;
    return_expr.return_value() = true_exprt();
    func_body.move_to_operands(return_expr);
  }

  added_symbol.set_value(migrate_expr(func_body));
  new_expr = symbol_expr(added_symbol);
  return false;
}

bool solidity_convertert::get_send_definition(
  const std::string &cname,
  exprt &new_expr,
  bool is_library)
{
  std::string call_name = "send";
  std::string call_id = "sol:@C@" + cname + "@F@$send#0";
  code_typet t;
  t.return_type() = bool_t;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  symbolt s;
  get_default_symbol(s, debug_modulename, t, call_name, call_id, locationt());
  auto &added_symbol = *move_symbol_to_context(s);
  get_function_this_pointer_param(
    cname, call_id, debug_modulename, locationt(), t);

  // param: address _addr;
  std::string addr_name = "_addr";
  std::string addr_id = "sol:@C@" + cname + "@F@send@" + addr_name + "#0";

  symbolt addr_s;
  get_default_symbol(
    addr_s, debug_modulename, addrp_t, addr_name, addr_id, locationt());
  auto addr_added_symbol = *move_symbol_to_context(addr_s);

  code_typet::argumentt param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(addr_name);
  param.cmt_identifier(addr_id);
  t.arguments().push_back(param);

  // param: uint _val;
  std::string val_name = "_val";
  std::string val_id = "sol:@C@" + cname + "@F@send@" + val_name + "#0";
  typet val_t = unsignedbv_typet(256);
  symbolt val_s;
  get_default_symbol(
    val_s, debug_modulename, val_t, val_name, val_id, locationt());
  auto val_added_symbol = *move_symbol_to_context(val_s);

  param = code_typet::argumentt();
  param.type() = val_t;
  param.cmt_base_name(val_name);
  param.cmt_identifier(val_id);
  t.arguments().push_back(param);

  added_symbol.set_type(migrate_type(t));

  // Library mode: see `get_transfer_definition` / `get_call_value_definition`
  // — caller-side debit + mutex + sender swap to this.$address are
  // skipped; credit side + receive/fallback dispatch kept; returns
  // nondet bool on tracked-target success (send is specced to return
  // false on failure, and library-scope reverts are opaque to the
  // caller).  EOA fallback credits the global EOA map and returns
  // nondet bool, matching send's failure-observable semantics.

  code_blockt func_body;
  exprt addr_expr = symbol_expr(addr_added_symbol);
  exprt val_expr = symbol_expr(val_added_symbol);

  // add __ESBMC_HIDE label
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  const symbolt &msg_sender_sym = *context.find_symbol("c:@msg_sender");
  exprt msg_sender = symbol_expr(msg_sender_sym);
  const symbolt &msg_value_sym = *context.find_symbol("c:@msg_value");
  exprt msg_value = symbol_expr(msg_value_sym);
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  exprt this_address;
  exprt this_balance;
  if (!is_library)
  {
    this_address = member_exprt(this_expr, "$address", addr_t);
    this_balance = member_exprt(this_expr, "$balance", val_t);
  }

  // uint256_t old_value = msg_value;
  symbolt old_value;
  get_default_symbol(
    old_value,
    debug_modulename,
    unsignedbv_typet(256),
    "old_value",
    "sol:@C@" + cname + "@F@send@old_value#" + std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_value = *move_symbol_to_context(old_value);
  code_declt old_val_decl(symbol_expr(added_old_value));
  added_old_value.set_value(symbol_expr2tc(msg_value_sym));
  old_val_decl.operands().push_back(msg_value);
  func_body.move_to_operands(old_val_decl);

  // uint160_t old_sender =  msg_sender;
  symbolt old_sender;
  get_default_symbol(
    old_sender,
    debug_modulename,
    addr_t,
    "old_sender",
    "sol:@C@" + cname + "@F@send@old_sender#" + std::to_string(aux_counter++),
    locationt());
  symbolt &added_old_sender = *move_symbol_to_context(old_sender);
  code_declt old_sender_decl(symbol_expr(added_old_sender));
  added_old_sender.set_value(symbol_expr2tc(msg_sender_sym));
  old_sender_decl.operands().push_back(msg_sender);
  func_body.move_to_operands(old_sender_decl);

  for (auto str : contractNamesList)
  {
    // skip interface/abstract contract/library
    if (nonContractNamesList.count(str) != 0 && str != cname)
      continue;
    // Check if this contract has a payable receive or fallback function.
    // Balance updates always happen; the receive/fallback call is optional.
    nlohmann::json decl_ref;
    bool has_payable_callback = false;
    if (has_target_function(str, "receive"))
      decl_ref = get_func_decl_ref(str, "receive");
    else if (has_target_function(str, "fallback"))
      decl_ref = get_func_decl_ref(str, "fallback");
    if (
      !decl_ref.empty() && !decl_ref.is_null() &&
      decl_ref["stateMutability"] == "payable")
      has_payable_callback = true;

    code_blockt then;

    // _addr == _ESBMC_Object_str.$address
    exprt static_ins;
    get_static_contract_instance_ref(str, static_ins);
    exprt mem_addr = member_exprt(static_ins, "$address", addr_t);

    exprt _equal = exprt("=", bool_t);
    _equal.operands().push_back(addr_expr);
    _equal.operands().push_back(mem_addr);

    // msg_value = _val;
    exprt assign_val = side_effect_exprt("assign", val_expr.type());
    assign_val.copy_to_operands(msg_value, val_expr);
    convert_expression_to_code(assign_val);
    then.move_to_operands(assign_val);

    // msg_sender swap: contracts use this.$address; libraries use
    // the ambient _ESBMC_enclosing_contract_address.
    exprt new_sender;
    if (is_library)
    {
      new_sender = symbol_expr(
        *context.find_symbol("c:@_ESBMC_enclosing_contract_address"));
    }
    else
    {
      new_sender = reentrant_msg_sender(cname, this_address);
    }
    exprt assign_sender = side_effect_exprt("assign", addr_t);
    assign_sender.copy_to_operands(msg_sender, new_sender);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

    if (!is_library)
    {
      // if(this.balance < val) return false;
      exprt less_than = exprt("<", val_expr.type());
      less_than.copy_to_operands(this_balance, val_expr);
      //! "ifthenelse" has to be declared as codet, not exprt and use
      //! convert_expr_to_code
      codet cmp_less_than("ifthenelse");
      code_returnt ret_false;
      ret_false.return_value() = false_exprt();
      cmp_less_than.copy_to_operands(less_than, ret_false);
      then.move_to_operands(cmp_less_than);

      // this.balance -= _val;
      exprt sub_assign = side_effect_exprt("assign-", val_t);
      sub_assign.copy_to_operands(this_balance, val_expr);
      convert_expression_to_code(sub_assign);
      then.move_to_operands(sub_assign);
    }
    else
    {
      // Library caller: debit enclosing contract's $balance.
      side_effect_expr_function_callt debit_call;
      get_library_function_call_no_args(
        "_ESBMC_enclosing_debit",
        "c:@F@_ESBMC_enclosing_debit",
        empty_typet(),
        locationt(),
        debit_call);
      debit_call.arguments().push_back(val_expr);
      convert_expression_to_code(debit_call);
      then.move_to_operands(debit_call);
    }

    // _ESBMC_Object_str.balance += _val;
    exprt target_balance = member_exprt(static_ins, "$balance", val_t);
    exprt add_assign = side_effect_exprt("assign+", val_t);
    add_assign.copy_to_operands(target_balance, val_expr);
    convert_expression_to_code(add_assign);
    then.move_to_operands(add_assign);

    // Only call receive/fallback if the contract has one.  When it does,
    // model call failure as `ok = !reverted`; without a callback there is
    // no body that can revert, so the success path stays deterministic.
    symbol_exprt saved_revert;
    bool wrapped_revert = false;
    if (has_payable_callback)
    {
      if (is_reentry_check && !is_library)
      {
        exprt _mutex;
        get_contract_mutex_expr(cname, this_expr, _mutex);

        // _ESBMC_mutex = true;
        exprt assign_lock = side_effect_exprt("assign", bool_t);
        assign_lock.copy_to_operands(_mutex, true_exprt());
        convert_expression_to_code(assign_lock);
        then.move_to_operands(assign_lock);
      }

      emit_call_revert_clear(then, saved_revert, locationt());
      wrapped_revert = true;

      // func_call, e.g. receive(&_ESBMC_Object_str)
      side_effect_expr_function_callt call;
      if (get_non_library_function_call(decl_ref, empty_json, call))
        return true;
      call.arguments().at(0) = static_ins;
      convert_expression_to_code(call);
      then.move_to_operands(call);

      if (is_reentry_check && !is_library)
      {
        exprt _mutex;
        get_contract_mutex_expr(cname, this_expr, _mutex);

        // _ESBMC_mutex = false;
        exprt assign_unlock = side_effect_exprt("assign", bool_t);
        assign_unlock.copy_to_operands(_mutex, false_exprt());
        convert_expression_to_code(assign_unlock);
        then.move_to_operands(assign_unlock);
      }
    }

    // msg_value = old_value;
    exprt assign_val_restore = side_effect_exprt("assign", val_expr.type());
    assign_val_restore.copy_to_operands(
      msg_value, symbol_expr(added_old_value));
    convert_expression_to_code(assign_val_restore);
    then.move_to_operands(assign_val_restore);

    // msg_sender = old_sender;
    exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
    assign_sender_restore.copy_to_operands(
      msg_sender, symbol_expr(added_old_sender));
    convert_expression_to_code(assign_sender_restore);
    then.move_to_operands(assign_sender_restore);

    // Tracked-target match: receive/fallback was invoked on a known
    // contract.  Return `ok = !reverted` when a callback ran (a reverting
    // receive/fallback makes `send` return false), undoing the value
    // transfer the EVM rolls back; deterministic `true` only when the
    // target has no callback that could revert.
    if (wrapped_revert)
    {
      code_blockt value_rollback;
      const exprt *vr = nullptr;
      if (!is_library)
      {
        // this.$balance += _val;  (undo the debit)
        exprt add_back = side_effect_exprt("assign+", val_t);
        add_back.copy_to_operands(this_balance, val_expr);
        convert_expression_to_code(add_back);
        value_rollback.move_to_operands(add_back);

        // _ESBMC_Object_str.$balance -= _val;  (undo the credit)
        exprt tgt_bal = member_exprt(static_ins, "$balance", val_t);
        exprt sub_back = side_effect_exprt("assign-", val_t);
        sub_back.copy_to_operands(tgt_bal, val_expr);
        convert_expression_to_code(sub_back);
        value_rollback.move_to_operands(sub_back);
        vr = &value_rollback;
      }
      emit_call_revert_return(then, saved_revert, vr, locationt());
    }
    else
    {
      code_returnt ret_outcome;
      ret_outcome.return_value() = true_exprt();
      then.move_to_operands(ret_outcome);
    }

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }

  // EOA / unknown-recipient fallback for `send`.  Same shape as
  // transfer: deduct from sender (contracts only), credit recipient
  // via the global EOA balance map.  See note in
  // get_transfer_definition.  send's bool return is `true` on EOA
  // success path for contracts; library mirrors — the non-library
  // model already ignores EVM-level revert reasons (out-of-gas,
  // cold-access), so library should too.
  {
    if (!is_library)
    {
      // this->$balance -= _val;
      exprt sub_assign = side_effect_exprt("assign-", val_t);
      sub_assign.copy_to_operands(this_balance, val_expr);
      convert_expression_to_code(sub_assign);
      func_body.move_to_operands(sub_assign);
    }
    else
    {
      // Library caller: debit enclosing contract's $balance.
      side_effect_expr_function_callt debit_call;
      get_library_function_call_no_args(
        "_ESBMC_enclosing_debit",
        "c:@F@_ESBMC_enclosing_debit",
        empty_typet(),
        locationt(),
        debit_call);
      debit_call.arguments().push_back(val_expr);
      convert_expression_to_code(debit_call);
      func_body.move_to_operands(debit_call);
    }

    // _ESBMC_eoa_credit(_addr, _val);
    side_effect_expr_function_callt eoa_credit;
    get_library_function_call_no_args(
      "_ESBMC_eoa_credit",
      "c:@F@_ESBMC_eoa_credit",
      empty_typet(),
      locationt(),
      eoa_credit);
    eoa_credit.arguments().push_back(addr_expr);
    eoa_credit.arguments().push_back(val_expr);
    convert_expression_to_code(eoa_credit);
    func_body.move_to_operands(eoa_credit);

    code_returnt ret_outcome;
    ret_outcome.return_value() = true_exprt();
    func_body.move_to_operands(ret_outcome);
  }

  added_symbol.set_value(migrate_expr(func_body));
  new_expr = symbol_expr(added_symbol);

  return false;
}

// add `staticcall(address _addr)` to the contract
// Semantically identical to call#0: dispatches to target's public functions.
// The EVM read-only enforcement is not modeled (state writes would revert
// at runtime but are not checked by ESBMC).
bool solidity_convertert::get_staticcall_definition(
  const std::string &cname,
  exprt &new_expr,
  bool is_library)
{
  std::string call_name = "staticcall";
  std::string call_id = "sol:@C@" + cname + "@F@$staticcall#0";
  symbolt s;
  code_typet t;
  t.return_type() = bool_t;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(s, debug_modulename, t, call_name, call_id, locationt());
  auto &added_symbol = *move_symbol_to_context(s);
  get_function_this_pointer_param(
    cname, call_id, debug_modulename, locationt(), t);

  // param: address _addr;
  std::string addr_name = "_addr";
  std::string addr_id = "sol:@C@" + cname + "@F@staticcall@" + addr_name + "#" +
                        std::to_string(aux_counter++);
  symbolt addr_s;
  get_default_symbol(
    addr_s, debug_modulename, addr_t, addr_name, addr_id, locationt());
  auto addr_added_symbol = *move_symbol_to_context(addr_s);

  code_typet::argumentt param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(addr_name);
  param.cmt_identifier(addr_id);
  t.arguments().push_back(param);

  added_symbol.set_type(migrate_type(t));

  // body: same as call#0
  code_blockt func_body;

  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.move_to_operands(label);

  exprt addr_expr = symbol_expr(addr_added_symbol);

  // Library mode: keep the static-call semantics (snapshot target +
  // Nondet_Extcall + restore) but drop the msg.sender swap and the
  // this-owning snapshot bookkeeping.  The dispatch still fires so
  // read effects propagate to the caller; writes observed from the
  // target are rolled back, matching static-call semantics.
  if (is_library)
  {
    for (auto str : contractNamesList)
    {
      if (nonContractNamesList.count(str) != 0 && str != cname)
        continue;
      if (!has_callable_func(str))
        continue;

      exprt static_ins;
      get_static_contract_instance_ref(str, static_ins);

      code_blockt then;

      symbolt snap_sym;
      get_default_symbol(
        snap_sym,
        debug_modulename,
        static_ins.type(),
        "sc_snap",
        "sol:@C@" + cname + "@F@staticcall@sc_snap_lib_" + str + "#" +
          std::to_string(aux_counter++),
        locationt());
      symbolt &added_snap = *move_symbol_to_context(snap_sym);
      added_snap.set_value(static_ins);
      code_declt snap_decl(symbol_expr(added_snap));
      snap_decl.operands().push_back(static_ins);
      then.move_to_operands(snap_decl);

      symbol_exprt saved_revert;
      emit_call_revert_clear(then, saved_revert, locationt());

      code_function_callt call;
      if (get_unbound_funccall(str, call))
        return true;
      then.move_to_operands(call);

      exprt assign_restore = side_effect_exprt("assign", static_ins.type());
      assign_restore.copy_to_operands(static_ins, symbol_expr(added_snap));
      convert_expression_to_code(assign_restore);
      then.move_to_operands(assign_restore);

      // return !reverted;  (explicit callee revert fails the staticcall)
      emit_call_revert_return(then, saved_revert, nullptr, locationt());

      exprt mem_addr = member_exprt(static_ins, "$address", addr_t);
      exprt _equal = exprt("=", bool_t);
      _equal.operands().push_back(addr_expr);
      _equal.operands().push_back(mem_addr);
      codet if_expr("ifthenelse");
      if_expr.copy_to_operands(_equal, then);
      func_body.move_to_operands(if_expr);
    }

    code_returnt return_expr;
    return_expr.return_value() = false_exprt();
    func_body.move_to_operands(return_expr);
    added_symbol.set_value(func_body);
    new_expr = symbol_expr(added_symbol);
    return false;
  }

  const symbolt &msg_sender_sym = *context.find_symbol("c:@msg_sender");
  exprt msg_sender = symbol_expr(msg_sender_sym);
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  exprt this_address = member_exprt(this_expr, "$address", addr_t);

  // uint160_t old_sender = msg_sender;
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
  added_old_sender.set_value(symbol_expr2tc(msg_sender_sym));
  old_sender_decl.operands().push_back(msg_sender);
  func_body.move_to_operands(old_sender_decl);

  for (auto str : contractNamesList)
  {
    if (nonContractNamesList.count(str) != 0 && str != cname)
      continue;

    if (!has_callable_func(str))
      continue;

    // Resolve target static instance once; used for snapshot, restore, and
    // the arm's address guard.
    exprt static_ins;
    get_static_contract_instance_ref(str, static_ins);

    code_blockt then;

    // Snapshot the target's full state into a local before dispatch, then
    // restore it after the nondet extcall returns. This enforces staticcall
    // read-only semantics: any writes the target performs during dispatch
    // are rolled back before control returns to the caller, matching real
    // EVM behavior (where a staticcall context causes state-modifying ops
    // to revert). Implemented as whole-struct copy (the static instance is
    // a plain struct symbol, so assignment is a memcpy-equivalent).
    symbolt snap_sym;
    get_default_symbol(
      snap_sym,
      debug_modulename,
      static_ins.type(),
      "sc_snap",
      "sol:@C@" + cname + "@F@staticcall@sc_snap_" + str + "#" +
        std::to_string(aux_counter++),
      locationt());
    symbolt &added_snap = *move_symbol_to_context(snap_sym);
    added_snap.set_value(migrate_expr(static_ins));
    code_declt snap_decl(symbol_expr(added_snap));
    snap_decl.operands().push_back(static_ins);
    then.move_to_operands(snap_decl);

    // msg_sender = this.address;
    exprt assign_sender = side_effect_exprt("assign", addr_t);
    assign_sender.copy_to_operands(
      msg_sender, reentrant_msg_sender(cname, this_address));
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

    // Note: no reentry mutex toggling here — staticcall cannot cause
    // reentrant state changes in the caller because any write attempt on
    // the callee side reverts under staticcall semantics, and the
    // snapshot/restore below guarantees the target's state observed by the
    // caller is identical before and after the dispatch. Leaving the mutex
    // alone avoids spurious reentrancy reports for view-only interactions.

    // Low-level-call failure modeling: clear, run callee, return !reverted.
    symbol_exprt saved_revert;
    emit_call_revert_clear(then, saved_revert, locationt());

    // _ESBMC_Nondet_Extcall_x();
    code_function_callt call;
    if (get_unbound_funccall(str, call))
      return true;
    then.move_to_operands(call);

    // _ESBMC_Object_str = sc_snap;  (rollback target writes)
    exprt assign_restore = side_effect_exprt("assign", static_ins.type());
    assign_restore.copy_to_operands(static_ins, symbol_expr(added_snap));
    convert_expression_to_code(assign_restore);
    then.move_to_operands(assign_restore);

    // msg_sender = old_sender;
    exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
    assign_sender_restore.copy_to_operands(
      msg_sender, symbol_expr(added_old_sender));
    convert_expression_to_code(assign_sender_restore);
    then.move_to_operands(assign_sender_restore);

    // return !reverted;  (an explicit revert in the callee fails the
    // staticcall; staticcall moves no value -> no rollback.  The implicit
    // "state-write under staticcall reverts" case stays modeled by the
    // snapshot restore above.)
    emit_call_revert_return(then, saved_revert, nullptr, locationt());

    // _addr == _ESBMC_Object_str.$address
    exprt mem_addr = member_exprt(static_ins, "$address", addr_t);
    exprt _equal = exprt("=", bool_t);
    _equal.operands().push_back(addr_expr);
    _equal.operands().push_back(mem_addr);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }

  code_returnt return_expr;
  return_expr.return_value() = false_exprt();
  func_body.move_to_operands(return_expr);

  added_symbol.set_value(migrate_expr(func_body));
  new_expr = symbol_expr(added_symbol);
  return false;
}

// add `delegatecall(address _addr)` to the contract
// delegatecall runs target code in the CALLER's storage context.
// msg.sender and msg.value are NOT changed (preserved from the original call).
// No ether transfer occurs.
// Note: true storage context switching is not modeled — the target's
// functions execute against their own storage. This correctly models
// reentrancy and control flow but not storage layout sharing.
bool solidity_convertert::get_delegatecall_definition(
  const std::string &cname,
  exprt &new_expr,
  bool is_library)
{
  std::string call_name = "delegatecall";
  std::string call_id = "sol:@C@" + cname + "@F@$delegatecall#0";
  symbolt s;
  code_typet t;
  t.return_type() = bool_t;
  std::string debug_modulename = get_modulename_from_path(absolute_path);
  get_default_symbol(s, debug_modulename, t, call_name, call_id, locationt());
  auto &added_symbol = *move_symbol_to_context(s);
  get_function_this_pointer_param(
    cname, call_id, debug_modulename, locationt(), t);

  // param: address _addr;
  std::string addr_name = "_addr";
  std::string addr_id = "sol:@C@" + cname + "@F@delegatecall@" + addr_name +
                        "#" + std::to_string(aux_counter++);
  symbolt addr_s;
  get_default_symbol(
    addr_s, debug_modulename, addr_t, addr_name, addr_id, locationt());
  auto addr_added_symbol = *move_symbol_to_context(addr_s);

  code_typet::argumentt param = code_typet::argumentt();
  param.type() = addr_t;
  param.cmt_base_name(addr_name);
  param.cmt_identifier(addr_id);
  t.arguments().push_back(param);

  added_symbol.set_type(migrate_type(t));

  // body:
  // Unlike call, delegatecall does NOT change msg.sender or msg.value.
  // It dispatches to the target contract's functions directly.
  code_blockt func_body;

  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.move_to_operands(label);

  // Delegatecall preserves msg.sender / msg.value (the target executes
  // in the caller's context).  In library scope we keep the per-target
  // dispatch — the target's external functions still fire and can
  // observe the ambient context — but skip the reentry-mutex toggles
  // (no `this` mutex to own) and return a nondet bool (library-scope
  // EVM reverts are opaque to the caller).

  exprt addr_expr = symbol_expr(addr_added_symbol);

  for (auto str : contractNamesList)
  {
    if (nonContractNamesList.count(str) != 0 && str != cname)
      continue;

    if (!has_callable_func(str))
      continue;

    code_blockt then;

    symbolt this_sym = *context.find_symbol(call_id + "#this");
    exprt this_expr = symbol_expr(this_sym);

    if (is_reentry_check && !is_library)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);
      exprt assign_lock = side_effect_exprt("assign", bool_t);
      assign_lock.copy_to_operands(_mutex, true_exprt());
      convert_expression_to_code(assign_lock);
      then.move_to_operands(assign_lock);
    }

    // Low-level-call failure modeling: clear, run callee, return !reverted.
    symbol_exprt saved_revert;
    emit_call_revert_clear(then, saved_revert, locationt());

    // _ESBMC_Nondet_Extcall_x();
    code_function_callt call;
    if (get_unbound_funccall(str, call))
      return true;
    then.move_to_operands(call);

    if (is_reentry_check && !is_library)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);
      exprt assign_unlock = side_effect_exprt("assign", bool_t);
      assign_unlock.copy_to_operands(_mutex, false_exprt());
      convert_expression_to_code(assign_unlock);
      then.move_to_operands(assign_unlock);
    }

    // Tracked-target match: the target's nondet-extcall fired.  Return
    // `ok = !reverted` so a reverting callee surfaces as a failed
    // delegatecall (gas not modeled).  delegatecall moves no value, so
    // there is no transfer to roll back.  Library-scope delegatecall to
    // an unknown address still falls through to `return false` below.
    emit_call_revert_return(then, saved_revert, nullptr, locationt());

    // _addr == _ESBMC_Object_str.$address
    exprt static_ins;
    get_static_contract_instance_ref(str, static_ins);
    exprt mem_addr = member_exprt(static_ins, "$address", addr_t);
    exprt _equal = exprt("=", bool_t);
    _equal.operands().push_back(addr_expr);
    _equal.operands().push_back(mem_addr);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }

  code_returnt return_expr;
  return_expr.return_value() = false_exprt();
  func_body.move_to_operands(return_expr);

  added_symbol.set_value(migrate_expr(func_body));
  new_expr = symbol_expr(added_symbol);
  return false;
}

std::string solidity_convertert::get_library_param_id(
  const std::string &lib_cname,
  const std::string &func_name,
  const std::string &param_name,
  int param_ast_id)
{
  return "sol:@C@" + lib_cname + "@F@" + func_name + "@" + param_name + "#" +
         std::to_string(param_ast_id);
}

// Find the name of the contract that originally defines the function with
// the given AST node id, by searching for the first non-inherited occurrence.
std::string solidity_convertert::find_contract_name_for_id(int func_id)
{
  if (!src_ast_json.contains("nodes"))
    return "";
  for (const auto &node : src_ast_json["nodes"])
  {
    if (!node.is_object())
      continue;
    if (!node.contains("nodeType") || node["nodeType"] != "ContractDefinition")
      continue;
    if (!node.contains("nodes") || !node.contains("name"))
      continue;
    for (const auto &sub : node["nodes"])
    {
      if (!sub.is_object() || !sub.contains("id"))
        continue;
      // Nodes added by merge_inheritance_ast are tagged is_inherited:true.
      // Skip those; we want only the original definition.
      if (sub.contains("is_inherited") && sub["is_inherited"].get<bool>())
        continue;
      if (sub["id"].get<int>() == func_id)
        return node["name"].get<std::string>();
    }
  }
  return "";
}

// Handle a super.method() call.
// The Solidity compiler has already resolved which base function to call via
// C3 linearization; member_access["referencedDeclaration"] is that function's
// id. We bypass the override map and call the base function directly on 'this'.
bool solidity_convertert::get_super_function_call(
  const nlohmann::json &member_access,
  const nlohmann::json &call_expr,
  exprt &new_expr)
{
  assert(member_access.contains("referencedDeclaration"));
  int func_id = member_access["referencedDeclaration"].get<int>();

  log_debug("solidity", "\t@@@ super call: resolving func_id={}", func_id);

  // Strategy: prefer the merged copy of the base function that was folded into
  // the derived contract (it carries the correct Derived* this type), unless
  // the override map redirected the lookup to a different function (meaning the
  // derived contract overrides this function).  In the override case we fall
  // back to the original definition in the base contract and insert a typecast.

  side_effect_expr_function_callt call;

  // 1. Direct lookup in the current (derived) contract scope.
  const nlohmann::json &direct = find_decl_ref(func_id);
  if (
    !direct.empty() && direct.contains("id") &&
    direct["id"].get<int>() == func_id)
  {
    // Found the exact node (merged copy inside the derived contract).
    // Its 'this' parameter already matches the derived contract type — no cast.
    if (get_non_library_function_call(direct, call_expr, call))
      return true;
  }
  else
  {
    // Either not found or override map redirected to a different function.
    // Locate the original definition in the base contract and call it with a
    // typecast on the 'this' argument.
    std::string base_cname = find_contract_name_for_id(func_id);
    if (base_cname.empty())
    {
      log_error(
        "super call: cannot find original contract for function id {}",
        func_id);
      return true;
    }
    log_debug(
      "solidity",
      "\t@@@ super call: override case, func_id={} in contract {}",
      func_id,
      base_cname);

    const nlohmann::json *decl_ptr;
    {
      ScopeGuard<std::string> guard(current_baseContractName, base_cname);
      decl_ptr = &find_decl_ref(func_id);
    }
    if (decl_ptr->empty())
    {
      log_error(
        "super call: cannot find function decl for id {} in contract {}",
        func_id,
        base_cname);
      return true;
    }

    if (get_non_library_function_call(*decl_ptr, call_expr, call))
      return true;

    // The base function's formal 'this' expects base_cname* but the current
    // function's this is Derived*.  Insert an explicit typecast so the
    // intent is clear (ESBMC would insert an implicit one regardless).
    if (!call.arguments().empty())
    {
      typet base_ptr_t = gen_pointer_type(symbol_typet(prefix + base_cname));
      exprt &this_arg = call.arguments().at(0);
      if (this_arg.type() != base_ptr_t)
        this_arg = typecast_exprt(this_arg, base_ptr_t);
    }
  }

  new_expr = call;
  return false;
}