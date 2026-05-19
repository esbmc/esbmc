/// \file solidity_convert_call.cpp
/// \brief Function call conversion for the Solidity frontend.
///
/// Converts Solidity function calls from the solc JSON AST into ESBMC's
/// side_effect_expr_function_callt representation. Handles library function
/// calls, using-for directive calls, and regular internal/external function
/// calls with argument conversion and this-pointer injection.

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
// the key difference comparing to the `get_non_library_function_call` is that we do not need a this-object as the first argument for the function call
// the key difference is that we do not add this pointer.
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
      }

      exprt single_arg;
      if (itr != itr_end && (*itr).contains("typeDescriptions"))
      {
        param = (*itr)["typeDescriptions"];
        ++itr;
      }
      else if (arg.value().contains("commonType"))
        param = arg.value()["commonType"];
      else if (arg.value().contains("typeDescriptions"))
        param = arg.value()["typeDescriptions"];

      if (get_expr(arg.value(), param, single_arg))
        return true;

      call.arguments().push_back(single_arg);
      param = nullptr;
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
    // otherwise, it's the auxiliary function we defined //e.g. call, delegatecall...

    call.arguments().push_back(this_object);
  }

  if (decl_ref.contains("parameters") && caller.contains("arguments"))
  {
    // * Assume it is a normal function call, including ctor call with params
    // set caller object as the first argument

    nlohmann::json param_nodes = decl_ref["parameters"]["parameters"];
    nlohmann::json param = nullptr;
    nlohmann::json::iterator itr = param_nodes.begin();

    for (const auto &arg : caller["arguments"].items())
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
        return true;

      call.arguments().push_back(single_arg);
      param = nullptr;
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
// we insert that contract name into the newContractSet if there is a new expresssion related to this contract
// e.g. Base x = new Base(); then we insert "Base" into newContractSet
// the idea is that if the contract is not used in 'new', then we can simply create a
// global static infinity array to play as a mapping structure
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
          if (get_type_description(
                node["typeName"]["typeDescriptions"], new_type))
          {
            log_error("failed to obtain typeDescriptions");
            abort();
          }
          if (get_sol_type(new_type) == SolidityGrammar::SolType::CONTRACT)
          {
            std::string contract_name = new_type.get("#sol_contract").c_str();
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
}

bool solidity_convertert::get_base_contract_name(
  const exprt &base,
  std::string &cname)
{
  log_debug("solidity", "\t\t@@@ get_base_contract_name");

  if (base.type().get("#sol_contract").as_string().empty())
  {
    log_error("cannot find base contract name");
    return true;
  }

  cname = base.type().get("#sol_contract").as_string();
  return false;
}

void solidity_convertert::get_nondet_expr(const typet &t, exprt &new_expr)
{
  new_expr = exprt("sideeffect", t);
  new_expr.statement("nondet");
}

// x._ESBMC_bind_cname = _ESBMC_get_nondet_cname();
//                        ^^^^^^^^^^^^^^^^^^^^^^^^
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
      typet t;
      if (get_type_description(p_node["typeDescriptions"], t))
        return true;
      if (get_sol_type(t) == SolidityGrammar::SolType::CONTRACT)
      {
        /*
            e.g. function run(Base x)
            ==>
            if(nondet_bool())
            {
              __ESBMC_Object_m.run(_ESBMC_Object_Base) 
              / / where its cname = ["Base", "Derive"]
            }
          */
        std::string base_cname = t.get("#sol_contract").as_string();
        assert(!base_cname.empty());
        exprt s;
        get_static_contract_instance_ref(base_cname, s);
        call.arguments().push_back(s);
      }
      else if (
        get_sol_type(t) == SolidityGrammar::SolType::STRING && is_pointer_check)
      {
        //! specific for string, we need to explicitly assign it as nondet_string()
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
      else
        call.arguments().push_back(static_cast<const exprt &>(get_nil_irep()));
    }
    ++cnt;
  }

  return false;
}

// check if the target contract have at least one non-ctor external or public function
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

/** 
 * Conversion: 
  constructor()
  {
    this->_ESBMC_bind_cname = get_nondet_cname(); // unless we have a new Base(), then = Base;
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
  @is_func_call: true if it's a function member access; false state variable access
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
        if (model_transaction(
              expr, cur_this_expr, base, balance, l, front_block, back_block))
        {
          log_error("failed to model the transaction property changes");
          return true;
        }
      }
      else if (get_high_level_call_wrapper(
                 cname, cur_this_expr, front_block, back_block))
        return true;

      for (auto op : front_block.operands())
        move_to_front_block(op);
      for (auto op : back_block.operands())
        move_to_back_block(op);
    }

    return false; // since it has only one possible option, no need to futher binding
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

  added_fsymbol.type = ft;
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

    // since we do not modify the string, and it always point to the known object
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
    // base->$address =  _ESBMC_Object_B.$address // note that pointer this->target == base

    bool is_revert = false;
    if (is_func_call)
    {
      // e.g. x.call() y.call(). we need to find the definition of the call beyond the contract x/y separately
      // get call
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

      if (inheritanceMap[_cname].count(str))
        memcall = member_exprt(_base, member.name(), member.type());
      else
      {
        // check if the state variable exsist in the target contract
        // signature: type + name
        // this is due to that the structureTypeMap only ensure the function signature matched
        if (is_var_getter_matched(
              str, member.name().as_string(), member.type()))
          memcall = member_exprt(_base, member.name(), member.type());
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
        if (model_transaction(
              expr, this_expr, new_base, balance, l, front_block, back_block))
        {
          log_error("failed to model the transaction property changes");
          return true;
        }
      }
      else if (get_high_level_call_wrapper(
                 cname, this_expr, front_block, back_block))
        return true;

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

  added_fsymbol.value = func_body;

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
 * e.g. x.call{value: val}("")
 *   @base: (this->)x
 *   @mem_name: call
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

      std::string func_id = "sol:@C@" + cname + "@F@$call#1";

      get_library_function_call_no_args(func_name, func_id, bool_t, loc, call);
      call.arguments().push_back(this_object);
      call.arguments().push_back(addr);
      call.arguments().push_back(value);
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

    // Emit the dispatch for its side effects, but leave the tuple's
    // `success` field as the nondet_bool initial value produced by
    // get_llc_ret_tuple(). Real EVM low-level calls can fail for many
    // reasons (out-of-gas, target revert, cold access, etc.); modeling
    // the boolean return as nondet is the soundest over-approximation
    // and matches delegatecall/staticcall below.
    convert_expression_to_code(call);
    move_to_front_block(call);

    symbolt dump;
    get_llc_ret_tuple(dump);
    new_expr = symbol_expr(dump);
  }
  else if (mem_name == "transfer")
  {
    // transfer(this, to_addr, balance_value)
    exprt addr = base;
    assert(!arg.is_nil());

    std::string func_name = "transfer";
    std::string func_id = "sol:@C@" + cname + "@F@$transfer#0";
    get_library_function_call_no_args(func_name, func_id, bool_t, loc, call);
    call.arguments().push_back(this_object);
    call.arguments().push_back(addr);
    call.arguments().push_back(arg);

    new_expr = call;
  }
  else if (mem_name == "send")
  {
    // send(this, to_addr, balance_value)
    exprt addr = base;
    assert(!arg.is_nil());

    std::string func_name = "send";
    std::string func_id = "sol:@C@" + cname + "@F@$send#0";
    get_library_function_call_no_args(func_name, func_id, bool_t, loc, call);
    call.arguments().push_back(this_object);
    call.arguments().push_back(addr);
    call.arguments().push_back(arg);

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

    // Dispatch for side effects; keep success as nondet (see call case).
    convert_expression_to_code(call);
    move_to_front_block(call);

    symbolt dump;
    get_llc_ret_tuple(dump);
    new_expr = symbol_expr(dump);
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

    // Dispatch for side effects; keep success as nondet (see call case).
    convert_expression_to_code(call);
    move_to_front_block(call);

    symbolt dump;
    get_llc_ret_tuple(dump);
    new_expr = symbol_expr(dump);
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
//   Logic.f           -> MemberAccess { memberName: "f", referencedDeclaration: <fn id> }
//   freeFunction f    -> Identifier   { referencedDeclaration: <fn id> }
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
  added_symbol.type = t;

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

  added_symbol.value = func_body;
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
      const nlohmann::json &decl = find_decl_ref(ref_id);
      if (
        !decl.empty() && !decl.is_null() &&
        decl.value("nodeType", "") == "VariableDeclaration" &&
        decl.value("stateVariable", false))
      {
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
  if (!fdecl.contains("body"))
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
    added_local.value = arg_exprs[i];

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
      rs.value = gen_zero(get_complete_type(rt, ns), true);
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
  if (extract_abi_encode_signature(
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
    if (!decl_ref.contains("body"))
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
    added_local.value = arg_exprs[i];

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
  added_succ.value = false_exprt();
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
        rs.value = gen_zero(get_complete_type(rt, ns), true);
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

    // Guard: _addr == _ESBMC_Object_cand.$address
    exprt static_ins;
    get_static_contract_instance_ref(cand.cname, static_ins);
    exprt mem_addr = member_exprt(static_ins, "$address", addr_t);
    exprt cond = exprt("=", bool_t);
    cond.operands().push_back(base);
    cond.operands().push_back(mem_addr);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(cond, then);
    wrapper_block.copy_to_operands(if_expr);
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
  if (extract_abi_encode_signature(
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
  call.type() = to_code_type(helper_sym->type).return_type();
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
// If it contains the function signature, it should be directly converted to the function calls rather than invoke this `call`
// e.g. addr.call(abi.encodeWithSignature("doSomething(uint256)", 123))
// => _ESBMC_Object_Base.doSomething(123);
bool solidity_convertert::get_call_definition(
  const std::string &cname,
  exprt &new_expr)
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

  added_symbol.type = t;

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
  exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  exprt this_address = member_exprt(this_expr, "$address", addr_t);

  // uint160_t old_sender =  msg_sender;
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
  func_body.move_to_operands(old_sender_decl);

  for (auto str : contractNamesList)
  {
    // skip interface/abstract contract/library
    if (nonContractNamesList.count(str) != 0 && str != cname)
      continue;

    if (!has_callable_func(str))
      continue;

    code_blockt then;

    // msg_sender = this.address;
    exprt assign_sender = side_effect_exprt("assign", addr_t);
    assign_sender.copy_to_operands(msg_sender, this_address);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = true;
      exprt assign_lock = side_effect_exprt("assign", bool_t);
      assign_lock.copy_to_operands(_mutex, true_exprt());
      convert_expression_to_code(assign_lock);
      then.move_to_operands(assign_lock);
    }

    // _ESBMC_Nondet_Extcall_x();
    code_function_callt call;
    if (get_unbound_funccall(str, call))
      return true;
    then.move_to_operands(call);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = false;
      exprt assign_unlock = side_effect_exprt("assign", bool_t);
      assign_unlock.copy_to_operands(_mutex, false_exprt());
      convert_expression_to_code(assign_unlock);
      then.move_to_operands(assign_unlock);
    }

    // msg_sender = old_sender;
    exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
    assign_sender_restore.copy_to_operands(
      msg_sender, symbol_expr(added_old_sender));
    convert_expression_to_code(assign_sender_restore);
    then.move_to_operands(assign_sender_restore);

    // return true;
    code_returnt ret_true;
    ret_true.return_value() = true_exprt();
    then.move_to_operands(ret_true);

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

  added_symbol.value = func_body;
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
  exprt msg_value = symbol_expr(*context.find_symbol("c:@msg_value"));

  if (get_high_level_call_wrapper(cname, this_expr, front_block, back_block))
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
  added_old_value.value = msg_value;
  old_val_decl.operands().push_back(msg_value);
  front_block.move_to_operands(old_val_decl);

  // msg_value = _val;
  exprt assign_val = side_effect_exprt("assign", value.type());
  assign_val.copy_to_operands(msg_value, value);
  convert_expression_to_code(assign_val);
  front_block.move_to_operands(assign_val);

  // if(this.balance < val) return false;
  exprt less_than = exprt("<", bool_t);
  less_than.copy_to_operands(this_balance, value);
  codet cmp_less_than("ifthenelse");
  code_returnt ret_false;
  ret_false.return_value() = false_exprt();
  cmp_less_than.copy_to_operands(less_than, ret_false);
  front_block.move_to_operands(cmp_less_than);

  // this.balance -= _val;
  exprt sub_assign = side_effect_exprt("assign-", val_t);
  sub_assign.copy_to_operands(this_balance, value);
  convert_expression_to_code(sub_assign);
  front_block.move_to_operands(sub_assign);

  // base.balance += _val;
  exprt target_balance = member_exprt(base, "$balance", val_t);
  exprt add_assign = side_effect_exprt("assign+", val_t);
  add_assign.copy_to_operands(target_balance, value);
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

// `call(address _addr, uint _val)`
bool solidity_convertert::get_call_value_definition(
  const std::string &cname,
  exprt &new_expr)
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

  added_symbol.type = t;

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

  exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));
  exprt msg_value = symbol_expr(*context.find_symbol("c:@msg_value"));
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  exprt this_address = member_exprt(this_expr, "$address", addrp_t);
  exprt this_balance = member_exprt(this_expr, "$balance", val_t);

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
  added_old_value.value = msg_value;
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
  added_old_sender.value = msg_sender;
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

    // msg_sender = this.$address;
    exprt assign_sender = side_effect_exprt("assign", addrp_t);
    assign_sender.copy_to_operands(msg_sender, this_address);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

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

    // _ESBMC_Object_str.balance += _val;
    exprt target_balance = member_exprt(static_ins, "$balance", val_t);
    exprt add_assign = side_effect_exprt("assign+", val_t);
    add_assign.copy_to_operands(target_balance, val_expr);
    convert_expression_to_code(add_assign);
    then.move_to_operands(add_assign);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);

      // _ESBMC_mutex = true;
      exprt assign_lock = side_effect_exprt("assign", bool_t);
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

    if (is_reentry_check)
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

    // return true;
    code_returnt ret_true;
    ret_true.return_value() = true_exprt();
    then.move_to_operands(ret_true);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }
  // add "Return false;" in the end
  code_returnt return_expr;
  return_expr.return_value() = false_exprt();
  func_body.move_to_operands(return_expr);

  added_symbol.value = func_body;
  new_expr = symbol_expr(added_symbol);
  return false;
}

bool solidity_convertert::get_transfer_definition(
  const std::string &cname,
  exprt &new_expr)
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

  added_symbol.type = t;

  code_blockt func_body;
  exprt addr_expr = symbol_expr(addr_added_symbol);
  exprt val_expr = symbol_expr(val_added_symbol);

  // add __ESBMC_HIDE label
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));
  exprt msg_value = symbol_expr(*context.find_symbol("c:@msg_value"));
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  exprt this_address = member_exprt(this_expr, "$address", addrp_t);
  exprt this_balance = member_exprt(this_expr, "$balance", val_t);

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
  added_old_value.value = msg_value;
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
  added_old_sender.value = msg_sender;
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

    // msg_sender = this.$address;
    exprt assign_sender = side_effect_exprt("assign", addrp_t);
    assign_sender.copy_to_operands(msg_sender, this_address);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

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

    // _ESBMC_Object_str.balance += _val;
    exprt target_balance = member_exprt(static_ins, "$balance", val_t);
    exprt add_assign = side_effect_exprt("assign+", val_t);
    add_assign.copy_to_operands(target_balance, val_expr);
    convert_expression_to_code(add_assign);
    then.move_to_operands(add_assign);

    // Only call receive/fallback if the contract has one
    if (has_payable_callback)
    {
      if (is_reentry_check)
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

      if (is_reentry_check)
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
  // add "Return false;" in the end
  code_returnt return_expr;
  return_expr.return_value() = false_exprt();
  func_body.move_to_operands(return_expr);

  added_symbol.value = func_body;
  new_expr = symbol_expr(added_symbol);
  return false;
}

bool solidity_convertert::get_send_definition(
  const std::string &cname,
  exprt &new_expr)
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

  added_symbol.type = t;

  code_blockt func_body;
  exprt addr_expr = symbol_expr(addr_added_symbol);
  exprt val_expr = symbol_expr(val_added_symbol);

  // add __ESBMC_HIDE label
  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.operands().push_back(label);

  exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));
  exprt msg_value = symbol_expr(*context.find_symbol("c:@msg_value"));
  symbolt this_sym = *context.find_symbol(call_id + "#this");
  exprt this_expr = symbol_expr(this_sym);
  exprt this_address = member_exprt(this_expr, "$address", addr_t);
  exprt this_balance = member_exprt(this_expr, "$balance", val_t);

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
  added_old_value.value = msg_value;
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
  added_old_sender.value = msg_sender;
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

    // msg_sender = this.$address;
    exprt assign_sender = side_effect_exprt("assign", addr_t);
    assign_sender.copy_to_operands(msg_sender, this_address);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

    // if(this.balance < val) return false;
    exprt less_than = exprt("<", val_expr.type());
    less_than.copy_to_operands(this_balance, val_expr);
    //! "ifthenelse" has to be declared as codet, not exprt and use convert_expr_to_code
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

    // _ESBMC_Object_str.balance += _val;
    exprt target_balance = member_exprt(static_ins, "$balance", val_t);
    exprt add_assign = side_effect_exprt("assign+", val_t);
    add_assign.copy_to_operands(target_balance, val_expr);
    convert_expression_to_code(add_assign);
    then.move_to_operands(add_assign);

    // Only call receive/fallback if the contract has one
    if (has_payable_callback)
    {
      if (is_reentry_check)
      {
        exprt _mutex;
        get_contract_mutex_expr(cname, this_expr, _mutex);

        // _ESBMC_mutex = true;
        exprt assign_lock = side_effect_exprt("assign", bool_t);
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

      if (is_reentry_check)
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

    // return true;
    code_returnt ret_true;
    ret_true.return_value() = true_exprt();
    then.move_to_operands(ret_true);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }

  // add "return false;" in the end
  code_returnt return_expr;
  return_expr.return_value() = false_exprt();
  func_body.move_to_operands(return_expr);

  added_symbol.value = func_body;
  new_expr = symbol_expr(added_symbol);

  return false;
}

// add `staticcall(address _addr)` to the contract
// Semantically identical to call#0: dispatches to target's public functions.
// The EVM read-only enforcement is not modeled (state writes would revert
// at runtime but are not checked by ESBMC).
bool solidity_convertert::get_staticcall_definition(
  const std::string &cname,
  exprt &new_expr)
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

  added_symbol.type = t;

  // body: same as call#0
  code_blockt func_body;

  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.move_to_operands(label);

  exprt addr_expr = symbol_expr(addr_added_symbol);
  exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));
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
  added_old_sender.value = msg_sender;
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
    added_snap.value = static_ins;
    code_declt snap_decl(symbol_expr(added_snap));
    snap_decl.operands().push_back(static_ins);
    then.move_to_operands(snap_decl);

    // msg_sender = this.address;
    exprt assign_sender = side_effect_exprt("assign", addr_t);
    assign_sender.copy_to_operands(msg_sender, this_address);
    convert_expression_to_code(assign_sender);
    then.move_to_operands(assign_sender);

    // Note: no reentry mutex toggling here — staticcall cannot cause
    // reentrant state changes in the caller because any write attempt on
    // the callee side reverts under staticcall semantics, and the
    // snapshot/restore below guarantees the target's state observed by the
    // caller is identical before and after the dispatch. Leaving the mutex
    // alone avoids spurious reentrancy reports for view-only interactions.

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

    // return true;
    code_returnt ret_true;
    ret_true.return_value() = true_exprt();
    then.move_to_operands(ret_true);

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

  added_symbol.value = func_body;
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
  exprt &new_expr)
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

  added_symbol.type = t;

  // body:
  // Unlike call, delegatecall does NOT change msg.sender or msg.value.
  // It dispatches to the target contract's functions directly.
  code_blockt func_body;

  code_labelt label;
  label.set_label("__ESBMC_HIDE");
  label.code() = code_skipt();
  func_body.move_to_operands(label);

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

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);
      exprt assign_lock = side_effect_exprt("assign", bool_t);
      assign_lock.copy_to_operands(_mutex, true_exprt());
      convert_expression_to_code(assign_lock);
      then.move_to_operands(assign_lock);
    }

    // _ESBMC_Nondet_Extcall_x();
    code_function_callt call;
    if (get_unbound_funccall(str, call))
      return true;
    then.move_to_operands(call);

    if (is_reentry_check)
    {
      exprt _mutex;
      get_contract_mutex_expr(cname, this_expr, _mutex);
      exprt assign_unlock = side_effect_exprt("assign", bool_t);
      assign_unlock.copy_to_operands(_mutex, false_exprt());
      convert_expression_to_code(assign_unlock);
      then.move_to_operands(assign_unlock);
    }

    // return true;
    code_returnt ret_true;
    ret_true.return_value() = true_exprt();
    then.move_to_operands(ret_true);

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

  added_symbol.value = func_body;
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
// C3 linearization; member_access["referencedDeclaration"] is that function's id.
// We bypass the override map and call the base function directly on 'this'.
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