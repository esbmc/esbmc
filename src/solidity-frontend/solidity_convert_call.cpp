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

bool solidity_convertert::get_library_function_call(
  const nlohmann::json &decl_ref,
  const nlohmann::json &caller,
  side_effect_expr_function_callt &call)
{
  assert(!decl_ref.empty());
  assert(decl_ref.contains("returnParameters"));

  exprt func;
  if (get_func_decl_ref(decl_ref, func))
    return true;

  code_typet t;
  if (get_type_description(decl_ref["returnParameters"], t.return_type()))
    return true;

  return get_library_function_call(func, t, decl_ref, caller, call);
}

// library/error/event functions have no definition node
// the key difference comparing to the `get_non_library_function_call` is that we do not need a this-object as the first argument for the function call
// the key difference is that we do not add this pointer.
bool solidity_convertert::get_library_function_call(
  const exprt &func,
  const typet &t,
  const nlohmann::json &decl_ref,
  const nlohmann::json &caller,
  side_effect_expr_function_callt &call)
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

    //  builtin functions do not need the this object as the first arguments
    for (const auto &arg : caller["arguments"].items())
    {
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

  // this object
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

  if (decl_ref.contains("parameters") && caller.contains("arguments"))
  {
    // * Assume it is a normal funciton call, including ctor call with params
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
          if (new_type.get("#sol_type") == "CONTRACT")
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
      if (t.get("#sol_type") == "CONTRACT")
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
      else if (t.get("#sol_type") == "STRING" && is_pointer_check)
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
  if (base.type().get("#sol_type") != "CONTRACT")
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

  if (member.type().get("#sol_type") == "TUPLE_RETURNS")
  {
    log_error("Unsupported return tuple");
    return true;
  }

  bool is_return_void = member.type().is_empty() ||
                        (member.type().is_code() &&
                         to_code_type(member.type()).return_type().is_empty());

  // construct auxilirary funciton
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
      // e.g. x.call() y.call(). we need to find the definiton of the call beyond the contract x/y seperately
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

/** e.g.
 * x.call{value: val}("")
 * @base: (this->)x
 * @mem_name: call
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
      addr.type().set("#sol_type", "ADDRESS_PAYABLE");
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
      addr.type().set("#sol_type", "ADDRESS");

      std::string func_id = "sol:@C@" + cname + "@F@$call#0";
      get_library_function_call_no_args(func_name, func_id, bool_t, loc, call);
      call.arguments().push_back(this_object);
      call.arguments().push_back(addr);
    }

    // convert the return value to tuple
    convert_expression_to_code(call);
    move_to_front_block(call);

    symbolt dump;
    get_llc_ret_tuple(dump);
    dump.value.op0() = call;
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
  temporary.set("#lvalue", true);
  temporary.type() = t;
  this_object = temporary;
}

// add `call(address _addr)` to the contract
// If it contains the funciton signature, it should be directly converted to the function calls rathe than invoke this `call`
// e.g. addr.call(abi.encodeWithSignature("doSomething(uint256)", 123))
// => _ESBMC_Object_Base.doSomething(123);
bool solidity_convertert::get_call_definition(
  const std::string &cname,
  exprt &new_expr)
{
  std::string call_name = "call";
  std::string call_id = "sol:@C@" + cname + "@F@$call#0";
  symbolt s;
  // the return value should be (bool, string)
  // however, we cannot handle the string, therefore we only return bool
  // and make it (x.call(), nondet_uint_expr)
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
  // the return value should be (bool, string)
  // however, we cannot handle the string, therefore we only return bool
  // and make it (x.call(), nondet_uint_expr) later
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
    exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
    assign_sender_restore.copy_to_operands(
      msg_sender, symbol_expr(added_old_sender));
    convert_expression_to_code(assign_sender_restore);
    then.move_to_operands(assign_sender_restore);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(_equal, then);
    func_body.move_to_operands(if_expr);
  }

  added_symbol.value = func_body;
  new_expr = symbol_expr(added_symbol);

  return false;
}