/// \file solidity_convert_constructor.cpp
/// \brief Constructor conversion for the Solidity frontend.
///
/// Handles parsing of explicit Solidity constructors and generation of
/// implicit default constructors for contracts that lack one. Manages
/// constructor parameter conversion, state variable initialization
/// ordering, and base contract constructor chaining for inheritance.

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

// parse the explicit ctor, or add the implicit ctor
bool solidity_convertert::get_constructor(
  const nlohmann::json &ast_node,
  const std::string &contract_name)
{
  log_debug("solidity", "Parsing Constructor {}...", contract_name);

  // check if we could find a explicit constructor
  nlohmann::json ast_nodes = ast_node["nodes"];
  for (nlohmann::json::iterator itr = ast_nodes.begin(); itr != ast_nodes.end();
       ++itr)
  {
    nlohmann::json ast_node = *itr;
    SolidityGrammar::ContractBodyElementT type =
      SolidityGrammar::get_contract_body_element_t(ast_node);
    switch (type)
    {
    case SolidityGrammar::ContractBodyElementT::FunctionDef:
    {
      if (
        ast_node.contains("kind") && !ast_node["kind"].empty() &&
        ast_node["kind"].get<std::string>() == "constructor")
        return get_function_definition(ast_node);
      continue;
    }
    default:
    {
      continue;
    }
    }
  }

  // reset
  assert(current_functionDecl == nullptr);

  // check if we need to add implicit constructor
  if (add_implicit_constructor(contract_name))
  {
    log_error("Failed to add implicit constructor");
    return true;
  }

  return false;
}

// add a empty constructor to the contract
bool solidity_convertert::add_implicit_constructor(
  const std::string &contract_name)
{
  log_debug("solidity", "\t@@@ Adding implicit constructor");
  std::string name, id;
  name = contract_name;

  // do nothing if there is already an explicit or implicit ctor
  get_ctor_call_id(contract_name, id);
  if (context.find_symbol(id) != nullptr)
    return false;

  // if we reach here, the id must be equal to get_implicit_ctor_id()
  // an implicit constructor is an void empty function
  code_typet type;
  typet tmp_rtn_type("constructor");
  type.return_type() = tmp_rtn_type;
  type.set("#member_name", prefix + contract_name);
  type.set("#inlined", true);

  locationt location_begin;

  std::string debug_modulename = get_modulename_from_path(absolute_path);

  symbolt symbol;
  get_default_symbol(symbol, debug_modulename, type, name, id, location_begin);

  symbol.lvalue = true;
  symbol.is_extern = false;
  symbol.file_local = false;

  auto &sym = *move_symbol_to_context(symbol);

  code_blockt body_exprt = code_blockt();
  sym.value = body_exprt;

  // add this pointer
  get_function_this_pointer_param(
    contract_name, id, debug_modulename, location_begin, type);

  sym.type = type;
  return false;
}

void solidity_convertert::get_temporary_object(exprt &call, exprt &new_expr)
{
  side_effect_exprt tmp_obj("temporary_object", call.type());
  codet code_expr("expression");
  code_expr.operands().push_back(call);
  tmp_obj.initializer(code_expr);
  tmp_obj.location() = call.location();
  call.swap(tmp_obj);
  new_expr = call;
}

void solidity_convertert::convert_unboundcall_nondet(
  exprt &new_expr,
  const typet common_type,
  const locationt &l)
{
  if (
    new_expr.is_code() && new_expr.statement() == "function_call" &&
    new_expr.operands().size() >= 1 &&
    new_expr.op1().name() == "_ESBMC_Nondet_Extcall")
  {
    move_to_front_block(new_expr);
    get_nondet_expr(common_type, new_expr);
    new_expr.location() = l;
  }
}

/* the member access is something like:
  class Base
  {
  public:
    static Base instance;
    void test()
    {
      instance.doSomething(); // 
    }
    void doSomething()
    {
      assert(0);
    }
  };
  Base Base::instance;

  int main()
  {
    Base::instance.test();
  }
*/
bool solidity_convertert::get_unbound_expr(
  const nlohmann::json expr,
  const std::string &c_name,
  exprt &new_expr)
{
  log_debug("solidity", "get_unbound_expr");
  if (c_name.empty())
  {
    log_error("got empty contract name");
    return true;
  }
  // it's not a member access, as it can only jump within current contract
  assert(!c_name.empty());
  code_function_callt func_call;
  if (get_unbound_funccall(c_name, func_call))
    return true;

  locationt l;
  get_location_from_node(expr, l);
  func_call.location() = l;

  // reentry check
  if (is_reentry_check)
  {
    exprt this_expr;
    assert(current_functionDecl);
    if (get_func_decl_this_ref(*current_functionDecl, this_expr))
    {
      log_error("cannot get internal this pointer reference");
      return true;
    }

    exprt _mutex;
    get_contract_mutex_expr(c_name, this_expr, _mutex);

    exprt assign_lock = side_effect_exprt("assign", bool_t);
    assign_lock.copy_to_operands(_mutex, true_exprt());
    convert_expression_to_code(assign_lock);

    exprt assign_unlock = side_effect_exprt("assign", bool_t);
    assign_unlock.copy_to_operands(_mutex, false_exprt());
    convert_expression_to_code(assign_unlock);

    // this should before the unbound_func_call
    move_to_front_block(assign_lock);
    move_to_back_block(assign_unlock);
  }

  move_to_front_block(func_call);
  new_expr = func_call;
  return false;
}

// construct the unbound verification harness
bool solidity_convertert::get_unbound_function(
  const std::string &c_name,
  symbolt &sym)
{
  std::string h_name = "_ESBMC_Nondet_Extcall_" + c_name;
  std::string h_id = "sol:@C@" + c_name + "@" + h_name + "#";
  log_debug("solidity", "\tget_unbound_function {}", h_name);

  symbolt h_sym;

  if (context.find_symbol(h_id) != nullptr)
    h_sym = *context.find_symbol(h_id);
  else
  {
    // construct unbound_function

    // 1.0 func body
    code_blockt func_body;
    func_body.make_block();

    // add __ESBMC_HIDE
    code_labelt label;
    label.set_label("__ESBMC_HIDE");
    label.code() = code_skipt();
    func_body.operands().push_back(label);

    // 1.1 get static contract instance
    exprt contract_var;
    get_static_contract_instance_ref(c_name, contract_var);

    // construct return; to avoid fall-through
    exprt return_expr = code_returnt();

    // 2.0 check visibility setting
    bool skip_vis =
      config.options.get_option("no-visibility").empty() ? false : true;
    if (skip_vis)
    {
      log_warning(
        "force to verify every function, even it's an unreachable "
        "internal/private function. This might lead to false positives.");
    }

    // 2.1 construct if-then-else statement
    const auto methods = funcSignatures[c_name];

    // --focus-function: when the caller is the target contract and a focus
    // function is set, restrict the dispatch loop to only that function.
    // Other contracts (e.g., cross-contract targets reached from inside the
    // focus function) keep their full nondet dispatch.
    const bool focus_applies = !focus_func.empty() && tgt_cnt_set.size() == 1 &&
                               c_name == *tgt_cnt_set.begin();

    for (const auto &method : methods)
    {
      // we only handle public (and external) function
      // as the private and internal function cannot be directly called
      if (
        !skip_vis && method.visibility != "public" &&
        method.visibility != "external")
        // skip internal and private
        continue;
      const std::string func_name = method.name;
      if (func_name == c_name)
        // skip constructor
        continue;
      // Dispatch fallback() and receive() as ordinary harness branches.
      // Skipping them used to hide assertion violations inside their bodies
      // (e.g. `assert(msg.sender == address(0))` would be unreachable because
      // the body was never invoked). The generic dispatch below is sound: the
      // harness entry already seeds msg_sender/msg_value to nondet values,
      // so the body is exercised under arbitrary caller state, which is the
      // correct over-approximation for both low-level entry points.
      if (focus_applies && func_name != focus_func)
        // focus-function mode: skip all non-focus functions on the target
        // contract to avoid unnecessary verification overhead.
        continue;

      // then: function_call
      // do member access
      exprt mem_access = member_exprt(contract_var, method.id, method.type);

      // find function definition json node
      nlohmann::json decl_ref = get_func_decl_ref(c_name, func_name);
      if (decl_ref.empty())
      {
        log_error(
          "Internal error: fail to find the definition of function {}",
          method.name);
        abort();
      }

      side_effect_expr_function_callt then_expr;
      if (get_non_library_function_call(decl_ref, empty_json, then_expr))
        return true;

      // set &_ESBMC_tmp as the first argument
      then_expr.arguments().at(0) = contract_var;
      convert_expression_to_code(then_expr);

      code_blockt then;
      then.copy_to_operands(then_expr, return_expr);

      // ifthenelse-statement:
      codet if_expr("ifthenelse");
      if_expr.copy_to_operands(nondet_bool_expr, then);

      func_body.copy_to_operands(if_expr);
    }

    // 3. construct harness
    symbolt new_symbol;
    code_typet h_type;
    typet e_type = empty_typet();
    e_type.set("cpp_type", "void");
    h_type.return_type() = e_type;
    std::string debug_modulename = get_modulename_from_path(absolute_path);
    get_default_symbol(
      new_symbol, debug_modulename, h_type, h_name, h_id, locationt());

    new_symbol.lvalue = true;
    new_symbol.is_extern = false;
    new_symbol.file_local = false;

    symbolt &added_sym = *context.move_symbol_to_context(new_symbol);

    // no params
    h_type.make_ellipsis();

    added_sym.type = h_type;
    added_sym.value = func_body;
    h_sym = added_sym;
  }

  sym = h_sym;
  return false;
}

// Normally, we would expect expr to be a code_declt expression
void solidity_convertert::move_to_initializer(const exprt &expr)
{
  // the initializer will clear its elements, so we populate the copy instead of origins
  if (!ctor_frontBlockDecl.operands().empty())
  {
    // reverse order
    for (auto &op : ctor_frontBlockDecl.operands())
    {
      convert_expression_to_code(op);
      initializers.copy_to_operands(op);
    }
    ctor_frontBlockDecl.clear();
  }

  initializers.copy_to_operands(expr);

  if (!ctor_backBlockDecl.operands().empty())
  {
    // reverse order
    for (auto &op : ctor_backBlockDecl.operands())
    {
      convert_expression_to_code(op);
      initializers.copy_to_operands(op);
    }
    ctor_backBlockDecl.clear();
  }
}

bool solidity_convertert::move_initializer_to_ctor(
  const nlohmann::json *based_contracts,
  const nlohmann::json &current_contract,
  const std::string contract_name)
{
  return move_initializer_to_ctor(
    based_contracts, current_contract, contract_name, false);
}

// move library initializer and bind-name initializer to main
bool solidity_convertert::move_initializer_to_main(codet &func_body)
{
  side_effect_expr_function_callt call;
  get_library_function_call_no_args(
    "initialize", "c:@F@initialize", empty_typet(), locationt(), call);
  convert_expression_to_code(call);
  func_body.move_to_operands(call);

  for (auto str : contractNamesList)
  {
    std::string fname, fid;
    get_bind_cname_func_name(str, fname, fid);
    if (context.find_symbol(fid) == nullptr)
      return true;
    exprt func = symbol_expr(*context.find_symbol(fid));
    side_effect_expr_function_callt _call;
    _call.function() = func;
    convert_expression_to_code(_call);
    func_body.move_to_operands(_call);
  }
  return false;
}

// convert the initialization of the state variable
// into the equivalent assignmment in the ctor
// for inheritance_ctor, we skip the builtin assignment
bool solidity_convertert::move_initializer_to_ctor(
  const nlohmann::json *based_contracts,
  const nlohmann::json &current_contract,
  const std::string contract_name,
  bool is_aux_ctor)
{
  log_debug(
    "solidity",
    "@@@ Moving initialization of the state variable to the constructor {}().",
    contract_name);

  std::string ctor_id;
  if (is_aux_ctor)
  {
    exprt dump;
    get_inherit_ctor_definition(contract_name, dump);
    ctor_id = dump.identifier().as_string();
  }
  else
  {
    if (get_ctor_call_id(contract_name, ctor_id))
    {
      log_error("cannot find the construcor");
      return true;
    }
  }

  if (context.find_symbol(ctor_id) == nullptr)
  {
    log_error("cannot find the ctor ref of {}", ctor_id);
    return true;
  }
  symbolt &sym = *context.find_symbol(ctor_id);

  // get this pointer
  exprt base;
  if (get_func_decl_this_ref(contract_name, ctor_id, base))
  {
    log_error("cannot find function's this pointer");
    return true;
  }

  // queue insert initialization of the state
  for (auto it = initializers.operands().rbegin();
       it != initializers.operands().rend();
       ++it)
  {
    if (
      it->type().is_code() &&
      to_code(*it).get_statement().as_string() == "decl")
    {
      exprt comp = to_code_decl(to_code(*it)).op0();
      log_debug(
        "solidity",
        "\t@@@ initializing symbol {} in the constructor",
        comp.name().as_string());

      bool is_state = comp.type().get("#sol_state_var") == "1";
      if (!is_state)
      {
        // auxiliary local variable we created
        exprt tmp = *it;
        sym.value.operands().insert(sym.value.operands().begin(), tmp);
        continue;
      }
      if (is_aux_ctor)
      {
        if (
          comp.name().empty() ||
          is_sol_builin_symbol(contract_name, comp.name().as_string()))
          continue;
      }

      exprt lhs = member_exprt(base, comp.name(), comp.type());
      if (context.find_symbol(comp.identifier()) == nullptr)
      {
        log_error("Interal Error: cannot find symbol");
        abort();
      }
      symbolt *symbol = context.find_symbol(comp.identifier());
      exprt rhs = symbol->value;
      exprt _assign;
      if (
        get_sol_type(lhs.type()) == SolidityGrammar::SolType::STRING &&
        rhs.get("#zero_initializer") != "1" && rhs.id() != "string-constant")
      {
        // p = NULL;
        // _str_assign(&p, "hello");
        // since it's in the intializer, there should be no memory leak
        get_string_assignment(lhs, rhs, _assign);
        convert_expression_to_code(_assign);
      }
      else
      {
        _assign = side_effect_exprt("assign", comp.type());
        _assign.location() = sym.location;
        assert(current_contract != nullptr);
        if (rhs.get("#zero_initializer") != "1")
          convert_type_expr(ns, rhs, comp, current_contract);
        _assign.copy_to_operands(lhs, rhs);
      }
      convert_expression_to_code(_assign);

      // insert before the sym.value.operands
      sym.value.operands().insert(sym.value.operands().begin(), _assign);

      // we might need to insert some expression
      // due to the convert_type_expr and get_string_assignment
      if (ctor_frontBlockDecl.operands().size() != 0)
      {
        for (auto &op : ctor_frontBlockDecl.operands())
        {
          convert_expression_to_code(op);
          sym.value.operands().insert(sym.value.operands().begin(), op);
        }
        ctor_frontBlockDecl.clear();
      }
    }
    else
    {
      exprt tmp = *it;
      convert_expression_to_code(tmp);
      sym.value.operands().insert(sym.value.operands().begin(), tmp);
    }
  }

  // insert parent ctor call in the front
  if (move_inheritance_to_ctor(based_contracts, contract_name, ctor_id, sym))
    return true;

  if (is_aux_ctor)
  {
    // hide it
    code_labelt label;
    label.set_label("__ESBMC_HIDE");
    label.code() = code_skipt();
    sym.value.operands().insert(sym.value.operands().begin(), label);
  }

  // _sol_init_
  side_effect_expr_function_callt init_call;
  get_library_function_call_no_args(
    "_sol_init_", "sol:@F@_sol_init_", empty_typet(), locationt(), init_call);
  convert_expression_to_code(init_call);
  sym.value.operands().insert(sym.value.operands().begin(), init_call);

  return false;
}

void solidity_convertert::move_to_front_block(const exprt &expr)
{
  if (current_functionDecl)
    expr_frontBlockDecl.copy_to_operands(expr);
  else
    ctor_frontBlockDecl.copy_to_operands(expr);
}

void solidity_convertert::move_to_back_block(const exprt &expr)
{
  if (current_functionDecl)
    expr_backBlockDecl.copy_to_operands(expr);
  else
    ctor_backBlockDecl.copy_to_operands(expr);
}

bool solidity_convertert::move_inheritance_to_ctor(
  const nlohmann::json *based_contracts,
  const std::string contract_name,
  std::string ctor_id,
  symbolt &sym)
{
  log_debug(
    "solidity",
    "@@@ Moving parents' constructor calls to the current constructor");

  std::string this_id = ctor_id + "#this";
  if (context.find_symbol(this_id) == nullptr)
  {
    log_error("Failed to find ctor this pointer {}", this_id);
    return true;
  }
  exprt this_expr = symbol_expr(*context.find_symbol(this_id));

  // As we are handling the constructor
  const std::string old_fname = current_functionName;
  current_functionName = contract_name;

  // queue insert the ctor initializaiton based on the linearizedBaseList
  if (based_contracts != nullptr && context.find_symbol(this_id) != nullptr)
  {
    /*
      Constructors are executed in the following order:
      1 - Base2
      2 - Base1
      3 - Derived3
      contract Derived3 is Base2, Base1 {
          constructor() Base1() Base2() {}
        }

      E.g. 
        contract DD is BB(3)
      Result ctor symbol table:
        Symbol......: c:@S@DD@F@DD#
        Module......: 1
        Base name...: DD
        Mode........: C++
        Type........: constructor  (struct DD *)
        Value.......: 
        {
          BB((struct BB *)this, 3);
        }
      However, since the c++ frontend is broken(esbmc/issues/1866),
      we convert it as 
        function ctor()
        {
          // create temporary object
          Base2 _ESBMC_ctor_Base2_tmp = new Base();
          // copy value
          this.x =  _ESBMC_ctor_Base2_tmp.x ;
          ...
        }
    */

    const std::vector<int> &id_list = linearizedBaseList[contract_name];
    for (auto it = id_list.begin() + 1; it != id_list.end(); ++it)
    {
      // handling inheritance
      // skip the first one as it is the contract itself
      std::string target_c_name = contractNamesMap[*it];

      for (const auto &c_node : (*based_contracts))
      {
        assert(c_node.contains("baseName"));
        std::string c_name = c_node["baseName"]["name"].get<std::string>();
        if (c_name != target_c_name)
          continue;

        typet c_type(irept::id_symbol);
        c_type.identifier(prefix + c_name);

        // get value
        // search for the parameter list for the constructor
        // they could be in two places:
        // - contract DD is BB(3)
        // or
        // - constructor() BB(3)
        nlohmann::json c_args_list_node = empty_json;
        const nlohmann::json &ctor_node = find_constructor_ref(contract_name);

        if (c_node.contains("arguments"))
          c_args_list_node = c_node;
        else if (!ctor_node.empty())
        {
          auto _ctor = ctor_node["modifiers"];
          for (const auto &c_mdf : _ctor)
          {
            if (
              !c_mdf.contains("modifierName") ||
              c_mdf["kind"] != "baseConstructorSpecifier")
              continue;

            if (c_mdf["modifierName"]["name"].get<std::string>() == c_name)
            {
              c_args_list_node = c_mdf;
              break;
            }
          }
        }

        // BB _ESBMC_aux_BB = BB(&this, 3, true);
        symbolt added_ctor_symbol;
        get_inherit_static_contract_instance(
          contract_name, c_name, c_args_list_node, added_ctor_symbol);

        // copy value e.g.  this.data = X.data
        struct_typet type_complete =
          to_struct_type(context.find_symbol(prefix + contract_name)->type);
        struct_typet c_type_complete =
          to_struct_type(context.find_symbol(prefix + c_name)->type);

        exprt lhs;
        exprt rhs;
        exprt _assign;
        for (const auto &c_comp : c_type_complete.components())
        {
          for (const auto &comp : type_complete.components())
          {
            if (c_comp.name() == comp.name())
            {
              assert(!comp.name().empty());
              assert(!c_comp.name().empty());

              if (is_sol_builin_symbol(c_name, c_comp.name().as_string()))
                // skip builtin symbol.
                //e.g. this->$address = _ESBMC_ctor_A_tmp.$address;
                continue;

              lhs = member_exprt(this_expr, comp.name(), comp.type());
              rhs = member_exprt(
                symbol_expr(added_ctor_symbol), c_comp.name(), c_comp.type());
              if (get_sol_type(comp.type()) == SolidityGrammar::SolType::STRING)
                // it have been initialized so should have no dereference failure
                get_string_assignment(lhs, rhs, _assign);
              else
              {
                _assign = side_effect_exprt("assign", comp.type());
                //? convert_type_expr(ns, rhs, comp.type(), current_contract);
                _assign.copy_to_operands(lhs, rhs);
              }

              convert_expression_to_code(_assign);
              sym.value.operands().insert(
                sym.value.operands().begin(), _assign);
              break;
            }
          }
        }

        // insert ctor call
        code_declt dl(symbol_expr(added_ctor_symbol));
        dl.operands().push_back(added_ctor_symbol.value);
        sym.value.operands().insert(sym.value.operands().begin(), dl);
      }
    }
  }

  current_functionName = old_fname;
  return false;
}

bool solidity_convertert::get_ctor_decl_this_ref(
  const std::string &c_name,
  exprt &this_object)
{
  std::string ctor_id;
  if (get_ctor_call_id(c_name, ctor_id))
  {
    log_error("failed to get the ctor id");
    return true;
  }

  if (get_func_decl_this_ref(c_name, ctor_id, this_object))
  {
    log_error("failed to get this ref of function {}", ctor_id);
    return true;
  }
  return false;
}

bool solidity_convertert::get_ctor_decl_this_ref(
  const nlohmann::json &caller,
  exprt &this_object)
{
  log_debug("solidity", "get_ctor_decl_this_ref");

  // ctor
  std::string current_cname;
  get_current_contract_name(caller, current_cname);
  if (current_cname.empty())
  {
    log_error("Failed to get caller's contract name");
    return true;
  }

  return get_ctor_decl_this_ref(current_cname, this_object);
}

// get the constructor symbol id
// noted that the ctor might not have been parsed yet
bool solidity_convertert::get_ctor_call_id(
  const std::string &contract_name,
  std::string &ctor_id)
{
  // we first try to find the explicit constructor defined in the source file.
  ctor_id = get_explicit_ctor_call_id(contract_name);
  if (ctor_id.empty())
    // then we try to find the implicit constructor we manually added
    ctor_id = get_implict_ctor_call_id(contract_name);

  if (ctor_id.empty())
  {
    // this means the neither explicit nor implicit constructor is found
    return true;
  }
  return false;
}

// get the explicit constructor symbol id
// retrun empty string if no explicit ctor
std::string
solidity_convertert::get_explicit_ctor_call_id(const std::string &contract_name)
{
  // get the constructor
  const nlohmann::json &ctor_ref = find_constructor_ref(contract_name);
  if (!ctor_ref.empty())
  {
    int id = ctor_ref["id"].get<int>();
    return "sol:@C@" + contract_name + "@F@" + contract_name + "#" +
           std::to_string(id);
  }

  // not found
  return "";
}

// get the implicit constructor symbol id
std::string
solidity_convertert::get_implict_ctor_call_id(const std::string &contract_name)
{
  // for implicit ctor, the id is manually set as 0
  return "sol:@C@" + contract_name + "@F@" + contract_name + "#";
}

// return construcor node based on the *contract* id
const nlohmann::json &solidity_convertert::find_constructor_ref(int contract_id)
{
  nlohmann::json &nodes = src_ast_json["nodes"];
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if (
      (*itr)["id"].get<int>() == contract_id &&
      (*itr)["nodeType"] == "ContractDefinition")
    {
      nlohmann::json &ast_nodes = (*itr)["nodes"];
      for (nlohmann::json::iterator ittr = ast_nodes.begin();
           ittr != ast_nodes.end();
           ++ittr)
      {
        if ((*ittr).contains("kind") && (*ittr)["kind"] == "constructor")
          return *ittr;
      }
    }
  }

  // implicit constructor call
  return empty_json;
}

const nlohmann::json &
solidity_convertert::find_constructor_ref(const std::string &contract_name)
{
  log_debug(
    "solidity", "\t@@@ finding reference of constructor {}", contract_name);
  nlohmann::json &nodes = src_ast_json["nodes"];
  for (nlohmann::json::iterator itr = nodes.begin(); itr != nodes.end(); ++itr)
  {
    if (
      (*itr).contains("name") &&
      (*itr)["name"].get<std::string>() == contract_name &&
      (*itr)["nodeType"] == "ContractDefinition")
    {
      nlohmann::json &ast_nodes = (*itr)["nodes"];
      for (nlohmann::json::iterator ittr = ast_nodes.begin();
           ittr != ast_nodes.end();
           ++ittr)
      {
        if ((*ittr).contains("kind") && (*ittr)["kind"] == "constructor")
          return *ittr;
      }
    }
  }

  log_debug("solidity", "\t@@@ Failed to find explicit constructor");
  // implicit constructor call
  return empty_json;
}

/**
 * @param decl_ref: the declaration of the ctor. Can be empty for implicit ctor.
 * @caller: the caller node that might contain the arguments
*/
bool solidity_convertert::get_ctor_call(
  const nlohmann::json &decl_ref,
  const nlohmann::json &caller,
  side_effect_expr_function_callt &call)
{
  /*
  we need to convert
    call(1)
  to
    call(&Base, 1)
  */
  log_debug("solidity", "\t\t@@@ get_ctor_call");
  locationt l;
  get_location_from_node(caller, l);

  if (!decl_ref.empty())
  {
    if (get_non_library_function_call(decl_ref, caller, call))
      return true;

    // reset the type. due to the empty returnParameters, the type of the call
    // is wrongly set as void.
    const auto &_contract =
      find_parent_contract(src_ast_json["nodes"], decl_ref);
    std::string c_name = _contract["name"].get<std::string>();
    call.type() = symbol_typet(prefix + c_name);

    // reset the this object
    exprt this_object;
    get_new_object(symbol_typet(prefix + c_name), this_object);
    this_object = address_of_exprt(this_object);
    call.arguments().at(0) = this_object;
    // set constructor
    call.set("constructor", 1);
  }
  else
  {
    log_error("unexpected implicit ctor");
    return true;
  }

  return false;
}

bool solidity_convertert::get_new_object_ctor_call(
  const nlohmann::json &caller,
  const bool is_object,
  exprt &new_expr)
{
  log_debug("solidity", "generating new contract object");
  // 1. get the ctor call expr
  nlohmann::json callee_expr_json;
  // if the caller's nextnode is a NewExpression, we can use it's expression directly
  // else, we need to use the expression's expression
  if (caller["expression"]["nodeType"] == "NewExpression")
    callee_expr_json = caller["expression"];
  else
    callee_expr_json = caller["expression"]["expression"];
  int ref_decl_id = callee_expr_json["typeName"]["referencedDeclaration"];
  // get contract name
  const std::string contract_name = contractNamesMap[ref_decl_id];
  if (contract_name.empty())
  {
    log_error("cannot find the contract name");
    abort();
  }
  if (get_new_object_ctor_call(contract_name, caller, is_object, new_expr))
    return true;

  return false;
}

// return a new expression: new Base(2);
bool solidity_convertert::get_new_object_ctor_call(
  const std::string &contract_name,
  const nlohmann::json caller,
  const bool is_object,
  exprt &new_expr)
{
  log_debug("solidity", "get_new_object_ctor_call");
  assert(linearizedBaseList.count(contract_name) && !contract_name.empty());

  // setup initializer, i.e. call the constructor
  side_effect_expr_function_callt call;
  const nlohmann::json &constructor_ref = find_constructor_ref(contract_name);
  if (constructor_ref.empty())
    return get_implicit_ctor_ref(contract_name, is_object, new_expr);

  if (get_ctor_call(constructor_ref, caller, call))
    return true;

  // construct temporary object
  if (is_object)
  {
    // Base x = &sideefect(..)
    get_temporary_object(call, new_expr);
  }
  else
  {
    // Base *x = new Base();
    exprt tmp_obj;
    get_temporary_object(call, tmp_obj);
    convert_expression_to_code(tmp_obj);
    new_expr = side_effect_exprt(
      "cpp_new", pointer_typet(symbol_typet(prefix + contract_name)));
    new_expr.initializer(tmp_obj);
  }
  return false;
}

bool solidity_convertert::get_implicit_ctor_ref(
  const std::string &contract_name,
  const bool is_object,
  exprt &new_expr)
{
  log_debug("solidity", "\t\tgetting implicit ctor call");

  // to obtain the type info
  std::string id;
  id = get_implict_ctor_call_id(contract_name);
  if (context.find_symbol(id) == nullptr)
  {
    if (add_implicit_constructor(contract_name))
      return true;
  }

  exprt ctor = symbol_expr(*context.find_symbol(id));
  code_typet type;
  type.return_type() = symbol_typet(prefix + contract_name);

  side_effect_expr_function_callt call;
  call.function() = ctor;
  call.set("constructor", 1);
  call.type() = type.return_type();
  call.location().file(absolute_path);

  exprt this_object;
  get_new_object(symbol_typet(prefix + contract_name), this_object);
  this_object = address_of_exprt(this_object);
  call.arguments().push_back(this_object);

  if (is_object)
  {
    // Base x = &sideefect(..)
    get_temporary_object(call, new_expr);
  }
  else
  {
    // Base *x = new Base();
    exprt tmp_obj;
    get_temporary_object(call, tmp_obj);
    convert_expression_to_code(tmp_obj);
    new_expr = side_effect_exprt(
      "cpp_new", pointer_typet(symbol_typet(prefix + contract_name)));
    new_expr.initializer(tmp_obj);
  }
  return false;
}
