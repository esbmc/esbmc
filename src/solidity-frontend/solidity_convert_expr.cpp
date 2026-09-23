/// \file solidity_convert_expr.cpp
/// \brief Expression conversion for the Solidity frontend.
///
/// Converts Solidity expressions (binary/unary operations, assignments,
/// conditional expressions, index access, member access, type conversions,
/// function calls, and identifier references) from the solc JSON AST into
/// ESBMC's irep2 expression tree.

#include <functional>
#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_expr.h>
#include <util/std_code.h>
#include <util/config.h>
#include <util/message.h>
#include <fstream>

bool solidity_convertert::get_expr(const nlohmann::json &expr, exprt &new_expr)
{
  return get_expr(expr, nullptr, new_expr);
}

/**
     * @brief Populate the out parameter with the expression based on
     * the solidity expression grammar. 
     * 
     * More specifically, parse each expression in the AST json and
     * convert it to a exprt ("new_expr"). The expression may have sub-expression
     * 
     * !Always check if the expression is a Literal before calling get_expr
     * !Unless you are 100% sure it will not be a constant
     * 
     * This function is called through two paths:
     * 1. get_non_function_decl => get_var_decl => get_expr
     * 2. get_non_function_decl => get_function_definition => get_statement => get_expr
     * 
     * @param expr The expression that is to be converted to the IR
     * @param literal_type Type information ast to create the the literal
     * type in the IR (only needed for when the expression is a literal).
     * A literal_type is a "typeDescriptions" ast_node.
     * we need this due to some info is missing in the child node.
     * @param new_expr Out parameter to hold the conversion
     * @return true iff the conversion has failed
     * @return false iff the conversion was successful
     */
bool solidity_convertert::get_expr(
  const nlohmann::json &expr,
  const nlohmann::json &literal_type,
  exprt &new_expr)
{
  assert(literal_type.is_null() || !literal_type.contains("typeDescriptions"));
  // For rule expression
  // We need to do location settings to match clang C's number of times to set the locations when recurring
  locationt location;
  get_start_location_from_stmt(expr, location);

  std::string current_contractName;
  get_current_contract_name(expr, current_contractName);

  SolidityGrammar::ExpressionT type = SolidityGrammar::get_expression_t(expr);
  log_debug(
    "solidity",
    "  @@@ got Expr: SolidityGrammar::ExpressionT::{}",
    SolidityGrammar::expression_to_str(type));

  switch (type)
  {
  case SolidityGrammar::ExpressionT::BinaryOperatorClass:
  {
    if (get_binary_operator_expr(expr, new_expr))
      return true;

    break;
  }
  case SolidityGrammar::ExpressionT::UnaryOperatorClass:
  {
    if (get_unary_operator_expr(expr, literal_type, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::ConditionalOperatorClass:
  {
    // for Ternary Operator (...?...:...) only
    if (get_conditional_operator_expr(expr, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::DeclRefExprClass:
  {
    if (get_decl_ref_expr(expr, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::LiteralWithRational:
  {
    // extract integer literal
    std::string typeString = expr["typeDescriptions"]["typeString"];
    // Remove "int_const " prefix
    std::string value_str = typeString.substr(10);

    BigInt z_ext_value = string2integer(value_str);
    unsignedbv_typet type(256);
    new_expr = constant_exprt(
      integer2binary(z_ext_value, bv_width(type)),
      integer2string(z_ext_value),
      type);

    break;
  }
  case SolidityGrammar::ExpressionT::Literal:
  {
    if (get_literal_expr(expr, literal_type, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::LiteralWithWei:
  case SolidityGrammar::ExpressionT::LiteralWithGwei:
  case SolidityGrammar::ExpressionT::LiteralWithSzabo:
  case SolidityGrammar::ExpressionT::LiteralWithFinney:
  case SolidityGrammar::ExpressionT::LiteralWithEther:
  case SolidityGrammar::ExpressionT::LiteralWithSeconds:
  case SolidityGrammar::ExpressionT::LiteralWithMinutes:
  case SolidityGrammar::ExpressionT::LiteralWithHours:
  case SolidityGrammar::ExpressionT::LiteralWithDays:
  case SolidityGrammar::ExpressionT::LiteralWithWeeks:
  case SolidityGrammar::ExpressionT::LiteralWithYears:
  {
    // e.g. _ESBMC_ether(1);
    assert(expr.contains("subdenomination"));
    std::string unit_name = expr["subdenomination"];

    nlohmann::json node = expr; // do copy
    // remove unit
    //! note that this will leads to "failed to get current contract name" error
    // however, since this can only be int_literal, we should be safe to do so
    node.erase("subdenomination");
    exprt l_expr;
    if (get_expr(node, literal_type, l_expr))
      return true;

    std::string f_name = "_ESBMC_" + unit_name;
    std::string f_id = "c:@F@" + f_name;

    side_effect_expr_function_callt call;
    get_library_function_call_no_args(
      f_name, f_id, unsignedbv_typet(256), location, call);
    call.arguments().push_back(l_expr);

    new_expr = call;
    break;
  }
  case SolidityGrammar::ExpressionT::Tuple:
  {
    if (get_tuple_expr(expr, literal_type, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::CallOptionsExprClass:
  {
    // e.g.
    // 1.
    // address(tmp).call{gas: 1000000, value: 1 ether}(abi.encodeWithSignature("register(string)", "MyName"));
    // 2.
    // function foo(uint a, uint b) public pure returns (uint) {
    //   return a + b;
    // }
    // function callFoo() public pure returns (uint) {
    //     return foo({a: 1, b: 2});
    // }
    assert(expr.contains("expression"));
    nlohmann::json callee_expr_json = expr["expression"];

    // Extract only the "value" option by matching names[].
    // The AST has parallel arrays: names=["value","gas"], options=[expr1,expr2].
    // We look up "value" by name so the order doesn't matter.
    nlohmann::json value_opts = empty_json;
    if (expr.contains("names") && expr.contains("options"))
    {
      const auto &names = expr["names"];
      const auto &opts = expr["options"];
      for (size_t i = 0; i < names.size(); ++i)
      {
        if (names[i] == "value")
        {
          value_opts = nlohmann::json::array();
          value_opts.push_back(opts[i]);
          break;
        }
      }
    }

    if (SolidityGrammar::is_address_member_call(callee_expr_json))
    {
      if (!is_bound)
      {
        if (get_unbound_expr(expr, current_contractName, new_expr))
          return true;

        symbolt dump;
        get_llc_ret_tuple(dump);
        new_expr = symbol_expr(dump);
      }
      else
      {
        if (get_expr(callee_expr_json, value_opts, new_expr))
          return true;
      }
    }
    else
    {
      if (!is_bound)
      {
        if (get_unbound_expr(expr, current_contractName, new_expr))
          return true;

        break;
      }
      else
      {
        // e.g. target.deposit{value: msg.value}();
        exprt memcall;
        // target.deposit
        if (get_expr(callee_expr_json, value_opts, memcall))
          return true;

        new_expr = memcall;
      }
    }

    break;
  }
  case SolidityGrammar::ExpressionT::CallExprClass:
  {
    if (get_call_expr(expr, literal_type, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::ContractMemberCall:
  {
    if (get_contract_member_call_expr(expr, literal_type, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::ImplicitCastExprClass:
  {
    if (get_cast_expr(expr, new_expr, literal_type))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::IndexAccess:
  {
    if (get_index_access_expr(expr, literal_type, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::IndexRangeAccess:
  {
    if (get_index_range_access_expr(expr, literal_type, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::NewExpression:
  {
    if (get_new_object_expr(expr, literal_type, new_expr))
      return true;
    break;
  }

  // 1. ContractMemberCall: contractInstance.call()
  //                        contractInstanceArray[0].call()
  //                        contractInstance.x
  //    this should be handled in CallExprClass
  // 2. StructMemberCall: struct.member
  // 3. EnumMemberCall: enum.member
  // 4. AddressMemberCall: tx.origin, msg.sender, ...
  case SolidityGrammar::ExpressionT::StructMemberCall:
  {
    assert(expr.contains("expression"));
    const nlohmann::json caller_expr_json = expr["expression"];

    exprt base;
    if (get_expr(caller_expr_json, base))
      return true;

    const int struct_var_id = expr["referencedDeclaration"].get<int>();
    const nlohmann::json &struct_var_ref = find_decl_ref(struct_var_id);
    if (struct_var_ref == empty_json)
    {
      log_error("cannot find struct member reference");
      return true;
    }
    exprt comp;
    if (get_var_decl_ref(struct_var_ref, true, comp))
      return true;

    assert(comp.name() == expr["memberName"]);

    // If the member is a mapping (infinite array in --bound mode),
    // it was skipped from the struct type definition (breaks padding/gen_zero).
    // Redirect to a global infinite array keyed by the struct variable path.
    if (
      get_sol_type(comp.type()) == SolidityGrammar::SolType::MAPPING &&
      comp.type().is_array())
    {
      // Build path: base_name + "$" + field_name
      std::string base_name;
      if (caller_expr_json.contains("name"))
        base_name = caller_expr_json["name"].get<std::string>();
      else if (caller_expr_json.contains("memberName"))
        base_name = caller_expr_json["memberName"].get<std::string>();
      else
        base_name = std::to_string(expr["referencedDeclaration"].get<int>());

      std::string field_name = expr["memberName"].get<std::string>();
      std::string path_name = base_name + "$" + field_name;

      std::string current_cName;
      get_current_contract_name(expr, current_cName);

      std::string arr_name, arr_id;
      get_mapping_inf_arr_name(current_cName, path_name, arr_name, arr_id);

      if (context.find_symbol(arr_id) == nullptr)
      {
        locationt loc;
        get_start_location_from_stmt(expr, loc);
        std::string mod = get_modulename_from_path(absolute_path);

        // Populate the array subtype by walking the valueType chain,
        // same as get_var_decl() does for direct mapping state variables.
        typet arr_t = comp.type();
        assert(struct_var_ref.contains("typeName"));
        {
          typet *cur_type = &arr_t;
          const nlohmann::json *cur_node = &struct_var_ref["typeName"];
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

        symbolt arr_sym;
        get_default_symbol(arr_sym, mod, arr_t, arr_name, arr_id, loc);
        arr_sym.static_lifetime = true;
        arr_sym.file_local = true;
        arr_sym.lvalue = true;
        auto &added = *move_symbol_to_context(arr_sym);
        added.value = gen_zero(get_complete_type(arr_t, ns), true);
      }

      new_expr = symbol_expr(*context.find_symbol(arr_id));
      break;
    }

    new_expr = member_exprt(base, comp.name(), comp.type());

    break;
  }
  case SolidityGrammar::ExpressionT::EnumMemberCall:
  {
    assert(expr.contains("expression"));
    const int enum_id = expr["referencedDeclaration"].get<int>();
    const nlohmann::json &enum_member_ref =
      find_node_by_id(src_ast_json, enum_id);
    if (enum_member_ref == empty_json)
    {
      log_error("cannot find enum member reference for id {}", enum_id);
      return true;
    }

    if (get_enum_member_ref(enum_member_ref, new_expr))
      return true;

    break;
  }
  case SolidityGrammar::ExpressionT::AddressMemberCall:
  {
    // property: <address>.balance
    // function_call: <address>.transfer()
    // examples:
    // 1. address(this).balance;
    // 2.  A tmp = new A();
    //    address(tmp).balance;
    // 3. address x;
    //    x.balance;
    //    msg.sender.balance;
    //! Note that member call like msg.sender will not be handled here
    // The main difference is that, for case 1 we do not need to guess the contract instance
    // While in case 2, we need to utilize over-approximate modelling to bind the all possible instance
    //
    // algo:
    // 1. we add the property and function to the contract definition (not handled here)
    // 2. we create an auxiliary mapping to store the <addr, contract-instance-ptr> pair (not handled here)
    // 3. For case 2, where we only have the address, we need to obtain the object from the mapping
    // For case 1: => this->balance
    // For case 3: => tmp.balance
    const nlohmann::json &caller_expr_json = expr["expression"];
    const std::string mem_name = expr["memberName"].get<std::string>();

    SolidityGrammar::ExpressionT _type =
      SolidityGrammar::get_expression_t(caller_expr_json);
    log_debug(
      "solidity",
      "\t\t@@@ got = {}",
      SolidityGrammar::expression_to_str(_type));

    exprt base;
    switch (_type)
    {
    case SolidityGrammar::TypeConversionExpression:
    case SolidityGrammar::DeclRefExprClass:
    case SolidityGrammar::BuiltinMemberCall:
    case SolidityGrammar::StructMemberCall:
    case SolidityGrammar::IndexAccess:
    case SolidityGrammar::CallExprClass:
    {
      // e.g.
      // - address(msg.sender).call
      // - details.solver.call             (StructMemberCall base)
      // - leaders[i].call                 (IndexAccess base)
      // - getAddr().call                  (CallExprClass base)
      if (get_expr(caller_expr_json, base))
        return true;
      break;
    }
    default:
    {
      log_error(
        "unexpected address member access, got {}",
        SolidityGrammar::expression_to_str(_type));
      return true;
    }
    }

    // case 1, which is a type conversion node
    if (is_low_level_call(mem_name))
    {
      // `transfer` / `send` move ETH out of the sender regardless of who
      // receives it.  The bound model implements this correctly (deduct
      // from this->$balance, credit known recipients, fall back to
      // EOA-deduct for unknown addresses).  The unbound model used to
      // skip the deduct entirely and only emit the nondet re-entry, which
      // makes any balance-monotonicity / TOD-Balance property vacuously
      // hold.  Route value-moving builtins through the bound model in
      // both modes; keep `call` / `staticcall` / `delegatecall` /
      // `callcode` on the legacy unbound path because they may carry no
      // value and the existing nondet harness is the right
      // over-approximation for their reentry semantics.
      const bool moves_value = (mem_name == "transfer" || mem_name == "send");
      if (!is_bound && !moves_value)
      {
        if (get_unbound_expr(expr, current_contractName, new_expr))
          return true;

        // call, staticcall ...
        symbolt dump;
        get_llc_ret_tuple(dump);
        new_expr = symbol_expr(dump);
      }
      else
      {
        if (get_bound_low_level_call(
              expr, literal_type, mem_name, base, new_expr))
          return true;
      }
    }
    else if (is_low_level_property(mem_name))
    {
      // property i.e balance/codehash
      //
      // For address(this).balance and address(<localContractInstance>).balance
      // we know the underlying contract instance, so route to
      // get_builtin_property_expr in BOTH bound and unbound modes — that path
      // emits `this->$balance` (the same SSA cell the constructor wrote)
      // instead of allocating a fresh nondet that no constraint connects to
      // the constructor's balance initialization.
      //
      // For opaque addresses (msg.sender, address values that came from
      // outside the closed-world view), we used to keep the unbound
      // nondet short-circuit, but that silently ignored the EOA
      // balance map credited by library transfer/send/call — a
      // "write, read, see nothing" inconsistency.  Now route the
      // `balance` property through `get_builtin_property_expr`
      // unconditionally (the `_ESBMC_eoa_balance_of` helper's
      // `_ESBMC_eoa_get_or_init` auto-allocates a slot with nondet
      // initial value, which is identical soundness to the old
      // short-circuit for first-sight addresses, and honours the
      // credit updates afterwards).  Other properties
      // (`code`/`codehash`/`address`) keep the unbound short-circuit
      // — they have no equivalent persistent map.
      bool know_instance = false;
      if (
        _type == SolidityGrammar::TypeConversionExpression &&
        caller_expr_json.contains("arguments") &&
        caller_expr_json["arguments"].is_array() &&
        caller_expr_json["arguments"].size() == 1)
      {
        const auto &inner = caller_expr_json["arguments"][0];
        const std::string ts = inner.value("typeDescriptions", nlohmann::json{})
                                 .value("typeString", "");
        // typeString starts with "contract " for both `this` and any
        // declared contract-typed local (V x = new V(); ... address(x).balance)
        if (ts.rfind("contract ", 0) == 0)
          know_instance = true;
      }

      bool reads_balance = (mem_name == "balance");
      if (!is_bound && !know_instance && !reads_balance)
        new_expr = nondet_uint_expr;
      else
        get_builtin_property_expr(
          current_contractName, mem_name, base, location, new_expr);
    }
    else
    {
      log_error("unexpected address member access");
      return true;
    }

    break;
  }
  case SolidityGrammar::ExpressionT::LibraryMemberCall:
  case SolidityGrammar::ExpressionT::TypeMemberCall:
  {
    // TypeMemberCall
    // - A.call(); // A is a contract or library
    // - enum ActionChoices { GoLeft, GoRight, GoStraight, SitStill }
    //   ActionChoices constant defaultChoice = ActionChoices.GoStraight;

    // `super.f` captured as an r-value (e.g.
    // `function() internal returns (uint) x = super.f;`) — not a call.
    // The super-call path in get_call_expr only triggers when the
    // MemberAccess is actually the callee of a FunctionCall; for
    // r-value captures we must produce an opaque fn-ptr here.
    // [APPROX: UNDER] Indirect calls through the captured pointer
    // return nondet values (same treatment as other fn-ptr stores).
    if (
      expr.contains("expression") && expr["expression"].is_object() &&
      expr["expression"].contains("name") &&
      expr["expression"]["name"].is_string() &&
      expr["expression"]["name"].get<std::string>() == "super")
    {
      typet ptr_t = gen_pointer_type(empty_typet());
      ptr_t.set("#sol_func_ptr", true);
      set_sol_type(ptr_t, SolidityGrammar::SolType::FUNC_PTR);
      int fn_id = expr.value("referencedDeclaration", -1);
      if (fn_id >= 0)
      {
        exprt id_const = constant_exprt(
          integer2binary(fn_id + 1, bv_width(size_type())),
          integer2string(fn_id + 1),
          size_type());
        new_expr = typecast_exprt(id_const, ptr_t);
      }
      else
      {
        get_nondet_expr(ptr_t, new_expr);
      }
      break;
    }

    exprt base;
    const nlohmann::json caller_expr_json = expr["expression"];
    typet t;
    if (get_type_description(caller_expr_json["typeDescriptions"], t))
      return true;
    // referencedDeclaration may be absent/null on builtin member accesses
    // (.length, block.number, msg.sender, abi.encode, string.concat, ...).
    // In that case we just leave func_ref empty and let later branches
    // decide. Use int64_t narrowing to tolerate larger ids.
    static const nlohmann::json empty_ref = nlohmann::json::object();
    int ref_id = 0;
    if (
      expr.contains("referencedDeclaration") &&
      expr["referencedDeclaration"].is_number())
      ref_id = static_cast<int>(expr["referencedDeclaration"].get<int64_t>());
    const auto &func_ref =
      ref_id > 0 ? find_node_by_id(src_ast_json, ref_id) : empty_ref;

    if (get_sol_type(t) == SolidityGrammar::SolType::ENUM)
    {
      /*
      "expression": {
          "id": 12,
          "name": "ActionChoices",
          "nodeType": "Identifier",
          "overloadedDeclarations": [],
          "referencedDeclaration": 6,
          "typeDescriptions": {
              "typeIdentifier": "t_type$_t_enum$_ActionChoices_$6_$",
              "typeString": "type(enum test.ActionChoices)"
          }
      },
      "memberName": "GoStraight",
      "nodeType": "MemberAccess",
      "referencedDeclaration": 4,
      "typeDescriptions": {
          "typeIdentifier": "t_enum$_ActionChoices_$6",
          "typeString": "enum test.ActionChoices"
      }
      */
      if (get_enum_member_ref(func_ref, new_expr))
        return true;
      break;
    };

    // Handle qualified struct constructor: e.g. `Pairing.G1Point(1, 2)`.
    // The MemberAccess's referencedDeclaration points at a StructDefinition,
    // not a function. Build a struct-valued initializer from the positional
    // arguments and return early — the struct-constructor handler in the
    // generic CallExprClass branch (below) is only reached for unqualified
    // struct calls like `G1Point(1, 2)`.
    if (
      !func_ref.empty() && func_ref.contains("nodeType") &&
      func_ref["nodeType"] == "StructDefinition")
    {
      const nlohmann::json &parent_call =
        find_last_parent(src_ast_json["nodes"], expr);
      assert(parent_call.contains("arguments"));

      typet struct_t;
      if (get_type_description(parent_call["typeDescriptions"], struct_t))
        return true;
      // get_type_description returns a symbol_typet for struct types; resolve
      // via the namespace to the underlying struct_typet.
      const typet &resolved = ns.follow(struct_t);
      if (resolved.id() != irept::id_struct)
      {
        log_error("expected struct type for qualified struct constructor");
        return true;
      }
      struct_t = resolved;

      exprt inits = gen_zero(struct_t);
      const nlohmann::json &members = func_ref["members"];
      const nlohmann::json &ctor_args = parent_call["arguments"];
      for (size_t i = 0; i < inits.operands().size() && i < ctor_args.size() &&
                         i < members.size();
           i++)
      {
        exprt init;
        if (get_expr(ctor_args.at(i), members.at(i)["typeDescriptions"], init))
          return true;
        const struct_union_typet::componentt *c =
          &to_struct_type(struct_t).components().at(i);
        solidity_gen_typecast(ns, init, c->type());
        inits.operands().at(i) = init;
      }

      new_expr = inits;
      break;
    }

    if (get_expr(caller_expr_json, literal_type, base))
      return true;

    // If func_ref doesn't point at a real FunctionDefinition (e.g. the
    // MemberAccess was a builtin like `string.concat` with a null
    // referencedDeclaration) fail loudly instead of feeding an empty
    // node into get_library_function_call, which would dereference
    // missing `name`/`parameters` fields.
    // Events and errors qualified by their containing library/contract
    // (e.g. `L.Ev(arg)` from `emit L.Ev(arg)`) reference an
    // EventDefinition / ErrorDefinition rather than a FunctionDefinition.
    // Events have no runtime effect in our model and errors lower to a
    // bottom assertion at the emit/revert site, so the call can be
    // modelled as a no-op skip here.
    if (
      !func_ref.empty() && func_ref.contains("nodeType") &&
      (func_ref["nodeType"] == "EventDefinition" ||
       func_ref["nodeType"] == "ErrorDefinition"))
    {
      new_expr = code_skipt();
      return false;
    }

    // `C.x` where x is a state variable of the contract type C — e.g.
    // `C.x = g;` referencing a function-typed state variable. The
    // member access lowers to an opaque nondet of the variable's type;
    // assignments through it lose their effect (UNDER-approximation of
    // function-pointer storage), but the program becomes verifiable.
    if (
      !func_ref.empty() && func_ref.contains("nodeType") &&
      func_ref["nodeType"] == "VariableDeclaration")
    {
      typet var_t;
      if (
        func_ref.contains("typeName") &&
        func_ref["typeName"].contains("typeDescriptions"))
      {
        if (get_type_description(
              func_ref["typeName"]["typeDescriptions"], var_t))
          return true;
      }
      else
      {
        var_t = empty_typet();
      }
      get_nondet_expr(var_t, new_expr);
      return false;
    }

    if (
      func_ref.empty() || !func_ref.contains("nodeType") ||
      func_ref["nodeType"] != "FunctionDefinition")
    {
      log_error(
        "TypeMemberCall: referencedDeclaration does not point at a "
        "FunctionDefinition (likely an unsupported builtin member access)");
      return true;
    }

    // Function reference used as a VALUE rather than as a call target.
    // e.g. `Base._b` stored in a function-pointer array literal or
    // assigned to a function-typed variable: the nearest parent is a
    // TupleExpression / VariableDeclarationStatement / Assignment, not a
    // FunctionCall whose `.expression` is this MemberAccess. Mirror the
    // `super.f` r-value lowering above — emit an opaque void* typecast
    // of the referenced FunctionDefinition's AST id so identity
    // comparisons stay sound; indirect calls through the pointer already
    // lower to nondet returns in convert_call.
    //
    // Without this branch, the assert below fires (the TupleExpression
    // has `components`, not `arguments`) and the converter crashes.
    {
      const nlohmann::json &parent_json =
        find_last_parent(src_ast_json["nodes"], expr);
      bool used_as_call_target =
        !parent_json.empty() && parent_json.contains("nodeType") &&
        (parent_json["nodeType"] == "FunctionCall" ||
         parent_json["nodeType"] == "FunctionCallOptions") &&
        parent_json.contains("expression") &&
        parent_json["expression"].contains("id") && expr.contains("id") &&
        parent_json["expression"]["id"] == expr["id"];
      if (!used_as_call_target)
      {
        typet ptr_t = gen_pointer_type(empty_typet());
        ptr_t.set("#sol_func_ptr", true);
        set_sol_type(ptr_t, SolidityGrammar::SolType::FUNC_PTR);
        int fn_id = expr.value("referencedDeclaration", -1);
        if (fn_id >= 0)
        {
          exprt id_const = constant_exprt(
            integer2binary(fn_id + 1, bv_width(size_type())),
            integer2string(fn_id + 1),
            size_type());
          new_expr = typecast_exprt(id_const, ptr_t);
        }
        else
        {
          get_nondet_expr(ptr_t, new_expr);
        }
        break;
      }
    }

    side_effect_expr_function_callt call;

    const nlohmann::json &args_json =
      find_last_parent(src_ast_json["nodes"], expr);
    assert(args_json.contains("arguments"));
    bool is_using_for =
      !(base.is_code() &&
        get_sol_type(base.type()) == SolidityGrammar::SolType::LIBRARY);
    if (get_library_function_call(func_ref, args_json, call, is_using_for))
      return true;

    if (is_using_for)
      // this means it is a using for library
      call.arguments().insert(call.arguments().begin(), base);

    // For library calls, copy modified storage-reference parameters back to
    // the caller's state variable via the global $out bridge created in
    // solidity_convert_modifier.cpp. The bridge is needed because the
    // parameter's SSA version is scoped to the function and not visible
    // after the call returns.
    //
    // Skip `view`/`pure` functions: they provably cannot modify a storage
    // parameter, so the copy-back is a no-op — and for a tuple-return call
    // (e.g. the 1inch/aqua `(amount, count) = balance.load()` pattern) the
    // copy-back is queued to the back-block DURING call conversion while the
    // call itself is queued AFTER (solidity_convert_tuple.cpp), so the
    // assignment would land BEFORE the call and clobber the caller's state
    // variable with the still-zero `$out` bridge. Gating on mutability is
    // both sound (view/pure never writes) and avoids that mis-ordering.
    const std::string state_mutability =
      func_ref.value("stateMutability", std::string());
    const bool call_can_modify_storage =
      state_mutability != "view" && state_mutability != "pure";
    if (
      call_can_modify_storage && func_ref.contains("id") &&
      SolidityGrammar::is_sol_library_function(func_ref["id"].get<int>()) &&
      func_ref.contains("parameters") &&
      func_ref["parameters"].contains("parameters"))
    {
      const auto &params = func_ref["parameters"]["parameters"];
      bool has_storage_param = false;
      for (const auto &p : params)
      {
        if (p.contains("storageLocation") && p["storageLocation"] == "storage")
        {
          has_storage_param = true;
          break;
        }
      }

      if (has_storage_param)
      {
        std::string lib_cname =
          find_contract_name_for_id(func_ref["id"].get<int>());
        std::string func_name = func_ref.value("name", std::string());

        for (size_t i = 0; i < params.size() && i < call.arguments().size();
             ++i)
        {
          const auto &p = params[i];
          if (
            !p.contains("storageLocation") || p["storageLocation"] != "storage")
            continue;

          std::string out_id = get_library_param_id(
                                 lib_cname,
                                 func_name,
                                 p["name"].get<std::string>(),
                                 p["id"].get<int>()) +
                               "$out";
          const symbolt *out_sym = context.find_symbol(out_id);
          if (!out_sym)
            continue;

          exprt &arg = call.arguments()[i];
          if (arg.type() != out_sym->type)
            continue;

          move_to_back_block(code_assignt(arg, symbol_expr(*out_sym)));
        }
      }
    }

    new_expr = call;
    break;
  }
  case SolidityGrammar::ExpressionT::BuiltinMemberCall:
  {
    if (get_sol_builtin_ref(expr, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ExpressionT::TypePropertyExpression:
  {
    // e.g.
    // Integers: 'type(uint256)'.max/min
    // Contracts: 'type(MyContract)'.creationCode/runtimecode

    // create a dump expr with no value, but set the correct type
    assert(
      expr.contains("expression") &&
      expr["expression"].contains("argumentTypes"));
    const auto &json = expr["expression"]["argumentTypes"];

    /*
    e.g.
      "argumentTypes": [
          {
              "typeIdentifier": "t_type$_t_int256_$",
              "typeString": "type(int256)"
          }
      ],
    */
    typet t;
    if (get_type_description(json[0], t))
      return true;

    exprt dump;
    dump.type() = t;

    new_expr = dump;
    break;
  }
  case SolidityGrammar::ExpressionT::TypeConversionExpression:
  {
    // perform type conversion
    // e.g.
    // address payable public owner = payable(msg.sender);
    // or
    // uint32 a = 0x432178;
    // uint16 b = uint16(a); // b will be 0x2178 now

    assert(expr.contains("expression"));
    const nlohmann::json conv_expr = expr["expression"];
    typet type;
    exprt from_expr;

    // 2. get target type (compute first so we can use it to pick a safe
    //    literal_type to pass down to the argument)
    if (get_type_description(conv_expr["typeDescriptions"], type))
      return true;

    // 1. get source expr.
    //
    // Default: forward the outer caller's `literal_type` so that downstream
    // literal handlers (e.g. `bytes32(0)` → bytes-zero helper) keep working.
    //
    // Exception 1: if the *outer* literal_type is a contract pointer (the
    // caller is, e.g., comparing this conversion's result against a state
    // var of a contract type), passing it down for a nested cast like
    // `Pool(address(0))` cascades to `convert_integer_literal`, which
    // calls `bv_width(pointer)` on the literal `0` and emits a
    // constant_exprt whose value() slot is empty — that then trips
    // `migrate expr failed` in the goto layer. Strip the contract-pointer
    // hint and let the argument be sized against its own typeDescriptions.
    //
    // Exception 2 (2026-05-10): if the outer literal_type would force a
    // NESTED inner-cast's argument to be lowered through a wrong-type
    // path, override with this cast's own dest type. Concrete trigger:
    // `bytes32(uint256(1))` — outer caller's literal_type is bytes32, so
    // both casts forward bytes32 down to the literal `1`; the literal
    // gets emitted as a BytesStatic, the inner uint256 cast then routes
    // through `bytes_static_to_uint`, and the outer bytes32 cast wraps
    // it with `bytes_static_from_uint` — a 64-step round-trip emitting
    // ~45 post-slice SSA per occurrence (Stage 0 hotspot at 15-30% of
    // post-slice live SSA on `napp_struct_multifield_fail`). Detect a
    // type mismatch between the outer hint and this cast's dest and
    // strip the outer hint so the literal is sized against this cast's
    // OWN dest instead.
    nlohmann::json arg_literal_type = literal_type;
    {
      typet hint_type;
      if (
        literal_type != nullptr && !literal_type.is_null() &&
        !get_type_description(literal_type, hint_type) &&
        get_sol_type(hint_type) == SolidityGrammar::SolType::CONTRACT &&
        hint_type.is_pointer())
      {
        arg_literal_type = expr["arguments"][0].contains("typeDescriptions")
                             ? expr["arguments"][0]["typeDescriptions"]
                             : nlohmann::json(nullptr);
      }
      // Type-incompatibility strip: outer hint differs from this cast's
      // dest type. The hint is a hint for ABOVE this cast; passing it
      // through this cast's argument processing forces the argument
      // (often a literal) into the wrong type before THIS cast even
      // gets to convert it. Use this cast's `typeDescriptions` instead
      // — the conversion EXPLICITLY changes the type, so the argument
      // should be lowered to match this cast's dest, not the outer
      // caller's expectation.
      else if (
        literal_type != nullptr && !literal_type.is_null() &&
        !get_type_description(literal_type, hint_type) && !(hint_type == type))
      {
        arg_literal_type = conv_expr.contains("typeDescriptions")
                             ? conv_expr["typeDescriptions"]
                             : nlohmann::json(nullptr);
      }
    }
    //! assume: only one argument
    assert(expr["arguments"].size() == 1);
    if (get_expr(expr["arguments"][0], arg_literal_type, from_expr))
      return true;

    // Save the pre-cast address expression — for contract casts we need
    // the ORIGINAL address value to populate the per-pointer bind shadow
    // by matching against each singleton's `$address`.  After
    // `convert_type_expr` runs, `from_expr` has been rewritten to the
    // target singleton pointer and the argument address is no longer
    // visible.  (The cast itself also clobbers `_ESBMC_Object_<target>.
    // $address` globally — see convert_type_expr:1622 — so the if-ladder
    // must skip the declared-type branch, otherwise the clobber would
    // trivially match.)
    exprt pre_cast = from_expr;

    // 3. generate the type casting expr
    convert_type_expr(ns, from_expr, type, expr);

    new_expr = from_expr;

    // Contract-cast bind shadow emission.
    // `new C()` writes both the singleton field and the per-pointer
    // shadow to the declared C.  For cast form `C(_addr)` there is no
    // single "correct" binding — the declared type C covers any of the
    // structural cluster, and the real binding depends on which
    // singleton's address matches `_addr`.  Emit an address-match
    // if-ladder updating the shadow only (leaving the singleton field
    // untouched so the legacy function-call dispatcher is unaffected).
    //
    // Ladder structure (declared_cname skipped inside the loop):
    //   if (_addr == _ESBMC_Object_X.$address) shadow = X
    //   else if (_addr == _ESBMC_Object_Y.$address) shadow = Y
    //   else shadow = declared_cname
    //
    // Skipping the declared-cname comparison is essential: the cast
    // already rewrote `_ESBMC_Object_<declared>.$address = _addr`
    // before this ladder runs, so that comparison would trivially hit
    // and short-circuit the legitimate alternatives.
    if (get_sol_type(type) == SolidityGrammar::SolType::CONTRACT)
    {
      const std::string declared_cname = type.get("#sol_contract").as_string();
      if (!declared_cname.empty())
      {
        const nlohmann::json &parent = find_last_parent(src_ast_json, expr);
        exprt lvar;
        bool got_lvar = false;
        if (parent.contains("nodeType"))
        {
          if (parent["nodeType"] == "VariableDeclarationStatement")
          {
            if (!get_var_decl_ref(parent["declarations"][0], true, lvar))
              got_lvar = true;
          }
          else if (parent["nodeType"] == "VariableDeclaration")
          {
            if (!get_var_decl_ref(parent, true, lvar))
              got_lvar = true;
          }
          else if (parent["nodeType"] == "Assignment")
          {
            if (!get_expr(parent["leftHandSide"], lvar))
              got_lvar = true;
          }
        }

        if (got_lvar)
        {
          exprt shadow;
          if (!get_or_create_bind_shadow(lvar, declared_cname, shadow))
          {
            // Prefer "shadow propagation" when the cast's argument is of
            // the shape `address(src_var)` where src_var is a contract-
            // typed local variable that already has a `$bind` shadow.
            // In that case the correct binding for the new pointer is
            // whatever src_var's shadow says — this sidesteps the
            // singleton-`$address` comparison approach, which doesn't
            // work when `_ESBMC_Object_<X>.$address` is never
            // meaningfully initialised (e.g. `--contract Test` only
            // runs Test's ctor; A2's singleton ctor never executes, so
            // `_ESBMC_Object_A2.$address` holds a default value that
            // doesn't match any `new`-created pointer's address).
            bool propagated_from_shadow = false;
            const nlohmann::json &cast_arg = expr["arguments"][0];
            if (
              cast_arg.is_object() &&
              cast_arg.value("nodeType", "") == "FunctionCall" &&
              cast_arg.value("kind", "") == "typeConversion" &&
              cast_arg.contains("arguments") &&
              cast_arg["arguments"].is_array() &&
              cast_arg["arguments"].size() == 1)
            {
              const nlohmann::json &inner = cast_arg["arguments"][0];
              if (
                inner.is_object() &&
                inner.value("nodeType", "") == "Identifier")
              {
                exprt src_var;
                if (!get_expr(inner, src_var))
                {
                  exprt src_shadow;
                  if (!get_bind_shadow_read(src_var, src_shadow))
                  {
                    exprt a = side_effect_exprt("assign", shadow.type());
                    solidity_gen_typecast(ns, src_shadow, shadow.type());
                    a.copy_to_operands(shadow, src_shadow);
                    convert_expression_to_code(a);
                    codet c = to_code(a);
                    c.location() = location;
                    move_to_back_block(c);
                    propagated_from_shadow = true;
                  }
                }
              }
            }

            if (!propagated_from_shadow)
            {
              // Fallback: address-match if-ladder against each
              // singleton.  Only works when singletons' `$address`
              // values are populated (multi-contract mode where every
              // contract's ctor runs).  Declared cname is always the
              // innermost else so that the clobbered-singleton trap on
              // the declared type can't short-circuit.
              auto make_shadow_assign = [&](const std::string &cn) -> codet {
                exprt rhs;
                get_cname_expr(cn, rhs);
                solidity_gen_typecast(ns, rhs, shadow.type());
                exprt a = side_effect_exprt("assign", shadow.type());
                a.copy_to_operands(shadow, rhs);
                convert_expression_to_code(a);
                return to_code(a);
              };

              exprt addr_val = pre_cast;
              solidity_gen_typecast(ns, addr_val, addr_t);

              codet ladder = make_shadow_assign(declared_cname);

              std::unordered_set<std::string> cname_set =
                structureTypingMap[declared_cname];
              for (auto non_cname : nonContractNamesList)
              {
                if (non_cname == declared_cname)
                  continue;
                cname_set.erase(non_cname);
              }

              for (const auto &alt_cname : cname_set)
              {
                if (alt_cname == declared_cname)
                  continue;

                const symbolt *struct_sym =
                  context.find_symbol(prefix + alt_cname);
                if (
                  struct_sym == nullptr || struct_sym->type.id() != "struct" ||
                  !to_struct_type(struct_sym->type).has_component("$address"))
                  continue;

                exprt obj_ref;
                get_static_contract_instance_ref(alt_cname, obj_ref);
                exprt obj_addr = member_exprt(obj_ref, "$address", addr_t);

                exprt cmp = exprt("=", bool_type());
                cmp.copy_to_operands(addr_val, obj_addr);

                codet if_expr("ifthenelse");
                if_expr.copy_to_operands(
                  cmp, make_shadow_assign(alt_cname), ladder);
                ladder = if_expr;
              }

              ladder.location() = location;
              move_to_back_block(ladder);
            }
          }
        }
      }
    }
    break;
  }
  case SolidityGrammar::ExpressionT::NullExpr:
  {
    // e.g. (, x) = (1, 2);
    // the first component in lhs is nil
    new_expr = nil_exprt();
    break;
  }
  case SolidityGrammar::ExpressionT::ElementaryTypeNameExpr:
  {
    // A bare type used as an expression value
    // (e.g. `string` in `string.concat(...)`). There is no runtime
    // value; callers read the type off of typeDescriptions.
    typet type;
    if (get_type_description(expr["typeDescriptions"], type))
      return true;
    exprt dump;
    dump.type() = type;
    new_expr = dump;
    break;
  }
  default:
  {
    log_error(
      "Unimplemented expression type: {}",
      SolidityGrammar::expression_to_str(type));
    return true;
  }
  }

  new_expr.location() = location;

  log_debug(
    "solidity",
    "@@@ Finish parsing expresion = {}",
    SolidityGrammar::expression_to_str(type));
  return false;
}

bool solidity_convertert::get_decl_ref_expr(
  const nlohmann::json &expr,
  exprt &new_expr)
{
  if (expr["referencedDeclaration"] > 0)
  {
    // Delegate-shadow parameter remap: when inlining a target function
    // body at a .delegatecall(...) call site, references to the target's
    // formal parameters are redirected to pre-declared locals in the
    // caller's scope. Check this first so we never even look up the
    // target function's parameter symbol (which would be out of scope).
    int ref_id = expr["referencedDeclaration"].get<int>();
    if (!delegate_shadow_param_remap.empty())
    {
      auto it = delegate_shadow_param_remap.find(ref_id);
      if (it != delegate_shadow_param_remap.end())
      {
        if (context.find_symbol(it->second) == nullptr)
        {
          log_error(
            "delegate_shadow_param_remap target {} not in symbol table",
            it->second);
          return true;
        }
        new_expr = symbol_expr(*context.find_symbol(it->second));
        return false;
      }
    }

    // Resolve storage reference aliases: if this identifier refers to a
    // local storage variable that aliases another, redirect to the source.
    // Loop handles chained aliases (Wrapper storage a = b; ... = a;).
    for (auto it = storage_ref_aliases.find(ref_id);
         it != storage_ref_aliases.end();
         it = storage_ref_aliases.find(ref_id))
      ref_id = it->second;

    // Resolve expression-based storage ref aliases: e.g. campaigns[0]
    {
      auto it = storage_ref_expr_aliases.find(ref_id);
      if (it != storage_ref_expr_aliases.end())
      {
        const nlohmann::json &init_expr = it->second;
        if (get_expr(init_expr, new_expr))
          return true;
        return false;
      }
    }

    // Solidity uses +ve odd numbers to refer to var or functions declared in the contract
    nlohmann::json decl = find_decl_ref(ref_id);
    if (decl.empty())
    {
      log_error(
        "failed to find the reference AST node, base contract name {}, "
        "reference id {}",
        current_baseContractName,
        std::to_string(expr["referencedDeclaration"].get<int>()));
      return true;
    }

    if (!check_intrinsic_function(decl))
    {
      log_debug(
        "solidity",
        "\t\t@@@ got nodeType={}",
        decl["nodeType"].get<std::string>());
      if (decl["nodeType"] == "VariableDeclaration")
      {
        if (get_var_decl_ref(decl, true, new_expr))
          return true;
      }
      else if (decl["nodeType"] == "FunctionDefinition")
      {
        if (get_func_decl_ref(decl, new_expr))
          return true;
      }
      else if (
        decl["nodeType"] == "StructDefinition" ||
        decl["nodeType"] == "ErrorDefinition" ||
        decl["nodeType"] == "EventDefinition" ||
        (decl["nodeType"] == "ContractDefinition" &&
         decl["contractKind"] == "library"))
      {
        if (get_noncontract_decl_ref(decl, new_expr))
          return true;
      }
      else if (decl["nodeType"] == "ContractDefinition")
      {
        if (current_functionDecl)
        {
          if (get_func_decl_this_ref(*current_functionDecl, new_expr))
            return true;
        }
        else if (!expr.empty())
        {
          if (get_ctor_decl_this_ref(expr, new_expr))
            return true;
        }
      }
      else if (decl["nodeType"] == "UserDefinedValueTypeDefinition")
      {
        // Bare identifier reference to a UDVT (e.g. `MyInt` used as
        // the callee of `MyInt(x)` or the base of `MyInt.wrap(x)`).
        // Lower it to a typecast_exprt carrying the UDVT's underlying
        // type and no operand — the FunctionCall path fills op0, and
        // the MemberAccess `.wrap` / `.unwrap` handler in
        // solidity_convert_ref.cpp short-circuits via the member name
        // rather than descending into the base. For type-argument
        // positions (e.g. `abi.decode(ret, (MyInt, ...))`) the
        // builtin path filters by typeIdentifier `t_type$...` and
        // never uses this exprt.
        typet t;
        if (get_type_description(decl["underlyingType"]["typeDescriptions"], t))
          return true;
        new_expr = typecast_exprt(t);
      }
      else
      {
        log_error(
          "Unsupported DeclRefExprClass type, got nodeType={}",
          decl["nodeType"].get<std::string>());
        return true;
      }
    }
    else
    {
      // for special functions, we need to deal with it separately
      if (get_esbmc_builtin_ref(expr, new_expr))
        return true;
    }
  }
  else
  {
    if (expr.contains("name") && expr["name"] == "this")
    {
      log_debug("solidity", "\t\tgot this ref");

      exprt this_expr;
      // Prefer the enclosing function's `this` when available. Outside a
      // function (e.g. state-variable initializers processed during
      // constructor synthesis) fall back to the current base contract's
      // constructor-this ref. solc 0.8.x emits `address(this)` inside
      // state-variable initializers that the frontend lowers in ctor
      // context without `current_functionDecl` being set.
      if (current_functionDecl)
      {
        if (get_func_decl_this_ref(*current_functionDecl, this_expr))
          return true;
      }
      else if (!current_baseContractName.empty())
      {
        if (get_ctor_decl_this_ref(current_baseContractName, this_expr))
          return true;
      }
      else
      {
        log_error("`this` referenced outside any function context");
        return true;
      }
      new_expr = this_expr;
    }
    else
    {
      // Solidity uses -ve odd numbers to refer to built-in var or functions that
      // are NOT declared in the contract
      if (get_esbmc_builtin_ref(expr, new_expr))
        return true;
    }
  }

  return false;
}

bool solidity_convertert::get_literal_expr(
  const nlohmann::json &expr,
  const nlohmann::json &literal_type,
  exprt &new_expr)
{
  locationt location;
  get_start_location_from_stmt(expr, location);

  std::string current_contractName;
  get_current_contract_name(expr, current_contractName);

  // make a type-name json for integer literal conversion
  // hex string literals (e.g. `hex"123456789a"`) ship with kind ==
  // "hexString" and only a `hexValue` field — solc omits `value`
  // entirely. Querying `expr["value"]` on those throws json
  // type_error.302 ("type must be string, but is discarded"). Fall
  // back to hexValue when value is absent.
  std::string the_value;
  if (expr.contains("value") && !expr["value"].is_null())
    the_value = expr["value"].get<std::string>();
  else if (expr.contains("hexValue") && !expr["hexValue"].is_null())
    the_value = expr["hexValue"].get<std::string>();
  const nlohmann::json &literal = expr["typeDescriptions"];
  SolidityGrammar::ElementaryTypeNameT type_name =
    SolidityGrammar::get_elementary_type_name_t(literal);
  log_debug(
    "solidity",
    "	@@@ got Literal: SolidityGrammar::ElementaryTypeNameT::{}",
    SolidityGrammar::elementary_type_name_to_str(type_name));

  if (
    literal_type != nullptr && literal_type.contains("typeString") &&
    literal_type["typeString"].get<std::string>().find("bytes") !=
      std::string::npos)
  {
    // e.g.
    // bytes a = hex"1234";
    // bytes2 b = hex"1234";
    typet byte_t;
    if (get_type_description(literal_type, byte_t))
      return true;
    assert(is_byte_type(byte_t));
    std::string fname = is_bytesN_type(byte_t) ? "static" : "dynamic";
    bool is_static = fname == "static";
    std::string expected_size;
    if (is_static)
    {
      assert(!byte_t.get("#sol_bytesn_size").empty());
      expected_size = byte_t.get("#sol_bytesn_size").as_string();
    }

    if (type_name == SolidityGrammar::ElementaryTypeNameT::INT_LITERAL)
    {
      assert(is_static);
      side_effect_expr_function_callt call;
      std::string val_str = expr["value"].get<std::string>();

      if (val_str.rfind("0x", 0) == 0)
      {
        // e.g. 0x12, expected size is 2 → pad to 0x0012
        std::string hex_part = val_str.substr(2);
        size_t actual_len = hex_part.length() / 2; // actual bytes
        size_t expected_len = std::stoul(expected_size);

        if (actual_len < expected_len)
        {
          size_t missing = expected_len - actual_len;
          std::string padding(missing * 2, '0');
          hex_part = padding + hex_part;
          val_str = "0x" + hex_part;
        }

        get_library_function_call_no_args(
          "bytes_static_from_hex",
          "c:@F@bytes_static_from_hex",
          byte_t,
          location,
          call);

        exprt str = string_constantt(val_str);
        call.arguments().push_back(str);
        call.arguments().push_back(from_integer(val_str.length(), uint_type()));
      }
      else if (val_str == "0")
      {
        // e.g. bytes32 data3 = 0;
        get_library_function_call_no_args(
          "bytes_static_init_zero",
          "c:@F@bytes_static_init_zero",
          byte_t,
          location,
          call);
        assert(!byte_t.get("#sol_bytesn_size").empty());
        exprt len = from_integer(std::stoul(expected_size), uint_type());
        call.arguments().push_back(len);
      }
      else
      {
        // decimal integer literal, e.g. bytes1(uint8(7)).
        // Convert to a zero-padded hex string of the target width.
        size_t expected_len = std::stoul(expected_size);
        BigInt v = string2integer(val_str);
        std::string hex_part;
        if (v == 0)
          hex_part = "";
        else
          hex_part = integer2string(v, 16);
        if (hex_part.length() % 2 != 0)
          hex_part = "0" + hex_part;
        if (hex_part.length() / 2 < expected_len)
        {
          size_t missing = expected_len - hex_part.length() / 2;
          hex_part = std::string(missing * 2, '0') + hex_part;
        }
        else if (hex_part.length() / 2 > expected_len)
        {
          log_error(
            "Integer literal {} does not fit in bytes{}",
            val_str,
            expected_len);
          return true;
        }
        std::string hex_val = "0x" + hex_part;

        get_library_function_call_no_args(
          "bytes_static_from_hex",
          "c:@F@bytes_static_from_hex",
          byte_t,
          location,
          call);

        exprt str = string_constantt(hex_val);
        call.arguments().push_back(str);
        call.arguments().push_back(from_integer(hex_val.length(), uint_type()));
      }
      set_sol_type(call.type(), SolidityGrammar::SolType::BYTES_STATIC);
      call.type().set("#sol_bytesn_size", expected_size);
      new_expr = make_aux_var(call, location);
      return false;
    }
    else if (type_name == SolidityGrammar::ElementaryTypeNameT::STRING_LITERAL)
    {
      assert(expr.contains("kind") && expr.contains("hexValue"));
      std::string val_str;
      // solc emits `kind: "string"` for a Solidity string literal, but
      // when the literal contains non-printable bytes (e.g.
      // "\x42\x00\xef") it omits the `value` field entirely and only
      // ships the raw `hexValue`. Treat any literal without a `value`
      // field as a hex string so we don't crash trying to read it as
      // UTF-8 — the bytes_*_from_hex helper interprets the same hex
      // payload correctly regardless of the source-level kind.
      bool is_hex_string = expr["kind"] == "hexString" ||
                           !expr.contains("value") || expr["value"].is_null();
      if (is_hex_string)
        val_str = expr["hexValue"].get<std::string>();
      else
        val_str = expr["value"].get<std::string>();

      // add padding
      if (is_static && is_hex_string)
      {
        assert(!byte_t.get("#sol_bytesn_size").empty());
        size_t actual_len = val_str.length() / 2;
        size_t expected_len = std::stoul(expected_size);

        if (actual_len < expected_len)
        {
          size_t missing = expected_len - actual_len;
          std::string padding(missing * 2, '0');
          val_str = padding + val_str;
        }
        else if (actual_len > expected_len)
        {
          log_error(
            "String literal is longer than target bytesN size ({} > {})",
            actual_len,
            expected_len);
          return true;
        }
      }
      if (is_hex_string)
        val_str = "0x" + val_str;

      exprt str = string_constantt(val_str);
      std::string posfix = is_hex_string ? "hex" : "string";

      side_effect_expr_function_callt str_call;
      get_library_function_call_no_args(
        "bytes_" + fname + "_from_" + posfix,
        "c:@F@bytes_" + fname + "_from_" + posfix,
        byte_t,
        location,
        str_call);

      str_call.arguments().push_back(str);
      if (is_hex_string)
        str_call.arguments().push_back(
          from_integer(val_str.length(), uint_type()));
      else if (is_static)
        str_call.arguments().push_back(
          from_integer(std::stoul(expected_size), uint_type()));
      if (!is_static)
      {
        exprt dynamic_pool;
        if (get_dynamic_pool(current_contractName, dynamic_pool))
          return true;
        set_sol_type(str_call.type(), SolidityGrammar::SolType::BYTES_DYN);
        str_call.arguments().push_back(dynamic_pool);
      }
      else
      {
        set_sol_type(str_call.type(), SolidityGrammar::SolType::BYTES_STATIC);
        str_call.type().set("#sol_bytesn_size", byte_t.get("#sol_bytesn_size"));
      }
      new_expr = make_aux_var(str_call, location);
      return false;
    }

    log_error("Unsupported bytes literal type in expression: {}", expr.dump());
    return true;
  }

  switch (type_name)
  {
  case SolidityGrammar::ElementaryTypeNameT::INT_LITERAL:
  {
    // literal_type may be null when the surrounding context did not
    // pin a destination type — most commonly through the LiteralWith*
    // (wei/ether/seconds/...) wrapper, which strips its subdenomination
    // and re-enters get_expr with the same literal_type it received,
    // and that literal_type came from a non-Literal AST node so the
    // return-statement plumbing left it as `nullptr`. In that case the
    // literal's own typeDescriptions (e.g. `t_rational_X_by_1`) is the
    // right fallback — it carries the int_const width metadata that
    // convert_integer_literal needs to size the result.
    const nlohmann::json &int_lit_t =
      (literal_type != nullptr) ? literal_type : literal;
    bool is_hex = false;
    if (the_value.length() >= 2 && the_value.substr(0, 2) == "0x")
      is_hex = true;
    else if (expr["kind"] == "hexString")
    {
      the_value = expr["hexValue"];
      is_hex = true;
    }

    if (is_hex) // meaning hex-string
    {
      if (convert_hex_literal(the_value, new_expr))
        return true;
      set_sol_type(new_expr.type(), SolidityGrammar::SolType::INT_CONST);
    }
    else if (convert_integer_literal(int_lit_t, the_value, new_expr))
      return true;

    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::BOOL:
  {
    if (convert_bool_literal(literal, the_value, new_expr))
      return true;
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::STRING_LITERAL:
  {
    if (convert_string_literal(the_value, new_expr))
      return true;

    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::ADDRESS:
  {
    if (convert_hex_literal(the_value, new_expr, 160))
      return true;
    set_sol_type(new_expr.type(), SolidityGrammar::SolType::ADDRESS);
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::ADDRESS_PAYABLE:
  {
    // 20 bytes
    if (convert_hex_literal(the_value, new_expr, 160))
      return true;
    set_sol_type(new_expr.type(), SolidityGrammar::SolType::ADDRESS_PAYABLE);
    break;
  }
  default:
    log_error("Unimplemented literal type");
    return true;
  }

  return false;
}

bool solidity_convertert::get_tuple_expr(
  const nlohmann::json &expr_in,
  const nlohmann::json &literal_type,
  exprt &new_expr)
{
  // "nodeType": "TupleExpression":
  //    1. InitList: uint[3] x = [1, 2, 3];
  //                         x = [1];  x = [1,2];
  //    2. Operator:
  //        - (x+1) % 2
  //        - if( x && (y || z) )
  //    3. TupleExpr:
  //        - multiple returns: return (x, y);
  //        - swap: (x, y) = (y, x)
  //        - constant: (1, 2)

  // Unwrap redundant parens around an inline array literal:
  //   return ([1, 2, 3, 4, 5]);
  // parses as TupleExpression{components:[InlineArray{...}]}. Without
  // unwrapping we'd treat the outer as an array of size 1 and recurse
  // into the inline array with the element typeDescriptions, which then
  // crashes make_array_elementary_type on the non-array t_uint8 id.
  const nlohmann::json *exprp = &expr_in;
  while (exprp->is_object() &&
         exprp->value("nodeType", "") == "TupleExpression" &&
         exprp->contains("components") && (*exprp)["components"].is_array() &&
         (*exprp)["components"].size() == 1 &&
         !(*exprp)["components"][0].is_null() &&
         (*exprp)["components"][0].is_object() &&
         (*exprp)["components"][0].value("nodeType", "") == "TupleExpression" &&
         (*exprp)["components"][0].value("isInlineArray", false))
    exprp = &(*exprp)["components"][0];
  const nlohmann::json &expr = *exprp;

  if (!expr.contains("components"))
  {
    log_error("Unexpected ast json structure, expecting component");
    abort();
  }
  SolidityGrammar::TypeNameT type =
    SolidityGrammar::get_type_name_t(expr["typeDescriptions"]);

  switch (type)
  {
  // case 1
  case SolidityGrammar::TypeNameT::ArrayTypeName:
  {
    // Prefer the tuple's own typeDescriptions: for array literals like
    // `return [a, b, c]` the Return stmt leaves literal_type as null
    // because the expression isn't a plain Literal, but expr itself
    // already carries the array type (e.g. "uint256[3] memory").
    const nlohmann::json &arr_literal_type =
      (literal_type != nullptr) ? literal_type : expr["typeDescriptions"];

    // get elem type
    nlohmann::json elem_literal_type =
      make_array_elementary_type(arr_literal_type);

    // get size
    exprt size;
    size = constant_exprt(
      integer2binary(expr["components"].size(), bv_width(int_type())),
      integer2string(expr["components"].size()),
      int_type());

    // get array type
    typet arr_type;
    if (get_type_description(arr_literal_type, arr_type))
      return true;

    // reallocate array size
    arr_type = array_typet(arr_type.subtype(), size);

    // declare static array tuple
    exprt inits;
    inits = gen_zero(arr_type);
    set_sol_type(inits.type(), SolidityGrammar::SolType::ARRAY_LITERAL);
    inits.type().set("#sol_array_size", size.cformat().as_string());

    // populate array
    int i = 0;
    for (const auto &arg : expr["components"].items())
    {
      exprt init;
      if (get_expr(arg.value(), elem_literal_type, init))
        return true;

      inits.operands().at(i) = init;
      i++;
    }
    inits.id("array");

    // They will be covnerted to an aux array in convert_type_expr() function
    new_expr = inits;
    break;
  }

  // case 3
  case SolidityGrammar::TypeNameT::TupleTypeName: // case 3
  {
    /*
      we assume there are three types of tuple expr:
      0. dump: (x,y);
      1. fixed: (x,y) = (y,x);
      2. function-related: 
          2.1. (x,y) = func();
          2.2. return (x,y);

      case 0:
        1. create a struct type
        2. create a struct type instance
        3. new_expr = instance
        e.g.
        (x , y) ==>
        struct Tuple
        {
          uint x,
          uint y
        };
        Tuple tuple;

      case 1:
        1. add special handling in binary operation.
           when matching struct_expr A = struct_expr B,
           divided into A.operands()[i] = B.operands()[i]
           and populated into a code_block.
        2. new_expr = code_block
        e.g.
        (x, y) = (1, 2) ==>
        {
          tuple.x = 1;
          tuple.y = 2;
        }

      case 2:
        1. when parsing the function definition, if the returnParam > 1
           make the function return void instead, and create a struct type
        2. when parsing the return statement, if the return value is a tuple,
           create a struct type instance, do assignments,  and return empty;
        3. when the lhs is tuple and rhs is func_call, get_tuple_instance_expr based 
           on the func_call, and do case 1.
        e.g.
        function test() returns (uint, uint)
        {
          return (1,2);
        }
        ==>
        struct Tuple
        {
          uint x;
          uint y;
        }
        function test()
        {
          Tuple tuple;
          tuple.x = 1;
          tuple.y = 2;
          return;
        }
      */

    if (current_lhsDecl)
    {
      // avoid nested
      assert(!current_rhsDecl);

      // we do not create struct-tuple instance for lhs
      // Null components (omitted positions like `(x, , y)`) become nil_exprt
      // to preserve positional alignment with the RHS tuple struct.
      code_blockt _block;
      for (const auto &i : expr["components"])
      {
        if (i.is_null() || !i.contains("typeDescriptions"))
        {
          _block.operands().push_back(nil_exprt());
          continue;
        }
        exprt op;
        if (get_expr(i, i["typeDescriptions"], op))
          return true;
        _block.operands().push_back(op);
      }
      new_expr = _block;
    }
    else
    {
      // 1. construct struct type
      if (get_tuple_definition(expr))
        return true;

      //2. construct struct_type instance
      if (get_tuple_instance(expr, new_expr))
        return true;
    }

    break;
  }

  // case 2
  default:
  {
    if (get_expr(expr["components"][0], literal_type, new_expr))
      return true;
    break;
  }
  }

  return false;
}

bool solidity_convertert::get_call_expr(
  const nlohmann::json &expr,
  const nlohmann::json &literal_type,
  exprt &new_expr)
{
  side_effect_expr_function_callt call;
  // Unwrap redundant parens around the callee: `(L.f)()` parses as a
  // FunctionCall whose `.expression` is a single-component TupleExpression
  // wrapping the real callee (Identifier or MemberAccess). The downstream
  // dispatch tests `nodeType == "MemberAccess"` etc. directly and would
  // otherwise miss it, eventually feeding a tuple node into
  // get_library_function_call which dereferences fields that only exist on
  // a true callee.
  const nlohmann::json *callee_p = &expr["expression"];
  while (callee_p->is_object() &&
         callee_p->value("nodeType", "") == "TupleExpression" &&
         callee_p->contains("components") &&
         (*callee_p)["components"].is_array() &&
         (*callee_p)["components"].size() == 1 &&
         !(*callee_p)["components"][0].is_null())
    callee_p = &(*callee_p)["components"][0];

  // Constant-fold inline array literal with constant index in callee:
  //   [f, g][0](args)  →  f(args)
  // When the callee is an IndexAccess whose base is an inline array
  // literal (TupleExpression with isInlineArray) and the index is a
  // compile-time constant, redirect callee_p to the selected component
  // in the *original* AST.  This lets the resolved element (which
  // carries referencedDeclaration) go through the normal call dispatch.
  if (
    callee_p->value("nodeType", "") == "IndexAccess" &&
    callee_p->contains("baseExpression") &&
    callee_p->contains("indexExpression"))
  {
    const auto &base = (*callee_p)["baseExpression"];
    const auto &idx = (*callee_p)["indexExpression"];
    if (
      base.value("nodeType", "") == "TupleExpression" &&
      base.value("isInlineArray", false) && base.contains("components") &&
      idx.value("nodeType", "") == "Literal" &&
      idx.value("kind", "") == "number")
    {
      size_t k = std::stoull(idx["value"].get<std::string>());
      const auto &comps = base["components"];
      if (k < comps.size() && !comps[k].is_null())
        callee_p = &comps[k];
    }
  }

  const nlohmann::json &callee_expr_json = *callee_p;

  // * __ESOL_deep_copy intrinsic:
  //   user-written TOD harnesses can call `__ESOL_deep_copy(a)` to
  //   get a state-equivalent, isolated clone of a contract instance.
  //   The user declares a per-type stub so solc accepts the call:
  //     function __ESOL_deep_copy(C src) pure returns (C) { return src; }
  //   ESBMC ignores the stub body and lowers the call to
  //   `_ESBMC_clone_<C>(arg)`, which performs a per-field deep copy
  //   (scalars by value, pointer-backed fixed arrays reallocated via
  //   _ESBMC_arrcpy, mappings retargeted to a fresh $address) and
  //   mints a fresh contract identity.  See build_tod_clone_helper +
  //   emit_clone_deep_copy_fixup in solidity_convert_constructor.cpp
  //   for the semantics.
  if (
    callee_expr_json.is_object() &&
    callee_expr_json.value("nodeType", "") == "Identifier" &&
    callee_expr_json.value("name", "") == "__ESOL_deep_copy")
  {
    if (
      !expr.contains("arguments") || !expr["arguments"].is_array() ||
      expr["arguments"].size() != 1)
    {
      log_error("__ESOL_deep_copy expects exactly one argument");
      return true;
    }

    const nlohmann::json &arg_json = expr["arguments"][0];
    exprt src_arg;
    if (get_expr(arg_json, arg_json["typeDescriptions"], src_arg))
      return true;

    // Extract contract name from the argument's type.  Contract
    // values flow as pointer-to-struct here, so check both the top
    // level type and, if it is a pointer, the pointee.
    std::string cname = src_arg.type().get("#sol_contract").as_string();
    if (cname.empty() && src_arg.type().is_pointer())
      cname = src_arg.type().subtype().get("#sol_contract").as_string();
    if (cname.empty())
    {
      log_error(
        "__ESOL_deep_copy: argument is not a contract instance "
        "(missing #sol_contract tag on type)");
      return true;
    }

    symbolt clone_sym;
    if (build_tod_clone_helper(cname, clone_sym))
      return true;

    side_effect_expr_function_callt clone_call;
    clone_call.function() = symbol_expr(clone_sym);
    clone_call.type() = to_code_type(clone_sym.type).return_type();
    clone_call.location() = clone_sym.location;
    clone_call.arguments().push_back(src_arg);
    new_expr = clone_call;
    return false;
  }

  // * __ESOL_nondet_state_forward intrinsic:
  //   drive a contract instance in place through a nondet sequence of
  //   its own public/external calls.  Lets user harnesses express
  //   "reach any reachable state S" before, e.g., a subsequent
  //   __ESOL_deep_copy.  User declares a per-type stub so solc
  //   accepts the call:
  //     function __ESOL_nondet_state_forward(C c) { }
  //   ESBMC ignores the stub body and lowers the call to
  //   `_ESBMC_state_forward_<C>(arg)`.
  if (
    callee_expr_json.is_object() &&
    callee_expr_json.value("nodeType", "") == "Identifier" &&
    callee_expr_json.value("name", "") == "__ESOL_nondet_state_forward")
  {
    if (
      !expr.contains("arguments") || !expr["arguments"].is_array() ||
      expr["arguments"].size() != 1)
    {
      log_error("__ESOL_nondet_state_forward expects exactly one argument");
      return true;
    }

    const nlohmann::json &arg_json = expr["arguments"][0];
    exprt src_arg;
    if (get_expr(arg_json, arg_json["typeDescriptions"], src_arg))
      return true;

    std::string cname = src_arg.type().get("#sol_contract").as_string();
    if (cname.empty() && src_arg.type().is_pointer())
      cname = src_arg.type().subtype().get("#sol_contract").as_string();
    if (cname.empty())
    {
      log_error(
        "__ESOL_nondet_state_forward: argument is not a contract "
        "instance (missing #sol_contract tag on type)");
      return true;
    }

    symbolt fwd_sym;
    if (build_esol_state_forward_helper(cname, fwd_sym))
      return true;

    side_effect_expr_function_callt fwd_call;
    fwd_call.function() = symbol_expr(fwd_sym);
    fwd_call.type() = to_code_type(fwd_sym.type).return_type();
    fwd_call.location() = fwd_sym.location;
    fwd_call.arguments().push_back(src_arg);
    new_expr = fwd_call;
    return false;
  }

  // * __ESBMC_nondet_* family intrinsics:
  //   Return a fresh nondet value of the call's declared return type.
  //   Closes the "MODELING-2" gap: instrumenters that need a fresh
  //   nondet at a specific point (e.g. for self-composition oracles
  //   like TD-guard manipulability) cannot inject one through a
  //   state variable (state vars start at their post-constructor
  //   default in --contract mode, not havoc'd) nor through a new
  //   function parameter (that would break every internal caller).
  //
  //   User declares an empty stub so solc accepts the call:
  //     function __ESBMC_nondet_uint() internal returns (uint256) {}
  //     function __ESBMC_nondet_bool() internal returns (bool) {}
  //     function __ESBMC_nondet_address() internal returns (address) {}
  //     // any return type works — the name prefix is the only trigger
  //   ESBMC ignores the stub body and lowers the call directly to a
  //   side-effect-nondet expression of the AST's return type.
  if (
    callee_expr_json.is_object() &&
    callee_expr_json.value("nodeType", "") == "Identifier")
  {
    const std::string &nm = callee_expr_json.value("name", "");
    static const std::string prefix = "__ESBMC_nondet_";
    if (nm.compare(0, prefix.size(), prefix) == 0 && nm.size() > prefix.size())
    {
      if (!expr.contains("typeDescriptions"))
      {
        log_error(
          "{}: FunctionCall node has no typeDescriptions — cannot "
          "resolve nondet return type",
          nm);
        return true;
      }
      typet ret_type;
      if (get_type_description(expr["typeDescriptions"], ret_type))
        return true;
      exprt nondet("sideeffect", ret_type);
      nondet.statement("nondet");
      locationt l;
      get_location_from_node(expr, l);
      nondet.location() = l;
      new_expr = nondet;
      return false;
    }
  }

  // * check if it's a low-level call
  if (SolidityGrammar::is_address_member_call(callee_expr_json))
  {
    log_debug("solidity", "\t\t@@@ got address member call");
    if (get_expr(callee_expr_json, new_expr))
      return true;
    return false;
  }

  // * delegate-shadow helper inlining
  // If we're currently inlining a target contract's body and this call
  // resolves to a FunctionDefinition inside that same target contract,
  // inline the helper body instead of emitting a `(Target*)this` call.
  // The cast-based call silently depends on struct layout coincidences
  // between caller and target and is unsound for proxies with different
  // field order.
  if (!delegate_shadow_target_cname.empty())
  {
    int ref_id = -1;
    if (
      callee_expr_json.is_object() &&
      callee_expr_json.contains("referencedDeclaration") &&
      callee_expr_json["referencedDeclaration"].is_number_integer())
      ref_id = callee_expr_json["referencedDeclaration"].get<int>();
    if (ref_id > 0)
    {
      const nlohmann::json &fdecl = find_decl_ref(ref_id);
      if (
        !fdecl.empty() && !fdecl.is_null() &&
        fdecl.value("nodeType", "") == "FunctionDefinition" &&
        fdecl.contains("scope"))
      {
        // Only inline if the helper lives in the target contract we're
        // currently shadowing. External/library functions still go
        // through the normal path.
        const nlohmann::json &owner =
          find_node_by_id(src_ast_json, fdecl["scope"].get<int>());
        if (
          !owner.empty() && !owner.is_null() &&
          owner.value("nodeType", "") == "ContractDefinition" &&
          owner.value("name", "") == delegate_shadow_target_cname)
        {
          if (!try_inline_delegate_shadow_helper_call(expr, fdecl, new_expr))
            return false;
          // Fall through on failure — normal call path will take over.
        }
      }
    }
  }

  // * using-for free-function binding:  x.f(args)  ==>  f(x, args)
  //   Solidity's `using { f } for S` lets a free function be called as
  //   a member on the bound type. Rewrite into a plain call so the
  //   normal call path handles it.
  if (
    callee_expr_json.is_object() &&
    callee_expr_json.value("nodeType", "") == "MemberAccess" &&
    callee_expr_json.contains("referencedDeclaration") &&
    callee_expr_json["referencedDeclaration"].is_number_integer())
  {
    int ref_id = callee_expr_json["referencedDeclaration"].get<int>();
    const nlohmann::json &fdecl = find_decl_ref(ref_id);
    if (
      !fdecl.empty() && !fdecl.is_null() &&
      fdecl.value("nodeType", "") == "FunctionDefinition" &&
      fdecl.value("kind", "") == "freeFunction" &&
      callee_expr_json.contains("expression"))
    {
      nlohmann::json rewritten = expr;
      nlohmann::json new_callee;
      new_callee["nodeType"] = "Identifier";
      new_callee["name"] = fdecl.value("name", "");
      new_callee["referencedDeclaration"] = ref_id;
      if (callee_expr_json.contains("src"))
        new_callee["src"] = callee_expr_json["src"];
      if (callee_expr_json.contains("typeDescriptions"))
        new_callee["typeDescriptions"] = callee_expr_json["typeDescriptions"];
      rewritten["expression"] = new_callee;
      nlohmann::json new_args = nlohmann::json::array();
      new_args.push_back(callee_expr_json["expression"]);
      if (rewritten.contains("arguments") && rewritten["arguments"].is_array())
        for (const auto &a : rewritten["arguments"])
          new_args.push_back(a);
      rewritten["arguments"] = new_args;
      return get_call_expr(rewritten, literal_type, new_expr);
    }
  }

  // * check if it's a solidity built-in function
  if (
    !get_esbmc_builtin_ref(callee_expr_json, new_expr) ||
    !get_sol_builtin_ref(expr, new_expr))
  {
    log_debug("solidity", "\t\t@@@ got builtin function call");
    if (new_expr.id() == "typecast")
    {
      // assume it's a wrap/unwrap
      exprt args;
      const nlohmann::json &arg0 = expr["arguments"][0];
      if (get_expr(arg0, arg0["typeDescriptions"], args))
        return true;
      new_expr.op0() = args;
      return false;
    }

    if (new_expr.id() == "sideeffect")
    {
      // mapping(K=>V)[] push/pop: already a complete assign expression
      if (new_expr.statement() == "assign")
        return false;

      std::string func_id = new_expr.op0().identifier().as_string();
      if (
        func_id == "c:@F@_ESBMC_array_push" ||
        func_id == "c:@F@_ESBMC_array_push_uint256")
      {
        // signed short _tmpzero#5 = 0;
        // this->data1 = _ESBMC_array_push((void *)this->data1, (void *)&_tmpzero#5, 2);
        // For `_ESBMC_array_push_uint256` the emission shape is the
        // same (call returning a new data pointer that must land back
        // in the base slot); the only difference is the push arg list,
        // already built correctly in solidity_convert_ref.cpp.
        exprt base;
        if (get_expr(callee_expr_json["expression"], base))
          return true;

        typet base_t;
        if (get_type_description(
              callee_expr_json["expression"]["typeDescriptions"], base_t))
          return true;

        exprt tmp = side_effect_exprt("assign", base_t);
        convert_type_expr(ns, new_expr, base_t, expr);
        tmp.copy_to_operands(base, new_expr);
        new_expr = tmp;
        return false;
      }
      if (
        func_id == "c:@F@_ESBMC_array_pop" ||
        func_id == "c:@F@_ESBMC_array_length")
        return false;
      if (func_id.compare(0, 11, "c:@F@bytes_") == 0)
        return false;
      if (func_id == "c:@F@string_concat")
        return false;
    }
    if (new_expr.is_member() && new_expr.component_name() == "length")
      return false;

    std::string sol_name = new_expr.type().get("#sol_name").as_string();
    if (sol_name == "revert")
    {
      // EVM revert with state-rollback semantics.  Try to lower as
      //   { *this = _sol_save_this; return [nondet]; }
      // (see build_revert_rollback_block).  When the function context
      // does not support rollback (constructor, library, tuple return,
      // etc.), fall through to the legacy `__ESBMC_assume(false)`
      // lowering — sound but lossy.
      exprt rollback;
      if (!build_revert_rollback_block(nullptr, rollback))
      {
        new_expr = rollback;
        return false;
      }
      call.function() = new_expr;
      call.type() = to_code_type(new_expr.type()).return_type();
      call.arguments().resize(1);
      call.arguments().at(0) = false_exprt();
    }
    else if (sol_name == "require")
    {
      // Special case: require
      // __ESBMC_assume only handle one param.
      // drop the potential second param.
      exprt single_arg;
      if (get_expr(
            expr["arguments"].at(0),
            expr["arguments"].at(0)["typeDescriptions"],
            single_arg))
        return true;
      // EVM revert with state-rollback semantics.  Try to lower as
      //   if (!cond) { *this = _sol_save_this; return [nondet]; }
      // (see build_revert_rollback_block).  Falls back to the legacy
      // `__ESBMC_assume(cond)` lowering if rollback is not applicable.
      exprt rollback;
      if (!build_revert_rollback_block(&single_arg, rollback))
      {
        new_expr = rollback;
        return false;
      }
      call.function() = new_expr;
      call.type() = to_code_type(new_expr.type()).return_type();
      call.arguments().resize(1);
      call.arguments().at(0) = single_arg;
    }
    else
    {
      // other solidity built-in functions
      std::string func_id_str = new_expr.identifier().as_string();

      // [APPROX: UNDER] ecrecover short-circuit.
      // The C model `ecrecover(uint256 hash, uint v, uint256 r,
      // uint256 s)` expects four scalar uint256 arguments, but the
      // Solidity-level arguments (`bytes32 h, uint8 v, bytes32 r,
      // bytes32 s`) are lowered as BytesStatic structs. Feeding the
      // structs verbatim crashes symex with a type mismatch on the
      // first argument. The crypto model documents ecrecover as a
      // deterministic function of `hash` only and the README already
      // classifies it as a nondet abstraction; emit a plain nondet
      // of the declared return type and skip the library call.
      if (func_id_str == "c:@F@ecrecover")
      {
        locationt ecr_loc;
        get_start_location_from_stmt(expr, ecr_loc);
        log_warning(
          "[approx] ecrecover at {}:{}: (v,r,s) ignored, returns "
          "nondet address (BytesStatic args bypass the crypto model; "
          "signature forgery is not modelled)",
          ecr_loc.get_file().c_str(),
          ecr_loc.get_line().c_str());
        typet ret_type = to_code_type(new_expr.type()).return_type();
        exprt nondet = exprt("sideeffect", ret_type);
        nondet.statement("nondet");
        new_expr = nondet;
        return false;
      }

      // B3 — selfdestruct(address payable to) fund drain.  The legacy
      // C model `void selfdestruct() { exit(0); }` discards the
      // recipient argument and prunes the path entirely, so no
      // observable balance transfer happens and post-destruct
      // dispatcher iterations are unreachable.  Replace the call with
      // an inline emission that:
      //   1. credits the recipient via `_ESBMC_eoa_credit(to,
      //      this->$balance)` — works precisely for EOA recipients;
      //      under-approximates for tracked-contract recipients
      //      (their `$balance` field stays unchanged) — sound, will
      //      be tightened in B3 v2 with a per-contract dispatch
      //      mirroring transfer/send;
      //   2. zeroes `this->$balance`;
      //   3. emits `return` so the rest of the calling Solidity
      //      function is skipped (matches "current call frame
      //      ends" semantics) but the dispatcher's outer loop can
      //      proceed to the next iteration with the now-empty
      //      contract balance visible — giving meaningful coverage
      //      to post-destruct properties (still over-approximates
      //      vs real EVM where the destroyed contract's code is
      //      wiped; a `$destroyed` gate is a B3 follow-up).
      // Falls back to the legacy `selfdestruct()` call when the
      // function context is not available (no `#this` symbol) or
      // the AST has no recipient argument.
      if (func_id_str == "c:@F@selfdestruct")
      {
        std::string this_id = current_functionId + "#this";
        const symbolt *this_sym =
          current_functionId.empty() ? nullptr : context.find_symbol(this_id);
        if (
          this_sym != nullptr && expr.contains("arguments") &&
          expr["arguments"].is_array() && !expr["arguments"].empty())
        {
          exprt recipient;
          if (get_expr(
                expr["arguments"][0],
                expr["arguments"][0]["typeDescriptions"],
                recipient))
            return true;

          typet val_t = unsignedbv_typet(256);
          typet addr_t = unsignedbv_typet(160);
          if (recipient.type() != addr_t)
            solidity_gen_typecast(ns, recipient, addr_t);

          exprt this_expr = symbol_expr(*this_sym);
          exprt this_balance = member_exprt(this_expr, "$balance", val_t);

          // _ESBMC_eoa_credit(to, this->$balance);
          side_effect_expr_function_callt credit_call;
          get_library_function_call_no_args(
            "_ESBMC_eoa_credit",
            "c:@F@_ESBMC_eoa_credit",
            empty_typet(),
            locationt(),
            credit_call);
          credit_call.arguments().push_back(recipient);
          credit_call.arguments().push_back(this_balance);
          convert_expression_to_code(credit_call);

          // this->$balance = 0;
          exprt zero_v = from_integer(0, val_t);
          code_assignt zero_balance(this_balance, zero_v);

          // return; (selfdestruct is void)
          code_returnt return_stmt;

          code_blockt block;
          block.copy_to_operands(credit_call);
          block.copy_to_operands(zero_balance);
          block.copy_to_operands(return_stmt);
          new_expr = block;
          return false;
        }
        // Legacy fallback: call `selfdestruct()` which is `exit(0)`.
      }

      bool is_abi_func = func_id_str.find("c:@F@abi_") == 0;
      // [APPROX: OVER] crypto + abi identity/nondet abstraction.
      // keccak256/sha256/ripemd160 and abi.encode* are modelled as a
      // single-uint256 identity hash + nondet decoder. See
      // src/c2goto/library/solidity/solidity_crypto.c and solidity_abi.c.
      // Soundness: sound for equality-based reasoning (same input →
      //   same hash). False positives possible when a property depends
      //   on *specific* hash bits (e.g. preimage resistance).
      // Completeness: incomplete — cannot refute preimage-resistance or
      //   collision properties; abi.decode may admit values that never
      //   round-trip from a real encoder. Normally they work
      // because inner abi.encode(...) already lowered to uint256, but
      // passing a raw BytesDynamic (e.g. keccak256(bytes(calldata)))
      // crashes in get_library_function_call with a type mismatch.
      // For that case, fall back to a nondet uint256 result. We keep
      // non-bytes argument paths on the regular library call so inputs
      // like keccak256(abi.encode(42)) still use the identity chain.
      bool is_hash_func = func_id_str == "c:@F@keccak256" ||
                          func_id_str == "c:@F@sha256" ||
                          func_id_str == "c:@F@ripemd160";
      bool hash_needs_nondet = false;
      if (
        is_hash_func && expr.contains("arguments") &&
        expr["arguments"].is_array() && !expr["arguments"].empty())
      {
        const auto &arg0 = expr["arguments"][0];
        const std::string tid =
          arg0.value("typeDescriptions", nlohmann::json::object())
            .value("typeIdentifier", "");
        const std::string node = arg0.value("nodeType", "");
        // Raw source-level bytes values (Identifier / MemberAccess /
        // IndexAccess) are BytesDynamic / BytesStatic structs at
        // runtime and can't flow through the keccak256(uint256) C
        // model.
        //
        // FunctionCall arguments are trickier: an inner abi.encode*
        // call lowers to the uint256 identity in solidity_abi.c (so
        // its result IS a 256-bit scalar despite the Solidity-level
        // bytes type) and must keep the precise library path to
        // preserve the equality semantics of `keccak256(abi.encode(x))
        // == keccak256(abi.encode(x))`. A non-builtin user function
        // declared `returns (bytes memory)`, however, really does
        // return BytesDynamic at runtime (especially after the
        // return-statement coercion that maps scalar→bytes via
        // llc_nondet_bytes), and feeding that struct to keccak256
        // would crash symex on a struct/scalar mismatch — fall back
        // to the nondet uint256 result for that case.
        bool is_func_call = (node == "FunctionCall") &&
                            (arg0.value("kind", "") != "typeConversion");
        bool is_builtin_abi_call = false;
        if (is_func_call && arg0.contains("expression"))
        {
          const auto &callee = arg0["expression"];
          std::string callee_name;
          if (callee.contains("memberName"))
            callee_name = callee["memberName"].get<std::string>();
          else if (callee.contains("name"))
            callee_name = callee["name"].get<std::string>();
          static const std::set<std::string> abi_calls = {
            "encode",
            "encodePacked",
            "encodeWithSelector",
            "encodeWithSignature",
            "encodeCall",
            "decode"};
          if (abi_calls.count(callee_name) > 0)
            is_builtin_abi_call = true;
        }
        if (
          !is_builtin_abi_call && (tid.compare(0, 8, "t_bytes_") == 0 ||
                                   (tid.compare(0, 7, "t_bytes") == 0 &&
                                    tid.size() > 7 && std::isdigit(tid[7]))))
          hash_needs_nondet = true;
      }
      if (is_abi_func || hash_needs_nondet)
      {
        // [F1 closure, ledger #3] Replace the unsound multiplicative
        // _ESBMC_abi_fold (`acc * 0x100000001b3 + next`) with bit-vector
        // concat into a wide-BV-indexed table.  The legacy fold was
        // non-injective under SMT — multiplication mod 2^256 is not a
        // permutation, so the solver finds (a,b) ≠ (c,d) with equal
        // fold output and breaks
        //   `keccak256(abi.encode(a,b)) != keccak256(abi.encode(c,d))`
        // for distinct args.
        //
        // New encoding: each arg's pre-cast bit-width is captured;
        // args are concatenated in declaration order (arg[0] in the
        // high bits, arg[N-1] in the low bits) into a wide BV; the
        // total width is rounded UP to the smallest enclosing bucket
        // W ∈ {256, 512, 1024, 2048}; the bucketed table for the
        // matching hash family is then indexed by the W-bit concat.
        // The SMT array axiom gives same-key-same-result for free;
        // per-callsite distinctness assumes (emitted below) cover the
        // distinct-key-distinct-result direction (injectivity).

        std::vector<std::pair<unsigned, exprt>> fold_args;

        std::function<bool(const nlohmann::json &)> fold_arg =
          [&](const nlohmann::json &a) -> bool {
          std::string tid =
            a.value("typeDescriptions", nlohmann::json::object())
              .value("typeIdentifier", "");
          // Skip type expressions and function declarations.
          if (
            tid.compare(0, 7, "t_type$") == 0 ||
            tid.compare(0, 23, "t_function_declaration_") == 0)
            return false;
          // Tuple arguments — descend into value-tuples; skip type-only.
          if (tid.compare(0, 8, "t_tuple$") == 0)
          {
            if (tid.find("t_type$") != std::string::npos)
              return false;
            if (a.contains("components"))
            {
              for (const auto &c : a["components"])
                if (!c.is_null() && fold_arg(c))
                  return true;
            }
            return false;
          }
          // Skip fixed bytesN — same exclusion as legacy fold;
          // BytesStatic.data[N] descent is deferred.
          if (
            tid.compare(0, 7, "t_bytes") == 0 && tid.size() > 7 &&
            std::isdigit(tid[7]))
            return false;

          exprt single_arg;
          if (get_expr(a, a["typeDescriptions"], single_arg))
            return true;

          // Dynamic bytes: fold by .length (closes B7-C from T2.1).
          if (tid.compare(0, 8, "t_bytes_") == 0)
            single_arg = member_exprt(single_arg, "length", size_type());

          // Capture original bit-width before any cast.
          unsigned bw = 256;
          if (
            single_arg.type().id() == "unsignedbv" ||
            single_arg.type().id() == "signedbv")
          {
            bw = atoi(single_arg.type().width().c_str());
            if (bw == 0)
              bw = 256;
          }
          else if (single_arg.type().id() == "bool")
            bw = 8;

          // Normalise to bw-bit unsignedbv for uniform concat.
          typet single_t = unsignedbv_typet(bw);
          if (single_arg.type() != single_t)
            solidity_gen_typecast(ns, single_arg, single_t);

          fold_args.emplace_back(bw, single_arg);
          return false;
        };

        for (const auto &arg : expr["arguments"])
        {
          if (fold_arg(arg))
            return true;
        }

        // Compute total width and pick bucket.
        unsigned total_W = 0;
        for (const auto &p : fold_args)
          total_W += p.first;

        unsigned W = 0; // 0 = nondet fallback
        if (!fold_args.empty())
        {
          if (total_W <= 256)
            W = 256;
          else if (total_W <= 512)
            W = 512;
          else if (total_W <= 1024)
            W = 1024;
          else if (total_W <= 2048)
            W = 2048;
          else
            log_warning(
              "[approx] keccak/abi fold total arg-width {} exceeds "
              "2048 bits; falling back to nondet result",
              total_W);
        }

        // Determine hash family + result type.
        std::string hash_name;
        bool is_ripemd_call = (func_id_str == "c:@F@ripemd160");
        typet res_t = unsignedbv_typet(256);
        if (is_abi_func)
          hash_name = "abi";
        else if (func_id_str == "c:@F@keccak256")
          hash_name = "keccak";
        else if (func_id_str == "c:@F@sha256")
          hash_name = "sha256";
        else if (is_ripemd_call)
        {
          hash_name = "ripemd160";
          res_t = unsignedbv_typet(160);
        }
        else
          hash_name = "abi";

        exprt result_expr;

        if (W == 0)
        {
          exprt nondet_arg = exprt("sideeffect");
          nondet_arg.type() = res_t;
          nondet_arg.statement("nondet");
          result_expr = nondet_arg;
        }
        else
        {
          typet wide_t = unsignedbv_typet(W);

          // Build chained shift+or concat: arg[0] high, arg[N-1] low.
          exprt concat_W = fold_args[0].second;
          if (concat_W.type() != wide_t)
            solidity_gen_typecast(ns, concat_W, wide_t);
          for (size_t i = 1; i < fold_args.size(); ++i)
          {
            exprt shift_amt = from_integer(fold_args[i].first, wide_t);
            exprt shifted("shl", wide_t);
            shifted.copy_to_operands(concat_W, shift_amt);

            exprt next_W = fold_args[i].second;
            if (next_W.type() != wide_t)
              solidity_gen_typecast(ns, next_W, wide_t);

            exprt or_expr("bitor", wide_t);
            or_expr.copy_to_operands(shifted, next_W);
            concat_W = or_expr;
          }

          // Allocate fresh per-callsite globals so distinctness
          // assumes at later call sites can reference them across
          // function boundaries.
          unsigned n = ++hash_callsite_counter_;
          std::string key_name = "__esbmc_hash_key_" + hash_name + "_" +
                                 std::to_string(W) + "_" + std::to_string(n);
          std::string key_id = "c:@" + key_name;
          std::string res_name = "__esbmc_hash_result_" + hash_name + "_" +
                                 std::to_string(W) + "_" + std::to_string(n);
          std::string res_id = "c:@" + res_name;
          locationt cs_loc;

          if (context.find_symbol(key_id) == nullptr)
          {
            symbolt key_sym;
            get_default_symbol(
              key_sym, "C++", wide_t, key_name, key_id, cs_loc);
            key_sym.lvalue = true;
            key_sym.is_extern = false;
            key_sym.file_local = false;
            key_sym.static_lifetime = true;
            context.move_symbol_to_context(key_sym);
          }
          if (context.find_symbol(res_id) == nullptr)
          {
            symbolt res_sym;
            get_default_symbol(res_sym, "C++", res_t, res_name, res_id, cs_loc);
            res_sym.lvalue = true;
            res_sym.is_extern = false;
            res_sym.file_local = false;
            res_sym.static_lifetime = true;
            context.move_symbol_to_context(res_sym);
          }

          exprt key_lhs = symbol_exprt(key_id, wide_t);
          exprt res_lhs = symbol_exprt(res_id, res_t);

          // __esbmc_hash_key_<n> = concat_W;
          code_assignt assign_key(key_lhs, concat_W);
          move_to_front_block(assign_key);

          // Table-memoised hash with sentinel-0:
          //   - First call at a given key writes a fresh nondet (≠0)
          //     into table[key].  Subsequent calls (same syntactic
          //     site OR any other site at the same key) read the
          //     same value back via the SMT array axiom, giving
          //     `same key → same hash` for free.
          //   - The sentinel 0 distinguishes "uninitialised" from
          //     "previously written".  This carves out one specific
          //     hash value (0); since real keccak collisions to 0
          //     have probability ≈ 2^-256, the under-approximation
          //     is practically negligible.
          //   - Cross-site distinctness (`distinct keys → distinct
          //     hashes`) is enforced by the per-pair assume below;
          //     the SMT array axiom gives only consistency, not
          //     injectivity.
          std::string table_name =
            "_ESBMC_" + hash_name + "_table_" + std::to_string(W);
          std::string table_id = "c:@" + table_name;
          const symbolt *table_sym_ptr = context.find_symbol(table_id);
          if (table_sym_ptr == nullptr)
          {
            log_error(
              "F1 fold path: table symbol {} not found — sol_glue "
              "unlinked or annotation parser failed",
              table_id);
            return true;
          }
          exprt table_ref = symbol_expr(*table_sym_ptr);

          // __esbmc_hash_result_<n> = table[key];   (existing or 0)
          index_exprt existing_lookup(table_ref, key_lhs, res_t);
          code_assignt assign_existing(res_lhs, existing_lookup);
          move_to_front_block(assign_existing);

          // fresh = nondet_<res_t>(); on cache-miss path.
          exprt nondet_fresh = exprt("sideeffect");
          nondet_fresh.type() = res_t;
          nondet_fresh.statement("nondet");

          // __esbmc_hash_result_<n> =
          //   (existing != 0) ? existing : fresh;
          exprt zero = from_integer(0, res_t);
          equality_exprt is_zero(res_lhs, zero);
          if_exprt pick(is_zero, nondet_fresh, res_lhs);
          pick.type() = res_t;
          code_assignt assign_pick(res_lhs, pick);
          move_to_front_block(assign_pick);

          // __ESBMC_assume(__esbmc_hash_result_<n> != 0);
          //   Carves out the sentinel — preserves the
          //   `existing != 0 ↔ already memoised` invariant for any
          //   subsequent call at this key.
          binary_relation_exprt res_nz(res_lhs, "notequal", zero);
          side_effect_expr_function_callt assume_nz;
          get_library_function_call_no_args(
            "__ESBMC_assume",
            "c:@F@__ESBMC_assume",
            empty_typet(),
            cs_loc,
            assume_nz);
          assume_nz.arguments().push_back(res_nz);
          convert_expression_to_code(assume_nz);
          move_to_front_block(assume_nz);

          // table[key] = __esbmc_hash_result_<n>;     (memoise)
          //   Idempotent if `existing != 0` (we wrote
          //   `res = existing` above); fresh write on first call.
          index_exprt store_lhs(table_ref, key_lhs, res_t);
          code_assignt store_back(store_lhs, res_lhs);
          move_to_front_block(store_back);

          // Per-callsite distinctness assume against prior matching
          // entries.  Encoded as `prior_key == this_key ||
          // prior_result != this_result`, the implication
          // `prior_key != this_key → prior_result != this_result`.
          // Memoisation already gives `same key → same hash` (both
          // sites read back the same table[key]), so only the
          // distinctness direction needs an explicit assume.
          for (const auto &prior : hash_callsites_)
          {
            if (prior.hash_name != hash_name)
              continue;
            if (prior.width_bucket != W)
              continue;

            exprt prior_key = symbol_exprt(
              prior.key_sym_id, unsignedbv_typet(prior.width_bucket));
            exprt this_key = symbol_exprt(key_id, wide_t);
            equality_exprt eq_keys(prior_key, this_key);

            exprt prior_res =
              symbol_exprt(prior.result_sym_id, prior.result_type);
            exprt this_res = symbol_exprt(res_id, res_t);
            binary_relation_exprt neq_res(prior_res, "notequal", this_res);

            exprt impl("or", bool_typet());
            impl.copy_to_operands(eq_keys, neq_res);

            side_effect_expr_function_callt assume_call;
            get_library_function_call_no_args(
              "__ESBMC_assume",
              "c:@F@__ESBMC_assume",
              empty_typet(),
              cs_loc,
              assume_call);
            assume_call.arguments().push_back(impl);
            convert_expression_to_code(assume_call);
            move_to_front_block(assume_call);
          }

          hash_callsites_.push_back(
            {hash_name, W, key_name, key_id, res_name, res_id, res_t});

          result_expr = res_lhs;
        }

        // Wrap hash result in bytes_static_from_uint for keccak/sha256
        // (Solidity returns bytes32; ripemd160 returns address; abi.*
        // returns raw uint256).
        if (is_hash_func && func_id_str != "c:@F@ripemd160")
        {
          side_effect_expr_function_callt pack_call;
          get_library_function_call_no_args(
            "bytes_static_from_uint",
            "c:@F@bytes_static_from_uint",
            byte_static_t,
            locationt(),
            pack_call);
          pack_call.arguments().push_back(result_expr);
          pack_call.arguments().push_back(from_integer(32, size_type()));
          new_expr = pack_call;
        }
        else
        {
          new_expr = result_expr;
        }
        return false;
      }
      else
      {
        if (get_library_function_call(
              new_expr, new_expr.type(), empty_json, expr, call))
          return true;
      }

      // Solidity's keccak256 / sha256 ALWAYS return bytes32, whereas
      // their C-model counterparts return uint256.  Without an explicit
      // pack, the call expression's type is uint256, so any Solidity
      // context that expects bytes32 (return of `returns (bytes32)`,
      // assignment to `bytes32`, comparison against bytes32) produces
      // a struct/scalar shape mismatch that crashes value-set / symex
      // at assignment time.  Wrap the call in bytes_static_from_uint
      // so the emitted expression has the correct BytesStatic shape
      // while keeping the identity-hash abstraction (same input
      // uint256 -> same bytes32 pack -> equality preserved; pack/unpack
      // is identity for len=32 so a subsequent `uint256(keccak(...))`
      // round-trips cleanly).  ripemd160 returns `address` in
      // Solidity, so its scalar result already matches and must not
      // be packed.  is_abi_func keeps the raw uint256 because abi.*
      // genuinely models `bytes memory` as the uint256 identity.
      //
      // NOTE: the pack routes through `bytes_static_from_uint` which
      // writes 32 bytes in a loop — tests that exercise keccak/sha
      // results need `--unwind 32` (or more) to unroll it; smaller
      // unwind bounds truncate the pack and leave the result
      // unconstrained.
      if (is_hash_func && func_id_str != "c:@F@ripemd160")
      {
        side_effect_expr_function_callt pack_call;
        get_library_function_call_no_args(
          "bytes_static_from_uint",
          "c:@F@bytes_static_from_uint",
          byte_static_t,
          locationt(),
          pack_call);
        pack_call.arguments().push_back(call);
        pack_call.arguments().push_back(from_integer(32, size_type()));
        new_expr = pack_call;
        return false;
      }
    }

    new_expr = call;
    return false;
  }

  // * check if its a call-with-options
  if (
    !expr.contains("name") && callee_expr_json.contains("nodeType") &&
    callee_expr_json["nodeType"] == "FunctionCallOptions")
  {
    if (get_expr(callee_expr_json, new_expr))
      return true;
    return false;
  }

  // * check if it's a member access call
  if (
    callee_expr_json.contains("nodeType") &&
    callee_expr_json["nodeType"] == "MemberAccess")
  {
    // super.method() — bypass override map and call the base function directly
    if (
      callee_expr_json.contains("expression") &&
      callee_expr_json["expression"].contains("name") &&
      callee_expr_json["expression"]["name"] == "super")
    {
      return get_super_function_call(callee_expr_json, expr, new_expr);
    }

    if (get_expr(callee_expr_json, literal_type, new_expr))
      return true;
    return false;
  }

  // wrap it in an ImplicitCastExpr to perform conversion of FunctionToPointerDecay
  nlohmann::json implicit_cast_expr =
    make_implicit_cast_expr(callee_expr_json, "FunctionToPointerDecay");
  exprt callee_expr;
  if (get_expr(implicit_cast_expr, callee_expr))
    return true;

  if (
    callee_expr.is_code() && callee_expr.statement() == "function_call" &&
    callee_expr.op1().name() == "_ESBMC_Nondet_Extcall")
  {
    new_expr = callee_expr;
    return false;
  }

  // * check if it's a struct call
  if (expr.contains("kind") && expr["kind"] == "structConstructorCall")
  {
    log_debug("solidity", "\t\t@@@ got struct constructor call");
    // e.g. Book book = Book('Learn Java', 'TP', 1);
    if (callee_expr.type().id() != irept::id_struct)
    {
      log_error("expected struct type for struct constructor call");
      return true;
    }

    typet t = callee_expr.type();
    exprt inits = gen_zero(t);

    int ref_id = callee_expr_json["referencedDeclaration"].get<int>();
    const nlohmann::json &struct_ref = find_decl_ref(ref_id);
    if (struct_ref == empty_json)
    {
      log_error("cannot find struct definition for ref_id {}", ref_id);
      return true;
    }

    const nlohmann::json members = struct_ref["members"];
    const nlohmann::json args = expr["arguments"];

    // popluate components
    for (size_t i = 0; i < inits.operands().size() && i < args.size(); i++)
    {
      exprt init;
      if (get_expr(args.at(i), members.at(i)["typeDescriptions"], init))
        return true;

      const struct_union_typet::componentt *c =
        &to_struct_type(t).components().at(i);
      typet elem_type = c->type();

      solidity_gen_typecast(ns, init, elem_type);
      inits.operands().at(i) = init;
    }

    new_expr = inits;
    return false;
  }

  // function call expr
  // [APPROX: UNDER] Chained indirect call through a function pointer,
  // e.g. `x()()()()` where `x` is `function() returns (function() ...)`.
  // The inner FunctionCall returns a sideeffect nondet with a pointer
  // (FUNC_PTR) type — not a code type — so we can't take its return
  // type via to_code_type. Fall back to the outer call's declared
  // typeDescriptions and emit a nondet of that type. Same rationale as
  // the fallback below for callees without a referencedDeclaration.
  if (!callee_expr.type().is_code())
  {
    log_debug(
      "solidity",
      "\t\t@@@ got chained indirect call (callee type is not code), "
      "synthesizing nondet");
    typet ret_type = empty_typet();
    if (expr.contains("typeDescriptions"))
      get_type_description(expr["typeDescriptions"], ret_type);
    exprt nondet = exprt("sideeffect", ret_type);
    nondet.statement("nondet");
    new_expr = nondet;
    return false;
  }
  typet type = to_code_type(callee_expr.type()).return_type();

  // [APPROX: UNDER] Indirect callees without a referencedDeclaration
  // (ternary selecting between two function references, IndexAccess on
  // a function-pointer array/mapping, etc.) cannot be resolved to a
  // single target. Model the call result as a nondet value of the
  // declared return type. This NEVER executes the real callee body, so
  // bugs inside functions reachable only through an indirect call are
  // invisible. Side effects of the call on contract state are also
  // lost — UNDER-approximate for state-modifying indirect calls.
  if (
    !callee_expr_json.contains("referencedDeclaration") ||
    callee_expr_json["referencedDeclaration"].is_null())
  {
    log_debug(
      "solidity",
      "\t\t@@@ got indirect call with no referencedDeclaration "
      "(nodeType={}), synthesizing nondet",
      callee_expr_json.value("nodeType", "?"));
    exprt nondet = exprt("sideeffect", type);
    nondet.statement("nondet");
    new_expr = nondet;
    return false;
  }
  const auto &decl_ref =
    find_decl_ref(callee_expr_json["referencedDeclaration"].get<int>());
  std::string node_type = decl_ref["nodeType"].get<std::string>();

  // * check if it's a event, error function call
  if (node_type == "EventDefinition" || node_type == "ErrorDefinition")
  {
    log_debug("solidity", "\t\t@@@ got event/error function call");
    assert(expr.contains("arguments"));
    // Named-argument errors (`revert E({a: 1, b: 2})`) must be
    // reordered to parameter declaration order before being passed
    // to get_library_function_call, which iterates parameters and
    // arguments positionally. Otherwise the types would line up by
    // call-site order and produce "type mismatch" during symex.
    auto it = expr.find("names");
    if (it != expr.end() && it->is_array() && !it->empty())
    {
      nlohmann::json clean_expr =
        reorder_arguments(expr, src_ast_json, callee_expr_json);
      if (get_library_function_call(
            callee_expr, type, decl_ref, clean_expr, call))
        return true;
    }
    else
    {
      if (get_library_function_call(callee_expr, type, decl_ref, expr, call))
        return true;
    }
    new_expr = call;
    return false;
  }

  // * indirect call through a function-typed parameter / field
  //   (FunctionTypeName, internal or external). We cannot resolve
  //   the target statically, so synthesize a nondet value of the
  //   declared return type. This makes the program verifiable
  //   (no crash) but loses the link between input and output —
  //   counterexamples involving specific return values may be
  //   spurious. See docs/Solidity_KnownLimitations.md.
  if (
    node_type == "VariableDeclaration" && decl_ref.contains("typeName") &&
    decl_ref["typeName"].is_object() &&
    decl_ref["typeName"].value("nodeType", "") == "FunctionTypeName")
  {
    log_debug(
      "solidity", "\t\t@@@ got function-pointer indirect call (nondet)");

    // Derive the return type from the FunctionTypeName AST. If the
    // function has no return value, emit a nondet of empty type so
    // the call statement is well-formed. For multiple returns,
    // fall back to the first one (docs examples use 0 or 1 return).
    typet ret_type = empty_typet();
    const auto &ftn = decl_ref["typeName"];
    if (
      ftn.contains("returnParameterTypes") &&
      ftn["returnParameterTypes"].contains("parameters") &&
      ftn["returnParameterTypes"]["parameters"].is_array() &&
      !ftn["returnParameterTypes"]["parameters"].empty())
    {
      const auto &ret_param = ftn["returnParameterTypes"]["parameters"][0];
      if (ret_param.contains("typeDescriptions"))
      {
        if (get_type_description(ret_param["typeDescriptions"], ret_type))
          return true;
      }
    }

    exprt nondet = exprt("sideeffect", ret_type);
    nondet.statement("nondet");
    new_expr = nondet;
    return false;
  }

  // * check if it's the function inside library node
  // Library functions have no this-pointer parameter, so use
  // get_library_function_call instead of get_non_library_function_call.
  if (SolidityGrammar::is_sol_library_function(
        callee_expr_json["referencedDeclaration"].get<int>()))
  {
    log_debug("solidity", "\t\t@@@ got library-internal function call");
    assert(expr.contains("arguments"));
    if (get_library_function_call(callee_expr, type, decl_ref, expr, call))
      return true;
    new_expr = call;
    return false;
  }

  // Foundry forge-std assertion lowering (F1.b): assertEq/assertTrue/... are
  // Test-base helpers (real Foundry tests never use native assert). Recognize
  // by name and lower to a native assert of the comparison so a wrong test's
  // expectation actually surfaces. Placed before the normal internal-call path
  // that would otherwise convert the (no-op) stub body.
  {
    const std::string callee_name = callee_expr_json.value("name", "");
    locationt al;
    get_location_from_node(expr, al);
    bool ah = false;
    if (handle_forge_std_assert(callee_name, expr, al, new_expr, ah))
      return true;
    if (ah)
      return false;
  }

  log_debug("solidity", "\t\t@@@ got normal function call");
  // * we had ruled out all the special cases
  // * we now confirm it is called by another contract inside current contract
  // * func() ==> current_func_this.func(&current_func_this);

  // * check if the function call has named arguments
  // e.g. func({a: 1, b: 2});
  // reorder the arguments based on the parameter order
  auto it = expr.find("names");
  if (it != expr.end() && it->is_array() && !it->empty())
  {
    nlohmann::json clean_expr =
      reorder_arguments(expr, src_ast_json, callee_expr_json);
    if (get_non_library_function_call(decl_ref, clean_expr, call))
      return true;

    new_expr = call;
    return false;
  }

  if (get_non_library_function_call(decl_ref, expr, call))
    return true;

  new_expr = call;

  return false;
}

bool solidity_convertert::get_contract_member_call_expr(
  const nlohmann::json &expr,
  const nlohmann::json &literal_type,
  exprt &new_expr)
{
  locationt location;
  get_start_location_from_stmt(expr, location);

  std::string current_contractName;
  get_current_contract_name(expr, current_contractName);

  // ContractMemberCall
  // - x.setAddress();
  // - x.address();
  // - x.val(); ==> property
  // The later one is quite special, as in Solidity variables behave like functions from the perspective of other contracts.
  // e.g. b._addr is not an address, but a function that returns an address.

  // find the parent json which contains arguments
  const auto &func_call_json = find_last_parent(src_ast_json["nodes"], expr);
  assert(!func_call_json.empty());

  // `c.f{value: v}(args)` parses as
  //     FunctionCall( FunctionCallOptions( MemberAccess(c.f) ), args )
  // so `func_call_json` above -- the MemberAccess's IMMEDIATE parent -- is the
  // FunctionCallOptions node. That node carries `names`/`options` but NO
  // `arguments`; the actual arguments hang off the enclosing FunctionCall.
  // get_non_library_function_call() binds arguments only when its `caller`
  // node contains "arguments", so handing it the options node silently falls
  // back to assign_param_nondet() and the callee then runs with completely
  // unconstrained parameters (measured: after `c.one{value:5}(7)`, all three
  // of `x == 7`, `x != 7` and `x != 5` were refutable in the callee).
  // Resolve the argument-carrying node separately and use it ONLY for
  // argument binding: `func_call_json` itself must stay the immediate parent,
  // because the function-reference identity test below compares
  // `func_call_json["expression"]["id"]` against this MemberAccess -- stepping
  // it over the options node makes that test fail, the call is lowered to an
  // opaque function pointer and the callee body is never executed at all
  // (measured: `assert(false)` inside the callee became unreachable).
  // Same skip as get_bound_low_level_call().
  const nlohmann::json *_args_call_p = &func_call_json;
  if (
    func_call_json.contains("nodeType") &&
    func_call_json["nodeType"] == "FunctionCallOptions")
    _args_call_p = &find_last_parent(src_ast_json["nodes"], func_call_json);
  const nlohmann::json &args_call_json = *_args_call_p;

  auto callee_expr_json = expr;
  const nlohmann::json &caller_expr_json = callee_expr_json["expression"];
  assert(callee_expr_json.contains("referencedDeclaration"));

  // The caller expression is normally an Identifier with referencedDeclaration
  // (e.g. `creator.method()`). But it can also be an inline type cast like
  // `TokenCreator(address(creator)).method()`, which is a FunctionCall with
  // kind "typeConversion". In that case, unwrap the cast chain to find the
  // innermost contract variable reference. Also unwrap any surrounding
  // parenthesised TupleExpressions (`(new C()).x()`).
  nlohmann::json resolved_caller = caller_expr_json;
  bool progress = true;
  while (progress)
  {
    progress = false;
    if (
      resolved_caller.contains("nodeType") &&
      resolved_caller["nodeType"] == "FunctionCall" &&
      resolved_caller.value("kind", "") == "typeConversion" &&
      resolved_caller.contains("arguments") &&
      resolved_caller["arguments"].size() == 1)
    {
      resolved_caller = resolved_caller["arguments"][0];
      progress = true;
    }
    else if (
      resolved_caller.contains("nodeType") &&
      resolved_caller["nodeType"] == "TupleExpression" &&
      resolved_caller.contains("components") &&
      resolved_caller["components"].is_array() &&
      resolved_caller["components"].size() == 1)
    {
      resolved_caller = resolved_caller["components"][0];
      progress = true;
    }
  }

  // Inline `new C()` as the call target: e.g. `(new C()).x()` or
  // `(new Other()).addTwo`. The resolved_caller is a FunctionCall
  // whose `expression` is a NewExpression — there is no persistent
  // contract variable to bind the call to. Model the invocation as
  // a nondet value of the callee's declared return type. We lose the
  // semantic link to C's state vars / method body, but the program
  // becomes verifiable and downstream reads stay sound (nondet
  // over-approximation).
  if (
    resolved_caller.contains("nodeType") &&
    resolved_caller["nodeType"] == "FunctionCall" &&
    resolved_caller.contains("expression") &&
    resolved_caller["expression"].is_object() &&
    resolved_caller["expression"].value("nodeType", "") == "NewExpression")
  {
    log_debug(
      "solidity",
      "\t\t@@@ got member call on inline `new C()`, synthesizing "
      "nondet return");
    int fn_ref = expr.value("referencedDeclaration", -1);
    typet ret_t;
    bool have_type = false;
    if (fn_ref > 0)
    {
      const nlohmann::json &fdecl = find_decl_ref(fn_ref);
      if (!fdecl.empty() && !fdecl.is_null())
      {
        std::string fnode = fdecl.value("nodeType", "");
        if (
          fnode == "FunctionDefinition" && fdecl.contains("returnParameters") &&
          fdecl["returnParameters"].contains("parameters") &&
          fdecl["returnParameters"]["parameters"].is_array() &&
          !fdecl["returnParameters"]["parameters"].empty())
        {
          const auto &rp = fdecl["returnParameters"]["parameters"][0];
          if (rp.contains("typeDescriptions"))
          {
            if (get_type_description(rp["typeDescriptions"], ret_t))
              return true;
            have_type = true;
          }
        }
        else if (
          fnode == "VariableDeclaration" && fdecl.contains("typeName") &&
          fdecl["typeName"].contains("typeDescriptions"))
        {
          // Public state var getter — the MemberAccess references
          // the VariableDeclaration; the getter returns the var's
          // declared type.
          if (get_type_description(
                fdecl["typeName"]["typeDescriptions"], ret_t))
            return true;
          have_type = true;
        }
      }
    }
    // Fall back to the member access's own typeDescriptions if we
    // could not recover the declared return type.
    if (!have_type && expr.contains("typeDescriptions"))
    {
      if (get_type_description(expr["typeDescriptions"], ret_t))
        return true;
      have_type = true;
    }
    if (!have_type)
      ret_t = empty_typet();
    exprt nondet = exprt("sideeffect", ret_t);
    nondet.statement("nondet");
    new_expr = nondet;
    return false;
  }

  // Recover the cast target contract/interface name when the caller is
  // an explicit type cast to a contract type (e.g.
  // `ERC721TokenReceiver(to).onERC721Received(...)` or
  // `ICallback(msg.sender).cb()`).  Hoisted so both the variable-backed
  // path and the no-variable address-cast path below can use it.
  std::string cast_target_cname;
  if (
    caller_expr_json.contains("nodeType") &&
    caller_expr_json["nodeType"] == "FunctionCall" &&
    caller_expr_json.value("kind", "") == "typeConversion" &&
    caller_expr_json.contains("typeDescriptions"))
  {
    const std::string ts =
      caller_expr_json["typeDescriptions"].value("typeString", "");
    // ts is like "contract ERC721TokenReceiver" or "contract IERC20"
    auto sp = ts.rfind(' ');
    if (sp != std::string::npos && ts.compare(0, 9, "contract ") == 0)
      cast_target_cname = ts.substr(sp + 1);
  }

  side_effect_expr_function_callt call;
  int contract_var_id = -1;
  exprt base;
  std::string base_cname = "";

  if (
    !resolved_caller.contains("referencedDeclaration") && is_bound &&
    !cast_target_cname.empty() && structureTypingMap.count(cast_target_cname))
  {
    // The cast operand is not a persistent contract variable — e.g.
    // `ICallback(msg.sender).cb()`.  The operand is an address-valued
    // expression (msg.sender, a literal, address(this), a computed
    // address) with no variable to which a contract type was bound.
    //
    // Under --bound the address still denotes a concrete system
    // contract: `Contract(addr).f()` must dispatch to whichever tracked
    // instance is deployed at `addr` (EVM dynamic dispatch by address).
    // Resolve `addr`, rewrap it as a CONTRACT pointer to the cast
    // target's singleton, and fall through to the same high-level
    // dispatch used for typed-variable callees.  Without this the call
    // is havoc'd and any property that depends on the callee executing
    // (re-entrancy via `I(msg.sender).f()`, ERC777 `tokensReceived`,
    // approveAndCall, ...) is silently missed — a soundness false
    // negative.
    if (get_expr(
          resolved_caller,
          resolved_caller.contains("typeDescriptions")
            ? resolved_caller["typeDescriptions"]
            : nlohmann::json(nullptr),
          base))
      return true;

    // Rewrap the raw address as a CONTRACT pointer to the cast target's
    // singleton (mirrors the variable-backed cast path below).
    // convert_type_expr patches the singleton's $address to carry `addr`
    // and returns `&singleton` as the CONTRACT-typed base.
    typet target_type = symbol_typet(prefix + cast_target_cname);
    target_type.set("#sol_contract", cast_target_cname);
    set_sol_type(target_type, SolidityGrammar::SolType::CONTRACT);
    convert_type_expr(ns, base, target_type, expr);
    base.type().set("#sol_contract", cast_target_cname);
    base_cname = cast_target_cname;
  }
  else if (!resolved_caller.contains("referencedDeclaration"))
  {
    // e.g. `C(address(0x1234)).fun` in unbound mode, or no tracked
    // implementer of the cast target. Model the read as a nondet value
    // of the member's declared type; if the member is later called, the
    // call site will be handled by the opaque fn-ptr path. This mirrors
    // the `(new C()).x` fallback above.
    typet ret_t;
    bool have_type = false;
    int fn_ref = expr.value("referencedDeclaration", -1);
    if (fn_ref > 0)
    {
      const nlohmann::json &fdecl = find_decl_ref(fn_ref);
      if (!fdecl.empty() && !fdecl.is_null())
      {
        std::string fnode = fdecl.value("nodeType", "");
        if (
          fnode == "FunctionDefinition" && fdecl.contains("returnParameters") &&
          fdecl["returnParameters"].contains("parameters") &&
          fdecl["returnParameters"]["parameters"].is_array() &&
          !fdecl["returnParameters"]["parameters"].empty())
        {
          const auto &rp = fdecl["returnParameters"]["parameters"][0];
          if (rp.contains("typeDescriptions"))
          {
            if (get_type_description(rp["typeDescriptions"], ret_t))
              return true;
            have_type = true;
          }
        }
        else if (
          fnode == "VariableDeclaration" && fdecl.contains("typeName") &&
          fdecl["typeName"].contains("typeDescriptions"))
        {
          if (get_type_description(
                fdecl["typeName"]["typeDescriptions"], ret_t))
            return true;
          have_type = true;
        }
      }
    }
    if (!have_type && expr.contains("typeDescriptions"))
    {
      if (get_type_description(expr["typeDescriptions"], ret_t))
        return true;
      have_type = true;
    }
    if (!have_type)
      ret_t = empty_typet();
    exprt nondet = exprt("sideeffect", ret_t);
    nondet.statement("nondet");
    new_expr = nondet;
    return false;
  }
  else
  {
    contract_var_id = resolved_caller["referencedDeclaration"].get<int>();
    assert(!current_baseContractName.empty());
    const nlohmann::json &base_expr_json =
      find_decl_ref(contract_var_id); // contract

    // contract C{ Base x; x.call();} where base.contractname != current_ContractName;
    // therefore, we need to extract the based contract name
    if (base_expr_json.empty())
    {
      // assume it's 'this'
      if (contract_var_id < 0 && resolved_caller.value("name", "") == "this")
      {
        exprt this_expr;
        assert(current_functionDecl);
        if (get_func_decl_this_ref(*current_functionDecl, this_expr))
          return true;
        base = this_expr;

        assert(!current_contractName.empty());
        base_cname = current_contractName;
      }
      else
      {
        log_error("Unexpect base expression");
        return true;
      }
    }
    else if (
      base_expr_json.contains("nodeType") &&
      base_expr_json["nodeType"] == "ContractDefinition")
    {
      // Static function reference: `ContractName.funcName` (e.g.
      // `ERC721TokenReceiver.onERC721Received.selector`). There is
      // no instance — use the contract name as the scope and leave
      // base as nil.
      base_cname = base_expr_json.value("name", "");
      assert(!base_cname.empty());
    }
    else
    {
      if (get_var_decl_ref(base_expr_json, true, base))
        return true;
      base_cname = base.type().get("#sol_contract").as_string();
      // The inner variable may be an address that was cast to a
      // contract type (e.g. `ERC721TokenReceiver(to).f()`); in
      // that case the cast's target type is the correct scope.
      if (base_cname.empty() && !cast_target_cname.empty())
      {
        base_cname = cast_target_cname;

        // Re-wrap `base` so its type is CONTRACT (pointer to the
        // target singleton) instead of the raw address.  Without this
        // `get_high_level_member_access` aborts with
        //   ERROR: Expecting contract type  (unsignedbv width=160)
        // for every `Contract(addrVar).method(...)` call pattern — a
        // pattern that is pervasive in DeFi contracts (IERC20(token).
        // transfer(...), Callback(addr).onX(...), Oracle(feed).
        // latestAnswer(), etc.).
        //
        // Semantics: Solidity `Contract(addr)` is a pure type
        // annotation at runtime — the underlying value is still the
        // 160-bit address.  ESBMC's address↔contract conversion logic
        // in convert_type_expr mutates `_ESBMC_Object_<C>.$address` to
        // carry the cast's address and returns `&singleton` as the
        // CONTRACT-typed base.  Downstream dispatch is mode-aware:
        //   - --bound: calls the singleton's method body, with
        //     $address carrying the cast addr so subsequent
        //     `address(c) == someAddr` checks line up.
        //   - --unbound: routes through _ESBMC_Nondet_Extcall_<C>,
        //     which havocs state regardless of the concrete address
        //     (correct over-approximation for opaque externals).
        if (get_sol_type(base.type()) != SolidityGrammar::SolType::CONTRACT)
        {
          typet target_type = symbol_typet(prefix + cast_target_cname);
          target_type.set("#sol_contract", cast_target_cname);
          set_sol_type(target_type, SolidityGrammar::SolType::CONTRACT);
          convert_type_expr(ns, base, target_type, expr);
          // `convert_type_expr` sets `#sol_type = CONTRACT` on the
          // returned pointer but leaves `#sol_contract` unset on the
          // pointer itself.  `get_base_contract_name` (called later by
          // `get_high_level_member_access`) looks at the pointer's
          // `#sol_contract`, so stamp it explicitly.
          base.type().set("#sol_contract", cast_target_cname);
        }
      }
      assert(!base_cname.empty());
    }
  }

  const int member_id = callee_expr_json["referencedDeclaration"].get<int>();
  const nlohmann::json *member_decl_ptr;
  {
    ScopeGuard<std::string> guard(current_baseContractName, base_cname);
    member_decl_ptr = &find_decl_ref(member_id); // methods or variables
  }
  const nlohmann::json &member_decl_ref = *member_decl_ptr;

  if (member_decl_ref.empty())
  {
    log_error("cannot find member json node reference");
    return true;
  }

  auto elem_type =
    SolidityGrammar::get_contract_body_element_t(member_decl_ref);
  log_debug(
    "solidity",
    "\t\t@@@ got contrant body element = {}",
    SolidityGrammar::contract_body_element_to_str(elem_type));
  switch (elem_type)
  {
  case SolidityGrammar::VarDecl:
  {
    // e.g. x.data()
    // ==> x.data, where data is a state variable in the contract
    // in Solidity the x.data() is read-only
    exprt comp;
    if (get_var_decl_ref(member_decl_ref, false, comp))
      return true;
    const irep_idt comp_name = comp.name();

    // special checks for mapping
    // e.g. _b.map(0)
    // => map[0] or map_uint_get(ins , 0);
    if (get_sol_type(comp.type()) == SolidityGrammar::SolType::MAPPING)
    {
      assert(func_call_json.contains("arguments"));
      assert(member_decl_ref.contains("typeName"));
      assert(member_decl_ref["typeName"].contains("valueType"));
      const nlohmann::json &args = func_call_json["arguments"];

      bool is_new_expr = should_treat_as_new(base_cname);
      const bool nested = args.size() > 1;

      // Walk one level at a time, collecting per-level keys (raw
      // 256-bit for single-level so we match in-contract writes,
      // xor-folded for nested so multiple keys can be packed into
      // one slot via combine_mapping_keys_256). Recover the LEAF
      // value type along the way.
      std::vector<exprt> per_level_keys;
      const nlohmann::json *cur_typename = &member_decl_ref["typeName"];
      typet leaf_value_t;
      SolidityGrammar::SolType leaf_val_sol_type =
        SolidityGrammar::SolType::UNSET;
      for (size_t i = 0; i < args.size(); ++i)
      {
        if (
          !cur_typename->is_object() ||
          cur_typename->value("nodeType", "") != "Mapping" ||
          !cur_typename->contains("keyType") ||
          !cur_typename->contains("valueType"))
        {
          log_error(
            "nested mapping getter has more arguments than mapping "
            "levels");
          return true;
        }
        typet level_key_t;
        if (get_type_description(
              (*cur_typename)["keyType"]["typeDescriptions"], level_key_t))
          return true;
        exprt key_expr;
        if (get_expr(
              args[i],
              (*cur_typename)["keyType"]["typeDescriptions"],
              key_expr))
          return true;
        gen_mapping_key_typecast(
          current_contractName, key_expr, location, level_key_t);
        if (nested)
          xor_fold_key_to_64bit(key_expr);
        per_level_keys.push_back(key_expr);

        cur_typename = &(*cur_typename)["valueType"];
        if (get_type_description(
              (*cur_typename)["typeDescriptions"], leaf_value_t))
          return true;
        leaf_val_sol_type = get_sol_type(leaf_value_t);
      }

      if (!is_bound && !is_new_created_decl(contract_var_id))
      {
        // Truly unbound mapping getter — return nondet of the leaf type.
        get_nondet_expr(leaf_value_t, new_expr);
        break;
      }

      if (!is_new_expr)
      {
        // !is_new_expr: comp.type() is a chain of array_typet whose
        // subtype is the next level (decl-time populated). Index level
        // by level so reads here match in-contract writes that go
        // through the same chained-subtype index_exprt. Apply
        // xor-folding consistently with the in-contract path
        // (which folds for the !is_new_expr array model).
        assert(comp.type().is_array());
        exprt cur = comp;
        for (size_t i = 0; i < per_level_keys.size(); ++i)
        {
          exprt k = per_level_keys[i];
          if (!nested)
            xor_fold_key_to_64bit(k);
          typet sub = cur.type().subtype();
          cur = index_exprt(cur, k, sub);
        }
        new_expr = cur;
      }
      else
      {
        // is_new_expr: comp is mapping_t.
        //   - single-level: pass the raw key through so we share a
        //     slot with the in-contract write (which also passes raw).
        //   - nested: combine folded keys into one packed slot —
        //     mirrors the in-contract write path in
        //     get_index_access_expr's nested-mapping intercept.
        exprt key_for_call;
        if (nested)
          combine_mapping_keys_256(per_level_keys, key_for_call);
        else
          key_for_call = per_level_keys[0];

        // Dispatch-singleton routing.  Cross-contract function calls
        // like `c1.approve(...)` are wrapped by the dispatcher when
        // cname_set has more than one element (structural typing
        // matched multiple contract types), and the wrapper executes
        // against `_ESBMC_Object_<cname>` for whichever cname matches
        // the runtime bind_cname.  Reads via the mapping getter must
        // use the same backing store; otherwise writes land in the
        // singleton's storage and reads against `&c1->allowed` see
        // nothing.  When cname_set.size() == 1, no dispatch wrapper
        // is built — the call goes directly to the local pointer's
        // storage, so reads must too.
        //
        // Polymorphic read via the per-pointer bind shadow: when the
        // cluster has more than one candidate, consult the shadow
        // symbol on `base` (populated by `new` and by `C(_addr)` cast)
        // to select the right singleton.  Fall back to the declared
        // `base_cname` singleton when no shadow exists (e.g. state var
        // or parameter).
        std::unordered_set<std::string> cname_set =
          structureTypingMap[base_cname];
        const bool polymorphic = cname_set.size() > 1;

        if (!polymorphic)
        {
          auto _mem_call = member_exprt(base, comp.name(), comp.type());
          bool is_mapping_set = false;
          if (get_new_mapping_index_access(
                leaf_value_t,
                leaf_val_sol_type,
                is_mapping_set,
                _mem_call,
                key_for_call,
                location,
                new_expr))
            return true;
          break;
        }

        // Polymorphic: strip non-contract cluster entries.
        for (auto non_cname : nonContractNamesList)
        {
          if (non_cname == base_cname)
            continue;
          cname_set.erase(non_cname);
        }

        // If base has a per-pointer shadow, key the if-ladder on it.
        // Otherwise the dispatch is static — use the declared base_cname
        // singleton only.
        exprt shadow;
        const bool have_shadow = !get_bind_shadow_read(base, shadow);

        if (!have_shadow)
        {
          // Static route to base_cname's singleton (legacy behaviour,
          // unchanged).  Without a shadow we cannot distinguish
          // structurally-identical singletons at this point.
          exprt storage_base;
          get_static_contract_instance_ref(base_cname, storage_base);
          auto _mem_call = member_exprt(storage_base, comp.name(), comp.type());
          bool is_mapping_set = false;
          if (get_new_mapping_index_access(
                leaf_value_t,
                leaf_val_sol_type,
                is_mapping_set,
                _mem_call,
                key_for_call,
                location,
                new_expr))
            return true;
          break;
        }

        // Build the ladder.  Innermost else reads from the declared
        // base_cname's singleton (covers the common bind == declared
        // case; shadow defaults to declared at init so this branch
        // always has a consistent meaning).
        exprt default_storage;
        get_static_contract_instance_ref(base_cname, default_storage);
        auto default_mem =
          member_exprt(default_storage, comp.name(), comp.type());
        exprt ladder;
        {
          bool is_mapping_set = false;
          if (get_new_mapping_index_access(
                leaf_value_t,
                leaf_val_sol_type,
                is_mapping_set,
                default_mem,
                key_for_call,
                location,
                ladder))
            return true;
        }

        for (const auto &alt_cname : cname_set)
        {
          if (alt_cname == base_cname)
            continue;

          const symbolt *struct_sym = context.find_symbol(prefix + alt_cname);
          if (
            struct_sym == nullptr || struct_sym->type.id() != "struct" ||
            !to_struct_type(struct_sym->type).has_component(comp.name()))
            continue;

          exprt cname_string;
          typet ct = string_t;
          ct.cmt_constant(true);
          get_symbol_decl_ref(alt_cname, "sol:@" + alt_cname, ct, cname_string);

          exprt cmp = exprt("=", bool_type());
          cmp.copy_to_operands(shadow, cname_string);

          exprt alt_storage;
          get_static_contract_instance_ref(alt_cname, alt_storage);
          auto alt_mem = member_exprt(alt_storage, comp.name(), comp.type());
          exprt alt_read;
          bool is_mapping_set = false;
          if (get_new_mapping_index_access(
                leaf_value_t,
                leaf_val_sol_type,
                is_mapping_set,
                alt_mem,
                key_for_call,
                location,
                alt_read))
            return true;

          ladder = if_exprt(cmp, alt_read, ladder);
          ladder.type() = alt_read.type();
        }
        new_expr = ladder;
      }
      break;
    }

    if (current_contractName == base_cname)
      // this.member();
      // comp can be either symbol_expr or member_expr
      new_expr = member_exprt(base, comp_name, comp.type());
    else if (!is_bound && !is_new_created_decl(contract_var_id))
      // Truly unbound state-variable getter — return nondet.
      get_nondet_expr(comp.type(), new_expr);
    else
    {
      assert(!comp.is_member());
      auto _mem_call = member_exprt(base, comp_name, comp.type());
      if (get_high_level_member_access(
            func_call_json, base, comp, _mem_call, false, new_expr))
        return true;
    }

    break;
  }
  case SolidityGrammar::FunctionDef:
  {
    // When the MemberAccess itself is a function-reference VALUE rather
    // than a call target (e.g. `this.oracleResponse` passed as the second
    // argument to `ORACLE_CONST.query("USD", this.oracleResponse)`), the
    // nearest parent FunctionCall is the *outer* call — not a call whose
    // `.expression` is this MemberAccess. In that case we must not try
    // to build a function_call (there are no arguments, and find_last_parent
    // returns the outer call so get_non_library_function_call crashes).
    // Lower to an opaque void* nondet, mirroring the internal function-type
    // lowering in solidity_convert_type.cpp. Indirect calls through this
    // pointer have already been rewritten to nondet returns elsewhere.
    bool used_as_value = true;
    if (
      func_call_json.contains("nodeType") &&
      (func_call_json["nodeType"] == "FunctionCall" ||
       func_call_json["nodeType"] == "FunctionCallOptions") &&
      func_call_json.contains("expression") &&
      func_call_json["expression"].contains("id") &&
      callee_expr_json.contains("id") &&
      func_call_json["expression"]["id"] == callee_expr_json["id"])
    {
      used_as_value = false;
    }
    if (used_as_value)
    {
      typet ptr_t = gen_pointer_type(empty_typet());
      ptr_t.set("#sol_func_ptr", true);
      // [APPROX: UNDER for content / OVER for identity]
      // Function-reference r-values (e.g. passing `this.f` as an argument
      // or storing it in a variable) lower to an opaque void* whose bit
      // pattern is a constant cast of the AST id of the referenced
      // FunctionDefinition. This gives stable identity:
      //   - `this.f == this.f` is always true
      //   - `this.f != this.g` is always true for distinct functions.
      // Indirect calls through such pointers NEVER execute the real
      // function body — convert_call lowers them to a nondet of the
      // declared return type (see solidity_convert_call.cpp). That is an
      // UNDER-approximation of control flow (we miss bugs inside the
      // callee) but an OVER-approximation of the return value.
      // False positives: none for the identity comparison itself.
      // False negatives: bugs inside a function invoked only through an
      //   indirect fn-ptr call cannot be detected.
      int fn_id = -1;
      if (
        callee_expr_json.contains("referencedDeclaration") &&
        !callee_expr_json["referencedDeclaration"].is_null())
        fn_id = callee_expr_json["referencedDeclaration"].get<int>();
      if (fn_id >= 0)
      {
        exprt id_const = constant_exprt(
          integer2binary(fn_id + 1, bv_width(size_type())),
          integer2string(fn_id + 1),
          size_type());
        new_expr = typecast_exprt(id_const, ptr_t);
      }
      else
      {
        get_nondet_expr(ptr_t, new_expr);
      }
      break;
    }

    // e.g. x.func()
    // x    --> base
    // func --> comp
    exprt comp;
    if (get_func_decl_ref(member_decl_ref, comp))
      return true;

    if (get_non_library_function_call(member_decl_ref, args_call_json, call))
      return true;
    call.arguments().at(0) = base;

    // Foundry cheatcode interception. Must precede the bound / new-instance
    // branching below: the `vm` handle is a `constant` (not created via `new`),
    // so in unbound mode a `vm.<name>(...)` call would otherwise fall to the
    // nondet path and the cheatcode become a silent no-op. Recognized
    // cheatcodes lower to their effect here; unrecognized ones fall through.
    // forge-std splits cheatcodes across `interface VmSafe` (view/pure) and
    // `interface Vm is VmSafe` (state-changing). A handle may be typed as either
    // (e.g. StdUtils uses `VmSafe constant vm`), so both must be gated or a
    // VmSafe call escapes to the nondet no-op path and breaks prune soundness.
    if (base_cname == "Vm" || base_cname == "VmSafe")
    {
      locationt cl;
      get_location_from_node(func_call_json, cl);
      bool handled = false;
      if (handle_foundry_cheatcode(func_call_json, cl, new_expr, handled))
        return true;
      if (handled)
        break;
    }

    if (current_contractName == base_cname)
    {
      // this.init(); we know the implementation thus cannot model it as unbound_harness
      // note that here is comp.identifier not comp.name
      // in unbound mode, we cannot determine the sender
      // wrap with msg_sender update:
      //  old_sender = msg_sender
      //  msg_sender = this.address
      //  ...
      //  msg_sender = old_sender
      // uint160_t old_sender =  msg_sender;
      std::string debug_modulename = get_modulename_from_path(absolute_path);
      exprt msg_sender = symbol_expr(*context.find_symbol("c:@msg_sender"));
      exprt this_expr;
      assert(current_functionDecl);
      if (get_func_decl_this_ref(*current_functionDecl, this_expr))
        return true;

      symbolt old_sender;
      get_default_symbol(
        old_sender,
        debug_modulename,
        addr_t,
        "old_sender",
        "sol:@C@" + current_contractName + "@F@old_sender#" +
          std::to_string(aux_counter++),
        locationt());
      symbolt &added_old_sender = *move_symbol_to_context(old_sender);
      code_declt old_sender_decl(symbol_expr(added_old_sender));
      added_old_sender.value = msg_sender;
      old_sender_decl.operands().push_back(msg_sender);
      move_to_front_block(old_sender_decl);

      // msg_sender = this.address;
      exprt this_address = member_exprt(this_expr, "$address", addr_t);
      exprt assign_sender = side_effect_exprt("assign", addr_t);
      assign_sender.copy_to_operands(msg_sender, this_address);
      convert_expression_to_code(assign_sender);
      move_to_front_block(assign_sender);

      // msg_sender = old_sender;
      exprt assign_sender_restore = side_effect_exprt("assign", addr_t);
      assign_sender_restore.copy_to_operands(
        msg_sender, symbol_expr(added_old_sender));
      convert_expression_to_code(assign_sender_restore);
      move_to_back_block(assign_sender_restore);

      new_expr = call;
    }
    else if (!is_bound && !is_new_created_decl(contract_var_id))
    {
      // Truly unbound: the variable was NOT created via `new`, so we
      // have no concrete implementation to call — return nondet.
      if (get_unbound_expr(func_call_json, current_contractName, new_expr))
        return true;

      typet t = to_code_type(comp.type()).return_type();
      get_nondet_expr(t, new_expr);
    }
    else
    {
      // Bound mode, or unbound but the variable was created via `new C()`:
      // dispatch to the concrete implementation (the callee body is executed).
      assert(!comp.is_member());
      if (get_high_level_member_access(
            func_call_json, literal_type, base, comp, call, true, new_expr))
        return true;
      // Consume a pending vm.expectRevert(): assert the call just built
      // reverted. Emitted into the back block so it runs AFTER the call; the
      // revert flag reflects the callee's outcome (public entry clears it, a
      // revert marks it). Requires uses_revert_observation (set for expectRevert).
      if (pending_expect_revert)
      {
        pending_expect_revert = false;
        const symbolt *flag =
          context.find_symbol("c:@_ESBMC_sol_reverted_flag");
        if (flag != nullptr)
        {
          locationt rl;
          get_location_from_node(func_call_json, rl);
          code_assertt ra(symbol_expr(*flag));
          ra.location() = rl;
          move_to_back_block(ra);
        }
      }
    }

    break;
  }
  default:
  {
    log_error(
      "Unexpected Member Access Element Type, Got {}",
      SolidityGrammar::contract_body_element_to_str(elem_type));
    return true;
  }
  }

  return false;
}

bool solidity_convertert::get_index_access_expr(
  const nlohmann::json &expr,
  const nlohmann::json &literal_type,
  exprt &new_expr)
{
  locationt location;
  get_start_location_from_stmt(expr, location);

  std::string current_contractName;
  get_current_contract_name(expr, current_contractName);

  // Nested mapping access in is_new_expr mode: m[k1][k2]...[kN] where the
  // base resolves to a mapping_t.  The default chained lowering does
  // `*(map_generic_get(&m, k1) + k2)` which has no read/write storage
  // link in the linked-list mapping model — writes never persist for a
  // matching read and TOD comparisons devolve into solver-chosen nondet.
  // Collapse the whole chain into one combined-key map_<leaf>_get/set
  // here so both sides of an assignment touch the same backing entry.
  // Only intercept when:
  //   - the immediate base is itself an IndexAccess (nested), AND
  //   - the deepest base variable is typed MAPPING, AND
  //   - we are in is_new_expr mode for the current contract.
  // The !is_new_expr path uses chained array_typet indexing (correct as-is).
  if (
    expr.contains("baseExpression") && expr["baseExpression"].is_object() &&
    expr["baseExpression"].value("nodeType", "") == "IndexAccess" &&
    should_treat_as_new(current_contractName))
  {
    // Walk down baseExpression chain to collect indices (outer→inner)
    // and find the base variable node.
    std::vector<const nlohmann::json *> idx_jsons;
    const nlohmann::json *cur = &expr;
    while (cur->is_object() && cur->value("nodeType", "") == "IndexAccess")
    {
      idx_jsons.push_back(&(*cur)["indexExpression"]);
      cur = &(*cur)["baseExpression"];
    }
    const nlohmann::json &base_var_json = *cur;

    // The base must be typed MAPPING, otherwise this is multi-dim arrays
    // or some other nested shape — let the default code handle it.
    typet base_var_t;
    if (
      base_var_json.is_object() && base_var_json.contains("typeDescriptions") &&
      !get_type_description(base_var_json["typeDescriptions"], base_var_t) &&
      get_sol_type(base_var_t) == SolidityGrammar::SolType::MAPPING &&
      base_var_json.contains("referencedDeclaration"))
    {
      // Resolve the var ref and fetch the mapping decl JSON for typeName walking.
      exprt array;
      if (get_expr(base_var_json, literal_type, array))
        return true;
      const nlohmann::json &map_decl =
        find_decl_ref(base_var_json["referencedDeclaration"].get<int>());

      // Pre-validate: count Mapping levels available in the typeName
      // chain.  If the index chain is deeper than the mapping nesting
      // (e.g. `mapping(K => V[])[k][i]` — one mapping level but two
      // indices, the second is an array index on the mapping's
      // array-typed value), the intercept does not apply.  Skip it
      // and let the default nested-IndexAccess path handle the outer
      // index via regular array indexing on the mapping-helper
      // result.  Without this guard we walk off the end of the
      // typeName chain and log a spurious error.
      size_t mapping_depth = 0;
      {
        const nlohmann::json *probe = &map_decl["typeName"];
        while (probe->is_object() &&
               probe->value("nodeType", "") == "Mapping" &&
               probe->contains("valueType"))
        {
          ++mapping_depth;
          probe = &(*probe)["valueType"];
        }
      }
      if (mapping_depth >= idx_jsons.size())
      {
        // Walk the typeName chain alongside idx_jsons to lower each key
        // and recover the leaf value type.
        std::vector<exprt> folded_keys;
        const nlohmann::json *tn = &map_decl["typeName"];
        typet leaf_value_t;
        SolidityGrammar::SolType leaf_val_sol_type =
          SolidityGrammar::SolType::UNSET;
        // idx_jsons is outer→inner (i.e., k1 first, then k2, etc.) — same
        // order as Solidity's m[k1][k2] write/read.
        // Reverse: idx_jsons was pushed outer-first as we walked DOWN,
        // so idx_jsons[0] is the OUTERMOST index, which in Solidity
        // m[k1][k2][k3] corresponds to k3 (the last bracket).
        // To align with the inner-first typeName walk (typeName.keyType
        // is k1's type), reverse the index list.
        std::reverse(idx_jsons.begin(), idx_jsons.end());

        for (size_t i = 0; i < idx_jsons.size(); ++i)
        {
          // Guarded by the pre-validation above; assert defensively.
          assert(
            tn->is_object() && tn->value("nodeType", "") == "Mapping" &&
            tn->contains("keyType") && tn->contains("valueType"));
          typet level_key_t;
          if (get_type_description(
                (*tn)["keyType"]["typeDescriptions"], level_key_t))
            return true;
          exprt key_expr;
          if (get_expr(
                *idx_jsons[i], (*idx_jsons[i])["typeDescriptions"], key_expr))
            return true;
          gen_mapping_key_typecast(
            current_contractName, key_expr, location, level_key_t);
          xor_fold_key_to_64bit(key_expr);
          folded_keys.push_back(key_expr);

          tn = &(*tn)["valueType"];
          if (get_type_description((*tn)["typeDescriptions"], leaf_value_t))
            return true;
          leaf_val_sol_type = get_sol_type(leaf_value_t);
        }

        // Combine and emit a single map_<leaf>_get/set on the base mapping.
        exprt combined_key;
        combine_mapping_keys_256(folded_keys, combined_key);
        bool is_mapping_set = is_mapping_set_lvalue(expr);
        if (get_new_mapping_index_access(
              leaf_value_t,
              leaf_val_sol_type,
              is_mapping_set,
              array,
              combined_key,
              location,
              new_expr))
          return true;
        return false;
      }
      // else: fall through below — idx chain extends past the
      // mapping nesting (mapping value is an array or struct).  The
      // default path handles the outer indices as regular array
      // accesses on the result of the mapping helper.
    }
    // else: fall through to existing nested-array handling.
  }

  // Per-mapping flat-array encoder leaf access.
  // For mapping(K => T[N]) / mapping(K => T[N0]...[Nk-1]) declared with the
  // flat encoder (see solidity_convert_decl.cpp:245-311 — `t.set("#sol_mapping_flat_encoded", true)`),
  // every access reaches the scalar leaf (the decl-time gate proved no
  // partial accesses).  Walk the full IndexAccess chain and emit a single
  // composed index_exprt — no slab pointer, no helper call.
  //
  // Composed index = (key_zext_to_W << inner_bits) | inner_offset, where
  //   inner_offset = ((... ((idx_0 * N0) + idx_1) * N1 + idx_2) * N2 + ...)
  //   inner_bits   = ceil_log2(N0 * N1 * ... * N_{k-1})
  //   W            = 256 + inner_bits  (key portion is always 256, see scalar
  //                  fast path at solidity_convert_type.cpp:611)
  if (
    expr.contains("baseExpression") && expr["baseExpression"].is_object() &&
    expr["baseExpression"].value("nodeType", "") == "IndexAccess" &&
    !should_treat_as_new(current_contractName))
  {
    // Walk the chain bottom-up to find the Identifier root.
    std::vector<const nlohmann::json *> idx_jsons; // outer-to-inner
    const nlohmann::json *cur = &expr;
    while (cur->is_object() && cur->value("nodeType", "") == "IndexAccess")
    {
      idx_jsons.push_back(&(*cur)["indexExpression"]);
      cur = &(*cur)["baseExpression"];
    }
    // idx_jsons is currently inner-to-outer (we pushed each as we descended);
    // reverse to outer-to-inner so idx_jsons[0] is the key.
    std::reverse(idx_jsons.begin(), idx_jsons.end());

    if (
      cur->is_object() && cur->value("nodeType", "") == "Identifier" &&
      cur->contains("referencedDeclaration"))
    {
      // Resolve the identifier to inspect its symbol's type tags.
      exprt map_array;
      if (get_expr(*cur, literal_type, map_array))
        return true;

      if (map_array.type().get_bool("#sol_mapping_flat_encoded"))
      {
        // Walk the original mapping value type to collect inner dim
        // sizes (outer-to-inner).
        const nlohmann::json &map_decl_json =
          find_decl_ref((*cur)["referencedDeclaration"].get<int>());
        typet val_t;
        if (get_type_description(
              map_decl_json["typeName"]["valueType"]["typeDescriptions"],
              val_t))
          return true;

        std::vector<unsigned long> dim_sizes;
        const typet *cw = &val_t;
        while (cw->is_array())
        {
          const exprt &sz = to_array_type(*cw).size();
          BigInt n;
          if (
            sz.is_nil() || sz.id() == "infinity" || to_integer(sz, n) ||
            !n.is_uint64())
            break;
          dim_sizes.push_back(n.to_uint64());
          cw = &cw->subtype();
        }

        // Sanity: chain depth must be 1 (key) + dim_sizes.size().
        // The decl-time gate enforces this; if it doesn't match, fall
        // through to the existing dispatch which will produce an error
        // (the type isn't `mapping_t`, so the slow helper path won't fire).
        if (idx_jsons.size() == 1 + dim_sizes.size())
        {
          unsigned index_width =
            std::stoul(map_array.type().get("#esbmc_index_width").as_string());
          unsigned inner_bits = std::stoul(
            map_array.type().get("#sol_flat_inner_bits").as_string());
          typet idx_t = unsignedbv_typet(index_width);

          // Lower the key.
          exprt key_pos;
          if (get_expr(
                *idx_jsons[0], (*idx_jsons[0])["typeDescriptions"], key_pos))
            return true;
          typet key_t;
          if (get_type_description(
                map_decl_json["typeName"]["keyType"]["typeDescriptions"],
                key_t))
            return true;
          gen_mapping_key_typecast(
            current_contractName, key_pos, location, key_t);
          xor_fold_key_to_64bit(key_pos); // identity-to-256-bit
          solidity_gen_typecast(ns, key_pos, idx_t);
          // shifted_key = key_pos << inner_bits
          exprt shifted_key("shl", idx_t);
          shifted_key.copy_to_operands(
            key_pos, from_integer(inner_bits, idx_t));

          // Compose inner offset.  accum starts at 0; for each dim j
          // (1-indexed in idx_jsons), accum = accum * dim_sizes[j-1] + idx_j.
          exprt accum = from_integer(0, idx_t);
          for (size_t j = 1; j < idx_jsons.size(); ++j)
          {
            exprt this_idx;
            if (get_expr(
                  *idx_jsons[j], (*idx_jsons[j])["typeDescriptions"], this_idx))
              return true;
            solidity_gen_typecast(ns, this_idx, idx_t);
            exprt mul("*", idx_t);
            mul.copy_to_operands(accum, from_integer(dim_sizes[j - 1], idx_t));
            exprt add("+", idx_t);
            add.copy_to_operands(mul, this_idx);
            accum = add;
          }

          // Final composed index: shifted_key | accum
          exprt final_idx("bitor", idx_t);
          final_idx.copy_to_operands(shifted_key, accum);

          new_expr =
            index_exprt(map_array, final_idx, map_array.type().subtype());
          return false;
        }
      }
    }
  }

  const nlohmann::json &base_json = expr["baseExpression"];
  const nlohmann::json &index_json = expr["indexExpression"];

  // 1. get type, this is the base type of array
  typet t;
  if (get_type_description(expr["typeDescriptions"], t))
    return true;

  // 2. get the decl ref of the array
  exprt array;

  // 2.1 arr[n] / x.arr[n]
  if (base_json.contains("referencedDeclaration"))
  {
    if (get_expr(base_json, literal_type, array))
      return true;
  }
  // 2.2 nested mapping: m[k1][k2] — base is itself an IndexAccess
  else if (base_json.value("nodeType", "") == "IndexAccess")
  {
    if (get_expr(base_json, literal_type, array))
      return true;
  }
  else
  {
    // 2.3 func()[n]
    const nlohmann::json &decl = base_json;
    nlohmann::json implicit_cast_expr =
      make_implicit_cast_expr(decl, "ArrayToPointerDecay");
    if (get_expr(implicit_cast_expr, literal_type, array))
      return true;
  }

  // 3. get the position index
  exprt pos;
  if (get_expr(index_json, index_json["typeDescriptions"], pos))
    return true;

  // for MAPPING
  typet base_t;
  if (get_type_description(base_json["typeDescriptions"], base_t))
    return true;
  if (get_sol_type(base_t) == SolidityGrammar::SolType::MAPPING)
  {
    bool is_new_expr = should_treat_as_new(current_contractName);

    if (base_json.contains("referencedDeclaration"))
    {
      // Direct mapping access: m[k]
      const nlohmann::json &map_node =
        find_decl_ref(base_json["referencedDeclaration"].get<int>());

      // get key/value type
      typet key_t, value_t;
      SolidityGrammar::SolType key_sol_type, val_sol_type;
      if (get_mapping_key_value_type(
            map_node, key_t, value_t, key_sol_type, val_sol_type))
      {
        log_error("cannot get mapping key/value type");
        return true;
      }
      gen_mapping_key_typecast(current_contractName, pos, location, key_t);

      // Fast path: static-singleton mapping lowered to `array<V, inf>`
      // with a scalar / struct V. Skip when the decl layer has forced
      // `mapping_t` (fixed-array-value mapping, `#sol_mapping_fixed_arr_value`),
      // in which case `array.type()` is the mapping_t struct and must
      // flow through the helper below.
      if (!is_new_expr && array.type().is_array())
      {
        xor_fold_key_to_64bit(pos);
        // Use the array's declared subtype rather than `t` from
        // get_type_description, which may lack subtypes for nested mappings.
        new_expr = index_exprt(array, pos, array.type().subtype());
      }
      else
      {
        bool is_mapping_set = is_mapping_set_lvalue(expr);
        if (get_new_mapping_index_access(
              value_t,
              val_sol_type,
              is_mapping_set,
              array,
              pos,
              location,
              new_expr))
          return true;
      }
    }
    else
    {
      // Nested mapping access: m[k1][k2] — base is itself an IndexAccess
      // The inner access was already resolved; just index into the result.
      // Route through gen_mapping_key_typecast so bytesN / string / bytes
      // keys go through their dedicated mapping-key lowerings
      // (bytes_static_to_mapping_key, str2uint, bytes_dynamic_to_mapping_key),
      // not a raw solidity_gen_typecast that leaves the struct in place
      // and trips xor_fold_key_to_64bit on an irep2-unmappable
      // shr(struct, uint256). The lowered helpers already return uint256.
      gen_mapping_key_typecast(current_contractName, pos, location, pos.type());
      xor_fold_key_to_64bit(pos);
      // Use the resolved base array's declared subtype rather than
      // `t` from get_type_description: for nested mappings (>=3
      // levels) `t` is the AST value type and lacks the inner array
      // dimensions, so it under-nests the intermediate index node
      // (e.g. `array<uint256,inf>` collapsed to `uint256`).  That
      // mis-typed index later makes symex_assign_array build a
      // with2t whose value is non-array while the source subtype is
      // an array -> with2t::assert_consistency / value_sett::assign
      // abort on the deep nested-mapping WRITE.  This mirrors the
      // direct-access fast path (array.type().subtype(), above).
      // Fall back to `t` only when the resolved base is not
      // array-typed (mapping_t struct: fixed-array-value / new-expr
      // / cloned shapes), where index_exprt typing is unaffected.
      new_expr = index_exprt(
        array, pos, array.type().is_array() ? array.type().subtype() : t);
    }
    return false;
  }

  // for BYTESN or BYTES, read-only access
  if (is_byte_type(base_t))
  {
    bool is_bytes_set = is_mapping_set_lvalue(expr); // set vs get
    typet result_type = byte_static_t;
    result_type.set("#sol_bytesn_size", 1);

    std::string aux_name, aux_id;
    get_aux_var(aux_name, aux_id);
    std::string mod_name = get_modulename_from_path(absolute_path);
    symbolt aux_sym;
    get_default_symbol(
      aux_sym, mod_name, result_type, aux_name, aux_id, location);
    aux_sym.file_local = true;
    aux_sym.lvalue = true;
    auto &added_sym = *move_symbol_to_context(aux_sym);

    exprt arg_val = array;

    if (!arg_val.is_symbol())
    {
      std::string temp_name, temp_id;
      get_aux_var(temp_name, temp_id);
      symbolt temp_sym;
      get_default_symbol(
        temp_sym, mod_name, array.type(), temp_name, temp_id, location);
      temp_sym.file_local = true;
      temp_sym.lvalue = true;
      auto &tmp_added_sym = *move_symbol_to_context(temp_sym);
      tmp_added_sym.value = array;
      code_declt decl(symbol_expr(tmp_added_sym));
      decl.operands().push_back(array);
      move_to_front_block(decl);
      arg_val = symbol_expr(tmp_added_sym);
    }

    // static bytes (bytesN)
    if (is_bytesN_type(base_t))
    {
      if (!is_bytes_set)
      {
        side_effect_expr_function_callt get_call;
        get_library_function_call_no_args(
          "bytes_static_get",
          "c:@F@bytes_static_get",
          result_type,
          location,
          get_call);
        get_call.arguments().push_back(address_of_exprt(arg_val));
        get_call.arguments().push_back(pos);
        added_sym.value = get_call;

        code_declt decl(symbol_expr(added_sym));
        decl.operands().push_back(get_call);
        move_to_front_block(decl);

        new_expr = symbol_expr(added_sym);
      }
      else
      {
        side_effect_expr_function_callt set_call;
        get_library_function_call_no_args(
          "bytes_static_set",
          "c:@F@bytes_static_set",
          empty_typet(),
          location,
          set_call);
        set_call.arguments().push_back(address_of_exprt(arg_val));
        set_call.arguments().push_back(pos);
        set_call.arguments().push_back(symbol_expr(added_sym));
        move_to_back_block(set_call);

        new_expr = symbol_expr(added_sym);
      }
    }
    // dynamic bytes
    else
    {
      exprt dynamic_pool;
      if (get_dynamic_pool(current_contractName, dynamic_pool))
        return true;

      if (!is_bytes_set)
      {
        side_effect_expr_function_callt get_call;
        get_library_function_call_no_args(
          "bytes_dynamic_get",
          "c:@F@bytes_dynamic_get",
          result_type,
          location,
          get_call);
        get_call.arguments().push_back(address_of_exprt(arg_val));
        get_call.arguments().push_back(dynamic_pool);
        get_call.arguments().push_back(pos);
        added_sym.value = get_call;

        code_declt decl(symbol_expr(added_sym));
        decl.operands().push_back(get_call);
        move_to_front_block(decl);

        new_expr = symbol_expr(added_sym);
      }
      else
      {
        side_effect_expr_function_callt set_call;
        get_library_function_call_no_args(
          "bytes_dynamic_set",
          "c:@F@bytes_dynamic_set",
          empty_typet(),
          location,
          set_call);
        set_call.arguments().push_back(address_of_exprt(arg_val));
        set_call.arguments().push_back(pos);
        set_call.arguments().push_back(symbol_expr(added_sym));
        set_call.arguments().push_back(dynamic_pool);
        move_to_back_block(set_call);
        new_expr = symbol_expr(added_sym);
      }
    }
    return false;
  }

  // [APPROX: UNDER] Array-of-dynamic-bytes / array-of-string calldata
  // element read. `a[i]` where `a` is `bytes[] calldata` / `string[]
  // calldata` / `bytes[N] calldata` currently lowers to a plain
  // index_exprt over an uninitialised struct, so subsequent reads of
  // `a[i].length` / `a[i][j]` trigger spurious init and bounds-check
  // failures (the element's `.length` and `.initialized` are fully
  // unconstrained nondet).  We replace the element read with a nondet
  // BytesDynamic via llc_nondet_bytes() — the same helper used for
  // top-level `bytes calldata` harness parameters. Post-T1.2: length
  // is fully nondet (no [32, 1024] clamp); contracts must require()
  // a tighter range or materialise into a local memory bytes if they
  // need stable identity across reads.
  //
  // Scope:
  //  - Only fires when the base array is calldata (`#sol_data_loc ==
  //    "calldata"`), so user-declared storage `bytes[] x;` keeps the
  //    precise index_exprt path and still catches real OOB.
  //  - Only for rvalue reads (calldata is read-only in Solidity; any
  //    `a[i] = ...` would be rejected by solc anyway).
  //
  // Trade-off recorded in approximation ledger #10:
  //  - UNDER: repeated reads of the same index are independent samples
  //    (no `a[i] == a[i]` invariant). Tests that depend on calldata
  //    element equality across reads would fail spuriously; the
  //    workaround is `bytes memory b = a[i];` to materialise once.
  if (
    get_sol_type(t) == SolidityGrammar::SolType::BYTES_DYN &&
    base_t.get("#sol_data_loc").as_string() == "calldata")
  {
    log_warning(
      "[approx] calldata bytes[] element access at {}:{}: repeated "
      "reads of a[i] return independent nondet samples (no a[i]==a[i] "
      "invariant); materialise into local memory with 'bytes memory "
      "b = a[i];' if stable identity is required",
      location.get_file().c_str(),
      location.get_line().c_str());
    side_effect_expr_function_callt nondet_b;
    get_library_function_call_no_args(
      "llc_nondet_bytes", "c:@F@llc_nondet_bytes", t, location, nondet_b);
    new_expr = nondet_b;
    return false;
  }

  // Same approximation for `string[] calldata a; ... a[i]`. Solidity
  // `string` lowers to `char *` so the BytesDynamic helper above doesn't
  // match. Substitute a bounded nondet_string() — valid zero-terminated
  // char* up to _ESBMC_NONDET_STRING_MAX — so downstream `bytes(a[i])`,
  // `a[i].length` and `a[i][j]` reads operate on a real buffer.
  if (
    get_sol_type(t) == SolidityGrammar::SolType::STRING &&
    base_t.get("#sol_data_loc").as_string() == "calldata")
  {
    side_effect_expr_function_callt nondet_s;
    get_library_function_call_no_args(
      "nondet_string", "c:@F@nondet_string", t, location, nondet_s);
    new_expr = nondet_s;
    return false;
  }

  // --bounds-check (Solidity opt-in): local dynamic (`new T[](n)`) and fixed
  // (`T[k] memory`) arrays lower to a *pointer*-backed buffer, so goto-check's
  // built-in array-bounds check (which only fires on a native array_typet index
  // source) is skipped, and `--no-standard-checks` disables the pointer-deref
  // check too — leaving these accesses unguarded. Emit an explicit `pos < len`
  // claim for pointer-backed arrays so an out-of-bounds access is caught. The
  // length source is the fixed `#sol_array_size` metadata for fixed arrays, or
  // the runtime `_ESBMC_array_length(a)` header for dynamic ones. State-var
  // dyn-arrays and the byte/string/calldata approximations are handled on their
  // own paths above; this only augments the plain pointer fall-through below.
  auto emit_pointer_array_bounds_check = [&](
                                           const exprt &array_expr,
                                           const typet &base_type) {
    if (
      !config.options.get_bool_option("bounds-check") ||
      config.options.get_bool_option("no-bounds-check"))
      return;
    if (!array_expr.type().is_pointer())
      return;

    SolidityGrammar::SolType base_sol = get_sol_type(base_type);
    if (
      base_sol != SolidityGrammar::SolType::DYNARRAY &&
      base_sol != SolidityGrammar::SolType::ARRAY &&
      base_sol != SolidityGrammar::SolType::ARRAY_LITERAL)
      return;

    // Locate the length. Fixed arrays carry it as `#sol_array_size` metadata
    // (on the pointer type, the base type, or its element subtype); dynamic
    // arrays keep it in the runtime allocation header.
    exprt len_expr;
    std::string fixed_size;
    if (!array_expr.type().get("#sol_array_size").empty())
      fixed_size = array_expr.type().get("#sol_array_size").as_string();
    else if (!base_type.get("#sol_array_size").empty())
      fixed_size = base_type.get("#sol_array_size").as_string();
    else if (
      base_type.has_subtype() &&
      !base_type.subtype().get("#sol_array_size").empty())
      fixed_size = base_type.subtype().get("#sol_array_size").as_string();

    if (!fixed_size.empty())
      len_expr = from_integer(std::stoul(fixed_size), unsignedbv_typet(256));
    else if (base_sol == SolidityGrammar::SolType::DYNARRAY)
    {
      // The dynamic length is read with `_ESBMC_array_length(array)`, which
      // embeds `array` a second time (the access itself still indexes it). If
      // the base is not a plain symbol (e.g. `make()[i]`), evaluating it twice
      // would duplicate its side effects, so only emit the check for a stable
      // symbol base; otherwise degrade to no check.
      if (!array_expr.is_symbol())
        return;
      // No length header symbol → cannot bound; degrade to no check.
      if (context.find_symbol("c:@F@_ESBMC_array_length") == nullptr)
        return;
      side_effect_expr_function_callt length_call;
      get_library_function_call_no_args(
        "_ESBMC_array_length",
        "c:@F@_ESBMC_array_length",
        uint_type(),
        location,
        length_call);
      length_call.arguments().push_back(array_expr);
      // _ESBMC_array_length returns uint32; widen to uint256 for the compare.
      solidity_gen_typecast(ns, length_call, unsignedbv_typet(256));
      len_expr = length_call;
    }
    else
      return;

    exprt bounds_pos = pos;
    solidity_gen_typecast(ns, bounds_pos, len_expr.type());
    exprt in_bounds = binary_relation_exprt(bounds_pos, "<", len_expr);

    code_assertt bounds_assert(in_bounds);
    bounds_assert.location() = location;
    bounds_assert.location().comment(
      "dereference failure: array bounds violated");
    bounds_assert.location().property("array bounds");
    move_to_front_block(bounds_assert);
  };

  emit_pointer_array_bounds_check(array, base_t);

  // For mapping arrays (mapping(K=>V)[]), use the array's declared subtype
  // which has fully populated mapping subtypes, rather than `t` which lacks
  // them.  This ensures the result carries the inner mapping's element type
  // so that subsequent m[k] indexing works correctly.
  if (
    array.type().is_array() && array.type().get_bool("#sol_mapping_array") &&
    array.type().has_subtype())
  {
    new_expr = index_exprt(array, pos, array.type().subtype());
  }
  // T1.1 Stage S2: state-var dyn-array element access — replace the index
  // with a hash-fold of (this->$address, pos) so two `new C()` instances
  // resolve to disjoint SMT-array slots.
  else if (array.is_symbol() && array.type().get_bool("#sol_dynarray_state"))
  {
    // --bounds-check (Solidity opt-in): a dyn-array state var is modelled with
    // an infinity-sized element buffer, so goto-symex's built-in dereference
    // bounds check never fires. Emit an explicit `pos < <arr>_dynarray_len`
    // claim so an out-of-bounds access is caught. This lowering serves both a
    // read and an assignment LHS, so one check covers both.
    // The length companion is only created for a direct state-var dyn-array
    // (solidity_convert_decl.cpp `is_dynarray_state` block); some paths set the
    // `#sol_dynarray_state` type flag without a `_dynarray_len` symbol (e.g. an
    // early return on an already-registered symbol). If it is absent, skip the
    // check rather than assert — degrade to the prior (no-bounds) behaviour.
    const symbolt *len_sym = nullptr;
    if (
      config.options.get_bool_option("bounds-check") &&
      !config.options.get_bool_option("no-bounds-check"))
      len_sym = ns.lookup(array.identifier().as_string() + "_dynarray_len");
    if (len_sym)
    {
      exprt len_ref;
      if (get_dynarr_len_ref(*len_sym, len_ref))
        return true;

      exprt bounds_pos = pos;
      solidity_gen_typecast(ns, bounds_pos, len_ref.type());
      exprt in_bounds = binary_relation_exprt(bounds_pos, "<", len_ref);

      code_assertt bounds_assert(in_bounds);
      bounds_assert.location() = location;
      bounds_assert.location().comment(
        "dereference failure: array bounds violated");
      bounds_assert.location().property("array bounds");
      move_to_front_block(bounds_assert);
    }

    exprt fold_idx;
    if (get_dynarr_elem_idx(pos, fold_idx))
      return true;
    new_expr = index_exprt(array, fold_idx, t);
  }
  else
    new_expr = index_exprt(array, pos, t);

  return false;
}

bool solidity_convertert::get_index_range_access_expr(
  const nlohmann::json &expr,
  const nlohmann::json &literal_type,
  exprt &new_expr)
{
  // [APPROX: OVER] IndexRangeAccess: data[start:end] on calldata
  // arrays/bytes. The slice value itself is still a fresh nondet
  // (no link to parent array content), but the path-condition is
  // tightened with real-EVM bounds:
  //   __ESBMC_assume(s <= e);
  //   __ESBMC_assume(e <= base.length);
  // Real EVM reverts when these are violated, so a feasible slice
  // implies the bounds. This closes ledger #14's slice-bounds gap;
  // the same-content-as-parent gap remains separately deferred.
  //
  // False positives: same-content slice assertions cannot be verified
  //   (slice value is still detached nondet).
  // False negatives: none for safety.
  locationt location;
  get_start_location_from_stmt(expr, location);

  typet t;
  if (get_type_description(expr["typeDescriptions"], t))
    return true;

  // Resolve start/end. Solidity allows omission: `b[:e]` (s == 0) and
  // `b[s:]` (e == base.length).
  exprt s_expr;
  bool have_s =
    expr.contains("startExpression") && !expr["startExpression"].is_null();
  if (have_s)
  {
    if (get_expr(
          expr["startExpression"],
          expr["startExpression"]["typeDescriptions"],
          s_expr))
      return true;
  }
  else
    s_expr = from_integer(BigInt(0), unsignedbv_typet(256));

  exprt e_expr;
  bool have_e =
    expr.contains("endExpression") && !expr["endExpression"].is_null();
  if (have_e)
  {
    if (get_expr(
          expr["endExpression"],
          expr["endExpression"]["typeDescriptions"],
          e_expr))
      return true;
  }

  // Try to read base.length when needed for the e <= length bound. For
  // t_bytes_calldata_ptr the BytesDynamic struct exposes `.length`; for
  // T[] calldata the array's length is reachable via the same name on
  // the C-level layout. If we can't resolve the base expression we
  // skip the length-side assume (still emit s <= e — a strict
  // improvement over no constraint).
  exprt base_length;
  bool have_base_length = false;
  if (expr.contains("baseExpression"))
  {
    exprt base_expr;
    if (!get_expr(
          expr["baseExpression"],
          expr["baseExpression"]["typeDescriptions"],
          base_expr))
    {
      const std::string base_tid =
        expr["baseExpression"]["typeDescriptions"].value("typeIdentifier", "");
      // bytes calldata / memory: BytesDynamic has a `length` field.
      if (base_tid.compare(0, 8, "t_bytes_") == 0)
      {
        base_length = member_exprt(base_expr, "length", size_type());
        solidity_gen_typecast(ns, base_length, unsignedbv_typet(256));
        have_base_length = true;
      }
      // T[] calldata / memory: similar path; member name 'length' on
      // the C-level struct that backs the dynamic array. Skip if the
      // member access fails at IR build time.
    }
  }
  if (!have_e && have_base_length)
  {
    e_expr = base_length;
    have_e = true;
  }

  auto emit_assume = [&](const exprt &cond) {
    side_effect_expr_function_callt assume_call;
    get_library_function_call_no_args(
      "__ESBMC_assume",
      "c:@F@__ESBMC_assume",
      empty_typet(),
      location,
      assume_call);
    assume_call.arguments().push_back(cond);
    convert_expression_to_code(assume_call);
    move_to_front_block(assume_call);
  };

  // Emit `s <= e` whenever both operands are known.
  if (have_e)
  {
    // Coerce both to a common 256-bit width for the comparison.
    exprt s_for_cmp = s_expr;
    exprt e_for_cmp = e_expr;
    solidity_gen_typecast(ns, s_for_cmp, unsignedbv_typet(256));
    solidity_gen_typecast(ns, e_for_cmp, unsignedbv_typet(256));
    binary_relation_exprt s_le_e(s_for_cmp, "<=", e_for_cmp);
    emit_assume(s_le_e);
  }

  // Emit `e <= base.length` when length is resolvable.
  if (have_e && have_base_length)
  {
    exprt e_for_cmp = e_expr;
    exprt len_for_cmp = base_length;
    solidity_gen_typecast(ns, e_for_cmp, unsignedbv_typet(256));
    solidity_gen_typecast(ns, len_for_cmp, unsignedbv_typet(256));
    binary_relation_exprt e_le_len(e_for_cmp, "<=", len_for_cmp);
    emit_assume(e_le_len);
  }

  // The slice value itself is still a fresh nondet of the result type.
  new_expr = exprt("sideeffect", t);
  new_expr.statement("nondet");
  new_expr.location() = location;

  return false;
}

bool solidity_convertert::get_new_object_expr(
  const nlohmann::json &expr,
  const nlohmann::json &literal_type,
  exprt &new_expr)
{
  locationt location;
  get_start_location_from_stmt(expr, location);

  // 1. new dynamic array, e.g.
  //    uint[] memory a = new uint[](7);
  //    uint[] memory a = new uint[](len);
  // 2. new bytes array e.g.
  //    bytes memory b = new bytes(7)
  // 3. new object, e.g.
  //    Base x = new Base(1, 2);
  // 4. new object with options, e.g.
  //    Base x = new Base{value: 1 ether}(1, 2);
  nlohmann::json callee_expr_json;
  if (
    expr.contains("expression") &&
    expr["expression"]["nodeType"] == "FunctionCallOptions")
  {
    callee_expr_json = expr["expression"]["expression"];
  }
  else
  {
    callee_expr_json = expr["expression"];
  }
  // nlohmann::json callee_expr_json = expr["expression"];
  if (callee_expr_json.contains("typeName"))
  {
    // case 1
    // e.g.
    //    new uint[](7)
    // convert to
    //    uint y[7] = {0,0,0,0,0,0,0};
    if (is_dyn_array(callee_expr_json["typeName"]))
    {
      if (get_empty_array_ref(expr, new_expr))
        return true;
      return false;
    }
    // case 2
    // the contract/constructor name cannot be "bytes"
    if (
      callee_expr_json["typeName"]["typeDescriptions"]["typeString"]
        .get<std::string>() == "bytes")
    {
      // populate 0x00 to bytes array
      // same process in case SolidityGrammar::ExpressionT::Literal
      assert(expr.contains("arguments") && expr["arguments"].size() == 1);
      exprt size_expr;
      if (get_expr(
            expr["arguments"][0],
            expr["expression"]["argumentTypes"][0],
            size_expr))
        return true;

      // Prepare function call: bytes_dynamic_init_zero(len, pool)
      side_effect_expr_function_callt call;
      get_library_function_call_no_args(
        "bytes_dynamic_init_zero",
        "c:@F@bytes_dynamic_init_zero",
        byte_dynamic_t,
        location,
        call);

      call.arguments().push_back(size_expr);

      member_exprt pool_member;
      if (get_dynamic_pool(expr, pool_member))
        return true;
      call.arguments().push_back(pool_member);

      // assert(b[0] ==  (new bytes(4))[0]);
      new_expr = make_aux_var(call, location);
      set_sol_type(new_expr.type(), SolidityGrammar::SolType::BYTES_DYN);
      return false;
    }
    // new string(N): allocate a char* buffer of size N. ESBMC's
    // Solidity string_t is `char*`, so malloc(N) cast to char* is the
    // right shape. The contents are left nondet — enough to let the
    // frontend not crash on contracts that use `new string(...)`.
    if (
      callee_expr_json["typeName"]["typeDescriptions"]["typeString"]
        .get<std::string>() == "string")
    {
      assert(expr.contains("arguments") && expr["arguments"].size() == 1);
      exprt size_expr;
      if (get_expr(
            expr["arguments"][0],
            expr["expression"]["argumentTypes"][0],
            size_expr))
        return true;

      side_effect_expr_function_callt mcall;
      get_malloc_function_call(location, mcall);
      solidity_gen_typecast(ns, size_expr, size_type());
      mcall.arguments().push_back(size_expr);

      exprt tc = typecast_exprt(mcall, string_t);
      new_expr = make_aux_var(tc, location);
      set_sol_type(new_expr.type(), SolidityGrammar::SolType::STRING);
      return false;
    }
  }
  // case 3
  // is equal to Base *x = new base(x);
  exprt call;
  if (get_new_object_ctor_call(expr, false, call))
    return true;

  new_expr = call;
  // check if the new expression has options
  // Options (e.g. {value: amount}) live in the FunctionCallOptions node,
  // which is expr["expression"] when present, not in the outer FunctionCall.
  const nlohmann::json &opts_src =
    (expr.contains("expression") &&
     expr["expression"]["nodeType"] == "FunctionCallOptions")
      ? expr["expression"]
      : expr;
  if (
    opts_src.contains("options") && opts_src.contains("names") &&
    !opts_src["options"].empty() && !opts_src["names"].empty())
  {
    const auto &options = opts_src["options"];
    const auto &names = opts_src["names"];

    for (size_t i = 0; i < options.size(); ++i)
    {
      const auto &opt = options[i];
      std::string opt_name = names[i];
      // model transaction when the option is "value"
      if (opt_name == "value")
      {
        exprt value_expr;
        // The option's own typeDescriptions is the right hint for
        // parsing its literal — argumentTypes[] on the surrounding
        // FunctionCallOptions reflects the OUTER FunctionCall's args
        // and is empty when the constructor takes no parameters
        // (e.g. `new C{value: x}()`), which would otherwise out-of-
        // range crash this branch.
        nlohmann::json val_type = opt.contains("typeDescriptions")
                                    ? opt["typeDescriptions"]
                                    : nlohmann::json(nullptr);
        if (get_expr(opt, val_type, value_expr))
          return true;

        exprt this_expr;
        if (current_functionDecl)
        {
          if (get_func_decl_this_ref(*current_functionDecl, this_expr))
            return true;
        }
        else
        {
          if (get_ctor_decl_this_ref(expr, this_expr))
            return true;
        }
        exprt front_block = code_blockt();
        exprt back_block = code_blockt();
        if (model_transaction(
              expr,
              this_expr,
              new_expr,
              value_expr,
              location,
              front_block,
              back_block))
          return true;

        // Remove the last front_block operand (base.$balance += value).
        // For contract creation, the recipient's $balance is initialized
        // to msg.value inside the payable constructor instead.
        if (!front_block.operands().empty())
          front_block.operands().pop_back();

        for (auto op : front_block.operands())
          move_to_front_block(op);
        for (auto op : back_block.operands())
          move_to_back_block(op);
        break;
      }
    }
  }

  // Emit _ESBMC_bind_cname assignment so that subsequent cross-contract
  // calls can dispatch to the correct singleton.  In --bound mode this
  // was always done; we now also emit it in unbound mode because
  // new-created instances are auto-bound (their calls should execute
  // the callee body, not return nondet).
  //
  // We write to BOTH the singleton's struct field (kept for the legacy
  // function-call dispatcher which reads the field via member_exprt) and
  // the per-pointer shadow symbol (used by the mapping-getter polymorphism
  // read path, which must disambiguate between structurally identical
  // contracts that share a singleton).  For `new`, both agree — the
  // declared type is the true binding.  For cast (`C(_addr)`) they can
  // diverge; see the TypeConversionExpression handler.
  {
    int ref_decl_id = callee_expr_json["typeName"]["referencedDeclaration"];
    const std::string contract_name = contractNamesMap[ref_decl_id];

    // Write singleton member (legacy dispatcher compatibility).
    exprt lhs;
    if (!get_bind_cname_expr(expr, lhs))
    {
      exprt rhs;
      get_cname_expr(contract_name, rhs);
      exprt _assign = side_effect_exprt("assign", lhs.type());
      solidity_gen_typecast(ns, rhs, lhs.type());
      _assign.operands().push_back(lhs);
      _assign.operands().push_back(rhs);
      convert_expression_to_code(_assign);
      move_to_back_block(_assign);
    }
    else
      return false; // no lvalue — skip both assignments

    // Also write to the per-pointer shadow.
    // Walk the same parent chain to recover the declared lvar.
    const nlohmann::json &parent = find_last_parent(src_ast_json, expr);
    exprt lvar;
    bool got_lvar = false;
    if (parent["nodeType"] == "VariableDeclarationStatement")
    {
      if (!get_var_decl_ref(parent["declarations"][0], true, lvar))
        got_lvar = true;
    }
    else if (parent["nodeType"] == "VariableDeclaration")
    {
      if (!get_var_decl_ref(parent, true, lvar))
        got_lvar = true;
    }
    else if (parent["nodeType"] == "Assignment")
    {
      if (!get_expr(parent["leftHandSide"], lvar))
        got_lvar = true;
    }
    if (got_lvar)
    {
      exprt shadow;
      if (!get_or_create_bind_shadow(lvar, contract_name, shadow))
      {
        exprt rhs;
        get_cname_expr(contract_name, rhs);
        solidity_gen_typecast(ns, rhs, shadow.type());
        exprt _assign = side_effect_exprt("assign", shadow.type());
        _assign.copy_to_operands(shadow, rhs);
        convert_expression_to_code(_assign);
        move_to_back_block(_assign);
      }
    }
  }

  return false;
}

// get the initial value for the variable declaration
bool solidity_convertert::get_init_expr(
  const nlohmann::json &init_value,
  const nlohmann::json &literal_type,
  const typet &dest_type,
  exprt &new_expr)
{
  if (literal_type == nullptr)
    return true;

  if (get_expr(init_value, literal_type, new_expr))
    return true;

  convert_type_expr(ns, new_expr, dest_type, init_value);
  return false;
}

// get the name of the contract that contains the target ast_node, including library
// note that the contract_name might be empty
void solidity_convertert::get_current_contract_name(
  const nlohmann::json &ast_node,
  std::string &contract_name)
{
  log_debug("solidity", "\tfinding current contract name");
  if (ast_node.is_null() || ast_node.empty())
  {
    log_debug("solidity", "got empty contract name");
    contract_name = "";
    return;
  }
  if (!ast_node.contains("id"))
  {
    // this could be manually created json.
    //TODO: avoid this kind of implementation
    if (ast_node.is_object() && ast_node["nodeType"] == "ImplicitCastExprClass")
    {
      get_current_contract_name(ast_node["subExpr"], contract_name);
    }
    else
    {
      log_warning("target node do not have id.");
      if (!ast_node.is_object())
        abort();
      log_status("{}", ast_node.dump());
    }
    return;
  }

  const auto &json = find_parent_contract(src_ast_json["nodes"], ast_node);
  if (json.empty() || json.is_null())
  {
    log_debug(
      "solidity",
      "failed to get current contract name, trying to "
      "find id {}, target json is \n{}\n",
      std::to_string(ast_node["id"].get<int>()),
      ast_node.dump());
    return;
  }

  assert(json.contains("name"));

  contract_name = json["name"].get<std::string>();
  log_debug("solidity", "\tcurrent contract name={}", contract_name);
}

bool solidity_convertert::get_binary_operator_expr(
  const nlohmann::json &expr,
  exprt &new_expr)
{
  // preliminary step for recursive BinaryOperation
  StackGuard<const nlohmann::json *> binop_guard(
    current_BinOp_type, &(expr["typeDescriptions"]));

  // 1. Convert LHS and RHS
  // For "Assignment" expression, it's called "leftHandSide" or "rightHandSide".
  // For "BinaryOperation" expression, it's called "leftExpression" or "leftExpression"
  exprt lhs, rhs;
  nlohmann::json rhs_json;
  locationt l;
  get_location_from_node(expr, l);

  if (expr.contains("leftHandSide"))
  {
    nlohmann::json literalType_l = expr["leftHandSide"]["typeDescriptions"];
    nlohmann::json literalType_r = expr["rightHandSide"]["typeDescriptions"];

    current_lhsDecl = true;
    if (get_expr(expr["leftHandSide"], literalType_l, lhs))
      return true;
    current_lhsDecl = false;

    current_rhsDecl = true;
    if (get_expr(expr["rightHandSide"], literalType_r, rhs))
      return true;
    current_rhsDecl = false;

    rhs_json = expr["rightHandSide"];
  }
  else if (expr.contains("leftExpression"))
  {
    nlohmann::json literalType_l = expr["leftExpression"]["typeDescriptions"];
    nlohmann::json literalType_r = expr["rightExpression"]["typeDescriptions"];

    current_lhsDecl = true;
    if (get_expr(expr["leftExpression"], literalType_l, lhs))
      return true;
    current_lhsDecl = false;

    current_rhsDecl = true;
    if (get_expr(expr["rightExpression"], literalType_r, rhs))
      return true;
    current_rhsDecl = false;

    rhs_json = expr["rightExpression"];
  }
  else
    assert(!"should not be here - unrecognized LHS and RHS keywords in expression JSON");

  // 2. Get type
  typet t;
  assert(current_BinOp_type.size());
  const nlohmann::json &binop_type = *(current_BinOp_type.top());
  if (get_type_description(binop_type, t))
    return true;

  typet common_type;
  if (expr.contains("commonType"))
  {
    if (get_type_description(expr["commonType"], common_type))
      return true;
  }

  typet lt = lhs.type();
  typet rt = rhs.type();
  SolidityGrammar::SolType lt_sol = get_sol_type(lt);
  SolidityGrammar::SolType rt_sol = get_sol_type(rt);

  // 2.1 special handling for the sol_unbound harness
  convert_unboundcall_nondet(lhs, common_type, l);
  convert_unboundcall_nondet(rhs, common_type, l);

  // 3. Convert opcode
  SolidityGrammar::ExpressionT opcode =
    SolidityGrammar::get_expr_operator_t(expr);
  log_debug(
    "solidity",
    "	@@@ got binop.getOpcode: SolidityGrammar::{}",
    SolidityGrammar::expression_to_str(opcode));

  if (is_byte_type(lhs.type()) || is_byte_type(rhs.type()))
  {
    log_debug("solidity", "\t\tHandling BYTES/BYTESN operators");

    bool is_static = is_bytesN_type(lt) && is_bytesN_type(rt);
    bool is_dynamic = is_bytes_type(lt) && is_bytes_type(rt);

    switch (opcode)
    {
    case SolidityGrammar::ExpressionT::BO_EQ:
    case SolidityGrammar::ExpressionT::BO_NE:
    {
      side_effect_expr_function_callt call_expr;
      std::string fname, fid;
      if (is_static)
      {
        fname = "bytes_static_equal";
        fid = "c:@F@bytes_static_equal";

        // Solidity 0.5.x permits direct `bytesN == bytesM` (N != M) via
        // implicit conversion of the narrower operand to the wider type
        // (right-padding zeros). bytes_static_equal returns false on
        // length mismatch, so we must widen the narrower side here. The
        // common_type computed by solc on the BinaryOperation is the
        // wider bytesN — convert each side that differs from it.
        const std::string lt_sz =
          lhs.type().get("#sol_bytesn_size").as_string();
        const std::string rt_sz =
          rhs.type().get("#sol_bytesn_size").as_string();
        if (!lt_sz.empty() && !rt_sz.empty() && lt_sz != rt_sz)
        {
          if (
            lhs.type().get("#sol_bytesn_size") !=
            common_type.get("#sol_bytesn_size"))
            convert_type_expr(ns, lhs, common_type, expr);
          if (
            rhs.type().get("#sol_bytesn_size") !=
            common_type.get("#sol_bytesn_size"))
            convert_type_expr(ns, rhs, common_type, expr);
        }
      }
      else if (is_dynamic)
      {
        fname = "bytes_dynamic_equal";
        fid = "c:@F@bytes_dynamic_equal";
      }
      else
      {
        assert(common_type.id() != "");
        // try to convert non-bytes operand to matching type
        // e.g. data2 == 0x00746573
        if (!is_byte_type(rhs.type()))
        {
          current_rhsDecl = true;
          if (get_expr(expr["rightExpression"], expr["commonType"], rhs))
            return true;
          current_rhsDecl = false;
          convert_type_expr(ns, rhs, common_type, expr);
        }
        else
        {
          current_rhsDecl = true;
          if (get_expr(expr["leftExpression"], expr["commonType"], lhs))
            return true;
          current_rhsDecl = false;
          convert_type_expr(ns, lhs, common_type, expr);
        }

        lt_sol = get_sol_type(lhs.type());
        rt_sol = get_sol_type(rhs.type());
        is_static = is_bytesN_type(lhs.type()) && is_bytesN_type(rhs.type());
        is_dynamic = is_bytes_type(lhs.type()) && is_bytes_type(rhs.type());

        if (is_static)
        {
          fname = "bytes_static_equal";
          fid = "c:@F@bytes_static_equal";
        }
        else if (is_dynamic)
        {
          fname = "bytes_dynamic_equal";
          fid = "c:@F@bytes_dynamic_equal";
        }
        else
        {
          log_error(
            "Incompatible bytes comparison: {} vs {}",
            SolidityGrammar::sol_type_to_str(lt_sol),
            SolidityGrammar::sol_type_to_str(rt_sol));
          return true;
        }
      }

      get_library_function_call_no_args(fname, fid, bool_t, l, call_expr);

      exprt lhs_tmp = make_aux_var(lhs, l);
      exprt rhs_tmp = make_aux_var(rhs, l);

      call_expr.arguments().push_back(address_of_exprt(lhs_tmp));
      call_expr.arguments().push_back(address_of_exprt(rhs_tmp));

      if (is_dynamic)
      {
        exprt pool_member;
        if (get_dynamic_pool(expr, pool_member))
          return true;
        call_expr.arguments().push_back(pool_member);
      }

      if (opcode == SolidityGrammar::ExpressionT::BO_EQ)
        new_expr = call_expr;
      else
        new_expr = not_exprt(call_expr);
      new_expr.location() = l;
      return false;
    }

    case SolidityGrammar::ExpressionT::BO_Shl:
    case SolidityGrammar::ExpressionT::BO_Shr:
    {
      if (!is_bytesN_type(lt))
      {
        log_error(
          "Shift operations only supported on bytesN types, got {}",
          SolidityGrammar::sol_type_to_str(lt_sol));
        return true;
      }

      std::string fname = (opcode == SolidityGrammar::ExpressionT::BO_Shl)
                            ? "bytes_static_shl"
                            : "bytes_static_shr";

      side_effect_expr_function_callt call_expr;
      get_library_function_call_no_args(
        fname, "c:@F@" + fname, lhs.type(), l, call_expr);

      exprt lhs_tmp = make_aux_var(lhs, l);
      call_expr.arguments().push_back(address_of_exprt(lhs_tmp));
      call_expr.arguments().push_back(rhs);

      new_expr = call_expr;
      new_expr.location() = l;
      return false;
    }

    case SolidityGrammar::ExpressionT::BO_And:
    case SolidityGrammar::ExpressionT::BO_Or:
    case SolidityGrammar::ExpressionT::BO_Xor:
    {
      if (!is_static)
      {
        log_error("Bitwise operations only supported for static bytesN");
        return true;
      }

      std::string fname, fid;
      if (opcode == SolidityGrammar::ExpressionT::BO_And)
      {
        fname = "bytes_static_and";
        fid = "c:@F@bytes_static_and";
      }
      else if (opcode == SolidityGrammar::ExpressionT::BO_Or)
      {
        fname = "bytes_static_or";
        fid = "c:@F@bytes_static_or";
      }
      else
      {
        fname = "bytes_static_xor";
        fid = "c:@F@bytes_static_xor";
      }

      side_effect_expr_function_callt call_expr;
      get_library_function_call_no_args(fname, fid, lhs.type(), l, call_expr);

      exprt lhs_tmp = make_aux_var(lhs, l);
      exprt rhs_tmp = make_aux_var(rhs, l);

      call_expr.arguments().push_back(address_of_exprt(lhs_tmp));
      call_expr.arguments().push_back(address_of_exprt(rhs_tmp));

      new_expr = call_expr;
      new_expr.location() = l;
      return false;
    }
    case SolidityGrammar::ExpressionT::BO_Assign:
    {
      // data2 = 0x0074657374;
      if (!is_byte_type(rhs.type()))
      {
        auto l_json = expr.contains("commonType") ? expr["commonType"]
                                                  : expr["typeDescriptions"];
        // redo get expr
        if (get_expr(expr["rightHandSide"], l_json, rhs))
          return true;
      }
      break;
    }
    default:
      break;
    }
  }

  switch (opcode)
  {
  case SolidityGrammar::ExpressionT::BO_Assign:
  {
    // Nested tuple assignment: ((a,b), c) = (f(), 3)
    // Detect by checking if the LHS TupleExpression contains nested TupleExpressions
    if (
      expr.contains("leftHandSide") &&
      expr["leftHandSide"].value("nodeType", "") == "TupleExpression" &&
      expr["leftHandSide"].contains("components"))
    {
      bool has_nested = false;
      for (const auto &comp : expr["leftHandSide"]["components"])
      {
        if (!comp.is_null() && comp.value("nodeType", "") == "TupleExpression")
        {
          has_nested = true;
          break;
        }
      }
      if (has_nested)
      {
        if (flatten_nested_tuple_assignment(
              expr, expr["leftHandSide"], expr["rightHandSide"]))
          return true;
        new_expr = code_skipt();
        return false;
      }
    }

    // Standard (non-nested) tuple assignment.
    // Also dispatch when the LHS is a code_blockt (the marker the
    // tuple-LHS path produces in get_tuple_expr) but the RHS is no longer
    // a tuple struct — convert_unboundcall_nondet may have rewritten an
    // external-call RHS into a plain `sideeffect/nondet` of the binop
    // common type, erasing the TUPLE_RETURNS tag. construct_tuple_assigments
    // has a non-struct sideeffect fallback that assigns an independent
    // nondet to each LHS slot.
    const bool lhs_is_tuple_block =
      lhs.is_code() && to_code(lhs).statement() == "block";
    if (
      rt_sol == SolidityGrammar::SolType::TUPLE_INSTANCE ||
      rt_sol == SolidityGrammar::SolType::TUPLE_RETURNS || lhs_is_tuple_block)
    {
      construct_tuple_assigments(expr, lhs, rhs);
      new_expr = code_skipt();
      return false;
    }
    else if (
      (rt_sol == SolidityGrammar::SolType::ARRAY ||
       rt_sol == SolidityGrammar::SolType::ARRAY_LITERAL) &&
      lhs.is_symbol() && lt.get_bool("#sol_dynarray_state"))
    {
      // Dynarray state var: element-wise assignment from array literal
      // e.g. items = [1,2,3] → items[0]=1; items[1]=2; items[2]=3; items_len=3
      typet elem_type = lt.subtype();
      const nlohmann::json &rhs_json = expr["rightHandSide"];
      unsigned count = 0;
      if (rhs_json.contains("components"))
      {
        for (unsigned i = 0; i < rhs_json["components"].size(); i++)
        {
          exprt val;
          if (get_expr(
                rhs_json["components"][i],
                rhs_json["components"][i]["typeDescriptions"],
                val))
            return true;
          solidity_gen_typecast(ns, val, elem_type);
          exprt idx = constant_exprt(
            integer2binary(i, bv_width(unsignedbv_typet(256))),
            std::to_string(i),
            unsignedbv_typet(256));
          // T1.1 Stage S2: fold the literal index by (this->$address, i)
          // so the write lands in the same slot the future read will hit.
          exprt fold_idx;
          if (get_dynarr_elem_idx(idx, fold_idx))
            return true;
          exprt elem_assign = side_effect_exprt("assign", elem_type);
          elem_assign.copy_to_operands(
            index_exprt(lhs, fold_idx, elem_type), val);
          convert_expression_to_code(elem_assign);
          move_to_front_block(elem_assign);
          count++;
        }
      }
      // Set length — T1.1 Stage S1: addr-keyed via get_dynarr_len_ref.
      std::string len_id = lhs.identifier().as_string() + "_dynarray_len";
      const symbolt *len_sym = ns.lookup(len_id);
      assert(len_sym);
      exprt len_ref;
      if (get_dynarr_len_ref(*len_sym, len_ref))
        return true;
      exprt count_expr = constant_exprt(
        integer2binary(count, bv_width(unsignedbv_typet(256))),
        std::to_string(count),
        unsignedbv_typet(256));
      exprt len_assign = side_effect_exprt("assign", unsignedbv_typet(256));
      len_assign.copy_to_operands(len_ref, count_expr);
      convert_expression_to_code(len_assign);
      move_to_front_block(len_assign);
      new_expr = code_skipt();
      return false;
    }
    else if (
      rt_sol == SolidityGrammar::SolType::ARRAY ||
      rt_sol == SolidityGrammar::SolType::ARRAY_LITERAL)
    {
      if (rt_sol == SolidityGrammar::SolType::ARRAY_LITERAL)
        // construct aux_array while adding padding
        // e.g. data1 = [1,2] ==> data1 = aux_array$1
        convert_type_expr(ns, rhs, lhs, expr);

      // get size
      exprt size_expr;
      get_size_expr(rhs, size_expr);

      // get sizeof
      exprt size_of_expr;
      // e.g. uint[] public tt; t = [1, 2, 3];
      // lt.subtype = uint256
      // rt.subtype = uint8
      get_size_of_expr(lt.subtype(), size_of_expr);

      // do array copy
      side_effect_expr_function_callt acpy_call;
      get_arrcpy_function_call(lhs.location(), acpy_call);

      acpy_call.arguments().push_back(rhs);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      solidity_gen_typecast(ns, acpy_call, lt);

      rhs = acpy_call;
    }
    else if (
      rt_sol == SolidityGrammar::SolType::DYNARRAY && lhs.is_symbol() &&
      lt.get_bool("#sol_dynarray_state"))
    {
      // Dynarray state var (infinite-size array): copy elements from
      // memory dynarray (pointer + inline header) into the infinite
      // array via a bounded for-loop, then update _dynarray_len.
      // The loop is bounded by --unwind (fundamental BMC limitation).
      typet elem_type = lt.subtype();
      locationt loc = lhs.location();
      std::string debug_modulename = get_modulename_from_path(absolute_path);

      // 1. _copy_len = _ESBMC_array_length(rhs)
      side_effect_expr_function_callt length_call;
      get_library_function_call_no_args(
        "_ESBMC_array_length",
        "c:@F@_ESBMC_array_length",
        uint_type(),
        loc,
        length_call);
      length_call.arguments().push_back(rhs);
      // _ESBMC_array_length returns uint32; cast to uint256 for consistency
      // with the loop counter and _dynarray_len.
      solidity_gen_typecast(ns, length_call, unsignedbv_typet(256));
      exprt len_var = make_aux_var(length_call, loc);

      // 2. Create loop counter: uint256 _i
      std::string ctr_name, ctr_id;
      get_aux_var(ctr_name, ctr_id);
      symbolt ctr_sym;
      get_default_symbol(
        ctr_sym,
        debug_modulename,
        unsignedbv_typet(256),
        ctr_name,
        ctr_id,
        loc);
      ctr_sym.lvalue = true;
      ctr_sym.file_local = true;
      ctr_sym.value = gen_zero(unsignedbv_typet(256));
      auto &added_ctr = *move_symbol_to_context(ctr_sym);
      exprt ctr_ref = symbol_expr(added_ctr);

      // Declare counter
      code_declt ctr_decl(ctr_ref);
      ctr_decl.operands().push_back(gen_zero(unsignedbv_typet(256)));
      move_to_front_block(ctr_decl);

      // init: _i = 0
      code_assignt init_assign(ctr_ref, gen_zero(unsignedbv_typet(256)));

      // cond: _i < _copy_len
      exprt cond = gen_binary("<", bool_typet(), ctr_ref, len_var);

      // iter: _i = _i + 1
      exprt one = constant_exprt(
        integer2binary(1, bv_width(unsignedbv_typet(256))),
        "1",
        unsignedbv_typet(256));
      code_assignt iter_assign(
        ctr_ref, gen_binary("+", unsignedbv_typet(256), ctr_ref, one));

      // body: data1[_i] = rhs[_i]
      // T1.1 Stage S2: lhs (state-var dyn-array) write must use the
      // addr-keyed fold so subsequent reads from the same instance hit
      // the same slot.  rhs is a memory dyn-array (heap-malloc, no
      // addr-keying), so its index stays as plain `ctr_ref`.
      exprt lhs_fold_idx;
      if (get_dynarr_elem_idx(ctr_ref, lhs_fold_idx))
        return true;
      exprt lhs_elem = index_exprt(lhs, lhs_fold_idx, elem_type);
      exprt rhs_elem = index_exprt(rhs, ctr_ref, elem_type);
      solidity_gen_typecast(ns, rhs_elem, elem_type);
      code_assignt body_assign(lhs_elem, rhs_elem);

      // Emit for-loop
      code_fort copy_loop;
      copy_loop.init() = init_assign;
      copy_loop.cond() = cond;
      copy_loop.iter() = iter_assign;
      copy_loop.body() = body_assign;
      move_to_front_block(copy_loop);

      // 3. _dynarray_len = _copy_len — T1.1 Stage S1: addr-keyed.
      std::string dlen_id = lhs.identifier().as_string() + "_dynarray_len";
      const symbolt *dlen_sym = ns.lookup(dlen_id);
      assert(dlen_sym);
      exprt dlen_ref;
      if (get_dynarr_len_ref(*dlen_sym, dlen_ref))
        return true;
      exprt dlen_assign = side_effect_exprt("assign", unsignedbv_typet(256));
      solidity_gen_typecast(ns, len_var, unsignedbv_typet(256));
      dlen_assign.copy_to_operands(dlen_ref, len_var);
      convert_expression_to_code(dlen_assign);
      move_to_front_block(dlen_assign);

      new_expr = code_skipt();
      return false;
    }
    else if (rt_sol == SolidityGrammar::SolType::DYNARRAY)
    {
      /* Dynarray-to-dynarray assignment.
       *
       * Storage target (data1 = ac): deep copy via _ESBMC_arrcpy so
       * the storage gets its own allocation independent of memory.
       *
       * Memory target (r.limbs = newLimbs): Solidity memory arrays
       * are reference types — assignment is just a pointer copy.
       * No arrcpy needed; fall through to plain assignment.  */
      bool lhs_is_state = lhs.is_symbol() && lt.get_bool("#sol_dynarray_state");
      if (lhs_is_state)
      {
        // get size
        exprt size_expr;
        get_size_expr(rhs, size_expr);

        // get sizeof
        exprt size_of_expr;
        get_size_of_expr(lt.subtype(), size_of_expr);

        // do array copy
        side_effect_expr_function_callt acpy_call;
        get_arrcpy_function_call(lhs.location(), acpy_call);
        acpy_call.arguments().push_back(rhs);
        acpy_call.arguments().push_back(size_expr);
        acpy_call.arguments().push_back(size_of_expr);
        solidity_gen_typecast(ns, acpy_call, lt);

        rhs = acpy_call;
      }
      else
      {
        // Memory-to-memory: just typecast and assign pointer directly
        solidity_gen_typecast(ns, rhs, lt);
      }
      // fall through to do assignment
    }
    else if (
      rt_sol == SolidityGrammar::SolType::ARRAY_CALLOC && lhs.is_symbol() &&
      lt.get_bool("#sol_dynarray_state"))
    {
      // Dynarray state var: `items = new uint[](n)` → just set length = n
      exprt size_expr;
      if (!rhs_json.contains("arguments"))
        abort();
      nlohmann::json callee_arg_json = rhs_json["arguments"][0];
      const nlohmann::json lit_type = callee_arg_json["typeDescriptions"];
      if (get_expr(callee_arg_json, lit_type, size_expr))
        return true;
      solidity_gen_typecast(ns, size_expr, unsignedbv_typet(256));

      // T1.1 Stage S1: addr-keyed via get_dynarr_len_ref.
      std::string len_id = lhs.identifier().as_string() + "_dynarray_len";
      const symbolt *len_sym = ns.lookup(len_id);
      assert(len_sym);
      exprt len_ref;
      if (get_dynarr_len_ref(*len_sym, len_ref))
        return true;
      exprt len_assign = side_effect_exprt("assign", unsignedbv_typet(256));
      len_assign.copy_to_operands(len_ref, size_expr);
      convert_expression_to_code(len_assign);
      move_to_front_block(len_assign);
      new_expr = code_skipt();
      return false;
    }
    else if (rt_sol == SolidityGrammar::SolType::ARRAY_CALLOC)
    {
      /* e.g.
        int[] memory ac;
        ac = new int[](10);

       _ESBMC_alloc_array already allocated the array with an inline
       header.  No arrcpy needed — just assign the pointer directly.
       The _ESBMC_store_array call (emitted separately by the frontend)
       updates the header if the count differs. */
      solidity_gen_typecast(ns, rhs, lt);
      // fall through to do plain pointer assignment
    }
    else if (lt_sol == SolidityGrammar::SolType::STRING)
    {
      get_string_assignment(lhs, rhs, new_expr);
      return false;
    }

    new_expr = side_effect_exprt("assign", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Add:
  {
    if (t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("+", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Sub:
  {
    if (t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("-", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Mul:
  {
    if (t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("*", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Div:
  {
    if (t.is_floatbv())
      assert(!"Solidity does not support FP arithmetic as of v0.8.6.");
    else
      new_expr = exprt("/", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Rem:
  {
    new_expr = exprt("mod", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Shl:
  {
    new_expr = exprt("shl", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Shr:
  {
    new_expr = exprt("shr", t);
    break;
  }
  case SolidityGrammar::BO_And:
  {
    new_expr = exprt("bitand", t);
    break;
  }
  case SolidityGrammar::BO_Xor:
  {
    new_expr = exprt("bitxor", t);
    break;
  }
  case SolidityGrammar::BO_Or:
  {
    new_expr = exprt("bitor", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_GT:
  {
    new_expr = exprt(">", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LT:
  {
    new_expr = exprt("<", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_GE:
  {
    new_expr = exprt(">=", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LE:
  {
    new_expr = exprt("<=", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_NE:
  {
    new_expr = exprt("notequal", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_EQ:
  {
    new_expr = exprt("=", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LAnd:
  {
    new_expr = exprt("and", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_LOr:
  {
    new_expr = exprt("or", t);
    break;
  }
  case SolidityGrammar::ExpressionT::BO_Pow:
  {
    // lhs**rhs => pow(lhs, rhs)

    // optimization: if both base and exponent are constant, use bigint::power
    exprt new_lhs = lhs;
    exprt new_rhs = rhs;
    // remove typecast
    while (new_lhs.id() == "typecast")
      new_lhs = new_lhs.op0();
    while (new_rhs.id() == "typecast")
      new_rhs = new_rhs.op0();
    if (new_lhs.is_constant() && new_rhs.is_constant())
    {
      //? it seems the solc cannot generate ast_json for constant power like 2**20
      BigInt base;
      if (to_integer(new_lhs, base))
      {
        log_error("failed to convert constant: {}", new_lhs.pretty());
        abort();
      }

      BigInt exp;
      if (to_integer(new_rhs, exp))
      {
        log_error("failed to convert constant: {}", new_rhs.pretty());
        abort();
      }

      BigInt res = ::power(base, exp);
      exprt tmp = from_integer(res, unsignedbv_typet(256));

      if (tmp.is_nil())
        return true;

      new_expr.swap(tmp);
    }
    else
    {
      // Use integer power function to avoid fixedbv/floatbv type mismatch.
      // Solidity ** is purely integer arithmetic.
      unsignedbv_typet u256(256);
      side_effect_expr_function_callt call_expr;
      get_library_function_call_no_args(
        "sol_pow_uint", "c:@F@sol_pow_uint", u256, lhs.location(), call_expr);

      call_expr.arguments().push_back(typecast_exprt(lhs, u256));
      call_expr.arguments().push_back(typecast_exprt(rhs, u256));

      new_expr = call_expr;
    }
    new_expr.location() = l;
    // do not fall through
    return false;
  }
  default:
  {
    if (get_compound_assign_expr(expr, lhs, rhs, common_type, new_expr))
    {
      log_error("Unimplemented binary operator");
      return true;
    }

    return false;
  }
  }

  // 4.1 check if it needs implicit type conversion
  if (common_type.id() != "")
  {
    convert_type_expr(ns, lhs, common_type, expr);
    convert_type_expr(ns, rhs, common_type, expr);
  }
  else if (lhs.type() != rhs.type())
    convert_type_expr(ns, rhs, lhs, expr);

  // 4.2 Copy to operands
  new_expr.copy_to_operands(lhs, rhs);

  return false;
}

bool solidity_convertert::get_compound_assign_expr(
  const nlohmann::json &expr,
  exprt &lhs,
  exprt &rhs,
  typet &common_type,
  exprt &new_expr)
{
  // equivalent to clang_c_convertert::get_compound_assign_expr
  SolidityGrammar::ExpressionT opcode =
    SolidityGrammar::get_expr_operator_t(expr);

  locationt location;
  get_location_from_node(expr, location);

  typet lt = lhs.type();

  if (is_bytesN_type(lt))
  {
    std::string fname;
    switch (opcode)
    {
    case SolidityGrammar::ExpressionT::BO_ShlAssign:
      fname = "bytes_static_shl";
      break;
    case SolidityGrammar::ExpressionT::BO_ShrAssign:
      fname = "bytes_static_shr";
      break;
    case SolidityGrammar::ExpressionT::BO_AndAssign:
      fname = "bytes_static_and";
      break;
    case SolidityGrammar::ExpressionT::BO_OrAssign:
      fname = "bytes_static_or";
      break;
    case SolidityGrammar::ExpressionT::BO_XorAssign:
      fname = "bytes_static_xor";
      break;
    default:
      log_error("Unsupported compound op for bytesN");
      return true;
    }

    side_effect_expr_function_callt call_expr;
    get_library_function_call_no_args(
      fname, "c:@F@" + fname, lt, location, call_expr);

    exprt lhs_tmp = make_aux_var(lhs, location);
    call_expr.arguments().push_back(address_of_exprt(lhs_tmp));

    if (
      opcode == SolidityGrammar::ExpressionT::BO_ShlAssign ||
      opcode == SolidityGrammar::ExpressionT::BO_ShrAssign)
    {
      call_expr.arguments().push_back(rhs);
    }
    else
    {
      exprt rhs_tmp = make_aux_var(rhs, location);
      call_expr.arguments().push_back(address_of_exprt(rhs_tmp));
    }

    code_assignt assign(lhs, call_expr);
    assign.location() = location;
    new_expr = assign;
    return false;
  }

  switch (opcode)
  {
  case SolidityGrammar::ExpressionT::BO_AddAssign:
  {
    new_expr = side_effect_exprt("assign+");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_SubAssign:
  {
    new_expr = side_effect_exprt("assign-");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_MulAssign:
  {
    new_expr = side_effect_exprt("assign*");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_DivAssign:
  {
    new_expr = side_effect_exprt("assign_div");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_RemAssign:
  {
    new_expr = side_effect_exprt("assign_mod");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_ShlAssign:
  {
    new_expr = side_effect_exprt("assign_shl");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_ShrAssign:
  {
    new_expr = side_effect_exprt("assign_shr");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_AndAssign:
  {
    new_expr = side_effect_exprt("assign_bitand");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_XorAssign:
  {
    new_expr = side_effect_exprt("assign_bitxor");
    break;
  }
  case SolidityGrammar::ExpressionT::BO_OrAssign:
  {
    new_expr = side_effect_exprt("assign_bitor");
    break;
  }
  default:
    log_error("Unimplemented compound assignment operator");
    return true;
  }

  if (common_type.id() != "")
  {
    convert_type_expr(ns, lhs, common_type, expr);
    convert_type_expr(ns, rhs, common_type, expr);
  }
  else if (lhs.type() != rhs.type())
    convert_type_expr(ns, rhs, lhs, expr);

  new_expr.copy_to_operands(lhs, rhs);
  new_expr.location() = location;
  return false;
}

bool solidity_convertert::get_unary_operator_expr(
  const nlohmann::json &expr,
  const nlohmann::json &literal_type,
  exprt &new_expr)
{
  // 1. get UnaryOperation opcode
  SolidityGrammar::ExpressionT opcode =
    SolidityGrammar::get_unary_expr_operator_t(expr, expr["prefix"]);
  log_debug(
    "solidity",
    "	@@@ got uniop.getOpcode: SolidityGrammar::{}",
    SolidityGrammar::expression_to_str(opcode));

  // delete-correctness plan (S1): recursive `delete` lowering.  See
  // `emit_delete_block` docstring in solidity_convert.h.  Closes four
  // empirically-confirmed Solidity-spec deviations regression-locked by
  // `delete_dyn_array_length_pass_knownbug`,
  // `delete_storage_alias_length_pass_knownbug`,
  // `delete_fixed_array_elements_pass_knownbug`,
  // `delete_struct_with_fixed_array_pass_knownbug`,
  // `delete_nested_struct_pass_knownbug`,
  // `delete_bytes_array_pass_knownbug`.
  if (opcode == SolidityGrammar::ExpressionT::UO_Delete)
  {
    exprt unary_sub;
    if (get_expr(expr["subExpression"], literal_type, unary_sub))
      return true;

    std::vector<exprt> assigns;
    if (emit_delete_block(unary_sub, unary_sub.type(), assigns))
      return true;

    // No-op (e.g. struct of only mapping fields after the recursive walk
    // skipped them per spec).  Emit a self-assign so the surrounding
    // statement-conversion stays well-formed.
    if (assigns.empty())
    {
      new_expr = side_effect_exprt("assign", unary_sub.type());
      new_expr.operands().push_back(unary_sub);
      new_expr.operands().push_back(unary_sub);
      return false;
    }

    // Push N-1 assigns to the front-block; return the last as the
    // surface expression so the caller's expression-to-statement wrapper
    // picks it up unchanged.
    for (size_t i = 0; i + 1 < assigns.size(); ++i)
      move_to_front_block(assigns[i]);
    new_expr = assigns.back();
    return false;
  }

  // 2. get type
  typet uniop_type;
  if (get_type_description(expr["typeDescriptions"], uniop_type))
    return true;

  // 3. get subexpr
  exprt unary_sub;
  if (get_expr(expr["subExpression"], literal_type, unary_sub))
    return true;

  switch (opcode)
  {
  case SolidityGrammar::ExpressionT::UO_PreDec:
  {
    new_expr = side_effect_exprt("predecrement", uniop_type);
    break;
  }
  case SolidityGrammar::ExpressionT::UO_PreInc:
  {
    new_expr = side_effect_exprt("preincrement", uniop_type);
    break;
  }
  case SolidityGrammar::UO_PostDec:
  {
    new_expr = side_effect_exprt("postdecrement", uniop_type);
    break;
  }
  case SolidityGrammar::UO_PostInc:
  {
    new_expr = side_effect_exprt("postincrement", uniop_type);
    break;
  }
  case SolidityGrammar::ExpressionT::UO_Minus:
  {
    new_expr = exprt("unary-", uniop_type);
    break;
  }
  case SolidityGrammar::ExpressionT::UO_Not:
  {
    new_expr = exprt("bitnot", uniop_type);
    break;
  }

  case SolidityGrammar::ExpressionT::UO_LNot:
  {
    new_expr = exprt("not", bool_t);
    break;
  }
  default:
  {
    log_error("Unimplemented unary operator");
    return true;
  }
  }

  new_expr.operands().push_back(unary_sub);
  return false;
}

// delete-correctness plan (S1).  Recursive `delete` lowering — see
// solidity_convert.h `emit_delete_block` declaration for the contract.
//
// Per Solidity spec:
//  - primitives → 0 / false / address(0)
//  - fixed-size arrays → element-by-element reset (length stays)
//  - dynamic arrays → length := 0 (data implicitly empty)
//  - structs → field-by-field, EXCEPT mapping members preserved
//  - mappings (inside structs) → preserved; standalone `delete map` is
//    rejected by solc itself
//  - bytes / string → length := 0 with `initialized=1` so downstream
//    push/copy paths don't trip init checks
//
// Handles four type-representation quirks the naive `gen_zero(t)` can't:
//  1. State-var dyn-arrays carry a separate `<arr>_dynarray_len[$address]`
//     companion (per-instance addr-keyed since T1.1).  The data-array
//     reset alone leaves the length at its pre-delete value.
//  2. State-var fixed arrays (`uint[N]`) lower to heap-pointer-backed
//     `pointer_typet(elem)` with `#sol_array_size=N`.  `gen_zero(pointer)`
//     returns NULL — assigning that clobbers the pointer instead of
//     element-zeroing through it.
//  3. Nested struct fields are stored as `symbol_typet("tag-Inner")`.
//     `gen_zero(symbol)` returns nil (no case in expr_util.cpp), so the
//     resulting struct constant has nil components and symex crashes.
//  4. Symbol-typed array element types (e.g. `BytesDynamic` inside
//     `bytes[]`) — same root cause as (3) — produces `ARRAY_OF(nil)`.
bool solidity_convertert::emit_delete_block(
  const exprt &lhs,
  const typet &type,
  std::vector<exprt> &assigns)
{
  // Resolve symbol-typed wrappers via ns to inspect the underlying type.
  // The lhs keeps the original (possibly symbol) type for assignment
  // compatibility; only `t` is dereferenced.
  typet t = type;
  if (t.id() == "symbol")
  {
    const symbolt *s = ns.lookup(t.identifier());
    if (s)
      t = s->type;
  }

  // Mapping placeholder (`_ESBMC_Mapping`): standalone `delete` is
  // rejected by solc; in struct recursion this branch is unreachable
  // (caller skips mapping fields).  Defensive no-op.
  if (t.id() == "struct" && t.tag().as_string() == "_ESBMC_Mapping")
    return false;

  // BytesDynamic: explicit field-level reset preserving `initialized=1`.
  // `gen_zero` of the struct would also zero `initialized`, which while
  // empirically harmless for tested patterns may trip future stricter
  // init checks — locked by `delete_bytes_clear_then_push_pass`.
  if (t.id() == "struct" && t.tag().as_string() == "BytesDynamic")
  {
    typet sz_t = size_type();
    typet int_t = int_type();
    side_effect_exprt a_off("assign", sz_t);
    a_off.copy_to_operands(member_exprt(lhs, "offset", sz_t), gen_zero(sz_t));
    assigns.push_back(a_off);

    side_effect_exprt a_len("assign", sz_t);
    a_len.copy_to_operands(member_exprt(lhs, "length", sz_t), gen_zero(sz_t));
    assigns.push_back(a_len);

    side_effect_exprt a_cap("assign", sz_t);
    a_cap.copy_to_operands(member_exprt(lhs, "capacity", sz_t), gen_zero(sz_t));
    assigns.push_back(a_cap);

    side_effect_exprt a_init("assign", int_t);
    a_init.copy_to_operands(
      member_exprt(lhs, "initialized", int_t), gen_one(int_t));
    assigns.push_back(a_init);
    return false;
  }

  // Heap-pointer-backed fixed array (state-var `uint[N]`): element-zero
  // through the pointer.  Detected via `#sol_array_size` annotation
  // preserved on the pointer type by the ctor walker (see
  // solidity_convert_constructor.cpp:1207, 1297, 1521).
  if (t.id() == "pointer" && !type.get("#sol_array_size").empty())
  {
    const std::string sz_str = type.get("#sol_array_size").as_string();
    BigInt N = string2integer(sz_str);
    typet elem_t = t.subtype();
    for (uint64_t i = 0; i < N.to_uint64(); ++i)
    {
      exprt idx = constant_exprt(
        integer2binary(BigInt(i), bv_width(int_type())),
        integer2string(BigInt(i)),
        int_type());
      exprt elem_lhs = index_exprt(lhs, idx, elem_t);
      if (emit_delete_block(elem_lhs, elem_t, assigns))
        return true;
    }
    return false;
  }

  // State-var dynamic array (carries `#sol_dynarray_state` flag set in
  // solidity_convert_decl.cpp).  Per Solidity spec, `delete arr` for
  // `T[]` resets length to 0 — that alone makes the array logically
  // empty since out-of-bounds element access reverts.  The underlying
  // SMT data array isn't reset; reads past length are OOB-checked by
  // the existing _ESBMC_array bounds machinery.  Skipping the data
  // write also avoids the `array_of(partial_struct)` symex crash that
  // affected `bytes[]` (Bug D root: gen_zero on BytesDynamic struct
  // member fields returned partially-nil constants).
  if (t.id() == "array" && type.get_bool("#sol_dynarray_state"))
  {
    // Reset length: `<arr>_dynarray_len[this->$address] := 0`.  Mirrors
    // the read path in solidity_convert_ref.cpp:712-718 and the push
    // path at 826-832.  Only fires when lhs is a direct symbol (state
    // var); nested struct-member dyn arrays use the legacy
    // `_ESBMC_array_*` model and don't have a `_dynarray_len` companion.
    if (lhs.is_symbol())
    {
      std::string len_id = lhs.identifier().as_string() + "_dynarray_len";
      const symbolt *len_sym = ns.lookup(len_id);
      if (len_sym)
      {
        exprt len_ref;
        if (!get_dynarr_len_ref(*len_sym, len_ref))
        {
          side_effect_exprt assign_len("assign", len_ref.type());
          assign_len.copy_to_operands(len_ref, gen_zero(len_ref.type()));
          assigns.push_back(assign_len);
        }
      }
    }
    return false;
  }

  // Generic struct: recurse per-component, skipping mapping fields per
  // Solidity spec.  Handles q5 (nested-struct crash) — recursion fully
  // resolves nested symbol-typed components instead of leaving nil.
  if (t.id() == "struct")
  {
    const struct_typet &st = to_struct_type(t);
    for (const auto &comp : st.components())
    {
      // Skip mapping placeholder fields — both inline-struct form and
      // symbol-wrapped form.  Per Solidity spec mappings inside structs
      // are preserved by `delete struct`.
      bool is_mapping_field = false;
      const typet &ct = comp.type();
      if (ct.id() == "struct" && ct.tag().as_string() == "_ESBMC_Mapping")
        is_mapping_field = true;
      else if (ct.id() == "symbol")
      {
        const symbolt *cs = ns.lookup(ct.identifier());
        if (
          cs && cs->type.id() == "struct" &&
          cs->type.tag().as_string() == "_ESBMC_Mapping")
          is_mapping_field = true;
      }
      if (is_mapping_field)
        continue;
      // Skip compiler-internal padding fields (anonymous).
      if (comp.name().empty())
        continue;
      exprt field = member_exprt(lhs, comp.name(), comp.type());
      if (emit_delete_block(field, comp.type(), assigns))
        return true;
    }
    return false;
  }

  // Default: scalar / unannotated pointer / anything else — single
  // assign with gen_zero.  Unchanged from pre-S1 behaviour.
  exprt zero = gen_zero(t);
  if (zero.is_nil())
  {
    log_error(
      "emit_delete_block: cannot generate default value for type {}",
      t.id_string());
    return true;
  }
  if (zero.type() != type)
    zero.type() = type;
  side_effect_exprt assign("assign", type);
  assign.copy_to_operands(lhs, zero);
  assigns.push_back(assign);
  return false;
}

// push/pop spec-conformance plan (P1 fix).  See solidity_convert.h
// `gen_default_value_resolved` declaration for the contract.
//
// Mirrors the dispatch of `emit_delete_block` (above) but produces a
// single value expression rather than a code-block of assigns.  The
// caller of no-arg `push()` lowering uses the result as the rhs of
// `arr[len] = <result>`.
//
// Soundness mirrors the delete-fix:
//   - BytesDynamic gets `initialized = 1` (post-init invariant — see
//     delete-correctness commit 277f815478)
//   - mapping placeholder fields default to a 0-byte struct (matches
//     the storage layout; mapping data lives outside the placeholder
//     anyway, so no observable change)
//   - generic struct components recurse — closes the `bytes[]` and
//     `S[]` no-arg push crashes
exprt solidity_convertert::gen_default_value_resolved(const typet &t_in)
{
  // Resolve symbol-typed wrappers via ns.
  typet t = t_in;
  if (t.id() == "symbol")
  {
    const symbolt *s = ns.lookup(t.identifier());
    if (s)
      t = s->type;
  }

  // Mapping placeholder: 1-byte struct with no observable state.  The
  // placeholder's address (a compile-time linker constant) is the `mid`
  // for the global `_ESBMC_map_storage`; gen_zero on this struct is
  // safe (yields struct{0}) but we route through the explicit case to
  // make the intent clear and to keep parity with emit_delete_block.
  if (t.id() == "struct" && t.tag().as_string() == "_ESBMC_Mapping")
  {
    exprt result = struct_exprt(t);
    for (const auto &comp : to_struct_type(t).components())
      result.copy_to_operands(gen_zero(comp.type()));
    return result;
  }

  // BytesDynamic: explicit field-defaults preserving `initialized=1`
  // so downstream push/copy/length-read paths don't trip init checks.
  // Mirrors the delete-fix invariant.
  if (t.id() == "struct" && t.tag().as_string() == "BytesDynamic")
  {
    typet sz_t = size_type();
    typet int_t = int_type();
    exprt result = struct_exprt(t);
    for (const auto &comp : to_struct_type(t).components())
    {
      const std::string &nm = comp.name().as_string();
      if (nm == "initialized")
        result.copy_to_operands(gen_one(int_t));
      else if (nm == "offset" || nm == "length" || nm == "capacity")
        result.copy_to_operands(gen_zero(sz_t));
      else
        // Unknown future field: fall back to recursive default.
        result.copy_to_operands(gen_default_value_resolved(comp.type()));
    }
    return result;
  }

  // Generic struct: recurse on each component, restoring the original
  // (possibly symbol) type on the result so downstream type-equality
  // checks against the array's declared element type still hold.
  if (t.id() == "struct")
  {
    exprt result = struct_exprt(t);
    for (const auto &comp : to_struct_type(t).components())
    {
      exprt field_default = gen_default_value_resolved(comp.type());
      if (field_default.is_nil())
        return field_default; // propagate failure
      result.copy_to_operands(field_default);
    }
    if (t_in.id() == "symbol")
      result.type() = t_in;
    return result;
  }

  // Generic array: handle BEFORE gen_zero fallback so recursion re-enters
  // this function (which resolves symbol_typet at the top), rather than
  // gen_zero (which has no symbol-id case and returns make_nil() for every
  // operand of an array<symbol-wrapped element>, producing
  // constant_array{nil, nil, ...} that crashes symex on subsequent
  // foreach_operand walks).  Closes Bug 6
  // (napp_struct_multifield_{pass,fail} SIGSEGV on bytes32[3] zero-init):
  // bytes32 is symbol_typet("tag-BytesStatic"); without this case
  // gen_default_value_resolved(bytes32[3]) bypasses to gen_zero which
  // returns nil for each symbol-typed element, leaving null-typed operands
  // in the constant_array that null-deref later in goto_symex_state.cpp:67.
  if (t.id() == "array")
  {
    array_typet arr_type = to_array_type(t);
    if (arr_type.size().id() == "infinity")
    {
      exprt elem = gen_default_value_resolved(t.subtype());
      if (elem.is_nil())
        return elem;
      exprt result = array_of_exprt(elem, t);
      if (t_in.id() == "symbol")
        result.type() = t_in;
      return result;
    }
    BigInt size = string2integer(arr_type.size().value().as_string(), 2);
    exprt result = exprt("constant", t);
    for (uint64_t i = 0; i < size.to_uint64(); i++)
    {
      exprt elem = gen_default_value_resolved(t.subtype());
      if (elem.is_nil())
        return elem;
      result.copy_to_operands(elem);
    }
    if (t_in.id() == "symbol")
      result.type() = t_in;
    return result;
  }

  // Default: gen_zero (handles primitives, plain pointers).  May still
  // return nil for genuinely unhandled types; caller responsibility to
  // detect.
  exprt z = gen_zero(t);
  if (!z.is_nil() && t_in.id() == "symbol")
    z.type() = t_in;
  return z;
}

bool solidity_convertert::get_conditional_operator_expr(
  const nlohmann::json &expr,
  exprt &new_expr)
{
  exprt cond;
  if (get_expr(expr["condition"], cond))
    return true;

  exprt then;
  if (get_expr(expr["trueExpression"], expr["typeDescriptions"], then))
    return true;

  exprt else_expr;
  if (get_expr(expr["falseExpression"], expr["typeDescriptions"], else_expr))
    return true;

  typet t;
  if (get_type_description(expr["typeDescriptions"], t))
    return true;

  exprt if_expr("if", t);
  if_expr.copy_to_operands(cond, then, else_expr);

  new_expr = if_expr;

  return false;
}

bool solidity_convertert::get_cast_expr(
  const nlohmann::json &cast_expr,
  exprt &new_expr,
  const nlohmann::json literal_type)
{
  // 1. convert subexpr
  exprt expr;
  if (get_expr(cast_expr["subExpr"], literal_type, expr))
    return true;

  // 2. get type
  typet type;
  if (cast_expr["castType"].get<std::string>() == "ArrayToPointerDecay")
  {
    // Array's cast_expr will have cast_expr["subExpr"]["typeDescriptions"]:
    //  "typeIdentifier": "t_array$_t_uint8_$2_memory_ptr"
    //  "typeString": "uint8[2] memory"
    // For the data above, SolidityGrammar::get_type_name_t will return ArrayTypeName.
    // But we want Pointer type. Hence, adjusting the type manually to make it like:
    //   "typeIdentifier": "ArrayToPtr",
    //   "typeString": "uint8[2] memory"
    nlohmann::json adjusted_type =
      make_array_to_pointer_type(cast_expr["subExpr"]["typeDescriptions"]);
    if (get_type_description(adjusted_type, type))
      return true;
  }
  // TODO: Maybe can just type = expr.type() for other types as well. Need to make sure types are all set in get_expr (many functions are called multiple times to perform the same action).
  else
  {
    type = expr.type();
  }

  // 3. get cast type and generate typecast
  SolidityGrammar::ImplicitCastTypeT cast_type =
    SolidityGrammar::get_implicit_cast_type_t(
      cast_expr["castType"].get<std::string>());
  switch (cast_type)
  {
  case SolidityGrammar::ImplicitCastTypeT::LValueToRValue:
  {
    solidity_gen_typecast(ns, expr, type);
    break;
  }
  case SolidityGrammar::ImplicitCastTypeT::FunctionToPointerDecay:
  case SolidityGrammar::ImplicitCastTypeT::ArrayToPointerDecay:
  {
    break;
  }
  default:
  {
    log_error("Unimplemented implicit cast type");
    return true;
  }
  }

  new_expr = expr;
  return false;
}
