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
     * This function is called throught two paths:
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
    if (expr["referencedDeclaration"] > 0)
    {
      // Soldity uses +ve odd numbers to refer to var or functions declared in the contract
      nlohmann::json decl =
        find_decl_ref(src_ast_json, expr["referencedDeclaration"]);
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
        assert(current_functionDecl);
        if (get_func_decl_this_ref(*current_functionDecl, this_expr))
          return true;
        new_expr = this_expr;
      }
      else
      {
        // Soldity uses -ve odd numbers to refer to built-in var or functions that
        // are NOT declared in the contract
        if (get_esbmc_builtin_ref(expr, new_expr))
          return true;
      }
    }

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
    // make a type-name json for integer literal conversion
    std::string the_value = expr["value"].get<std::string>();
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

      SolidityGrammar::ElementaryTypeNameT target_type =
        SolidityGrammar::get_elementary_type_name_t(literal_type);

      std::string sol_type_str =
        SolidityGrammar::elementary_type_name_to_str(target_type);
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
          call.arguments().push_back(
            from_integer(val_str.length(), uint_type()));
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
          log_error(
            "Unsupported bytes literal type in expression: {}", expr.dump());
          return true;
        }
        call.type().set("#sol_type", "BytesStatic");
        call.type().set("#sol_bytesn_size", expected_size);
        new_expr = make_aux_var(call, location);
        break;
      }
      else if (
        type_name == SolidityGrammar::ElementaryTypeNameT::STRING_LITERAL)
      {
        assert(expr.contains("kind") && expr.contains("hexValue"));
        std::string val_str;
        bool is_hex_string = expr["kind"] == "hexString";
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
          else if (val_str.length() > expected_len)
          {
            log_error(
              "String literal is longer than target bytesN size ({} > {})",
              val_str.length(),
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
        else if (is_static && !is_hex_string)
          str_call.arguments().push_back(
            from_integer(std::stoul(expected_size), uint_type()));
        if (!is_static)
        {
          exprt dynamic_pool;
          if (get_dynamic_pool(current_contractName, dynamic_pool))
            return true;
          str_call.type().set("#sol_type", "BytesDynamic");
          str_call.arguments().push_back(dynamic_pool);
        }
        else
        {
          str_call.type().set("#sol_type", "BytesStatic");
          str_call.type().set(
            "#sol_bytesn_size", byte_t.get("#sol_bytesn_size"));
        }
        new_expr = make_aux_var(str_call, location);
        break;
      }

      log_error(
        "Unsupported bytes literal type in expression: {}", expr.dump());
      return true;
    }

    switch (type_name)
    {
    case SolidityGrammar::ElementaryTypeNameT::INT_LITERAL:
    {
      assert(literal_type != nullptr);
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
        new_expr.type().set("#sol_type", "INT_CONST");
      }
      else if (convert_integer_literal(literal_type, the_value, new_expr))
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
      new_expr.type().set("#sol_type", "ADDRESS");
      break;
    }
    case SolidityGrammar::ElementaryTypeNameT::ADDRESS_PAYABLE:
    {
      // 20 bytes
      if (convert_hex_literal(the_value, new_expr, 160))
        return true;
      new_expr.type().set("#sol_type", "ADDRESS_PAYABLE");
      break;
    }
    default:
      assert(!"Literal not implemented");
    }
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
      assert(literal_type != nullptr);

      // get elem type
      nlohmann::json elem_literal_type =
        make_array_elementary_type(literal_type);

      // get size
      exprt size;
      size = constant_exprt(
        integer2binary(expr["components"].size(), bv_width(int_type())),
        integer2string(expr["components"].size()),
        int_type());

      // get array type
      typet arr_type;
      if (get_type_description(literal_type, arr_type))
        return true;

      // reallocate array size
      arr_type = array_typet(arr_type.subtype(), size);

      // declare static array tuple
      exprt inits;
      inits = gen_zero(arr_type);
      inits.type().set("#sol_type", "ARRAY_LITERAL");
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
        1. when parsing the funciton definition, if the returnParam > 1
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
        code_blockt _block;
        exprt op = nil_exprt();
        for (auto i : expr["components"])
        {
          if (
            i.contains("typeDescriptions") &&
            get_expr(i, i["typeDescriptions"], op))
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
        assert(expr.contains("options"));
        // pass the ["options"] to addressmembercall, via literal_type
        if (get_expr(callee_expr_json, expr["options"], new_expr))
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
        if (!expr.contains("options"))
        {
          log_error("Unsupported CallOptionsExprClass");
          return true;
        }
        // e.g. target.deposit{value: msg.value}();
        exprt memcall;
        // target.deposit
        if (get_expr(callee_expr_json, expr["options"], memcall))
          return true;

        new_expr = memcall;
      }
    }

    break;
  }
  case SolidityGrammar::ExpressionT::CallExprClass:
  {
    side_effect_expr_function_callt call;
    const nlohmann::json &callee_expr_json = expr["expression"];
    // * check if it's a low-level call
    if (SolidityGrammar::is_address_member_call(callee_expr_json))
    {
      log_debug("solidity", "\t\t@@@ got address member call");
      if (get_expr(callee_expr_json, new_expr))
        return true;
      break;
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
        if (get_expr(expr["arguments"][0], args))
          return true;
        new_expr.op0() = args;
        break;
      }

      if (new_expr.id() == "sideeffect")
      {
        std::string func_id = new_expr.op0().identifier().as_string();
        if (func_id == "c:@F@_ESBMC_array_push")
        {
          // signed short _tmpzero#5 = 0;
          // this->data1 = _ESBMC_array_push((void *)this->data1, (void *)&_tmpzero#5, 2);
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
          break;
        }
        if (
          func_id == "c:@F@_ESBMC_array_pop" ||
          func_id == "c:@F@_ESBMC_array_length")
          break;
        if (func_id.compare(0, 11, "c:@F@bytes_") == 0)
          break;
      }
      if (new_expr.is_member() && new_expr.component_name() == "length")
        break;

      std::string sol_name = new_expr.type().get("#sol_name").as_string();
      if (sol_name == "revert")
      {
        // Special case: revert
        // insert a bool false as the first argument.
        // drop the rest of params.
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
        call.function() = new_expr;
        call.type() = to_code_type(new_expr.type()).return_type();
        call.arguments().resize(1);
        call.arguments().at(0) = single_arg;
      }
      else
      {
        // other solidity built-in functions
        if (get_library_function_call(
              new_expr, new_expr.type(), empty_json, expr, call))
          return true;
      }

      new_expr = call;
      break;
    }

    // * check if its a call-with-options
    if (
      !expr.contains("name") && callee_expr_json.contains("nodeType") &&
      callee_expr_json["nodeType"] == "FunctionCallOptions")
    {
      if (get_expr(callee_expr_json, new_expr))
        return true;
      break;
    }

    // * check if it's a member access call
    if (
      callee_expr_json.contains("nodeType") &&
      callee_expr_json["nodeType"] == "MemberAccess")
    {
      if (get_expr(callee_expr_json, literal_type, new_expr))
        return true;
      break;
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
        return true;

      typet t = callee_expr.type();
      exprt inits = gen_zero(t);

      int ref_id = callee_expr_json["referencedDeclaration"].get<int>();
      const nlohmann::json &struct_ref = find_decl_ref(src_ast_json, ref_id);
      if (struct_ref == empty_json)
        return true;

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
      break;
    }

    // funciton call expr
    assert(callee_expr.type().is_code());
    typet type = to_code_type(callee_expr.type()).return_type();

    assert(
      callee_expr_json.contains("referencedDeclaration") &&
      !callee_expr_json["referencedDeclaration"].is_null());
    const auto &decl_ref = find_decl_ref(
      src_ast_json, callee_expr_json["referencedDeclaration"].get<int>());
    std::string node_type = decl_ref["nodeType"].get<std::string>();

    // * check if it's a event, error function call
    if (node_type == "EventDefinition" || node_type == "ErrorDefinition")
    {
      log_debug("solidity", "\t\t@@@ got event/error function call");
      assert(expr.contains("arguments"));
      if (get_library_function_call(callee_expr, type, decl_ref, expr, call))
        return true;
      new_expr = call;
      break;
    }

    // * check if it's the funciton inside library node
    //TODO
    log_debug("solidity", "\t\t@@@ got normal function call");
    // * we had ruled out all the special cases
    // * we now confirm it is called by aother contract inside current contract
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
      break;
    }

    if (get_non_library_function_call(decl_ref, expr, call))
      return true;

    new_expr = call;
    break;
  }
  case SolidityGrammar::ExpressionT::ContractMemberCall:
  {
    // ContractMemberCall
    // - x.setAddress();
    // - x.address();
    // - x.val(); ==> property
    // The later one is quite special, as in Solidity variables behave like functions from the perspective of other contracts.
    // e.g. b._addr is not an address, but a function that returns an address.

    // find the parent json which contains arguments
    const auto &func_call_json = find_last_parent(src_ast_json["nodes"], expr);
    assert(!func_call_json.empty());

    auto callee_expr_json = expr;
    const nlohmann::json &caller_expr_json = callee_expr_json["expression"];
    assert(callee_expr_json.contains("referencedDeclaration"));
    assert(caller_expr_json.contains("referencedDeclaration"));

    side_effect_expr_function_callt call;
    const int contract_var_id =
      caller_expr_json["referencedDeclaration"].get<int>();
    assert(!current_baseContractName.empty());
    const nlohmann::json &base_expr_json =
      find_decl_ref(src_ast_json["nodes"], contract_var_id); // contract

    // contract C{ Base x; x.call();} where base.contractname != current_ContractName;
    // therefore, we need to extract the based contract name
    exprt base;
    std::string base_cname = "";
    if (base_expr_json.empty())
    {
      // assume it's 'this'
      if (contract_var_id < 0 && caller_expr_json["name"] == "this")
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
    else
    {
      if (get_var_decl_ref(base_expr_json, true, base))
        return true;
      base_cname = base.type().get("#sol_contract").as_string();
      assert(!base_cname.empty());
    }

    const int member_id = callee_expr_json["referencedDeclaration"].get<int>();
    std::string old = current_baseContractName;
    current_baseContractName = base_cname;
    const nlohmann::json &member_decl_ref =
      find_decl_ref(src_ast_json["nodes"], member_id); // methods or variables
    current_baseContractName = old;

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
      if (comp.type().get("#sol_type") == "MAPPING")
      {
        assert(func_call_json.contains("arguments"));
        assert(member_decl_ref.contains("typeName"));
        assert(member_decl_ref["typeName"].contains("valueType"));
        exprt pos;
        if (get_expr(
              func_call_json["arguments"][0],
              member_decl_ref["typeName"]["valueType"]["typeDescriptions"],
              pos))
          return true;

        bool is_new_expr = newContractSet.count(base_cname);
        if (is_new_expr)
        {
          if (
            !is_bound && tgt_cnt_set.count(base_cname) > 0 &&
            tgt_cnt_set.size() == 1)
            is_new_expr = false;
        }
        // get key/value type
        typet key_t, value_t;
        std::string key_sol_type, val_sol_type;
        if (get_mapping_key_value_type(
              member_decl_ref, key_t, value_t, key_sol_type, val_sol_type))
        {
          log_error("cannot get mapping key/value type");
          return true;
        }
        gen_mapping_key_typecast(current_contractName, pos, location, key_t);

        if (!is_bound)
        {
          // x.member(); ==> nondet();
          get_nondet_expr(value_t, new_expr);
          break;
        }

        if (!is_new_expr)
        {
          assert(comp.type().is_array());
          //TODO: the index type of ESBMC is limited to unsigned long long
          // we need to extend such limit up to unsignedbv_type(256)
          new_expr = index_exprt(comp, pos, value_t);
        }
        else
        {
          bool is_mapping_set = false;
          auto _mem_call = member_exprt(base, comp.name(), comp.type());
          if (get_new_mapping_index_access(
                value_t,
                val_sol_type,
                is_mapping_set,
                _mem_call,
                pos,
                location,
                new_expr))
            return true;
        }

        break;
      }

      if (current_contractName == base_cname)
        // this.member();
        // comp can be either symbol_expr or member_expr
        new_expr = member_exprt(base, comp_name, comp.type());
      else if (!is_bound)
        // x.member(); ==> nondet();
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
      // e.g. x.func()
      // x    --> base
      // func --> comp
      exprt comp;
      if (get_func_decl_ref(member_decl_ref, comp))
        return true;

      if (get_non_library_function_call(member_decl_ref, func_call_json, call))
        return true;
      call.arguments().at(0) = base;

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
      else if (!is_bound)
      {
        if (get_unbound_expr(func_call_json, current_contractName, new_expr))
          return true;

        typet t = to_code_type(comp.type()).return_type();
        get_nondet_expr(t, new_expr);
      }
      else
      {
        assert(!comp.is_member());
        if (get_high_level_member_access(
              func_call_json, literal_type, base, comp, call, true, new_expr))
          return true;
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
    else
    {
      // 2.2 func()[n]
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
    if (base_t.get("#sol_type").as_string() == "MAPPING")
    {
      // hack to improve the verification speed
      bool is_new_expr = newContractSet.count(current_contractName);
      if (is_new_expr)
      {
        if (
          !is_bound && tgt_cnt_set.count(current_contractName) > 0 &&
          tgt_cnt_set.size() == 1)
          is_new_expr = false;
      }

      // find mapping definition
      assert(base_json.contains("referencedDeclaration"));
      const nlohmann::json &map_node = find_decl_ref(
        src_ast_json["nodes"], base_json["referencedDeclaration"].get<int>());

      // get key/value type
      typet key_t, value_t;
      std::string key_sol_type, val_sol_type;
      if (get_mapping_key_value_type(
            map_node, key_t, value_t, key_sol_type, val_sol_type))
      {
        log_error("cannot get mapping key/value type");
        return true;
      }
      gen_mapping_key_typecast(current_contractName, pos, location, key_t);

      if (!is_new_expr)
      {
        assert(array.type().is_array());
        //TODO: the index type of ESBMC is limited to unsigned long long
        // we need to extend such limit up to unsignedbv_type(256)
        new_expr = index_exprt(array, pos, t);
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
      break;
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
      break;
    }

    new_expr = index_exprt(array, pos, t);
    break;
  }
  case SolidityGrammar::ExpressionT::NewExpression:
  {
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
        break;
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
        new_expr.type().set("#sol_type", "BytesDynamic");
        break;
      }
    }
    // case 3
    // is equal to Base *x = new base(x);
    exprt call;
    if (get_new_object_ctor_call(expr, false, call))
      return true;

    new_expr = call;
    // check if the new expression has options
    if (
      expr.contains("options") && expr.contains("names") &&
      !expr["options"].empty() && !expr["names"].empty())
    {
      const auto &options = expr["options"];
      const auto &names = expr["names"];

      for (size_t i = 0; i < options.size(); ++i)
      {
        const auto &opt = options[i];
        std::string opt_name = names[i];
        // model transaction when the option is "value"
        if (opt_name == "value")
        {
          exprt value_expr;
          nlohmann::json val_type = expr["expression"]["argumentTypes"][0];
          if (get_expr(opt, val_type, value_expr))
            return true;

          exprt this_expr;
          if (get_func_decl_this_ref(expr, this_expr))
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

          move_to_back_block(back_block);
          break;
        }
      }
    }

    if (is_bound)
    {
      // fix: x._ESBMC_bind_cname = Base
      // lhs
      exprt lhs;
      if (get_bind_cname_expr(expr, lhs))
        break; // no lhs

      // rhs
      int ref_decl_id = callee_expr_json["typeName"]["referencedDeclaration"];
      const std::string contract_name = contractNamesMap[ref_decl_id];
      exprt rhs;
      get_cname_expr(contract_name, rhs);

      // do assignment
      exprt _assign = side_effect_exprt("assign", lhs.type());
      solidity_gen_typecast(ns, rhs, lhs.type());
      _assign.operands().push_back(lhs);
      _assign.operands().push_back(rhs);

      convert_expression_to_code(_assign);
      move_to_back_block(_assign);
    }

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
    const nlohmann::json &struct_var_ref =
      find_decl_ref(src_ast_json, struct_var_id);
    if (struct_var_ref == empty_json)
    {
      log_error("cannot find struct member reference");
      return true;
    }
    exprt comp;
    if (get_var_decl_ref(struct_var_ref, true, comp))
      return true;

    assert(comp.name() == expr["memberName"]);
    new_expr = member_exprt(base, comp.name(), comp.type());

    break;
  }
  case SolidityGrammar::ExpressionT::EnumMemberCall:
  {
    assert(expr.contains("expression"));
    const int enum_id = expr["referencedDeclaration"].get<int>();
    const nlohmann::json &enum_member_ref =
      find_decl_ref_unique_id(src_ast_json, enum_id);
    if (enum_member_ref == empty_json)
      return true;

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
    {
      // e.g.
      // - address(msg.sender)
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
      if (!is_bound)
      {
        if (get_unbound_expr(expr, current_contractName, new_expr))
          return true;

        if (mem_name == "send")
          new_expr = nondet_bool_expr;
        else
        {
          // call, staticcall ...
          symbolt dump;
          get_llc_ret_tuple(dump);
          new_expr = symbol_expr(dump);
        }
      }
      else
      {
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
          return true;

        exprt arg = nil_exprt();
        assert((*func_call).contains("arguments"));

        // only one possible arguemnt
        if ((*func_call)["arguments"].size() > 0)
        {
          auto &arguments = (*func_call)["arguments"][0];
          if (get_expr(arguments, expr["argumentTypes"][0], arg))
            return true;
        }

        if (get_low_level_member_accsss(
              expr, literal_type, mem_name, base, arg, new_expr))
          return true;
      }
    }
    else if (is_low_level_property(mem_name))
    {
      // property i.e balance/codehash
      if (!is_bound)
        // make it as a NODET_UINT
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
    exprt base;
    const nlohmann::json caller_expr_json = expr["expression"];
    typet t;
    if (get_type_description(caller_expr_json["typeDescriptions"], t))
      return true;
    const auto &func_ref = find_decl_ref_unique_id(
      src_ast_json, expr["referencedDeclaration"].get<int>());

    if (t.get("#sol_type") == "ENUM")
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

    if (get_expr(caller_expr_json, literal_type, base))
      return true;

    side_effect_expr_function_callt call;

    const nlohmann::json &args_json =
      find_last_parent(src_ast_json["nodes"], expr);
    assert(args_json.contains("arguments"));
    if (get_library_function_call(func_ref, args_json, call))
      return true;

    if (!(base.is_code() && base.type().get("#sol_type") == "LIBRARY"))
      // this means it is a using for library
      call.arguments().insert(call.arguments().begin(), base);

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

    // 1. get source expr
    //! assume: only one argument
    assert(expr["arguments"].size() == 1);
    if (get_expr(expr["arguments"][0], literal_type, from_expr))
      return true;

    // 2. get target type
    if (get_type_description(conv_expr["typeDescriptions"], type))
      return true;

    // 3. generate the type casting expr
    convert_type_expr(ns, from_expr, type, expr);

    new_expr = from_expr;
    break;
  }
  case SolidityGrammar::ExpressionT::NullExpr:
  {
    // e.g. (, x) = (1, 2);
    // the first component in lhs is nil
    new_expr = nil_exprt();
    break;
  }
  default:
  {
    assert(!"Unimplemented type in rule expression");
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
  current_BinOp_type.push(&(expr["typeDescriptions"]));

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
  std::string lt_sol = lt.get("#sol_type").as_string();
  std::string rt_sol = rt.get("#sol_type").as_string();

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

        lt_sol = lhs.type().get("#sol_type").as_string();
        rt_sol = rhs.type().get("#sol_type").as_string();
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
          log_error("Incompatible bytes comparison: {} vs {}", lt_sol, rt_sol);
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
          "Shift operations only supported on bytesN types, got {}", lt_sol);
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
    // special handling for tuple-type assignment;
    //TODO: handle nested tuple
    if (rt_sol == "TUPLE_INSTANCE" || rt_sol == "TUPLE_RETURNS")
    {
      construct_tuple_assigments(expr, lhs, rhs);
      new_expr = code_skipt();
      current_BinOp_type.pop();
      return false;
    }
    else if (rt_sol == "ARRAY" || rt_sol == "ARRAY_LITERAL")
    {
      if (rt_sol == "ARRAY_LITERAL")
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
    else if (rt_sol == "DYNARRAY")
    {
      /* e.g. 
        int[] public data1;
        int[] memory ac;
        ac = new int[](10);
        data1 = ac;  // target

      we convert it as 
        data1 = _ESBMC_arrcpy(ac, get_array_size(ac), type_size); 
      */

      // get size
      exprt size_expr;
      get_size_expr(rhs, size_expr);

      // get sizeof
      exprt size_of_expr;
      get_size_of_expr(rt.subtype(), size_of_expr);

      // do array copy
      side_effect_expr_function_callt acpy_call;
      get_arrcpy_function_call(lhs.location(), acpy_call);
      acpy_call.arguments().push_back(rhs);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      solidity_gen_typecast(ns, acpy_call, lt);

      rhs = acpy_call;
      // fall through to do assignment
    }
    else if (rt_sol == "ARRAY_CALLOC")
    {
      /* e.g. 
        int[] memory ac;
        ac = new int[](10);
      */
      exprt size_expr;
      if (!rhs_json.contains("arguments"))
        abort();
      nlohmann::json callee_arg_json = rhs_json["arguments"][0];
      const nlohmann::json literal_type = callee_arg_json["typeDescriptions"];

      // get new array
      if (get_expr(callee_arg_json, literal_type, size_expr))
        return true;

      // get sizeof
      exprt size_of_expr;
      get_size_of_expr(rt.subtype(), size_of_expr);

      // do array copy
      side_effect_expr_function_callt acpy_call;
      get_arrcpy_function_call(lhs.location(), acpy_call);

      acpy_call.arguments().push_back(rhs);
      acpy_call.arguments().push_back(size_expr);
      acpy_call.arguments().push_back(size_of_expr);
      solidity_gen_typecast(ns, acpy_call, lt);

      rhs = acpy_call;
    }
    else if (lt_sol == "STRING")
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
      // Not sure why but it seems esbmc's pow works worse in solidity,
      // so I write my own version
      //? maybe this should convert this to BigInt too?
      side_effect_expr_function_callt call_expr;
      get_library_function_call_no_args(
        "pow", "c:@F@pow", double_type(), lhs.location(), call_expr);

      call_expr.arguments().push_back(typecast_exprt(lhs, double_type()));
      call_expr.arguments().push_back(typecast_exprt(rhs, double_type()));

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
      assert(!"Unimplemented binary operator");
      return true;
    }

    current_BinOp_type.pop();

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

  // Pop current_BinOp_type.push as we've finished this conversion
  current_BinOp_type.pop();

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
  // TODO: Fix me! Currently just support prefix == true,e.g. pre-increment

  // 1. get UnaryOperation opcode
  SolidityGrammar::ExpressionT opcode =
    SolidityGrammar::get_unary_expr_operator_t(expr, expr["prefix"]);
  log_debug(
    "solidity",
    "	@@@ got uniop.getOpcode: SolidityGrammar::{}",
    SolidityGrammar::expression_to_str(opcode));

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
    assert(!"Unimplemented unary operator");
  }
  }

  new_expr.operands().push_back(unary_sub);
  return false;
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
    assert(!"Unimplemented implicit cast type");
  }
  }

  new_expr = expr;
  return false;
}
