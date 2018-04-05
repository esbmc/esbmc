/*
 * clang_c_adjust.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: mramalho
 */

#include <clang-c-frontend/clang_c_adjust.h>
#include <clang-c-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/c_sizeof.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <util/ieee_float.h>
#include <util/prefix.h>
#include <util/std_code.h>

clang_c_adjust::clang_c_adjust(contextt &_context)
  : context(_context), ns(namespacet(context))
{
}

bool clang_c_adjust::adjust()
{
  // warning! hash-table iterators are not stable

  symbol_listt symbol_list;
  context.Foreach_operand(
    [&symbol_list](symbolt &s) { symbol_list.push_back(&s); });

  // Adjust types first, so that symbolic-type resolution always receives
  // fixed up types.
  Forall_symbol_list(it, symbol_list)
  {
    symbolt &symbol = **it;
    if(symbol.is_type)
      adjust_type(symbol.type);
  }

  Forall_symbol_list(it, symbol_list)
  {
    symbolt &symbol = **it;
    adjust_type(symbol.type);

    if(symbol.is_type)
      continue;

    adjust_symbol(symbol);
  }

  return false;
}

void clang_c_adjust::adjust_symbol(symbolt &symbol)
{
  if(!symbol.value.is_nil())
    adjust_expr(symbol.value);

  if(symbol.type.is_code() && symbol.name == "main")
    adjust_argc_argv(symbol);
}

void clang_c_adjust::adjust_expr(exprt &expr)
{
  adjust_type(expr.type());

  if(expr.id() == "sideeffect")
  {
    adjust_side_effect(to_side_effect_expr(expr));
  }
  else if(expr.id() == "symbol")
  {
    adjust_symbol(expr);
  }
  else if(expr.id() == "not")
  {
    adjust_expr_unary_boolean(expr);
  }
  else if(expr.is_and() || expr.is_or())
  {
    adjust_expr_binary_boolean(expr);
  }
  else if(expr.is_address_of())
  {
    adjust_address_of(expr);
  }
  else if(expr.is_dereference())
  {
    adjust_dereference(expr);
  }
  else if(expr.is_member())
  {
    adjust_member(to_member_expr(expr));
  }
  else if(
    expr.id() == "=" || expr.id() == "notequal" || expr.id() == "<" ||
    expr.id() == "<=" || expr.id() == ">" || expr.id() == ">=")
  {
    adjust_expr_rel(expr);
  }
  else if(expr.is_index())
  {
    adjust_index(to_index_expr(expr));
  }
  else if(expr.id() == "sizeof")
  {
    adjust_sizeof(expr);
  }
  else if(
    expr.id() == "+" || expr.id() == "-" || expr.id() == "*" ||
    expr.id() == "/" || expr.id() == "mod" || expr.id() == "shl" ||
    expr.id() == "shr" || expr.id() == "bitand" || expr.id() == "bitxor" ||
    expr.id() == "bitor")
  {
    adjust_expr_binary_arithmetic(expr);
  }
  else if(expr.id() == "comma")
  {
    adjust_comma(expr);
  }
  else if(expr.id() == "if")
  {
    // Check all operands
    adjust_operands(expr);

    // If the condition is not of boolean type, it must be casted
    gen_typecast(ns, expr.op0(), bool_type());

    // Typecast both the true and false results
    gen_typecast_arithmetic(ns, expr.op1(), expr.op2());
  }
  else if(expr.id() == "builtin_va_arg")
  {
    adjust_builtin_va_arg(expr);
  }
  else if(expr.is_code())
  {
    adjust_code(to_code(expr));
  }
  else
  {
    // Just check operands of everything else
    adjust_operands(expr);
  }
}

void clang_c_adjust::adjust_symbol(exprt &expr)
{
  const irep_idt &identifier = expr.identifier();

  // look it up
  symbolt *s = context.find_symbol(identifier);

  if(s == nullptr)
    return;

  // found it
  const symbolt &symbol = *s;

  // save location
  locationt location = expr.location();

  if(symbol.is_macro)
  {
    expr = symbol.value;

    // put it back
    expr.location() = location;
  }
  else
  {
    expr = symbol_expr(symbol);

    // put it back
    expr.location() = location;

    if(symbol.lvalue)
      expr.cmt_lvalue(true);

    if(expr.type().is_code()) // function designator
    {
      // special case: this is sugar for &f
      address_of_exprt tmp(expr);
      tmp.implicit(true);
      tmp.location() = expr.location();
      expr.swap(tmp);
    }
  }
}

void clang_c_adjust::adjust_side_effect(side_effect_exprt &expr)
{
  const irep_idt &statement = expr.get_statement();

  if(statement == "function_call")
    adjust_side_effect_function_call(to_side_effect_expr_function_call(expr));
  else
  {
    adjust_operands(expr);

    if(
      statement == "preincrement" || statement == "predecrement" ||
      statement == "postincrement" || statement == "postdecrement")
    {
    }
    else if(has_prefix(id2string(statement), "assign"))
      adjust_side_effect_assignment(expr);
    else if(statement == "statement_expression")
      adjust_side_effect_statement_expression(expr);
    else if(statement == "gcc_conditional_expression")
    {
    }
    else
    {
      std::cout << "unknown side effect: " << statement;
      std::cout << " at " << expr.location() << std::endl;
      abort();
    }
  }
}

void clang_c_adjust::adjust_member(member_exprt &expr)
{
  adjust_operands(expr);

  exprt &base = expr.struct_op();
  if(base.type().is_pointer())
  {
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
}

void clang_c_adjust::adjust_expr_binary_arithmetic(exprt &expr)
{
  adjust_operands(expr);

  exprt &op0 = expr.op0();
  exprt &op1 = expr.op1();

  const typet type0 = ns.follow(op0.type());
  const typet type1 = ns.follow(op1.type());

  if(expr.id() == "shr" || expr.id() == "shl")
  {
    gen_typecast_arithmetic(ns, op0);
    gen_typecast_arithmetic(ns, op1);

    if(is_number(op0.type()) && is_number(op1.type()))
    {
      if(expr.id() == "shr") // shifting operation depends on types
      {
        if(type0.id() == "unsignedbv")
        {
          expr.id("lshr");
          return;
        }
        if(type0.id() == "signedbv")
        {
          expr.id("ashr");
          return;
        }
      }

      return;
    }
  }
  else
  {
    gen_typecast_arithmetic(ns, op0, op1);

    const typet &type0 = ns.follow(op0.type());
    const typet &type1 = ns.follow(op1.type());

    if(
      expr.id() == "+" || expr.id() == "-" || expr.id() == "*" ||
      expr.id() == "/")
    {
      if(type0.id() == "pointer" || type1.id() == "pointer")
      {
        //        typecheck_expr_pointer_arithmetic(expr);
        return;
      }

      adjust_float_arith(expr);
      return;
    }
  }
}

void clang_c_adjust::adjust_index(index_exprt &index)
{
  adjust_operands(index);

  exprt &array_expr = index.op0();
  exprt &index_expr = index.op1();

  // we might have to swap them

  {
    const typet &array_full_type = ns.follow(array_expr.type());
    const typet &index_full_type = ns.follow(index_expr.type());

    if(
      !array_full_type.is_array() && !array_full_type.is_pointer() &&
      (index_full_type.is_array() || index_full_type.is_pointer()))
      std::swap(array_expr, index_expr);
  }

  const typet &final_array_type = ns.follow(array_expr.type());

  if(final_array_type.is_array() || final_array_type.id() == "incomplete_array")
  {
    if(array_expr.cmt_lvalue())
      index.cmt_lvalue(true);
  }
  else if(final_array_type.id() == "pointer")
  {
    // p[i] is syntactic sugar for *(p+i)

    exprt addition("+", array_expr.type());
    addition.operands().swap(index.operands());
    index.move_to_operands(addition);
    index.id("dereference");
    index.cmt_lvalue(true);
  }

  index.type() = final_array_type.subtype();
}

void clang_c_adjust::adjust_expr_rel(exprt &expr)
{
  adjust_operands(expr);

  expr.type() = bool_type();

  exprt &op0 = expr.op0();
  exprt &op1 = expr.op1();

  gen_typecast_arithmetic(ns, op0, op1);
}

void clang_c_adjust::adjust_float_arith(exprt &expr)
{
  // equality and disequality on float is not mathematical equality!
  assert(expr.operands().size() == 2);

  if(ns.follow(expr.type()).is_floatbv())
  {
    // And change id
    if(expr.id() == "+")
    {
      expr.id("ieee_add");
    }
    else if(expr.id() == "-")
    {
      expr.id("ieee_sub");
    }
    else if(expr.id() == "*")
    {
      expr.id("ieee_mul");
    }
    else if(expr.id() == "/")
    {
      expr.id("ieee_div");
    }

    // Add rounding mode
    expr.set(
      "rounding_mode",
      symbol_exprt(CPROVER_PREFIX "rounding_mode", int_type()));
  }
}

void clang_c_adjust::adjust_address_of(exprt &expr)
{
  adjust_operands(expr);

  exprt &op = expr.op0();

  // special case: address of function designator
  // ANSI-C 99 section 6.3.2.1 paragraph 4

  if(
    op.is_address_of() && op.implicit() && op.operands().size() == 1 &&
    op.op0().id() == "symbol" && op.op0().type().is_code())
  {
    // make the implicit address_of an explicit address_of
    exprt tmp;
    tmp.swap(op);
    tmp.implicit(false);
    expr.swap(tmp);
    return;
  }

  expr.type() = typet("pointer");

  // turn &array into &(array[0])
  if(op.type().is_array())
  {
    index_exprt index;
    index.array() = op;
    index.index() = gen_zero(index_type());
    index.type() = op.type().subtype();
    index.location() = expr.location();
    op.swap(index);
  }

  expr.type().subtype() = op.type();
}

void clang_c_adjust::adjust_dereference(exprt &deref)
{
  adjust_operands(deref);

  exprt &op = deref.op0();

  const typet op_type = ns.follow(op.type());

  if(op_type.is_array())
  {
    // *a is the same as a[0]
    deref.id("index");
    deref.type() = op_type.subtype();
    deref.copy_to_operands(gen_zero(index_type()));
    assert(deref.operands().size() == 2);
  }
  else if(op_type.id() == "pointer")
  {
    deref.type() = op_type.subtype();
  }

  deref.cmt_lvalue(true);

  // if you dereference a pointer pointing to
  // a function, you get a pointer again
  // allowing ******...*p
  if(deref.type().is_code())
  {
    exprt tmp("address_of", pointer_typet());
    tmp.implicit(true);
    tmp.type().subtype() = deref.type();
    tmp.location() = deref.location();
    tmp.move_to_operands(deref);
    deref.swap(tmp);
  }
}

void clang_c_adjust::adjust_sizeof(exprt &expr)
{
  typet type;
  if(expr.operands().size() == 0)
  {
    type = ((typet &)expr.c_sizeof_type());
    adjust_type(type);
  }
  else if(expr.operands().size() == 1)
  {
    type.swap(expr.op0().type());
    adjust_type(type);
  }
  else
  {
    std::cout << "sizeof operator expects zero or one operand, "
              << "but got" << expr.operands().size() << std::endl;
    abort();
  }

  exprt new_expr = c_sizeof(type, ns);

  if(new_expr.is_nil())
  {
    std::cout << "type has no size, " << type.name() << std::endl;
    abort();
  }

  new_expr.swap(expr);
  expr.c_sizeof_type(type);
}

void clang_c_adjust::adjust_type(typet &type)
{
  if(type.id() == "symbol")
  {
    const irep_idt &identifier = type.identifier();

    // look it up
    symbolt *s = context.find_symbol(identifier);

    if(s == nullptr)
    {
      std::cout << "type symbol `" << identifier << "' not found" << std::endl;
      abort();
    }

    const symbolt &symbol = *s;

    if(!symbol.is_type)
    {
      std::cout << "expected type symbol, but got " << std::endl;
      symbol.dump();
      abort();
    }

    if(symbol.is_macro)
      type = symbol.type; // overwrite
  }
}

void clang_c_adjust::adjust_side_effect_assignment(exprt &expr)
{
  const irep_idt &statement = expr.statement();

  exprt &op0 = expr.op0();
  exprt &op1 = expr.op1();

  const typet type0 = op0.type();

  if(statement == "assign")
  {
    gen_typecast(ns, op1, type0);
    return;
  }
  if(statement == "assign_shl" || statement == "assign_shr")
  {
    gen_typecast_arithmetic(ns, op1);

    if(is_number(op1.type()))
    {
      if(statement == "assign_shl")
      {
        return;
      }

      if(type0.id() == "unsignedbv")
      {
        expr.statement("assign_lshr");
        return;
      }
      if(type0.id() == "signedbv")
      {
        expr.statement("assign_ashr");
        return;
      }
    }
  }
}

void clang_c_adjust::adjust_side_effect_function_call(
  side_effect_expr_function_callt &expr)
{
  exprt &f_op = expr.function();

  if(f_op.is_symbol())
  {
    const irep_idt &identifier = f_op.identifier();
    symbolt *s = context.find_symbol(identifier);
    if(s == nullptr)
    {
      // maybe this is an undeclared function
      // let's just add it
      symbolt new_symbol;

      new_symbol.name = identifier;
      new_symbol.base_name = f_op.name();
      new_symbol.location = expr.location();
      new_symbol.type = f_op.type();
      new_symbol.mode = "C";
      new_symbol.is_used = true;

      // Adjust type
      to_code_type(new_symbol.type).make_ellipsis();
      to_code_type(f_op.type()).make_ellipsis();

      symbolt *symbol_ptr;
      bool res = context.move(new_symbol, symbol_ptr);
      assert(!res);
      (void)res; // ndebug

      // clang will complain about this already, no need for us to do the same!
    }
    else
    {
      // Pull symbol informations, like parameter types and location

      // Save previous location
      locationt location = f_op.location();

      const symbolt &symbol = *s;
      f_op = symbol_expr(symbol);

      // Restore location
      f_op.location() = location;

      if(symbol.lvalue)
        f_op.cmt_lvalue(true);
    }
  }
  else
    adjust_expr(f_op);

  // do implicit dereference
  if(f_op.is_address_of() && f_op.implicit() && (f_op.operands().size() == 1))
  {
    exprt tmp;
    tmp.swap(f_op.op0());
    f_op.swap(tmp);
  }
  else if(f_op.type().is_pointer())
  {
    exprt tmp("dereference", f_op.type().subtype());
    tmp.implicit(true);
    tmp.location() = f_op.location();
    tmp.move_to_operands(f_op);
    f_op.swap(tmp);
  }

  adjust_function_call_arguments(expr);

  do_special_functions(expr);
}

void clang_c_adjust::adjust_function_call_arguments(
  side_effect_expr_function_callt &expr)
{
  exprt &f_op = expr.function();
  const code_typet &code_type = to_code_type(f_op.type());
  exprt::operandst &arguments = expr.arguments();
  const code_typet::argumentst &argument_types = code_type.arguments();

  for(unsigned i = 0; i < arguments.size(); i++)
  {
    exprt &op = arguments[i];
    adjust_expr(op);

    if(i < argument_types.size())
    {
      const code_typet::argumentt &argument_type = argument_types[i];
      const typet &op_type = argument_type.type();
      gen_typecast(ns, op, op_type);
    }
    else
    {
      // don't know type, just do standard conversion

      const typet &type = ns.follow(op.type());
      if(type.is_array())
        gen_typecast(ns, op, pointer_typet(empty_typet()));
    }
  }
}

void clang_c_adjust::do_special_functions(side_effect_expr_function_callt &expr)
{
  const exprt &f_op = expr.function();
  const locationt location = expr.location();

  // some built-in functions
  if(f_op.is_symbol())
  {
    const irep_idt &identifier = to_symbol_expr(f_op).get_identifier();

    if(identifier == CPROVER_PREFIX "same_object")
    {
      if(expr.arguments().size() != 2)
      {
        std::cout << "same_object expects two operands" << std::endl;
        expr.dump();
        abort();
      }

      exprt same_object_expr("same-object", bool_typet());
      same_object_expr.operands() = expr.arguments();
      expr.swap(same_object_expr);
    }
    else if(identifier == CPROVER_PREFIX "POINTER_OFFSET")
    {
      if(expr.arguments().size() != 1)
      {
        std::cout << "pointer_offset expects one argument" << std::endl;
        expr.dump();
        abort();
      }

      exprt pointer_offset_expr = exprt("pointer_offset", expr.type());
      pointer_offset_expr.operands() = expr.arguments();
      expr.swap(pointer_offset_expr);
    }
    else if(identifier == CPROVER_PREFIX "POINTER_OBJECT")
    {
      if(expr.arguments().size() != 1)
      {
        std::cout << "pointer_object expects one argument" << std::endl;
        expr.dump();
        abort();
      }

      exprt pointer_object_expr = exprt("pointer_object", expr.type());
      pointer_object_expr.operands() = expr.arguments();
      expr.swap(pointer_object_expr);
    }
    else if(
      identifier == CPROVER_PREFIX "isnanf" ||
      identifier == CPROVER_PREFIX "isnand" ||
      identifier == CPROVER_PREFIX "isnanld" || identifier == "__builtin_isnan")
    {
      if(expr.arguments().size() != 1)
      {
        std::cout << "isnan expects one operand" << std::endl;
        expr.dump();
        abort();
      }

      exprt isnan_expr("isnan", bool_typet());
      isnan_expr.operands() = expr.arguments();
      expr.swap(isnan_expr);
    }
    else if(
      identifier == CPROVER_PREFIX "isfinitef" ||
      identifier == CPROVER_PREFIX "isfinited" ||
      identifier == CPROVER_PREFIX "isfiniteld")
    {
      if(expr.arguments().size() != 1)
      {
        std::cout << "isfinite expects one operand" << std::endl;
        expr.dump();
        abort();
      }

      exprt isfinite_expr("isfinite", bool_typet());
      isfinite_expr.operands() = expr.arguments();
      expr.swap(isfinite_expr);
    }
    else if(
      identifier == CPROVER_PREFIX "inff" ||
      identifier == CPROVER_PREFIX "inf" ||
      identifier == CPROVER_PREFIX "infld" || identifier == "__builtin_inff" ||
      identifier == "__builtin_inf" || identifier == "__builtin_infld" ||
      identifier == "__builtin_huge_valf" ||
      identifier == "__builtin_huge_val" || identifier == "__builtin_huge_vall")
    {
      typet t = expr.type();

      constant_exprt infl_expr;
      if(config.ansi_c.use_fixed_for_float)
      {
        // We saturate to the biggest value
        mp_integer value = power(2, bv_width(t) - 1) - 1;
        infl_expr = constant_exprt(
          integer2binary(value, bv_width(t)), integer2string(value, 10), t);
      }
      else
      {
        infl_expr =
          ieee_floatt::plus_infinity(ieee_float_spect(to_floatbv_type(t)))
            .to_expr();
      }

      expr.swap(infl_expr);
    }
    else if(
      identifier == CPROVER_PREFIX "nanf" ||
      identifier == CPROVER_PREFIX "nan" ||
      identifier == CPROVER_PREFIX "nanld" || identifier == "__builtin_nanf" ||
      identifier == "__builtin_nan" || identifier == "__builtin_nanl")
    {
      typet t = expr.type();

      constant_exprt nan_expr;
      if(config.ansi_c.use_fixed_for_float)
      {
        mp_integer value = 0;
        nan_expr = constant_exprt(
          integer2binary(value, bv_width(t)), integer2string(value, 10), t);
      }
      else
      {
        nan_expr =
          ieee_floatt::NaN(ieee_float_spect(to_floatbv_type(t))).to_expr();
      }

      expr.swap(nan_expr);
    }
    else if(
      identifier == CPROVER_PREFIX "abs" ||
      identifier == CPROVER_PREFIX "labs" ||
      identifier == CPROVER_PREFIX "llabs" ||
      identifier == CPROVER_PREFIX "fabsd" ||
      identifier == CPROVER_PREFIX "fabsf" ||
      identifier == CPROVER_PREFIX "fabsld" || identifier == "__builtin_fabs" ||
      identifier == "__builtin_fabsf")
    {
      if(expr.arguments().size() != 1)
      {
        std::cout << "abs expects one operand" << std::endl;
        expr.dump();
        abort();
      }

      exprt abs_expr("abs", expr.type());
      abs_expr.operands() = expr.arguments();
      expr.swap(abs_expr);
    }
    else if(
      identifier == CPROVER_PREFIX "isinf" ||
      identifier == CPROVER_PREFIX "isinff" ||
      identifier == CPROVER_PREFIX "isinfd" ||
      identifier == CPROVER_PREFIX "isinfld" ||
      identifier == "__builtin_isinf" || identifier == "__builtin_isinff" ||
      identifier == "__builtin_isinfd" || identifier == "__builtin_isinfld")
    {
      if(expr.arguments().size() != 1)
      {
        std::cout << "isinf expects one operand" << std::endl;
        expr.dump();
        abort();
      }

      exprt isinf_expr("isinf", bool_typet());
      isinf_expr.operands() = expr.arguments();
      expr.swap(isinf_expr);
    }
    else if(
      identifier == CPROVER_PREFIX "isnormal" ||
      identifier == CPROVER_PREFIX "isnormalf" ||
      identifier == CPROVER_PREFIX "isnormald" ||
      identifier == CPROVER_PREFIX "isnormalld" ||
      identifier == "__builtin_isnormal" ||
      identifier == "__builtin_isnormalf" ||
      identifier == "__builtin_isnormald" ||
      identifier == "__builtin_isnormalld")
    {
      if(expr.arguments().size() != 1)
      {
        std::cout << "finite expects one operand" << std::endl;
        expr.dump();
        abort();
      }

      exprt isnormal_expr("isnormal", bool_typet());
      isnormal_expr.operands() = expr.arguments();
      expr.swap(isnormal_expr);
    }
    else if(
      identifier == CPROVER_PREFIX "signf" ||
      identifier == CPROVER_PREFIX "signd" ||
      identifier == CPROVER_PREFIX "signld" ||
      identifier == "__builtin_signbit" || identifier == "__builtin_signbitf" ||
      identifier == "__builtin_signbitl")
    {
      if(expr.arguments().size() != 1)
      {
        std::cout << "sign expects one operand" << std::endl;
        expr.dump();
        abort();
      }

      exprt sign_expr("signbit", int_type());
      sign_expr.operands() = expr.arguments();
      expr.swap(sign_expr);
    }
    else if(identifier == "__builtin_expect")
    {
      // this is a gcc extension to provide branch prediction
      if(expr.arguments().size() != 2)
      {
        std::cout << "__builtin_expect expects two arguments" << std::endl;
        expr.dump();
        abort();
      }

      exprt tmp = expr.arguments()[0];
      expr.swap(tmp);
    }
    else if(identifier == "__builtin_isgreater")
    {
      // this is a gcc extension to provide branch prediction
      if(expr.arguments().size() != 2)
      {
        std::cout << "__builtin_isgreater expects two arguments" << std::endl;
        expr.dump();
        abort();
      }

      exprt op(">", bool_typet());
      op.copy_to_operands(expr.arguments()[0], expr.arguments()[1]);

      expr.swap(op);
    }
    else if(identifier == "__builtin_isgreaterequal")
    {
      // this is a gcc extension to provide branch prediction
      if(expr.arguments().size() != 2)
      {
        std::cout << "__builtin_isgreaterequal expects two arguments"
                  << std::endl;
        expr.dump();
        abort();
      }

      exprt op(">=", bool_typet());
      op.copy_to_operands(expr.arguments()[0], expr.arguments()[1]);

      expr.swap(op);
    }
    else if(identifier == "__builtin_isless")
    {
      // this is a gcc extension to provide branch prediction
      if(expr.arguments().size() != 2)
      {
        std::cout << "__builtin_isless expects two arguments" << std::endl;
        expr.dump();
        abort();
      }

      exprt op("<", bool_typet());
      op.copy_to_operands(expr.arguments()[0], expr.arguments()[1]);

      expr.swap(op);
    }
    else if(identifier == "__builtin_islessequal")
    {
      // this is a gcc extension to provide branch prediction
      if(expr.arguments().size() != 2)
      {
        std::cout << "__builtin_islessequal expects two arguments" << std::endl;
        expr.dump();
        abort();
      }

      exprt op("<=", bool_typet());
      op.copy_to_operands(expr.arguments()[0], expr.arguments()[1]);

      expr.swap(op);
    }
    else if(identifier == "__builtin_islessgreater")
    {
      // this is a gcc extension to provide branch prediction
      if(expr.arguments().size() != 2)
      {
        std::cout << "__builtin_islessgreater expects two arguments"
                  << std::endl;
        expr.dump();
        abort();
      }

      exprt op1("<", bool_typet());
      op1.copy_to_operands(expr.arguments()[0], expr.arguments()[1]);

      exprt op2(">", bool_typet());
      op2.copy_to_operands(expr.arguments()[0], expr.arguments()[1]);

      exprt op("or", bool_typet());
      op.copy_to_operands(op1, op2);

      expr.swap(op);
    }
    else if(identifier == "__builtin_isunordered")
    {
      // this is a gcc extension to provide branch prediction
      if(expr.arguments().size() != 2)
      {
        std::cout << "__builtin_islessequal expects two arguments" << std::endl;
        expr.dump();
        abort();
      }

      exprt op1("isnan", bool_typet());
      op1.copy_to_operands(expr.arguments()[0]);

      exprt op2("isnan", bool_typet());
      op2.copy_to_operands(expr.arguments()[1]);

      exprt op("or", bool_typet());
      op.copy_to_operands(op1, op2);

      expr.swap(op);
    }
    else if(
      identifier == CPROVER_PREFIX "nearbyintf" ||
      identifier == CPROVER_PREFIX "nearbyintd" ||
      identifier == CPROVER_PREFIX "nearbyintld")
    {
      if(expr.arguments().size() != 1)
      {
        std::cout << "nearbyint expects one operand" << std::endl;
        expr.dump();
        abort();
      }

      exprt new_expr("nearbyint", expr.type());
      new_expr.operands() = expr.arguments();
      expr.swap(new_expr);
    }
    else if(
      identifier == CPROVER_PREFIX "fmaf" ||
      identifier == CPROVER_PREFIX "fmad" ||
      identifier == CPROVER_PREFIX "fmald")
    {
      if(expr.arguments().size() != 3)
      {
        std::cout << "fma expects three operand" << std::endl;
        expr.dump();
        abort();
      }

      exprt new_expr("ieee_fma", expr.type());
      new_expr.operands() = expr.arguments();
      expr.swap(new_expr);
    }
    else if(identifier == CPROVER_PREFIX "floatbv_mode")
    {
      exprt new_expr;
      if(config.ansi_c.use_fixed_for_float)
        new_expr = false_exprt();
      else
        new_expr = true_exprt();

      expr.swap(new_expr);
    }
    else if(
      identifier == CPROVER_PREFIX "sqrtf" ||
      identifier == CPROVER_PREFIX "sqrtd" ||
      identifier == CPROVER_PREFIX "sqrtld")
    {
      if(expr.arguments().size() != 1)
      {
        std::cout << "sqrt expects one operand" << std::endl;
        expr.dump();
        abort();
      }

      exprt new_expr("ieee_sqrt", expr.type());
      new_expr.operands() = expr.arguments();
      expr.swap(new_expr);
    }
  }

  // Restore location
  expr.location() = location;
}

void clang_c_adjust::adjust_side_effect_statement_expression(
  side_effect_exprt &expr)
{
  codet &code = to_code(expr.op0());
  assert(code.statement() == "block");

  // the type is the type of the last statement in the
  // block
  codet &last = to_code(code.operands().back());

  irep_idt last_statement = last.get_statement();

  if(last_statement == "expression")
  {
    assert(last.operands().size() == 1);
    expr.type() = last.op0().type();
  }
  else if(last_statement == "function_call")
  {
    // make the last statement an expression

    code_function_callt &fc = to_code_function_call(last);

    side_effect_expr_function_callt sideeffect;

    sideeffect.function() = fc.function();
    sideeffect.arguments() = fc.arguments();
    sideeffect.location() = fc.location();

    sideeffect.type() =
      static_cast<const typet &>(fc.function().type().return_type());

    expr.type() = sideeffect.type();

    if(fc.lhs().is_nil())
    {
      codet code_expr("expression");
      code_expr.location() = fc.location();
      code_expr.move_to_operands(sideeffect);
      last.swap(code_expr);
    }
    else
    {
      codet code_expr("expression");
      code_expr.location() = fc.location();

      exprt assign("sideeffect");
      assign.statement("assign");
      assign.location() = fc.location();
      assign.move_to_operands(fc.lhs(), sideeffect);
      assign.type() = assign.op1().type();

      code_expr.move_to_operands(assign);
      last.swap(code_expr);
    }
  }
  else
    expr.type() = typet("empty");
}

void clang_c_adjust::adjust_expr_unary_boolean(exprt &expr)
{
  adjust_operands(expr);

  expr.type() = bool_type();

  exprt &operand = expr.op0();
  gen_typecast_bool(ns, operand);
}

void clang_c_adjust::adjust_expr_binary_boolean(exprt &expr)
{
  adjust_operands(expr);

  expr.type() = bool_type();

  gen_typecast_bool(ns, expr.op0());
  gen_typecast_bool(ns, expr.op1());
}

void clang_c_adjust::adjust_argc_argv(const symbolt &main_symbol)
{
  const code_typet::argumentst &arguments =
    to_code_type(main_symbol.type).arguments();

  if(arguments.size() == 0)
    return;

  if(arguments.size() != 2 && arguments.size() != 3)
  {
    std::cerr << "main expected to have no or two or three arguments"
              << std::endl;
    abort();
  }

  const exprt &op0 = arguments[0];
  const exprt &op1 = arguments[1];

  symbolt argc_symbol;
  argc_symbol.base_name = "argc";
  argc_symbol.name = "argc'";
  argc_symbol.type = op0.type();
  argc_symbol.static_lifetime = true;
  argc_symbol.lvalue = true;
  argc_symbol.is_used = true;

  symbolt *argc_new_symbol;
  context.move(argc_symbol, argc_new_symbol);

  // need to add one to the size -- the array is terminated
  // with NULL
  exprt one_expr = from_integer(1, argc_new_symbol->type);

  exprt size_expr("+", argc_new_symbol->type);
  size_expr.copy_to_operands(symbol_expr(*argc_new_symbol), one_expr);

  symbolt argv_symbol;
  argv_symbol.base_name = "argv";
  argv_symbol.name = "argv'";
  argv_symbol.type = array_typet(op1.type().subtype(), size_expr);
  argv_symbol.static_lifetime = true;
  argv_symbol.lvalue = true;
  argv_symbol.is_used = true;

  symbolt *argv_new_symbol;
  context.move(argv_symbol, argv_new_symbol);

  if(arguments.size() == 3)
  {
    const exprt &op2 = arguments[2];

    symbolt envp_size_symbol;
    envp_size_symbol.base_name = "envp_size";
    envp_size_symbol.name = "envp_size'";
    envp_size_symbol.type = op0.type(); // same type as argc!
    envp_size_symbol.static_lifetime = true;
    envp_size_symbol.is_used = true;

    symbolt *envp_new_size_symbol;
    context.move(envp_size_symbol, envp_new_size_symbol);

    symbolt envp_symbol;
    envp_symbol.base_name = "envp";
    envp_symbol.name = "envp'";
    envp_symbol.type = op2.type();
    envp_symbol.static_lifetime = true;
    envp_symbol.is_used = true;
    exprt size_expr = symbol_expr(*envp_new_size_symbol);
    envp_symbol.type = array_typet(envp_symbol.type.subtype(), size_expr);

    symbolt *envp_new_symbol;
    context.move(envp_symbol, envp_new_symbol);
  }
}

void clang_c_adjust::adjust_comma(exprt &expr)
{
  adjust_operands(expr);

  expr.type() = expr.op1().type();

  // make this an l-value if the last operand is one
  if(expr.op1().cmt_lvalue())
    expr.cmt_lvalue(true);
}

void clang_c_adjust::adjust_builtin_va_arg(exprt &expr)
{
  // The first parameter is the va_list, and the second
  // is the type, which will need to be fixed and checked.
  // The type is given by the parser as type of the expression.

  typet arg_type = expr.type();

  code_typet new_type;
  new_type.return_type().swap(arg_type);
  new_type.arguments().resize(1);
  new_type.arguments()[0].type() = pointer_typet(empty_typet());

  assert(expr.operands().size() == 1);
  exprt arg = expr.op0();

  gen_typecast(ns, arg, pointer_typet(empty_typet()));

  // turn into function call
  side_effect_expr_function_callt result;
  result.location() = expr.location();
  result.function() = symbol_exprt("builtin_va_arg");
  result.function().location() = expr.location();
  result.function().type() = new_type;
  result.arguments().push_back(arg);
  result.type() = new_type.return_type();

  expr.swap(result);

  // Make sure symbol exists, but we have it return void
  // to avoid collisions of the same symbol with different
  // types.

  code_typet symbol_type = new_type;
  symbol_type.return_type() = empty_typet();

  symbolt symbol;
  symbol.base_name = "builtin_va_arg";
  symbol.name = "builtin_va_arg";
  symbol.type = symbol_type;
  symbol.is_used = true;

  context.move(symbol);
}

void clang_c_adjust::adjust_operands(exprt &expr)
{
  if(!expr.has_operands())
    return;

  for(auto &op : expr.operands())
    adjust_expr(op);
}
