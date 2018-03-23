/*******************************************************************\

Module: Program Transformation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-programs/goto_convert_class.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/rename.h>
#include <util/std_expr.h>

void goto_convertt::make_temp_symbol(exprt &expr, goto_programt &dest)
{
  const locationt location = expr.find_location();

  symbolt &new_symbol = new_tmp_symbol(expr.type());

  code_assignt assignment;
  assignment.lhs() = symbol_expr(new_symbol);
  assignment.rhs() = expr;
  assignment.location() = location;

  convert(assignment, dest);

  expr = symbol_expr(new_symbol);
}

bool goto_convertt::has_sideeffect(const exprt &expr)
{
  forall_operands(it, expr)
    if(has_sideeffect(*it))
      return true;

  if(expr.id() == "sideeffect")
    return true;

  return false;
}

void goto_convertt::remove_sideeffects(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  if(!has_sideeffect(expr))
    return;

  if(expr.is_and() || expr.is_or())
  {
    if(!expr.is_boolean())
      throw expr.id_string() + " must be Boolean, but got " + expr.pretty();

    exprt tmp;

    if(expr.is_and())
      tmp = true_exprt();
    else
      // ID_or
      tmp = false_exprt();

    exprt::operandst &ops = expr.operands();

    // start with last one
    for(exprt::operandst::reverse_iterator it = ops.rbegin(); it != ops.rend();
        ++it)
    {
      exprt &op = *it;

      if(!op.is_boolean())
        throw expr.id().as_string() + " takes boolean operands only";

      if(expr.is_and())
      {
        if_exprt if_e(op, tmp, false_exprt());
        tmp.swap(if_e);
      }
      else // ID_or
      {
        if_exprt if_e(op, true_exprt(), tmp);
        tmp.swap(if_e);
      }
    }

    expr.swap(tmp);

    remove_sideeffects(expr, dest, result_is_used);
    return;
  }
  if(expr.id() == "if")
  {
    // first clean condition
    remove_sideeffects(expr.op0(), dest);

    // possibly done now
    if(
      !has_sideeffect(to_if_expr(expr).true_case()) &&
      !has_sideeffect(to_if_expr(expr).false_case()))
      return;

    // copy expression
    if_exprt if_expr = to_if_expr(expr);

    if(!if_expr.cond().is_boolean())
      throw "first argument of `if' must be boolean, but got ";

    const locationt location = expr.location();

    goto_programt tmp_true;
    remove_sideeffects(if_expr.true_case(), tmp_true, result_is_used);

    goto_programt tmp_false;
    remove_sideeffects(if_expr.false_case(), tmp_false, result_is_used);

    if(result_is_used)
    {
      symbolt &new_symbol = new_tmp_symbol(expr.type());

      code_assignt assignment_true;
      assignment_true.lhs() = symbol_expr(new_symbol);
      assignment_true.rhs() = if_expr.true_case();
      assignment_true.location() = location;
      convert(assignment_true, tmp_true);

      code_assignt assignment_false;
      assignment_false.lhs() = symbol_expr(new_symbol);
      assignment_false.rhs() = if_expr.false_case();
      assignment_false.location() = location;
      convert(assignment_false, tmp_false);

      // overwrites expr
      expr = symbol_expr(new_symbol);
    }
    else
    {
      // preserve the expressions for possible later checks
      if(if_expr.true_case().is_not_nil())
      {
        code_expressiont code_expression(if_expr.true_case());
        convert(code_expression, tmp_true);
      }

      if(if_expr.false_case().is_not_nil())
      {
        code_expressiont code_expression(if_expr.false_case());
        convert(code_expression, tmp_false);
      }

      expr = nil_exprt();
    }

    // generate guard for argument side-effects
    generate_ifthenelse(if_expr.cond(), tmp_true, tmp_false, location, dest);

    return;
  }
  else if(expr.id() == "comma")
  {
    if(result_is_used)
    {
      exprt result;

      Forall_operands(it, expr)
      {
        bool last = (it == --expr.operands().end());

        // special treatment for last one
        if(last)
        {
          result.swap(*it);
          remove_sideeffects(result, dest, true);
        }
        else
        {
          remove_sideeffects(*it, dest, false);

          // remember these for later checks
          if(it->is_not_nil())
            convert(code_expressiont(*it), dest);
        }
      }

      expr.swap(result);
    }
    else // result not used
    {
      Forall_operands(it, expr)
      {
        remove_sideeffects(*it, dest, false);

        // remember as expression statement for later checks
        if(it->is_not_nil())
          convert(code_expressiont(*it), dest);
      }

      expr = nil_exprt();
    }

    return;
  }
  else if(expr.id() == "typecast")
  {
    if(expr.operands().size() != 1)
      throw "typecast takes one argument";

    // preserve 'result_is_used'
    remove_sideeffects(expr.op0(), dest, result_is_used);

    if(expr.op0().is_nil())
      expr.make_nil();

    return;
  }
  else if(expr.id() == "sideeffect")
  {
    // some of the side-effects need special treatment!
    const irep_idt statement = expr.statement();
    if(statement == "gcc_conditional_expression")
    {
      remove_gcc_conditional_expression(expr, dest);
      return;
    }
    if(statement == "statement_expression")
    {
      remove_statement_expression(expr, dest, result_is_used);
      return;
    }
    else if(statement == "assign")
    {
      // we do a special treatment for x=f(...)
      assert(expr.operands().size() == 2);

      if(
        expr.op1().id() == "sideeffect" &&
        to_side_effect_expr(expr.op1()).get_statement() == "function_call")
      {
        remove_sideeffects(expr.op0(), dest);
        exprt lhs = expr.op0();

        // turn into code
        code_assignt assignment;
        assignment.lhs() = lhs;
        assignment.rhs() = expr.op1();
        assignment.location() = expr.location();
        convert_assign(assignment, dest);

        if(result_is_used)
          expr.swap(lhs);
        else
          expr.make_nil();
        return;
      }
    }
  }

  // TODO: evaluation order
  Forall_operands(it, expr)
    remove_sideeffects(*it, dest);

  if(expr.id() == "sideeffect")
  {
    const irep_idt &statement = expr.statement();

    if(statement == "function_call") // might do anything
      remove_function_call(expr, dest, result_is_used);
    else if(
      statement == "assign" || statement == "assign+" ||
      statement == "assign-" || statement == "assign*" ||
      statement == "assign_div" || statement == "assign_bitor" ||
      statement == "assign_bitxor" || statement == "assign_bitand" ||
      statement == "assign_lshr" || statement == "assign_ashr" ||
      statement == "assign_shl" || statement == "assign_mod")
      remove_assignment(expr, dest, result_is_used);
    else if(statement == "postincrement" || statement == "postdecrement")
      remove_post(expr, dest, result_is_used);
    else if(statement == "preincrement" || statement == "predecrement")
      remove_pre(expr, dest, result_is_used);
    else if(statement == "cpp_new" || statement == "cpp_new[]")
      remove_cpp_new(expr, dest, result_is_used);
    else if(statement == "temporary_object")
      remove_temporary_object(expr, dest);
    else if(statement == "nondet")
    {
      // these are fine
    }
    else if(statement == "skip")
    {
      expr.make_nil();
    }
    else if(statement == "cpp-throw")
    {
      goto_programt::targett t = dest.add_instruction(THROW);
      codet tmp("cpp-throw");
      tmp.operands().swap(expr.operands());
      tmp.location() = expr.location();
      tmp.set("exception_list", expr.find("exception_list"));
      migrate_expr(tmp, t->code);
      t->location = expr.location();

      // the result can't be used, these are void
      expr.make_nil();
    }
    else
    {
      str << "cannot remove side effect (" << statement << ")";
      throw 0;
    }
  }
}

void goto_convertt::remove_assignment(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  const irep_idt &statement = expr.statement();

  if(statement == "assign")
  {
    exprt tmp = expr;
    tmp.id("code");
    convert_assign(to_code_assign(to_code(tmp)), dest);
  }
  else if(
    statement == "assign+" || statement == "assign-" ||
    statement == "assign*" || statement == "assign_div" ||
    statement == "assign_mod" || statement == "assign_shl" ||
    statement == "assign_ashr" || statement == "assign_lshr" ||
    statement == "assign_bitand" || statement == "assign_bitxor" ||
    statement == "assign_bitor")
  {
    if(expr.operands().size() != 2)
    {
      err_location(expr);
      str << statement << " takes two arguments";
      throw 0;
    }

    exprt rhs;

    if(statement == "assign+")
    {
      if(expr.type().is_floatbv())
      {
        rhs.id("ieee_add");
      }
      else
      {
        rhs.id("+");
      }
    }
    else if(statement == "assign-")
    {
      if(expr.type().is_floatbv())
      {
        rhs.id("ieee_sub");
      }
      else
      {
        rhs.id("-");
      }
    }
    else if(statement == "assign*")
    {
      if(expr.type().is_floatbv())
      {
        rhs.id("ieee_mul");
      }
      else
      {
        rhs.id("*");
      }
    }
    else if(statement == "assign_div")
    {
      if(expr.type().is_floatbv())
      {
        rhs.id("ieee_div");
      }
      else
      {
        rhs.id("/");
      }
    }
    else if(statement == "assign_mod")
    {
      rhs.id("mod");
    }
    else if(statement == "assign_shl")
    {
      rhs.id("shl");
    }
    else if(statement == "assign_ashr")
    {
      rhs.id("ashr");
    }
    else if(statement == "assign_lshr")
    {
      rhs.id("lshr");
    }
    else if(statement == "assign_bitand")
    {
      rhs.id("bitand");
    }
    else if(statement == "assign_bitxor")
    {
      rhs.id("bitxor");
    }
    else if(statement == "assign_bitor")
    {
      rhs.id("bitor");
    }
    else
    {
      err_location(expr);
      str << statement << " not yet supported";
      throw 0;
    }

    rhs.copy_to_operands(expr.op0(), expr.op1());
    rhs.type() = expr.op0().type();

    if(rhs.op0().type().is_bool())
    {
      rhs.op0().make_typecast(int_type());
      rhs.op1().make_typecast(int_type());
      rhs.type() = int_type();
      rhs.make_typecast(typet("bool"));
    }

    exprt lhs(expr.op0());

    code_assignt assignment(lhs, rhs);
    assignment.location() = expr.location();

    convert(assignment, dest);
  }

  // revert assignment in the expression to its LHS
  if(result_is_used)
  {
    exprt lhs;
    lhs.swap(expr.op0());
    expr.swap(lhs);
  }
  else
    expr.make_nil();
}

void goto_convertt::remove_pre(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  const irep_idt statement = expr.statement();

  assert(statement == "preincrement" || statement == "predecrement");

  if(expr.operands().size() != 1)
  {
    err_location(expr);
    str << statement << " takes one argument";
    throw 0;
  }

  exprt rhs;
  rhs.location() = expr.location();

  if(statement == "preincrement")
  {
    if(expr.type().is_floatbv())
      rhs.id("ieee_add");
    else
      rhs.id("+");
  }
  else
  {
    if(expr.type().is_floatbv())
      rhs.id("ieee_sub");
    else
      rhs.id("-");
  }

  const typet &op_type = ns.follow(expr.op0().type());

  if(op_type.is_bool())
  {
    rhs.copy_to_operands(expr.op0(), gen_one(int_type()));
    rhs.op0().make_typecast(int_type());
    rhs.type() = int_type();
    rhs.make_typecast(bool_type());
  }
  else if(op_type.id() == "c_enum" || op_type.id() == "incomplete_c_enum")
  {
    rhs.copy_to_operands(expr.op0(), gen_one(int_type()));
    rhs.op0().make_typecast(int_type());
    rhs.type() = int_type();
    rhs.make_typecast(op_type);
  }
  else
  {
    typet constant_type;

    if(op_type.is_pointer())
      constant_type = index_type();
    else if(is_number(op_type))
      constant_type = op_type;
    else
    {
      err_location(expr);
      throw "no constant one of type " + op_type.to_string();
    }

    exprt constant = gen_one(constant_type);

    rhs.copy_to_operands(expr.op0());
    rhs.move_to_operands(constant);
    rhs.type() = expr.op0().type();
  }

  code_assignt assignment(expr.op0(), rhs);
  assignment.location() = expr.location();

  convert(assignment, dest);

  if(result_is_used)
  {
    // revert to argument of pre-inc/pre-dec
    exprt tmp = expr.op0();
    expr.swap(tmp);
  }
  else
    expr.make_nil();
}

void goto_convertt::remove_post(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  const irep_idt statement = expr.statement();

  assert(statement == "postincrement" || statement == "postdecrement");

  if(expr.operands().size() != 1)
  {
    err_location(expr);
    str << statement << " takes one argument";
    throw 0;
  }

  exprt rhs;
  rhs.location() = expr.location();

  if(statement == "postincrement")
  {
    if(expr.type().is_floatbv())
      rhs.id("ieee_add");
    else
      rhs.id("+");
  }
  else
  {
    if(expr.type().is_floatbv())
      rhs.id("ieee_sub");
    else
      rhs.id("-");
  }

  const typet &op_type = ns.follow(expr.op0().type());

  if(op_type.is_bool())
  {
    rhs.copy_to_operands(expr.op0(), gen_one(int_type()));
    rhs.op0().make_typecast(int_type());
    rhs.type() = int_type();
    rhs.make_typecast(bool_type());
  }
  else if(op_type.id() == "c_enum" || op_type.id() == "incomplete_c_enum")
  {
    rhs.copy_to_operands(expr.op0(), gen_one(int_type()));
    rhs.op0().make_typecast(int_type());
    rhs.type() = int_type();
    rhs.make_typecast(op_type);
  }
  else
  {
    typet constant_type;

    if(op_type.is_pointer())
      constant_type = index_type();
    else if(is_number(op_type))
      constant_type = op_type;
    else
    {
      err_location(expr);
      throw "no constant one of type " + op_type.to_string();
    }

    exprt constant = gen_one(constant_type);

    rhs.copy_to_operands(expr.op0());
    rhs.move_to_operands(constant);
    rhs.type() = expr.op0().type();
  }

  code_assignt assignment(expr.op0(), rhs);
  assignment.location() = expr.location();

  goto_programt tmp;
  convert(assignment, tmp);

  // fix up the expression, if needed

  if(result_is_used)
  {
    exprt tmp = expr.op0();
    make_temp_symbol(tmp, dest);
    expr.swap(tmp);
  }
  else
    expr.make_nil();

  dest.destructive_append(tmp);
}

void goto_convertt::remove_function_call(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  if(!result_is_used)
  {
    assert(expr.operands().size() == 2);
    code_function_callt call;
    call.function() = expr.op0();
    call.arguments() = expr.op1().operands();
    call.location() = expr.location();
    call.lhs().make_nil();
    convert_function_call(call, dest);
    expr.make_nil();
    return;
  }

  symbolt new_symbol;

  new_symbol.base_name = "return_value$";
  new_symbol.type = expr.type();
  new_symbol.location = expr.location();

  // get name of function, if available

  if(expr.id() != "sideeffect" || expr.statement() != "function_call")
    throw "expected function call";

  if(expr.operands().empty())
    throw "function_call expects at least one operand";

  if(expr.op0().is_symbol())
  {
    const irep_idt &identifier = expr.op0().identifier();
    const symbolt &symbol = ns.lookup(identifier);

    std::string new_base_name = id2string(new_symbol.base_name);

    new_base_name += '_';
    new_base_name += id2string(symbol.base_name);
    new_base_name += "$" + std::to_string(++temporary_counter);

    new_symbol.base_name = new_base_name;
    new_symbol.mode = symbol.mode;
  }

  new_symbol.name = tmp_symbol_prefix + id2string(new_symbol.base_name);
  new_name(new_symbol);
  scoped_variables.push_front(new_symbol.name);

  code_function_callt call;
  call.lhs() = symbol_expr(new_symbol);
  call.function() = expr.op0();
  call.arguments() = expr.op1().operands();
  call.location() = new_symbol.location;

  codet assignment("assign");
  assignment.reserve_operands(2);
  assignment.copy_to_operands(symbol_expr(new_symbol));
  assignment.move_to_operands(call);

  goto_programt tmp_program;
  convert(assignment, tmp_program);
  dest.destructive_append(tmp_program);

  expr = symbol_expr(new_symbol);
}

void goto_convertt::replace_new_object(const exprt &object, exprt &dest)
{
  if(dest.id() == "new_object")
    dest = object;
  else
    Forall_operands(it, dest)
      replace_new_object(object, *it);
}

void goto_convertt::remove_cpp_new(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  codet call;

  symbolt new_symbol;

  new_symbol.base_name = "new_ptr$" + std::to_string(++temporary_counter);
  new_symbol.type = expr.type();
  new_symbol.name = tmp_symbol_prefix + id2string(new_symbol.base_name);

  new_name(new_symbol);

  code_declt decl(symbol_expr(new_symbol));
  decl.location() = new_symbol.location;
  convert_decl(decl, dest);

  call = code_assignt(symbol_expr(new_symbol), expr);

  if(result_is_used)
    expr = symbol_expr(new_symbol);
  else
    expr.make_nil();

  convert(call, dest);
}

void goto_convertt::remove_temporary_object(exprt &expr, goto_programt &dest)
{
  if(expr.operands().size() != 1 && expr.operands().size() != 0)
    throw "temporary_object takes 0 or 1 operands";

  symbolt &new_symbol = new_tmp_symbol(expr.type());

  new_symbol.mode = expr.mode();

  if(expr.operands().size() == 1)
  {
    codet assignment("assign");
    assignment.reserve_operands(2);
    assignment.copy_to_operands(symbol_expr(new_symbol));
    assignment.move_to_operands(expr.op0());

    goto_programt tmp_program;
    convert(assignment, tmp_program);
    dest.destructive_append(tmp_program);
  }

  if(expr.initializer().is_not_nil())
  {
    assert(expr.operands().empty());
    exprt initializer = static_cast<const exprt &>(expr.initializer());
    replace_new_object(symbol_expr(new_symbol), initializer);

    goto_programt tmp_program;
    convert(to_code(initializer), tmp_program);
    dest.destructive_append(tmp_program);
  }

  expr = symbol_expr(new_symbol);
}

void goto_convertt::remove_statement_expression(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  if(expr.operands().size() != 1)
    throw "statement_expression takes 1 operand";

  if(!expr.op0().is_code())
    throw "statement_expression takes code as operand";

  codet &code = to_code(expr.op0());

  if(!result_is_used)
  {
    convert(code, dest);
    expr.make_nil();
    return;
  }

  // get last statement from block
  if(code.get_statement() != "block")
    throw "statement_expression expects block";

  if(code.operands().empty())
    throw "statement_expression expects non-empty block";

  exprt &last = code.operands().back();
  locationt location = last.location();

  symbolt &new_symbol = new_tmp_symbol(expr.type());

  symbol_exprt tmp_symbol_expr(new_symbol.name, new_symbol.type);
  tmp_symbol_expr.location() = location;

  if(last.statement() == "expression")
  {
    // we turn this into an assignment
    exprt e = to_code_expression(to_code(last)).expression();
    last = code_assignt(tmp_symbol_expr, e);
    last.location() = location;
  }
  else if(last.statement() == "assign")
  {
    exprt e = to_code_assign(to_code(last)).lhs();
    code_assignt assignment(tmp_symbol_expr, e);
    assignment.location() = location;
    code.operands().push_back(assignment);
  }
  else
    throw "statement_expression expects expression or assignment";

  {
    goto_programt tmp;
    convert(code, tmp);
    dest.destructive_append(tmp);
  }

  expr = tmp_symbol_expr;
}

void goto_convertt::remove_gcc_conditional_expression(
  exprt &expr,
  goto_programt &dest)
{
  if(expr.operands().size() != 2)
    throw "conditional_expression takes two operands";

  // first remove side-effects from condition
  remove_sideeffects(expr.op0(), dest);

  if_exprt if_expr;

  if_expr.cond() = expr.op0();
  if_expr.true_case() = expr.op0();
  if_expr.false_case() = expr.op1();
  if_expr.type() = expr.type();
  if_expr.location() = expr.location();

  if(!if_expr.op0().type().is_bool())
    if_expr.op0().make_typecast(bool_typet());

  expr.swap(if_expr);

  // there might still be one in expr.op2()
  remove_sideeffects(expr, dest);
}
