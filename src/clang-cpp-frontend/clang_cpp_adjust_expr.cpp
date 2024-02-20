#include <clang-c-frontend/typecast.h>
#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <util/c_sizeof.h>
#include <util/c_types.h>
#include <util/destructor.h>
#include <util/expr_util.h>

clang_cpp_adjust::clang_cpp_adjust(contextt &_context)
  : clang_c_adjust(_context)
{
}

void clang_cpp_adjust::adjust_symbol(symbolt &symbol)
{
  clang_c_adjust::adjust_symbol(symbol);

  /*
   * implicit code generation for vptr initializations:
   * The idea is to get the constructor method symbol and
   * add implicit code to set each virtual pointer of this
   * class to point to the corresponding virtual table.
   */
  gen_vptr_initializations(symbol);
}

void clang_cpp_adjust::adjust_side_effect(side_effect_exprt &expr)
{
  const irep_idt &statement = expr.statement();

  if (statement == "cpp_new" || statement == "cpp_new[]")
  {
    adjust_new(expr);
  }
  else if (statement == "cpp_delete" || statement == "cpp_delete[]")
  {
    // adjust side effect node to explicitly call class destructor
    // e.g. the adjustment here will add the following instruction in GOTO:
    // FUNCTION_CALL:  ~t2(&(*p))
    code_function_callt destructor = get_destructor(ns, expr.type());
    if (destructor.is_not_nil())
    {
      exprt new_object("new_object", expr.type());
      new_object.cmt_lvalue(true);

      destructor.arguments().push_back(address_of_exprt(new_object));
      expr.set("destructor", destructor);
    }
  }
  else if (statement == "temporary_object")
  {
    exprt &initializer = static_cast<exprt &>(expr.add("initializer"));

    side_effect_expr_function_callt &constructor_call =
      to_side_effect_expr_function_call(initializer.op0());

    adjust_function_call_arguments(constructor_call);
  }
  else if (statement == "assign")
  {
    adjust_side_effect_assign(expr);
  }
  else
    clang_c_adjust::adjust_side_effect(expr);
}

void clang_cpp_adjust::adjust_new(exprt &expr)
{
  if (expr.initializer().is_not_nil())
  {
    exprt &initializer = static_cast<exprt &>(expr.add("initializer"));
    adjust_expr(initializer);
  }

  if (expr.size_irep().is_not_nil())
  {
    exprt new_size = static_cast<const exprt &>(expr.size_irep());
    adjust_expr(new_size);
    expr.size(new_size);
  }

  // Set sizeof and cmt_sizeof_type
  exprt size_of = c_sizeof(expr.type().subtype(), ns);
  size_of.set("#c_sizeof_type", expr.type().subtype());

  expr.set("sizeof", size_of);
}

void clang_cpp_adjust::adjust_member(member_exprt &expr)
{
  clang_c_adjust::adjust_member(expr);

  /*
   * Additional adjustment is required for C++ class member access:
   * e.g. when we got a class/struct member function call via:
   * dot operator, e.g. OBJECT.setX();
   * or arrow operator, e.g.OBJECT->setX();
   */
  if (expr.type().is_code() && !expr.get_string("component_name").empty())
  {
    adjust_cpp_member(expr);
  }
}

void clang_cpp_adjust::adjust_cpp_member(member_exprt &expr)
{
  /*
   * For class member function call:
   * Replace OBJECT.setX() OR OBJECT->setX() with setX(), where
   * OBJECT.setX() is represented by:
   *    member:
   *      * type: ...
   *      * operands:
   *      * symbol: <object_symbol>
   *      * component_name: <setX_clang_ID>
   *  OBJECT->setX() is represented by:
   *    member:
   *      * type: ...
   *      * operands:
   *        * dereference: <object_ptr>
   *      * component_name: <setX_clang_ID>
   *  and setX() is represented by the symbol expr:
   *    symbol:
   *      * type: ...
   *      * id: <setX_clang_ID>
   */
  const symbolt *comp_symb = namespacet(context).lookup(expr.component_name());
  assert(comp_symb);
  // compoment's type shall be the same as member_exprt's type
  // and both are of the type `code`
  assert(comp_symb->type.is_code());
  exprt method_call = symbol_expr(*comp_symb);
  expr.swap(method_call);
}

void clang_cpp_adjust::adjust_if(exprt &expr)
{
  // Check all operands
  adjust_operands(expr);

  // If the condition is not of boolean type, it must be casted
  gen_typecast(ns, expr.op0(), bool_type());

  // Typecast both the true and false results
  gen_typecast(ns, expr.op1(), expr.type());
  gen_typecast(ns, expr.op2(), expr.type());
}

void clang_cpp_adjust::adjust_side_effect_assign(side_effect_exprt &expr)
{
  // sideeffect assign got be representing a binary operator
  assert(expr.operands().size() == 2);

  exprt &lhs = expr.op0();
  exprt &rhs = expr.op1();

  if (
    rhs.id() == "sideeffect" && rhs.statement() == "function_call" &&
    rhs.get_bool("constructor"))
  {
    // turn assign expression bleh = BLAH() into one instruction:
    // BLAH(&bleh);
    // where bleh has been declared and there exists a corresponding symbol
    side_effect_expr_function_callt &rhs_func_call =
      to_side_effect_expr_function_call(rhs);

    // callee must be a constructor
    assert(rhs_func_call.function().type().return_type().id() == "constructor");

    // just populate rhs' argument and replace the entire expression
    exprt &lhs = expr.op0();
    exprt arg = address_of_exprt(lhs);
    exprt base_symbol = arg.op0();
    assert(base_symbol.op0().id() == "symbol");
    // TODO: wrap base symbol into dereference if it's a member
    rhs_func_call.arguments().push_back(arg);

    expr.swap(rhs);

    // let's go through C's side_effect_function_call to make sure
    // there's no missing adjustment we need to carry out
    clang_c_adjust::adjust_side_effect_function_call(
      to_side_effect_expr_function_call(expr));
  }
  else if (
    lhs.is_symbol() &&
    (is_reference(lhs.type()) || is_rvalue_reference(lhs.type())))
  {
    // since we modelled lvalue reference as pointers
    // turn assign expression r = 1, where r is an lvalue reference
    // into *r = 1
    convert_ref_to_deref_symbol(lhs);
  }
  else if (lhs.id() == "sideeffect" && lhs.statement() == "function_call")
  {
    // deal with X(a) = 5; where X(a) returns an lvalue reference which
    // is modelled as a pointer. Hence we got to align the LHS with RHS:
    // *X(a) = 5;
    // rather than align RHS with LHS:
    // X(a) = (int &)5; // which is far removed from the original symatics

    // first adjust LHS to make sure its type aligns with the
    // function return type
    adjust_side_effect_function_call(to_side_effect_expr_function_call(lhs));
    if (is_reference(lhs.type()))
    {
      convert_lvalue_ref_to_deref_sideeffect(lhs);
    }
    adjust_expr(rhs);
  }
  else
    clang_c_adjust::adjust_side_effect(expr);
}

void clang_cpp_adjust::adjust_expr_rel(exprt &expr)
{
  clang_c_adjust::adjust_expr_rel(expr);

  exprt &op0 = expr.op0();
  if (op0.is_typecast())
  {
    // special treatment for lvalue reference typecasting
    // if lhs is a typecast of lvalue reference, e.g. (int)r == 1
    // where r is a reference
    // we turn it into (int)*r == 1
    exprt &tp_op0 = op0.op0();
    if (
      tp_op0.is_symbol() &&
      (is_reference(tp_op0.type()) || is_rvalue_reference(tp_op0.type())))
      convert_ref_to_deref_symbol(tp_op0);
  }
  if (is_reference(op0.type()) || is_rvalue_reference(op0.type()))
  {
    // special treatment for lvalue reference
    // if LHS is an lvalue reference, e.g.
    //  r == 1 or F(a) == 1 where F is a function that returns a reference
    // but RHS is not a pointer. We got to dereference the LHS
    // and turn it into:
    //  *r == 1 *F(a) == 1
    dereference_exprt tmp_deref(op0, op0.type());
    tmp_deref.location() = op0.location();
    tmp_deref.set("#lvalue", true);
    tmp_deref.set("#implicit", true);
    op0.swap(tmp_deref);
  }
}

void clang_cpp_adjust::convert_ref_to_deref_symbol(exprt &expr)
{
  assert(
    expr.is_symbol() &&
    (is_reference(expr.type()) || is_rvalue_reference(expr.type())));

  dereference_exprt tmp(expr, expr.type());
  tmp.location() = expr.location();
  expr.swap(tmp);
}

void clang_cpp_adjust::convert_lvalue_ref_to_deref_sideeffect(exprt &expr)
{
  assert(expr.id() == "sideeffect" && is_reference(expr.type()));
  dereference_exprt tmp_deref(expr, expr.type());
  tmp_deref.location() = expr.location();
  tmp_deref.set("#lvalue", true);
  tmp_deref.set("#implicit", true);
  expr.swap(tmp_deref);
}

void clang_cpp_adjust::adjust_function_call_arguments(
  side_effect_expr_function_callt &expr)
{
  clang_c_adjust::adjust_function_call_arguments(expr);

  exprt::operandst &arguments = expr.arguments();

  for (unsigned i = 0; i < arguments.size(); i++)
  {
    exprt &op = arguments[i];
    if (op.is_typecast() && op.type().get_bool("#reference"))
    {
      // special treatment for lvalue reference in function parameter list
      // e.g. for F(a), where F is a function taking an int lvalue reference.
      // Since we've modelled lvalue references as pointers, clang_c_aduster
      // would have adjusted the parameter to a typecasting expression:
      //  F((int &)a);
      // but what we really want is an address_of expression:
      //  F(*a);
      address_of_exprt tmp_expr;
      assert(op.type().is_pointer());
      tmp_expr.type() = op.type();
      tmp_expr.operands().resize(0);
      tmp_expr.move_to_operands(op.op0());
      tmp_expr.location() = op.location();
      op.swap(tmp_expr);
    }
  }
}

void clang_cpp_adjust::align_se_function_call_return_type(
  exprt &f_op,
  side_effect_expr_function_callt &expr)
{
  // align the side effect's type at callsite with the
  // function return type. But ignore constructors
  const typet &return_type = (typet &)f_op.type().return_type();
  if (return_type.id() != "constructor")
    expr.type() = return_type;
}
