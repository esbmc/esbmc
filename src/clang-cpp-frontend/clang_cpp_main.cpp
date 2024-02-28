#include <clang-cpp-frontend/clang_cpp_main.h>
#include <util/std_expr.h>

clang_cpp_maint::clang_cpp_maint(contextt &_context) : clang_c_maint(_context)
{
}

void clang_cpp_maint::adjust_init(code_assignt &assignment, codet &adjusted)
{
  // adjust the init statement for global variables
  assert(assignment.operands().size() == 2);

  exprt &rhs = assignment.rhs();
  if (
    rhs.id() == "sideeffect" && rhs.statement() == "function_call" &&
    rhs.get_bool("constructor"))
  {
    // First, create new decl without rhs
    code_declt object(assignment.lhs());
    adjusted.copy_to_operands(object);

    // Get rhs - this represents the constructor call
    side_effect_expr_function_callt &init =
      to_side_effect_expr_function_call(rhs);

    // Get lhs - this represents the `this` pointer
    exprt::operandst &rhs_args = init.arguments();
    // the original lhs needs to be the first arg, then followed by others:
    //  BLAH(&bleh, arg1, arg2, ...);
    rhs_args[0] = address_of_exprt(assignment.lhs());

    // Now convert the side_effect into an expression
    convert_expression_to_code(init);

    // and copy to adjusted
    adjusted.copy_to_operands(init);
  }
}

void clang_cpp_maint::convert_expression_to_code(exprt &expr)
{
  if (expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}
