#include <clang-cpp-frontend/clang_cpp_adjust.h>

void clang_cpp_adjust::convert_expression_to_code(exprt &expr)
{
  if(expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

void clang_cpp_adjust::adjust_ifthenelse(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // if(void *p=...) ...
  if(code.op0().is_code())
  {
    codet decl_block = to_code(code.op0());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // replace declaration by its symbol
    code_declt decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());

    code.op0() = decl.op0();
    clang_c_adjust::adjust_ifthenelse(code);

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block.op0(), code);
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_ifthenelse(code);
}

void clang_cpp_adjust::adjust_while(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // while(void *p=...) ...
  if(code.op0().is_code())
  {
    codet decl_block = to_code(code.op0());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // replace declaration by its symbol
    code_declt decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());

    code.op0() = decl.op0();
    clang_c_adjust::adjust_while(code);

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block.op0(), code);
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_while(code);
}

void clang_cpp_adjust::adjust_switch(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // switch(int i=...) ...
  if(code.op0().is_code())
  {
    codet decl_block = to_code(code.op0());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // replace declaration by its symbol
    code_declt decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());

    code.op0() = decl.op0();
    clang_c_adjust::adjust_switch(code);

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block.op0(), code);
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_switch(code);
}

void clang_cpp_adjust::adjust_for(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // for( ; int i=...; ) ...
  if(code.op1().is_code())
  {
    codet decl_block = to_code(code.op1());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // Create new cond assignment
    code_declt &decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());
    assert(decl.operands().size() == 2);

    side_effect_exprt new_cond("assign", decl.op0().type());
    new_cond.copy_to_operands(decl.op0(), decl.op1());
    adjust_expr(new_cond);

    code.op1() = new_cond;
    clang_c_adjust::adjust_for(code);

    // Remove assignment
    decl.operands().pop_back();

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block, code);
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_for(code);
}

void clang_cpp_adjust::adjust_decl_block(codet &code)
{
  codet new_block("decl-block");

  Forall_operands(it, code)
  {
    if(it->is_code() && (it->statement() == "skip"))
      continue;

    code_declt &code_decl = to_code_decl(to_code(*it));

    if(code_decl.operands().size() == 2)
    {
      exprt &rhs = code_decl.rhs();
      exprt &lhs = code_decl.lhs();
      if(
        rhs.id() == "sideeffect" && rhs.statement() == "function_call" &&
        rhs.get_bool("constructor"))
      {
        // turn struct BLAH bleh = BLAH() into two instructions:
        // struct BLAH bleh;
        // BLAH(&bleh);

        // First, create new decl without rhs
        code_declt object(code_decl.lhs());
        new_block.copy_to_operands(object);

        // Get rhs - this represents the constructor call
        side_effect_expr_function_callt &init =
          to_side_effect_expr_function_call(rhs);

        // Get lhs - this represents the `this` pointer
        exprt::operandst &rhs_args = init.arguments();
        // the original lhs needs to be the first arg, then followed by others:
        //  BLAH(&bleh, arg1, arg2, ...);
        rhs_args.insert(rhs_args.begin(), address_of_exprt(code_decl.lhs()));

        // Now convert the side_effect into an expression
        convert_expression_to_code(init);

        // and copy to new_block
        new_block.copy_to_operands(init);

        continue;
      }

      if(lhs.type().get_bool("#reference"))
      {
        // adjust rhs to address_off:
        // `int &r = g;` is turned into `int &r = &g;`
        exprt result_expr = exprt("address_of", rhs.type());
        result_expr.copy_to_operands(rhs.op0());
        rhs.swap(result_expr);
      }
    }

    new_block.copy_to_operands(code_decl);
  }

  code.swap(new_block);
}
