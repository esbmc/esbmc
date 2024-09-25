#include <clang-cpp-frontend/clang_cpp_adjust.h>

void clang_cpp_adjust::convert_expression_to_code(exprt &expr)
{
  if (expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

void clang_cpp_adjust::adjust_code(codet &code)
{
  const irep_idt &statement = code.statement();

  if (statement == "cpp-catch")
  {
    adjust_catch(code);
  }
  else if (statement == "throw_decl")
  {
    adjust_throw_decl(code);
  }
  else
    clang_c_adjust::adjust_code(code);
}

void clang_cpp_adjust::adjust_ifthenelse(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // if(void *p=...) ...
  if (code.op0().is_code())
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
  if (code.op0().is_code())
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
  if (code.op0().is_code())
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
  if (code.op1().is_code())
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
  else if (code.op0().get_bool("for_range"))
  {
    clang_c_adjust::adjust_for(code);
    // speaial case: for (int num : arr)
    // it has several inits: __range1 = &arr[0]
    // __begin1 = (&(*__range1)[0])
    // __end1 = &(*__range1)[0] + size
    codet decls = to_code(code.op0());

    code_blockt code_block;
    Forall_operands (it, decls)
    {
      adjust_expr(*it);
      codet &code_decl = to_code(*it);

      code_block.move_to_operands(code_decl);
    }

    code_block.move_to_operands(code.op1());
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_for(code);
}

void clang_cpp_adjust::adjust_decl_block(codet &code)
{
  codet new_block("decl-block");

  Forall_operands (it, code)
  {
    if (it->is_code() && it->statement() == "skip")
      continue;

    adjust_expr(*it);
    code_declt &code_decl = to_code_decl(to_code(*it));

    new_block.copy_to_operands(code_decl);
  }

  code.swap(new_block);
}

void clang_cpp_adjust::adjust_catch(codet &code)
{
  codet::operandst &operands = code.operands();
  // adjust try block
  adjust_expr(operands[0]);

  // First operand is always the try block, skip it
  for (auto it = ++operands.begin(); it != operands.end(); it++)
  {
    // The following operands are the catchs
    adjust_expr(*it);
    code_blockt &block = to_code_block(to_code(*it));

    std::vector<irep_idt> ids;
    convert_exception_id(block.type(), "", ids);

    block.type() = code_typet();
    block.set("exception_id", ids.front());
  }
}

void clang_cpp_adjust::adjust_throw_decl(codet &code)
{
  codet::operandst &operands = code.operands();

  for (auto &op : operands)
  {
    std::vector<irep_idt> ids;
    convert_exception_id(op.type(), "", ids);

    op.type() = code_typet();
    op.set("throw_decl_id", ids.front());
  }
}
