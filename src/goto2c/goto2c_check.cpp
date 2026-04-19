#include <goto2c/goto2c.h>

void goto2ct::check()
{
  // Iterating through all available functions
  for (auto &goto_function : goto_functions.function_map)
    check(goto_function.first.as_string(), goto_function.second);
}

void goto2ct::check(
  std::string function_id [[maybe_unused]],
  goto_functiont &goto_function)
{
  if (!goto_function.body_available)
    return;

  // Checks for when the function body is available
  assert(
    goto_function.body.instructions.size() > 0 &&
    "A non-empty GOTO must contain at least one instruction");
  assert(
    goto_function.body.instructions.back().type == END_FUNCTION &&
    "Last instruction in the GOTO function body must be END_FUNCTION");
  check(goto_function.body);
}

void goto2ct::check(goto_programt &goto_program)
{
  for (auto instruction : goto_program.instructions)
    check(instruction);
}

void goto2ct::check(goto_programt::instructiont &instruction)
{
  // Fedor: locations cannot be empty
  // Fedor: do something for targets here

  // Only GOTO instructions can have targets
  if (instruction.type != GOTO)
    assert(
      !instruction.has_target() &&
      "A non-GOTO instruction cannot have targets");

  // Now specific checks for each instruction
  switch (instruction.type)
  {
  case NO_INSTRUCTION_TYPE:
    break;
  case ASSERT:
    check_assert(instruction);
    break;
  case ASSUME:
    check_assume(instruction);
    break;
  case GOTO:
    check_goto(instruction);
    break;
  case FUNCTION_CALL:
    check_function_call(instruction);
    break;
  case RETURN:
    check_return(instruction);
    break;
  case END_FUNCTION:
    check_end_function(instruction);
    break;
  case DECL:
    check_decl(instruction);
    break;
  case DEAD:
    check_dead(instruction);
    break;
  case ASSIGN:
    check_assign(instruction);
    break;
  case LOCATION:
    check_location(instruction);
    break;
  case SKIP:
    check_skip(instruction);
    break;
  case THROW:
    check_throw(instruction);
    break;
  case CATCH:
    check_catch(instruction);
    break;
  case ATOMIC_BEGIN:
    check_atomic_begin(instruction);
    break;
  case ATOMIC_END:
    check_atomic_end(instruction);
    break;
  case THROW_DECL:
    check_throw_decl(instruction);
    break;
  case THROW_DECL_END:
    check_throw_decl_end(instruction);
    break;
  case OTHER:
    check_other(instruction);
    break;
  default:
    assert(!"Unknown instruction type");
  }
}

void goto2ct::check_assert(goto_programt::instructiont instruction)
{
  assert(
    !is_nil_expr(instruction.guard) && "The ASSERT guard must contain a value");
  assert(is_nil_expr(instruction.code) && "The ASSERT code must be empty");
  check_guard(instruction.guard);
}

void goto2ct::check_assume(goto_programt::instructiont instruction)
{
  assert(
    !is_nil_expr(instruction.guard) && "The ASSUME guard must contain a value");
  assert(is_nil_expr(instruction.code) && "The ASSUME code must be empty");
  check_guard(instruction.guard);
}

void goto2ct::check_goto(goto_programt::instructiont instruction)
{
  assert(
    !is_nil_expr(instruction.guard) && "The GOTO guard must contain a value");
  check_guard(instruction.guard);
  // Fedor: I think it will be best if we harmonize both:
  // each GOTO instruction must contain a code expression
  // of type "code_goto2t" OR the "code_goto2t" code information
  // is transferred to the list of targets
  assert(is_nil_expr(instruction.code) || is_code_goto2t(instruction.code));
  assert(instruction.targets.size() == 1);
}

void goto2ct::check_function_call(goto_programt::instructiont instruction
                                  [[maybe_unused]])
{
  assert(is_code_function_call2t(instruction.code));
}

void goto2ct::check_return(goto_programt::instructiont instruction
                           [[maybe_unused]])
{
  assert(is_code_return2t(instruction.code));
}

void goto2ct::check_end_function(goto_programt::instructiont instruction
                                 [[maybe_unused]])
{
  assert(is_nil_expr(instruction.code));
  assert(is_true(instruction.guard));
}

void goto2ct::check_decl(goto_programt::instructiont instruction
                         [[maybe_unused]])
{
  assert(is_code_decl2t(instruction.code));
}

void goto2ct::check_dead(goto_programt::instructiont instruction
                         [[maybe_unused]])
{
  assert(is_code_dead2t(instruction.code));
}

void goto2ct::check_assign(goto_programt::instructiont instruction
                           [[maybe_unused]])
{
  assert(is_code_assign2t(instruction.code));
}

void goto2ct::check_location(goto_programt::instructiont instruction
                             [[maybe_unused]])
{
  // Fedor: maybe use it just for targets in the attempt
  // to establish a unique "target" type of instruction
}

void goto2ct::check_skip(goto_programt::instructiont instruction
                         [[maybe_unused]])
{
  // Fedor: at the moment SKIP's can have code of type "code_skip2t"
  // or something else
}

void goto2ct::check_throw(goto_programt::instructiont instruction
                          [[maybe_unused]])
{
  assert(is_code_cpp_throw2t(instruction.code));
}

void goto2ct::check_catch(goto_programt::instructiont instruction
                          [[maybe_unused]])
{
  assert(is_code_cpp_catch2t(instruction.code));
}

void goto2ct::check_atomic_begin(goto_programt::instructiont instruction
                                 [[maybe_unused]])
{
}

void goto2ct::check_atomic_end(goto_programt::instructiont instruction
                               [[maybe_unused]])
{
}

void goto2ct::check_throw_decl(goto_programt::instructiont instruction
                               [[maybe_unused]])
{
  assert(is_code_cpp_throw_decl2t(instruction.code));
}

void goto2ct::check_throw_decl_end(goto_programt::instructiont instruction
                                   [[maybe_unused]])
{
  assert(is_code_cpp_throw_decl_end2t(instruction.code));
}

void goto2ct::check_other(goto_programt::instructiont instruction
                          [[maybe_unused]])
{
}

void check_if_sideeffect_or_assign_expr(expr2tc expr [[maybe_unused]])
{
  assert(!is_code_assign2t(expr));
  assert(!is_sideeffect2t(expr));
}

// This method iterates recursively through subexpressions
// and checks whether any of them is a "sideeffect"
// or "assign" expression
void goto2ct::check_guard(expr2tc expr)
{
  check_if_sideeffect_or_assign_expr(expr);
  for (size_t i = 0; i < expr->get_num_sub_exprs(); i++)
    check_guard(*(expr->get_sub_expr(i)));
}
