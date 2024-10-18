#include <goto-programs/assign_params_as_non_det.h>

bool assign_params_as_non_det::runOnFunction(
  std::pair<const dstring, goto_functiont> &F)
{
  if (context.find_symbol(F.first) == nullptr)
    return false; // Not exist

  exprt symbol = symbol_expr(*context.find_symbol(F.first));

  if (!symbol.type().is_code())
    return false; // Not expected
  code_typet t = to_code_type(symbol.type());

  if (symbol.name().as_string() != target_function)
    return false; // Not target function

  if (!F.second.body_available)
    return false; // Empty function

  /*
    Foreach parameter, create an assignment to nondet value and insert in the front
    E.g. func(int x, bool y) {...}
    =>
    func(int x, bool y)  { x = nondet_int(); y = nondet_bool(); ...}
  */
  goto_programt &goto_program = F.second.body;
  auto it = (goto_program).instructions.begin();

  for (const auto &arg : t.arguments())
  {
    // lhs
    const auto &_id = arg.get("#identifier");
    if (context.find_symbol(_id) == nullptr)
      return false; // Not expected
    exprt lhs = symbol_expr(*context.find_symbol(_id));

    // rhs
    exprt rhs = exprt("sideeffect", lhs.type());
    rhs.statement("nondet");
    rhs.location() = it->location;

    // assignment
    goto_programt tmp;
    goto_programt::targett assignment = tmp.add_instruction(ASSIGN);
    assignment->function = it->location.get_function();

    code_assignt code_assign(lhs, rhs);
    code_assign.location() = it->location;
    migrate_expr(code_assign, assignment->code);

    // insert
    goto_program.insert_swap(it++, *assignment);
    --it;
  }

  goto_program.update();
  return true;
}