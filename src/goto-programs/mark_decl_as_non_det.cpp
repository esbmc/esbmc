#include <goto-programs/mark_decl_as_non_det.h>

bool mark_decl_as_non_det::runOnFunction(
  std::pair<const dstring, goto_functiont> &F)
{
  if(!F.second.body_available)
    return false; // Didn't changed anything

  for(auto it = F.second.body.instructions.begin();
      it != F.second.body.instructions.end();
      it++)
  {
    if(!it->is_decl())
      continue;

    auto decl = to_code_decl2t(it->code);
    symbolt *s = context.find_symbol(decl.value);

    // Global variables and function declaration shouldn`t reach here
    assert(!s->static_lifetime || !s->type.is_code());

    // Is the value initialized?
    if(s->value.is_nil())
    {
      // Initialize it with nondet then
      exprt nondet = exprt("sideeffect", s->type);
      nondet.statement("nondet");
      code_assignt assign(symbol_exprt(decl.value), nondet);
      assign.location() = it->location;

      auto next_inst = it;
      next_inst++;
      goto_programt::targett t =
        F.second.body.instructions.insert(next_inst, ASSIGN);
      migrate_expr(assign, t->code);
      t->location = assign.location();
    }
  }

  return true;
}