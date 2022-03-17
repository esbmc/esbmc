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

    // find_symbol should always work here
    assert(s);
    // Global variables and function declaration shouldn`t reach here
    assert(!s->static_lifetime || !s->type.is_code());

    // Is the value initialized?
    if(s->value.is_nil())
    {
      // Initialize it with nondet then
      expr2tc new_value =
        code_assign2tc(symbol2tc(decl.type, decl.value), gen_nondet(decl.type));

      // Due to the value set analysis, we need to split declarations and assignments
      auto insert_pos = it;
      insert_pos++;
      auto t = F.second.body.instructions.insert(insert_pos, ASSIGN);
      t->location = it->location;
      t->code = new_value;
    }
  }

  return true;
}