#include <functional>
#include <goto-programs/goto_functions.h>

std::unordered_set<irep_idt, irep_id_hash>
reachable_functions(const goto_functionst &goto_functions)
{
  std::unordered_set<irep_idt, irep_id_hash> seen{goto_functions.main_id()};
  std::vector<irep_idt> work{goto_functions.main_id()};
  std::function<void(const expr2tc &)> visit = [&](const expr2tc &e) {
    if (is_nil_expr(e))
      return;
    if (
      is_symbol2t(e) && is_code_type(e->type) &&
      seen.insert(to_symbol2t(e).thename).second)
      work.push_back(to_symbol2t(e).thename);
    e->foreach_operand([&visit](const expr2tc &op) { visit(op); });
  };
  while (!work.empty())
  {
    auto it = goto_functions.function_map.find(work.back());
    work.pop_back();
    if (it == goto_functions.function_map.end() || !it->second.body_available)
      continue;
    for (const auto &instr : it->second.body.instructions)
    {
      visit(instr.code);
      visit(instr.guard);
    }
  }
  return seen;
}
