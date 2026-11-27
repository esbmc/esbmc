#include <transition-system/ts_flatten.h>

#include <irep2/irep2_utils.h>
#include <util/irep/migrate.h>
#include <util/lang/c_types.h>
#include <util/message/message.h>
#include <util/symtab/symbol.h>

#include <map>

namespace
{
using targett = goto_programt::targett;

goto_programt::instructiont
instruction_at(goto_program_instruction_typet type, const targett &at)
{
  goto_programt::instructiont i(type);
  i.location = at->location;
  i.function = at->function;
  return i;
}
} // namespace

void flatten_loops(goto_functionst &goto_functions, contextt &context)
{
  auto main_fn = goto_functions.function_map.find("c:@F@main");
  if (
    main_fn == goto_functions.function_map.end() ||
    !main_fn->second.body_available)
    return;
  goto_programt &body = main_fn->second.body;

  std::unordered_map<const goto_programt::instructiont *, size_t> pos;
  std::vector<targett> gotos;
  size_t n = 0;
  for (auto it = body.instructions.begin(); it != body.instructions.end(); ++it)
  {
    pos[&*it] = n++;
    if (it->is_goto())
      gotos.push_back(it);
  }
  for (const targett &g : gotos)
    if (g->targets.size() != 1)
      return;
  auto at = [&](const targett &it) { return pos.at(&*it); };
  auto target = [](const targett &g) { return g->targets.front(); };

  // Loops as [head, back edge] positions.
  std::vector<std::pair<size_t, size_t>> loops;
  std::map<size_t, targett> cuts;
  for (const targett &g : gotos)
    if (at(target(g)) <= at(g))
    {
      loops.emplace_back(at(target(g)), at(g));
      cuts.emplace(at(target(g)), target(g));
    }
  for (const targett &g : gotos)
    for (const auto &[head, back] : loops)
      if (
        at(target(g)) > head && at(target(g)) <= back &&
        (at(g) < head || at(g) > back))
        cuts.emplace(at(target(g)), target(g));
  if (cuts.size() < 2)
    return;

  const targett end = std::prev(body.instructions.end());
  const targett first = cuts.begin()->second;
  // A prefix jump past the first cut point would enter the new loop past its
  // head, so its target is a cut point too.
  for (const targett &g : gotos)
    if (at(g) < at(first) && at(target(g)) > at(first) && target(g) != end)
      cuts.emplace(at(target(g)), target(g));

  unsigned bits = 1;
  while ((1ULL << bits) < cuts.size())
    bits++;
  const type2tc loc_type = get_uint_type(bits);
  symbolt sym;
  // goto_loopst skips a non-USR name with a '$', so the havoc would miss it;
  // no clang USR of a local lacks the file part.
  sym.id = "c:@F@main@ts_location";
  sym.name = "ts_location";
  sym.mode = "C";
  sym.set_type(migrate_type_back(loc_type));
  sym.lvalue = true;
  context.move_symbol_to_context(sym);
  const expr2tc loc = symbol2tc(loc_type, sym.id);
  goto_programt::instructiont decl_loc = instruction_at(DECL, first);
  decl_loc.code = code_decl2tc(loc_type, sym.id);
  body.instructions.insert(body.instructions.begin(), decl_loc);

  // The last dispatch is unconditional: a havocked location out of range
  // starts at the last cut point, so the dispatcher never falls through.
  targett dispatch = first;
  unsigned id = 0;
  for (const auto &[p, c] : cuts)
  {
    const expr2tc is_c =
      equality2tc(loc, constant_int2tc(loc_type, BigInt(id)));
    const targett jump =
      body.instructions.insert(first, instruction_at(GOTO, c));
    jump->make_goto(c, id + 1 == cuts.size() ? gen_true_expr() : is_c);
    if (id++ == 0)
      dispatch = jump;
  }
  body.instructions.insert(end, instruction_at(GOTO, end))->make_goto(end);
  const targett latch =
    body.instructions.insert(end, instruction_at(GOTO, end));
  latch->make_goto(dispatch);

  // Jumps from before the dispatcher enter through it, the rest through the
  // latch, whose back edge is the loop's only one; so the stubs that set the
  // location sit after every block, and a block falling into a cut point
  // jumps forward to its stub.
  auto stub = [&](targett before, unsigned id, targett via) {
    goto_programt::instructiont set = instruction_at(ASSIGN, before);
    set.code = code_assign2tc(loc, constant_int2tc(loc_type, BigInt(id)));
    const targett assign = body.instructions.insert(before, set);
    body.instructions.insert(before, instruction_at(GOTO, before))
      ->make_goto(via);
    return assign;
  };
  std::map<size_t, std::pair<targett, targett>> stubs;
  id = 0;
  for (const auto &[p, c] : cuts)
  {
    const targett from_block = stub(latch, id, latch);
    body.instructions.insert(c, instruction_at(GOTO, c))->make_goto(from_block);
    stubs.emplace(p, std::make_pair(stub(dispatch, id, dispatch), from_block));
    id++;
  }
  for (const targett &g : gotos)
  {
    auto s = stubs.find(at(target(g)));
    if (s != stubs.end())
      g->targets.front() =
        at(g) < at(first) ? s->second.first : s->second.second;
  }

  goto_functions.update();
  log_status("Flattened main into one loop over {} cut points", cuts.size());
}
