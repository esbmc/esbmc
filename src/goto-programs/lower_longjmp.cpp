#include <goto-programs/lower_longjmp.h>
#include <goto-programs/goto_functions.h>
#include <irep2/irep2_utils.h>
#include <util/irep/migrate.h>
#include <util/symtab/context.h>

namespace
{
bool calls(const goto_programt::instructiont &insn, const char *fn)
{
  if (!insn.is_function_call())
    return false;
  const expr2tc &f = to_code_function_call2t(insn.code).function;
  return is_symbol2t(f) && to_symbol2t(f).thename == fn;
}

bool is_setjmp(const goto_programt::instructiont &insn)
{
  return calls(insn, "c:@F@setjmp") || calls(insn, "c:@F@_setjmp") ||
         calls(insn, "c:@F@sigsetjmp") || calls(insn, "c:@F@__sigsetjmp");
}

bool mentions_longjmp(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (is_symbol2t(e))
  {
    const irep_idt &id = to_symbol2t(e).thename;
    return id == "c:@F@longjmp" || id == "c:@F@_longjmp" ||
           id == "c:@F@siglongjmp";
  }
  bool found = false;
  e->foreach_operand(
    [&found](const expr2tc &op) { found = found || mentions_longjmp(op); });
  return found;
}

// A call through a pointer counts too, so any reference to longjmp does.
bool program_uses_longjmp(const goto_functionst &goto_functions)
{
  // The library's longjmp wrappers are linked whenever <setjmp.h> is
  // included, so their references do not count.
  for (const auto &[id, f] : goto_functions.function_map)
    if (id != "c:@F@_longjmp" && id != "c:@F@siglongjmp")
      for (const auto &insn : f.body.instructions)
        if (mentions_longjmp(insn.code) || mentions_longjmp(insn.guard))
          return true;
  return false;
}

bool is_main(const irep_idt &id)
{
  return id == "c:@F@main" || has_prefix(id, "c:@F@main#");
}

bool mentions_main(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (is_symbol2t(e) && is_main(to_symbol2t(e).thename))
    return true;
  bool found = false;
  e->foreach_operand(
    [&](const expr2tc &op) { found = found || mentions_main(op); });
  return found;
}

/// Whether anything but __ESBMC_main can call main (C allows it, C++ does
/// not), so a frame of main may end while an outer one still has a setjmp.
bool main_reentrant(const goto_functionst &goto_functions)
{
  for (const auto &[id, f] : goto_functions.function_map)
    if (id != goto_functions.main_id())
      for (const auto &insn : f.body.instructions)
        if (mentions_main(insn.code) || mentions_main(insn.guard))
          return true;
  return false;
}

class longjmp_loweringt
{
public:
  longjmp_loweringt(contextt &context, bool check_main)
    : context(context), check_main(check_main)
  {
  }

  bool find_globals()
  {
    return global(pending, "c:@__ESBMC_longjmp_pending") &&
           global(value, "c:@__ESBMC_longjmp_value") &&
           global(token, "c:@__ESBMC_longjmp_token") &&
           global(count, "c:@__ESBMC_setjmp_count");
  }

  void lower(const irep_idt &fn, goto_programt &body)
  {
    std::vector<goto_programt::targett> sites, others;
    for (auto it = body.instructions.begin(); it != body.instructions.end();
         ++it)
      if (it->is_function_call())
        (!body.hide && is_setjmp(*it) ? sites : others).push_back(it);

    if (sites.empty() && others.empty())
      return;

    auto end = std::prev(body.instructions.end());
    auto epilogue = insert(body, end, end);
    if (check_main && is_main(fn))
      make_unmatched_check(*epilogue);
    else
      epilogue->make_skip();

    auto dispatch =
      sites.empty() ? epilogue : build_dispatch(fn, body, sites, epilogue);

    // Checked at each call: the thread ends inside the trampoline before a
    // jump to its epilogue would be merged there.
    const bool thread_entry = fn == "c:@F@pthread_trampoline";
    for (auto call : others)
    {
      auto check = insert(body, std::next(call), call);
      if (thread_entry)
        make_unmatched_check(*check);
      else
      {
        check->make_goto(dispatch, pending);
        check->location.property("skipped");
      }
    }
  }

private:
  contextt &context;
  const bool check_main;
  expr2tc pending, value, token, count;
  unsigned slots = 0;

  /// A jump still pending when a thread's outermost function ends found no
  /// active setjmp to resume at (C11 7.13.2.1p2).
  void make_unmatched_check(goto_programt::instructiont &insn) const
  {
    insn.make_assertion(not2tc(pending));
    insn.location.property("longjmp");
    insn.location.comment("longjmp to a jmp_buf whose setjmp is not active");
  }

  bool global(expr2tc &e, const char *id)
  {
    const symbolt *s = context.find_symbol(id);
    if (!s)
      return false;
    e = symbol2tc(migrate_type(s->get_type()), s->id);
    return true;
  }

  static goto_programt::targett insert(
    goto_programt &body,
    goto_programt::targett before,
    goto_programt::const_targett from)
  {
    auto t = body.insert(before);
    t->location = from->location;
    t->function = from->function;
    return t;
  }

  expr2tc new_slot(const irep_idt &fn, goto_programt::const_targett at)
  {
    symbolt s;
    s.id = id2string(fn) + "::$setjmp_token$" + std::to_string(slots++);
    s.name = s.id;
    s.location = at->location;
    s.set_type(migrate_type_back(count->type));
    s.lvalue = true;
    symbolt *added;
    context.move(s, added);
    return symbol2tc(count->type, added->id);
  }

  goto_programt::targett build_dispatch(
    const irep_idt &fn,
    goto_programt &body,
    const std::vector<goto_programt::targett> &sites,
    goto_programt::targett epilogue)
  {
    auto first = body.instructions.begin();
    auto check = insert(body, epilogue, epilogue);
    check->make_goto(epilogue, not2tc(pending));
    auto done = insert(body, epilogue, epilogue);
    done->make_goto(epilogue);

    for (auto site : sites)
    {
      // Zeroed on entry so a site this frame has not reached matches nothing.
      const expr2tc slot = new_slot(fn, site);
      auto decl = insert(body, first, first);
      decl->make_decl();
      decl->code = code_decl2tc(slot->type, to_symbol2t(slot).thename);
      auto zero = insert(body, first, first);
      zero->make_assignment();
      zero->code = code_assign2tc(slot, gen_zero(slot->type));

      auto resume = std::next(site);
      auto record = insert(body, resume, site);
      record->make_assignment();
      record->code = code_assign2tc(slot, count);

      auto match = insert(body, done, site);
      auto land = insert(body, epilogue, site);
      // A never-set static jmp_buf holds token 0, which must not match a slot
      // this frame zeroed.
      match->make_goto(
        land,
        and2tc(
          notequal2tc(slot, gen_zero(slot->type)), equality2tc(token, slot)));
      const expr2tc &ret = to_code_function_call2t(site->code).ret;
      if (!is_nil_expr(ret))
      {
        land->make_assignment();
        land->code = code_assign2tc(ret, typecast2tc(ret->type, value));
        land = insert(body, epilogue, site);
      }
      land->make_assignment();
      land->code = code_assign2tc(pending, gen_false_expr());
      insert(body, epilogue, site)->make_goto(resume);
    }
    return check;
  }
};
} // namespace

void lower_longjmp(goto_functionst &goto_functions, contextt &context)
{
  if (!program_uses_longjmp(goto_functions))
    return;
  longjmp_loweringt lowering(context, !main_reentrant(goto_functions));
  if (!lowering.find_globals())
    return;
  for (auto &[id, f] : goto_functions.function_map)
    if (f.body_available)
      lowering.lower(id, f.body);
  goto_functions.update();
}
