#include <goto-programs/lower_longjmp.h>
#include <goto-programs/goto_functions.h>
#include <irep2/irep2_utils.h>
#include <util/irep/migrate.h>
#include <util/symtab/context.h>
#include <util/config/options.h>

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
  return calls(insn, "c:@F@setjmp") || calls(insn, "c:@F@_setjmp");
}

bool program_calls_longjmp(const goto_functionst &goto_functions)
{
  for (const auto &[id, f] : goto_functions.function_map)
    for (const auto &insn : f.body.instructions)
      if (calls(insn, "c:@F@longjmp"))
        return true;
  return false;
}

class longjmp_loweringt
{
public:
  longjmp_loweringt(contextt &context) : context(context)
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
    epilogue->make_skip();

    auto dispatch =
      sites.empty() ? epilogue : build_dispatch(fn, body, sites, epilogue);

    for (auto call : others)
    {
      auto check = insert(body, std::next(call), call);
      check->make_goto(dispatch, pending);
      check->location.property("skipped");
    }
  }

private:
  contextt &context;
  expr2tc pending, value, token, count;
  unsigned slots = 0;

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
      expr2tc slot = new_slot(fn, site);
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
      match->make_goto(land, equality2tc(token, slot));
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

void lower_longjmp(
  goto_functionst &goto_functions,
  contextt &context,
  const optionst &options)
{
  if (
    options.get_bool_option("enable-unreachability-intrinsic") ||
    !program_calls_longjmp(goto_functions))
    return;
  longjmp_loweringt lowering(context);
  if (!lowering.find_globals())
    return;
  for (auto &[id, f] : goto_functions.function_map)
    if (f.body_available)
      lowering.lower(id, f.body);
  goto_functions.update();
}
