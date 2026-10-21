#include <goto-programs/goto_logic_functions.h>
#include <irep2/irep2_utils.h>
#include <util/irep/migrate.h>
#include <util/message/message.h>
#include <functional>
#include <map>
#include <set>

// An __ESBMC_logic function is a definition, never executed. Each call outside
// a definition is an application whose result symex keeps congruent across
// equal arguments and unchanged memory, plus one unfolding of the body that
// assumes result == body. Inside the unfolding, calls are applications only.
// The assumed equations have a model as long as every unfolded self-call
// decreases a non-negative measure (the last argument); that is asserted, so
// a definition can never make a proof vacuous.

const char *const kLogicAppPrefix = "__ESBMC_logic_app$";

namespace
{
bool is_logic(const namespacet &ns, const irep_idt &id)
{
  const symbolt *s = ns.lookup(id);
  return s != nullptr && s->get_type().get_bool("#logic_function");
}

const code_function_call2t *logic_call(
  const goto_programt::instructiont &i,
  const namespacet &ns)
{
  if (!i.is_function_call() || !is_code_function_call2t(i.code))
    return nullptr;
  const code_function_call2t &c = to_code_function_call2t(i.code);
  if (!is_symbol2t(c.function) || !is_logic(ns, to_symbol2t(c.function).thename))
    return nullptr;
  return &c;
}

[[noreturn]] void refuse(const irep_idt &f, const std::string &why)
{
  log_error("__ESBMC_logic function {}: {}", f, why);
  abort();
}

void collect_names(const expr2tc &e, std::set<irep_idt> &out)
{
  if (is_nil_expr(e))
    return;
  if (is_symbol2t(e))
    out.insert(to_symbol2t(e).thename);
  e->foreach_operand([&out](const expr2tc &op) { collect_names(op, out); });
}

bool has_deref(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (is_dereference2t(e))
    return true;
  bool found = false;
  e->foreach_operand([&found](const expr2tc &op) { found |= has_deref(op); });
  return found;
}

/// Reads must stay inside what the application's key covers: parameters,
/// locals, and memory reached directly through a pointer parameter.
void check_reads(
  const irep_idt &f,
  const expr2tc &e,
  const std::set<irep_idt> &params,
  const std::set<irep_idt> &locals)
{
  if (is_nil_expr(e))
    return;
  if (is_symbol2t(e))
  {
    const symbol2t &s = to_symbol2t(e);
    // Float operations carry the rounding mode; symex keys applications on it.
    if (
      !is_code_type(e->type) && !params.count(s.thename) &&
      !locals.count(s.thename) && s.thename != "c:@__ESBMC_rounding_mode")
      refuse(f, "reads " + id2string(s.thename) + ", which is not a parameter");
    return;
  }
  if (is_dereference2t(e))
  {
    const expr2tc &ptr = to_dereference2t(e).value;
    if (has_deref(ptr))
      refuse(f, "dereferences a pointer read from memory");
    std::set<irep_idt> names;
    collect_names(ptr, names);
    for (const irep_idt &n : names)
      if (!params.count(n))
        refuse(f, "dereferences a pointer that is not a parameter");
  }
  e->foreach_operand(
    [&](const expr2tc &op) { check_reads(f, op, params, locals); });
}

void check_body(
  const irep_idt &f,
  const goto_functiont &gf,
  const namespacet &ns,
  std::set<irep_idt> &callees)
{
  const code_type2t &ct = to_code_type(gf.type);
  if (is_empty_type(ct.ret_type))
    refuse(f, "returns void");
  std::set<irep_idt> params(ct.argument_names.begin(), ct.argument_names.end());
  std::set<irep_idt> locals;
  std::map<const goto_programt::instructiont *, unsigned> index;
  unsigned n = 0;
  forall_goto_program_instructions (it, gf.body)
  {
    index[&*it] = n++;
    if (it->is_decl())
      locals.insert(to_code_decl2t(it->code).value);
  }

  forall_goto_program_instructions (it, gf.body)
  {
    switch (it->type)
    {
    case ASSIGN:
    {
      const expr2tc &lhs = to_code_assign2t(it->code).target;
      if (!is_symbol2t(lhs) || !locals.count(to_symbol2t(lhs).thename))
        refuse(f, "assigns something other than a local");
      check_reads(f, to_code_assign2t(it->code).source, params, locals);
      break;
    }
    case FUNCTION_CALL:
    {
      const code_function_call2t *c = logic_call(*it, ns);
      if (c == nullptr)
        refuse(f, "calls a function that is not __ESBMC_logic");
      if (
        !is_nil_expr(c->ret) && (!is_symbol2t(c->ret) ||
                                 !locals.count(to_symbol2t(c->ret).thename)))
        refuse(f, "stores a call result outside a local");
      for (const expr2tc &a : c->operands)
        check_reads(f, a, params, locals);
      callees.insert(to_symbol2t(c->function).thename);
      break;
    }
    case GOTO:
      for (const auto &t : it->targets)
        if (index.at(&*t) <= index.at(&*it))
          refuse(f, "contains a loop; recurse instead");
      check_reads(f, it->guard, params, locals);
      break;
    case RETURN:
      check_reads(f, to_code_return2t(it->code).operand, params, locals);
      break;
    case ASSERT:
      check_reads(f, it->guard, params, locals);
      break;
    case DECL:
    case DEAD:
    case SKIP:
    case LOCATION:
    case END_FUNCTION:
      break;
    default:
      refuse(f, "has a statement a pure definition cannot");
    }
  }

  if (callees.count(f))
  {
    if (ct.arguments.empty() || !is_bv_type(ct.arguments.back()))
      refuse(f, "recurses without an integer last argument to decrease");
  }
}

/// Mutual recursion would need a measure across functions; only self
/// recursion is allowed.
void check_acyclic(const std::map<irep_idt, std::set<irep_idt>> &calls)
{
  for (const auto &[f, direct] : calls)
  {
    std::set<irep_idt> seen;
    std::vector<irep_idt> work;
    for (const irep_idt &g : direct)
      if (g != f)
        work.push_back(g);
    while (!work.empty())
    {
      const irep_idt g = work.back();
      work.pop_back();
      if (g == f)
        refuse(f, "is mutually recursive");
      if (!seen.insert(g).second)
        continue;
      auto it = calls.find(g);
      if (it != calls.end())
        for (const irep_idt &h : it->second)
          if (h != g)
            work.push_back(h);
    }
  }
}

/// Logic function -> the first one with a structurally identical definition.
/// Applications are named after it, so equal definitions under different names
/// (a version-isolated copy of one) are congruent: they are one function.
std::map<irep_idt, irep_idt> representative;

/// The definition with parameters and locals named by position and logic
/// callees by their own canonical form; equal strings mean equal functions.
std::string canonical(
  const irep_idt &f,
  const goto_functionst &goto_functions,
  const namespacet &ns,
  std::map<irep_idt, std::string> &memo)
{
  auto known = memo.find(f);
  if (known != memo.end())
    return known->second;
  memo[f] = "";
  const goto_functiont &gf = goto_functions.function_map.at(f);
  const code_type2t &ct = to_code_type(gf.type);
  std::map<irep_idt, std::string> names;
  for (size_t i = 0; i < ct.argument_names.size(); i++)
    names[ct.argument_names[i]] = "$p" + std::to_string(i);
  std::map<const goto_programt::instructiont *, unsigned> index;
  unsigned n = 0, locals = 0;
  forall_goto_program_instructions (it, gf.body)
  {
    index[&*it] = n++;
    if (it->is_decl())
      names[to_code_decl2t(it->code).value] = "$l" + std::to_string(locals++);
  }

  std::function<void(expr2tc &)> rename = [&](expr2tc &e) {
    if (is_nil_expr(e))
      return;
    if (is_symbol2t(e))
    {
      const irep_idt &id = to_symbol2t(e).thename;
      auto it = names.find(id);
      if (it != names.end())
        e = symbol2tc(e->type, it->second);
      else if (is_code_type(e->type) && is_logic(ns, id))
        // The code type names its parameters, so leave it out.
        e = symbol2tc(
          get_empty_type(),
          id == f ? std::string("$self")
                  : "$logic(" + canonical(id, goto_functions, ns, memo) + ")");
      return;
    }
    e = e->clone();
    e->Foreach_operand(rename);
  };

  std::string out = ct.ret_type->pretty(0);
  for (const type2tc &t : ct.arguments)
    out += t->pretty(0);
  forall_goto_program_instructions (it, gf.body)
  {
    expr2tc code = it->code, guard = it->guard;
    rename(code);
    rename(guard);
    out += "\n" + std::to_string(it->type);
    // A DECL/DEAD carries its local's name as a field, not an operand.
    if (is_nil_expr(code))
      ;
    else if (is_code_decl2t(code))
      out += names[to_code_decl2t(code).value] + code->type->pretty(0);
    else if (is_code_dead2t(code))
      out += names[to_code_dead2t(code).value];
    else
      out += code->pretty(0);
    if (!is_nil_expr(guard))
      out += guard->pretty(0);
    for (const auto &t : it->targets)
      out += " ->" + std::to_string(index.at(&*t));
  }
  return memo[f] = out;
}

expr2tc app_symbol(const code_function_call2t &c)
{
  const irep_idt &f = to_symbol2t(c.function).thename;
  auto rep = representative.find(f);
  return symbol2tc(
    c.function->type,
    kLogicAppPrefix +
      id2string(rep == representative.end() ? f : rep->second));
}

irep_idt unfold_name(const irep_idt &f)
{
  return id2string(f) + "$logic_unfold";
}

type2tc unfold_type(const code_type2t &ct, const irep_idt &name)
{
  std::vector<type2tc> args = ct.arguments;
  std::vector<irep_idt> names = ct.argument_names;
  args.push_back(ct.ret_type);
  names.push_back(id2string(name) + "::__ESBMC_logic_result");
  return code_type2tc(args, get_empty_type(), names, false);
}

/// `unfold_f(args..., result)`: assume \p c's result equals f's body.
goto_programt::instructiont unfold_call(
  const code_function_call2t &c,
  const goto_programt::instructiont &at,
  const namespacet &ns)
{
  const symbolt *usym =
    ns.lookup(unfold_name(to_symbol2t(c.function).thename));
  std::vector<expr2tc> args = c.operands;
  args.push_back(c.ret);
  goto_programt::instructiont unfold(FUNCTION_CALL);
  unfold.code = code_function_call2tc(
    expr2tc(), symbol2tc(migrate_type(usym->get_type()), usym->id), args);
  unfold.location = at.location;
  unfold.function = at.function;
  return unfold;
}

void declare_unfold(const irep_idt &f, const goto_functiont &gf, contextt &context)
{
  const irep_idt name = unfold_name(f);
  const code_type2t &ct = to_code_type(gf.type);
  symbolt fsym = *context.find_symbol(f);
  fsym.id = name;
  fsym.name = name;
  set_symbol_type(fsym, unfold_type(ct, name));
  context.add(fsym);
  symbolt rsym;
  rsym.id = id2string(name) + "::__ESBMC_logic_result";
  rsym.name = "__ESBMC_logic_result";
  set_symbol_type(rsym, ct.ret_type);
  rsym.lvalue = true;
  rsym.is_parameter = true;
  context.add(rsym);
  symbolt oksym;
  oksym.id = id2string(name) + "::__ESBMC_logic_ok";
  oksym.name = "__ESBMC_logic_ok";
  set_symbol_type(oksym, get_bool_type());
  oksym.lvalue = true;
  context.add(oksym);
}

/// The body of \p f with each return turned into `assume(result == value)`.
/// A self-call asserts the measure and stays an application; a call to
/// another logic function is unfolded too, which ends as calls are acyclic.
void build_unfold(
  const irep_idt &f,
  const goto_functiont &gf,
  goto_functionst &goto_functions,
  const namespacet &ns)
{
  const irep_idt name = unfold_name(f);
  const code_type2t &ct = to_code_type(gf.type);
  const expr2tc result =
    symbol2tc(ct.ret_type, id2string(name) + "::__ESBMC_logic_result");
  const expr2tc measure =
    ct.arguments.empty()
      ? expr2tc()
      : symbol2tc(ct.arguments.back(), ct.argument_names.back());

  // The definition is assumed only where every self-call decreased the
  // measure, so it stays consistent even if the assertion below is dropped or
  // turned into an assumption by a caller.
  const expr2tc ok =
    symbol2tc(get_bool_type(), id2string(name) + "::__ESBMC_logic_ok");

  goto_functiont &uf = goto_functions.function_map[name];
  uf.body.copy_from(gf.body);
  uf.type = unfold_type(ct, name);
  uf.body_available = true;
  {
    goto_programt::instructiont decl(DECL), init(ASSIGN);
    decl.code = code_decl2tc(get_bool_type(), to_symbol2t(ok).thename);
    init.code = code_assign2tc(ok, gen_true_expr());
    decl.location = init.location = uf.body.instructions.front().location;
    uf.body.insert_swap(uf.body.instructions.begin(), init);
    uf.body.insert_swap(uf.body.instructions.begin(), decl);
  }

  Forall_goto_program_instructions (it, uf.body)
  {
    // A definition is not executed code: the safety checks goto_check gave
    // the body would fire on unconstrained application values. A read out of
    // bounds only leaves the value free, which cannot make a proof vacuous.
    if (it->is_assert())
    {
      it->make_skip();
      continue;
    }
    if (it->type == RETURN)
    {
      expr2tc value = to_code_return2t(it->code).operand;
      if (value->type != ct.ret_type)
        value = typecast2tc(ct.ret_type, value);
      it->type = ASSUME;
      it->guard = implies2tc(ok, equality2tc(result, value));
      it->code = expr2tc();
      it->location.comment("logic function definition");
      continue;
    }
    const code_function_call2t *c = logic_call(*it, ns);
    if (c == nullptr)
      continue;
    code_function_call2t call = *c;
    call.function = app_symbol(*c);
    if (to_symbol2t(c->function).thename == f)
    {
      expr2tc arg = call.operands.back();
      if (arg->type != measure->type)
        arg = typecast2tc(measure->type, arg);
      const expr2tc decreases = and2tc(
        greaterthanequal2tc(arg, gen_zero(measure->type)),
        lessthan2tc(arg, measure));
      goto_programt::instructiont record(ASSIGN);
      record.code = code_assign2tc(ok, and2tc(ok, decreases));
      record.location = it->location;
      record.function = it->function;
      uf.body.insert_swap(it, record);
      ++it;
      goto_programt::instructiont check(ASSERT);
      check.guard = decreases;
      check.location = it->location;
      check.location.comment("logic function recursion measure");
      check.location.property("logic-measure");
      check.function = it->function;
      goto_programt::instructiont app = *it;
      app.code = code_function_call2tc(call);
      // Keep jumps into the call landing on the check.
      uf.body.insert_swap(it, check);
      ++it;
      *it = app;
    }
    else if (!is_nil_expr(c->ret))
    {
      goto_programt::instructiont unfold = unfold_call(*c, *it, ns);
      it->code = code_function_call2tc(call);
      it = uf.body.insert(std::next(it));
      *it = unfold;
    }
    else
      it->make_skip();
  }
  uf.update_instructions_function(name, true);
}

/// Each call in an ordinary body becomes an application followed by an
/// unfolding. A call whose result nothing reads (a clause dropped as inactive)
/// is removed rather than unfolded at a state it was never meant for.
void lower_calls(goto_programt &body, const namespacet &ns)
{
  std::map<irep_idt, unsigned> uses;
  forall_goto_program_instructions (it, body)
  {
    std::set<irep_idt> names;
    collect_names(it->code, names);
    collect_names(it->guard, names);
    for (const irep_idt &n : names)
      ++uses[n];
  }

  Forall_goto_program_instructions (it, body)
  {
    const code_function_call2t *c = logic_call(*it, ns);
    if (c == nullptr)
      continue;
    const irep_idt f = to_symbol2t(c->function).thename;
    if (!is_symbol2t(c->ret) || uses[to_symbol2t(c->ret).thename] <= 1)
    {
      it->make_skip();
      continue;
    }
    std::set<irep_idt> in_args;
    for (const expr2tc &a : c->operands)
      collect_names(a, in_args);
    if (in_args.count(to_symbol2t(c->ret).thename))
      refuse(f, "is called with its own result as an argument");

    goto_programt::instructiont unfold = unfold_call(*c, *it, ns);
    code_function_call2t call = *c;
    call.function = app_symbol(*c);
    it->code = code_function_call2tc(call);
    it = body.insert(std::next(it));
    *it = unfold;
  }
}
} // namespace

void goto_logic_functions(goto_functionst &goto_functions, contextt &context)
{
  const namespacet ns(context);
  std::map<irep_idt, std::set<irep_idt>> calls;
  for (const auto &[id, gf] : goto_functions.function_map)
    if (gf.body_available && is_logic(ns, id))
      check_body(id, gf, ns, calls[id]);
  if (calls.empty())
    return;
  check_acyclic(calls);

  representative.clear();
  std::map<irep_idt, std::string> memo;
  std::map<std::string, irep_idt> first;
  for (const auto &[id, callees] : calls)
    representative[id] =
      first.emplace(canonical(id, goto_functions, ns, memo), id).first->second;

  for (const auto &[id, callees] : calls)
    declare_unfold(id, goto_functions.function_map.at(id), context);
  for (const auto &[id, callees] : calls)
    build_unfold(id, goto_functions.function_map.at(id), goto_functions, ns);

  for (auto &[id, gf] : goto_functions.function_map)
    if (
      gf.body_available && !calls.count(id) &&
      !calls.count(id2string(id).substr(0, id2string(id).rfind('$'))))
      lower_calls(gf.body, ns);

  goto_functions.update();
}
