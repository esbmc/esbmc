#include <goto-programs/goto_logic_functions.h>
#include <irep2/irep2_utils.h>
#include <util/irep/migrate.h>
#include <util/message/message.h>
#include <algorithm>
#include <functional>
#include <map>
#include <regex>
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

const code_function_call2t *
logic_call(const goto_programt::instructiont &i, const namespacet &ns)
{
  if (!i.is_function_call() || !is_code_function_call2t(i.code))
    return nullptr;
  const code_function_call2t &c = to_code_function_call2t(i.code);
  if (
    !is_symbol2t(c.function) || !is_logic(ns, to_symbol2t(c.function).thename))
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

/// Whether \p e reads memory: a dereference, or a member, whose object symex
/// does not key an application on even when it is a parameter.
bool has_deref(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (is_dereference2t(e) || is_member2t(e))
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
    names.erase("c:@__ESBMC_rounding_mode");
    for (const irep_idt &n : names)
      if (!params.count(n))
        refuse(
          f,
          "dereferences " + id2string(n) +
            ", which is not a pointer parameter");
  }
  e->foreach_operand(
    [&](const expr2tc &op) { check_reads(f, op, params, locals); });
}

/// The logic callee of the call \p i in the definition of \p f, refusing
/// any other callee or a result stored outside a local.
irep_idt check_call(
  const irep_idt &f,
  const goto_programt::instructiont &i,
  const namespacet &ns,
  const std::set<irep_idt> &params,
  const std::set<irep_idt> &locals)
{
  const code_function_call2t *c = logic_call(i, ns);
  if (c == nullptr)
    refuse(f, "calls a function that is not __ESBMC_logic");
  if (
    !is_nil_expr(c->ret) &&
    (!is_symbol2t(c->ret) || !locals.count(to_symbol2t(c->ret).thename)))
    refuse(f, "stores a call result outside a local");
  for (const expr2tc &a : c->operands)
    check_reads(f, a, params, locals);
  return to_symbol2t(c->function).thename;
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
      callees.insert(check_call(f, *it, ns, params, locals));
      break;
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

  if (
    callees.count(f) &&
    (ct.arguments.empty() || !is_bv_type(ct.arguments.back())))
    refuse(f, "recurses without an integer last argument to decrease");
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
using representativest = std::map<irep_idt, irep_idt>;

/// Member names and types, following struct tags, so a struct and its
/// version-isolated copy (same members, another tag) spell the same.
std::string layout(const type2tc &t, const namespacet &ns, unsigned depth = 0)
{
  const type2tc f = ns.follow(t);
  if (depth > 8 || (!is_struct_type(f) && !is_union_type(f)))
    return f->pretty(0);
  const auto &members =
    is_struct_type(f) ? to_struct_type(f).members : to_union_type(f).members;
  const auto &names = is_struct_type(f) ? to_struct_type(f).member_names
                                        : to_union_type(f).member_names;
  std::string out = is_struct_type(f) ? "struct{" : "union{";
  for (size_t i = 0; i < members.size(); i++)
  {
    // An anonymous member's name embeds its source location.
    const std::string &n = id2string(names[i]);
    out += (n.find("(anonymous") != std::string::npos ? "#" + std::to_string(i)
                                                      : n) +
           ":" + layout(members[i], ns, depth + 1) + ";";
  }
  return out + "}";
}

/// \p text with every struct/union tag replaced by that type's layout.
std::string by_layout(const std::string &text, const namespacet &ns)
{
  static const std::regex tag(
    R"(symbol_name : (tag-(struct|union|class) [^\n]*))");
  std::string out;
  auto begin = std::sregex_iterator(text.begin(), text.end(), tag);
  size_t last = 0;
  for (auto m = begin; m != std::sregex_iterator(); ++m)
  {
    out += text.substr(last, m->position() - last);
    const symbolt *sym = ns.lookup((*m)[1].str());
    out += sym == nullptr
             ? m->str()
             : "layout : " + layout(migrate_type(sym->get_type()), ns);
    last = m->position() + m->length();
  }
  out += text.substr(last);
  // An expanded struct type carries its tag too; its members follow it.
  static const std::regex name(R"(\* typename : (struct|union|class) [^\n]*)");
  // An anonymous member's name embeds its source location; its own members
  // are injected into the enclosing scope, so they alone identify a field.
  static const std::regex anon(R"([\w:]*\(anonymous[^)]*\))");
  return std::regex_replace(
    std::regex_replace(out, name, ""), anon, "(anonymous)");
}

/// \p code as text, with a DECL/DEAD's local by its position name: it carries
/// the name as a field, not an operand.
std::string
code_text(const expr2tc &code, std::map<irep_idt, std::string> &names)
{
  if (is_nil_expr(code))
    return "";
  if (is_code_decl2t(code))
    return names[to_code_decl2t(code).value] + code->type->pretty(0);
  if (is_code_dead2t(code))
    return names[to_code_dead2t(code).value];
  return code->pretty(0);
}

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
    out += "\n" + std::to_string(it->type) + code_text(code, names);
    if (!is_nil_expr(guard))
      out += guard->pretty(0);
    for (const auto &t : it->targets)
      out += " ->" + std::to_string(index.at(&*t));
  }
  return memo[f] = by_layout(out, ns);
}

expr2tc
app_symbol(const code_function_call2t &c, const representativest &representative)
{
  const irep_idt &f = to_symbol2t(c.function).thename;
  auto rep = representative.find(f);
  return symbol2tc(
    c.function->type,
    kLogicAppPrefix + id2string(rep == representative.end() ? f : rep->second));
}

const char *const kUnfoldSuffix = "$logic_unfold";

irep_idt unfold_name(const irep_idt &f)
{
  return id2string(f) + kUnfoldSuffix;
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
  const symbolt *usym = ns.lookup(unfold_name(to_symbol2t(c.function).thename));
  std::vector<expr2tc> args = c.operands;
  args.push_back(c.ret);
  goto_programt::instructiont unfold(FUNCTION_CALL);
  unfold.code = code_function_call2tc(
    expr2tc(), symbol2tc(migrate_type(usym->get_type()), usym->id), args);
  unfold.location = at.location;
  unfold.function = at.function;
  return unfold;
}

void declare_unfold(
  const irep_idt &f,
  const goto_functiont &gf,
  contextt &context)
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
  const namespacet &ns,
  const representativest &representative)
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
      // A RETURN jumps to the end by itself; the ASSUME would fall through
      // into the next return's and make the path, and every claim on it,
      // vacuous.
      goto_programt::instructiont leave(GOTO);
      leave.guard = gen_true_expr();
      leave.targets.push_back(std::prev(uf.body.instructions.end()));
      leave.location = it->location;
      leave.function = it->function;
      it = uf.body.insert(std::next(it));
      *it = leave;
      continue;
    }
    const code_function_call2t *c = logic_call(*it, ns);
    if (c == nullptr)
      continue;
    code_function_call2t call = *c;
    call.function = app_symbol(*c, representative);
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

/// Whether nothing reads the result \p ret of a call: there is no target, or
/// it is a local named only by the call. An element, a member, a dereference
/// or a global may be read anywhere.
bool result_unread(
  const expr2tc &ret,
  const std::set<irep_idt> &locals,
  std::map<irep_idt, unsigned> &uses)
{
  if (is_nil_expr(ret))
    return true;
  if (!is_symbol2t(ret))
    return false;
  const irep_idt &n = to_symbol2t(ret).thename;
  return locals.count(n) && uses[n] <= 1;
}

bool shares_name(const expr2tc &a, const std::vector<expr2tc> &bs)
{
  std::set<irep_idt> in_a, in_bs;
  collect_names(a, in_a);
  for (const expr2tc &b : bs)
    collect_names(b, in_bs);
  return std::any_of(in_a.begin(), in_a.end(), [&in_bs](const irep_idt &n) {
    return in_bs.count(n) > 0;
  });
}

/// Each call in an ordinary body becomes an application followed by an
/// unfolding. A call whose result nothing reads (a clause dropped as inactive)
/// is removed rather than unfolded at a state it was never meant for.
void lower_calls(
  goto_programt &body,
  const namespacet &ns,
  const representativest &representative)
{
  std::map<irep_idt, unsigned> uses;
  std::set<irep_idt> locals;
  forall_goto_program_instructions (it, body)
  {
    if (it->is_decl())
      locals.insert(to_code_decl2t(it->code).value);
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
    if (result_unread(c->ret, locals, uses))
    {
      it->make_skip();
      continue;
    }
    if (shares_name(c->ret, c->operands))
      refuse(f, "is called with its own result as an argument");

    goto_programt::instructiont unfold = unfold_call(*c, *it, ns);
    code_function_call2t call = *c;
    call.function = app_symbol(*c, representative);
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

  representativest representative;
  std::map<irep_idt, std::string> memo;
  std::map<std::string, irep_idt> first;
  for (const auto &[id, callees] : calls)
    representative[id] =
      first.emplace(canonical(id, goto_functions, ns, memo), id).first->second;

  for (const auto &[id, callees] : calls)
    declare_unfold(id, goto_functions.function_map.at(id), context);
  for (const auto &[id, callees] : calls)
    build_unfold(
      id, goto_functions.function_map.at(id), goto_functions, ns, representative);

  for (auto &[id, gf] : goto_functions.function_map)
    if (
      gf.body_available && !calls.count(id) &&
      !calls.count(id2string(id).substr(0, id2string(id).rfind('$'))))
      lower_calls(gf.body, ns, representative);

  goto_functions.update();
}

bool is_logic_unfold(const irep_idt &function)
{
  const std::string &s = id2string(function);
  const std::string suffix = kUnfoldSuffix;
  return s.size() > suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
