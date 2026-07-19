#include <goto-programs/goto_convert_class.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/message.h>
#include <util/message/format.h>
#include <util/migrate.h>
#include <util/rename.h>
#include <util/std_expr.h>

/// Recursively flatten a (possibly nested) binary && expression into a flat
/// list of conjuncts.  Clang represents A && B && C as and(and(A,B),C).
static void collect_and_conjuncts(const exprt &expr, exprt::operandst &out)
{
  if (expr.is_and())
    for (const auto &op : expr.operands())
      collect_and_conjuncts(op, out);
  else
    out.push_back(expr);
}

/// Rebuild a right-associative nested binary && from a flat list of conjuncts
/// starting at index @p i.
static exprt rebuild_and_chain(const exprt::operandst &conjuncts, std::size_t i)
{
  assert(i < conjuncts.size());
  if (i + 1 == conjuncts.size())
    return conjuncts[i];
  exprt result = exprt("and", bool_type());
  result.copy_to_operands(conjuncts[i], rebuild_and_chain(conjuncts, i + 1));
  return result;
}

/// Recursively flatten a (possibly nested) binary || expression into a flat
/// list of disjuncts.  Clang represents A || B || C as or(or(A,B),C).
static void collect_or_disjuncts(const exprt &expr, exprt::operandst &out)
{
  if (expr.id() == "or")
    for (const auto &op : expr.operands())
      collect_or_disjuncts(op, out);
  else
    out.push_back(expr);
}

/// Rebuild a right-to-left nested binary || from a flat list of disjuncts
/// starting at index @p i.
static exprt rebuild_or_chain(const exprt::operandst &disjuncts, std::size_t i)
{
  assert(i < disjuncts.size());
  if (i + 1 == disjuncts.size())
    return disjuncts[i];
  exprt result = exprt("or", bool_type());
  result.copy_to_operands(disjuncts[i], rebuild_or_chain(disjuncts, i + 1));
  return result;
}

/// Extract the symbol bound by a quantifier from its first argument,
/// which has the shape (possibly typecast) address_of(symbol).
static const exprt *quantifier_bound_var(const exprt &arg)
{
  const exprt *e = &arg;
  while (e->id() == "typecast" && e->operands().size() == 1)
    e = &e->op0();
  if (
    e->id() == "address_of" && e->operands().size() == 1 &&
    e->op0().is_symbol())
    return &e->op0();
  return nullptr;
}

static irep_idt quantifier_bound_var_id(const exprt &arg)
{
  const exprt *sym = quantifier_bound_var(arg);
  return sym ? sym->identifier() : irep_idt();
}

static bool mentions_symbol(const exprt &e, const std::set<irep_idt> &ids)
{
  if (e.is_symbol())
    return ids.count(e.identifier()) != 0;
  forall_operands (it, e)
    if (mentions_symbol(*it, ids))
      return true;
  return false;
}

/// A side effect other than a nested function call (e.g. ++ on a parameter)
/// cannot be replicated by argument substitution.
static bool has_non_call_sideeffect(const exprt &e)
{
  if (e.id() == "sideeffect" && e.statement() != "function_call")
    return true;
  forall_operands (it, e)
    if (has_non_call_sideeffect(*it))
      return true;
  return false;
}

static std::size_t count_symbol_occurrences(const exprt &e, const irep_idt &id)
{
  std::size_t n = e.is_symbol() && e.identifier() == id ? 1 : 0;
  forall_operands (it, e)
    n += count_symbol_occurrences(*it, id);
  return n;
}

/// Post-order (evaluation-order) walk: true when the occurrence of @p id
/// precedes every other side effect in @p e.  Substituting a side-effecting
/// value at a later position would reorder calls across a sequence point.
static bool
symbol_before_sideeffects(const exprt &e, const irep_idt &id, bool &seen_id)
{
  forall_operands (it, e)
    if (!symbol_before_sideeffects(*it, id, seen_id))
      return false;
  if (e.is_symbol() && e.identifier() == id)
    seen_id = true;
  else if (e.id() == "sideeffect" && !seen_id)
    return false;
  return true;
}

static bool substitutes_in_eval_order(const exprt &e, const irep_idt &id)
{
  bool seen_id = false;
  return symbol_before_sideeffects(e, id, seen_id);
}

/// Substituting an argument under address_of would take the address of an
/// rvalue or conflate distinct parameter objects (e.g. `&a != &b`).
static bool
param_under_address_of(const exprt &e, const std::set<irep_idt> &ids)
{
  if (e.id() == "address_of")
    return mentions_symbol(e, ids);
  forall_operands (it, e)
    if (param_under_address_of(*it, ids))
      return true;
  return false;
}

static void
replace_symbols_in_expr(exprt &e, const std::map<irep_idt, exprt> &map)
{
  if (e.is_symbol())
  {
    auto it = map.find(e.identifier());
    if (it != map.end())
    {
      e = it->second;
      return;
    }
  }
  Forall_operands (it, e)
    replace_symbols_in_expr(*it, map);
}

/// Returns the __ESBMC_forall/__ESBMC_exists symbol when @p e is a call to a
/// quantifier intrinsic, nullptr otherwise.
static const symbolt *
quantifier_intrinsic_call(const exprt &e, const namespacet &ns)
{
  if (
    e.id() != "sideeffect" || e.statement() != "function_call" ||
    e.operands().size() < 2 || !e.op0().is_symbol())
    return nullptr;
  const symbolt *fsym = ns.lookup(e.op0().identifier());
  if (
    fsym && (fsym->name == "__ESBMC_forall" || fsym->name == "__ESBMC_exists"))
    return fsym;
  return nullptr;
}

static constexpr unsigned max_quantifier_inline_depth = 16;

static bool
flatten_body_statements(const codet &code, std::vector<const codet *> &out)
{
  const irep_idt &statement = code.get_statement();
  if (statement == "block" || statement == "decl-block")
  {
    forall_operands (it, code)
      if (!it->is_code() || !flatten_body_statements(to_code(*it), out))
        return false;
    return true;
  }
  if (statement == "skip")
    return true;
  out.push_back(&code);
  return true;
}

/// Build the inlined form of @p call to @p fsym: the callee's body must be
/// local declarations with initializers followed by a single
/// `return <expr>;`.  The declarations and then the actual arguments are
/// substituted into the return expression.  Returns false when the callee
/// does not have this shape.
bool goto_convertt::try_inline_pure_call(
  const symbolt &fsym,
  const exprt &call,
  exprt &out)
{
  const exprt &value = fsym.get_value();
  if (!value.is_code() || fsym.get_type().id() != "code")
    return false;

  std::vector<const codet *> stmts;
  if (!flatten_body_statements(to_code(value), stmts) || stmts.empty())
    return false;

  const codet &ret = *stmts.back();
  if (
    ret.get_statement() != "return" || ret.operands().size() != 1 ||
    ret.op0().is_nil())
    return false;

  const code_typet &ftype = to_code_type(fsym.get_type());
  if (ftype.has_ellipsis())
    return false;
  const code_typet::argumentst &params = ftype.arguments();
  const exprt::operandst &args = call.op1().operands();
  if (params.size() != args.size())
    return false;

  exprt result = ret.op0();

  // Fold local declarations into the return expression, last first, so that
  // each occurrence count sees the fully substituted downstream uses.
  for (std::size_t i = stmts.size() - 1; i-- > 0;)
  {
    const codet &decl = *stmts[i];
    if (
      decl.get_statement() != "decl" || decl.operands().size() != 2 ||
      !decl.op0().is_symbol())
      return false;
    const irep_idt &id = decl.op0().identifier();
    // A static local is initialized once, not per call.
    const symbolt *local = ns.lookup(id);
    if (!local || local->static_lifetime)
      return false;
    exprt init = decl.op1();
    if (has_non_call_sideeffect(init) || mentions_symbol(init, {id}))
      return false;
    // Same rule as for arguments below: a side-effecting initializer must
    // end up evaluated exactly once, before every side effect sequenced
    // after it.
    if (
      has_sideeffect(init) && (count_symbol_occurrences(result, id) != 1 ||
                               !substitutes_in_eval_order(result, id)))
      return false;
    if (param_under_address_of(result, {id}))
      return false;
    if (init.type() != decl.op0().type())
      init.make_typecast(decl.op0().type());
    std::map<irep_idt, exprt> local_map;
    local_map.emplace(id, std::move(init));
    replace_symbols_in_expr(result, local_map);
  }

  if (has_non_call_sideeffect(result))
    return false;

  std::map<irep_idt, exprt> param_map;
  std::set<irep_idt> param_ids;
  for (std::size_t i = 0; i < params.size(); i++)
  {
    const irep_idt &id = params[i].get_identifier();
    if (id.empty())
      return false;
    exprt arg = args[i];
    if (has_non_call_sideeffect(arg))
      return false;
    // A side-effecting argument must be evaluated exactly once; substitution
    // would drop it (unused parameter) or duplicate it (parameter used more
    // than once).  It is also sequenced before the callee body, so its
    // occurrence must precede every side effect already in the body.
    if (
      has_sideeffect(arg) && (count_symbol_occurrences(result, id) != 1 ||
                              !substitutes_in_eval_order(result, id)))
      return false;
    if (arg.type() != params[i].type())
      arg.make_typecast(params[i].type());
    param_ids.insert(id);
    param_map.emplace(id, std::move(arg));
  }

  if (param_under_address_of(result, param_ids))
    return false;

  replace_symbols_in_expr(result, param_map);
  if (result.type() != call.type())
    result.make_typecast(call.type());
  result.location() = call.find_location();
  out.swap(result);
  return true;
}

namespace
{
/// State threaded through the structured-body summarizer.  @ref env maps each
/// in-scope local (parameters seeded from the call's actual arguments) to its
/// current symbolic value; @ref pc is the path condition reaching the current
/// statement; @ref returns records (guard, value) for every `return' in program
/// order, with the earliest matching guard winning; @ref locals holds every
/// callee-local identifier, so a read of a not-yet-assigned local can be
/// detected; @ref budget bounds loop unrolling.
struct summary_statet
{
  std::map<irep_idt, exprt> env;
  exprt pc;
  std::vector<std::pair<exprt, exprt>> returns;
  std::set<irep_idt> locals;
  unsigned budget;
};

bool summary_has_sideeffect(const exprt &e)
{
  if (e.id() == "sideeffect")
    return true;
  forall_operands (it, e)
    if (summary_has_sideeffect(*it))
      return true;
  return false;
}

/// Fold @p e to a constant boolean: 1 = true, 0 = false, -1 = not a constant.
/// -1 covers both "genuinely symbolic" and "constant, but the simplifier could
/// not fold it"; either way the caller rejects the callee, so a stronger
/// simplifier only widens the set of accepted shapes, it never changes a result.
int summary_fold_bool(const exprt &e)
{
  expr2tc e2;
  migrate_expr(e, e2);
  simplify(e2);
  BigInt v;
  if (to_integer(e2, v))
    return -1;
  return v.is_zero() ? 0 : 1;
}

/// Keeping the path condition literally `true' (rather than and(true, x)) lets
/// the totality check below fold it without leaning on the simplifier.
exprt summary_and(const exprt &a, const exprt &b)
{
  return a.is_true() ? b : gen_and(a, b);
}

void summary_coerce(exprt &e, const typet &t)
{
  if (e.type() != t)
    e.make_typecast(t);
}

std::size_t summary_node_count(const exprt &e)
{
  std::size_t n = 1;
  forall_operands (it, e)
    n += summary_node_count(*it);
  return n;
}

/// Total size of the summary built so far.  A branch merge embeds the
/// pre-branch value in both arms, so an unrolled loop containing an if/else
/// doubles the tracked values per iteration: bounding the iteration count alone
/// still admits a 2^n-sized expression (a trip count of 18 already costs GBs).
/// Bounding size instead makes an over-large body degrade to a clean rejection.
std::size_t summary_size(const summary_statet &st)
{
  std::size_t n = 0;
  for (const auto &kv : st.env)
    n += summary_node_count(kv.second);
  for (const auto &r : st.returns)
    n += summary_node_count(r.first) + summary_node_count(r.second);
  return n;
}

static constexpr std::size_t max_summary_nodes = 20000;
} // namespace

/// Substitute tracked locals in @p e with their env values.  Fails when @p e
/// has a side effect, takes the address of a tracked local, or — after
/// substitution — still mentions a callee-local (a read before assignment the
/// summarizer cannot model).
static bool summary_eval(const exprt &e, const summary_statet &st, exprt &out)
{
  if (summary_has_sideeffect(e) || param_under_address_of(e, st.locals))
    return false;
  exprt r = e;
  replace_symbols_in_expr(r, st.env);
  if (mentions_symbol(r, st.locals))
    return false;
  out.swap(r);
  return true;
}

/// Apply a discarded expression statement (or a for-loop iterator) to @p st.
/// A plain `=' assignment or a bare `++`/`--' on a tracked local scalar updates
/// its value; a side-effect-free expression has no effect; anything else fails.
/// The clang frontend lowers an assignment used as a statement (including an
/// `if'/`for' controlled statement) to a side_effect_exprt("assign") rather
/// than a code_assignt, which is why this is the only assignment form handled.
static bool summary_apply_effect(const exprt &e, summary_statet &st)
{
  if (!summary_has_sideeffect(e))
    return true;
  // A `nondet' side effect carries no operands, so check before reaching op0().
  if (e.id() != "sideeffect" || e.operands().empty() || !e.op0().is_symbol())
    return false;
  const irep_idt &id = e.op0().identifier();
  if (!st.locals.count(id))
    return false;
  const typet &t = e.op0().type();
  const irep_idt &s = e.statement();

  if (s == "assign" && e.operands().size() == 2)
  {
    exprt val;
    if (!summary_eval(e.op1(), st, val))
      return false;
    summary_coerce(val, t);
    st.env[id] = val;
    return true;
  }

  const bool inc = (s == "preincrement" || s == "postincrement");
  const bool dec = (s == "predecrement" || s == "postdecrement");
  auto it = st.env.find(id);
  // Whitelist the types `cur +/- from_integer(1, t)' is meaningful for, so an
  // unhandled type degrades to a rejection.  from_integer() returns nil for
  // anything but an integer bitvector or bool, and a nil operand here left an
  // ill-formed `cur + nil' in env that crashed once the summary reached the
  // solver (a `double' local incremented in a summarized loop).
  if (
    (!inc && !dec) || e.operands().size() != 1 || it == st.env.end() ||
    (t.id() != "signedbv" && t.id() != "unsignedbv"))
    return false;
  exprt cur = it->second;
  summary_coerce(cur, t);
  exprt val(inc ? "+" : "-", t);
  val.copy_to_operands(cur, from_integer(1, t));
  it->second = val;
  return true;
}

/// Symbolically execute one structured statement of the callee, updating
/// @p st.  Returns false for any construct the summarizer cannot soundly turn
/// into a pure expression.
static bool
summarize_code(const codet &code, summary_statet &st, const namespacet &ns)
{
  const irep_idt &s = code.get_statement();

  if (s == "skip")
    return true;

  if (s == "block" || s == "decl-block")
  {
    forall_operands (it, code)
      if (!it->is_code() || !summarize_code(to_code(*it), st, ns))
        return false;
    return true;
  }

  if (s == "decl")
  {
    if (code.operands().empty() || !code.op0().is_symbol())
      return false;
    const irep_idt &id = code.op0().identifier();
    const symbolt *ls = ns.lookup(id);
    if (!ls || ls->static_lifetime)
      return false;
    st.locals.insert(id);
    if (code.operands().size() == 2 && code.op1().is_not_nil())
    {
      exprt val;
      if (!summary_eval(code.op1(), st, val))
        return false;
      summary_coerce(val, code.op0().type());
      st.env[id] = val;
    }
    else
      st.env.erase(id);
    return true;
  }

  // A source-level assignment reaches us as a side_effect_exprt("assign")
  // wrapped in a code_expressiont, never as a bare code_assignt, so only the
  // "expression" form below is handled; anything else falls through to the
  // conservative rejection at the end.
  if (s == "expression")
    return code.operands().size() == 1 && summary_apply_effect(code.op0(), st);

  if (s == "return")
  {
    if (code.operands().size() != 1)
      return false;
    const code_returnt &r = to_code_return(code);
    exprt val;
    if (!r.has_return_value() || !summary_eval(r.return_value(), st, val))
      return false;
    st.returns.emplace_back(st.pc, val);
    return true;
  }

  if (s == "ifthenelse")
  {
    if (code.operands().size() < 2)
      return false;
    const code_ifthenelset &i = to_code_ifthenelse(code);
    // The frontend emits an ifthenelse with two operands when there is no
    // else branch; op2() is only present with three (cf. convert_ifthenelse).
    const bool has_else =
      code.operands().size() == 3 && code.op2().is_not_nil();
    exprt cv;
    if (!summary_eval(i.cond(), st, cv))
      return false;
    const int f = summary_fold_bool(cv);
    if (f == 1)
      return summarize_code(i.then_case(), st, ns);
    if (f == 0)
      return !has_else || summarize_code(i.else_case(), st, ns);

    const std::map<irep_idt, exprt> base = st.env;
    const std::set<irep_idt> outer = st.locals;
    const exprt saved_pc = st.pc;

    st.pc = summary_and(saved_pc, cv);
    if (!summarize_code(i.then_case(), st, ns))
      return false;
    const std::map<irep_idt, exprt> then_env = st.env;

    st.env = base;
    st.pc = summary_and(saved_pc, gen_not(cv));
    if (has_else && !summarize_code(i.else_case(), st, ns))
      return false;
    const std::map<irep_idt, exprt> else_env = st.env;

    st.pc = saved_pc;
    st.env = base;
    st.locals = outer;
    // Merge every outer-scope local written on either path; a variable written
    // in only one branch takes its pre-branch value on the other.  Values only
    // defined on one path (declared without initializer, assigned in one branch)
    // are left unset so a later unconditional read bails.  Locals declared
    // inside a branch drop out of scope with the restore above.
    std::set<irep_idt> keys;
    for (const auto &kv : then_env)
      if (outer.count(kv.first))
        keys.insert(kv.first);
    for (const auto &kv : else_env)
      if (outer.count(kv.first))
        keys.insert(kv.first);
    for (const irep_idt &k : keys)
    {
      auto ti = then_env.find(k), ei = else_env.find(k), bi = base.find(k);
      const exprt *tv = ti != then_env.end() ? &ti->second
                        : bi != base.end()   ? &bi->second
                                             : nullptr;
      const exprt *ev = ei != else_env.end() ? &ei->second
                        : bi != base.end()   ? &bi->second
                                             : nullptr;
      if (!tv || !ev)
        continue;
      if (*tv == *ev)
        st.env[k] = *tv;
      else
      {
        exprt e = *ev;
        summary_coerce(e, tv->type());
        st.env[k] = if_exprt(cv, *tv, e);
      }
    }
    return true;
  }

  if (s == "for" || s == "while" || s == "dowhile")
  {
    const exprt *cond, *init = nullptr, *iter = nullptr;
    const codet *body;
    if (s == "for")
    {
      if (code.operands().size() != 4)
        return false;
      const code_fort &l = to_code_for(code);
      init = &l.init();
      cond = &l.cond();
      iter = &l.iter();
      body = &l.body();
    }
    else if (s == "while")
    {
      if (code.operands().size() != 2)
        return false;
      const code_whilet &l = to_code_while(code);
      cond = &l.cond();
      body = &l.body();
    }
    else
    {
      if (code.operands().size() != 2)
        return false;
      const code_dowhilet &l = to_code_dowhile(code);
      cond = &l.cond();
      body = &l.body();
    }

    if (
      init && init->is_not_nil() &&
      (!init->is_code() || !summarize_code(to_code(*init), st, ns)))
      return false;

    // Unroll while the loop condition folds to a constant true, threading the
    // loop counter through env each iteration.  A condition that never folds
    // (data-dependent trip count), an exhausted budget, or a summary that grew
    // too large leaves the loop as a call the summarizer cannot turn into a
    // pure expression.
    for (bool first = true;; first = false)
    {
      if (summary_size(st) > max_summary_nodes)
        return false;
      if (!(s == "dowhile" && first))
      {
        if (cond->is_nil())
          return false;
        exprt cv;
        if (!summary_eval(*cond, st, cv))
          return false;
        const int f = summary_fold_bool(cv);
        if (f == -1)
          return false;
        if (f == 0)
          break;
      }
      if (st.budget == 0)
        return false;
      st.budget--;
      if (!summarize_code(*body, st, ns))
        return false;
      // The frontend wraps the loop iterator in a code_expressiont; route it
      // through summarize_code, which unwraps the statement to apply its effect.
      if (iter && iter->is_not_nil())
      {
        if (iter->is_code())
        {
          if (!summarize_code(to_code(*iter), st, ns))
            return false;
        }
        else if (!summary_apply_effect(*iter, st))
          return false;
      }
    }
    return true;
  }

  return false;
}

bool goto_convertt::summarize_pure_call(
  const symbolt &fsym,
  const exprt &call,
  exprt &out)
{
  const exprt &value = fsym.get_value();
  if (!value.is_code() || fsym.get_type().id() != "code")
    return false;

  const code_typet &ftype = to_code_type(fsym.get_type());
  if (ftype.has_ellipsis())
    return false;
  const code_typet::argumentst &params = ftype.arguments();
  const exprt::operandst &args = call.op1().operands();
  if (params.size() != args.size())
    return false;

  summary_statet st;
  st.pc = true_exprt();
  st.budget = max_quantifier_inline_depth * max_quantifier_inline_depth;

  for (std::size_t i = 0; i < params.size(); i++)
  {
    const irep_idt &id = params[i].get_identifier();
    if (id.empty() || summary_has_sideeffect(args[i]))
      return false;
    exprt arg = args[i];
    summary_coerce(arg, params[i].type());
    st.locals.insert(id);
    st.env[id] = arg;
  }

  if (!summarize_code(to_code(value), st, ns))
    return false;

  // A trailing unconditional return guarantees the summary is total: the last
  // recorded value is the fall-through, earlier returns wrap it as nested
  // if-then-else so the earliest matching guard wins.
  if (st.returns.empty() || summary_fold_bool(st.returns.back().first) != 1)
    return false;

  exprt result = st.returns.back().second;
  for (std::size_t i = st.returns.size() - 1; i-- > 0;)
  {
    exprt v = st.returns[i].second;
    summary_coerce(v, result.type());
    result = if_exprt(st.returns[i].first, v, result);
  }

  summary_coerce(result, call.type());
  result.location() = call.find_location();
  out.swap(result);
  return true;
}

/// Inline calls to pure single-return functions occurring under a quantifier
/// binder, so that the bound variable stays visible in the body for
/// replace_name_in_body() in smt_solver.cpp.  Hoisting such a call to a temp
/// would freeze the bound variable at its pre-quantifier value, detaching the
/// body from the binder (GitHub discussion #6100).  Nested quantifier
/// intrinsic calls are recursed into but left intact; calls that cannot be
/// inlined are left in place.
void goto_convertt::inline_calls_in_quantifier_body(exprt &expr, unsigned depth)
{
  if (quantifier_intrinsic_call(expr, ns))
  {
    exprt::operandst &args = expr.op1().operands();
    if (args.size() == 2)
      inline_calls_in_quantifier_body(args[1], depth);
    return;
  }

  if (
    expr.id() == "sideeffect" && expr.statement() == "function_call" &&
    expr.operands().size() >= 2 && expr.op0().is_symbol())
  {
    const symbolt *fsym = ns.lookup(expr.op0().identifier());
    exprt inlined;
    if (
      !fsym || depth == 0 ||
      (!try_inline_pure_call(*fsym, expr, inlined) &&
       !summarize_pure_call(*fsym, expr, inlined)))
      return;
    expr.swap(inlined);
    inline_calls_in_quantifier_body(expr, depth - 1);
    return;
  }

  Forall_operands (it, expr)
    inline_calls_in_quantifier_body(*it, depth);
}

/// Find a side effect that mentions one of @p bound_vars.  Such a side
/// effect cannot be hoisted soundly: the temp would freeze the bound
/// variable at its pre-quantifier value.  Quantifier intrinsic calls are
/// skipped, with their own bound variable added while scanning their body.
const exprt *goto_convertt::find_sideeffect_on_bound_var(
  const exprt &expr,
  const std::set<irep_idt> &bound_vars)
{
  if (quantifier_intrinsic_call(expr, ns))
  {
    const exprt::operandst &args = expr.op1().operands();
    if (args.size() != 2)
      return nullptr;
    std::set<irep_idt> inner_vars = bound_vars;
    const irep_idt inner_id = quantifier_bound_var_id(args[0]);
    if (!inner_id.empty())
      inner_vars.insert(inner_id);
    return find_sideeffect_on_bound_var(args[1], inner_vars);
  }

  if (expr.id() == "sideeffect" && mentions_symbol(expr, bound_vars))
    return &expr;

  forall_operands (it, expr)
    if (const exprt *found = find_sideeffect_on_bound_var(*it, bound_vars))
      return found;
  return nullptr;
}

/// In an assertion, a __ESBMC_forall in positive polarity is equivalent to
/// its body over a fresh unconstrained variable: assert(forall x. B) fails
/// exactly when some x violates B.  When the body keeps a side effect on the
/// bound variable that inlining cannot remove (e.g. a call to a libc model
/// whose body is only linked at the GOTO level), rewrite the binder away so
/// the call is hoisted as an ordinary call on the fresh variable.  Bodies
/// the binder path can model are left alone, as are __ESBMC_exists and
/// quantifiers in negative positions (!, ==>, ?:, __ESBMC_assume).
void goto_convertt::skolemize_asserted_foralls(exprt &expr, goto_programt &dest)
{
  if (expr.id() == "typecast" && expr.operands().size() == 1)
  {
    skolemize_asserted_foralls(expr.op0(), dest);
    return;
  }

  if (expr.is_and() || expr.id() == "or")
  {
    Forall_operands (it, expr)
      skolemize_asserted_foralls(*it, dest);
    return;
  }

  const symbolt *fsym = quantifier_intrinsic_call(expr, ns);
  if (!fsym || fsym->name != "__ESBMC_forall")
    return;
  exprt::operandst &args = expr.op1().operands();
  if (args.size() != 2 || !has_sideeffect(args[1]))
    return;
  const exprt *bvar = quantifier_bound_var(args[0]);
  if (!bvar)
    return;

  inline_calls_in_quantifier_body(args[1], max_quantifier_inline_depth);
  if (!find_sideeffect_on_bound_var(args[1], {bvar->identifier()}))
    return;

  symbolt &skolem = new_tmp_symbol(bvar->type());
  skolem.location = expr.find_location();
  code_declt decl(symbol_expr(skolem));
  decl.location() = skolem.location;
  convert_decl(decl, dest);

  exprt body;
  body.swap(args[1]);
  std::map<irep_idt, exprt> skolem_map;
  skolem_map.emplace(bvar->identifier(), symbol_expr(skolem));
  replace_symbols_in_expr(body, skolem_map);
  if (body.type() != expr.type())
    body.make_typecast(expr.type());
  expr.swap(body);
  skolemize_asserted_foralls(expr, dest);
}

void goto_convertt::convert_quantifier_calls(exprt &expr)
{
  if (const symbolt *fsym = quantifier_intrinsic_call(expr, ns))
  {
    exprt::operandst &args = expr.op1().operands();
    if (args.size() == 2)
    {
      // Bottom-up: convert any nested quantifier calls first, then summarize
      // the remaining calls so the bound variable stays free in the body.
      convert_quantifier_calls(args[1]);
      inline_calls_in_quantifier_body(args[1], max_quantifier_inline_depth);
      if (!has_sideeffect(args[1]))
      {
        exprt quant(
          fsym->name == "__ESBMC_forall" ? "forall" : "exists", typet("bool"));
        quant.copy_to_operands(args[0], args[1]);
        quant.location() = expr.find_location();
        expr.swap(quant);
      }
    }
    return;
  }

  Forall_operands (it, expr)
    convert_quantifier_calls(*it);
}

void goto_convertt::remove_sideeffects_for_quantifier_body(
  exprt &body,
  const std::set<irep_idt> &bound_vars,
  goto_programt &dest)
{
  if (!has_sideeffect(body))
    return;

  // Look through any implicit typecasts that Clang may insert (e.g. an
  // explicit (int) cast on the body followed by an implicit _Bool conversion
  // for the function parameter produces two typecast layers).
  exprt *expr = &body;
  while (expr->id() == "typecast" && expr->operands().size() == 1)
    expr = &expr->op0();

  // Recurse into || and && without converting them to short-circuit ITE chains.
  // This preserves the logical structure so that replace_name_in_body() in
  // smt_solver.cpp can substitute the bound variable throughout the full
  // body.
  if (expr->is_or() || expr->is_and())
  {
    for (auto &op : expr->operands())
      remove_sideeffects_for_quantifier_body(op, bound_vars, dest);
    return;
  }

  // If the leaf is a nested quantifier call, convert it to a quantifier
  // expression inline instead of creating a temp variable.  This keeps the
  // full expression tree intact so that goto_check's guard propagation from
  // || / && correctly constrains array accesses within the nested quantifier
  // body.  Without this, the temp assignment is a separate instruction
  // without the enclosing quantifier's guard context, causing spurious
  // array-bounds violations (GitHub #3995).
  if (const symbolt *fsym = quantifier_intrinsic_call(*expr, ns))
  {
    exprt::operandst &args = expr->op1().operands();
    if (args.size() == 2)
    {
      if (has_sideeffect(args[1]))
      {
        std::set<irep_idt> inner_vars = bound_vars;
        const irep_idt inner_id = quantifier_bound_var_id(args[0]);
        if (!inner_id.empty())
          inner_vars.insert(inner_id);
        remove_sideeffects_for_quantifier_body(args[1], inner_vars, dest);
      }

      bool is_forall = (fsym->name == "__ESBMC_forall");
      exprt quant(is_forall ? "forall" : "exists", typet("bool"));
      quant.copy_to_operands(args[0]);
      quant.copy_to_operands(args[1]);
      quant.location() = expr->find_location();
      body = quant;
      return;
    }
  }

  // A remaining side effect on a bound variable cannot be hoisted: fail
  // loudly rather than produce a vacuous quantifier (GitHub discussion
  // #6100).
  if (const exprt *se = find_sideeffect_on_bound_var(body, bound_vars))
  {
    const bool is_call = se->statement() == "function_call" &&
                         se->operands().size() >= 1 && se->op0().is_symbol();
    log_error(
      "{}: cannot model {} on a quantified variable inside "
      "__ESBMC_forall/__ESBMC_exists; the quantifier body must be a "
      "side-effect-free expression, possibly calling functions built from "
      "local declarations, assignments, if/else, and loops with a statically "
      "constant trip count, and whose side-effecting arguments are each used "
      "exactly once",
      se->find_location().as_string(),
      is_call ? "call to `" + se->op0().identifier().as_string() + "'"
              : "side effect `" + se->statement().as_string() + "'");
    abort();
  }

  // Leaf that is not a boolean connector or nested quantifier: use the
  // standard handler.
  remove_sideeffects(body, dest, /*result_is_used=*/true);
}

void goto_convertt::make_temp_symbol(exprt &expr, goto_programt &dest)
{
  const locationt location = expr.find_location();

  symbolt &new_symbol = new_tmp_symbol(expr.type());

  // declare this symbol first
  code_declt decl(symbol_expr(new_symbol));
  decl.location() = location;
  convert_decl(decl, dest);

  code_assignt assignment;
  assignment.lhs() = symbol_expr(new_symbol);
  assignment.rhs() = expr;
  assignment.location() = location;

  convert(assignment, dest);

  expr = symbol_expr(new_symbol);
}

bool goto_convertt::has_sideeffect(const exprt &expr)
{
  forall_operands (it, expr)
    if (has_sideeffect(*it))
      return true;

  if (expr.id() == "sideeffect")
    return true;

  return false;
}

bool goto_convertt::has_sideeffect(const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return false;

  // A legacy "sideeffect" exprt migrates to sideeffect2t (function_call, malloc,
  // ++/--, …) OR sideeffect_assign2t (assignment / compound-assignment used as
  // an expression); the legacy has_sideeffect treats both as side effects.
  if (is_sideeffect2t(expr) || is_sideeffect_assign2t(expr))
    return true;

  bool found = false;
  expr->foreach_operand([this, &found](const expr2tc &op) {
    if (!found && has_sideeffect(op))
      found = true;
  });
  return found;
}

void goto_convertt::remove_sideeffects(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  // Always enter for ternary (if_exprt) so that --validate-violation-witness
  // can lower sideeffect-free ternaries to IF/GOTO with the ? column recorded
  // on expr.location(), enabling column-accurate branching waypoint matching.
  if (!has_sideeffect(expr) && expr.id() != "if")
    return;

  // Turn quantifier intrinsic calls into forall/exists expressions before the
  // &&/|| and generic side-effect lowering below, so a nested quantifier's body
  // is never hoisted into a temp (which would freeze the bound variable).  A
  // quantifier call is itself a side effect, so the guard above never skips a
  // convertible expression.
  convert_quantifier_calls(expr);

  // Snapshot for the discarded-temporary_object arm below: entries pushed
  // while lowering this expression describe its full-expression temporaries.
  std::size_t stack_size = targets.destructor_stack.size();

  if (expr.is_and() || expr.is_or())
  {
    if (!expr.is_boolean())
    {
      log_error(
        "{} must be Boolean, but got {}", expr.id_string(), expr.pretty());
      abort();
    }

    exprt tmp;

    if (expr.is_and())
      tmp = true_exprt();
    else
      // ID_or
      tmp = false_exprt();

    // Make sure we do not lose the location in tmp
    tmp.location() = expr.location();

    exprt::operandst &ops = expr.operands();

    // start with last one
    for (exprt::operandst::reverse_iterator it = ops.rbegin(); it != ops.rend();
         ++it)
    {
      exprt &op = *it;

      // This is a hack for now. We need to solve this properly by
      // correctly tracking all locations through all GOTO transformations
      op.location() = expr.location();

      if (!op.is_boolean())
      {
        log_error("{} takes boolean operands only", expr.id().as_string());
        abort();
      }

      if (expr.is_and())
      {
        // We need to record the location of the newly generated expression
        exprt false_expr = false_exprt();
        false_expr.location() = op.location();
        if_exprt if_e(op, tmp, false_expr);
        if_e.location() = op.location();
        tmp.swap(if_e);
      }
      else // ID_or
      {
        // We need to record the location of the newly generated expression
        exprt true_expr = true_exprt();
        true_expr.location() = op.location();
        if_exprt if_e(op, true_expr, tmp);
        if_e.location() = op.location();
        tmp.swap(if_e);
      }
    }

    expr.swap(tmp);
    expr.location() = tmp.location();

    remove_sideeffects(expr, dest, result_is_used);
    return;
  }

  if (expr.id() == "if")
  {
    // first clean condition sideeffects
    remove_sideeffects(expr.op0(), dest);

    // If neither branch has a sideeffect and we are not validating a violation
    // witness, the ternary can stay as an if_exprt — no lowering needed.
    // Under --validate-violation-witness we always lower so that the resulting
    // IF instruction carries the ? column from expr.location(), which symex_goto
    // uses for column-accurate branching waypoint matching.
    if (
      !has_sideeffect(to_if_expr(expr).true_case()) &&
      !has_sideeffect(to_if_expr(expr).false_case()) &&
      !options.get_bool_option("validate-violation-witness"))
      return;

    if_exprt if_expr = to_if_expr(expr);

    if (!if_expr.cond().is_boolean())
      throw "first argument of `if' must be boolean, but got ";

    const locationt location = expr.location();

    goto_programt tmp_true;
    remove_sideeffects(if_expr.true_case(), tmp_true, result_is_used);

    goto_programt tmp_false;
    remove_sideeffects(if_expr.false_case(), tmp_false, result_is_used);

    if (result_is_used)
    {
      symbolt &new_symbol = new_tmp_symbol(expr.type());

      code_declt decl(symbol_expr(new_symbol));
      decl.location() = location;
      convert_decl(decl, dest);

      code_assignt assignment_true;
      assignment_true.lhs() = symbol_expr(new_symbol);
      assignment_true.rhs() = if_expr.true_case();
      assignment_true.location() = location;
      convert(assignment_true, tmp_true);

      code_assignt assignment_false;
      assignment_false.lhs() = symbol_expr(new_symbol);
      assignment_false.rhs() = if_expr.false_case();
      assignment_false.location() = location;
      convert(assignment_false, tmp_false);

      expr = symbol_expr(new_symbol);
    }
    else
    {
      if (if_expr.true_case().is_not_nil())
      {
        code_expressiont code_expression(if_expr.true_case());
        convert(code_expression, tmp_true);
      }

      if (if_expr.false_case().is_not_nil())
      {
        code_expressiont code_expression(if_expr.false_case());
        convert(code_expression, tmp_false);
      }

      expr = nil_exprt();
    }

    generate_ifthenelse(if_expr.cond(), tmp_true, tmp_false, location, dest);
    return;
  }

  if (expr.id() == "comma")
  {
    if (result_is_used)
    {
      exprt result;

      Forall_operands (it, expr)
      {
        bool last = (it == --expr.operands().end());

        // special treatment for last one
        if (last)
        {
          result.swap(*it);
          remove_sideeffects(result, dest, true);
        }
        else
        {
          remove_sideeffects(*it, dest, false);

          // remember these for later checks
          if (it->is_not_nil())
            convert(code_expressiont(*it), dest);
        }
      }

      expr.swap(result);
    }
    else // result not used
    {
      Forall_operands (it, expr)
      {
        remove_sideeffects(*it, dest, false);

        // remember as expression statement for later checks
        if (it->is_not_nil())
          convert(code_expressiont(*it), dest);
      }

      expr = nil_exprt();
    }

    return;
  }

  if (expr.id() == "typecast")
  {
    if (expr.operands().size() != 1)
      throw "typecast takes one argument";

    // preserve 'result_is_used'
    remove_sideeffects(expr.op0(), dest, result_is_used);

    if (expr.op0().is_nil())
      expr.make_nil();

    return;
  }

  if (expr.id() == "sideeffect")
  {
    // some of the side-effects need special treatment!
    const irep_idt statement = expr.statement();
    if (statement == "gcc_conditional_expression")
    {
      remove_gcc_conditional_expression(expr, dest);
      return;
    }

    if (statement == "statement_expression")
    {
      remove_statement_expression(expr, dest, result_is_used);
      return;
    }

    if (statement == "assign")
    {
      // we do a special treatment for x=f(...)
      assert(expr.operands().size() == 2);

      if (
        expr.op1().id() == "sideeffect" &&
        to_side_effect_expr(expr.op1()).get_statement() == "function_call")
      {
        remove_sideeffects(expr.op0(), dest);
        exprt lhs = expr.op0();

        // turn into code
        code_assignt assignment;
        assignment.lhs() = lhs;
        assignment.rhs() = expr.op1();
        assignment.location() = expr.location();
        convert_assign(assignment, dest);

        if (result_is_used)
          expr.swap(lhs);
        else
          expr.make_nil();
        return;
      }
    }

    // Special handling for __ESBMC_loop_invariant(A && f(x) && B && ...):
    //
    // The normal Forall_operands path below would call remove_sideeffects()
    // on the entire && argument, which converts it to a short-circuit
    // if-then-else GOTO pattern and loses the guard condition A from the
    // stored LOOP_INVARIANT expression.
    //
    // Instead: flatten the top-level && chain, call remove_sideeffects()
    // on each conjunct independently (producing simple DECL+CALL pairs for
    // function-call conjuncts), and rebuild a plain and(...) from the
    // cleaned-up conjuncts.  The result is stored correctly in
    // LOOP_INVARIANT and re-evaluated after HAVOC.  All other call sites
    // are completely unaffected.
    if (
      statement == "function_call" && expr.operands().size() >= 2 &&
      expr.op0().is_symbol())
    {
      const symbolt *fsym = ns.lookup(expr.op0().identifier());

      // In non-contract mode, drop all contract annotation calls entirely so
      // they have zero effect on the GOTO program (no FUNCTION_CALL step, no
      // side-effect processing of the argument).  The declarations remain in
      // the unconditional intrinsics section so annotated files still compile.
      if (
        fsym && options.get_option("enforce-contract").empty() &&
        options.get_option("replace-call-with-contract").empty() &&
        !options.get_bool_option("enforce-all-contracts") &&
        !options.get_bool_option("replace-all-contracts"))
      {
        const std::string &fname = id2string(fsym->name);
        if (
          fname == "__ESBMC_requires" || fname == "__ESBMC_ensures" ||
          fname == "__ESBMC_assigns_impl")
        {
          expr.make_nil();
          return;
        }
      }
      if (fsym && fsym->name == "__ESBMC_loop_invariant")
      {
        exprt::operandst &args = expr.op1().operands();
        exprt *and_expr = nullptr;
        if (args.size() == 1 && has_sideeffect(args.front()))
        {
          // Locate the && expression, looking through any implicit typecast
          // that Clang inserts when converting the && result to _Bool.
          if (args.front().is_and())
            and_expr = &args.front();
          else if (
            args.front().id() == "typecast" &&
            args.front().operands().size() == 1 && args.front().op0().is_and())
            and_expr = &args.front().op0();

          if (and_expr)
          {
            exprt::operandst conjuncts;
            collect_and_conjuncts(*and_expr, conjuncts);
            for (auto &c : conjuncts)
              remove_sideeffects(c, dest);
            // Replace the argument with the rebuilt and-chain; drop any
            // surrounding typecast since and(...) is already boolean.
            args.front() = rebuild_and_chain(conjuncts, 0);
          }
        }
        // Only bypass Forall_operands when we actually rewrote the && chain;
        // otherwise (e.g. single foo(x) or foo(x)==0) fall through to normal path.
        if (and_expr)
        {
          remove_function_call(expr, dest, result_is_used);
          return;
        }
      }

      // Special handling for __ESBMC_ensures and __ESBMC_requires with
      // && or || chains containing side effects (e.g. __ESBMC_old()).
      if (
        fsym &&
        (fsym->name == "__ESBMC_ensures" || fsym->name == "__ESBMC_requires"))
      {
        exprt::operandst &args = expr.op1().operands();
        bool rewrote = false;
        if (args.size() == 1 && has_sideeffect(args.front()))
        {
          exprt *inner = &args.front();
          if (
            inner->id() == "typecast" && inner->operands().size() == 1 &&
            (inner->op0().is_and() || inner->op0().id() == "or"))
            inner = &inner->op0();

          if (inner->is_and())
          {
            exprt::operandst parts;
            collect_and_conjuncts(*inner, parts);
            for (auto &p : parts)
              remove_sideeffects(p, dest);
            args.front() = rebuild_and_chain(parts, 0);
            rewrote = true;
          }
          else if (inner->id() == "or")
          {
            exprt::operandst parts;
            collect_or_disjuncts(*inner, parts);
            for (auto &p : parts)
              remove_sideeffects(p, dest);
            args.front() = rebuild_or_chain(parts, 0);
            rewrote = true;
          }
        }
        if (rewrote)
        {
          remove_function_call(expr, dest, result_is_used);
          return;
        }
      }

      // A __ESBMC_forall the binder path cannot model (a residual side
      // effect on the bound variable) is sound to rewrite over a fresh
      // unconstrained variable when it sits in positive polarity of an
      // assertion; see skolemize_asserted_foralls.
      if (
        fsym && (fsym->name == "__ESBMC_assert" || fsym->name == "assert") &&
        !expr.op1().operands().empty())
        skolemize_asserted_foralls(expr.op1().operands().front(), dest);

      // Special handling for __ESBMC_forall/__ESBMC_exists(ptr, body):
      // the body must be processed under the binder so the bound variable
      // remains visible for substitution in smt_solver.cpp.
      if (
        fsym &&
        (fsym->name == "__ESBMC_forall" || fsym->name == "__ESBMC_exists"))
      {
        exprt::operandst &args = expr.op1().operands();
        if (args.size() == 2 && has_sideeffect(args[1]))
        {
          inline_calls_in_quantifier_body(args[1], max_quantifier_inline_depth);
          std::set<irep_idt> bound_vars;
          const irep_idt id = quantifier_bound_var_id(args[0]);
          if (!id.empty())
            bound_vars.insert(id);
          remove_sideeffects_for_quantifier_body(args[1], bound_vars, dest);
          remove_function_call(expr, dest, result_is_used);
          return;
        }
      }
    }
  }

  // TODO: evaluation order
  Forall_operands (it, expr)
    remove_sideeffects(*it, dest);

  if (expr.id() == "sideeffect")
  {
    const irep_idt &statement = expr.statement();

    if (statement == "function_call")
      remove_function_call(expr, dest, result_is_used);
    else if (
      statement == "assign" || statement == "assign+" ||
      statement == "assign-" || statement == "assign*" ||
      statement == "assign_div" || statement == "assign_bitor" ||
      statement == "assign_bitxor" || statement == "assign_bitand" ||
      statement == "assign_lshr" || statement == "assign_ashr" ||
      statement == "assign_shl" || statement == "assign_mod")
      remove_assignment(expr, dest, result_is_used);
    else if (statement == "postincrement" || statement == "postdecrement")
      remove_post(expr, dest, result_is_used);
    else if (statement == "preincrement" || statement == "predecrement")
      remove_pre(expr, dest, result_is_used);
    else if (statement == "cpp_new" || statement == "cpp_new[]")
      remove_cpp_new(expr, dest, result_is_used);
    else if (statement == "cpp_delete" || statement == "cpp_delete[]")
      remove_cpp_delete(expr, dest);
    else if (statement == "temporary_object")
    {
      const locationt location = expr.find_location();
      remove_temporary_object(expr, dest, result_is_used);

      // A discarded temporary dies at the end of its full expression
      // (C++ [class.temporary]/4, github #6076), not at block exit: emit
      // the scope-exit entries pushed while lowering this expression
      // (destructors before DEADs, innermost temporaries first) right here.
      // A result that is used keeps block-level scope, as does anything
      // without a pending destructor call (plain DEADs of C-style temps).
      if (!result_is_used)
      {
        bool have_destructor = false;
        for (std::size_t i = stack_size; i < targets.destructor_stack.size();
             i++)
          if (targets.destructor_stack[i].get_statement() == "function_call")
          {
            have_destructor = true;
            break;
          }

        if (have_destructor)
        {
          while (targets.destructor_stack.size() > stack_size)
          {
            codet d_code = targets.destructor_stack.back();
            targets.destructor_stack.pop_back();
            d_code.location() = location;
            convert(d_code, dest);
          }
          expr.make_nil();
        }
      }
    }
    else if (statement == "nondet")
    {
      // these are fine
    }
    else if (statement == "skip")
    {
      expr.make_nil();
    }
    else if (statement == "cpp-throw")
    {
      codet c("cpp-throw");
      c.operands() = expr.operands();
      c.location() = expr.location();
      c.set("exception_list", expr.find("exception_list"));
      convert_throw(c, dest);
      // the result can't be used, these are void
      expr.make_nil();
    }
    else
    {
      log_error("cannot remove side effect ({})", statement);
      abort();
    }
  }
}

// IREP2 dual-API seam (W1, esbmc/esbmc#4715): an expr2tc overload of
// remove_sideeffects. The side-effect-free common case is handled natively (no
// migration round-trip — mirroring the first line of the legacy overload). A
// side-effect-bearing or ternary expression is delegated to the legacy exprt
// path for now (migrate out, run the unchanged legacy removal, migrate back),
// which is behaviour-identical by construction; the per-kind hoisting is ported
// to native expr2tc in a later phase behind this same signature.
void goto_convertt::remove_sideeffects(
  expr2tc &expr,
  goto_programt &dest,
  bool result_is_used)
{
  // Native fast path: nil, or a side-effect-free non-ternary expression, needs
  // no hoisting. The legacy overload always enters for a ternary (if2t) so that
  // --validate-violation-witness can lower it; preserve that by delegating.
  if (is_nil_expr(expr) || (!has_sideeffect(expr) && !is_if2t(expr)))
    return;

  exprt legacy = migrate_expr_back(expr);
  remove_sideeffects(legacy, dest, result_is_used);
  migrate_expr(legacy, expr);
}

void goto_convertt::remove_assignment(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  const irep_idt &statement = expr.statement();

  if (statement == "assign")
  {
    exprt tmp = expr;
    tmp.id("code");
    convert_assign(to_code_assign(to_code(tmp)), dest);
  }
  else if (
    statement == "assign+" || statement == "assign-" ||
    statement == "assign*" || statement == "assign_div" ||
    statement == "assign_mod" || statement == "assign_shl" ||
    statement == "assign_ashr" || statement == "assign_lshr" ||
    statement == "assign_bitand" || statement == "assign_bitxor" ||
    statement == "assign_bitor")
  {
    if (expr.operands().size() != 2)
    {
      log_error(
        "{} takes two arguments\nLocation: {}", statement, expr.location());
      abort();
    }

    exprt rhs;

    if (statement == "assign+")
    {
      if (expr_has_floatbv(expr))
      {
        rhs.id("ieee_add");
      }
      else
      {
        rhs.id("+");
      }
    }
    else if (statement == "assign-")
    {
      if (expr_has_floatbv(expr))
      {
        rhs.id("ieee_sub");
      }
      else
      {
        rhs.id("-");
      }
    }
    else if (statement == "assign*")
    {
      if (expr_has_floatbv(expr))
      {
        rhs.id("ieee_mul");
      }
      else
      {
        rhs.id("*");
      }
    }
    else if (statement == "assign_div")
    {
      if (expr_has_floatbv(expr))
      {
        rhs.id("ieee_div");
      }
      else
      {
        rhs.id("/");
      }
    }
    else if (statement == "assign_mod")
    {
      rhs.id("mod");
    }
    else if (statement == "assign_shl")
    {
      rhs.id("shl");
    }
    else if (statement == "assign_ashr")
    {
      rhs.id("ashr");
    }
    else if (statement == "assign_lshr")
    {
      rhs.id("lshr");
    }
    else if (statement == "assign_bitand")
    {
      rhs.id("bitand");
    }
    else if (statement == "assign_bitxor")
    {
      rhs.id("bitxor");
    }
    else if (statement == "assign_bitor")
    {
      rhs.id("bitor");
    }
    else
    {
      std::ostringstream str;
      str << statement << " not yet supported\n";
      str << "Location: " << expr.location();
      log_error("{}", str.str());
      abort();
    }

    rhs.copy_to_operands(expr.op0(), expr.op1());
    rhs.type() = expr.op0().type();

    if (rhs.op0().type().is_bool())
    {
      rhs.op0().make_typecast(int_type());
      rhs.op1().make_typecast(int_type());
      rhs.type() = int_type();
      rhs.make_typecast(typet("bool"));
    }

    exprt lhs(expr.op0());

    // C11 compound assignments on _Atomic are RMW: the read and write must
    // happen in a single indivisible atomic section (no context-switch window
    // between the load and the store, unlike plain "x = x op y").
    if (is_atomic_symbol(lhs, ns))
      convert_assign_rmw_atomic(lhs, rhs, expr.location(), dest);
    else
    {
      code_assignt assignment(lhs, rhs);
      assignment.location() = expr.location();
      convert(assignment, dest);
    }
  }

  // revert assignment in the expression to its LHS
  if (result_is_used)
  {
    exprt lhs;
    lhs.swap(expr.op0());
    expr.swap(lhs);
  }
  else
    expr.make_nil();
}

void goto_convertt::remove_pre(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  const irep_idt statement = expr.statement();

  assert(statement == "preincrement" || statement == "predecrement");

  if (expr.operands().size() != 1)
  {
    std::ostringstream str;
    str << statement << " takes one argument\n";
    str << "Location: " << expr.location();
    log_error("{}", str.str());
    abort();
  }

  exprt rhs;
  rhs.location() = expr.location();

  if (statement == "preincrement")
  {
    if (expr.type().is_floatbv())
      rhs.id("ieee_add");
    else
      rhs.id("+");
  }
  else
  {
    if (expr.type().is_floatbv())
      rhs.id("ieee_sub");
    else
      rhs.id("-");
  }

  const typet &op_type = ns.follow(expr.op0().type());

  if (op_type.is_bool())
  {
    rhs.copy_to_operands(expr.op0(), gen_one(int_type()));
    rhs.op0().make_typecast(int_type());
    rhs.type() = int_type();
    rhs.make_typecast(bool_type());
  }
  else if (op_type.id() == "c_enum" || op_type.id() == "incomplete_c_enum")
  {
    rhs.copy_to_operands(expr.op0(), gen_one(int_type()));
    rhs.op0().make_typecast(int_type());
    rhs.type() = int_type();
    rhs.make_typecast(op_type);
  }
  else
  {
    typet constant_type;

    if (op_type.is_pointer())
      constant_type = index_type();
    else if (is_number(op_type))
      constant_type = op_type;
    else
    {
      log_error(
        "no constant one of type {}\nLocation: {}",
        op_type.to_string(),
        expr.location());
      abort();
    }

    exprt constant = gen_one(constant_type);

    rhs.copy_to_operands(expr.op0());
    rhs.move_to_operands(constant);
    rhs.type() = expr.op0().type();
  }

  // C11 pre-increment/decrement on _Atomic is a sequentially-consistent RMW:
  // the load and store must be indivisible.
  if (is_atomic_symbol(expr.op0(), ns))
    convert_assign_rmw_atomic(expr.op0(), rhs, expr.location(), dest);
  else
  {
    code_assignt assignment(expr.op0(), rhs);
    assignment.location() = expr.location();
    convert(assignment, dest);
  }

  if (result_is_used)
  {
    // revert to argument of pre-inc/pre-dec
    exprt tmp = expr.op0();
    expr.swap(tmp);
  }
  else
    expr.make_nil();
}

void goto_convertt::remove_post(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  const irep_idt statement = expr.statement();

  assert(statement == "postincrement" || statement == "postdecrement");

  if (expr.operands().size() != 1)
  {
    std::ostringstream str;
    str << statement << " takes one argument";
    str << "Location: " << expr.location();
    log_error("{}", str.str());
    abort();
  }

  exprt rhs;
  rhs.location() = expr.location();

  if (statement == "postincrement")
  {
    if (expr.type().is_floatbv())
      rhs.id("ieee_add");
    else
      rhs.id("+");
  }
  else
  {
    if (expr.type().is_floatbv())
      rhs.id("ieee_sub");
    else
      rhs.id("-");
  }

  const typet &op_type = ns.follow(expr.op0().type());

  if (op_type.is_bool())
  {
    rhs.copy_to_operands(expr.op0(), gen_one(int_type()));
    rhs.op0().make_typecast(int_type());
    rhs.type() = int_type();
    rhs.make_typecast(bool_type());
  }
  else if (op_type.id() == "c_enum" || op_type.id() == "incomplete_c_enum")
  {
    rhs.copy_to_operands(expr.op0(), gen_one(int_type()));
    rhs.op0().make_typecast(int_type());
    rhs.type() = int_type();
    rhs.make_typecast(op_type);
  }
  else
  {
    typet constant_type;

    if (op_type.is_pointer())
      constant_type = index_type();
    else if (is_number(op_type))
      constant_type = op_type;
    else
    {
      log_error(
        "no constant one of type {}\nLocation: {}",
        op_type.to_string(),
        expr.location());
      abort();
    }

    exprt constant = gen_one(constant_type);

    rhs.copy_to_operands(expr.op0());
    rhs.move_to_operands(constant);
    rhs.type() = expr.op0().type();
  }

  // C11 post-increment/decrement on _Atomic is a sequentially-consistent RMW.
  // The old value and the store must be captured inside a single atomic section
  // so no other thread can observe an intermediate state.
  if (is_atomic_symbol(expr.op0(), ns))
  {
    if (result_is_used)
    {
      // Declare the "old value" temporary outside the atomic section.
      symbolt &old_sym = new_tmp_symbol(expr.op0().type());
      code_declt decl(symbol_expr(old_sym));
      decl.location() = expr.location();
      convert_decl(decl, dest);

      dest.add_instruction(ATOMIC_BEGIN);
      // Save old value then modify — all inside one atomic block.
      code_assignt save(symbol_expr(old_sym), expr.op0());
      save.location() = expr.location();
      copy(save, ASSIGN, dest);
      code_assignt modify(expr.op0(), rhs);
      modify.location() = expr.location();
      copy(modify, ASSIGN, dest);
      dest.add_instruction(ATOMIC_END);

      expr = symbol_expr(old_sym);
    }
    else
    {
      convert_assign_rmw_atomic(expr.op0(), rhs, expr.location(), dest);
      expr.make_nil();
    }
    return;
  }

  code_assignt assignment(expr.op0(), rhs);
  assignment.location() = expr.location();

  goto_programt tmp;
  convert(assignment, tmp);

  // fix up the expression, if needed

  if (result_is_used)
  {
    exprt tmp = expr.op0();
    make_temp_symbol(tmp, dest);
    expr.swap(tmp);
  }
  else
    expr.make_nil();

  dest.destructive_append(tmp);
}

// The below code introduces some special treatment for function calls.
// For example, the following code:
//    ...
//    a = b + foo();
//    ...
// is transformed to:
//    ...
//    <type> return_value$_foo;
//    return_value$_foo = foo();
//    a = b + return_value$_foo;
//    ...
//  However, if the return value of function "foo()" is never
//  used in the program, or the return type <type> of "foo()"
//  is void, the same code is transformed to:
//    ...
//    foo();
//    a = b;
//    ...
void goto_convertt::remove_function_call(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  // If the result of the function call is never used
  // or the return type of the invoked function is "void",
  // we can just call the above function without any
  // further modifications.
  if (!result_is_used || expr.type().id() == "empty")
  {
    assert(expr.operands().size() == 2);
    code_function_callt call;
    call.function() = expr.op0();
    call.arguments() = expr.op1().operands();
    call.location() = expr.location();
    call.lhs().make_nil();
    convert_function_call(call, dest);
    expr.make_nil();
    return;
  }

  symbolt new_symbol;

  new_symbol.name = "return_value$";
  new_symbol.set_type(expr.type());
  new_symbol.location = expr.location();

  // get name of function, if available

  if (expr.id() != "sideeffect" || expr.statement() != "function_call")
    throw "expected function call";

  if (expr.operands().empty())
    throw "function_call expects at least one operand";

  if (expr.op0().is_symbol())
  {
    const irep_idt &identifier = expr.op0().identifier();
    const symbolt *symbol = ns.lookup(identifier);
    assert(symbol);

    std::string new_base_name = id2string(new_symbol.name);

    new_base_name += '_';
    new_base_name += id2string(symbol->name);
    new_base_name += "$" + std::to_string(++tmp_symbol.counter);

    new_symbol.name = new_base_name;
    new_symbol.mode = symbol->mode;
  }

  new_symbol.id = tmp_symbol.prefix + id2string(new_symbol.name);
  new_name(new_symbol);

  {
    // temporary declaration:
    // T return_value$_BLAH$1;
    // where T denotes the type of the return value from the function,
    // BLAH denotes the name of the function
    code_declt decl(symbol_expr(new_symbol));
    decl.location() = new_symbol.location;
    convert_decl(decl, dest);
  }

  // up to this point, we've got the declaration for the temporary return value
  // The next step is to flatten the side effect of this function call
  code_function_callt call;
  call.lhs() = symbol_expr(new_symbol);
  call.function() = expr.op0();
  call.arguments() = expr.op1().operands();
  call.location() = new_symbol.location;

  goto_programt tmp_program;
  const typet &ftype = call.function().type();
  if (ftype.return_type().id() == "constructor")
  {
    // for constructor, we need to add the implicit `this` as the first argument,
    // so convert to:
    // BLAH(&return_value$_BLAH$1, ...)
    side_effect_expr_function_callt ctor_call;
    ctor_call.function() = call.function();
    exprt::operandst &args = ctor_call.arguments();
    address_of_exprt tmp_result = address_of_exprt(call.lhs());
    // first push the implicit `this` arg
    args.push_back(tmp_result);
    // then append the remaining operands
    args.insert(args.end(), call.arguments().begin(), call.arguments().end());
    ctor_call.location() = call.location();
    // now convert this expr to code
    codet ctor_call_code("expression");
    ctor_call_code.location() = call.location();
    ctor_call_code.move_to_operands(ctor_call);

    convert(ctor_call_code, tmp_program);
  }
  else
  {
    // otherwise just convert to something like:
    // return_value$_BLAH$1 = BLAH(...),
    // where BLAH denotes the function name
    codet assignment("assign");
    assignment.reserve_operands(2);
    assignment.copy_to_operands(symbol_expr(new_symbol));
    assignment.move_to_operands(call);
    assignment.location() = new_symbol.location;

    convert(assignment, tmp_program);
  }

  dest.destructive_append(tmp_program);

  expr = symbol_expr(new_symbol);
}

void goto_convertt::replace_new_object(const exprt &object, exprt &dest)
{
  if (dest.id() == "new_object")
    dest = object;
  else
    Forall_operands (it, dest)
      replace_new_object(object, *it);
}

void goto_convertt::remove_cpp_new(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  // For side effect with 'cpp_new' statement, `expr' refers to the side effect that
  // contains an initializer. Technically, this function converts the cpp_new side effect
  // and replaces it with a new symbol if `result_is_used` is true. It's not just simply
  // removing the side effect node in the exprt tree.
  codet call;

  symbolt new_symbol;

  new_symbol.name = "new_ptr$" + std::to_string(++tmp_symbol.counter);
  new_symbol.set_type(expr.type());
  new_symbol.id = tmp_symbol.prefix + id2string(new_symbol.name);

  new_name(new_symbol);

  code_declt decl(symbol_expr(new_symbol));
  decl.location() = new_symbol.location;
  convert_decl(decl, dest);

  call = code_assignt(symbol_expr(new_symbol), expr);

  if (result_is_used) // e.g. used to construct an assignment 'code_assignt'
    expr = symbol_expr(new_symbol);
  else
    expr.make_nil();

  convert(call, dest);
}

void goto_convertt::remove_cpp_delete(exprt &expr, goto_programt &dest)
{
  assert(expr.operands().size() == 1); // cpp_delete expects one operand

  codet tmp(expr.statement());
  tmp.location() = expr.location();
  tmp.copy_to_operands(to_unary_expr(expr).op0());
  tmp.set("destructor", expr.find("destructor"));

  convert_cpp_delete(tmp, dest);

  expr.make_nil();
}

void goto_convertt::remove_temporary_object(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  if (expr.operands().size() != 1 && expr.operands().size() != 0)
    throw "temporary_object takes 0 or 1 operands";

  // A temporary whose value is already materialized in a symbol (e.g. the
  // return_value$ of a call lowered by remove_function_call) needs no second
  // copy: reuse that symbol as the object's identity, so it is destructed
  // exactly once (github #6076, #6075). A prvalue only lowers to a symbol
  // when that symbol is such a compiler temporary, never a named variable.
  if (expr.operands().size() == 1 && expr.op0().is_symbol())
  {
    exprt tmp = expr.op0();
    expr.swap(tmp);
    return;
  }

  symbolt &new_symbol = new_tmp_symbol(expr.type());

  new_symbol.mode = expr.mode();

  // declare this symbol first
  code_declt decl(symbol_expr(new_symbol));
  decl.location() = expr.location();
  convert_decl(decl, dest);

  if (expr.operands().size() == 1)
  {
    codet assignment("assign");
    assignment.reserve_operands(2);
    new_symbol.set_value(expr.op0());
    assignment.copy_to_operands(symbol_expr(new_symbol));
    assignment.move_to_operands(expr.op0());
    assignment.location() = expr.location();

    goto_programt tmp_program;
    convert(assignment, tmp_program);
    dest.destructive_append(tmp_program);
  }

  if (expr.initializer().is_not_nil())
  {
    assert(expr.operands().empty());
    exprt initializer = static_cast<const exprt &>(expr.initializer());
    replace_new_object(symbol_expr(new_symbol), initializer);

    goto_programt tmp_program;
    convert(to_code(initializer), tmp_program);
    dest.destructive_append(tmp_program);
  }

  const locationt loc = expr.location();
  expr = symbol_expr(new_symbol);
  expr.location() = loc;
}

void goto_convertt::remove_statement_expression(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  if (expr.operands().size() != 1)
    throw "statement_expression takes 1 operand";

  if (!expr.op0().is_code())
    throw "statement_expression takes code as operand";

  codet &code = to_code(expr.op0());

  if (!result_is_used)
  {
    convert(code, dest);
    expr.make_nil();
    return;
  }

  // get last statement from block
  if (code.get_statement() != "block")
    throw "statement_expression expects block";

  if (code.operands().empty())
    throw "statement_expression expects non-empty block";

  exprt &last = code.operands().back();
  locationt location = last.location();

  symbolt &new_symbol = new_tmp_symbol(expr.type());

  // declare this symbol first
  code_declt decl(symbol_expr(new_symbol));
  decl.location() = location;
  convert_decl(decl, dest);

  symbol_exprt tmp_symbol_expr(new_symbol.id, new_symbol.get_type());
  tmp_symbol_expr.location() = location;

  if (last.statement() == "expression")
  {
    // we turn this into an assignment
    exprt e = to_code_expression(to_code(last)).expression();
    last = code_assignt(tmp_symbol_expr, e);
    last.location() = location;
  }
  else if (last.statement() == "assign")
  {
    exprt e = to_code_assign(to_code(last)).lhs();
    code_assignt assignment(tmp_symbol_expr, e);
    assignment.location() = location;
    code.operands().push_back(assignment);
  }
  else
    throw "statement_expression expects expression or assignment";

  {
    goto_programt tmp;
    convert(code, tmp);
    dest.destructive_append(tmp);
  }

  expr = tmp_symbol_expr;
}

void goto_convertt::remove_gcc_conditional_expression(
  exprt &expr,
  goto_programt &dest)
{
  if (expr.operands().size() != 2)
    throw "conditional_expression takes two operands";

  // first remove side-effects from condition
  remove_sideeffects(expr.op0(), dest);

  if_exprt if_expr;

  if_expr.cond() = expr.op0();
  if_expr.true_case() = expr.op0();
  if_expr.false_case() = expr.op1();
  if_expr.type() = expr.type();
  if_expr.location() = expr.location();

  if (!if_expr.op0().type().is_bool())
    if_expr.op0().make_typecast(bool_typet());
  if (if_expr.true_case().type() != expr.type())
    if_expr.true_case().make_typecast(expr.type());

  expr.swap(if_expr);

  // there might still be one in expr.op2()
  remove_sideeffects(expr, dest);
}
