#include <goto-programs/python_claim_discharge.h>
#include <goto-programs/exception_globals.h>
#include <goto-programs/remove_exceptions.h>
#include <irep2/irep2_utils.h>
#include <util/symtab/context.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <vector>

namespace
{
bool names(const expr2tc &e, const char *id)
{
  return !is_nil_expr(e) && is_symbol2t(e) && to_symbol2t(e).thename == id;
}

bool mentions(const expr2tc &e, const std::function<bool(const expr2tc &)> &p)
{
  if (is_nil_expr(e))
    return false;
  if (p(e))
    return true;
  bool found = false;
  e->foreach_operand(
    [&](const expr2tc &o) { found = found || mentions(o, p); });
  return found;
}

bool mentions_exception_global(const expr2tc &e)
{
  return mentions(e, [](const expr2tc &x) {
    return names(x, exception_globals::thrown_id) ||
           names(x, exception_globals::typeid_id);
  });
}

std::optional<BigInt> constant_of(const expr2tc &e)
{
  if (is_constant_int2t(e))
    return to_constant_int2t(e).value;
  if (is_constant_bool2t(e))
    return BigInt(to_constant_bool2t(e).value ? 1 : 0);
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Uncaught-exception checks
// ---------------------------------------------------------------------------

/// The values of `__ESBMC_exc_thrown` and `__ESBMC_exc_typeid` in one state.
struct exception_statet
{
  BigInt thrown, type_id;
};

std::optional<BigInt>
eval_exception(const expr2tc &e, const exception_statet &s)
{
  if (names(e, exception_globals::thrown_id))
    return s.thrown;
  if (names(e, exception_globals::typeid_id))
    return s.type_id;
  // A cast that cannot drop bits keeps the value.
  if (is_typecast2t(e) && is_bv_type(e->type))
  {
    const type2tc &from = to_typecast2t(e).from->type;
    if (
      is_bool_type(from) ||
      (is_bv_type(from) && e->type->get_width() >= from->get_width()))
      return eval_exception(to_typecast2t(e).from, s);
  }
  return constant_of(e);
}

std::optional<bool>
eval_exception_bool(const expr2tc &e, const exception_statet &s);

std::optional<bool> eval_exception_binary(
  const expr2tc &a,
  const expr2tc &b,
  const exception_statet &s,
  bool is_and)
{
  const auto x = eval_exception_bool(a, s), y = eval_exception_bool(b, s);
  if (!x || !y)
    return std::nullopt;
  return is_and ? (*x && *y) : (*x || *y);
}

std::optional<bool> eval_exception_equal(
  const expr2tc &a,
  const expr2tc &b,
  const exception_statet &s)
{
  const auto x = eval_exception(a, s), y = eval_exception(b, s);
  if (!x || !y)
    return std::nullopt;
  return *x == *y;
}

std::optional<bool> negated(std::optional<bool> v)
{
  return v ? std::optional<bool>(!*v) : std::nullopt;
}

std::optional<bool>
eval_exception_bool(const expr2tc &e, const exception_statet &s)
{
  if (is_not2t(e))
    return negated(eval_exception_bool(to_not2t(e).value, s));
  if (is_and2t(e))
    return eval_exception_binary(
      to_and2t(e).side_1, to_and2t(e).side_2, s, true);
  if (is_or2t(e))
    return eval_exception_binary(
      to_or2t(e).side_1, to_or2t(e).side_2, s, false);
  if (is_equality2t(e))
    return eval_exception_equal(
      to_equality2t(e).side_1, to_equality2t(e).side_2, s);
  if (is_notequal2t(e))
    return negated(eval_exception_equal(
      to_notequal2t(e).side_1, to_notequal2t(e).side_2, s));
  const auto v = eval_exception(e, s);
  return v ? std::optional<bool>(!v->is_zero()) : std::nullopt;
}

bool assigns(const goto_programt::instructiont &i, const char *id)
{
  return i.is_assign() && names(to_code_assign2t(i.code).target, id);
}

/// `typeid = c` with c a constant other than 0.
bool arms_type_id(const goto_programt::instructiont &i)
{
  if (!assigns(i, exception_globals::typeid_id))
    return false;
  const auto c = constant_of(to_code_assign2t(i.code).source);
  return c && !c->is_zero();
}

/// `thrown = 0` right before @p it, with nothing jumping in between.
bool follows_clear(
  const goto_programt::instructionst &body,
  goto_programt::instructionst::const_iterator it)
{
  return it != body.begin() && !it->is_target() &&
         assigns(*std::prev(it), exception_globals::thrown_id) &&
         is_false(to_code_assign2t(std::prev(it)->code).source);
}

bool spawns_thread(const goto_programt::instructiont &i)
{
  return i.is_function_call() && names(
                                   to_code_function_call2t(i.code).function,
                                   "c:@F@__ESBMC_spawn_thread");
}

/// Whether the instruction at @p it keeps "thrown implies a non-zero type id
/// that was written" true, recording the type id it writes.
bool accepts_exception_write(
  const goto_programt::instructionst &body,
  goto_programt::instructionst::const_iterator it,
  std::set<BigInt> &ids)
{
  if (!it->is_assign())
    return !mentions_exception_global(it->code) && !spawns_thread(*it);
  const code_assign2t &a = to_code_assign2t(it->code);
  const bool thrown = names(a.target, exception_globals::thrown_id);
  const bool type_id = names(a.target, exception_globals::typeid_id);
  if (mentions_exception_global(a.source))
    return false;
  if (!thrown && !type_id)
    return !mentions_exception_global(a.target);
  const std::optional<BigInt> c = constant_of(a.source);
  if (!c)
    return false;
  if (thrown)
    return c->is_zero() ||
           (std::next(it) != body.end() && arms_type_id(*std::next(it)));
  if (c->is_zero())
    return follows_clear(body, it);
  ids.insert(*c);
  return true;
}

/// The type ids `__ESBMC_exc_typeid` can hold while an exception is in
/// flight, or none when a reachable write to the exception globals has
/// another shape. Accepted writes are `thrown = 0`; `thrown = 1` followed at
/// once by `typeid = c` with c not 0; `typeid = c` with c not 0; and
/// `typeid = 0` right after `thrown = 0` and not a jump target, as at program
/// start. A second thread could see the state between `thrown = 1` and its
/// type id, so a program that spawns one gets none.
std::optional<std::set<BigInt>> exception_type_ids(
  const goto_functionst &goto_functions,
  const std::set<irep_idt> &reachable)
{
  std::set<BigInt> ids;
  for (const auto &[id, fn] : goto_functions.function_map)
  {
    if (!fn.body_available || !reachable.count(id))
      continue;
    const auto &body = fn.body.instructions;
    for (auto it = body.begin(); it != body.end(); ++it)
      if (!accepts_exception_write(body, it, ids))
        return std::nullopt;
  }
  return ids;
}

bool holds_for_every_exception_state(
  const expr2tc &guard,
  const std::set<BigInt> &ids)
{
  if (!mentions_exception_global(guard))
    return false;
  std::vector<exception_statet> states{{BigInt(0), BigInt(0)}};
  for (const BigInt &id : ids)
  {
    states.push_back({BigInt(0), id});
    states.push_back({BigInt(1), id});
  }
  return std::all_of(states.begin(), states.end(), [&](const auto &s) {
    const auto v = eval_exception_bool(guard, s);
    return v && *v;
  });
}

// ---------------------------------------------------------------------------
// Facts that hold on every path to an instruction
// ---------------------------------------------------------------------------

/// The signs an integer local may have, as a mask of these.
enum : unsigned
{
  negative = 1,
  zero = 2,
  positive = 4,
  any_sign = 7
};

/// What is known about one integer or pointer local. A local with no entry
/// may have any value.
struct factt
{
  unsigned signs = any_sign;
  bool non_null = false;

  bool operator==(const factt &o) const
  {
    return signs == o.signs && non_null == o.non_null;
  }
};

using factst = std::map<irep_idt, factt>;

factst join(const factst &a, const factst &b)
{
  factst out;
  for (const auto &[name, fa] : a)
  {
    auto it = b.find(name);
    if (it == b.end())
      continue;
    const factt f{
      fa.signs | it->second.signs, fa.non_null && it->second.non_null};
    if (!(f == factt()))
      out[name] = f;
  }
  return out;
}

/// `x rel 0` or `p rel NULL` for a local, with @p signs the signs of x for
/// which it holds.
struct atomt
{
  irep_idt name;
  unsigned signs;
  bool pointer;
};

bool is_zero_constant(const expr2tc &e)
{
  if (is_typecast2t(e))
    return is_zero_constant(to_typecast2t(e).from);
  if (is_symbol2t(e))
    return to_symbol2t(e).thename == "NULL";
  return is_constant_int2t(e) && to_constant_int2t(e).value.is_zero();
}

/// The signs of x for which `x rel 0` holds, or 0 for no comparison.
unsigned signs_of_comparison(const expr2tc &e)
{
  if (is_lessthan2t(e))
    return negative;
  if (is_lessthanequal2t(e))
    return negative | zero;
  if (is_greaterthan2t(e))
    return positive;
  if (is_greaterthanequal2t(e))
    return positive | zero;
  if (is_equality2t(e))
    return zero;
  return is_notequal2t(e) ? negative | positive : 0;
}

/// The signs of x for which `0 rel x` holds, given those for `x rel 0`.
unsigned mirror(unsigned signs)
{
  return ((signs & negative) ? positive : 0) |
         ((signs & positive) ? negative : 0) | (signs & zero);
}

bool is_equality_test(unsigned signs)
{
  return signs == zero || signs == (negative | positive);
}

class functiont
{
public:
  functiont(const goto_functiont &fn, const contextt &context)
    : fn(fn), context(context)
  {
    for (const auto &i : fn.body.instructions)
    {
      collect_address_taken(i.code);
      collect_address_taken(i.guard);
    }
  }

  const goto_functiont &fn;

  /// Integer and pointer locals that nothing but their own function writes.
  bool tracked(const expr2tc &e) const
  {
    if (is_nil_expr(e) || !is_symbol2t(e))
      return false;
    if (!is_bv_type(e->type) && !is_pointer_type(e->type))
      return false;
    const irep_idt &name = to_symbol2t(e).thename;
    const symbolt *s = context.find_symbol(name);
    return !taken.count(name) && s && !s->static_lifetime;
  }

  /// `x rel 0`, `0 rel x`, `-x == 0`, `-x != 0`, `p == NULL`, `p != NULL`.
  std::optional<atomt> atom(const expr2tc &e) const
  {
    unsigned signs = signs_of_comparison(e);
    if (!signs)
      return std::nullopt;
    std::vector<expr2tc> sides;
    e->foreach_operand([&](const expr2tc &o) { sides.push_back(o); });
    expr2tc x = sides[0], k = sides[1];
    if (!is_zero_constant(k))
    {
      std::swap(x, k);
      signs = mirror(signs);
    }
    // -x is 0 exactly when x is, in two's complement too.
    if (is_neg2t(x) && is_equality_test(signs))
      x = to_neg2t(x).value;
    if (!tracked(x) || !is_zero_constant(k))
      return std::nullopt;
    const bool pointer = is_pointer_type(x->type);
    if (pointer && !is_equality_test(signs))
      return std::nullopt;
    return atomt{to_symbol2t(x).thename, signs, pointer};
  }

private:
  const contextt &context;
  std::set<irep_idt> taken;

  // Every symbol under an address-of counts as taken.
  void collect_address_taken(const expr2tc &e)
  {
    mentions(e, [this](const expr2tc &x) {
      if (is_address_of2t(x))
        mentions(to_address_of2t(x).ptr_obj, [this](const expr2tc &y) {
          if (is_symbol2t(y))
            taken.insert(to_symbol2t(y).thename);
          return false;
        });
      return false;
    });
  }
};

void add_condition(
  const functiont &f,
  factst &facts,
  const expr2tc &e,
  bool positive)
{
  if (is_not2t(e))
    return add_condition(f, facts, to_not2t(e).value, !positive);
  if ((is_and2t(e) && positive) || (is_or2t(e) && !positive))
  {
    e->foreach_operand(
      [&](const expr2tc &o) { add_condition(f, facts, o, positive); });
    return;
  }
  const auto a = f.atom(e);
  if (!a)
    return;
  const unsigned signs = positive ? a->signs : any_sign & ~a->signs;
  factt &fact = facts[a->name];
  if (!a->pointer)
    fact.signs &= signs;
  else if (signs != zero)
    fact.non_null = true;
}

std::optional<bool> eval_atom(const factst &facts, const atomt &a)
{
  auto it = facts.find(a.name);
  if (it == facts.end())
    return std::nullopt;
  const factt &f = it->second;
  if (a.pointer)
    return f.non_null ? std::optional<bool>(a.signs != zero) : std::nullopt;
  if (!(f.signs & ~a.signs))
    return true;
  return (f.signs & a.signs) ? std::nullopt : std::optional<bool>(false);
}

std::optional<bool>
eval(const functiont &f, const factst &facts, const expr2tc &e);

std::optional<bool> eval_connective(
  const functiont &f,
  const factst &facts,
  const expr2tc &e,
  bool is_and)
{
  // A side that decides the connective decides it alone.
  bool all_known = true, decided = false;
  e->foreach_operand([&](const expr2tc &o) {
    const auto v = eval(f, facts, o);
    all_known = all_known && v;
    decided = decided || (v && *v != is_and);
  });
  if (decided)
    return !is_and;
  return all_known ? std::optional<bool>(is_and) : std::nullopt;
}

std::optional<bool>
eval(const functiont &f, const factst &facts, const expr2tc &e)
{
  if (is_true(e) || is_false(e))
    return is_true(e);
  if (is_not2t(e))
    return negated(eval(f, facts, to_not2t(e).value));
  if (is_and2t(e) || is_or2t(e))
    return eval_connective(f, facts, e, is_and2t(e));
  if (const auto a = f.atom(e))
    return eval_atom(facts, *a);
  return std::nullopt;
}

/// alloca has no failure outcome for a size it can represent.
bool is_fixed_size_alloca(const code_function_call2t &call)
{
  if (!names(call.function, "c:@F@__ESBMC_alloca") || call.operands.size() != 1)
    return false;
  expr2tc size = call.operands[0];
  while (is_typecast2t(size))
    size = to_typecast2t(size).from;
  return is_constant_int2t(size) &&
         to_constant_int2t(size).value < BigInt((unsigned long)1 << 40);
}

// Forgets every symbol @p target mentions, so a write through a member,
// index or cast is covered too.
void kill(factst &facts, const expr2tc &target)
{
  mentions(target, [&](const expr2tc &x) {
    if (is_symbol2t(x))
      facts.erase(to_symbol2t(x).thename);
    return false;
  });
}

using call_argst = std::map<irep_idt, std::vector<std::vector<bool>>>;
using instruction_factst =
  std::map<const goto_programt::instructiont *, factst>;
} // namespace

struct python_claim_dischargert::implt
{
  std::set<irep_idt> address_taken;
  std::optional<std::set<BigInt>> type_ids;
  std::map<irep_idt, std::unique_ptr<functiont>> functions;
  std::map<irep_idt, std::set<unsigned>> non_null_params;
  std::set<irep_idt> non_null_returns;
  instruction_factst at;
  std::map<const goto_programt::instructiont *, irep_idt> owner;
  bool converged = false;

  implt(
    const goto_functionst &gf,
    const contextt &context,
    const std::set<irep_idt> &reachable)
    : address_taken(address_taken_functions(gf, context)),
      type_ids(exception_type_ids(gf, reachable))
  {
    for (const auto &[id, fn] : gf.function_map)
      if (fn.body_available && reachable.count(id))
        functions[id] = std::make_unique<functiont>(fn, context);
    // Parameter and return facts only grow, so this reaches a fixpoint. The
    // bound guards against a slow one, after which nothing is proved.
    for (unsigned round = 0; round < 50 && !converged; ++round)
      converged = !step();
  }

  bool
  is_non_null(const functiont &f, const factst &facts, const expr2tc &e) const
  {
    if (is_address_of2t(e))
      return true;
    if (is_typecast2t(e) && is_pointer_type(to_typecast2t(e).from->type))
      return is_non_null(f, facts, to_typecast2t(e).from);
    if (!f.tracked(e))
      return false;
    auto it = facts.find(to_symbol2t(e).thename);
    return it != facts.end() && it->second.non_null;
  }

  factst entry_facts(const irep_idt &id, const functiont &f) const
  {
    factst facts;
    auto it = non_null_params.find(id);
    if (it == non_null_params.end())
      return facts;
    const auto &names = to_code_type(f.fn.type).argument_names;
    for (unsigned i : it->second)
      if (i < names.size())
        facts[names[i]].non_null = true;
    return facts;
  }

  void assign(
    const functiont &f,
    factst &facts,
    const expr2tc &target,
    bool non_null,
    const expr2tc &source) const
  {
    kill(facts, target);
    if (!f.tracked(target))
      return;
    factt &fact = facts[to_symbol2t(target).thename];
    fact.non_null = non_null;
    if (is_nil_expr(source) || !is_constant_int2t(source))
      return;
    const BigInt &v = to_constant_int2t(source).value;
    fact.signs = v.is_zero() ? zero : v.is_negative() ? negative : positive;
  }

  /// The facts after @p i for the fall-through edge, given those before it.
  /// False for an instruction this does not model.
  bool transfer(
    const functiont &f,
    const goto_programt::instructiont &i,
    factst &facts) const
  {
    switch (i.type)
    {
    case ASSIGN:
    {
      const code_assign2t &a = to_code_assign2t(i.code);
      const bool non_null =
        is_pointer_type(a.target->type) && is_non_null(f, facts, a.source);
      assign(f, facts, a.target, non_null, a.source);
      return true;
    }
    case FUNCTION_CALL:
    {
      const code_function_call2t &c = to_code_function_call2t(i.code);
      const bool non_null =
        is_fixed_size_alloca(c) ||
        (is_symbol2t(c.function) &&
         non_null_returns.count(to_symbol2t(c.function).thename));
      assign(f, facts, c.ret, non_null, expr2tc());
      return true;
    }
    case DECL:
      facts.erase(to_code_decl2t(i.code).value);
      return true;
    case DEAD:
      facts.erase(to_code_dead2t(i.code).value);
      return true;
    case ASSUME:
    case ASSERT:
      add_condition(f, facts, i.guard, true);
      return true;
    case SKIP:
    case LOCATION:
    case RETURN:
    case OTHER:
    case ATOMIC_BEGIN:
    case ATOMIC_END:
    case END_FUNCTION:
      return true;
    default:
      return false;
    }
  }

  /// The facts before each instruction of @p f, or false when an instruction
  /// this does not model is reached or the iteration bound is hit.
  bool
  analyse(const irep_idt &id, const functiont &f, instruction_factst &in) const
  {
    const auto &body = f.fn.body.instructions;
    if (body.empty())
      return true;
    std::vector<goto_programt::const_targett> work{body.begin()};
    in[&*body.begin()] = entry_facts(id, f);
    auto flow = [&](goto_programt::const_targett t, const factst &facts) {
      auto [it, fresh] = in.emplace(&*t, facts);
      if (!fresh)
      {
        factst joined = join(it->second, facts);
        if (joined == it->second)
          return;
        it->second = std::move(joined);
      }
      work.push_back(t);
    };
    for (unsigned steps = 0; !work.empty(); ++steps)
    {
      const goto_programt::const_targett t = work.back();
      work.pop_back();
      factst facts = in.at(&*t);
      if (t->is_goto())
      {
        factst taken = facts;
        add_condition(f, taken, t->guard, true);
        for (const auto &target : t->targets)
          flow(target, taken);
        add_condition(f, facts, t->guard, false);
      }
      else if (steps > 200000 || !transfer(f, *t, facts))
        return false;
      const bool falls_through =
        !t->is_end_function() && !(t->is_goto() && is_true(t->guard));
      if (falls_through && std::next(t) != body.end())
        flow(std::next(t), facts);
    }
    return true;
  }

  void record_call(
    const functiont &f,
    const goto_programt::instructiont &i,
    const factst &facts,
    call_argst &call_args) const
  {
    const code_function_call2t &c = to_code_function_call2t(i.code);
    if (!is_symbol2t(c.function))
      return;
    std::vector<bool> args;
    for (const expr2tc &a : c.operands)
      args.push_back(is_pointer_type(a->type) && is_non_null(f, facts, a));
    call_args[to_symbol2t(c.function).thename].push_back(args);
  }

  /// Records @p f's calls and assertions. Returns whether @p f returns and
  /// every return of it is non-null.
  bool summarise(
    const irep_idt &id,
    const functiont &f,
    const instruction_factst &in,
    call_argst &call_args)
  {
    bool all_non_null = true, any = false;
    for (const auto &i : f.fn.body.instructions)
    {
      auto it = in.find(&i);
      if (it == in.end())
        continue;
      if (i.is_function_call())
        record_call(f, i, it->second, call_args);
      if (i.is_assert())
      {
        at[&i] = it->second;
        owner[&i] = id;
      }
      if (!i.is_return())
        continue;
      const expr2tc &v = to_code_return2t(i.code).operand;
      any = true;
      all_non_null = all_non_null && !is_nil_expr(v) &&
                     is_pointer_type(v->type) && is_non_null(f, it->second, v);
    }
    return any && all_non_null;
  }

  /// The parameters every call site passes non-null, for the functions whose
  /// address is never taken, so that every caller is known.
  std::map<irep_idt, std::set<unsigned>>
  non_null_arguments(const call_argst &call_args) const
  {
    std::map<irep_idt, std::set<unsigned>> params;
    for (const auto &[callee, sites] : call_args)
    {
      if (address_taken.count(callee))
        continue;
      for (unsigned n = 0; n < sites.front().size(); ++n)
        if (std::all_of(sites.begin(), sites.end(), [n](const auto &args) {
              return n < args.size() && args[n];
            }))
          params[callee].insert(n);
    }
    return params;
  }

  /// One forward pass over every function. Returns whether a parameter or
  /// return fact changed.
  bool step()
  {
    call_argst call_args;
    std::set<irep_idt> returns;
    for (const auto &[id, f] : functions)
    {
      instruction_factst in;
      if (!analyse(id, *f, in))
      {
        // Nothing is known at its calls, so its callees learn nothing.
        for (const auto &i : f->fn.body.instructions)
          if (i.is_function_call())
            record_call(*f, i, factst(), call_args);
        continue;
      }
      if (summarise(id, *f, in, call_args))
        returns.insert(id);
    }
    auto params = non_null_arguments(call_args);
    const bool changed =
      returns != non_null_returns || params != non_null_params;
    non_null_returns = std::move(returns);
    non_null_params = std::move(params);
    return changed;
  }
};

python_claim_dischargert::python_claim_dischargert(
  const goto_functionst &goto_functions,
  const contextt &context,
  const std::set<irep_idt> &reachable)
  : impl(std::make_unique<implt>(goto_functions, context, reachable))
{
}

python_claim_dischargert::~python_claim_dischargert() = default;

bool python_claim_dischargert::proves(goto_programt::const_targett a) const
{
  if (
    impl->type_ids &&
    holds_for_every_exception_state(a->guard, *impl->type_ids))
    return true;
  if (!impl->converged)
    return false;
  auto facts = impl->at.find(&*a);
  if (facts == impl->at.end())
    return false;
  const functiont &f = *impl->functions.at(impl->owner.at(&*a));
  const auto v = eval(f, facts->second, a->guard);
  return v && *v;
}
