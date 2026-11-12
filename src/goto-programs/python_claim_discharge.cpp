#include <goto-programs/python_claim_discharge.h>
#include <goto-programs/exception_globals.h>
#include <irep2/irep2_utils.h>

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
} // namespace

struct python_claim_dischargert::implt
{
  std::optional<std::set<BigInt>> type_ids;
};

python_claim_dischargert::python_claim_dischargert(
  const goto_functionst &goto_functions,
  const std::set<irep_idt> &reachable)
  : impl(
      std::make_unique<implt>(
        implt{exception_type_ids(goto_functions, reachable)}))
{
}

python_claim_dischargert::~python_claim_dischargert() = default;

bool python_claim_dischargert::proves(goto_programt::const_targett a) const
{
  return impl->type_ids &&
         holds_for_every_exception_state(a->guard, *impl->type_ids);
}
