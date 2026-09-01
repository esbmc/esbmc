#include <goto-programs/goto_houdini_invariants.h>
#include <goto-programs/goto_loop_invariant.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/loopst.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/expr/expr_util.h>
#include <util/arith/ieee_float.h>
#include <util/message/message.h>
#include <algorithm>
#include <vector>

namespace
{
/// Pool caps. The cost of a round is linear in the pool, and every candidate
/// that survives adds a conjunct to the assumption the other candidates are
/// checked under, so a large pool is slow rather than wrong. These bounds keep
/// a round comparable to a plain BMC run on the same program.
constexpr size_t kMaxConstantsPerType = 6;
constexpr size_t kMaxCandidatesPerLoop = 24;

bool is_scalar_arith(const type2tc &t)
{
  return is_unsignedbv_type(t) || is_signedbv_type(t) || is_floatbv_type(t);
}

/// Literals the program itself mentions. Guessing from the program's own
/// constants is what makes a small pool enough: the bound that closes a proof
/// is nearly always already written down in the loop body or the property
/// (`x > 1` for `x = 2*x - 1`, whose literals are 2 and 1).
void collect_constants(const expr2tc &expr, std::vector<expr2tc> &out)
{
  if (!expr)
    return;

  if (is_constant_int2t(expr) || is_constant_floatbv2t(expr))
  {
    if (std::find(out.begin(), out.end(), expr) == out.end())
      out.push_back(expr);
    return;
  }

  expr->foreach_operand(
    [&out](const expr2tc &sub) { collect_constants(sub, out); });
}

void collect_function_constants(
  const goto_programt &body,
  std::vector<expr2tc> &out)
{
  for (auto it = body.instructions.begin(); it != body.instructions.end(); ++it)
  {
    collect_constants(it->guard, out);
    if (it->is_assign())
      collect_constants(to_code_assign2t(it->code).source, out);
  }
}

/// Re-type a literal to the variable it will be compared against.
///
/// The pool is built from the program's own constants, and the frontend records
/// those at the type they were *written*: `float x = 2` leaves a signedbv 2
/// behind a typecast, so comparing a float variable against the collected
/// constant needs the value rebuilt at the float type. A typecast wrapper is
/// not enough -- it does not fold to a literal here, and an unfolded typecast
/// in a candidate defeats the point of guessing simple facts. Returns nil when
/// the value has no representation at the target type.
expr2tc retype_constant(const expr2tc &constant, const type2tc &target)
{
  if (constant->type == target)
    return constant;

  if (is_floatbv_type(target) && is_constant_floatbv2t(constant))
  {
    ieee_floatt f = to_constant_floatbv2t(constant).value;
    f.change_spec(ieee_float_spect(to_floatbv_type(target)));
    return constant_floatbv2tc(f);
  }

  BigInt value;
  if (is_constant_int2t(constant))
    value = to_constant_int2t(constant).value;
  else if (is_constant_floatbv2t(constant))
    value = to_constant_floatbv2t(constant).value.to_integer();
  else
    return expr2tc();

  if (is_floatbv_type(target))
  {
    ieee_floatt f(ieee_float_spect(to_floatbv_type(target)));
    f.from_integer(value);
    return constant_floatbv2tc(f);
  }

  // A negative value has no unsigned representation, and wrapping it would
  // make a guess about a number the program never mentions.
  if (is_unsignedbv_type(target) && value.is_negative())
    return expr2tc();

  return constant_int2tc(target, value);
}

/// The four order templates. Equality is deliberately absent: it survives only
/// for a variable the loop does not really change, where it buys nothing, and
/// it doubles the pool.
void build_templates(
  const expr2tc &var,
  const expr2tc &konst,
  std::vector<expr2tc> &out)
{
  out.push_back(greaterthanequal2tc(var, konst));
  out.push_back(greaterthan2tc(var, konst));
  out.push_back(lessthanequal2tc(var, konst));
  out.push_back(lessthan2tc(var, konst));
}

std::vector<expr2tc>
candidates_for_var(const expr2tc &var, const std::vector<expr2tc> &constants)
{
  std::vector<expr2tc> out;
  size_t used = 0;

  for (const expr2tc &c : constants)
  {
    if (used >= kMaxConstantsPerType)
      break;

    const expr2tc typed = retype_constant(c, var->type);
    if (!typed)
      continue;

    ++used;
    build_templates(var, typed, out);
  }
  return out;
}

/// Candidates over every scalar the loop modifies, ordered so the pool is
/// identical between runs (loop_varst hashes on the string-pool index, which
/// varies).
std::vector<expr2tc>
candidates_for_loop(const loopst &loop, const std::vector<expr2tc> &constants)
{
  std::vector<expr2tc> vars;
  for (const expr2tc &var : loop.get_modified_loop_vars())
    if (is_symbol2t(var) && is_scalar_arith(var->type))
      vars.push_back(var);

  std::sort(vars.begin(), vars.end(), [](const expr2tc &a, const expr2tc &b) {
    return a->pretty() < b->pretty();
  });

  std::vector<expr2tc> out;
  for (const expr2tc &var : vars)
  {
    for (expr2tc &cand : candidates_for_var(var, constants))
    {
      if (out.size() >= kMaxCandidatesPerLoop)
        return out;
      out.push_back(cand);
    }
  }
  return out;
}

/// True when a LOOP_INVARIANT already sits where goto_loop_invariant's
/// extractor searches. A user-written or affine-synthesised invariant is
/// authoritative: both would be folded into the same claim set, and a rejected
/// guess would then fail alongside the invariant that was actually wanted.
bool has_existing_invariant(
  const goto_programt::targett &head,
  const goto_programt::targett &begin)
{
  goto_programt::targett it = head;
  for (size_t steps = 0; it != begin && steps < 16; ++steps)
  {
    --it;
    if (it->is_loop_invariant())
      return true;
    if (it->is_goto() || it->is_function_call())
      return false;
  }
  return false;
}

void emit_candidate(
  goto_functiont &goto_function,
  const goto_programt::targett &anchor,
  const expr2tc &candidate,
  const std::string &id)
{
  goto_programt::instructiont inv;
  inv.type = LOOP_INVARIANT;
  inv.location = anchor->location;
  inv.function = anchor->function;
  inv.location.property(kHoudiniCandidatePrefix + id);
  inv.add_loop_invariant(candidate);
  goto_function.body.instructions.insert(anchor, inv);
}

} // namespace

std::set<std::string> goto_houdini_emit_candidates(
  goto_functionst &goto_functions,
  const std::optional<std::set<std::string>> &keep)
{
  std::set<std::string> emitted;
  size_t next_id = 0;

  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available || it->second.body.hide)
      continue;

    std::vector<expr2tc> constants;
    collect_function_constants(it->second.body, constants);

    goto_loopst loops(it->first, goto_functions, it->second);
    for (auto &loop : loops.get_loops())
    {
      const goto_programt::targett anchor = loop.get_original_loop_head();
      const goto_programt::targett begin =
        it->second.body.instructions.begin();

      if (has_existing_invariant(anchor, begin))
        continue;

      for (const expr2tc &cand : candidates_for_loop(loop, constants))
      {
        const std::string id = std::to_string(next_id++);
        if (keep && keep->count(id) == 0)
          continue;
        emit_candidate(it->second, anchor, cand, id);
        emitted.insert(id);
      }
    }
  }

  if (!emitted.empty())
    log_status("Houdini: {} candidate invariant(s)", emitted.size());

  return emitted;
}
