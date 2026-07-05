#include <goto-symex/goto_symex.h>
#include <goto-symex/witnesses.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/prefix.h>
#include <irep2/irep2.h>

static void substitute_result(expr2tc &e, const expr2tc &ret_val)
{
  if (is_symbol2t(e) && to_symbol2t(e).thename == "\\result")
  {
    e = ret_val;
    return;
  }
  if (is_constant_int2t(e))
  {
    if (
      is_unsignedbv_type(ret_val->type) &&
      to_constant_int2t(e).value.is_negative())
      log_warning(
        "witness: function_return constraint compares signed constant {} "
        "against unsigned return type; constraint may be trivially false",
        integer2string(to_constant_int2t(e).value));
    e = from_integer(to_constant_int2t(e).value, ret_val->type);
    return;
  }
  e->Foreach_operand([&](expr2tc &op) { substitute_result(op, ret_val); });
}

void goto_symext::symex_witness_function_return(
  expr2tc ret_val,
  const irep_idt &call_line)
{
  if (cur_state->cur_seg >= cur_state->witness_segs.size())
    return;

  const auto &seg = cur_state->witness_segs[cur_state->cur_seg];
  for (const waypoint &wp : seg)
  {
    if (wp.type != waypoint::function_return)
      continue;
    if (!wp.parsed_cond.valid)
      continue;
    if (wp.line_id != call_line)
      continue;

    cur_state->rename(ret_val);

    expr2tc constraint = wp.parsed_cond.expr;
    substitute_result(constraint, ret_val);

    if (wp.action == waypoint::avoid)
      assume(not2tc(constraint));
    else
    {
      assume(constraint);
      cur_state->advance_witness_position();
    }
    return;
  }
}

// Handle a function_enter waypoint for the call at call_line.
// Returns true if the path was killed (avoid matched); false otherwise.
// For avoid: kills the path with assume(false).
// For follow: advances the witness position.
// In both cases the caller should NOT enter the function body when true is returned.
bool goto_symext::symex_witness_function_enter(const irep_idt &call_line)
{
  if (cur_state->cur_seg >= cur_state->witness_segs.size())
    return false;

  const auto &seg = cur_state->witness_segs[cur_state->cur_seg];
  for (const waypoint &wp : seg)
  {
    if (wp.type != waypoint::function_enter)
      continue;

    const bool loc_matches = !wp.line_id.empty() && wp.line_id == call_line;

    if (loc_matches)
    {
      if (wp.action == waypoint::avoid)
      {
        assume(gen_false_expr());
        return true;
      }
      cur_state->advance_witness_position();
      return false;
    }

    // No match: FOLLOW is ordered (stop scanning); AVOID is persistent (continue).
    if (wp.action != waypoint::avoid)
      return false;
  }
  return false;
}

void goto_symext::symex_witness_branching(
  const expr2tc &old_guard,
  const expr2tc &new_guard,
  bool forward,
  bool &new_guard_true,
  bool &new_guard_false,
  const goto_programt::instructiont &instruction)
{
  if (!validate_witness || !forward || is_constant(old_guard))
    return;

  const auto &loc = cur_state->source.pc->location;

  auto force_goto = [&](bool taken) {
    new_guard_true = taken;
    new_guard_false = !taken;
    expr2tc dir_cond = taken ? new_guard : not2tc(new_guard);
    do_simplify(dir_cond);
    if (!is_true(dir_cond))
      assume(dir_cond);
  };

  // O(1) lookup for branching(cycle): persistent across all iterations,
  // independent of cur_seg.  Built once at init in execution_state.cpp.
  {
    auto it = cur_state->cycle_branch_map.find(loc.get_line());
    if (it != cur_state->cycle_branch_map.end())
    {
      const bool goto_taken =
        instruction.flipped_guard ? it->second : !it->second;
      force_goto(goto_taken);
      return;
    }
  }

  if (cur_state->cur_seg >= cur_state->witness_segs.size())
    return;

  const auto &seg = cur_state->witness_segs[cur_state->cur_seg];

  for (const auto &wp : seg)
  {
    if (wp.type != waypoint::branching)
      continue;

    const bool is_avoid = (wp.action == waypoint::avoid);
    const bool loc_matches =
      !wp.line_id.empty() && wp.line_id == loc.get_line() &&
      (wp.column_id.empty() || wp.column_id == loc.get_column());

    if (!loc_matches)
    {
      if (!is_avoid)
        break;
      continue;
    }

    if (wp.value != "true" && wp.value != "false")
    {
      bool case_matches = false;
      for (const auto &id : instruction.switch_case_ids)
        if (id == wp.value)
        {
          case_matches = true;
          break;
        }

      if (is_avoid)
      {
        if (!case_matches)
          continue;
        force_goto(false);
      }
      else
      {
        force_goto(case_matches);
        if (case_matches)
          cur_state->advance_witness_position();
      }
      break;
    }

    const bool direction_true = (wp.value == "true") ^ is_avoid;
    const bool goto_taken =
      instruction.flipped_guard ? direction_true : !direction_true;
    force_goto(goto_taken);
    if (wp.action == waypoint::cycle)
      cur_state->cycle_branch_map[loc.get_line()] = (wp.value == "true");
    if (!is_avoid)
      cur_state->advance_witness_position();
    break;
  }
}

void goto_symext::symex_witness_assert(
  expr2tc &new_expr,
  const std::string &msg)
{
  if (
    !validate_witness || has_prefix(msg, "unwinding assertion loop") ||
    has_prefix(msg, "termination per-loop marker") ||
    has_prefix(msg, "termination abort-call marker"))
    return;

  const size_t seg = cur_state->cur_seg;
  const waypoint *target_wp = nullptr;
  if (seg < cur_state->witness_segs.size())
    for (const auto &wp : cur_state->witness_segs[seg])
      if (wp.type == waypoint::target)
      {
        target_wp = &wp;
        break;
      }

  if (
    !target_wp ||
    cur_state->source.pc->location.get_line() != target_wp->line_id ||
    cur_state->witness_target_reached)
    new_expr = gen_true_expr();
  else
  {
    cur_state->advance_witness_position();
    cur_state->witness_target_reached = true;
  }
}
