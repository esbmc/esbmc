#include <goto-symex/engine/goto_symex.h>
#include <goto-symex/waypoint.h>
#include <goto-symex/witness/witnesses.h>
#include <util/arith/arith_tools.h>
#include <util/expr/expr_util.h>
#include <util/message/message.h>
#include <util/base/filesystem.h>
#include <util/base/prefix.h>
#include <irep2/irep2.h>

/// A float literal compared with \result of another type: C converts both to
/// the wider floating type, or an integer to the floating one (C11 6.3.1.8).
static void to_common_float_type(expr2tc &rel)
{
  std::vector<expr2tc *> ops;
  rel->Foreach_operand([&](expr2tc &op) { ops.push_back(&op); });
  type2tc common;
  for (const expr2tc *op : ops)
    if (
      is_floatbv_type(*op) &&
      (!common || (*op)->type->get_width() > common->get_width()))
      common = (*op)->type;
  if (!common)
    return;
  // A pointer or aggregate result never equals a float.
  for (const expr2tc *op : ops)
    if (!is_number_type(*op))
    {
      rel = gen_false_expr();
      return;
    }
  for (expr2tc *op : ops)
    if ((*op)->type != common)
      *op = typecast2tc(common, *op);
}

static void to_common_integer_type(expr2tc &rel)
{
  std::vector<expr2tc *> ops;
  rel->Foreach_operand([&](expr2tc &op) { ops.push_back(&op); });
  if (ops.size() != 2 || !is_bv_type(*ops[0]) || !is_bv_type(*ops[1]))
    return;
  type2tc common = (*ops[0])->type;
  const type2tc other = (*ops[1])->type;
  if (
    other->get_width() > common->get_width() ||
    (other->get_width() == common->get_width() && is_unsignedbv_type(other)))
    common = other;
  for (expr2tc *op : ops)
    if ((*op)->type != common)
      *op = typecast2tc(common, *op);
}

static void
substitute_result(expr2tc &e, const expr2tc &ret_val, bool integer_result)
{
  if (is_symbol2t(e) && to_symbol2t(e).thename == "\\result")
  {
    e = ret_val;
    return;
  }
  if (is_constant_int2t(e))
  {
    if (integer_result)
      return;
    e = from_integer(to_constant_int2t(e).value, ret_val->type);
    return;
  }
  e->Foreach_operand(
    [&](expr2tc &op) { substitute_result(op, ret_val, integer_result); });
  if (is_comparison_expr(e))
  {
    to_common_float_type(e);
    if (integer_result)
      to_common_integer_type(e);
  }
}

const std::vector<size_t> &goto_symext::witness_segments_at(
  waypoint::Type type,
  const irep_idt &line) const
{
  static const std::vector<size_t> empty;
  auto found = cur_state->witness_locations.find({type, line});
  return found == cur_state->witness_locations.end() ? empty : found->second;
}

expr2tc
goto_symext::witness_segment_guard(size_t segment, const expr2tc &position)
{
  expr2tc active =
    equality2tc(position, constant_int2tc(position->type, BigInt(segment)));
  do_simplify(active);
  return active;
}

expr2tc goto_symext::witness_position() const
{
  expr2tc position = witness_cursor;
  cur_state->rename(position);
  return position;
}

void goto_symext::advance_witness_position(
  size_t segment,
  const expr2tc &position)
{
  size_t next = segment + 1;
  if (
    next == cur_state->witness_segs.size() &&
    cur_state->witness_segs[segment].back().action == waypoint::cycle)
  {
    // After the last cycle segment, replay the ordered cycle from its start.
    for (next = 0; next < cur_state->witness_segs.size(); ++next)
      if (
        !cur_state->witness_segs[next].empty() &&
        cur_state->witness_segs[next].back().action == waypoint::cycle)
        break;
  }
  guard2tc guard;
  guard.add(witness_segment_guard(segment, position));
  symex_assign(
    code_assign2tc(
      witness_cursor, constant_int2tc(position->type, BigInt(next))),
    true,
    guard);
}

void goto_symext::symex_witness_function_return(
  expr2tc ret_val,
  const irep_idt &call_line)
{
  // Promote narrow results before comparing them (C11 6.3.1.1, 6.3.1.8).
  const bool promoted = is_bool_type(ret_val->type) ||
                        (is_bv_type(ret_val->type) &&
                         ret_val->type->get_width() < config.ansi_c.int_width);
  if (promoted)
    ret_val = typecast2tc(int_type2(), ret_val);
  const expr2tc position = witness_position();
  for (size_t seg : witness_segments_at(waypoint::function_return, call_line))
  {
    const expr2tc active = witness_segment_guard(seg, position);
    if (is_false(active))
      continue;
    for (const waypoint &wp : cur_state->witness_segs[seg])
    {
      if (
        wp.type != waypoint::function_return || !wp.parsed_cond.valid ||
        wp.line_id != call_line)
        continue;

      cur_state->rename(ret_val);
      expr2tc constraint = wp.parsed_cond.expr;
      substitute_result(constraint, ret_val, is_bv_type(ret_val->type));
      if (wp.action == waypoint::avoid)
        constraint = not2tc(constraint);
      assume(implies2tc(active, constraint));
      if (wp.action != waypoint::avoid)
        advance_witness_position(seg, position);
      break;
    }
  }
}

bool goto_symext::symex_witness_function_enter(const irep_idt &call_line)
{
  const expr2tc position = witness_position();
  for (size_t seg : witness_segments_at(waypoint::function_enter, call_line))
  {
    const expr2tc active = witness_segment_guard(seg, position);
    if (is_false(active))
      continue;
    for (const waypoint &wp : cur_state->witness_segs[seg])
    {
      if (wp.type != waypoint::function_enter)
        continue;
      if (!wp.line_id.empty() && wp.line_id == call_line)
      {
        if (wp.action == waypoint::avoid)
        {
          assume(not2tc(active));
          if (cur_state->guard.is_false())
            return true;
        }
        else
          advance_witness_position(seg, position);
        break;
      }
      if (wp.action != waypoint::avoid)
        break;
    }
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
  if (
    !validate_witness || !forward || is_constant(old_guard) ||
    cur_state->guard.is_false())
    return;

  const auto &loc = cur_state->source.pc->location;
  auto force_goto = [&](bool taken, const expr2tc &active) {
    if (is_true(active))
    {
      new_guard_true = taken;
      new_guard_false = !taken;
    }
    expr2tc dir_cond = taken ? new_guard : not2tc(new_guard);
    assume(implies2tc(active, dir_cond));
  };

  const expr2tc position = witness_position();
  for (size_t seg : witness_segments_at(waypoint::branching, loc.get_line()))
  {
    const expr2tc active = witness_segment_guard(seg, position);
    if (is_false(active))
      continue;
    for (const auto &wp : cur_state->witness_segs[seg])
    {
      if (wp.type != waypoint::branching)
        continue;
      const bool is_avoid = wp.action == waypoint::avoid;
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
          force_goto(instruction.flipped_guard, active);
        }
        else
        {
          force_goto(case_matches != instruction.flipped_guard, active);
          if (
            case_matches || (wp.value == "default" &&
                             is_last_switch_case_test(cur_state->source.pc)))
            advance_witness_position(seg, position);
        }
        break;
      }

      const bool direction_true = (wp.value == "true") ^ is_avoid;
      force_goto(
        instruction.flipped_guard ? direction_true : !direction_true, active);
      if (!is_avoid)
        advance_witness_position(seg, position);
      break;
    }
  }
}

expr2tc goto_symext::witness_target_guard(
  const locationt &loc,
  const expr2tc &position,
  bool return_statement)
{
  expr2tc matched = gen_false_expr();
  for (size_t seg : witness_segments_at(waypoint::target, loc.get_line()))
    for (const auto &wp : cur_state->witness_segs[seg])
      if (
        wp.type == waypoint::target && loc.get_line() == wp.line_id &&
        (wp.column_id.empty() ||
         (return_statement
            ? loc.get_column()
            : full_expression_start(loc).get_column()) == wp.column_id))
        matched = or2tc(matched, witness_segment_guard(seg, position));
  return matched;
}

void goto_symext::symex_witness_main_return()
{
  const auto &instruction = *cur_state->source.pc;
  if (
    (instruction.function != "c:@F@main" &&
     instruction.function != "c:@F@main#") ||
    cur_state->top().calling_location.pc->function != "__ESBMC_main")
    return;

  const expr2tc position = witness_position();
  guard2tc matched;
  matched.add(witness_target_guard(instruction.location, position, true));
  // Leak claims follow main's return; keep its target match in SSA meanwhile.
  symex_assign(
    code_assign2tc(
      witness_cursor,
      constant_int2tc(
        position->type, BigInt(cur_state->witness_segs.size() + 1))),
    true,
    matched);
}

expr2tc goto_symext::witness_assert_target_guard(
  const std::string &msg,
  bool application_assertion,
  const expr2tc &position)
{
  // A witness writer reports a claim in a bundled model, e.g. the leak check
  // in exit(), at its innermost call site in the program.
  const locationt *loc = &cur_state->source.pc->location;
  for (auto it = cur_state->call_stack.rbegin();
       it != cur_state->call_stack.rend() &&
       file_operations::is_bundled_source(loc->get_file().as_string());
       ++it)
    if (
      it->calling_location.is_set &&
      it->calling_location.pc != it->calling_location.prog->instructions.end())
      loc = &it->calling_location.pc->location;

  // A final segment matches a violation on any visit to the target, not only
  // the first (sv-witnesses Witness-Format.md, "Matching a Final Segment").
  expr2tc matched = gen_false_expr();
  if (cur_state->source.pc->location.property() != "error label")
    matched = witness_target_guard(*loc, position);
  if (
    memory_leak_check && !no_reachable_memleak && !application_assertion &&
    cur_state->source.pc->function == "__ESBMC_main" &&
    has_prefix(msg, "dereference failure: forgotten memory:"))
    matched = or2tc(
      matched,
      witness_segment_guard(cur_state->witness_segs.size() + 1, position));

  return matched;
}

void goto_symext::symex_witness_assert(
  expr2tc &new_expr,
  const std::string &msg,
  bool user_assertion)
{
  if (
    !validate_witness || has_prefix(msg, "unwinding assertion loop") ||
    msg == "recursion unwinding assertion" ||
    has_prefix(msg, "uncontrolled recursion") ||
    has_prefix(msg, "termination per-loop marker") ||
    has_prefix(msg, "termination abort-call marker"))
    return;

  const bool application_assertion =
    user_assertion && !file_operations::is_bundled_source(
                        cur_state->source.pc->location.get_file().as_string());
  const expr2tc position = witness_position();
  const expr2tc matched =
    witness_assert_target_guard(msg, application_assertion, position);

  // A memory violation before the target cannot be part of a valid prefix.
  const bool memory_check =
    (msg.find("dereference failure") != std::string::npos &&
     msg.find("forgotten memory") == std::string::npos) ||
    has_prefix(msg, "array bounds violated") ||
    msg == "Operand of free must have zero pointer offset";
  if (!application_assertion && memory_check)
    assume(or2tc(matched, new_expr));
  new_expr = implies2tc(matched, new_expr);
}
