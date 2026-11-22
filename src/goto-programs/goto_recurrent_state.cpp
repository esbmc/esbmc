#include <goto-programs/goto_recurrent_state.h>
#include <goto-programs/goto_loops.h>
#include <irep2/irep2_utils.h>
#include <util/base/prefix.h>
#include <util/config/config.h>
#include <util/symtab/symbol_generator.h>
#include <algorithm>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace
{
using targett = goto_programt::targett;

using instructionst = std::unordered_set<const goto_programt::instructiont *>;
using symbol_sett = std::unordered_set<expr2tc, irep2_hash>;

/// The prefix of the symbols this pass adds, which an enclosing loop's check
/// need not compare.
const std::string own_prefix = "c:@__ESBMC_recurrent$";

bool has_fixed_size(const type2tc &type)
{
  if (!is_array_type(type))
    return true;
  const array_type2t &array = to_array_type(type);
  return !array.size_is_infinite && is_constant_int2t(array.array_size) &&
         has_fixed_size(array.subtype);
}

/// Adds to @p written the symbol that an assignment to @p lhs writes, or
/// returns false when no symbol names it.
bool add_written(const expr2tc &lhs, symbol_sett &written)
{
  if (is_nil_expr(lhs))
    return true;
  if (is_member2t(lhs))
    return add_written(to_member2t(lhs).source_value, written);
  // p[i] writes through p, not to it.
  if (is_index2t(lhs))
    return !is_pointer_type(to_index2t(lhs).source_value->type) &&
           add_written(to_index2t(lhs).source_value, written);
  if (is_typecast2t(lhs))
    return add_written(to_typecast2t(lhs).from, written);
  if (!is_symbol2t(lhs))
    return false;
  if (!has_prefix(to_symbol2t(lhs).thename.as_string(), own_prefix))
    written.insert(lhs);
  return has_fixed_size(lhs->type);
}

/// Adds to @p written every symbol that [@p from, @p to) writes, with the
/// functions it calls. Returns false when some write has no symbol to name
/// it, or a call goes through a function pointer or recursion.
bool add_writes(
  const goto_functionst &goto_functions,
  goto_programt::const_targett from,
  goto_programt::const_targett to,
  std::vector<irep_idt> &stack,
  symbol_sett &written)
{
  for (auto i = from; i != to; ++i)
  {
    if (
      i->is_assign() && !add_written(to_code_assign2t(i->code).target, written))
      return false;
    if (!i->is_function_call())
      continue;
    const code_function_call2t &call = to_code_function_call2t(i->code);
    if (!add_written(call.ret, written) || !is_symbol2t(call.function))
      return false;
    const irep_idt &callee = to_symbol2t(call.function).thename;
    if (std::find(stack.begin(), stack.end(), callee) != stack.end())
      return false;
    const auto it = goto_functions.function_map.find(callee);
    if (it == goto_functions.function_map.end() || !it->second.body_available)
      continue;
    const goto_programt &body = it->second.body;
    stack.push_back(callee);
    if (!add_writes(
          goto_functions,
          body.instructions.begin(),
          body.instructions.end(),
          stack,
          written))
      return false;
    stack.pop_back();
  }
  return true;
}

/// What the loop, and what it calls, may write, or nothing when some write
/// cannot be named. This is the state a recurrence must repeat: everything
/// else keeps its value while the loop runs.
std::optional<symbol_sett>
loop_state(const goto_functionst &goto_functions, const loopst &loop)
{
  // goto_loopst sees the pointer writes; its modified set is no help, as it
  // leaves out the symbols k-induction need not havoc.
  if (
    loop.writes_through_pointer() || loop.unnamed_pointer_write() ||
    loop.pointer_array_write_unresolvable())
    return {};
  symbol_sett written;
  std::vector<irep_idt> stack;
  if (!add_writes(
        goto_functions,
        loop.get_original_loop_head(),
        std::next(loop.get_original_loop_exit()),
        stack,
        written))
    return {};
  return written;
}

instructionst instructions_of(const loopst &loop)
{
  instructionst inside;
  for (auto p = loop.get_original_loop_head();
       p != std::next(loop.get_original_loop_exit());
       ++p)
    inside.insert(&*p);
  return inside;
}

/// Whether every way into the loop passes its head, where the reset goes. A
/// jump into the body (`goto test;` before a rotated loop) would leave the
/// saved state of an earlier entry, or none, to compare against.
bool entered_at_head(
  const goto_programt &body,
  const loopst &loop,
  const instructionst &inside)
{
  const auto head = loop.get_original_loop_head();
  for (const auto &i : body.instructions)
    if (!inside.count(&i))
      for (const auto &t : i.targets)
        if (t != head && inside.count(&*t))
          return false;
  return true;
}

expr2tc same_value(const expr2tc &a, const expr2tc &b)
{
  // == on floats is IEEE: -0 equals +0, which behave differently, so compare
  // the bits.
  if (is_floatbv_type(a) && !config.options.get_bool_option("int-encoding"))
  {
    const type2tc bits = get_uint_type(a->type->get_width());
    return equality2tc(bitcast2tc(bits, a), bitcast2tc(bits, b));
  }
  return equality2tc(a, b);
}

/// Before the loop's head:
///
///   reset: stored = false
///   check: ASSERT !(stored && saved == vars)
///          flag = nondet; saved = flag ? vars : saved
///          stored = stored || flag
///
/// Entries into the loop go to reset; jumps from inside it go to check.
targett instrument(
  goto_programt &body,
  const loopst &loop,
  const instructionst &inside,
  const symbol_sett &state,
  const std::function<expr2tc(const type2tc &, const std::string &)> &fresh)
{
  const targett head = loop.get_original_loop_head();

  goto_programt block;
  auto add = [&](goto_program_instruction_typet type, const expr2tc &code) {
    targett t = block.add_instruction(type);
    t->code = code;
    t->location = head->location;
    t->function = head->function;
    return t;
  };
  const expr2tc stored = fresh(get_bool_type(), "stored");
  const expr2tc flag = fresh(get_bool_type(), "save");

  const targett reset = add(ASSIGN, code_assign2tc(stored, gen_false_expr()));
  const targett check = add(ASSERT, expr2tc());
  add(ASSIGN, code_assign2tc(flag, gen_nondet(get_bool_type())));
  expr2tc same = stored;
  for (const expr2tc &v : state)
  {
    const expr2tc saved = fresh(v->type, "saved");
    same = and2tc(same, same_value(saved, v));
    add(ASSIGN, code_assign2tc(saved, if2tc(v->type, flag, v, saved)));
  }
  add(ASSIGN, code_assign2tc(stored, or2tc(stored, flag)));
  check->guard = not2tc(same);
  check->location.comment("recurrent state");

  body.instructions.splice(head, block.instructions);
  for (auto &i : body.instructions)
    for (auto &t : i.targets)
      if (t == head)
        t = inside.count(&i) ? check : reset;
  return check;
}

} // namespace

std::vector<recurrent_state_loopt>
goto_recurrent_state(goto_functionst &goto_functions, contextt &context)
{
  Forall_goto_functions (it, goto_functions)
    for (auto &i : it->second.body.instructions)
      if (i.is_assert())
        i.type = ASSUME;

  symbol_generator names{own_prefix};
  auto fresh = [&](const type2tc &type, const std::string &name) {
    symbolt &s = names.new_symbol(context, type, name);
    s.static_lifetime = true;
    return symbol2tc(type, s.id);
  };

  std::vector<recurrent_state_loopt> loops;
  Forall_goto_functions (it, goto_functions)
  {
    goto_functiont &function = it->second;
    if (
      !function.body_available || function.body.hide ||
      it->first == "__ESBMC_main")
      continue;

    goto_loopst function_loops(it->first, goto_functions, function);
    // ponytail: loops sharing a head (nested do-whiles) are skipped; giving
    // each its own reset needs the inner one's entries told from its back edge.
    std::unordered_map<const goto_programt::instructiont *, unsigned> heads;
    for (const loopst &loop : function_loops.get_loops())
      ++heads[&*loop.get_original_loop_head()];

    for (const loopst &loop : function_loops.get_loops())
    {
      const instructionst inside = instructions_of(loop);
      if (
        heads[&*loop.get_original_loop_head()] != 1 ||
        !entered_at_head(function.body, loop, inside))
        continue;
      if (const auto state = loop_state(goto_functions, loop))
        loops.push_back(
          {loop.get_original_loop_head()->location,
           instrument(function.body, loop, inside, *state, fresh)});
    }

    function.body.update();
  }
  goto_functions.update();
  return loops;
}
