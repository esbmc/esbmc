#include <goto-programs/goto_array_loop_summary.h>
#include <irep2/irep2_utils.h>
#include <util/symtab/symbol_generator.h>

namespace
{
using targett = goto_programt::targett;

/// A loop `head: IF !(i < hi) GOTO exit; body; i = i + 1; GOTO head`.
struct loopt
{
  targett head, incr, back;
  expr2tc i, hi;
  // The single write `a[idx] = value`, if any, and the guards of the GOTOs
  // before it, each of which skips the rest of the iteration.
  targett write;
  expr2tc array, index;
  std::vector<expr2tc> skips;
  // Whether the body asserts anything.
  bool checks = false;
  // Whether the write stores a fresh nondet value each iteration.
  bool nondet = false;
};

const irep_idt verifier_assert = "c:@F@__VERIFIER_assert";

/// Whether calling f changes no state that outlives the call: it writes no
/// variable of static lifetime, and every function it calls is the same, or
/// is abort(), which ends the path. The summary runs such a call for one
/// index instead of every index, which loses nothing only then.
bool keeps_state(
  const irep_idt &f,
  const goto_functionst &goto_functions,
  const contextt &context,
  unsigned depth = 0)
{
  if (f == "c:@F@abort")
    return true;
  auto it = goto_functions.function_map.find(f);
  if (
    depth > 4 || it == goto_functions.function_map.end() ||
    !it->second.body_available)
    return false;
  for (const auto &ins : it->second.body.instructions)
  {
    if (ins.is_assign())
    {
      const expr2tc &target = to_code_assign2t(ins.code).target;
      const symbolt *s = is_symbol2t(target)
                           ? context.find_symbol(to_symbol2t(target).thename)
                           : nullptr;
      if (!s || s->static_lifetime)
        return false;
    }
    else if (ins.is_function_call())
    {
      const expr2tc &callee = to_code_function_call2t(ins.code).function;
      if (
        !is_symbol2t(callee) ||
        !keeps_state(
          to_symbol2t(callee).thename, goto_functions, context, depth + 1))
        return false;
    }
    else if (ins.is_other() || ins.type == THROW)
      return false;
  }
  return true;
}

bool is_verifier_assert(const goto_programt::instructiont &ins)
{
  if (!ins.is_function_call())
    return false;
  const code_function_call2t &call = to_code_function_call2t(ins.code);
  return is_nil_expr(call.ret) && is_symbol2t(call.function) &&
         to_symbol2t(call.function).thename == verifier_assert &&
         call.operands.size() == 1;
}

bool is_counter_index(const expr2tc &index, const expr2tc &i)
{
  if (index == i)
    return true;
  if (!is_typecast2t(index) || to_typecast2t(index).from != i)
    return false;
  // Widening only, so distinct counters are distinct indices.
  const type2tc &to = index->type, &from = i->type;
  return is_bv_type(to) && to->get_width() >= from->get_width() &&
         (is_signedbv_type(to) == is_signedbv_type(from) ||
          (is_unsignedbv_type(from) && to->get_width() > from->get_width()));
}

/// Whether evaluating e in iteration i depends on no other iteration: no
/// pointers or side effects, and the written array read only at a[i].
bool reads_only_own_element(const expr2tc &e, const loopt &loop)
{
  if (is_nil_expr(e))
    return true;
  if (
    is_dereference2t(e) || is_address_of2t(e) || is_sideeffect2t(e) ||
    is_array_lambda2t(e))
    return false;
  if (!is_nil_expr(loop.array))
  {
    if (is_index2t(e) && to_index2t(e).source_value == loop.array)
      return is_counter_index(to_index2t(e).index, loop.i);
    if (e == loop.array)
      return false;
  }
  bool ok = true;
  e->foreach_operand(
    [&](const expr2tc &op) { ok = ok && reads_only_own_element(op, loop); });
  return ok;
}

bool mentions(const expr2tc &e, const expr2tc &sym)
{
  if (is_nil_expr(e))
    return false;
  if (e == sym)
    return true;
  bool found = false;
  e->foreach_operand(
    [&](const expr2tc &op) { found = found || mentions(op, sym); });
  return found;
}

bool match(
  goto_programt &body,
  targett back,
  bool assert_keeps_state,
  loopt &loop)
{
  if (!is_true(back->guard) || back->targets.size() != 1)
    return false;
  loop.back = back;
  loop.head = back->targets.front();
  const targett exit = std::next(back);
  if (
    loop.head == back || !loop.head->is_goto() ||
    loop.head->targets.size() != 1 || loop.head->targets.front() != exit ||
    !is_not2t(loop.head->guard) ||
    !is_lessthan2t(to_not2t(loop.head->guard).value))
    return false;

  const lessthan2t &test = to_lessthan2t(to_not2t(loop.head->guard).value);
  loop.i = test.side_1;
  loop.hi = test.side_2;
  if (!is_symbol2t(loop.i) || !is_bv_type(loop.i))
    return false;

  loop.incr = std::prev(back);
  if (loop.incr == loop.head || !loop.incr->is_assign())
    return false;
  const code_assign2t &step = to_code_assign2t(loop.incr->code);
  if (
    step.target != loop.i || !is_add2t(step.source) ||
    to_add2t(step.source).side_1 != loop.i ||
    !is_constant_int2t(to_add2t(step.source).side_2) ||
    to_constant_int2t(to_add2t(step.source).side_2).value != 1)
    return false;

  loop.write = body.instructions.end();
  std::vector<expr2tc> reads;
  for (targett it = std::next(loop.head); it != loop.incr; ++it)
  {
    if (it->is_location() || it->is_skip())
      continue;
    if (it->is_assert())
    {
      reads.push_back(it->guard);
      loop.checks = true;
    }
    else if (assert_keeps_state && is_verifier_assert(*it))
    {
      reads.push_back(to_code_function_call2t(it->code).operands.front());
      loop.checks = true;
    }
    else if (it->is_goto())
    {
      // Only a skip of the rest of this iteration, before the write.
      if (
        it->targets.size() != 1 || it->targets.front() != loop.incr ||
        loop.write != body.instructions.end())
        return false;
      reads.push_back(it->guard);
      loop.skips.push_back(it->guard);
    }
    else if (it->is_assign() && loop.write == body.instructions.end())
    {
      const code_assign2t &assign = to_code_assign2t(it->code);
      if (
        !is_index2t(assign.target) ||
        !is_symbol2t(to_index2t(assign.target).source_value) ||
        !is_array_type(to_index2t(assign.target).source_value) ||
        !is_counter_index(to_index2t(assign.target).index, loop.i))
        return false;
      loop.write = it;
      loop.array = to_index2t(assign.target).source_value;
      loop.index = to_index2t(assign.target).index;
      if (
        is_sideeffect2t(assign.source) &&
        to_sideeffect2t(assign.source).kind == sideeffect2t::allockind::nondet)
        loop.nondet = true;
      else
        reads.push_back(assign.source);
    }
    else
      return false;
  }

  for (const expr2tc &e : reads)
    if (!reads_only_own_element(e, loop))
      return false;
  if (
    !reads_only_own_element(loop.hi, loop) || mentions(loop.hi, loop.i) ||
    (!is_nil_expr(loop.array) && mentions(loop.hi, loop.array)))
    return false;

  // Nothing outside the loop may jump into it past the head.
  for (const auto &ins : body.instructions)
    for (const targett &t : ins.targets)
    {
      bool inside = false;
      for (targett it = std::next(loop.head); it != exit && !inside; ++it)
        inside = it == t;
      if (!inside)
        continue;
      bool from_inside = false;
      for (targett it = std::next(loop.head); it != exit && !from_inside; ++it)
        from_inside = &*it == &ins;
      if (!from_inside)
        return false;
    }
  return true;
}

void replace(expr2tc &e, const expr2tc &from, const expr2tc &to)
{
  if (is_nil_expr(e))
    return;
  if (e == from)
  {
    e = to;
    return;
  }
  e->Foreach_operand([&](expr2tc &op) { replace(op, from, to); });
}

/// Replace every read or write of the loop's own element a[i] in e by
/// element.
void replace_own_element(expr2tc &e, const loopt &loop, const expr2tc &element)
{
  if (is_nil_expr(e))
    return;
  if (
    is_index2t(e) && to_index2t(e).source_value == loop.array &&
    is_counter_index(to_index2t(e).index, loop.i))
  {
    e = element;
    return;
  }
  e->Foreach_operand(
    [&](expr2tc &op) { replace_own_element(op, loop, element); });
}

/// The index the write uses when the counter is x.
expr2tc index_at(const loopt &loop, const expr2tc &x)
{
  expr2tc index = loop.index;
  replace(index, loop.i, x);
  return index;
}

/// Numbers the lambdas' bound variables, so a nested lambda never captures an
/// outer one's.
unsigned lambdas = 0;

void summarise(
  goto_programt &body,
  const loopt &loop,
  bool keep_iteration,
  contextt &context,
  symbol_generator &names)
{
  auto fresh = [&](const type2tc &type, const char *name) {
    symbolt &s = names.new_symbol(context, type, name);
    s.static_lifetime = true;
    return symbol2tc(type, s.id);
  };
  const bool writes = !is_nil_expr(loop.array);
  const goto_programt::instructiont &proto = *loop.head;
  goto_programt code;
  auto add = [&](goto_program_instruction_typet type) {
    targett t = code.add_instruction(type);
    t->location = proto.location;
    t->function = proto.function;
    return t;
  };

  // One iteration at a nondet index in [i, hi): the checks in the body hold
  // for every iteration exactly when they hold for this one.
  std::vector<targett> skips;
  if (keep_iteration)
  {
    const expr2tc k = fresh(loop.i->type, "loop_index$");
    // The body touches only a[i], so its iteration works on that element
    // alone: copying the whole array would hand the solver the summary as an
    // array.
    const expr2tc element =
      writes ? fresh(to_array_type(loop.array->type).subtype, "loop_element$")
             : expr2tc();
    auto at_k = [&](expr2tc e) {
      if (writes)
        replace_own_element(e, loop, element);
      replace(e, loop.i, k);
      return e;
    };

    targett t = add(ASSIGN);
    t->code = code_assign2tc(
      k,
      sideeffect2tc(
        k->type,
        expr2tc(),
        expr2tc(),
        std::vector<expr2tc>(),
        type2tc(),
        sideeffect2t::allockind::nondet));
    const targett enter = add(GOTO);
    enter->guard =
      not2tc(and2tc(lessthanequal2tc(loop.i, k), lessthan2tc(k, loop.hi)));
    skips.push_back(enter);
    bool loaded = false;
    for (targett it = std::next(loop.head); it != loop.incr; ++it)
    {
      // Load the element where the body first touches it, so the load's bounds
      // check sits on the same path as the original access.
      if (
        writes && !loaded &&
        (mentions(it->guard, loop.array) || mentions(it->code, loop.array)))
      {
        t = add(ASSIGN);
        t->code = code_assign2tc(
          element, index2tc(element->type, loop.array, index_at(loop, k)));
        loaded = true;
      }
      targett c = code.add_instruction(*it);
      c->targets.clear();
      c->guard = at_k(c->guard);
      if (!is_nil_expr(c->code))
        c->code = at_k(c->code);
      if (it->is_goto())
        skips.push_back(c);
    }
  }

  const targett summary = add(writes ? ASSIGN : SKIP);
  for (targett skip : skips)
    skip->targets.push_back(summary);
  targett lambda = summary;

  if (writes)
  {
    // Element j is written iff some iteration in [i, hi) indexes it and does
    // not skip; the counter of that iteration is j itself, narrowed back.
    const type2tc &index_type = loop.index->type;
    const expr2tc j = symbol2tc(
      index_type, fmt::format("__ESBMC_array_loop_index${}", lambdas++));
    const expr2tc counter =
      index_type == loop.i->type ? j : typecast2tc(loop.i->type, j);
    std::vector<expr2tc> conds{
      lessthanequal2tc(index_at(loop, loop.i), j),
      lessthan2tc(j, index_at(loop, loop.hi))};
    for (expr2tc skip : loop.skips)
    {
      replace(skip, loop.i, counter);
      conds.push_back(not2tc(skip));
    }
    expr2tc value = to_code_assign2t(loop.write->code).source;
    replace(value, loop.i, counter);
    if (loop.nondet)
    {
      // Each iteration stores its own nondet value: element j of one fresh
      // nondet array.
      const expr2tc values = fresh(loop.array->type, "loop_nondet$");
      summary->code = code_assign2tc(
        values,
        sideeffect2tc(
          values->type,
          expr2tc(),
          expr2tc(),
          std::vector<expr2tc>(),
          type2tc(),
          sideeffect2t::allockind::nondet));
      lambda = add(ASSIGN);
      value = index2tc(value->type, values, j);
    }
    const expr2tc element = if2tc(
      value->type,
      conjunction(conds),
      value,
      index2tc(value->type, loop.array, j));
    lambda->code =
      code_assign2tc(loop.array, array_lambda2tc(loop.array->type, j, element));
  }
  add(ASSIGN)->code = code_assign2tc(loop.i, loop.hi);

  const targett exit = std::next(loop.back);
  body.instructions.erase(std::next(loop.head), exit);
  body.destructive_insert(exit, code);
}
} // namespace

void goto_array_loop_summary(
  goto_functionst &goto_functions,
  contextt &context,
  const optionst &options)
{
  // A summary writes the whole array at once, so another thread could no
  // longer observe it half written.
  if (spawns_threads(goto_functions))
    return;

  // Whether goto_check, which runs later, can add a check to a body this pass
  // accepts. Pointer checks cannot: such a body has no pointers.
  const bool checked = !options.get_bool_option("no-bounds-check") ||
                       !options.get_bool_option("no-div-by-zero-check") ||
                       options.get_bool_option("overflow-check") ||
                       options.get_bool_option("unsigned-overflow-check") ||
                       options.get_bool_option("ub-shift-check") ||
                       options.get_bool_option("nan-check") ||
                       options.get_bool_option("clz-zero-check");
  const bool assert_keeps_state =
    keeps_state(verifier_assert, goto_functions, context);
  symbol_generator names("c:@__ESBMC_array_loop$");
  unsigned summarised = 0;
  Forall_goto_functions (f, goto_functions)
  {
    if (!f->second.body_available)
      continue;
    goto_programt &body = f->second.body;
    for (targett it = body.instructions.begin(); it != body.instructions.end();
         ++it)
    {
      loopt loop;
      if (
        !it->is_backwards_goto() || !match(body, it, assert_keeps_state, loop))
        continue;
      targett exit = std::next(it);
      summarise(body, loop, loop.checks || checked, context, names);
      ++summarised;
      it = std::prev(exit);
    }
  }
  if (summarised)
  {
    goto_functions.update();
    log_status("Summarised {} array loop(s)", summarised);
  }
}
