#include <goto-programs/goto_array_loop_summary.h>
#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/lang/c_types.h>
#include <util/symtab/symbol_generator.h>

namespace
{
using targett = goto_programt::targett;

/// A scalar `v` the body updates once per iteration by `v = v + delta`, or
/// `v - delta` when `minus`, with delta the same in every iteration.
struct inductiont
{
  targett update;
  expr2tc v, delta;
  bool minus;
};

/// A loop `head: IF !(i < hi) GOTO exit; body; i = i + step; GOTO head`.
struct loopt
{
  targett head, incr, back;
  expr2tc i, hi;
  BigInt step;
  std::vector<inductiont> vars;
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

/// The largest value of a bitvector type.
BigInt type_max(const type2tc &t)
{
  return power(2, t->get_width() - (is_signedbv_type(t) ? 1 : 0)) - 1;
}

/// An upper bound on e: its value if constant, else the largest value of the
/// narrower unsigned type it widens, else of its own type.
BigInt upper_bound(const expr2tc &e)
{
  if (is_constant_int2t(e))
    return to_constant_int2t(e).value;
  if (is_typecast2t(e))
  {
    const type2tc &from = to_typecast2t(e).from->type;
    if (is_unsignedbv_type(from) && from->get_width() < e->type->get_width())
      return type_max(from);
  }
  return type_max(e->type);
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
  bool exact_wrap,
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

  // Updates of other counters may follow i's: no read of i comes after it.
  auto updates_itself = [](const goto_programt::instructiont &ins) {
    if (!ins.is_assign())
      return false;
    const code_assign2t &assign = to_code_assign2t(ins.code);
    return is_symbol2t(assign.target) &&
           ((is_add2t(assign.source) &&
             to_add2t(assign.source).side_1 == assign.target) ||
            (is_sub2t(assign.source) &&
             to_sub2t(assign.source).side_1 == assign.target));
  };
  loop.incr = std::prev(back);
  while (loop.incr != loop.head && updates_itself(*loop.incr) &&
         to_code_assign2t(loop.incr->code).target != loop.i)
    --loop.incr;
  if (loop.incr == loop.head || !loop.incr->is_assign())
    return false;
  const code_assign2t &step = to_code_assign2t(loop.incr->code);
  if (
    step.target != loop.i || !is_add2t(step.source) ||
    to_add2t(step.source).side_1 != loop.i ||
    !is_constant_int2t(to_add2t(step.source).side_2))
    return false;
  loop.step = to_constant_int2t(to_add2t(step.source).side_2).value;
  // The last iteration's increment must not wrap past hi. Counting is in
  // unsigned arithmetic, which needs exact_wrap.
  if (
    loop.step < 1 || (loop.step != 1 && !exact_wrap) ||
    upper_bound(loop.hi) + loop.step - 1 > type_max(loop.i->type))
    return false;

  loop.write = body.instructions.end();
  std::vector<expr2tc> reads;
  for (targett it = std::next(loop.head); it != back; ++it)
  {
    if (it == loop.incr || it->is_location() || it->is_skip())
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
    else if (it->is_assign() && is_symbol2t(to_code_assign2t(it->code).target))
    {
      const code_assign2t &assign = to_code_assign2t(it->code);
      const expr2tc &v = assign.target;
      if (
        v == loop.i || !is_bv_type(v) || !exact_wrap ||
        !(is_add2t(assign.source) || is_sub2t(assign.source)))
        return false;
      const expr2tc &lhs = is_add2t(assign.source)
                             ? to_add2t(assign.source).side_1
                             : to_sub2t(assign.source).side_1;
      const expr2tc &delta = is_add2t(assign.source)
                               ? to_add2t(assign.source).side_2
                               : to_sub2t(assign.source).side_2;
      if (lhs != v)
        return false;
      for (const inductiont &var : loop.vars)
        if (var.v == v)
          return false;
      loop.vars.push_back({it, v, delta, is_sub2t(assign.source)});
    }
    else if (it->is_assign() && loop.write == body.instructions.end())
    {
      const code_assign2t &assign = to_code_assign2t(it->code);
      if (
        !is_index2t(assign.target) ||
        !is_symbol2t(to_index2t(assign.target).source_value) ||
        !is_array_type(to_index2t(assign.target).source_value) ||
        !is_number_type(assign.target) ||
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
  // Each update happens every iteration, by an amount no iteration changes.
  if (!loop.vars.empty() && !loop.skips.empty())
    return false;
  for (const inductiont &var : loop.vars)
  {
    if (
      !reads_only_own_element(var.delta, loop) || mentions(var.delta, loop.i) ||
      mentions(loop.hi, var.v) ||
      (!is_nil_expr(loop.array) && mentions(var.delta, loop.array)))
      return false;
    for (const inductiont &other : loop.vars)
      if (mentions(var.delta, other.v))
        return false;
  }

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

  // All counting is in unsigned arithmetic, which wraps as the loop's own
  // updates do; i still holds its value on entry until the end.
  auto as_unsigned = [](const expr2tc &x, unsigned width) {
    const type2tc u = get_uint_type(width);
    return x->type == u ? x : typecast2tc(u, x);
  };
  const unsigned width = loop.i->type->get_width();
  const expr2tc step = constant_int2tc(get_uint_type(width), loop.step);
  auto distance = [&](const expr2tc &x) {
    return sub2tc(
      step->type, as_unsigned(x, width), as_unsigned(loop.i, width));
  };
  // The number of iterations before the one whose counter is x.
  auto iterations = [&](const expr2tc &x) {
    return loop.step == 1 ? distance(x) : div2tc(step->type, distance(x), step);
  };
  // Whether some iteration's counter is x, given i <= x < hi.
  auto on_stride = [&](const expr2tc &x) -> expr2tc {
    if (loop.step == 1)
      return gen_true_expr();
    return equality2tc(
      modulus2tc(step->type, distance(x), step), gen_zero(step->type));
  };
  std::vector<expr2tc> entry;
  for (const inductiont &var : loop.vars)
  {
    entry.push_back(fresh(var.v->type, "loop_entry$"));
    add(ASSIGN)->code = code_assign2tc(entry.back(), var.v);
  }
  // v after n updates.
  auto value_after = [&](std::size_t idx, const expr2tc &n) {
    const inductiont &var = loop.vars[idx];
    const unsigned w = var.v->type->get_width();
    const expr2tc change =
      mul2tc(get_uint_type(w), as_unsigned(var.delta, w), as_unsigned(n, w));
    const expr2tc start = as_unsigned(entry[idx], w);
    const expr2tc sum = var.minus ? sub2tc(start->type, start, change)
                                  : add2tc(start->type, start, change);
    return sum->type == var.v->type ? sum : typecast2tc(var.v->type, sum);
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
    enter->guard = not2tc(conjunction(
      {lessthanequal2tc(loop.i, k), lessthan2tc(k, loop.hi), on_stride(k)}));
    skips.push_back(enter);
    for (std::size_t idx = 0; idx < loop.vars.size(); ++idx)
      add(ASSIGN)->code =
        code_assign2tc(loop.vars[idx].v, value_after(idx, iterations(k)));
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
      lessthan2tc(j, index_at(loop, loop.hi)),
      on_stride(counter)};
    for (expr2tc skip : loop.skips)
    {
      replace(skip, loop.i, counter);
      conds.push_back(not2tc(skip));
    }
    // Counter first: the closed forms read i as its value on entry.
    expr2tc value = to_code_assign2t(loop.write->code).source;
    replace(value, loop.i, counter);
    for (std::size_t idx = 0; idx < loop.vars.size(); ++idx)
    {
      // The updates the write sees: those of earlier iterations, and this
      // one's if it comes first.
      bool before = false;
      for (targett it = loop.head; it != loop.write; ++it)
        before = before || it == loop.vars[idx].update;
      expr2tc n = iterations(counter);
      if (before)
        n = add2tc(n->type, n, gen_one(n->type));
      replace(value, loop.vars[idx].v, value_after(idx, n));
    }
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
  // ceil((hi - i) / step) iterations; hi + step - 1 does not wrap.
  const expr2tc total = loop.step == 1
                          ? iterations(loop.hi)
                          : div2tc(
                              step->type,
                              add2tc(
                                step->type,
                                distance(loop.hi),
                                constant_int2tc(step->type, loop.step - 1)),
                              step);
  for (std::size_t idx = 0; idx < loop.vars.size(); ++idx)
    add(ASSIGN)->code =
      code_assign2tc(loop.vars[idx].v, value_after(idx, total));
  expr2tc last = loop.hi;
  if (loop.step != 1)
  {
    last = add2tc(
      step->type, as_unsigned(loop.i, width), mul2tc(step->type, step, total));
    if (last->type != loop.i->type)
      last = typecast2tc(loop.i->type, last);
  }
  add(ASSIGN)->code = code_assign2tc(loop.i, last);

  const targett exit = std::next(loop.back);
  body.instructions.erase(std::next(loop.head), exit);
  body.destructive_insert(exit, code);
}
} // namespace

void goto_array_loop_summary(
  goto_functionst &goto_functions,
  contextt &context,
  const optionst &options,
  bool assert_loops)
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
  // Closed forms count in unsigned arithmetic: an overflow check would flag
  // it, and --ir does not wrap it.
  const bool exact_wrap = !options.get_bool_option("overflow-check") &&
                          !options.get_bool_option("unsigned-overflow-check") &&
                          !options.get_bool_option("int-encoding");
  symbol_generator names("c:@__ESBMC_array_loop$");
  unsigned summarised = 0;
  Forall_goto_functions (f, goto_functions)
  {
    if (!f->second.body_available)
      continue;
    goto_programt &body = f->second.body;
    // __ESBMC_unroll bounds the loop after it only once this pass has run.
    bool unrolls = false;
    for (const auto &ins : body.instructions)
      unrolls =
        unrolls || (ins.is_function_call() &&
                    is_symbol2t(to_code_function_call2t(ins.code).function) &&
                    has_prefix(
                      to_symbol2t(to_code_function_call2t(ins.code).function)
                        .get_symbol_name(),
                      "c:@F@__ESBMC_unroll"));
    if (unrolls)
      continue;
    for (targett it = body.instructions.begin(); it != body.instructions.end();
         ++it)
    {
      loopt loop;
      if (
        !it->is_backwards_goto() || it->pragma_unroll_count != 0 ||
        !match(body, it, assert_keeps_state, exact_wrap, loop) ||
        (!assert_loops && is_nil_expr(loop.array)))
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
