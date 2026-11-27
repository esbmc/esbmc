/*******************************************************************
 Module: --partial-order SC memory model (#10990)

 Hand-built two-thread equations: thread 0 writes x and then reads it,
 thread 1 writes x. Each test fixes the clocks, the reads-from choices and
 the values, substitutes them into the constraints memory_model_sct
 prepends, and checks whether they all hold.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <goto-symex/equation/memory_model_sc.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>
#include <util/irep/std_types.h>

namespace
{
using valuation = std::map<expr2tc, expr2tc>;

expr2tc x(unsigned version)
{
  return symbol2tc(
    get_int32_type(),
    "c:@x",
    symbol2t::renaming_level::level2_global,
    0,
    version);
}

expr2tc clock(unsigned event)
{
  return symbol2tc(
    get_uint32_type(), "__ESBMC_po_clock$" + std::to_string(event));
}

expr2tc rf(unsigned write, unsigned read)
{
  return symbol2tc(
    get_bool_type(),
    "__ESBMC_po_rf$" + std::to_string(write) + "$" + std::to_string(read));
}

expr2tc num(int v)
{
  return constant_int2tc(get_int32_type(), BigInt(v));
}

expr2tc tick(unsigned v)
{
  return constant_int2tc(get_uint32_type(), BigInt(v));
}

void add_event(
  symex_target_equationt &eq,
  unsigned thread,
  symex_targett::step_typet type,
  const expr2tc &sym,
  const expr2tc &old = expr2tc())
{
  symex_targett::sourcet source;
  source.thread_nr = thread;
  eq.event(gen_true_expr(), type, sym, source, old);
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

void substitute(expr2tc &e, const valuation &v)
{
  auto it = v.find(e);
  if (it != v.end())
  {
    e = it->second;
    return;
  }
  e->Foreach_operand([&](expr2tc &op) { substitute(op, v); });
}

bool all_hold(const symex_target_equationt &eq, const valuation &v)
{
  for (const auto &step : eq.SSA_steps)
  {
    if (!step.is_assume())
      continue;
    expr2tc cond = step.cond;
    substitute(cond, v);
    simplify(cond);
    REQUIRE(is_constant_bool2t(cond));
    if (is_false(cond))
      return false;
  }
  return true;
}

expr2tc thread_id(unsigned t)
{
  return constant_int2tc(get_uint32_type(), BigInt(t));
}

/// Events 0: W x#1 (thread 0), 1: R x#3 (thread 0), 2: W x#2 (thread 1).
struct write_read_writet
{
  contextt ctx;
  namespacet ns{ctx};
  symex_target_equationt eq{ns};

  write_read_writet()
  {
    add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
    add_event(eq, 0, symex_targett::SHARED_READ, x(3));
    add_event(eq, 1, symex_targett::SHARED_WRITE, x(2));
    memory_model_sct().run(eq.SSA_steps);
  }

  bool holds(const valuation &v) const
  {
    return all_hold(eq, v);
  }
};

/// Thread 0 writes 0 at clock 1 and reads it at clock 2; thread 1 writes 1
/// at clock 3.
valuation reads_initial_write()
{
  return {
    {x(1), num(0)},
    {x(2), num(1)},
    {x(3), num(0)},
    {clock(0), tick(1)},
    {clock(1), tick(2)},
    {clock(2), tick(3)},
    {rf(0, 1), gen_true_expr()},
    {rf(2, 1), gen_false_expr()}};
}
} // namespace

TEST_CASE("constraints are hidden assumptions ahead of the events", "[po]")
{
  write_read_writet f;
  REQUIRE(f.eq.SSA_steps.size() > 3);
  auto it = f.eq.SSA_steps.begin();
  for (size_t i = 0; i < f.eq.SSA_steps.size() - 3; ++i, ++it)
  {
    CHECK(it->is_assume());
    CHECK(it->hidden);
  }
  for (; it != f.eq.SSA_steps.end(); ++it)
    CHECK(it->is_event());
}

TEST_CASE("each event keeps its clock for the trace", "[po]")
{
  write_read_writet f;
  auto it = std::prev(f.eq.SSA_steps.end(), 3);
  for (unsigned event = 0; event < 3; ++event, ++it)
    CHECK(it->cond == clock(event));
}

TEST_CASE("an SC execution satisfies the constraints", "[po]")
{
  write_read_writet f;
  CHECK(f.holds(reads_initial_write()));

  valuation v = reads_initial_write();
  v[clock(1)] = tick(3);
  v[clock(2)] = tick(2);
  v[rf(0, 1)] = gen_false_expr();
  v[rf(2, 1)] = gen_true_expr();
  v[x(3)] = num(1);
  CHECK(f.holds(v));
}

TEST_CASE("program order is kept within a thread", "[po]")
{
  write_read_writet f;
  // Thread 0 reads thread 1's write before its own write: every constraint
  // but program order holds.
  valuation v = reads_initial_write();
  v[clock(2)] = tick(0);
  v[clock(1)] = tick(1);
  v[clock(0)] = tick(2);
  v[rf(0, 1)] = gen_false_expr();
  v[rf(2, 1)] = gen_true_expr();
  v[x(3)] = num(1);
  CHECK_FALSE(f.holds(v));
}

TEST_CASE("writes to an object are totally ordered", "[po]")
{
  write_read_writet f;
  valuation v = reads_initial_write();
  v[clock(2)] = tick(1);
  CHECK_FALSE(f.holds(v));
}

TEST_CASE("a read has the value of the write it reads from", "[po]")
{
  write_read_writet f;
  valuation v = reads_initial_write();
  v[x(3)] = num(1);
  CHECK_FALSE(f.holds(v));
}

TEST_CASE("a read reads from some write", "[po]")
{
  write_read_writet f;
  valuation v = reads_initial_write();
  v[rf(0, 1)] = gen_false_expr();
  CHECK_FALSE(f.holds(v));
}

TEST_CASE("a read cannot read from a later write", "[po]")
{
  write_read_writet f;
  valuation v = reads_initial_write();
  v[rf(0, 1)] = gen_false_expr();
  v[rf(2, 1)] = gen_true_expr();
  v[x(3)] = num(1);
  CHECK_FALSE(f.holds(v));
}

TEST_CASE("a read cannot skip a write between", "[po]")
{
  write_read_writet f;
  valuation v = reads_initial_write();
  v[clock(2)] = tick(1);
  v[clock(0)] = tick(0);
  CHECK_FALSE(f.holds(v));
}

TEST_CASE("a thread never reads its own later write", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::SHARED_READ, x(3));
  add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
  add_event(eq, 1, symex_targett::SHARED_WRITE, x(2));
  memory_model_sct().run(eq.SSA_steps);

  bool own = false, other = false;
  for (const auto &step : eq.SSA_steps)
  {
    own |= mentions(step.cond, rf(1, 0));
    other |= mentions(step.cond, rf(2, 0));
  }
  CHECK_FALSE(own);
  CHECK(other);
}

/// Events 0: SPAWN 1 (thread 0), 1: W x#1 (thread 1), 2: JOIN 1 (thread 0).
struct spawn_joint
{
  contextt ctx;
  namespacet ns{ctx};
  symex_target_equationt eq{ns};

  spawn_joint()
  {
    add_event(eq, 0, symex_targett::SPAWN, thread_id(1));
    add_event(eq, 1, symex_targett::SHARED_WRITE, x(1));
    add_event(eq, 0, symex_targett::JOIN, thread_id(1));
    memory_model_sct().run(eq.SSA_steps);
  }
};

valuation clocks(unsigned c0, unsigned c1, unsigned c2)
{
  return {{clock(0), tick(c0)}, {clock(1), tick(c1)}, {clock(2), tick(c2)}};
}

TEST_CASE("a spawned thread runs between its spawn and join", "[po]")
{
  spawn_joint f;
  CHECK(all_hold(f.eq, clocks(1, 2, 3)));
  CHECK_FALSE(all_hold(f.eq, clocks(2, 1, 3)));
  CHECK_FALSE(all_hold(f.eq, clocks(1, 3, 2)));
}

/// Events 0: ATOMIC_BEGIN, 1: W x#1, 2: ATOMIC_END (thread 0), 3: W x#2
/// (thread 1).
struct atomic_sectiont
{
  contextt ctx;
  namespacet ns{ctx};
  symex_target_equationt eq{ns};

  atomic_sectiont()
  {
    add_event(eq, 0, symex_targett::ATOMIC_BEGIN, expr2tc());
    add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
    add_event(eq, 0, symex_targett::ATOMIC_END, expr2tc());
    add_event(eq, 1, symex_targett::SHARED_WRITE, x(2));
    memory_model_sct().run(eq.SSA_steps);
  }

  bool holds_with_other_at(unsigned c3) const
  {
    valuation v = clocks(2, 4, 6);
    v[clock(3)] = tick(c3);
    return all_hold(eq, v);
  }
};

TEST_CASE("no other thread's event falls inside an atomic section", "[po]")
{
  atomic_sectiont f;
  CHECK(f.holds_with_other_at(1));
  CHECK(f.holds_with_other_at(7));
  CHECK_FALSE(f.holds_with_other_at(3));
  CHECK_FALSE(f.holds_with_other_at(5));
}

TEST_CASE("a section left open lasts to its thread's end", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::ATOMIC_BEGIN, expr2tc());
  add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
  add_event(eq, 1, symex_targett::SHARED_WRITE, x(2));
  memory_model_sct().run(eq.SSA_steps);

  valuation v = {{clock(0), tick(1)}, {clock(1), tick(3)}};
  v[clock(2)] = tick(2);
  CHECK_FALSE(all_hold(eq, v));
  v[clock(2)] = tick(4);
  CHECK(all_hold(eq, v));
}

/// Events 0: W x#1 (thread 0), 1: W x#2 updating x#3 (thread 1), 2: W x#4
/// updating x#5 (thread 2), as two element writes to one array.
struct two_updatest
{
  contextt ctx;
  namespacet ns{ctx};
  symex_target_equationt eq{ns};

  two_updatest()
  {
    add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
    add_event(eq, 1, symex_targett::SHARED_WRITE, x(2), x(3));
    add_event(eq, 2, symex_targett::SHARED_WRITE, x(4), x(5));
    memory_model_sct().run(eq.SSA_steps);
  }
};

/// The initial write, then thread 1's update, then thread 2's, which reads
/// thread 1's.
valuation serial_updates()
{
  return {
    {x(1), num(0)},
    {x(2), num(1)},
    {x(3), num(0)},
    {x(4), num(2)},
    {x(5), num(1)},
    {clock(0), tick(1)},
    {clock(1), tick(2)},
    {clock(2), tick(3)},
    {rf(0, 1), gen_true_expr()},
    {rf(2, 1), gen_false_expr()},
    {rf(0, 2), gen_false_expr()},
    {rf(1, 2), gen_true_expr()}};
}

TEST_CASE("an update reads the latest write before it", "[po]")
{
  two_updatest f;
  CHECK(all_hold(f.eq, serial_updates()));

  valuation v = serial_updates();
  v[x(5)] = num(0);
  CHECK_FALSE(all_hold(f.eq, v));
}

TEST_CASE("no write falls between an update's read and its write", "[po]")
{
  two_updatest f;
  // Both updates read the initial write, so thread 1's is lost.
  valuation v = serial_updates();
  v[x(5)] = num(0);
  v[rf(0, 2)] = gen_true_expr();
  v[rf(1, 2)] = gen_false_expr();
  CHECK_FALSE(all_hold(f.eq, v));
}

TEST_CASE("an update never reads its own write", "[po]")
{
  two_updatest f;
  for (const auto &step : f.eq.SSA_steps)
  {
    CHECK_FALSE(mentions(step.cond, rf(1, 1)));
    CHECK_FALSE(mentions(step.cond, rf(2, 2)));
  }
}

/// Events 0: LOCK m, 1: UNLOCK m (thread 0), 2: LOCK m (thread 1), which it
/// never releases.
struct lockst
{
  contextt ctx;
  namespacet ns{ctx};
  symex_target_equationt eq{ns};

  lockst()
  {
    const expr2tc m = symbol2tc(get_int32_type(), "c:@m");
    add_event(eq, 0, symex_targett::LOCK, m);
    add_event(eq, 0, symex_targett::UNLOCK, m);
    add_event(eq, 1, symex_targett::LOCK, m);
    memory_model_sct().run(eq.SSA_steps);
  }
};

TEST_CASE("a lock waits for the mutex to be released", "[po]")
{
  lockst f;
  CHECK(all_hold(f.eq, clocks(1, 2, 3)));
  CHECK_FALSE(all_hold(f.eq, clocks(1, 3, 2)));
}

TEST_CASE("a lock never released keeps later locks out", "[po]")
{
  lockst f;
  CHECK_FALSE(all_hold(f.eq, clocks(2, 3, 1)));
}

TEST_CASE("two threads do not lock a mutex at once", "[po]")
{
  lockst f;
  CHECK_FALSE(all_hold(f.eq, clocks(1, 2, 1)));
}

TEST_CASE("an unlock off the path does not release the mutex", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  const expr2tc m = symbol2tc(get_int32_type(), "c:@m");
  add_event(eq, 0, symex_targett::LOCK, m);
  symex_targett::sourcet source;
  eq.event(gen_false_expr(), symex_targett::UNLOCK, m, source);
  add_event(eq, 1, symex_targett::LOCK, m);
  memory_model_sct().run(eq.SSA_steps);

  CHECK_FALSE(all_hold(eq, clocks(1, 2, 3)));
}

TEST_CASE("different mutexes do not exclude each other", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::LOCK, symbol2tc(get_int32_type(), "c:@m1"));
  add_event(eq, 1, symex_targett::LOCK, symbol2tc(get_int32_type(), "c:@m2"));
  memory_model_sct().run(eq.SSA_steps);

  valuation v = {{clock(0), tick(1)}, {clock(1), tick(1)}};
  CHECK(all_hold(eq, v));
  v[clock(1)] = tick(2);
  CHECK(all_hold(eq, v));
}

TEST_CASE("a mutex nobody unlocks is locked once", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  const expr2tc m = symbol2tc(get_int32_type(), "c:@m");
  add_event(eq, 0, symex_targett::LOCK, m);
  add_event(eq, 1, symex_targett::LOCK, m);
  memory_model_sct().run(eq.SSA_steps);

  CHECK_FALSE(all_hold(eq, {{clock(0), tick(1)}, {clock(1), tick(2)}}));
}

namespace
{
std::vector<const symex_target_equationt::SSA_stept *>
race_claims(const symex_target_equationt &eq)
{
  std::vector<const symex_target_equationt::SSA_stept *> out;
  for (const auto &step : eq.SSA_steps)
    if (step.is_assert())
      out.push_back(&step);
  return out;
}

bool races_under(const symex_target_equationt::SSA_stept &claim, valuation v)
{
  expr2tc adjacent = claim.cond_neg;
  substitute(adjacent, v);
  simplify(adjacent);
  REQUIRE(is_constant_bool2t(adjacent));
  return is_true(adjacent);
}

/// Events 0: W x (thread 0), 1: W y (thread 0), 2: R x (thread 1).
struct write_write_readt
{
  contextt ctx;
  namespacet ns{ctx};
  symex_target_equationt eq{ns};

  explicit write_write_readt(bool races)
  {
    add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
    add_event(
      eq, 0, symex_targett::SHARED_WRITE, symbol2tc(get_int32_type(), "c:@y"));
    add_event(eq, 1, symex_targett::SHARED_READ, x(2));
    memory_model_sct(races).run(eq.SSA_steps);
  }
};
} // namespace

TEST_CASE("races are claimed only when asked for", "[po]")
{
  CHECK(race_claims(write_write_readt(false).eq).empty());

  write_write_readt f(true);
  const auto claims = race_claims(f.eq);
  REQUIRE(claims.size() == 1);
  CHECK(claims[0] == &f.eq.SSA_steps.back());
  CHECK(claims[0]->comment == "R/W data race on c:@x");
}

TEST_CASE("conflicting events race when nothing lies between", "[po]")
{
  write_write_readt f(true);
  const auto &claim = *race_claims(f.eq).front();
  const valuation adjacent = {
    {clock(0), tick(1)}, {clock(1), tick(3)}, {clock(2), tick(2)}};
  CHECK(races_under(claim, adjacent));
  const valuation reversed = {
    {clock(0), tick(2)}, {clock(1), tick(3)}, {clock(2), tick(1)}};
  CHECK(races_under(claim, reversed));
  const valuation separated = {
    {clock(0), tick(1)}, {clock(1), tick(2)}, {clock(2), tick(3)}};
  CHECK_FALSE(races_under(claim, separated));
}

TEST_CASE("an event off the path does not separate a race", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
  symex_targett::sourcet source;
  eq.event(
    gen_false_expr(),
    symex_targett::SHARED_WRITE,
    symbol2tc(get_int32_type(), "c:@y"),
    source);
  add_event(eq, 1, symex_targett::SHARED_READ, x(2));
  memory_model_sct(true).run(eq.SSA_steps);

  const auto claims = race_claims(eq);
  REQUIRE(claims.size() == 1);
  CHECK(races_under(
    *claims[0],
    {{clock(0), tick(1)}, {clock(1), tick(2)}, {clock(2), tick(3)}}));
}

TEST_CASE("only conflicting pairs of different threads are claimed", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
  add_event(eq, 0, symex_targett::SHARED_READ, x(2));
  add_event(eq, 1, symex_targett::SHARED_READ, x(3));
  add_event(eq, 1, symex_targett::SHARED_WRITE, x(4));
  memory_model_sct(true).run(eq.SSA_steps);

  // W0/R2, W0/W3 and W3/R1; neither R/R nor a thread with itself.
  std::multiset<std::string> comments;
  for (const auto *claim : race_claims(eq))
    comments.insert(claim->comment.as_string());
  CHECK(
    comments == std::multiset<std::string>{
                  "R/W data race on c:@x",
                  "R/W data race on c:@x",
                  "W/W data race on c:@x"});
}

TEST_CASE("the race claims appended are counted", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
  add_event(eq, 1, symex_targett::SHARED_READ, x(2));
  memory_model_sct model(true);
  model.run(eq.SSA_steps);
  CHECK(model.claims_added() == race_claims(eq).size());
  CHECK(model.claims_added() == 1);
}

TEST_CASE("an access off the path does not race", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  symex_targett::sourcet source;
  eq.event(gen_false_expr(), symex_targett::SHARED_WRITE, x(1), source);
  add_event(eq, 1, symex_targett::SHARED_READ, x(2));
  memory_model_sct(true).run(eq.SSA_steps);

  const auto claims = race_claims(eq);
  REQUIRE(claims.size() == 1);
  CHECK_FALSE(
    races_under(*claims[0], {{clock(0), tick(1)}, {clock(1), tick(2)}}));
}

TEST_CASE("an atomic write races with a plain read", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::ATOMIC_BEGIN, expr2tc());
  add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
  add_event(eq, 0, symex_targett::ATOMIC_END, expr2tc());
  add_event(eq, 1, symex_targett::SHARED_READ, x(2));
  memory_model_sct(true).run(eq.SSA_steps);

  const auto claims = race_claims(eq);
  REQUIRE(claims.size() == 1);
  CHECK(races_under(
    *claims[0],
    {{clock(0), tick(1)},
     {clock(1), tick(2)},
     {clock(2), tick(3)},
     {clock(3), tick(4)}}));
}

TEST_CASE("accesses inside atomic sections do not race", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::ATOMIC_BEGIN, expr2tc());
  add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
  add_event(eq, 0, symex_targett::ATOMIC_END, expr2tc());
  add_event(eq, 1, symex_targett::ATOMIC_BEGIN, expr2tc());
  add_event(eq, 1, symex_targett::SHARED_READ, x(2));
  add_event(eq, 1, symex_targett::ATOMIC_END, expr2tc());
  memory_model_sct(true).run(eq.SSA_steps);
  CHECK(race_claims(eq).empty());
}

TEST_CASE("accesses to an _Atomic object do not race", "[po]")
{
  contextt ctx;
  symbolt atomic_x;
  atomic_x.id = "c:@x";
  typet type = signedbv_typet(32);
  type.set("#atomic", true);
  atomic_x.set_type(type);
  ctx.add(atomic_x);
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
  add_event(
    eq, 0, symex_targett::SHARED_WRITE, symbol2tc(get_int32_type(), "c:@y"));
  add_event(eq, 1, symex_targett::SHARED_READ, x(2));
  add_event(
    eq, 1, symex_targett::SHARED_READ, symbol2tc(get_int32_type(), "c:@y"));
  memory_model_sct(true, false, &ns).run(eq.SSA_steps);
  const auto claims = race_claims(eq);
  REQUIRE(claims.size() == 1);
  CHECK(claims[0]->comment == "R/W data race on c:@y");
}

namespace
{
expr2tc mutex(const char *name)
{
  return symbol2tc(get_int32_type(), name);
}

expr2tc stop()
{
  return symbol2tc(get_uint32_type(), "__ESBMC_po_stop");
}

/// The clocks \p ticks in event order, and the stop time \p at.
valuation timed(const std::vector<unsigned> &ticks, unsigned at)
{
  valuation v{{stop(), tick(at)}};
  for (unsigned e = 0; e < ticks.size(); ++e)
    v[clock(e)] = tick(ticks[e]);
  return v;
}

const symex_target_equationt::SSA_stept *
deadlock_claim(const symex_target_equationt &eq)
{
  for (const auto &step : eq.SSA_steps)
    if (step.is_assert() && step.comment == "Deadlocked state")
      return &step;
  return nullptr;
}

bool deadlocked(const symex_target_equationt &eq, const valuation &v)
{
  const auto *claim = deadlock_claim(eq);
  REQUIRE(claim);
  return races_under(*claim, v);
}

/// Events 0: LOCK a, 1: LOCK b, 2: UNLOCK b, 3: UNLOCK a (thread 0), and
/// 4: LOCK b, 5: LOCK a, 6: UNLOCK a, 7: UNLOCK b (thread 1).
struct lock_ordert
{
  contextt ctx;
  namespacet ns{ctx};
  symex_target_equationt eq{ns};

  explicit lock_ordert(bool deadlocks)
  {
    for (unsigned t = 0; t < 2; ++t)
    {
      const expr2tc first = mutex(t == 0 ? "c:@a" : "c:@b");
      const expr2tc second = mutex(t == 0 ? "c:@b" : "c:@a");
      add_event(eq, t, symex_targett::LOCK, first);
      add_event(eq, t, symex_targett::LOCK, second);
      add_event(eq, t, symex_targett::UNLOCK, second);
      add_event(eq, t, symex_targett::UNLOCK, first);
    }
    memory_model_sct(false, deadlocks).run(eq.SSA_steps);
  }
};

/// Each thread holds the mutex the other locks next.
const std::vector<unsigned> crossed{1, 3, 5, 6, 2, 4, 7, 8};
} // namespace

TEST_CASE("deadlocks are claimed only when asked for", "[po]")
{
  CHECK_FALSE(deadlock_claim(lock_ordert(false).eq));
  lock_ordert f(true);
  REQUIRE(deadlock_claim(f.eq));
  CHECK(deadlock_claim(f.eq) == &f.eq.SSA_steps.back());
}

TEST_CASE("a lock after the stop time need not wait for a release", "[po]")
{
  CHECK_FALSE(all_hold(lock_ordert(false).eq, timed(crossed, 3)));
  CHECK(all_hold(lock_ordert(true).eq, timed(crossed, 3)));
  CHECK_FALSE(all_hold(lock_ordert(true).eq, timed(crossed, 9)));
}

TEST_CASE("crossed locks deadlock", "[po]")
{
  lock_ordert f(true);
  CHECK(deadlocked(f.eq, timed(crossed, 3)));
  // Thread 1 has not taken b yet, so main can.
  CHECK_FALSE(deadlocked(f.eq, timed(crossed, 2)));
  CHECK_FALSE(deadlocked(f.eq, timed(crossed, 9)));
}

TEST_CASE("a thread waiting on main is no deadlock once main ends", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::LOCK, mutex("c:@m"));
  add_event(eq, 1, symex_targett::LOCK, mutex("c:@m"));
  memory_model_sct(false, true).run(eq.SSA_steps);

  CHECK_FALSE(deadlocked(eq, timed({1, 2}, 2)));
}

TEST_CASE("main joining a thread that waits on main deadlocks", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::LOCK, mutex("c:@m"));
  add_event(eq, 0, symex_targett::JOIN, thread_id(1));
  add_event(eq, 1, symex_targett::LOCK, mutex("c:@m"));
  memory_model_sct(false, true).run(eq.SSA_steps);

  CHECK(deadlocked(eq, timed({1, 3, 2}, 2)));
  // Before main takes the mutex, the thread can.
  CHECK_FALSE(deadlocked(eq, timed({1, 3, 2}, 1)));
}

TEST_CASE("an event off the path does not wait", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::LOCK, mutex("c:@m"));
  symex_targett::sourcet source;
  eq.event(gen_false_expr(), symex_targett::LOCK, mutex("c:@m"), source);
  add_event(eq, 0, symex_targett::UNLOCK, mutex("c:@m"));
  memory_model_sct(false, true).run(eq.SSA_steps);

  CHECK_FALSE(deadlocked(eq, timed({1, 2, 3}, 2)));
}

TEST_CASE("other claims hold until every thread ends", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  add_event(eq, 0, symex_targett::SHARED_WRITE, x(1));
  add_event(eq, 1, symex_targett::SHARED_READ, x(2));
  symex_targett::sourcet source;
  eq.assertion(
    gen_true_expr(),
    gen_false_expr(),
    gen_true_expr(),
    "user claim",
    {},
    source,
    0);
  memory_model_sct model(true, true);
  model.run(eq.SSA_steps);
  CHECK(model.claims_added() == 2);

  for (const auto &step : eq.SSA_steps)
  {
    if (!step.is_assert() || step.comment == "Deadlocked state")
      continue;
    for (unsigned at : {2u, 3u})
    {
      expr2tc cond = step.cond;
      substitute(cond, timed({1, 2}, at));
      simplify(cond);
      REQUIRE(is_constant_bool2t(cond));
      // Thread 1's read is still pending at 2, so the claim holds vacuously.
      CHECK(is_true(cond) == (at == 2));
    }
  }
}

/// Thread 0 spawns 1, claims false, joins 1, then reads x and claims false
/// again; thread 1 writes x and halts at an unwinding bound when h holds.
struct halted_joint
{
  contextt ctx;
  namespacet ns{ctx};
  symex_target_equationt eq{ns};
  const expr2tc h = symbol2tc(get_bool_type(), "h");

  void claim(const std::string &comment)
  {
    symex_targett::sourcet source;
    eq.assertion(
      gen_true_expr(),
      gen_false_expr(),
      gen_true_expr(),
      comment,
      {},
      source,
      0);
  }

  halted_joint()
  {
    add_event(eq, 0, symex_targett::SPAWN, thread_id(1));
    claim("before join");
    add_event(eq, 0, symex_targett::JOIN, thread_id(1));
    add_event(eq, 0, symex_targett::SHARED_READ, x(2));
    claim("after join");
    add_event(eq, 1, symex_targett::SHARED_WRITE, x(1));
    symex_targett::sourcet source;
    source.thread_nr = 1;
    eq.event(h, symex_targett::HALT, expr2tc(), source);
    memory_model_sct().run(eq.SSA_steps);
  }

  expr2tc under(expr2tc e, bool halted) const
  {
    substitute(e, {{h, halted ? gen_true_expr() : gen_false_expr()}});
    simplify(e);
    REQUIRE(is_constant_bool2t(e));
    return e;
  }

  const symex_target_equationt::SSA_stept &step(
    std::function<bool(const symex_target_equationt::SSA_stept &)> match) const
  {
    for (const auto &s : eq.SSA_steps)
      if (match(s))
        return s;
    FAIL("no such step");
    return eq.SSA_steps.front();
  }
};

TEST_CASE("a join on a halted thread does not return", "[po]")
{
  halted_joint f;
  const auto &read =
    f.step([](const auto &s) { return s.type == symex_targett::SHARED_READ; });
  const auto &join =
    f.step([](const auto &s) { return s.type == symex_targett::JOIN; });
  const auto &before =
    f.step([](const auto &s) { return s.comment == "before join"; });
  const auto &after =
    f.step([](const auto &s) { return s.comment == "after join"; });

  CHECK(is_false(f.under(read.guard, true)));
  CHECK(is_true(f.under(read.guard, false)));
  CHECK(is_false(f.under(join.guard, true)));
  CHECK(is_true(f.under(join.guard, false)));
  CHECK(is_true(f.under(after.cond, true)));
  CHECK(is_false(f.under(after.cond, false)));
  CHECK(is_false(f.under(before.cond, true)));
}

TEST_CASE("a thread joining a halted thread never ends either", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  const expr2tc h = symbol2tc(get_bool_type(), "h");
  add_event(eq, 0, symex_targett::JOIN, thread_id(1));
  add_event(eq, 0, symex_targett::SHARED_READ, x(2));
  add_event(eq, 1, symex_targett::JOIN, thread_id(2));
  symex_targett::sourcet source;
  source.thread_nr = 2;
  eq.event(h, symex_targett::HALT, expr2tc(), source);
  memory_model_sct().run(eq.SSA_steps);

  for (const auto &s : eq.SSA_steps)
    if (s.type == symex_targett::SHARED_READ)
    {
      expr2tc guard = s.guard;
      substitute(guard, {{h, gen_true_expr()}});
      simplify(guard);
      CHECK(is_false(guard));
    }
}

TEST_CASE("a thread halted in an atomic section keeps others out", "[po]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);
  const expr2tc h = symbol2tc(get_bool_type(), "h");
  add_event(eq, 0, symex_targett::SHARED_READ, x(2));
  add_event(eq, 1, symex_targett::ATOMIC_BEGIN, expr2tc());
  symex_targett::sourcet source;
  source.thread_nr = 1;
  eq.event(h, symex_targett::HALT, expr2tc(), source);
  eq.event(not2tc(h), symex_targett::ATOMIC_END, expr2tc(), source);
  memory_model_sct().run(eq.SSA_steps);

  // Main's read at 6, after the section's end, which is off the halted path.
  valuation v = {
    {clock(0), tick(6)},
    {clock(1), tick(2)},
    {clock(2), tick(3)},
    {clock(3), tick(5)}};
  v[h] = gen_true_expr();
  CHECK_FALSE(all_hold(eq, v));
  v[h] = gen_false_expr();
  v[clock(3)] = tick(3);
  v[clock(2)] = tick(2);
  v[clock(1)] = tick(1);
  v[clock(0)] = tick(4);
  CHECK(all_hold(eq, v));
}
