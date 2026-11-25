/*******************************************************************
 Module: --partial-order SC memory model (#10990)

 Hand-built two-thread equations: thread 0 writes x and then reads it,
 thread 1 writes x. Each test fixes the clocks, the reads-from choices and
 the values, substitutes them into the constraints memory_model_sct
 prepends, and checks whether they all hold.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <map>
#include <string>

#include <goto-symex/equation/memory_model_sc.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

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
