// H-A1 — Refcount conservation, single-free, and self-alias UAF-safety,
// verified against the *actual* irep_container<T> implementation
// (src/irep2/irep2.h), not a hand-written model.
//
// These Catch2 property tests drive the genuine expr2tc / irep_container
// value-semantics machinery — copy ctor (fetch_add), move ctor (steal),
// copy/move assignment (snapshot-bump-before-release), release()
// (fetch_sub + delete-iff-prev==1) and detach() (COW clone) — and assert
// the representation invariant I1 (refcount == number of live handles) by
// reading the real atomic `irep2t::refcount` after each operation.
//
// This is the Tier-B strategy of docs/irep2-verification-plan.md §2: the
// template/immer/fmt/atomic layering of irep2 cannot be soundly ingested
// by ESBMC end-to-end, so the observable ownership contract is exercised
// directly against the real classes (cf. unit/irep2/irep2.bench.cpp).
// The value/refcount assertions catch wrong-count and wrong-value
// regressions directly; built under the Sanitizer build type
// (`./scripts/build.sh -b Sanitizer -s ASAN`) the same cases additionally
// witness the double-free / use-after-free freedom the assertions alone
// cannot observe.

#define CATCH_CONFIG_MAIN // Catch provides main() for this test executable
#include <catch2/catch.hpp>
#include <utility>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/config.h>

namespace
{
// Read a node's live refcount WITHOUT perturbing it: the const get()
// overload hands back the pointee without detaching, and `refcount` is a
// public mutable atomic on irep2t.
unsigned live_refcount(const expr2tc &c)
{
  return c.get()->refcount.load(std::memory_order_relaxed);
}

// Node identity without detaching (const get()).
const expr2t *node_of(const expr2tc &c)
{
  return c.get();
}
} // namespace

// I1: refcount equals the number of live containers pointing at the node,
// tracked exactly across copy, move, and scoped destruction.
TEST_CASE("irep2 refcount tracks live handles (I1)", "[core][irep2][refcount]")
{
  config.ansi_c.word_size = 32;

  expr2tc a = gen_ulong(11); // make_irep adopts at refcount 1
  REQUIRE(live_refcount(a) == 1);

  {
    expr2tc b = a; // copy ctor: fetch_add -> 2
    REQUIRE(live_refcount(a) == 2);
    REQUIRE(node_of(a) == node_of(b)); // shared, same node

    {
      expr2tc c = a; // -> 3
      REQUIRE(live_refcount(a) == 3);

      expr2tc d = std::move(c); // move ctor: steals, NO bump
      REQUIRE(live_refcount(a) == 3);
      REQUIRE(!c);                       // moved-from container is empty
      REQUIRE(node_of(d) == node_of(a)); // d owns the stolen node
    } // d destroyed -> 2
    REQUIRE(live_refcount(a) == 2);
  } // b destroyed -> 1
  REQUIRE(live_refcount(a) == 1);
}

// Copy-assignment conserves the count: it bumps the source and releases
// the previous pointee (which, at refcount 1, is freed exactly once —
// witnessed under a `Sanitizer` (ASan) build). The surviving node's
// value is intact.
TEST_CASE(
  "irep2 copy-assign releases old and bumps new",
  "[core][irep2][refcount]")
{
  config.ansi_c.word_size = 32;

  expr2tc a = gen_ulong(7);  // node A, refcount 1
  expr2tc b = gen_ulong(99); // node B, refcount 1
  const expr2t *A = node_of(a);
  REQUIRE(node_of(b) != A);

  b = a; // release B (freed), fetch_add on A -> 2

  REQUIRE(live_refcount(a) == 2);
  REQUIRE(node_of(b) == A); // b now aliases A
  REQUIRE(to_constant_int2t(b).value.to_uint64() == 7);
}

// H-A3: the self-aliasing assignment `x = <member-of-x's-own-node>` must
// be use-after-free-safe. irep_container::operator= snapshots and bumps
// the source pointer BEFORE release() runs, so even though releasing x's
// sole-owned add node destroys the very operand the RHS references, the
// operand survives the assignment. This reproduces the hazard the header
// documents at irep2.h:190-225 (`x = to_array_type(x).subtype`).
TEST_CASE(
  "irep2 self-aliasing assignment is UAF-safe (H-A3)",
  "[core][irep2][refcount]")
{
  config.ansi_c.word_size = 32;
  type2tc word = get_uint_type(config.ansi_c.word_size);

  expr2tc x = add2tc(word, gen_ulong(3), gen_ulong(4)); // sole owner, rc 1
  REQUIRE(live_refcount(x) == 1);

  // RHS aliases a member (side_1) of the node x solely owns.
  const expr2tc &alias = to_add2t(x).side_1;
  x = alias; // must not read freed storage; x becomes the operand (== 3)

  REQUIRE(is_constant_int2t(x));
  REQUIRE(to_constant_int2t(x).value.to_uint64() == 3);
  REQUIRE(live_refcount(x) == 1);
}

// COW detach on a shared node clones into a fresh refcount-1 object,
// leaving the other handle sole owner of the original — value preserved,
// counts conserved on both sides.
TEST_CASE(
  "irep2 COW detach conserves refcount and isolates mutation",
  "[core][irep2][refcount]")
{
  config.ansi_c.word_size = 32;

  expr2tc a = gen_ulong(11);
  expr2tc b = a; // shared, refcount 2
  REQUIRE(live_refcount(a) == 2);
  REQUIRE(node_of(a) == node_of(b));

  (void)a.get(); // non-const get() -> detach(): a clones to a fresh node

  REQUIRE(node_of(a) != node_of(b)); // distinct nodes now
  REQUIRE(live_refcount(a) == 1);    // a sole-owns its clone
  REQUIRE(live_refcount(b) == 1);    // b sole-owns the original
  REQUIRE(*a == *b);                 // structural value unchanged
}
