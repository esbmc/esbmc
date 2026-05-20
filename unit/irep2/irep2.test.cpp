#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <limits>
#include <irep2/irep2.h>
#include <irep2/irep2_dispatch.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/crypto_hash.h>
#include <util/expr_util.h>
#include <util/mp_arith.h>

namespace
{
std::array<unsigned int, 5> to_array(const crypto_hash &h)
{
  std::array<unsigned int, 5> result;
  std::copy(h.hash, h.hash + 5, result.begin());
  return result;
}
type2tc testing_struct2t()
{
  std::vector<type2tc> struct_members{
    get_uint_type(config.ansi_c.word_size),
    get_uint_type(config.ansi_c.word_size)};
  std::vector<irep_idt> struct_memb_names{"a", "b"};
  return struct_type2tc(
    struct_members, struct_memb_names, struct_memb_names, "test_struct");
}

type2tc testing_overlap2t()
{
  return get_uint_type(1);
}

expr2tc gen_testing_overlap(unsigned v)
{
  return constant_int2tc(testing_overlap2t(), BigInt(v));
}

expr2tc gen_testing_struct(unsigned int a, unsigned int b)
{
  std::vector<expr2tc> members{gen_ulong(a), gen_ulong(b)};
  return constant_struct2tc(testing_struct2t(), members);
}

// Build a constant_array2t whose member vector has `count` ulong elements,
// each initialised to its index. The array type uses an infinite-sized array
// so its identity does not depend on the member count.
expr2tc gen_testing_array(unsigned int count)
{
  type2tc subtype = get_uint_type(config.ansi_c.word_size);
  type2tc array_ty = array_type2tc(subtype, expr2tc(), true);
  std::vector<expr2tc> members;
  members.reserve(count);
  for (unsigned int i = 0; i < count; ++i)
    members.push_back(gen_ulong(i));
  return constant_array2tc(array_ty, members);
}

void test_constructed_equally(const expr2tc e1, const expr2tc e2)
{
  crypto_hash c_hash;
  crypto_hash c_hash2;
  // "The == operator should return true"
  REQUIRE(e1 == e2);
  // "Their crc should be the same"
  REQUIRE(e1->crc() == e2->crc());
  // Their hash should be the same"
  e1->hash(c_hash);
  e2->hash(c_hash2);

  c_hash.fin();
  c_hash2.fin();
  REQUIRE(to_array(c_hash) == to_array(c_hash2));
  REQUIRE(c_hash.to_size_t() == c_hash2.to_size_t());
}

void test_constructed_differently(const expr2tc e1, const expr2tc e2)
{
  crypto_hash c_hash;
  crypto_hash c_hash2;

  // "The == operator should return false"
  REQUIRE(e1 != e2);
  // "Their crc should not be the same"
  REQUIRE(e1->crc() != e2->crc());
  // "Their hash should not be the same"

  e1->hash(c_hash);
  e2->hash(c_hash2);

  c_hash.fin();
  c_hash2.fin();
  REQUIRE(to_array(c_hash) != to_array(c_hash2));
  REQUIRE(c_hash.to_size_t() != c_hash2.to_size_t());
}

} // namespace

SCENARIO("irep2 hashing", "[core][irep2]")
{
  GIVEN("Expressions constructed in the same way")
  {
    std::vector<std::pair<expr2tc, expr2tc>> expressions{
      {gen_ulong(42), gen_ulong(42)},
      {gen_testing_struct(1, 2), gen_testing_struct(1, 2)}};

    THEN("Expressions and hashes should be equal to itself")
    {
      for (auto &e : expressions)
      {
        test_constructed_equally(e.first, e.first);
        test_constructed_equally(e.second, e.second);
      }
    }
    THEN("Expressions and hashes between pairs should be equal")
    {
      for (auto &e : expressions)
        test_constructed_equally(e.first, e.second);
    }
  }
  GIVEN("Expressions constructed differently")
  {
    std::vector<std::pair<expr2tc, expr2tc>> expressions{
      {gen_ulong(42), gen_ulong(64)},
      {gen_testing_struct(1, 2), gen_ulong(1)},
      {gen_testing_struct(1, 2), gen_ulong(2)},
      {gen_testing_overlap(0), gen_testing_overlap(2)}, // overlap
      {gen_testing_struct(1, 2), gen_testing_struct(1, 1)}};
    THEN("Expressions and hashes between pairs should not be equal")
    {
      for (auto &e : expressions)
        test_constructed_differently(e.first, e.second);
    }
  }
}

// Regression for do_type_lt(const std::vector<expr2tc>&, ...) iterating past
// the end of the shorter vector when operand vectors have different lengths.
// Before the fix the longer side dereferences side2.end(); with ASan this
// trips an OOB read, otherwise the comparison can return a stale value or
// crash depending on heap layout.
SCENARIO(
  "constant_array2t ordering with unequal member counts",
  "[core][irep2]")
{
  GIVEN("Two constant_array2ts with different numbers of members")
  {
    expr2tc small = gen_testing_array(2);
    expr2tc big = gen_testing_array(5);

    THEN("operator< is a strict weak ordering across both directions")
    {
      // Neither direction may dereference past either vector. Both calls
      // must terminate without OOB access, and they must be consistent
      // (not both true).
      bool small_lt_big = (*small < *big);
      bool big_lt_small = (*big < *small);
      REQUIRE(small_lt_big != big_lt_small);
    }

    THEN("lt is antisymmetric and non-zero")
    {
      int forward = small->lt(*big);
      int reverse = big->lt(*small);
      REQUIRE(forward != 0);
      REQUIRE(reverse != 0);
      REQUIRE(forward == -reverse);
    }
  }
}

// Regression for the NDEBUG dynamic_cast→static_cast redefine: a bad to_*2t /
// to_*_type used to invoke UB in release builds. The checked-cast helpers now
// throw irep2_cast_error in every build mode.
SCENARIO("bad to_* downcasts throw irep2_cast_error", "[core][irep2]")
{
  GIVEN("A bool type passed to to_signedbv_type")
  {
    type2tc bool_ty = get_bool_type();
    THEN("the cast throws")
    {
      REQUIRE_THROWS_AS(to_signedbv_type(bool_ty), irep2_cast_error);
    }
  }

  GIVEN("A constant_int expr passed to to_add2t")
  {
    type2tc int_ty = get_uint_type(config.ansi_c.word_size);
    expr2tc lit = constant_int2tc(int_ty, BigInt(7));
    THEN("the cast throws")
    {
      REQUIRE_THROWS_AS(to_add2t(lit), irep2_cast_error);
    }
  }

  GIVEN("A bool type passed to to_bool_type (matching kind)")
  {
    type2tc bool_ty = get_bool_type();
    THEN("the cast succeeds")
    {
      REQUIRE_NOTHROW(to_bool_type(bool_ty));
    }
  }
}

// Regression for the BigInt CRC path silently dropping bytes past 256 and
// ignoring the sign. Before the fix, two oversized BigInts that differ only
// past the buffer cutoff produced the same CRC, and negating a value did not
// change its CRC.
SCENARIO("BigInt CRC handles sign and oversized values", "[core][irep2]")
{
  // Build a hex string of `nibbles` characters, padded with leading zeros if
  // shorter. Two helpers keep the test cases readable below.
  auto hex_of_length = [](std::size_t nibbles, char tail_digit) {
    std::string s(nibbles, '0');
    s.front() = 'f'; // ensure number is large, not zero-padded into the void
    s.back() = tail_digit;
    return s;
  };
  auto crc_of = [](const std::string &hex) {
    BigInt v(hex.c_str(), 16);
    type2tc ty = get_int_type(64);
    expr2tc e = constant_int2tc(ty, v);
    return e->crc();
  };

  GIVEN("Positive vs negative of the same small magnitude")
  {
    type2tc ty = get_int_type(64);
    expr2tc pos = constant_int2tc(ty, BigInt(5));
    expr2tc neg = constant_int2tc(ty, BigInt(-5));
    THEN("CRCs differ")
    {
      REQUIRE(pos->crc() != neg->crc());
    }
  }

  GIVEN("Two oversized BigInts that differ only past byte 256 of the dump")
  {
    // 1024 hex digits = 512 bytes of magnitude; comfortably beyond the
    // 256-byte stack buffer. The two strings are identical for the leading
    // 512 nibbles (covering the first 256 bytes of dumped magnitude) and
    // differ only in the final nibble.
    std::string base = hex_of_length(1024, '1');
    std::string variant = base;
    variant.back() = '2';
    THEN("CRCs differ")
    {
      REQUIRE(crc_of(base) != crc_of(variant));
    }
  }

  GIVEN("Equal oversized BigInts built independently")
  {
    std::string s = hex_of_length(1024, 'a');
    THEN("CRCs are equal")
    {
      REQUIRE(crc_of(s) == crc_of(s));
    }
  }
}

// Regression for do_type_lt(const type2tc &, const type2tc &) dereferencing
// side1/side2 before checking for nil. The expr2tc overload above already
// handles nulls; the type2tc overload should be symmetric.
SCENARIO("do_type_lt(type2tc, type2tc) with nil sides", "[core][irep2]")
{
  type2tc nil_ty;
  type2tc int_ty = get_uint_type(config.ansi_c.word_size);
  type2tc bool_ty = get_bool_type();

  GIVEN("Both sides nil")
  {
    THEN("comparison is 0")
    {
      REQUIRE(do_type_lt(nil_ty, nil_ty) == 0);
    }
  }

  GIVEN("Only the left side is nil")
  {
    THEN("comparison is negative")
    {
      REQUIRE(do_type_lt(nil_ty, int_ty) < 0);
    }
  }

  GIVEN("Only the right side is nil")
  {
    THEN("comparison is positive")
    {
      REQUIRE(do_type_lt(int_ty, nil_ty) > 0);
    }
  }

  GIVEN("Two non-nil, equal types")
  {
    THEN("comparison is 0")
    {
      type2tc int_ty_2 = get_uint_type(config.ansi_c.word_size);
      REQUIRE(do_type_lt(int_ty, int_ty_2) == 0);
    }
  }

  GIVEN("Two non-nil, differing types")
  {
    THEN("comparison is antisymmetric and non-zero")
    {
      int forward = do_type_lt(int_ty, bool_ty);
      int reverse = do_type_lt(bool_ty, int_ty);
      REQUIRE(forward != 0);
      REQUIRE(reverse != 0);
      REQUIRE(forward == -reverse);
    }
  }
}

// Positive coverage for the debug-only single-writer stamp on irep2t.
// In NDEBUG builds the stamp doesn't exist and this scenario is a
// no-op (the storage and helpers are compiled out); we keep the test
// always-defined-but-conditional-bodied so the test count is stable
// across build modes.
SCENARIO(
  "irep2t writer-thread stamp tracks ownership transitions",
  "[core][irep2]")
{
#ifdef NDEBUG
  // Stamp is compiled out; there is nothing to observe. Pass through.
  SUCCEED("writer-thread stamp is disabled in release builds");
#else
  // Inspecting the stamp through `e->writer_thread` on a non-const
  // expr2tc would itself call the non-const operator->(), which goes
  // through get() and stamps the slot — defeating the test. Bind a
  // const reference and read through that to use the const operator->
  // (a pure pointer load, no stamping).
  GIVEN("A freshly constructed expr2tc")
  {
    expr2tc e = gen_ulong(42);
    const expr2tc &ce = e;
    THEN("the writer stamp is initially clear")
    {
      REQUIRE(
        ce->writer_thread.load(std::memory_order_relaxed) == std::uintptr_t{0});
    }

    WHEN("we obtain a mutable reference")
    {
      // Trigger the non-const get(), which detaches (no-op here, we're
      // the sole owner) and stamps the writer slot.
      (void)e.get();
      THEN("the stamp matches the current thread's tag")
      {
        std::uintptr_t tag = irep2t::current_thread_tag();
        REQUIRE(ce->writer_thread.load(std::memory_order_relaxed) == tag);
      }

      AND_WHEN("we mutate the same expr again from the same thread")
      {
        // No-op fast path: mark_writer sees a matching stamp and
        // returns without touching the atomic.
        (void)e.get();
        THEN("the stamp is still ours")
        {
          REQUIRE(
            ce->writer_thread.load(std::memory_order_relaxed) ==
            irep2t::current_thread_tag());
        }
      }
    }
  }

  GIVEN("Two containers sharing one node, then dropping back to one")
  {
    expr2tc a = gen_ulong(7);
    const expr2tc &ca = a;
    (void)a.get(); // stamp the node as written by us
    REQUIRE(
      ca->writer_thread.load(std::memory_order_relaxed) ==
      irep2t::current_thread_tag());

    {
      expr2tc b = a; // refcount 2; stamp is still 'us'
      REQUIRE(
        ca->writer_thread.load(std::memory_order_relaxed) ==
        irep2t::current_thread_tag());
      // b goes out of scope here; release() observes refcount transition
      // from 2 → 1 and clears the stamp so the remaining owner gets a
      // clean slate.
    }
    THEN("the stamp is cleared once refcount returns to 1")
    {
      REQUIRE(
        ca->writer_thread.load(std::memory_order_relaxed) == std::uintptr_t{0});
    }
  }
#endif // NDEBUG
}

// Copy-on-write: a non-const access through a sharing container must
// clone, leaving the other handles bound to the original (unchanged)
// node. Verifies the contract documented at irep_container::get() and
// in the irep2.h threading-contract preamble.
SCENARIO("COW detach leaves the other handle unchanged", "[core][irep2]")
{
  // Helper: const get() does not detach, so it's safe for identity
  // comparison without perturbing the refcount.
  auto raw = [](const expr2tc &c) { return c.get(); };

  GIVEN("Two containers sharing a single underlying node")
  {
    expr2tc a = gen_ulong(11);
    expr2tc b = a; // refcount 2, both point to the same node
    REQUIRE(raw(a) == raw(b));

    WHEN("the first handle is mutated via non-const access")
    {
      // Non-const get() triggers detach(): a clones to a fresh node,
      // b keeps the original. After this call, a and b alias
      // different underlying objects.
      (void)a.get();

      THEN("a and b now point to distinct nodes")
      {
        REQUIRE(raw(a) != raw(b));
      }
      THEN("the structural value of both is still equal")
      {
        REQUIRE(*a == *b);
      }
    }
  }

  GIVEN("A container with refcount 1 (no sharing)")
  {
    expr2tc a = gen_ulong(13);
    const expr2t *before = raw(a);
    WHEN("the sole owner takes a mutable view")
    {
      (void)a.get();
      THEN("no clone happens — the underlying pointer is unchanged")
      {
        REQUIRE(raw(a) == before);
      }
    }
  }
}

// Sanity-check the flat-layout switch dispatchers on a representative
// expr kind (not2t).
SCENARIO("not2t flat-layout dispatchers (issue #4560)", "[core][irep2]")
{
  GIVEN("A not2t wrapping a constant_int expression")
  {
    type2tc word = get_uint_type(config.ansi_c.word_size);
    expr2tc inner = constant_int2tc(word, BigInt(3));
    expr2tc e_not = not2tc(inner);
    expr2tc e_not2 = not2tc(inner); // independent copy with same value

    THEN("cmp returns true for structurally-equal operands")
    {
      REQUIRE(e_not->cmp(*e_not2));
      REQUIRE(e_not->cmp(*e_not));
    }

    THEN("lt is 0 for equal operands")
    {
      REQUIRE(e_not->lt(*e_not) == 0);
    }

    THEN("crc agrees between equal nodes")
    {
      REQUIRE(e_not->crc() == e_not2->crc());
    }

    THEN("get_num_sub_exprs returns 1")
    {
      REQUIRE(e_not->get_num_sub_exprs() == 1u);
    }

    THEN("get_sub_expr(0) is the wrapped operand")
    {
      const expr2tc *p = e_not->get_sub_expr(0);
      REQUIRE(p != nullptr);
      REQUIRE(*p == inner);
    }

    THEN("get_sub_expr out-of-range returns nullptr")
    {
      REQUIRE(e_not->get_sub_expr(1) == nullptr);
    }

    THEN("clone produces a structurally equal independent copy")
    {
      expr2tc cloned = e_not->clone();
      REQUIRE(cloned.get() != e_not.get());
      REQUIRE(*cloned == *e_not);
    }

    THEN("foreach_operand visits exactly the one operand")
    {
      std::vector<expr2tc> visited;
      e_not->foreach_operand(
        [&](const expr2tc &sub) { visited.push_back(sub); });
      REQUIRE(visited.size() == 1u);
      REQUIRE(visited[0] == inner);
    }
  }

  GIVEN("Two not2t with different operands")
  {
    type2tc word = get_uint_type(config.ansi_c.word_size);
    expr2tc a = not2tc(constant_int2tc(word, BigInt(1)));
    expr2tc b = not2tc(constant_int2tc(word, BigInt(2)));

    THEN("cmp returns false")
    {
      REQUIRE(!a->cmp(*b));
    }

    THEN("lt is antisymmetric")
    {
      int fwd = a->lt(*b);
      int rev = b->lt(*a);
      REQUIRE(fwd != 0);
      REQUIRE(rev != 0);
      REQUIRE((fwd < 0) != (rev < 0));
    }

    THEN("crc differs")
    {
      REQUIRE(a->crc() != b->crc());
    }
  }
}

// get_sub_expr(idx) is contractually nullable: it returns nullptr
// when idx is past the operand count. Several simplify paths and
// the indexed-loop bench rely on this for bounds termination.
SCENARIO("get_sub_expr returns nullptr past the operand count", "[core][irep2]")
{
  GIVEN("A binary expression (add2t) with two operands")
  {
    type2tc word = get_uint_type(config.ansi_c.word_size);
    expr2tc e = add2tc(word, gen_ulong(1), gen_ulong(2));

    THEN("the two in-range indices return non-null sub-expressions")
    {
      REQUIRE(e->get_sub_expr(0) != nullptr);
      REQUIRE(e->get_sub_expr(1) != nullptr);
    }
    THEN("indices >= get_num_sub_exprs() return nullptr")
    {
      const size_t n = e->get_num_sub_exprs();
      REQUIRE(n == 2);
      REQUIRE(e->get_sub_expr(n) == nullptr);
      REQUIRE(e->get_sub_expr(n + 5) == nullptr);
      REQUIRE(e->get_sub_expr(std::numeric_limits<size_t>::max()) == nullptr);
    }
  }

  GIVEN("A constant_array2t with multiple elements")
  {
    expr2tc arr = gen_testing_array(4);
    THEN("each element is reachable and out-of-range returns nullptr")
    {
      const size_t n = arr->get_num_sub_exprs();
      REQUIRE(n == 4);
      for (size_t i = 0; i < n; ++i)
        REQUIRE(arr->get_sub_expr(i) != nullptr);
      REQUIRE(arr->get_sub_expr(n) == nullptr);
    }
  }

  GIVEN("A leaf expression with no sub-expressions")
  {
    expr2tc leaf = gen_ulong(42);
    THEN("get_num_sub_exprs is zero and every index returns nullptr")
    {
      REQUIRE(leaf->get_num_sub_exprs() == 0);
      REQUIRE(leaf->get_sub_expr(0) == nullptr);
      REQUIRE(leaf->get_sub_expr(1) == nullptr);
    }
  }
}

// Foundation tests for bigint_type2t (issue #4642). Phase 2A introduces the
// kind with no SMT lowering and no symex behaviour, so coverage is limited
// to the type-level contract: identity, classification, equality, hashing,
// width semantics, migrate round-trip, and that constant_int2t can carry
// arbitrary-precision values when typed bigint without width clipping.
SCENARIO("bigint_type2t identity and classification", "[core][irep2]")
{
  type2tc t = bigint_type2tc();

  THEN("downcasts and predicates report bigint")
  {
    REQUIRE(t->type_id == type2t::bigint_id);
    REQUIRE(is_bigint_type(t));
    REQUIRE(try_to_bigint_type(t) != nullptr);
  }

  THEN("bigint is a number and scalar, but not a bv, fractional, or byte type")
  {
    REQUIRE(is_number_type(t));
    REQUIRE(is_scalar_type(t));
    REQUIRE_FALSE(is_bv_type(t));
    REQUIRE_FALSE(is_fractional_type(t));
    REQUIRE_FALSE(is_byte_type(t));
    REQUIRE_FALSE(is_bool_type(t));
  }

  THEN("get_width throws because bigint is unbounded")
  {
    REQUIRE_THROWS_AS(t->get_width(), type2t::symbolic_type_excp);
  }

  THEN("pretty-prints as bigint")
  {
    REQUIRE(t->pretty(0).find("bigint") != std::string::npos);
  }
}

SCENARIO("bigint_type2t structural equality and hashing", "[core][irep2]")
{
  type2tc a = bigint_type2tc();
  type2tc b = bigint_type2tc();
  type2tc i64 = signedbv_type2tc(64);
  type2tc boolt = bool_type2tc();

  THEN("two bigint types compare equal and hash identically")
  {
    REQUIRE(a == b);
    REQUIRE(a->crc() == b->crc());
  }

  THEN("bigint is distinct from signedbv(64) and from bool")
  {
    REQUIRE_FALSE(a == i64);
    REQUIRE_FALSE(a == boolt);
  }
}

SCENARIO("bigint_type2t migrate round-trip", "[core][irep2]")
{
  type2tc original = bigint_type2tc();

  WHEN("we migrate back to legacy irep and forward again")
  {
    typet legacy = migrate_type_back(original);
    type2tc round = migrate_type(legacy);

    THEN("the legacy form carries the bigint id")
    {
      REQUIRE(legacy.id() == typet::t_bigint);
    }

    THEN("the round-tripped irep2 type matches the original")
    {
      REQUIRE(round == original);
      REQUIRE(is_bigint_type(round));
    }
  }
}

SCENARIO(
  "constant_int2t carries arbitrary precision under bigint type",
  "[core][irep2]")
{
  // 2^200 — far beyond any fixed-width bitvector ESBMC currently uses. Phase
  // 2A's contract is that the value survives in the IR with the bigint type
  // attached; SMT lowering arrives in PR 2C / 2D.
  BigInt huge = BigInt::power2(200);
  expr2tc k = constant_int2tc(bigint_type2tc(), huge);

  THEN("the value round-trips unchanged")
  {
    REQUIRE(is_constant_int2t(k));
    REQUIRE(to_constant_int2t(k).value == huge);
  }

  THEN("the carried type is bigint")
  {
    REQUIRE(is_bigint_type(k->type));
  }

  THEN("two equally-constructed constants compare equal and hash equally")
  {
    expr2tc k2 = constant_int2tc(bigint_type2tc(), huge);
    REQUIRE(k == k2);
    REQUIRE(k->crc() == k2->crc());
  }

  THEN("from_integer(huge, bigint_type) preserves the value")
  {
    expr2tc via_helper = from_integer(huge, bigint_type2tc());
    REQUIRE(is_constant_int2t(via_helper));
    REQUIRE(to_constant_int2t(via_helper).value == huge);
    REQUIRE(via_helper == k);
  }

  THEN("legacy from_integer(huge, bigint_typet) emits the decimal encoding")
  {
    exprt legacy_helper =
      from_integer(huge, migrate_type_back(bigint_type2tc()));
    REQUIRE(legacy_helper.type().id() == typet::t_bigint);
    REQUIRE(legacy_helper.value().as_string() == integer2string(huge, 10));
  }

  THEN("gen_zero / gen_one on bigint_typet produce well-formed legacy exprs")
  {
    exprt z = gen_zero(migrate_type_back(bigint_type2tc()));
    exprt o = gen_one(migrate_type_back(bigint_type2tc()));
    REQUIRE_FALSE(z.is_nil());
    REQUIRE_FALSE(o.is_nil());
    REQUIRE(z.type().id() == typet::t_bigint);
    REQUIRE(o.type().id() == typet::t_bigint);
    REQUIRE(z.value().as_string() == "0");
    REQUIRE(o.value().as_string() == "1");
  }
}

SCENARIO(
  "constant_int2t with bigint type survives migrate round-trip",
  "[core][irep2]")
{
  // bigint has no fixed width; migrate_expr_back's default constant_int
  // encoding reads thetype.width() and emits an empty binary value. The
  // dedicated bigint encoding stores a decimal string in expr.value() so
  // the round-trip is lossless for any BigInt magnitude or sign. See PR
  // #4646 review (Copilot finding #3).
  const std::vector<BigInt> samples{
    BigInt::power2(200),
    -BigInt::power2(200) - BigInt(1),
    BigInt(0),
    BigInt(1),
    BigInt(-1),
    BigInt::power2(63),                          // INT64_MAX + 1
    -BigInt::power2(63) - BigInt(1),             // INT64_MIN - 1
    BigInt::power2(1024) + BigInt::power2(512)}; // not a single bit

  GIVEN("a representative bigint constant_int2t")
  {
    THEN("each sample round-trips losslessly via migrate")
    {
      for (const BigInt &v : samples)
      {
        expr2tc original = constant_int2tc(bigint_type2tc(), v);
        exprt legacy = migrate_expr_back(original);
        REQUIRE(legacy.type().id() == typet::t_bigint);
        REQUIRE(legacy.value().as_string() == integer2string(v, 10));

        expr2tc round;
        migrate_expr(legacy, round);
        REQUIRE(is_constant_int2t(round));
        REQUIRE(is_bigint_type(round->type));
        REQUIRE(to_constant_int2t(round).value == v);
        REQUIRE(round == original);
        REQUIRE(round->crc() == original->crc());
      }
    }
  }
}

SCENARIO(
  "bigint arithmetic preserves precision through simplify",
  "[core][irep2]")
{
  // Phase 2B contract: arithmetic over bigint-typed constant_int2t operands
  // must simplify without clipping the value to any fixed width. The
  // existing simplify_arith_2ops dispatch gated on is_bv_type, so before
  // PR 2B the bigint branch was unreachable — folds either declined or
  // truncated through a synthetic typecast.
  const type2tc bigint = bigint_type2tc();
  const BigInt a = BigInt::power2(200);
  const BigInt b = BigInt::power2(199);

  GIVEN("add2tc(bigint, 2^200, 2^200)")
  {
    expr2tc e =
      add2tc(bigint, constant_int2tc(bigint, a), constant_int2tc(bigint, a));
    THEN("simplify folds to 2^201 unclipped")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_int2t(s));
      REQUIRE(is_bigint_type(s->type));
      REQUIRE(to_constant_int2t(s).value == BigInt::power2(201));
    }
  }

  GIVEN("sub2tc(bigint, 2^200, 2^199)")
  {
    expr2tc e =
      sub2tc(bigint, constant_int2tc(bigint, a), constant_int2tc(bigint, b));
    THEN("simplify folds to 2^199 unclipped")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_int2t(s));
      REQUIRE(is_bigint_type(s->type));
      REQUIRE(to_constant_int2t(s).value == b);
    }
  }

  GIVEN("mul2tc(bigint, 2^200, 2^200)")
  {
    expr2tc e =
      mul2tc(bigint, constant_int2tc(bigint, a), constant_int2tc(bigint, a));
    THEN("simplify folds to 2^400 unclipped")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_int2t(s));
      REQUIRE(is_bigint_type(s->type));
      REQUIRE(to_constant_int2t(s).value == BigInt::power2(400));
    }
  }

  GIVEN("neg2tc(bigint, 2^200)")
  {
    expr2tc e = neg2tc(bigint, constant_int2tc(bigint, a));
    THEN("simplify folds to -(2^200) with the sign preserved")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_int2t(s));
      REQUIRE(is_bigint_type(s->type));
      REQUIRE(to_constant_int2t(s).value == -a);
    }
  }

  GIVEN("add2tc(bigint, 0, 2^200) — identity short-circuit")
  {
    expr2tc e = add2tc(
      bigint,
      constant_int2tc(bigint, BigInt(0)),
      constant_int2tc(bigint, a));
    THEN("add2t::do_simplify returns the non-zero side unclipped")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_int2t(s));
      REQUIRE(is_bigint_type(s->type));
      REQUIRE(to_constant_int2t(s).value == a);
    }
  }

  GIVEN("div2tc(bigint, 2^400, 2^200) — exact division of huge values")
  {
    expr2tc e = div2tc(
      bigint,
      constant_int2tc(bigint, BigInt::power2(400)),
      constant_int2tc(bigint, a));
    THEN("simplify folds to 2^200 unclipped")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_int2t(s));
      REQUIRE(is_bigint_type(s->type));
      REQUIRE(to_constant_int2t(s).value == a);
    }
  }

  GIVEN("modulus2tc(bigint, 2^400 + 7, 2^200)")
  {
    expr2tc e = modulus2tc(
      bigint,
      constant_int2tc(bigint, BigInt::power2(400) + BigInt(7)),
      constant_int2tc(bigint, a));
    THEN("simplify folds to 7 (no width clipping)")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_int2t(s));
      REQUIRE(is_bigint_type(s->type));
      REQUIRE(to_constant_int2t(s).value == BigInt(7));
    }
  }

  GIVEN("chained mul: ((2^200) * 2) * 2 — depth-2 fold")
  {
    expr2tc inner = mul2tc(
      bigint,
      constant_int2tc(bigint, a),
      constant_int2tc(bigint, BigInt(2)));
    expr2tc outer =
      mul2tc(bigint, inner, constant_int2tc(bigint, BigInt(2)));
    THEN("simplify folds the whole chain to 2^202 unclipped")
    {
      expr2tc s = outer->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_int2t(s));
      REQUIRE(is_bigint_type(s->type));
      REQUIRE(to_constant_int2t(s).value == BigInt::power2(202));
    }
  }

  GIVEN("mul of opposite-sign huge values")
  {
    expr2tc e =
      mul2tc(bigint, constant_int2tc(bigint, -a), constant_int2tc(bigint, a));
    THEN("simplify folds to -(2^400) with sign preserved")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_int2t(s));
      REQUIRE(is_bigint_type(s->type));
      REQUIRE(to_constant_int2t(s).value == -BigInt::power2(400));
    }
  }

  GIVEN("div(0, 2^200) — numerator zero short-circuit")
  {
    expr2tc e = div2tc(
      bigint,
      constant_int2tc(bigint, BigInt(0)),
      constant_int2tc(bigint, a));
    THEN("simplify folds to 0 (no width-dependent zero generation)")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_int2t(s));
      REQUIRE(is_bigint_type(s->type));
      REQUIRE(to_constant_int2t(s).value == BigInt(0));
    }
  }

  GIVEN("equality of two equal bigint constants")
  {
    expr2tc e = equality2tc(
      constant_int2tc(bigint, BigInt::power2(200)),
      constant_int2tc(bigint, BigInt::power2(200)));
    THEN("simplify folds to true via the relations BigInt path")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_bool2t(s));
      REQUIRE(to_constant_bool2t(s).value == true);
    }
  }

  GIVEN("equality of two distinct bigint constants")
  {
    expr2tc e = equality2tc(
      constant_int2tc(bigint, a),
      constant_int2tc(bigint, b));
    THEN("simplify folds to false")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_bool2t(s));
      REQUIRE(to_constant_bool2t(s).value == false);
    }
  }

  GIVEN("lessthan(2^199, 2^200)")
  {
    expr2tc e = lessthan2tc(
      constant_int2tc(bigint, b), constant_int2tc(bigint, a));
    THEN("simplify folds to true")
    {
      expr2tc s = e->simplify();
      REQUIRE(!is_nil_expr(s));
      REQUIRE(is_constant_bool2t(s));
      REQUIRE(to_constant_bool2t(s).value == true);
    }
  }
}
