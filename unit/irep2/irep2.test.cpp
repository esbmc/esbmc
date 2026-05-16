#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <limits>
#include <irep2/irep2.h>
#include <irep2/irep2_dispatch.h>
#include <irep2/irep2_template_utils.h>
#include <irep2/irep2_utils.h>
#include <util/crypto_hash.h>

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
  REQUIRE(e1->do_crc() == e2->do_crc());
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
  REQUIRE(e1->do_crc() != e2->do_crc());
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

    THEN("ltchecked is antisymmetric and non-zero")
    {
      int forward = small->ltchecked(*big);
      int reverse = big->ltchecked(*small);
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

// Verify that the step-2 flat-layout migration of not2t (issue #4560) is
// correct: has_fields_v<not2t> must be true, and every _v2 dispatcher must
// agree with the corresponding virtual method on the same operand.
SCENARIO(
  "not2t _v2 dispatchers agree with virtual methods (issue #4560)",
  "[core][irep2]")
{
  GIVEN("A not2t wrapping a constant_int expression")
  {
    type2tc word = get_uint_type(config.ansi_c.word_size);
    expr2tc inner = constant_int2tc(word, BigInt(3));
    expr2tc e_not = not2tc(inner);
    expr2tc e_not2 = not2tc(inner); // independent copy with same value

    THEN("has_fields_v<not2t> is true")
    {
      REQUIRE(esbmct::has_fields_v<not2t>);
    }

    THEN("cmp_v2 matches virtual cmp")
    {
      REQUIRE(e_not->cmp_v2(*e_not2) == e_not->cmp(*e_not2));
      REQUIRE(e_not->cmp_v2(*e_not) == e_not->cmp(*e_not));
    }

    THEN("lt_v2 matches virtual lt (same operand → 0)")
    {
      REQUIRE(e_not->lt_v2(*e_not) == 0);
      int virt_lt = e_not->lt(*e_not);
      REQUIRE(e_not->lt_v2(*e_not) == virt_lt);
    }

    THEN("do_crc_v2 matches virtual do_crc")
    {
      REQUIRE(e_not->do_crc_v2() == e_not->do_crc());
      REQUIRE(e_not->do_crc_v2() == e_not2->do_crc_v2());
    }

    THEN("hash_v2 produces same digest as virtual hash")
    {
      crypto_hash h1, h2;
      e_not->hash_v2(h1);
      e_not->hash(h2);
      h1.fin();
      h2.fin();
      REQUIRE(h1.to_size_t() == h2.to_size_t());
    }

    THEN("tostring_v2 matches virtual tostring")
    {
      auto v1 = e_not->tostring_v2(0);
      auto v2 = e_not->tostring(0);
      REQUIRE(v1 == v2);
    }

    THEN("get_num_sub_exprs_v2 matches virtual get_num_sub_exprs")
    {
      REQUIRE(e_not->get_num_sub_exprs_v2() == e_not->get_num_sub_exprs());
      REQUIRE(e_not->get_num_sub_exprs_v2() == 1u);
    }

    THEN("get_sub_expr_v2(0) is non-null and equals the wrapped operand")
    {
      const expr2tc *p = e_not->get_sub_expr_v2(0);
      const expr2tc *p_virt = e_not->get_sub_expr(0);
      REQUIRE(p != nullptr);
      REQUIRE(p_virt != nullptr);
      REQUIRE(*p == *p_virt);
    }

    THEN("get_sub_expr_v2 out-of-range returns nullptr")
    {
      REQUIRE(e_not->get_sub_expr_v2(1) == nullptr);
    }

    THEN("clone_v2 produces a structurally equal independent copy")
    {
      expr2tc cloned = e_not->clone_v2();
      REQUIRE(cloned.get() != e_not.get()); // distinct underlying object
      REQUIRE(*cloned == *e_not);           // same structural value
    }

    THEN("foreach_operand (public) visits exactly the one operand")
    {
      std::vector<expr2tc> visited;
      e_not->foreach_operand([&](const expr2tc &sub) { visited.push_back(sub); });
      REQUIRE(visited.size() == 1u);
      REQUIRE(visited[0] == inner);
    }
  }

  GIVEN("Two not2t with different operands")
  {
    type2tc word = get_uint_type(config.ansi_c.word_size);
    expr2tc a = not2tc(constant_int2tc(word, BigInt(1)));
    expr2tc b = not2tc(constant_int2tc(word, BigInt(2)));

    THEN("cmp_v2 returns false")
    {
      REQUIRE(!a->cmp_v2(*b));
    }

    THEN("lt_v2 is antisymmetric")
    {
      int fwd = a->lt_v2(*b);
      int rev = b->lt_v2(*a);
      REQUIRE(fwd != 0);
      REQUIRE(rev != 0);
      // signs must be opposite
      REQUIRE((fwd < 0) != (rev < 0));
    }

    THEN("do_crc_v2 differs")
    {
      REQUIRE(a->do_crc_v2() != b->do_crc_v2());
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
