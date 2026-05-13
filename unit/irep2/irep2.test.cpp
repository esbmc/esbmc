#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <irep2/irep2.h>
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
  GIVEN("A freshly constructed expr2tc")
  {
    expr2tc e = gen_ulong(42);
    THEN("the writer stamp is initially clear")
    {
      REQUIRE(
        e->writer_thread.load(std::memory_order_relaxed) == std::uintptr_t{0});
    }

    WHEN("we obtain a mutable reference")
    {
      // Trigger the non-const get(), which detaches (no-op here, we're
      // the sole owner) and stamps the writer slot.
      (void)e.get();
      THEN("the stamp matches the current thread's tag")
      {
        std::uintptr_t tag = irep2t::current_thread_tag();
        REQUIRE(e->writer_thread.load(std::memory_order_relaxed) == tag);
      }

      AND_WHEN("we mutate the same expr again from the same thread")
      {
        // No-op fast path: mark_writer sees a matching stamp and
        // returns without touching the atomic.
        (void)e.get();
        THEN("the stamp is still ours")
        {
          REQUIRE(
            e->writer_thread.load(std::memory_order_relaxed) ==
            irep2t::current_thread_tag());
        }
      }
    }
  }

  GIVEN("Two containers sharing one node, then dropping back to one")
  {
    expr2tc a = gen_ulong(7);
    (void)a.get(); // stamp the node as written by us
    REQUIRE(
      a->writer_thread.load(std::memory_order_relaxed) ==
      irep2t::current_thread_tag());

    {
      expr2tc b = a; // refcount 2; stamp is still 'us'
      REQUIRE(
        a->writer_thread.load(std::memory_order_relaxed) ==
        irep2t::current_thread_tag());
      // b goes out of scope here; release() observes refcount transition
      // from 2 → 1 and clears the stamp so the remaining owner gets a
      // clean slate.
    }
    THEN("the stamp is cleared once refcount returns to 1")
    {
      REQUIRE(
        a->writer_thread.load(std::memory_order_relaxed) == std::uintptr_t{0});
    }
  }
#endif // NDEBUG
}
