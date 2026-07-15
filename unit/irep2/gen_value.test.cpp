// H-A10: gen_zero / gen_one recursion & aggregate size loops (irep2_utils.cpp).
// gen_zero recurses over aggregate types (array/vector/struct/union), driving
// the element loops on constant_int sizes; the result must be a structurally
// correct zero for every supported type and its nesting. gen_one only handles
// scalars; both deliberately abort on an unsupported type. All checks run
// against the real irep2_utils functions on real type2tc inputs.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <functional>
#include <vector>

// The abort-path death-test forks and waits on a POSIX child; guard the whole
// mechanism (and the test case that uses it) out on Windows, which has neither
// fork() nor <sys/wait.h>.
#if !defined(_WIN32)
#  include <csignal>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>

namespace
{
expr2tc size_of(unsigned long n)
{
  return constant_int2tc(get_uint_type(32), BigInt(n));
}

#if !defined(_WIN32)
// Fork a child, run fn (expected to abort()), and require SIGABRT.
void require_aborts(const std::function<void()> &fn)
{
  pid_t pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0)
  {
    // Silence the diagnostic + Catch2's signal-handler report. Consume the
    // freopen result to satisfy -Werror=unused-result.
    if (
      !freopen("/dev/null", "w", stderr) || !freopen("/dev/null", "w", stdout))
      _exit(2);
    fn();
    _exit(0); // unreachable if fn aborts
  }
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFSIGNALED(status));
  REQUIRE(WTERMSIG(status) == SIGABRT);
}
#endif
} // namespace

TEST_CASE("gen_zero on scalar types (H-A10)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  REQUIRE(gen_zero(get_bool_type()) == gen_false_expr());
  REQUIRE(
    gen_zero(get_uint_type(32)) ==
    constant_int2tc(get_uint_type(32), BigInt(0)));
  REQUIRE(
    gen_zero(get_int_type(64)) == constant_int2tc(get_int_type(64), BigInt(0)));
}

TEST_CASE("gen_zero recurses over aggregates (H-A10)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  type2tc u8 = get_uint_type(8);
  type2tc i32 = get_int_type(32);
  expr2tc z8 = constant_int2tc(u8, BigInt(0));
  expr2tc z32 = constant_int2tc(i32, BigInt(0));

  SECTION("array expands to N zeroed members")
  {
    type2tc arr = array_type2tc(u8, size_of(3), false);
    expr2tc got = gen_zero(arr);
    REQUIRE(is_constant_array2t(got));
    REQUIRE(to_constant_array2t(got).datatype_members.size() == 3);
    REQUIRE(got == constant_array2tc(arr, std::vector<expr2tc>{z8, z8, z8}));
  }

  SECTION("array_as_array_of collapses to a single initializer")
  {
    type2tc arr = array_type2tc(u8, size_of(3), false);
    REQUIRE(gen_zero(arr, true) == constant_array_of2tc(arr, z8));
  }

  SECTION("vector expands to N zeroed lanes")
  {
    type2tc vec = vector_type2tc(u8, size_of(4));
    REQUIRE(
      gen_zero(vec) ==
      constant_vector2tc(vec, std::vector<expr2tc>{z8, z8, z8, z8}));
  }

  SECTION("struct zeroes each member")
  {
    std::vector<type2tc> memb{i32, u8};
    std::vector<irep_idt> names{"a", "b"};
    type2tc st = struct_type2tc(memb, names, names, "s");
    REQUIRE(
      gen_zero(st) == constant_struct2tc(st, std::vector<expr2tc>{z32, z8}));
  }

  SECTION("union zeroes only the first member")
  {
    std::vector<type2tc> memb{i32, u8};
    std::vector<irep_idt> names{"a", "b"};
    type2tc un = union_type2tc(memb, names, names, "u");
    REQUIRE(
      gen_zero(un) == constant_union2tc(un, "a", std::vector<expr2tc>{z32}));
  }

  SECTION("nested struct-of-array recurses correctly")
  {
    type2tc inner = array_type2tc(u8, size_of(2), false);
    std::vector<type2tc> memb{inner, i32};
    std::vector<irep_idt> names{"arr", "n"};
    type2tc st = struct_type2tc(memb, names, names, "nested");

    expr2tc inner_zero = constant_array2tc(inner, std::vector<expr2tc>{z8, z8});
    REQUIRE(
      gen_zero(st) ==
      constant_struct2tc(st, std::vector<expr2tc>{inner_zero, z32}));

    // Determinism: independent calls produce equal values.
    REQUIRE(gen_zero(st) == gen_zero(st));
  }
}

TEST_CASE("gen_one on scalar types (H-A10)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  REQUIRE(gen_one(get_bool_type()) == gen_true_expr());
  REQUIRE(
    gen_one(get_uint_type(32)) ==
    constant_int2tc(get_uint_type(32), BigInt(1)));
  REQUIRE(
    gen_one(get_int_type(64)) == constant_int2tc(get_int_type(64), BigInt(1)));
}

#if !defined(_WIN32)
TEST_CASE(
  "gen_zero / gen_one abort on unsupported types (H-A10)",
  "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  // gen_zero has no case for empty_type.
  type2tc empty = empty_type2tc();
  require_aborts([&]() { (void)gen_zero(empty); });

  // gen_one only handles scalars; an aggregate hits its abort path.
  std::vector<type2tc> memb{get_int_type(32)};
  std::vector<irep_idt> names{"a"};
  type2tc st = struct_type2tc(memb, names, names, "s");
  require_aborts([&]() { (void)gen_one(st); });
}
#endif
