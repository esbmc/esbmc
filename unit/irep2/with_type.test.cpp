// H-B5: expr2t::with_type / dispatch totality. Every expr kind is either
//   - with_type-supported: e.with_type(e->type) round-trips structurally, and
//     substituting a type then reverting returns the original; or
//   - unsupported (its ctor derives the type from operands): with_type routes
//     to a deliberate abort rather than silently returning a mistyped node.
// We also pin the dispatcher's exhaustiveness by deriving the kind count from
// the expr_kinds.inc / type_kinds.inc manifests (an X-macro fold) rather than a
// literal, and smoke-test crc/cmp/clone/tostring/get_num_sub_exprs on real
// nodes. All checks run against the genuine irep2 classes.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <utility>
#include <vector>

// The unsupported-kind death-test forks and waits on a POSIX child; guard the
// mechanism (and its test case) out on Windows, which has neither fork() nor
// <sys/wait.h>.
#if !defined(_WIN32)
#  include <csignal>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>

namespace
{
// Kind counts derived from the single-source-of-truth manifests. If a kind is
// added to the enum without a manifest line (or vice versa) these disagree.
constexpr unsigned expr_kind_count()
{
  unsigned n = 0;
#define IREP2_EXPR(kind, pretty) ++n;
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
  return n;
}
constexpr unsigned type_kind_count()
{
  unsigned n = 0;
#define IREP2_TYPE(kind, pretty) ++n;
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
  return n;
}
static_assert(expr_kind_count() == expr2t::end_expr_id, "expr manifest drift");
static_assert(type_kind_count() == type2t::end_type_id, "type manifest drift");
} // namespace

TEST_CASE("with_type round-trips on supported kinds (H-B5)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  type2tc u32 = get_uint_type(32);
  type2tc u64 = get_uint_type(64);
  type2tc s32 = get_int_type(32);
  expr2tc c5 = constant_int2tc(u32, BigInt(5));
  expr2tc c7 = constant_int2tc(u32, BigInt(7));

  // Kinds whose first field is &expr2t::type and whose ctor accepts it, each
  // paired with a valid alternate type. Arithmetic kinds assert that the
  // result-type width matches the operand width (assert_arith_2ops_consistency,
  // active in non-Release builds), so their alternate must keep the operand
  // width (u32 -> s32); pure type-slot kinds accept an arbitrary type (u64).
  std::vector<std::pair<expr2tc, type2tc>> supported{
    {constant_int2tc(u32, BigInt(5)), u64},
    {symbol2tc(u32, "x"), u64},
    {add2tc(u32, c5, c7), s32},
    {sub2tc(u32, c5, c7), s32},
    {mul2tc(u32, c5, c7), s32},
    {bitand2tc(u32, c5, c7), s32},
    {bitor2tc(u32, c5, c7), s32},
    {address_of2tc(u32, symbol2tc(u32, "x")), pointer_type2tc(u64)},
  };

  for (const auto &[e, alt] : supported)
  {
    // Re-applying the same type is a structural identity.
    REQUIRE(e->with_type(e->type) == e);

    // Substituting a type changes only the type; reverting restores the node.
    expr2tc retyped = e->with_type(alt);
    REQUIRE(retyped->type == alt);
    REQUIRE(retyped->with_type(e->type) == e);
  }
}

TEST_CASE("with_type rebuilds an address-of from the pointee", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  type2tc u32 = get_uint_type(32);
  type2tc u64 = get_uint_type(64);
  expr2tc obj = symbol2tc(u32, "x");

  // address_of2t's ctor takes the *pointee* type and builds the pointer, so a
  // generic rebuild that forwards new_type verbatim wraps it a second time.
  expr2tc retyped = address_of2tc(u32, obj)->with_type(pointer_type2tc(u64));
  const type2tc &pointee = to_pointer_type(retyped->type).subtype;
  REQUIRE(pointee == u64);
  REQUIRE(!is_pointer_type(pointee));
  REQUIRE(to_address_of2t(retyped).ptr_obj == obj);

  // implicit is in the fields tuple, so dropping it would also break equality.
  expr2tc sugar =
    address_of2tc(u32, obj, true)->with_type(pointer_type2tc(u64));
  REQUIRE(to_address_of2t(sugar).implicit);

  // The rebuild reads new_type as a pointer; a non-pointer is a caller bug.
  REQUIRE_THROWS_AS(address_of2tc(u32, obj)->with_type(u64), irep2_cast_error);
}

#if !defined(_WIN32)
TEST_CASE("with_type aborts on an unsupported kind (H-B5)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  // constant_bool2t derives its type from the value (its ctor takes no type),
  // so it is not with_type-substitutable and must hit the abort path.
  expr2tc b = constant_bool2tc(true);

  pid_t pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0)
  {
    // Child: silence the diagnostic and Catch2's signal-handler report (it
    // re-raises SIGABRT after printing), then trigger the unsupported path.
    // Consume the freopen result to satisfy -Werror=unused-result.
    if (
      !freopen("/dev/null", "w", stderr) || !freopen("/dev/null", "w", stdout))
      _exit(2);
    (void)b->with_type(get_uint_type(32));
    _exit(0); // unreachable if with_type aborts as required
  }

  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFSIGNALED(status));
  REQUIRE(WTERMSIG(status) == SIGABRT);
}
#endif

TEST_CASE("dispatch smoke over real nodes (H-B5)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  type2tc u32 = get_uint_type(32);
  expr2tc c5 = constant_int2tc(u32, BigInt(5));

  std::vector<expr2tc> nodes{
    constant_int2tc(u32, BigInt(5)),
    constant_bool2tc(true),
    symbol2tc(u32, "x"),
    add2tc(u32, c5, c5),
    not2tc(constant_bool2tc(true)),
  };

  for (const expr2tc &e : nodes)
  {
    // clone is a structural identity whose crc matches the original's.
    expr2tc twin = e->clone();
    REQUIRE(twin == e);
    REQUIRE(twin->crc() == e->crc());

    // tostring and the operand count run without crashing and agree with a
    // manual operand walk.
    REQUIRE(!e->pretty(0).empty());
    unsigned counted = 0;
    e->foreach_operand([&counted](const expr2tc &) { ++counted; });
    REQUIRE(e->get_num_sub_exprs() == counted);
  }
}
