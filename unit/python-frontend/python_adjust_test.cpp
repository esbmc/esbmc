#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/python_adjust.h>
#include <util/context.h>
#include <util/symbol.h>
#include <util/c_types.h>
#include <irep2/irep2_utils.h>

// Phase B.0 dead-but-tested gate for the V.1k (b) IREP2-native Python adjuster
// (docs/irep2-migration.md, "V.1k (b)-adjuster", phase B.0).
//
// The B.0 pass is a *structural no-op*: walking an IREP2 expression — and a
// symbol's IREP2 value through the full adjust() entry point — must leave it
// byte-identical. This pins the inert baseline that later phases (B.1+, which
// resolve member2t/index2t sources) validate against, the same
// "machinery-first, prove-inert, wire-later" gate used for the V.4.0
// structured-CF kinds (esbmc/esbmc#5265).

namespace
{
// (x + 1) == 0, a small nested IREP2 tree with leaves and interior nodes.
expr2tc make_sample_expr()
{
  const type2tc int_t = get_int32_type();
  const expr2tc x = symbol2tc(int_t, "x");
  const expr2tc sum = add2tc(int_t, x, gen_one(int_t));
  return equality2tc(sum, gen_zero(int_t));
}
} // namespace

TEST_CASE(
  "python_adjust B.0 leaves an expression byte-identical",
  "[python-adjust]")
{
  const expr2tc original = make_sample_expr();
  expr2tc walked = original;

  contextt ctx;
  python_adjust adjuster(ctx);
  adjuster.adjust_expr(walked);

  REQUIRE(walked == original);
}

TEST_CASE(
  "python_adjust B.0 adjust() leaves a symbol's IREP2 value unchanged",
  "[python-adjust]")
{
  const expr2tc value = make_sample_expr();

  symbolt symbol;
  symbol.id = "py_adjust_test_sym";
  symbol.name = "py_adjust_test_sym";
  symbol.mode = "Python";
  symbol.set_type(get_int32_type());
  symbol.set_value(value);

  contextt ctx;
  ctx.add(symbol);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());

  const symbolt *out = ctx.find_symbol("py_adjust_test_sym");
  REQUIRE(out != nullptr);
  REQUIRE(out->get_value2() == value);
}

TEST_CASE(
  "python_adjust B.0 ignores a nil-valued symbol",
  "[python-adjust]")
{
  symbolt symbol;
  symbol.id = "py_adjust_nil_sym";
  symbol.name = "py_adjust_nil_sym";
  symbol.mode = "Python";
  symbol.set_type(get_int32_type());

  contextt ctx;
  ctx.add(symbol);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());
}
