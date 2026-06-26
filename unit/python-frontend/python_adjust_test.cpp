#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/python_adjust.h>
#include <irep2/irep2_utils.h>
#include <util/config.h>
#include <util/context.h>
#include <util/c_types.h>

// Part V Phase V.1k (b) sub-phase B.0 (docs/irep2-migration.md): the IREP2-native
// Python adjuster ships first as a structural no-op walker. These tests pin the
// two B.0 contracts so B.1 (member/index source following) extends a verified
// traversal:
//   (1) adjust_expr visits every node of an IREP2 expression tree exactly once.
//   (2) adjust() leaves every symbol's IREP2 value byte-identical (no-op).

namespace
{
// Observes the traversal by counting every non-nil node the walker descends
// into. Relies on adjust_expr being virtual and recursing through itself.
struct counting_adjust : public python_adjust
{
  using python_adjust::python_adjust;
  unsigned visited = 0;

  void adjust_expr(const expr2tc &expr) override
  {
    if (!is_nil_expr(expr))
      ++visited;
    python_adjust::adjust_expr(expr);
  }
};
} // namespace

TEST_CASE(
  "python_adjust B.0 walker visits every IREP2 node once",
  "[python-frontend][irep2][python-adjust]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  // add( mul(1, 2), 3 ) — 2 internal arith nodes + 3 leaf constants = 5 nodes.
  const type2tc t = get_uint_type(config.ansi_c.word_size);
  expr2tc tree = add2tc(t, mul2tc(t, gen_ulong(1), gen_ulong(2)), gen_ulong(3));

  contextt context;
  counting_adjust walker(context);
  walker.adjust_expr(tree);

  REQUIRE(walker.visited == 5);
}

TEST_CASE(
  "python_adjust B.0 adjust() is a no-op over the symbol table",
  "[python-frontend][irep2][python-adjust]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  const type2tc t = get_uint_type(config.ansi_c.word_size);
  const expr2tc value =
    add2tc(t, mul2tc(t, gen_ulong(7), gen_ulong(11)), gen_ulong(13));

  contextt context;
  symbolt symbol;
  symbol.id = "py:test@F@adjust_noop";
  symbol.name = "adjust_noop";
  symbol.mode = "Python";
  symbol.set_value(value);
  symbol.set_type(t);
  context.add(symbol);

  // A type symbol carrying a value: adjust() must skip it (is_type branch) and
  // leave it byte-identical, just as clang_c_adjust::adjust() does.
  symbolt type_symbol;
  type_symbol.id = "py:test@T@tag-Skipped";
  type_symbol.name = "tag-Skipped";
  type_symbol.mode = "Python";
  type_symbol.is_type = true;
  type_symbol.set_value(value);
  type_symbol.set_type(t);
  context.add(type_symbol);

  REQUIRE_FALSE(python_adjust(context).adjust());

  const symbolt *out = context.find_symbol("py:test@F@adjust_noop");
  REQUIRE(out != nullptr);
  REQUIRE(out->get_value2() == value);

  const symbolt *type_out = context.find_symbol("py:test@T@tag-Skipped");
  REQUIRE(type_out != nullptr);
  REQUIRE(type_out->get_value2() == value);
}
