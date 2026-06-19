/*******************************************************************
 Module: goto_convertt::remove_sideeffects IREP2 dual-API seam (W1)

 The expr2tc overload of remove_sideeffects delegates to the legacy
 exprt path, so it must be behaviour-identical: a side-effect-free
 expression is returned unchanged with no emitted instructions, and a
 function-call side effect is hoisted into an instruction with the
 expression replaced by its result symbol.
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "../testing-utils/goto_factory.h"
#include <goto-programs/goto_convert_class.h>
#include <util/migrate.h>
#include <util/c_types.h>
#include <irep2/irep2_utils.h>

namespace
{
// Expose the protected remove_sideeffects overloads for testing.
struct test_convertt : public goto_convertt
{
  test_convertt(contextt &c, optionst &o) : goto_convertt(c, o)
  {
  }
  using goto_convertt::has_sideeffect;
  using goto_convertt::remove_sideeffects;
};

const char *const SRC =
  "int f(void) { return 1; }\n"
  "int main(void) { return 0; }\n";

optionst default_options()
{
  cmdlinet cmd = goto_factory::get_default_cmdline("test.c");
  return goto_factory::get_default_options(cmd);
}
} // namespace

TEST_CASE(
  "remove_sideeffects expr2tc overload leaves a side-effect-free expr "
  "unchanged",
  "[goto-convert][irep2]")
{
  std::string src = SRC;
  program p =
    goto_factory::get_goto_functions(src, goto_factory::Architecture::BIT_64);
  optionst opts = default_options();
  test_convertt conv(p.context, opts);

  expr2tc e = constant_int2tc(migrate_type(int_type()), BigInt(5));
  const expr2tc original = e;

  goto_programt dest;
  conv.remove_sideeffects(e, dest);

  REQUIRE(dest.instructions.empty());
  REQUIRE(e == original);
}

TEST_CASE(
  "remove_sideeffects expr2tc overload hoists a function-call side effect",
  "[goto-convert][irep2]")
{
  std::string src = SRC;
  program p =
    goto_factory::get_goto_functions(src, goto_factory::Architecture::BIT_64);
  const symbolt *f = p.context.find_symbol("c:@F@f");
  REQUIRE(f != nullptr);

  optionst opts = default_options();
  test_convertt conv(p.context, opts);

  expr2tc call = side_effect_function_call2tc(
    migrate_type(int_type()), symbol_expr2tc(*f), std::vector<expr2tc>{});

  goto_programt dest;
  conv.remove_sideeffects(call, dest, /*result_is_used=*/true);

  // The call is hoisted out into an instruction and the expression is replaced
  // by its result symbol; no side effect remains.
  REQUIRE_FALSE(dest.instructions.empty());
  REQUIRE_FALSE(is_sideeffect2t(call));
  REQUIRE(is_symbol2t(call));
}

TEST_CASE(
  "remove_sideeffects expr2tc overload hoists a side effect nested in an "
  "expression",
  "[goto-convert][irep2]")
{
  std::string src = SRC;
  program p =
    goto_factory::get_goto_functions(src, goto_factory::Architecture::BIT_64);
  const symbolt *f = p.context.find_symbol("c:@F@f");
  REQUIRE(f != nullptr);

  optionst opts = default_options();
  test_convertt conv(p.context, opts);

  // 1 + f(): the side effect is nested under an add, so the native
  // has_sideeffect scan must recurse into the operand to find it.
  type2tc int_t = migrate_type(int_type());
  expr2tc call =
    side_effect_function_call2tc(int_t, symbol_expr2tc(*f), std::vector<expr2tc>{});
  expr2tc nested = add2tc(int_t, constant_int2tc(int_t, BigInt(1)), call);

  goto_programt dest;
  conv.remove_sideeffects(nested, dest, /*result_is_used=*/true);

  // The nested call is hoisted; the add survives with its second operand
  // replaced by the result symbol, leaving no side effect.
  REQUIRE_FALSE(dest.instructions.empty());
  REQUIRE(is_add2t(nested));
  REQUIRE_FALSE(is_sideeffect2t(to_add2t(nested).side_2));
  REQUIRE(is_symbol2t(to_add2t(nested).side_2));
}

TEST_CASE(
  "has_sideeffect detects a sideeffect_assign2t (assignment side effect)",
  "[goto-convert][irep2]")
{
  std::string src = SRC;
  program p =
    goto_factory::get_goto_functions(src, goto_factory::Architecture::BIT_64);
  optionst opts = default_options();
  test_convertt conv(p.context, opts);

  // A legacy "assign" sideeffect migrates to sideeffect_assign2t, a distinct
  // kind from sideeffect2t; has_sideeffect must treat it as a side effect both
  // at the top level and nested under another expression.
  type2tc int_t = migrate_type(int_type());
  expr2tc assign = sideeffect_assign2tc(
    int_t,
    irep_idt("assign"),
    symbol2tc(int_t, "x"),
    constant_int2tc(int_t, BigInt(1)),
    locationt());

  REQUIRE(conv.has_sideeffect(assign));
  REQUIRE(
    conv.has_sideeffect(add2tc(int_t, constant_int2tc(int_t, BigInt(2)), assign)));
}
