/*******************************************************************
 Module: native goto conversion of an expression statement whose
 operand is code

 convert_expression() converts a code-typed operand of any statement
 kind through convert(). The native dispatcher must do the same rather
 than decline the whole body, and must emit what the legacy converter
 emits for the round-tripped body (frontends-to-irep2.md §31.3, §86).
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "../testing-utils/goto_factory.h"
#include <goto-programs/goto_convert_functions.h>
#include <util/irep/migrate.h>
#include <util/lang/c_types.h>
#include <irep2/irep2_utils.h>

namespace
{
struct test_convertt : public goto_convert_functionst
{
  test_convertt(contextt &c, optionst &o, goto_functionst &f)
    : goto_convert_functionst(c, o, f)
  {
  }
  using goto_convert_functionst::try_convert_body_native;
  using goto_convertt::goto_convert_rec;
};

locationt line(unsigned n)
{
  locationt loc;
  loc.set_file("test.c");
  loc.set_line(n);
  return loc;
}

// Convert `body2` natively and through the legacy converter, each with a
// fresh converter, and require both to succeed with identical output.
void require_native_matches_legacy(
  program &p,
  const expr2tc &body2,
  bool no_assertions = false,
  bool compare_locations = true)
{
  cmdlinet cmd = goto_factory::get_default_cmdline("test.c");
  optionst opts = goto_factory::get_default_options(cmd);
  opts.set_option("no-assertions", no_assertions);

  goto_functionst native_functions;
  test_convertt native_conv(p.context, opts, native_functions);
  goto_programt native;
  REQUIRE(native_conv.try_convert_body_native(body2, native));

  goto_functionst legacy_functions;
  test_convertt legacy_conv(p.context, opts, legacy_functions);
  goto_programt legacy;
  legacy_conv.goto_convert_rec(to_code(migrate_expr_back(body2)), legacy);

  REQUIRE(native.instructions.size() == legacy.instructions.size());
  auto n = native.instructions.begin();
  for (const auto &l : legacy.instructions)
  {
    CHECK(n->type == l.type);
    CHECK(n->code == l.code);
    CHECK(n->guard == l.guard);
    if (compare_locations)
      CHECK(n->location == l.location);
    ++n;
  }
}

const char *const SRC =
  "int x;\n"
  "int main(void) { return x; }\n";
} // namespace

TEST_CASE(
  "an assignment as an expression statement's operand converts natively",
  "[goto-convert][irep2]")
{
  std::string src = SRC;
  program p =
    goto_factory::get_goto_functions(src, goto_factory::Architecture::BIT_64);
  const symbolt *x = p.context.find_symbol("c:@x");
  REQUIRE(x != nullptr);

  const expr2tc assign = code_assign2tc(
    symbol_expr2tc(*x),
    constant_int2tc(migrate_type(int_type()), BigInt(1)),
    line(3));
  const expr2tc body2 =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(assign, line(3))});

  require_native_matches_legacy(p, body2);
}

TEST_CASE(
  "a block as an expression statement's operand converts natively",
  "[goto-convert][irep2]")
{
  std::string src = SRC;
  program p =
    goto_factory::get_goto_functions(src, goto_factory::Architecture::BIT_64);
  const symbolt *x = p.context.find_symbol("c:@x");
  REQUIRE(x != nullptr);

  const type2tc int_t = migrate_type(int_type());
  const expr2tc first = code_assign2tc(
    symbol_expr2tc(*x), constant_int2tc(int_t, BigInt(1)), line(4));
  const expr2tc second = code_assign2tc(
    symbol_expr2tc(*x), constant_int2tc(int_t, BigInt(2)), line(5));
  const expr2tc inner =
    code_block2tc(std::vector<expr2tc>{first, second}, line(4), line(6));
  const expr2tc body2 =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(inner, line(4))});

  require_native_matches_legacy(p, body2);
}

TEST_CASE(
  "a do-while with an empty native body is delegated, not declined",
  "[goto-convert][irep2]")
{
  std::string src = SRC;
  program p =
    goto_factory::get_goto_functions(src, goto_factory::Architecture::BIT_64);

  // An assert under --no-assertions emits nothing, so the loop body is empty.
  const expr2tc body2 = code_block2tc(std::vector<expr2tc>{code_dowhile2tc(
    gen_false_expr(), code_assert2tc(gen_true_expr(), line(3)), line(3))});

  // The legacy side here skips restore_value_locations, which stamps the
  // loop's back-edge GOTO from the statement location, so locations differ.
  require_native_matches_legacy(p, body2, true, false);
}
