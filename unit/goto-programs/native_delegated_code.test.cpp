/*******************************************************************
 Module: native goto conversion of the statement kinds that
 convert_native_rec delegates to convert()

 dead, printf, free and asm have no native
 arm. The dispatcher must accept them and emit what the legacy
 converter emits for the round-tripped body (frontends-to-irep2.md
 §89).
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "../testing-utils/goto_factory.h"
#include <goto-programs/goto_convert_functions.h>
#include <util/irep/migrate.h>
#include <util/lang/c_types.h>
#include <irep2/irep2_utils.h>
#include <functional>

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
void require_native_matches_legacy(program &p, const expr2tc &body2)
{
  cmdlinet cmd = goto_factory::get_default_cmdline("test.c");
  optionst opts = goto_factory::get_default_options(cmd);

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
    CHECK(n->location == l.location);
    ++n;
  }
}

const char *const SRC =
  "int x;\n"
  "int main(void) { return x; }\n";

// Wrap `stmt` in a block, between two assignments so that its position and
// the neighbours' conversion are exercised too.
void check_delegated(const std::function<expr2tc(const symbolt &)> &make)
{
  std::string src = SRC;
  program p =
    goto_factory::get_goto_functions(src, goto_factory::Architecture::BIT_64);
  const symbolt *x = p.context.find_symbol("c:@x");
  REQUIRE(x != nullptr);

  const type2tc int_t = migrate_type(int_type());
  const expr2tc before = code_assign2tc(
    symbol_expr2tc(*x), constant_int2tc(int_t, BigInt(1)), line(3));
  const expr2tc after = code_assign2tc(
    symbol_expr2tc(*x), constant_int2tc(int_t, BigInt(2)), line(5));
  const expr2tc body2 = code_block2tc(
    std::vector<expr2tc>{before, make(*x), after}, line(3), line(6));

  require_native_matches_legacy(p, body2);
}
} // namespace

TEST_CASE("a dead statement converts natively", "[goto-convert][irep2]")
{
  check_delegated([](const symbolt &x) {
    return code_dead2tc(migrate_type(int_type()), x.id, line(4));
  });
}

TEST_CASE("a printf statement converts natively", "[goto-convert][irep2]")
{
  check_delegated([](const symbolt &x) {
    return code_printf2tc(
      std::vector<expr2tc>{symbol_expr2tc(x)}, printf_kindt::PRINTF, line(4));
  });
}

TEST_CASE("a free statement converts natively", "[goto-convert][irep2]")
{
  check_delegated(
    [](const symbolt &x) { return code_free2tc(symbol_expr2tc(x), line(4)); });
}

TEST_CASE("an asm statement converts natively", "[goto-convert][irep2]")
{
  check_delegated([](const symbolt &) {
    return code_asm2tc(get_empty_type(), irep_idt("nop"), line(4));
  });
}
