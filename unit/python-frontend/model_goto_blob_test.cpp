/*******************************************************************
 Module: The Python operational-model blob ships lowered GOTO

 python2goto runs goto_convert before writing pysrc64, so each model body
 travels in the binary's function section and esbmc links it instead of
 lowering the same 100-odd functions on every run. Both halves are pinned
 here: the symbols arrive as declarations, and their bodies arrive beside
 them.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/python_library.h>

#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_functions.h>
#include <util/config/config.h>
#include <util/irep/migrate.h>
#include <util/config/options.h>
#include <util/symtab/context.h>

#include <string>

namespace
{
bool is_model(const symbolt &s)
{
  return s.id.as_string().compare(0, 3, "py:") == 0;
}

size_t functions_with_bodies(const goto_functionst &gf)
{
  size_t n = 0;
  for (const auto &named : gf.function_map)
    if (named.second.body_available)
      n++;
  return n;
}
} // namespace

TEST_CASE("the model blob ships bodies, not codet", "[python][models]")
{
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  // migrate reaches for this while deserialising the blob's types.
  const namespacet ns(ctx);
  migrate_namespace_lookup = &ns;

  add_cpython_library(ctx);

  size_t models = 0;
  size_t models_carrying_a_value = 0;
  ctx.foreach_operand([&](const symbolt &s) {
    if (s.is_type || !s.get_type().is_code() || !is_model(s))
      return;
    models++;
    if (!s.get_value().is_nil())
      models_carrying_a_value++;
  });

  // Reading the blob must produce model symbols at all, or everything below
  // passes vacuously.
  REQUIRE(models > 0);

  // The half that makes goto_convert cheap: nothing left for it to lower.
  REQUIRE(models_carrying_a_value == 0);

  goto_functionst gf;
  optionst opts;
  goto_convert(ctx, opts, gf);

  // goto_convert saw declarations, so it produced entries without bodies.
  const size_t before = functions_with_bodies(gf);
  link_cpython_library_bodies(gf);
  const size_t after = functions_with_bodies(gf);

  // The other half: the bodies the blob carries reach the program.
  REQUIRE(after > before);
  REQUIRE(after >= models);
}
