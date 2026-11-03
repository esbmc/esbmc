#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <clang-c-frontend/padding.h>
#include <util/config/config.h>
#include <util/symtab/context.h>

TEST_CASE("Extended integer padding preserves the record layout", "[padding]")
{
  config.ansi_c.set_data_model(configt::LP64);
  contextt context;
  namespacet ns(context);
  const unsigned width = GENERATE(8, 16, 32, 64, 128, 256, 480);
  unsignedbv_typet integer(width);
  integer.set("#extint", true);
  struct_typet record;
  record.components().emplace_back("value", "value", integer);
  record.components().emplace_back("tail", "tail", unsignedbv_typet(64));

  add_padding(record, ns);
  for (const auto &component : record.components())
  {
    if (component.get_is_padding())
      CHECK(component.type().width() != "0");
  }
  const struct_typet first_layout = record;
  add_padding(record, ns);
  CHECK(record == first_layout);
}

TEST_CASE("Unaligned extended integers retain their padding", "[padding]")
{
  config.ansi_c.set_data_model(configt::LP64);
  contextt context;
  namespacet ns(context);
  unsignedbv_typet integer(65);
  integer.set("#extint", true);
  struct_typet record;
  record.components().emplace_back("value", "value", integer);
  add_padding(record, ns);
  REQUIRE(record.components().size() == 2);
  CHECK(record.components()[1].get_is_padding());
  CHECK(record.components()[1].type().width() == "63");
  const struct_typet first_layout = record;
  add_padding(record, ns);
  CHECK(record == first_layout);
}
