#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2_utils.h>
#include <pointer-analysis/value_set.h>
#include <util/config/config.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

TEST_CASE(
  "symbolic array reads retain constant-index pointer writes",
  "[value-set]")
{
  config.ansi_c.set_data_model(configt::LP64);
  const bool old_mode = config.options.get_bool_option("solidity-mode");
  struct restore_mode
  {
    bool old;
    ~restore_mode()
    {
      config.options.set_option("solidity-mode", old);
    }
  } restore{old_mode};
  config.options.set_option("solidity-mode", true);
  contextt ctx;
  namespacet ns(ctx);
  value_sett vs(ns);
  const type2tc integer = get_int32_type();
  const type2tc pointer = pointer_type2tc(integer);
  const expr2tc zero = constant_int2tc(integer, 0);
  const expr2tc one = constant_int2tc(integer, 1);
  const expr2tc arr = symbol2tc(
    array_type2tc(pointer, constant_int2tc(integer, 2), false), "arr");
  const expr2tc a = symbol2tc(integer, "a");
  const expr2tc b = symbol2tc(integer, "b");
  value_setst::valuest result;
  SECTION("array index")
  {
    vs.assign(index2tc(pointer, arr, zero), address_of2tc(pointer, a));
    vs.assign(index2tc(pointer, arr, one), address_of2tc(pointer, b));
    vs.get_value_set(index2tc(pointer, arr, symbol2tc(integer, "i")), result);
  }
  SECTION("pointer offset")
  {
    const type2tc pointer_pointer = pointer_type2tc(pointer);
    const expr2tc base = symbol2tc(pointer_pointer, "base");
    const expr2tc storage = dynamic_object2tc(pointer, one, false, false);
    vs.assign(base, address_of2tc(pointer_pointer, storage));
    vs.assign(dereference2tc(pointer, base), address_of2tc(pointer, a));
    vs.assign(
      dereference2tc(pointer, add2tc(pointer_pointer, base, one)),
      address_of2tc(pointer, b));
    vs.get_value_set(
      dereference2tc(
        pointer, add2tc(pointer_pointer, base, symbol2tc(integer, "i"))),
      result);
  }
  bool has_a = false, has_b = false;
  for (const expr2tc &value : result)
    if (is_object_descriptor2t(value))
    {
      const expr2tc &object = to_object_descriptor2t(value).object;
      has_a |= object == a;
      has_b |= object == b;
    }
  REQUIRE(has_a);
  REQUIRE(has_b);
}
