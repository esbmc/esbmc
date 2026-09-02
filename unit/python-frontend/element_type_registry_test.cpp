#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/type/element_type_registry.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/module/global_scope.h>
#include <python-frontend/type/type_handler.h>
#include <util/arith/arith_tools.h>
#include <util/config/config.h>
#include <util/lang/c_types.h>
#include <util/symtab/context.h>

#include <nlohmann/json.hpp>

using nlohmann::json;

namespace
{
void ensure_config_initialized()
{
  static bool initialized = false;
  if (initialized)
    return;

  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));
  initialized = true;
}

json make_dummy_ast()
{
  return json::parse(R"json({
    "body": [],
    "filename": "test.py",
    "ast_output_dir": "/tmp"
  })json");
}

typet string_type()
{
  return array_typet(char_type(), from_integer(4, size_type()));
}
} // namespace

TEST_CASE("recording and reading element types", "[python-frontend][types]")
{
  ensure_config_initialized();
  element_type_registry registry;

  SECTION("an unknown instance reads as absent")
  {
    REQUIRE(registry.size("lst") == 0);
    REQUIRE(registry.find("lst") == nullptr);
    REQUIRE(registry.element_type("lst") == typet());
    REQUIRE(registry.element_id("lst", 0).empty());
    REQUIRE(registry.last_element_type("lst") == typet());
  }

  SECTION("entries read back in order")
  {
    registry.record("lst", "e0", int_type());
    registry.record("lst", "e1", double_type());

    REQUIRE(registry.size("lst") == 2);
    REQUIRE(registry.element_type("lst", 0) == int_type());
    REQUIRE(registry.element_type("lst", 1) == double_type());
    REQUIRE(registry.element_id("lst", 0) == "e0");
    REQUIRE(registry.element_id("lst", 1) == "e1");
    REQUIRE(registry.last_element_type("lst") == double_type());
  }

  SECTION("element_type falls back to the first entry, element_id does not")
  {
    registry.record("lst", "e0", int_type());

    REQUIRE(registry.element_type("lst", 99) == int_type());
    REQUIRE(registry.element_id("lst", 99).empty());
  }
}

TEST_CASE("slots are isolated namespaces", "[python-frontend][types]")
{
  ensure_config_initialized();
  element_type_registry registry;

  // Recording a dict's value types under its values-list id must stay
  // invisible to the elements slot: a .values()/.items() list read that saw
  // them would take the generic mixed-list path and misread dict value
  // storage (github_3719_4).
  registry.record(
    "vals", std::string(), int_type(), type_slot::dict_value_types);
  registry.record(
    "vals", std::string(), double_type(), type_slot::dict_value_types);

  REQUIRE(registry.size("vals", type_slot::dict_value_types) == 2);
  REQUIRE(registry.has_mixed_numeric("vals", type_slot::dict_value_types));

  REQUIRE(registry.size("vals") == 0);
  REQUIRE(registry.find("vals") == nullptr);
  REQUIRE(registry.element_type("vals") == typet());
  REQUIRE_FALSE(registry.has_mixed_numeric("vals"));

  REQUIRE(registry.size("vals", type_slot::dict_value_list_elems) == 0);

  // The elements slot stays independent under the same id.
  registry.record("vals", "e0", string_type());
  REQUIRE(registry.element_type("vals") == string_type());
  REQUIRE(registry.size("vals", type_slot::dict_value_types) == 2);
}

TEST_CASE("copying between instances", "[python-frontend][types]")
{
  ensure_config_initialized();
  element_type_registry registry;
  registry.record("src", "e0", int_type());
  registry.record("src", "e1", double_type());

  SECTION("assign_from replaces the destination")
  {
    registry.record("dst", "old", string_type());
    registry.assign_from("src", "dst");

    REQUIRE(registry.size("dst") == 2);
    REQUIRE(registry.element_type("dst", 0) == int_type());
    REQUIRE(registry.element_id("dst", 0) == "e0");
  }

  SECTION("assign_from leaves the destination alone for an unknown source")
  {
    registry.record("dst", "old", string_type());
    registry.assign_from("missing", "dst");

    REQUIRE(registry.size("dst") == 1);
    REQUIRE(registry.element_type("dst") == string_type());
  }

  SECTION("append_from leaves the destination alone for an unknown source")
  {
    registry.record("dst", "old", string_type());
    registry.append_from("missing", "dst");

    REQUIRE(registry.size("dst") == 1);
    REQUIRE(registry.element_type("dst") == string_type());
  }

  SECTION("append_from concatenates")
  {
    registry.record("dst", "old", string_type());
    registry.append_from("src", "dst");

    REQUIRE(registry.size("dst") == 3);
    REQUIRE(registry.element_type("dst", 0) == string_type());
    REQUIRE(registry.element_type("dst", 1) == int_type());
    REQUIRE(registry.element_type("dst", 2) == double_type());
  }

  SECTION("append_from onto itself doubles the sequence")
  {
    registry.append_from("src", "src");

    REQUIRE(registry.size("src") == 4);
    REQUIRE(registry.element_type("src", 2) == int_type());
    REQUIRE(registry.element_type("src", 3) == double_type());
  }

  SECTION("append_from repeats the source sequence")
  {
    registry.append_from("src", "dst", 3);

    REQUIRE(registry.size("dst") == 6);
    for (size_t i = 0; i < 3; ++i)
    {
      REQUIRE(registry.element_type("dst", 2 * i) == int_type());
      REQUIRE(registry.element_type("dst", 2 * i + 1) == double_type());
    }
  }

  SECTION("repeating onto itself scales linearly, not exponentially")
  {
    // `l = l * 4`: the result holds 4 copies of the source, not 2^4.
    registry.append_from("src", "src", 3);

    REQUIRE(registry.size("src") == 8);
    REQUIRE(registry.element_type("src", 6) == int_type());
    REQUIRE(registry.element_type("src", 7) == double_type());
  }

  SECTION("append_from is a no-op for zero repetitions")
  {
    registry.record("dst", "old", string_type());
    registry.append_from("src", "dst", 0);

    REQUIRE(registry.size("dst") == 1);
  }

  SECTION("copies are independent of the source")
  {
    registry.assign_from("src", "dst");
    registry.record("src", "e2", string_type());

    REQUIRE(registry.size("dst") == 2);
    REQUIRE(registry.size("src") == 3);
  }
}

TEST_CASE("mutating a recorded sequence", "[python-frontend][types]")
{
  ensure_config_initialized();
  element_type_registry registry;

  SECTION("pop_last drops the tail")
  {
    registry.record("lst", "e0", int_type());
    registry.record("lst", "e1", double_type());
    registry.pop_last("lst");

    REQUIRE(registry.size("lst") == 1);
    REQUIRE(registry.element_type("lst") == int_type());
  }

  SECTION("popping the only entry makes the instance absent")
  {
    registry.record("lst", "e0", int_type());
    registry.pop_last("lst");

    REQUIRE(registry.size("lst") == 0);
    REQUIRE(registry.find("lst") == nullptr);
  }

  SECTION("pop_last on an unknown instance is a no-op")
  {
    registry.pop_last("lst");
    REQUIRE(registry.size("lst") == 0);
  }

  SECTION("reverse mirrors an in-place reversal")
  {
    registry.record("lst", "e0", int_type());
    registry.record("lst", "e1", double_type());
    registry.record("lst", "e2", string_type());
    registry.reverse("lst");

    REQUIRE(registry.element_type("lst", 0) == string_type());
    REQUIRE(registry.element_id("lst", 0) == "e2");
    REQUIRE(registry.element_type("lst", 2) == int_type());
    REQUIRE(registry.element_id("lst", 2) == "e0");
  }

  SECTION("reverse of a single entry or unknown instance is a no-op")
  {
    registry.record("one", "e0", int_type());
    registry.reverse("one");
    registry.reverse("missing");

    REQUIRE(registry.element_type("one") == int_type());
    REQUIRE(registry.size("missing") == 0);
  }
}

TEST_CASE("derived type queries", "[python-frontend][types]")
{
  ensure_config_initialized();
  element_type_registry registry;

  SECTION("uniform_element_type")
  {
    registry.record("same", "", int_type());
    registry.record("same", "", int_type());
    REQUIRE(registry.uniform_element_type("same") == int_type());

    registry.record("mixed", "", int_type());
    registry.record("mixed", "", double_type());
    REQUIRE(registry.uniform_element_type("mixed") == typet());

    REQUIRE(registry.uniform_element_type("missing") == typet());
  }

  SECTION("has_mixed_numeric needs both an integer and a float")
  {
    registry.record("ints", "", int_type());
    registry.record("ints", "", long_int_type());
    REQUIRE_FALSE(registry.has_mixed_numeric("ints"));

    registry.record("floats", "", double_type());
    REQUIRE_FALSE(registry.has_mixed_numeric("floats"));

    registry.record("both", "", int_type());
    registry.record("both", "", double_type());
    REQUIRE(registry.has_mixed_numeric("both"));

    // A string alongside an int is not a numeric mix.
    registry.record("strs", "", int_type());
    registry.record("strs", "", string_type());
    REQUIRE_FALSE(registry.has_mixed_numeric("strs"));
  }

  SECTION("numeric_element_type promotes an int/float mix to double")
  {
    registry.record("mix", "", int_type());
    registry.record("mix", "", double_type());
    REQUIRE(registry.numeric_element_type("mix") == double_type());

    registry.record("ints", "", int_type());
    registry.record("ints", "", int_type());
    REQUIRE(registry.numeric_element_type("ints") == int_type());
  }

  SECTION("numeric_element_type rejects non-numeric and mixed widths")
  {
    registry.record("strs", "", string_type());
    REQUIRE(registry.numeric_element_type("strs") == typet());

    registry.record("widths", "", int_type());
    registry.record("widths", "", long_int_type());
    REQUIRE(registry.numeric_element_type("widths") == typet());

    REQUIRE(registry.numeric_element_type("missing") == typet());
  }

  SECTION("homogeneous_element_type accepts one type and an int/float mix")
  {
    registry.record("same", "", int_type());
    registry.record("same", "", int_type());
    REQUIRE(registry.homogeneous_element_type("same", "sorted") == int_type());

    registry.record("mix", "", int_type());
    registry.record("mix", "", double_type());
    REQUIRE(
      registry.homogeneous_element_type("mix", "sorted") == double_type());

    REQUIRE(registry.homogeneous_element_type("missing", "sorted") == typet());
  }

  SECTION("homogeneous_element_type treats all char sequences as strings")
  {
    registry.record("strs", "", string_type());
    registry.record("strs", "", pointer_typet(char_type()));
    REQUIRE_NOTHROW(registry.homogeneous_element_type("strs", "sorted"));
  }

  SECTION("homogeneous_element_type rejects an incompatible mix")
  {
    registry.record("bad", "", int_type());
    registry.record("bad", "", string_type());
    REQUIRE_THROWS_AS(
      registry.homogeneous_element_type("bad", "sorted"), std::runtime_error);
  }
}

TEST_CASE("pop-site type memo", "[python-frontend][types]")
{
  ensure_config_initialized();
  element_type_registry registry;

  SECTION("an unmemoized site reads as absent")
  {
    REQUIRE(registry.memoized_pop_type("site") == typet());
  }

  SECTION("memoize_pop_type is replayed by a later lookup")
  {
    registry.memoize_pop_type("site", int_type());
    REQUIRE(registry.memoized_pop_type("site") == int_type());
  }
}

TEST_CASE(
  "type_flags mirrors the list model encoding",
  "[python-frontend][types]")
{
  ensure_config_initialized();

  json ast = make_dummy_ast();
  contextt ctx;
  global_scope gs;
  python_converter converter(ctx, &ast, gs);
  const type_handler &th = converter.get_type_handler();

  element_type_registry registry;
  int flag = -1;
  size_t float_type_id = 0;

  SECTION("all integers encode as 0 with no float id")
  {
    registry.record("lst", "", int_type());
    registry.type_flags("lst", th, flag, float_type_id);

    REQUIRE(flag == 0);
    REQUIRE(float_type_id == 0);
  }

  SECTION("all floats encode as 1 and carry a float id")
  {
    registry.record("lst", "", double_type());
    registry.type_flags("lst", th, flag, float_type_id);

    REQUIRE(flag == 1);
    REQUIRE(float_type_id != 0);
  }

  SECTION("a string element encodes as 2 and wins over numerics")
  {
    registry.record("lst", "", int_type());
    registry.record("lst", "", string_type());
    registry.type_flags("lst", th, flag, float_type_id);

    REQUIRE(flag == 2);
  }

  SECTION("an int/float mix encodes as 3")
  {
    registry.record("lst", "", int_type());
    registry.record("lst", "", double_type());
    registry.type_flags("lst", th, flag, float_type_id);

    REQUIRE(flag == 3);
    REQUIRE(float_type_id != 0);
  }

  SECTION("an unknown instance encodes as 0")
  {
    registry.type_flags("missing", th, flag, float_type_id);

    REQUIRE(flag == 0);
    REQUIRE(float_type_id == 0);
  }
}

TEST_CASE(
  "each converter owns its own element types",
  "[python-frontend][types]")
{
  ensure_config_initialized();

  json ast = make_dummy_ast();
  contextt ctx_a;
  contextt ctx_b;
  global_scope gs_a;
  global_scope gs_b;

  python_converter converter_a(ctx_a, &ast, gs_a);
  python_converter converter_b(ctx_b, &ast, gs_b);

  converter_a.get_element_type_registry().record("shared_id", "e0", int_type());

  // Before the registry was owned, this state lived in a process-wide static
  // map and would leak between conversions.
  REQUIRE(converter_a.get_element_type_registry().size("shared_id") == 1);
  REQUIRE(converter_b.get_element_type_registry().size("shared_id") == 0);
}
