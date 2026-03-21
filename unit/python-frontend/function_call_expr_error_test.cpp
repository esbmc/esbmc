#define CATCH_CONFIG_MAIN

// Allow test access to private members
#define private public
#define protected public
#include <python-frontend/function_call_expr.h>
#include <python-frontend/python_converter.h>
#undef private
#undef protected

#include <catch2/catch.hpp>
#include <nlohmann/json.hpp>
#include <python-frontend/global_scope.h>
#include <python-frontend/symbol_id.h>
#include <util/config.h>
#include <util/context.h>
#include <util/python_types.h>

using json = nlohmann::json;

namespace
{

/// Minimal AST accepted by python_converter constructor and
/// type_handler::is_constructor_call (needs "body" array).
json make_dummy_ast()
{
  return json::parse(R"json({
    "body": [],
    "filename": "test.py",
    "ast_output_dir": "/tmp"
  })json");
}

/// Minimal Call node that satisfies the function_call_expr constructor
/// (get_function_type inspects func._type and func.id).
json make_dummy_call()
{
  return json::parse(R"json({
    "_type": "Call",
    "func": { "_type": "Name", "id": "test_func" },
    "args": [],
    "lineno": 1,
    "col_offset": 0
  })json");
}

} // namespace

TEST_CASE(
  "generate_attribute_error returns typed nondet fallback",
  "[python-frontend][attribute-error]")
{
  // Initialise global config so converters don't crash.
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  json ast = make_dummy_ast();
  json call = make_dummy_call();

  contextt ctx;
  global_scope gs;
  python_converter converter(ctx, &ast, gs);

  // Provide a block so add_instruction() can push the assert.
  code_blockt block;
  converter.current_block = &block;

  symbol_id sid("test.py", "", "test_func");
  function_call_expr fce(sid, call, converter);

  SECTION("returns sideeffect/nondet with given type")
  {
    typet expected = signedbv_typet(32);
    exprt result =
      fce.generate_attribute_error("foo", {"MyClass"}, expected);

    // The assert instruction should have been emitted into the block.
    REQUIRE(block.operands().size() == 1);
    REQUIRE(block.operands()[0].id() == "code_assert");

    // Returned expression must be a sideeffect/nondet with the expected type.
    REQUIRE(result.id() == "sideeffect");
    REQUIRE(result.statement() == "nondet");
    REQUIRE(result.type() == expected);
  }

  SECTION("falls back to any_type when expected type is empty")
  {
    // Reset block.
    block.operands().clear();

    exprt result =
      fce.generate_attribute_error("bar", {"A", "B"}, empty_typet());

    REQUIRE(block.operands().size() == 1);

    REQUIRE(result.id() == "sideeffect");
    REQUIRE(result.statement() == "nondet");
    // Must not be empty — should have fallen back to any_type().
    REQUIRE_FALSE(result.type() == empty_typet());
    REQUIRE(result.type() == any_type());
  }

  SECTION("falls back to any_type when expected type is nil")
  {
    block.operands().clear();

    exprt result =
      fce.generate_attribute_error("baz", {}, typet());

    REQUIRE(result.id() == "sideeffect");
    REQUIRE(result.statement() == "nondet");
    REQUIRE(result.type() == any_type());
  }
}
