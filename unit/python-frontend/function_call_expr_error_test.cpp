// Catch2 header with main
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

// System includes before custom ones with ESBMC macros
#include <nlohmann/json.hpp>

// Allow test access to private members
#define private public
#define protected public
#include <python-frontend/function_call_expr.h>
#include <python-frontend/python_converter.h>
#undef private
#undef protected
#include <python-frontend/function_call_cache.h>
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
    REQUIRE(block.operands()[0].is_code());
    REQUIRE(to_code(block.operands()[0]).get_statement() == "assert");

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

// =========================================================================
// Cache behavior and hierarchy corner-case tests
// =========================================================================

namespace
{

/// Build an AST with the given class definitions in "body".
json make_ast_with_classes(const json &body_items)
{
  json ast;
  ast["body"] = body_items;
  ast["filename"] = "test.py";
  ast["ast_output_dir"] = "/tmp";
  return ast;
}

/// Build a minimal ClassDef node.
json make_class(
  const std::string &name,
  const json &body = json::array(),
  const json &bases = json::array())
{
  json cls;
  cls["_type"] = "ClassDef";
  cls["name"] = name;
  cls["body"] = body;
  cls["bases"] = bases;
  return cls;
}

/// Build a minimal FunctionDef node.
json make_funcdef(const std::string &name)
{
  json fd;
  fd["_type"] = "FunctionDef";
  fd["name"] = name;
  fd["args"] = json::object();
  fd["body"] = json::array();
  fd["decorator_list"] = json::array();
  return fd;
}

/// Build a minimal AsyncFunctionDef node.
json make_async_funcdef(const std::string &name)
{
  json fd;
  fd["_type"] = "AsyncFunctionDef";
  fd["name"] = name;
  fd["args"] = json::object();
  fd["body"] = json::array();
  fd["decorator_list"] = json::array();
  return fd;
}

/// Build a base reference ({"_type":"Name","id":"BaseName"}).
json make_base(const std::string &name)
{
  return json{{"_type", "Name"}, {"id", name}, {"ctx", {{"_type", "Load"}}}};
}

} // namespace

TEST_CASE(
  "method_exists_in_class_hierarchy cache and cycle handling",
  "[python-frontend][hierarchy]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  json call = make_dummy_call();
  contextt ctx;
  global_scope gs;

  SECTION("cyclic inheritance does not infinite-recurse")
  {
    // A -> B -> A (cycle)
    json body = json::array(
      {make_class("A", {make_funcdef("foo")}, {make_base("B")}),
       make_class("B", {}, {make_base("A")})});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    converter.current_block = &block;

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    // "foo" is in A directly
    REQUIRE(fce.method_exists_in_class_hierarchy("A", "foo") == true);
    // "foo" in B should find it through A (and not loop forever)
    REQUIRE(fce.method_exists_in_class_hierarchy("B", "foo") == true);
  }

  SECTION("provisional false is upgraded to true when method found")
  {
    json body = json::array(
      {make_class("Base", {make_funcdef("greet")}),
       make_class("Child", {}, {make_base("Base")})});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    converter.current_block = &block;

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    REQUIRE(fce.method_exists_in_class_hierarchy("Child", "greet") == true);

    // Cache should now hold true.
    auto cached =
      converter.get_function_call_cache().get_method_exists("Child::greet");
    REQUIRE(cached.has_value());
    REQUIRE(cached.value() == true);
  }

  SECTION("ignores nested function defs — only top-level methods count")
  {
    // Class with a FunctionDef inside another FunctionDef
    json nested_func = make_funcdef("inner_helper");
    json outer_func = make_funcdef("outer");
    outer_func["body"] = json::array({nested_func});
    json body = json::array({make_class("MyClass", {outer_func})});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    converter.current_block = &block;

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    // "outer" should be found (top-level)
    REQUIRE(fce.method_exists_in_class_hierarchy("MyClass", "outer") == true);
    // "inner_helper" should NOT be found (nested, not top-level)
    REQUIRE(
      fce.method_exists_in_class_hierarchy("MyClass", "inner_helper") ==
      false);
  }

  SECTION("malformed class members are skipped without crash")
  {
    json bad_member = json::object({{"_type", 42}}); // _type is not a string
    json body = json::array({make_class("BadClass", {bad_member})});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    converter.current_block = &block;

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    REQUIRE(
      fce.method_exists_in_class_hierarchy("BadClass", "anything") == false);
  }

  SECTION("malformed bases are skipped without crash")
  {
    // Base entry with no "id" field
    json bad_base = json::object({{"_type", "Name"}});
    json body = json::array(
      {make_class("Child", {}, json::array({bad_base}))});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    converter.current_block = &block;

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    REQUIRE(
      fce.method_exists_in_class_hierarchy("Child", "foo") == false);
  }

  SECTION("async method detection")
  {
    json body = json::array(
      {make_class("AsyncClass", {make_async_funcdef("async_method")})});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    converter.current_block = &block;

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    REQUIRE(
      fce.method_exists_in_class_hierarchy("AsyncClass", "async_method") ==
      true);
  }

  SECTION("empty class/method names return false")
  {
    json body = json::array({make_class("X", {make_funcdef("m")})});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    converter.current_block = &block;

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    REQUIRE(fce.method_exists_in_class_hierarchy("", "m") == false);
    REQUIRE(fce.method_exists_in_class_hierarchy("X", "") == false);
    REQUIRE(fce.method_exists_in_class_hierarchy("", "") == false);
  }
}

TEST_CASE(
  "find_possible_class_types with malformed AST",
  "[python-frontend][class-inference]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  json call = make_dummy_call();
  contextt ctx;
  global_scope gs;

  SECTION("null symbol returns empty vector")
  {
    json ast = make_dummy_ast();
    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    converter.current_block = &block;

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    auto result = fce.find_possible_class_types(nullptr);
    REQUIRE(result.empty());
  }
}

