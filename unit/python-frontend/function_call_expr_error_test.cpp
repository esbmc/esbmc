// Catch2 header with main
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

// System includes before custom ones with ESBMC macros
#include <nlohmann/json.hpp>

#include <python-frontend/function_call_expr.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/function_call_cache.h>
#include <python-frontend/global_scope.h>
#include <python-frontend/symbol_id.h>
#include <util/config.h>
#include <util/context.h>
#include <util/python_types.h>

using json = nlohmann::json;

class python_converter_test_access
{
public:
  static void set_current_block(python_converter &converter, code_blockt &block)
  {
    converter.current_block = &block;
  }
};

class function_call_expr_test_access
{
public:
  static exprt generate_attribute_error(
    const function_call_expr &fce,
    const std::string &method_name,
    const std::vector<std::string> &possible_classes,
    const typet &expected_type = typet())
  {
    return fce.generate_attribute_error(
      method_name, possible_classes, expected_type);
  }

  static bool method_exists_in_class_hierarchy(
    const function_call_expr &fce,
    const std::string &class_name,
    const std::string &method_name)
  {
    return fce.method_exists_in_class_hierarchy(class_name, method_name);
  }

  static std::vector<std::string> find_possible_class_types(
    const function_call_expr &fce,
    const symbolt *symbol)
  {
    return fce.find_possible_class_types(symbol);
  }

  static std::string get_object_name(const function_call_expr &fce)
  {
    return fce.get_object_name();
  }
};

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
  ensure_config_initialized();

  json ast = make_dummy_ast();
  json call = make_dummy_call();

  contextt ctx;
  global_scope gs;
  python_converter converter(ctx, &ast, gs);

  // Provide a block so add_instruction() can push the assert.
  code_blockt block;
  python_converter_test_access::set_current_block(converter, block);

  symbol_id sid("test.py", "", "test_func");
  function_call_expr fce(sid, call, converter);

  SECTION("returns sideeffect/nondet with given type")
  {
    typet expected = signedbv_typet(32);
    exprt result = function_call_expr_test_access::generate_attribute_error(
      fce, "foo", {"MyClass"}, expected);

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

    exprt result = function_call_expr_test_access::generate_attribute_error(
      fce, "bar", {"A", "B"}, empty_typet());

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

    exprt result = function_call_expr_test_access::generate_attribute_error(
      fce, "baz", {}, typet());

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
  ensure_config_initialized();

  json call = make_dummy_call();
  contextt ctx;
  global_scope gs;

  SECTION("cyclic inheritance does not infinite-recurse")
  {
    // A -> B -> A (cycle)
    json body = json::array(
      {make_class(
         "A",
         json::array({make_funcdef("foo")}),
         json::array({make_base("B")})),
       make_class("B", json::array(), json::array({make_base("A")}))});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    python_converter_test_access::set_current_block(converter, block);

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    // "foo" is in A directly
    REQUIRE(
      function_call_expr_test_access::method_exists_in_class_hierarchy(
        fce, "A", "foo") == true);
    // "foo" in B should find it through A (and not loop forever)
    REQUIRE(
      function_call_expr_test_access::method_exists_in_class_hierarchy(
        fce, "B", "foo") == true);
  }

  SECTION("provisional false is upgraded to true when method found")
  {
    json body = json::array(
      {make_class("Base", json::array({make_funcdef("greet")})),
       make_class("Child", json::array(), json::array({make_base("Base")}))});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    python_converter_test_access::set_current_block(converter, block);

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    REQUIRE(
      function_call_expr_test_access::method_exists_in_class_hierarchy(
        fce, "Child", "greet") == true);

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
    json body = json::array({make_class("MyClass", json::array({outer_func}))});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    python_converter_test_access::set_current_block(converter, block);

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    // "outer" should be found (top-level)
    REQUIRE(
      function_call_expr_test_access::method_exists_in_class_hierarchy(
        fce, "MyClass", "outer") == true);
    // "inner_helper" should NOT be found (nested, not top-level)
    REQUIRE(
      function_call_expr_test_access::method_exists_in_class_hierarchy(
        fce, "MyClass", "inner_helper") == false);
  }

  SECTION("malformed class members are skipped without crash")
  {
    json bad_member = json::object({{"_type", 42}}); // _type is not a string
    json body =
      json::array({make_class("BadClass", json::array({bad_member}))});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    python_converter_test_access::set_current_block(converter, block);

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    REQUIRE(
      function_call_expr_test_access::method_exists_in_class_hierarchy(
        fce, "BadClass", "anything") == false);
  }

  SECTION("malformed bases are skipped without crash")
  {
    // Base entry with no "id" field
    json bad_base = json::object({{"_type", "Name"}});
    json body = json::array({make_class("Child", {}, json::array({bad_base}))});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    python_converter_test_access::set_current_block(converter, block);

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    REQUIRE(
      function_call_expr_test_access::method_exists_in_class_hierarchy(
        fce, "Child", "foo") == false);
  }

  SECTION("async method detection")
  {
    json body = json::array({make_class(
      "AsyncClass", json::array({make_async_funcdef("async_method")}))});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    python_converter_test_access::set_current_block(converter, block);

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    REQUIRE(
      function_call_expr_test_access::method_exists_in_class_hierarchy(
        fce, "AsyncClass", "async_method") == true);
  }

  SECTION("empty class/method names return false")
  {
    json body =
      json::array({make_class("X", json::array({make_funcdef("m")}))});
    json ast = make_ast_with_classes(body);

    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    python_converter_test_access::set_current_block(converter, block);

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    REQUIRE(
      function_call_expr_test_access::method_exists_in_class_hierarchy(
        fce, "", "m") == false);
    REQUIRE(
      function_call_expr_test_access::method_exists_in_class_hierarchy(
        fce, "X", "") == false);
    REQUIRE(
      function_call_expr_test_access::method_exists_in_class_hierarchy(
        fce, "", "") == false);
  }
}

TEST_CASE(
  "find_possible_class_types with malformed AST",
  "[python-frontend][class-inference]")
{
  ensure_config_initialized();

  json call = make_dummy_call();
  contextt ctx;
  global_scope gs;

  SECTION("null symbol returns empty vector")
  {
    json ast = make_dummy_ast();
    python_converter converter(ctx, &ast, gs);
    code_blockt block;
    python_converter_test_access::set_current_block(converter, block);

    symbol_id sid("test.py", "", "test_func");
    function_call_expr fce(sid, call, converter);

    auto result =
      function_call_expr_test_access::find_possible_class_types(fce, nullptr);
    REQUIRE(result.empty());
  }
}

TEST_CASE(
  "python_math dispatch classification uses cache consistently",
  "[python-frontend][math-dispatch-cache]")
{
  ensure_config_initialized();

  json ast = make_dummy_ast();
  contextt ctx;
  global_scope gs;
  python_converter converter(ctx, &ast, gs);

  auto &math = converter.get_math_handler();

  REQUIRE(math.is_math_dispatch_target_cached("math", "sin"));
  REQUIRE(math.is_math_dispatch_target_cached("", "__ESBMC_sin"));
  REQUIRE_FALSE(math.is_math_dispatch_target_cached("other", "sin"));

  auto attr_hit =
    converter.get_function_call_cache().get_math_dispatch_classification(
      "attr::math::sin");
  REQUIRE(attr_hit.has_value());
  REQUIRE(attr_hit.value());

  auto global_hit =
    converter.get_function_call_cache().get_math_dispatch_classification(
      "global::__ESBMC_sin");
  REQUIRE(global_hit.has_value());
  REQUIRE(global_hit.value());

  auto negative_hit =
    converter.get_function_call_cache().get_math_dispatch_classification(
      "attr::other::sin");
  REQUIRE(negative_hit.has_value());
  REQUIRE_FALSE(negative_hit.value());
}

TEST_CASE(
  "python_math dispatch cache keys do not collide between global and attribute "
  "calls",
  "[python-frontend][math-dispatch-cache]")
{
  ensure_config_initialized();

  json ast = make_dummy_ast();
  contextt ctx;
  global_scope gs;
  python_converter converter(ctx, &ast, gs);

  auto &math = converter.get_math_handler();

  // Global wrapper-style call name that includes "::" in function name.
  REQUIRE_FALSE(math.is_math_dispatch_target_cached("", "math::sin"));
  // Attribute-style normal math call.
  REQUIRE(math.is_math_dispatch_target_cached("math", "sin"));

  auto global_entry =
    converter.get_function_call_cache().get_math_dispatch_classification(
      "global::math::sin");
  REQUIRE(global_entry.has_value());
  REQUIRE_FALSE(global_entry.value());

  auto attr_entry =
    converter.get_function_call_cache().get_math_dispatch_classification(
      "attr::math::sin");
  REQUIRE(attr_entry.has_value());
  REQUIRE(attr_entry.value());

  // Empty function names are ignored and must not pollute cache.
  REQUIRE_FALSE(math.is_math_dispatch_target_cached("math", ""));
  REQUIRE_FALSE(converter.get_function_call_cache()
                  .get_math_dispatch_classification("attr::math::")
                  .has_value());
}

TEST_CASE(
  "get_object_name is resilient to malformed attribute call nodes",
  "[python-frontend][ast-robustness]")
{
  ensure_config_initialized();

  json ast = make_dummy_ast();
  // Missing "value" under func Attribute on purpose.
  json bad_call = json::parse(R"json({
    "_type": "Call",
    "func": { "_type": "Attribute", "attr": "sin" },
    "args": [],
    "lineno": 1,
    "col_offset": 0
  })json");

  contextt ctx;
  global_scope gs;
  python_converter converter(ctx, &ast, gs);

  symbol_id sid("test.py", "", "test_func");
  function_call_expr fce(sid, bad_call, converter);

  std::string obj_name;
  REQUIRE_NOTHROW(
    obj_name = function_call_expr_test_access::get_object_name(fce));
  REQUIRE(obj_name.empty());
}

TEST_CASE(
  "get_object_name handles malformed attribute members without throwing",
  "[python-frontend][ast-robustness]")
{
  ensure_config_initialized();

  json ast = make_dummy_ast();
  // Attribute call with malformed members:
  // - attr is non-string
  // - nested Name id is non-string
  json bad_call = json::parse(R"json({
    "_type": "Call",
    "func": {
      "_type": "Attribute",
      "attr": 42,
      "value": {
        "_type": "Name",
        "id": 123
      }
    },
    "args": [],
    "lineno": 1,
    "col_offset": 0
  })json");

  contextt ctx;
  global_scope gs;
  python_converter converter(ctx, &ast, gs);

  symbol_id sid("test.py", "", "test_func");
  function_call_expr fce(sid, bad_call, converter);

  std::string obj_name;
  REQUIRE_NOTHROW(
    obj_name = function_call_expr_test_access::get_object_name(fce));
  REQUIRE(obj_name.empty());
}
