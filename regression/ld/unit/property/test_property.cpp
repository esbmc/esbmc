// Unit tests for YamlPropertyParser and property_encoder.
// Standalone; return 0 on pass.

#include <ld-frontend/property/yaml_property_parser.h>
#include <ld-frontend/property/property_encoder.h>
#include <cassert>
#include <iostream>
#include <stdexcept>

#ifndef FIXTURE_DIR
#  define FIXTURE_DIR "."
#endif

static std::string fixture(const char *name)
{
  return std::string(FIXTURE_DIR) + "/" + name;
}

// Fix 10: mutual_exclusion with fewer than 2 variables must throw.
static void test_mutual_exclusion_one_var_rejected()
{
  YamlPropertyParser parser;
  bool threw = false;
  try
  {
    parser.parse(fixture("mutual_exclusion_one_var.yaml"));
  }
  catch (const LdPropertyParseError &)
  {
    threw = true;
  }
  assert(threw);
  std::cout << "PASS: test_mutual_exclusion_one_var_rejected\n";
}

// All valid property kinds must parse without error.
static void test_valid_properties_parse()
{
  YamlPropertyParser parser;
  auto props = parser.parse(fixture("valid_props.yaml"));
  assert(props.size() == 5);
  assert(props[0].kind == PropertyKind::mutual_exclusion);
  assert(props[0].variables.size() == 2);
  assert(props[1].kind == PropertyKind::invariant);
  assert(!props[1].expression.empty());
  assert(props[2].kind == PropertyKind::absence);
  assert(props[3].kind == PropertyKind::reachability);
  assert(!props[3].justification.empty());
  assert(props[4].kind == PropertyKind::response);
  assert(props[4].max_scans == 5);
  std::cout << "PASS: test_valid_properties_parse\n";
}

// Fix 9: parse_bool_expr must handle parenthesised sub-expressions.
// We test via the property encoder using a mock context+property.
// The simplest approach: exercise the parser path directly by constructing
// an invariant property and encoding it against a context with the variables.
static void test_parse_bool_expr_parentheses()
{
  // Build a minimal context with BOOL symbols A, B, C.
  contextt ctx;
  auto add_bool = [&](const std::string &name) {
    symbolt sym;
    sym.id = "ld::" + name;
    sym.name = name;
    sym.module = "ld";
    sym.mode = "LD";
    sym.set_type(typet("bool"));
    sym.set_value(false_exprt());
    sym.lvalue = true;
    sym.static_lifetime = true;
    ctx.move_symbol_to_context(sym);
  };
  add_bool("A");
  add_bool("B");
  add_bool("C");

  LdProperty p;
  p.id = "P_paren";
  p.kind = PropertyKind::invariant;
  p.expression = "(A || B) && C";

  property_encoder enc(ctx, "<test>");
  bool threw = false;
  try
  {
    // encode() calls parse_bool_expr internally.
    std::vector<LdProperty> props = {p};
    code_blockt blk = enc.encode(props);
    // Should produce one assertion (invariant).
    assert(blk.operands().size() == 1);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Unexpected exception: " << e.what() << "\n";
    threw = true;
  }
  assert(!threw);
  std::cout << "PASS: test_parse_bool_expr_parentheses\n";
}

// Fix 9: empty expression must throw.
static void test_parse_bool_expr_empty_throws()
{
  contextt ctx;
  LdProperty p;
  p.id = "P_empty";
  p.kind = PropertyKind::invariant;
  p.expression = "   "; // whitespace-only → empty after trim

  property_encoder enc(ctx, "<test>");
  bool threw = false;
  try
  {
    std::vector<LdProperty> props = {p};
    enc.encode(props);
  }
  catch (const std::runtime_error &)
  {
    threw = true;
  }
  assert(threw);
  std::cout << "PASS: test_parse_bool_expr_empty_throws\n";
}

int main()
{
  test_mutual_exclusion_one_var_rejected();
  test_valid_properties_parse();
  test_parse_bool_expr_parentheses();
  test_parse_bool_expr_empty_throws();
  std::cout << "All property tests passed.\n";
  return 0;
}
