#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <ld-frontend/property/property_encoder.h>
#include <ld-frontend/property/yaml_property_parser.h>
#include <util/context.h>
#include <util/std_expr.h>
#include <util/type.h>

#ifndef FIXTURE_DIR
#  define FIXTURE_DIR "."
#endif

static std::string fixture(const char *name)
{
  return std::string(FIXTURE_DIR) + "/" + name;
}

TEST_CASE(
  "mutual_exclusion with fewer than 2 variables rejected at parse",
  "[property_parser]")
{
  YamlPropertyParser parser;
  REQUIRE_THROWS_AS(
    parser.parse(fixture("mutual_exclusion_one_var.yaml")),
    LdPropertyParseError);
}

TEST_CASE("all valid property kinds parse without error", "[property_parser]")
{
  YamlPropertyParser parser;
  auto props = parser.parse(fixture("valid_props.yaml"));
  REQUIRE(props.size() == 5);
  REQUIRE(props[0].kind == PropertyKind::mutual_exclusion);
  REQUIRE(props[0].variables.size() == 2);
  REQUIRE(props[1].kind == PropertyKind::invariant);
  REQUIRE(!props[1].expression.empty());
  REQUIRE(props[2].kind == PropertyKind::absence);
  REQUIRE(props[3].kind == PropertyKind::reachability);
  REQUIRE(!props[3].justification.empty());
  REQUIRE(props[4].kind == PropertyKind::response);
  REQUIRE(props[4].max_scans == 5);
}

static void add_bool_sym(contextt &ctx, const std::string &name)
{
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
}

TEST_CASE(
  "parenthesised boolean expression encodes without error",
  "[property_encoder]")
{
  contextt ctx;
  add_bool_sym(ctx, "A");
  add_bool_sym(ctx, "B");
  add_bool_sym(ctx, "C");

  LdProperty p;
  p.id = "P_paren";
  p.kind = PropertyKind::invariant;
  p.expression = "(A || B) && C";

  property_encoder enc(ctx, "<test>");
  code_blockt blk;
  REQUIRE_NOTHROW(blk = enc.encode({p}));
  REQUIRE(blk.operands().size() == 1);
}

TEST_CASE("empty boolean expression throws", "[property_encoder]")
{
  contextt ctx;

  LdProperty p;
  p.id = "P_empty";
  p.kind = PropertyKind::invariant;
  p.expression = "   "; // whitespace-only

  property_encoder enc(ctx, "<test>");
  REQUIRE_THROWS_AS(enc.encode({p}), std::runtime_error);
}
