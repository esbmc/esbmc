#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/python_adjust.h>
#include <util/context.h>
#include <util/symbol.h>
#include <util/c_types.h>
#include <irep2/irep2_utils.h>
#include <string>
#include <vector>

// Dead-but-tested gate for the V.1k (b) IREP2-native Python adjuster
// (docs/irep2-migration.md, "V.1k (b)-adjuster", phases B.0/B.1).
//
// B.0 pins the inert baseline: a non-member/index expression is byte-identical
// after a walk. B.1 adds the resolution behaviour: a transient symbol_type2t
// member2t/index2t source is followed to its struct/array (and a pointer source
// is auto-dereferenced), re-establishing the strong source invariant the
// construction assert was relaxed for. B.2 wires the pass into
// python_languaget::typecheck behind --python-irep2-adjust (default off); these
// unit tests exercise the behaviour directly — the "machinery-first,
// prove-inert, wire-later" gate used for the V.4.0 structured-CF kinds
// (esbmc/esbmc#5265).

namespace
{
// (x + 1) == 0, a small nested IREP2 tree with leaves and interior nodes.
expr2tc make_sample_expr()
{
  const type2tc int_t = get_int32_type();
  const expr2tc x = symbol2tc(int_t, "x");
  const expr2tc sum = add2tc(int_t, x, gen_one(int_t));
  return equality2tc(sum, gen_zero(int_t));
}

// Register a type symbol `tag-<name>` = struct { <field> : int; } so that
// ns.follow(symbol_type2t("tag-<name>")) yields the resolved struct.
type2tc add_struct_type(
  contextt &ctx,
  const std::string &name,
  const std::string &field)
{
  const type2tc struct_t = struct_type2tc(
    std::vector<type2tc>{get_int32_type()},
    std::vector<irep_idt>{field},
    std::vector<irep_idt>{field},
    name,
    false);

  symbolt type_sym;
  type_sym.id = "tag-" + name;
  type_sym.name = "tag-" + name;
  type_sym.mode = "Python";
  type_sym.is_type = true;
  type_sym.set_type(struct_t);
  ctx.add(type_sym);

  return struct_t;
}
} // namespace

TEST_CASE(
  "python_adjust B.0 leaves an expression byte-identical",
  "[python-adjust]")
{
  const expr2tc original = make_sample_expr();
  expr2tc walked = original;

  contextt ctx;
  python_adjust adjuster(ctx);
  adjuster.adjust_expr(walked);

  REQUIRE(walked == original);
}

TEST_CASE(
  "python_adjust B.0 adjust() leaves a symbol's IREP2 value unchanged",
  "[python-adjust]")
{
  const expr2tc value = make_sample_expr();

  symbolt symbol;
  symbol.id = "py_adjust_test_sym";
  symbol.name = "py_adjust_test_sym";
  symbol.mode = "Python";
  symbol.set_type(get_int32_type());
  symbol.set_value(value);

  contextt ctx;
  ctx.add(symbol);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());

  const symbolt *out = ctx.find_symbol("py_adjust_test_sym");
  REQUIRE(out != nullptr);
  REQUIRE(out->get_value2() == value);
}

TEST_CASE("python_adjust B.0 ignores a nil-valued symbol", "[python-adjust]")
{
  symbolt symbol;
  symbol.id = "py_adjust_nil_sym";
  symbol.name = "py_adjust_nil_sym";
  symbol.mode = "Python";
  symbol.set_type(get_int32_type());

  contextt ctx;
  ctx.add(symbol);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());
}

TEST_CASE(
  "python_adjust B.1 follows a direct symbol_type member source to its struct",
  "[python-adjust]")
{
  contextt ctx;
  const type2tc struct_t = add_struct_type(ctx, "Foo", "x");

  // member2t over a source whose type is the transient symbol_type2t("tag-Foo")
  // (permitted by the relaxed construction assert).
  const expr2tc source = symbol2tc(symbol_type2tc("tag-Foo"), "obj");
  expr2tc member = member2tc(get_int32_type(), source, "x");

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(member);

  REQUIRE(is_member2t(member));
  REQUIRE(to_member2t(member).source_value->type == struct_t);
}

TEST_CASE(
  "python_adjust B.1 follows a symbol_type index source to its array",
  "[python-adjust]")
{
  // The index2t arm of resolve_source: a source typed as a transient
  // symbol_type2t("tag-Buf") is followed to its registered array type.
  contextt ctx;
  const type2tc array_t = array_type2tc(get_int32_type(), gen_ulong(4), false);

  symbolt type_sym;
  type_sym.id = "tag-Buf";
  type_sym.name = "tag-Buf";
  type_sym.mode = "Python";
  type_sym.is_type = true;
  type_sym.set_type(array_t);
  ctx.add(type_sym);

  const expr2tc source = symbol2tc(symbol_type2tc("tag-Buf"), "buf");
  expr2tc index = index2tc(get_int32_type(), source, gen_ulong(0));

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(index);

  REQUIRE(is_index2t(index));
  REQUIRE(to_index2t(index).source_value->type == array_t);
}

TEST_CASE(
  "python_adjust B.1 follows a symbol_type on a dereference member source",
  "[python-adjust]")
{
  contextt ctx;
  const type2tc struct_t = add_struct_type(ctx, "Bar", "y");

  // A Python class-instance access `obj.y`: obj is a pointer→symbol_type, so
  // the member source is *obj — a dereference2t whose result type is the
  // symbol_type pointee (a member2t cannot be built over the raw pointer).
  const expr2tc ptr =
    symbol2tc(pointer_type2tc(symbol_type2tc("tag-Bar")), "obj");
  const expr2tc deref = dereference2tc(symbol_type2tc("tag-Bar"), ptr);
  expr2tc member = member2tc(get_int32_type(), deref, "y");

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(member);

  REQUIRE(is_member2t(member));
  const expr2tc &resolved_source = to_member2t(member).source_value;
  REQUIRE(is_dereference2t(resolved_source));
  REQUIRE(resolved_source->type == struct_t);
}

TEST_CASE(
  "python_adjust B.1 resolves a nested member chain inner-to-outer",
  "[python-adjust]")
{
  // self.b : B, and B.a : int, with B a registered struct. The outer member
  // self.b.a has, as its source, the inner member self.b whose own source is a
  // symbol_type2t("tag-Outer"). Both levels must resolve.
  contextt ctx;
  const type2tc b_struct = add_struct_type(ctx, "B", "a");
  // Outer struct has a member `b` of (symbol) type tag-B.
  const type2tc outer_struct = struct_type2tc(
    std::vector<type2tc>{symbol_type2tc("tag-B")},
    std::vector<irep_idt>{"b"},
    std::vector<irep_idt>{"b"},
    "Outer",
    false);
  symbolt outer_sym;
  outer_sym.id = "tag-Outer";
  outer_sym.name = "tag-Outer";
  outer_sym.mode = "Python";
  outer_sym.is_type = true;
  outer_sym.set_type(outer_struct);
  ctx.add(outer_sym);

  const expr2tc self = symbol2tc(symbol_type2tc("tag-Outer"), "self");
  const expr2tc self_b = member2tc(symbol_type2tc("tag-B"), self, "b");
  expr2tc self_b_a = member2tc(get_int32_type(), self_b, "a");

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(self_b_a);

  // Outer member's source (self.b) resolved to struct B; its inner source
  // (self) resolved to struct Outer.
  REQUIRE(is_member2t(self_b_a));
  const expr2tc &outer_source = to_member2t(self_b_a).source_value;
  REQUIRE(outer_source->type == b_struct);
  REQUIRE(is_member2t(outer_source));
  REQUIRE(to_member2t(outer_source).source_value->type == outer_struct);
}

TEST_CASE(
  "python_adjust B.1 leaves an already-resolved struct source untouched",
  "[python-adjust]")
{
  // A member2t whose source is already a resolved struct must be left alone
  // (idempotence: re-running the adjuster is a no-op on resolved nodes).
  contextt ctx;
  const type2tc struct_t = add_struct_type(ctx, "Baz", "z");

  const expr2tc source = symbol2tc(struct_t, "obj");
  expr2tc member = member2tc(get_int32_type(), source, "z");
  const expr2tc original = member;

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(member);

  REQUIRE(member == original);
}
