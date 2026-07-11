#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/python_adjust.h>
#include <util/context.h>
#include <util/symbol.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/migrate.h>
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

// Register a type symbol `tag-<name>` = int[] so that
// ns.follow(symbol_type2t("tag-<name>")) yields the resolved array.
type2tc add_array_type(contextt &ctx, const std::string &name)
{
  const type2tc array_t =
    array_type2tc(get_int32_type(), expr2tc(), /*infinite=*/true);

  symbolt type_sym;
  type_sym.id = "tag-" + name;
  type_sym.name = "tag-" + name;
  type_sym.mode = "Python";
  type_sym.is_type = true;
  type_sym.set_type(array_t);
  ctx.add(type_sym);

  return array_t;
}

// A nullary `void()` code type — the discriminator adjust() walks on.
type2tc code_symbol_type()
{
  return code_type2tc(
    std::vector<type2tc>{},
    get_empty_type(),
    std::vector<irep_idt>{},
    /*ellipsis=*/false);
}

// Register a code symbol `id` whose IREP2 value (body) is `body`, so adjust()
// reaches it (only code symbols are walked, python_adjust.cpp).
void add_code_symbol(contextt &ctx, const std::string &id, const expr2tc &body)
{
  symbolt symbol;
  symbol.id = id;
  symbol.name = id;
  symbol.mode = "Python";
  symbol.set_type(code_symbol_type());
  symbol.set_value(body);
  ctx.add(symbol);
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
  // The index2t arm: a subscript `arr[0]` whose source carries the transient
  // symbol_type2t("tag-IntArr") is followed to the resolved array type, the
  // is_array_type arm of is_resolved_aggregate the member tests never reach.
  contextt ctx;
  const type2tc array_t = add_array_type(ctx, "IntArr");

  const expr2tc source = symbol2tc(symbol_type2tc("tag-IntArr"), "arr");
  const expr2tc zero = gen_zero(get_uint32_type());
  expr2tc index = index2tc(get_int32_type(), source, zero);

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

TEST_CASE(
  "python_adjust B.1 leaves a member untouched when the symbol_type follows to "
  "a non-aggregate",
  "[python-adjust]")
{
  // resolve_source only retypes a source whose followed type is a concrete
  // aggregate. A symbol_type that follows to a scalar (here tag-Scalar -> int)
  // is not an aggregate, so the member is left transient for the downstream
  // construction assert to reject — the pass must not retype it to a scalar.
  contextt ctx;
  symbolt scalar_type_sym;
  scalar_type_sym.id = scalar_type_sym.name = "tag-Scalar";
  scalar_type_sym.mode = "Python";
  scalar_type_sym.is_type = true;
  scalar_type_sym.set_type(get_int32_type());
  ctx.add(scalar_type_sym);

  const expr2tc source = symbol2tc(symbol_type2tc("tag-Scalar"), "obj");
  expr2tc member = member2tc(get_int32_type(), source, "x");
  const expr2tc original = member;

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(member);

  REQUIRE(member == original);
}

TEST_CASE(
  "python_adjust adjust() resolves a member source inside a code body and "
  "writes it back",
  "[python-adjust]")
{
  // adjust() walks code symbols, resolving transient member sources in the
  // body and writing the symbol back. The body wraps `obj.x` (obj : tag-Foo);
  // after adjust() the stored body's member source is the resolved struct.
  contextt ctx;
  const type2tc struct_t = add_struct_type(ctx, "Foo", "x");

  const expr2tc source = symbol2tc(symbol_type2tc("tag-Foo"), "obj");
  const expr2tc member = member2tc(get_int32_type(), source, "x");
  const expr2tc body =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(member)});
  add_code_symbol(ctx, "py_adjust_code_sym", body);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());

  const symbolt *out = ctx.find_symbol("py_adjust_code_sym");
  REQUIRE(out != nullptr);
  const expr2tc &stored = out->get_value2();
  REQUIRE(stored != body);
  const expr2tc &stmt = to_code_block2t(stored).operands.at(0);
  const expr2tc &resolved_member = to_code_expression2t(stmt).operand;
  REQUIRE(to_member2t(resolved_member).source_value->type == struct_t);
}

TEST_CASE(
  "python_adjust adjust() leaves a code body with no transient source "
  "unchanged",
  "[python-adjust]")
{
  // When the body has nothing to resolve (its member source is already a
  // resolved struct), adjust() must not rewrite the symbol — the value stays
  // byte-identical so the legacy cache and goto-convert body are untouched.
  contextt ctx;
  const type2tc struct_t = add_struct_type(ctx, "Qux", "w");

  const expr2tc source = symbol2tc(struct_t, "obj");
  const expr2tc member = member2tc(get_int32_type(), source, "w");
  const expr2tc body =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(member)});
  add_code_symbol(ctx, "py_adjust_noop_sym", body);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());

  const symbolt *out = ctx.find_symbol("py_adjust_noop_sym");
  REQUIRE(out != nullptr);
  REQUIRE(out->get_value2() == body);
}

TEST_CASE(
  "python_adjust adjust() skips a code symbol with a nil value",
  "[python-adjust]")
{
  // A code symbol with no body (nil value) is skipped before adjust_expr; the
  // int-typed nil test bails earlier at the is_code_type guard, so this pins
  // the is_nil_expr guard on the code path specifically.
  symbolt symbol;
  symbol.id = symbol.name = "py_adjust_code_nil_sym";
  symbol.mode = "Python";
  symbol.set_type(code_symbol_type());

  contextt ctx;
  ctx.add(symbol);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());

  // The body must remain nil — adjust() neither materialises nor rewrites it.
  REQUIRE(is_nil_expr(ctx.find_symbol("py_adjust_code_nil_sym")->get_value2()));
}

TEST_CASE(
  "python_adjust B.4 adjust() flags an unresolved member source post-adjust",
  "[python-adjust]")
{
  // A code body reads `obj.x` where obj's tag follows to a scalar (tag-Scalar ->
  // int): resolve_source must NOT retype the source to a non-aggregate, so the
  // transient symbol_type2t source survives adjust. The post-adjust
  // strong-invariant check catches the survivor and adjust() returns true
  // (error) — the negative of the resolved-struct case above, which returns
  // false. This is the B.5-era mis-resolution the safety net exists to catch.
  contextt ctx;
  symbolt scalar_type_sym;
  scalar_type_sym.id = scalar_type_sym.name = "tag-Scalar";
  scalar_type_sym.mode = "Python";
  scalar_type_sym.is_type = true;
  scalar_type_sym.set_type(get_int32_type());
  ctx.add(scalar_type_sym);

  const expr2tc source = symbol2tc(symbol_type2tc("tag-Scalar"), "obj");
  const expr2tc member = member2tc(get_int32_type(), source, "x");
  const expr2tc body =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(member)});
  add_code_symbol(ctx, "py_adjust_unresolved_sym", body);

  python_adjust adjuster(ctx);
  REQUIRE(adjuster.adjust());
}

TEST_CASE(
  "python_adjust B.4 adjust() flags an unresolved constant_struct type "
  "post-adjust",
  "[python-adjust]")
{
  // constant_struct2t is the third relaxed construction assert (irep2_expr.h):
  // its own type may be a transient by-name symbol_type2t. S2 resolves a
  // *registered* tag (tests below); this literal's tag-Rec is unregistered,
  // so it must survive resolution and be caught by the post-adjust invariant:
  // adjust() returns true (error).
  contextt ctx;
  const expr2tc lit = constant_struct2tc(
    symbol_type2tc("tag-Rec"),
    std::vector<expr2tc>{gen_zero(get_int32_type())});
  const expr2tc body =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(lit)});
  add_code_symbol(ctx, "py_adjust_struct_lit_sym", body);

  python_adjust adjuster(ctx);
  REQUIRE(adjuster.adjust());
}

// --- S1: IREP2-native adjust_type (V.1k/B.5 combined-milestone step S1) ---
//
// The padding tests need the target data model set (add_padding's alignment
// arithmetic reads config.ansi_c); set it explicitly so the tests do not
// depend on invocation order.

TEST_CASE(
  "python_adjust S1 adjust_type expands a macro symbol type, chained",
  "[python-adjust]")
{
  // py_alias2 -> py_alias1 -> int32: the macro arm must recurse until the
  // type is concrete, mirroring clang_c_adjust::adjust_type's is_macro loop.
  contextt ctx;
  symbolt alias1;
  alias1.id = alias1.name = "py_alias1";
  alias1.mode = "Python";
  alias1.is_type = true;
  alias1.is_macro = true;
  alias1.set_type(get_int32_type());
  ctx.add(alias1);

  symbolt alias2;
  alias2.id = alias2.name = "py_alias2";
  alias2.mode = "Python";
  alias2.is_type = true;
  alias2.is_macro = true;
  alias2.set_type(symbol_type2tc("py_alias1"));
  ctx.add(alias2);

  type2tc t = symbol_type2tc("py_alias2");
  python_adjust adjuster(ctx);
  adjuster.adjust_type(t);

  REQUIRE(t == get_int32_type());
}

TEST_CASE(
  "python_adjust S1 adjust_type leaves a non-macro tag by-name",
  "[python-adjust]")
{
  // The RV-adj6 parity subtlety: the legacy pass only overwrites is_macro
  // symbol types; a struct tag reference stays by-name and is followed at
  // consumption time. Eagerly resolving it here would diverge.
  contextt ctx;
  add_struct_type(ctx, "Foo", "x");

  type2tc t = symbol_type2tc("tag-Foo");
  const type2tc original = t;
  python_adjust adjuster(ctx);
  adjuster.adjust_type(t);

  REQUIRE(t == original);
}

TEST_CASE(
  "python_adjust S1 adjust_type completes an array element type",
  "[python-adjust]")
{
  // Array arm: the element type recurses through adjust_type (here a macro
  // alias expands to int32) while the size expression is preserved.
  contextt ctx;
  symbolt alias;
  alias.id = alias.name = "py_elem_alias";
  alias.mode = "Python";
  alias.is_type = true;
  alias.is_macro = true;
  alias.set_type(get_int32_type());
  ctx.add(alias);

  const expr2tc size = gen_long(get_uint64_type(), 4);
  type2tc t = array_type2tc(symbol_type2tc("py_elem_alias"), size, false);
  python_adjust adjuster(ctx);
  adjuster.adjust_type(t);

  REQUIRE(is_array_type(t));
  REQUIRE(to_array_type(t).subtype == get_int32_type());
  REQUIRE(to_array_type(t).array_size == size);
}

TEST_CASE(
  "python_adjust S1 adjust_type pads a misaligned struct like add_padding",
  "[python-adjust]")
{
  // struct { int8 c; int32 i; } needs 3 bytes of padding after `c` on any
  // data model with 4-byte int alignment. The pass must reproduce the legacy
  // add_padding layout exactly (RV-adj5): one unsignedbv(24) padding member
  // between the fields, total size a multiple of the alignment.
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  type2tc t = struct_type2tc(
    std::vector<type2tc>{signedbv_type2tc(8), get_int32_type()},
    std::vector<irep_idt>{"c", "i"},
    std::vector<irep_idt>{"c", "i"},
    "Packed2",
    false);

  python_adjust adjuster(ctx);
  adjuster.adjust_type(t);

  REQUIRE(is_struct_type(t));
  const struct_type2t &st = to_struct_type(t);
  REQUIRE(st.members.size() == 3);
  REQUIRE(st.member_names[0] == "c");
  REQUIRE(st.member_names[2] == "i");
  REQUIRE(is_unsignedbv_type(st.members[1]));
  REQUIRE(st.members[1]->get_width() == 24);

  // Idempotence: a second pass over the already-padded struct is a no-op —
  // the inertness guarantee for the live (post-clang_cpp_adjust) pipeline.
  const type2tc padded = t;
  adjuster.adjust_type(t);
  REQUIRE(t == padded);
}

TEST_CASE(
  "python_adjust S1 adjust_type pads a union like add_padding, idempotently",
  "[python-adjust]")
{
  // union { int8 b[5]; int32 i; } is 5 bytes wide with 4-byte alignment, so
  // add_padding appends a trailing `$pad` member widening it to 8 bytes. The
  // second pass exercises the `$pad` case of the #is_padding re-derivation:
  // the union arm must be as idempotent as the struct one.
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  const type2tc bytes5 =
    array_type2tc(signedbv_type2tc(8), gen_long(get_uint64_type(), 5), false);
  type2tc t = union_type2tc(
    std::vector<type2tc>{bytes5, get_int32_type()},
    std::vector<irep_idt>{"b", "i"},
    std::vector<irep_idt>{"b", "i"},
    "U1",
    false);

  python_adjust adjuster(ctx);
  adjuster.adjust_type(t);

  REQUIRE(is_union_type(t));
  const union_type2t &ut = to_union_type(t);
  REQUIRE(ut.members.size() == 3);
  REQUIRE(ut.member_names[2] == "$pad");
  REQUIRE(is_unsignedbv_type(ut.members[2]));

  const type2tc padded = t;
  adjuster.adjust_type(t);
  REQUIRE(t == padded);
}

TEST_CASE(
  "python_adjust S1 adjust_type completes a struct member's macro type",
  "[python-adjust]")
{
  // Aggregate-member recursion: a member whose type is a macro alias expands
  // before padding runs, so the layout is computed over concrete types.
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  symbolt alias;
  alias.id = alias.name = "py_field_alias";
  alias.mode = "Python";
  alias.is_type = true;
  alias.is_macro = true;
  alias.set_type(get_int32_type());
  ctx.add(alias);

  type2tc t = struct_type2tc(
    std::vector<type2tc>{symbol_type2tc("py_field_alias")},
    std::vector<irep_idt>{"v"},
    std::vector<irep_idt>{"v"},
    "Rec1",
    false);

  python_adjust adjuster(ctx);
  adjuster.adjust_type(t);

  REQUIRE(is_struct_type(t));
  REQUIRE(to_struct_type(t).members.at(0) == get_int32_type());
}

TEST_CASE(
  "python_adjust S2 completes a by-name struct literal to its resolved type",
  "[python-adjust]")
{
  // The OM exception-literal shape (raise IndexError(...)): a
  // constant_struct2t typed by-name whose operands are already complete. S2
  // retypes it to the followed struct — eagerly, unlike the legacy
  // adjust_struct (RV-adj6) — and the exit invariant stops flagging it.
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  const type2tc struct_t = add_struct_type(ctx, "Exc", "id");

  const expr2tc lit = constant_struct2tc(
    symbol_type2tc("tag-Exc"),
    std::vector<expr2tc>{gen_zero(get_int32_type())});
  const expr2tc body =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(lit)});
  add_code_symbol(ctx, "py_adjust_s2_lit_sym", body);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());

  const expr2tc &stored = ctx.find_symbol("py_adjust_s2_lit_sym")->get_value2();
  const expr2tc &stmt = to_code_block2t(stored).operands.at(0);
  const expr2tc &resolved = to_code_expression2t(stmt).operand;
  REQUIRE(is_constant_struct2t(resolved));
  REQUIRE(resolved->type == struct_t);
  REQUIRE(to_constant_struct2t(resolved).datatype_members.size() == 1);
}

TEST_CASE(
  "python_adjust S2 inserts missing padding operands into a struct literal",
  "[python-adjust]")
{
  // A converter-built literal carries only the value operands; the followed
  // struct pads to { c, anon_pad$, i } (S1), so S2 must insert a zero pad
  // operand at position 1, mirroring the legacy adjust_struct insertion.
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  const type2tc unpadded = struct_type2tc(
    std::vector<type2tc>{signedbv_type2tc(8), get_int32_type()},
    std::vector<irep_idt>{"c", "i"},
    std::vector<irep_idt>{"c", "i"},
    "PadLit",
    false);
  symbolt type_sym;
  type_sym.id = type_sym.name = "tag-PadLit";
  type_sym.mode = "Python";
  type_sym.is_type = true;
  type_sym.set_type(unpadded);
  ctx.add(type_sym);

  expr2tc lit = constant_struct2tc(
    symbol_type2tc("tag-PadLit"),
    std::vector<expr2tc>{
      gen_zero(signedbv_type2tc(8)), gen_zero(get_int32_type())});

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(lit);

  REQUIRE(is_constant_struct2t(lit));
  REQUIRE(is_struct_type(lit->type));
  const struct_type2t &st = to_struct_type(lit->type);
  const constant_struct2t &cs = to_constant_struct2t(lit);
  REQUIRE(st.members.size() == 3);
  REQUIRE(cs.datatype_members.size() == 3);
  REQUIRE(is_unsignedbv_type(cs.datatype_members[1]->type));
  REQUIRE(cs.datatype_members[1]->type->get_width() == 24);
  REQUIRE(cs.datatype_members[2]->type == get_int32_type());

  // Idempotence: the completed literal is stable under a second walk.
  const expr2tc done = lit;
  adjuster.adjust_expr(lit);
  REQUIRE(lit == done);
}

TEST_CASE(
  "python_adjust S2 completes a macro-tag struct literal",
  "[python-adjust]")
{
  // A literal typed by a macro alias of a struct: the leading type-completion
  // block must NOT retype it bare (that would skip the padding-operand
  // insertion); the S2 arm owns the shape end-to-end and pads it.
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  const type2tc unpadded = struct_type2tc(
    std::vector<type2tc>{signedbv_type2tc(8), get_int32_type()},
    std::vector<irep_idt>{"c", "i"},
    std::vector<irep_idt>{"c", "i"},
    "MacroLit",
    false);
  symbolt alias;
  alias.id = alias.name = "py_struct_alias";
  alias.mode = "Python";
  alias.is_type = true;
  alias.is_macro = true;
  alias.set_type(unpadded);
  ctx.add(alias);

  expr2tc lit = constant_struct2tc(
    symbol_type2tc("py_struct_alias"),
    std::vector<expr2tc>{
      gen_zero(signedbv_type2tc(8)), gen_zero(get_int32_type())});

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(lit);

  REQUIRE(is_constant_struct2t(lit));
  REQUIRE(is_struct_type(lit->type));
  REQUIRE(to_struct_type(lit->type).members.size() == 3);
  REQUIRE(to_constant_struct2t(lit).datatype_members.size() == 3);
}

TEST_CASE(
  "python_adjust S2 leaves a structurally short literal for the invariant",
  "[python-adjust]")
{
  // A literal missing a *value* operand (only `c`, no `i`): pad insertion
  // cannot make the counts match, so the literal must stay by-name and be
  // flagged — never rebuilt with misaligned operands.
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  const type2tc unpadded = struct_type2tc(
    std::vector<type2tc>{signedbv_type2tc(8), get_int32_type()},
    std::vector<irep_idt>{"c", "i"},
    std::vector<irep_idt>{"c", "i"},
    "ShortLit",
    false);
  symbolt type_sym;
  type_sym.id = type_sym.name = "tag-ShortLit";
  type_sym.mode = "Python";
  type_sym.is_type = true;
  type_sym.set_type(unpadded);
  ctx.add(type_sym);

  const expr2tc lit = constant_struct2tc(
    symbol_type2tc("tag-ShortLit"),
    std::vector<expr2tc>{gen_zero(signedbv_type2tc(8))});
  const expr2tc body =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(lit)});
  add_code_symbol(ctx, "py_adjust_s2_short_sym", body);

  python_adjust adjuster(ctx);
  REQUIRE(adjuster.adjust());
}

TEST_CASE(
  "python_adjust invariant flags a resolved literal with an operand-count "
  "mismatch",
  "[python-adjust]")
{
  // The durable half of the S2 safety net: even a literal that already
  // carries a resolved struct type is flagged when its operand count
  // disagrees with the component list (constant_struct2t's constructor
  // asserts only the type kind, so this would otherwise reach symex).
  contextt ctx;
  const type2tc struct_t = add_struct_type(ctx, "Mismatch", "x");

  const expr2tc lit = constant_struct2tc(struct_t, std::vector<expr2tc>{});
  const expr2tc body =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(lit)});
  add_code_symbol(ctx, "py_adjust_s2_mismatch_sym", body);

  python_adjust adjuster(ctx);
  REQUIRE(adjuster.adjust());
}

TEST_CASE(
  "python_adjust S2 leaves a scalar-tag struct literal for the invariant",
  "[python-adjust]")
{
  // tag-Scalar follows to int (not a struct): S2 must not retype the literal;
  // the survivor is flagged by the exit invariant instead.
  contextt ctx;
  symbolt scalar_type_sym;
  scalar_type_sym.id = scalar_type_sym.name = "tag-Scalar";
  scalar_type_sym.mode = "Python";
  scalar_type_sym.is_type = true;
  scalar_type_sym.set_type(get_int32_type());
  ctx.add(scalar_type_sym);

  const expr2tc lit = constant_struct2tc(
    symbol_type2tc("tag-Scalar"),
    std::vector<expr2tc>{gen_zero(get_int32_type())});
  const expr2tc body =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(lit)});
  add_code_symbol(ctx, "py_adjust_s2_scalar_sym", body);

  python_adjust adjuster(ctx);
  REQUIRE(adjuster.adjust());
}

TEST_CASE(
  "python_adjust pre-pass completes a Python tag symbol in the table",
  "[python-adjust]")
{
  // The type-symbol pre-pass (mirroring clang_c_adjust::adjust()'s first
  // loop): an unpadded Python-mode tag with a macro-typed member is
  // macro-expanded and padded in the symbol table itself, so later
  // resolution follows the fixed-up layout.
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  symbolt alias;
  alias.id = alias.name = "py_prepass_alias";
  alias.mode = "Python";
  alias.is_type = true;
  alias.is_macro = true;
  alias.set_type(signedbv_type2tc(8));
  ctx.add(alias);

  const type2tc unpadded = struct_type2tc(
    std::vector<type2tc>{symbol_type2tc("py_prepass_alias"), get_int32_type()},
    std::vector<irep_idt>{"c", "i"},
    std::vector<irep_idt>{"c", "i"},
    "PrePad",
    false);
  symbolt tag;
  tag.id = tag.name = "tag-PrePad";
  tag.mode = "Python";
  tag.is_type = true;
  tag.set_type(unpadded);
  ctx.add(tag);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());

  const type2tc &stored = ctx.find_symbol("tag-PrePad")->get_type2();
  REQUIRE(is_struct_type(stored));
  const struct_type2t &st = to_struct_type(stored);
  REQUIRE(st.members.size() == 3);
  REQUIRE(st.members[0] == signedbv_type2tc(8));
  REQUIRE(is_unsignedbv_type(st.members[1]));
  REQUIRE(st.member_names[2] == "i");
}

TEST_CASE(
  "python_adjust pre-pass skips a non-Python type symbol",
  "[python-adjust]")
{
  // C/C++-header types may carry bitfields whose #bitfield flag the IREP2
  // round-trip drops; the pre-pass must not re-pad them (RV-adj4 scoping).
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  const type2tc unpadded = struct_type2tc(
    std::vector<type2tc>{signedbv_type2tc(8), get_int32_type()},
    std::vector<irep_idt>{"c", "i"},
    std::vector<irep_idt>{"c", "i"},
    "CPad",
    false);
  symbolt tag;
  tag.id = tag.name = "tag-CPad";
  tag.mode = "C";
  tag.is_type = true;
  tag.set_type(unpadded);
  ctx.add(tag);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());

  REQUIRE(ctx.find_symbol("tag-CPad")->get_type2() == unpadded);
}

TEST_CASE(
  "python_adjust pre-pass output feeds the value walk's resolution",
  "[python-adjust]")
{
  // End-to-end across the two phases: the tag starts unpadded in the table;
  // the pre-pass pads it, and the value walk's resolve_source then follows a
  // transient member source to the *padded* layout — the reason the type
  // pre-pass must run first (clang_c_adjust's "so that symbolic-type
  // resolution always receives fixed up types").
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  const type2tc unpadded = struct_type2tc(
    std::vector<type2tc>{signedbv_type2tc(8), get_int32_type()},
    std::vector<irep_idt>{"c", "i"},
    std::vector<irep_idt>{"c", "i"},
    "Chain",
    false);
  symbolt tag;
  tag.id = tag.name = "tag-Chain";
  tag.mode = "Python";
  tag.is_type = true;
  tag.set_type(unpadded);
  ctx.add(tag);

  const expr2tc source = symbol2tc(symbol_type2tc("tag-Chain"), "obj");
  const expr2tc member = member2tc(get_int32_type(), source, "i");
  const expr2tc body =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(member)});
  add_code_symbol(ctx, "py_adjust_chain_sym", body);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());

  const expr2tc &stored = ctx.find_symbol("py_adjust_chain_sym")->get_value2();
  const expr2tc &stmt = to_code_block2t(stored).operands.at(0);
  const expr2tc &resolved = to_code_expression2t(stmt).operand;
  REQUIRE(is_member2t(resolved));
  const type2tc &src_type = to_member2t(resolved).source_value->type;
  REQUIRE(is_struct_type(src_type));
  REQUIRE(to_struct_type(src_type).members.size() == 3);
}

TEST_CASE(
  "python_adjust pre-pass leaves an empty (incomplete) tag unchanged",
  "[python-adjust]")
{
  // A forward-declared class tag (python_class_builder::ensure_sym) is an
  // incomplete struct with no components; the pre-pass must pass through
  // without padding or crashing, and without writing the symbol back.
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  const type2tc empty_struct = struct_type2tc(
    std::vector<type2tc>{},
    std::vector<irep_idt>{},
    std::vector<irep_idt>{},
    "Fwd",
    false);
  symbolt tag;
  tag.id = tag.name = "tag-Fwd";
  tag.mode = "Python";
  tag.is_type = true;
  tag.set_type(empty_struct);
  ctx.add(tag);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());

  REQUIRE(ctx.find_symbol("tag-Fwd")->get_type2() == empty_struct);
}

namespace
{
// Register `tag-<name>` whose LEGACY type is a struct carrying a "bases"
// sub-irep listing `tag-<base>` — the layout python_class_builder::get_bases
// produces and clang_cpp_adjust::convert_exception_id consumes.
void add_class_with_base(
  contextt &ctx,
  const std::string &name,
  const std::string &base)
{
  struct_typet legacy;
  legacy.tag(name);
  exprt &bases = static_cast<exprt &>(legacy.add("bases"));
  bases.get_sub().emplace_back(irept("tag-" + base));

  symbolt tag;
  tag.id = tag.name = "tag-" + name;
  tag.mode = "Python";
  tag.is_type = true;
  tag.set_type(legacy);
  ctx.add(tag);
}
} // namespace

TEST_CASE(
  "python_adjust completes an empty throw exception list from the class",
  "[python-adjust]")
{
  // Flip blocker #1: a cpp-throw whose exception_list the legacy pass never
  // derived must get [derived, direct bases...] from its operand's tag —
  // the same one-level chain convert_exception_id produces.
  contextt ctx;
  add_class_with_base(ctx, "E", "Exception");

  const expr2tc operand = symbol2tc(symbol_type2tc("tag-E"), "e");
  expr2tc thr =
    code_cpp_throw2tc(operand, std::vector<irep_idt>{}, locationt());

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(thr);

  REQUIRE(is_code_cpp_throw2t(thr));
  const code_cpp_throw2t &t = to_code_cpp_throw2t(thr);
  REQUIRE(t.exception_list.size() == 2);
  REQUIRE(t.exception_list[0] == "E");
  REQUIRE(t.exception_list[1] == "Exception");
}

TEST_CASE(
  "python_adjust leaves a legacy-filled throw exception list untouched",
  "[python-adjust]")
{
  // Inertness on today's pipeline: clang_cpp_adjust already derived the
  // list; the arm must not re-derive or reorder it.
  contextt ctx;
  add_class_with_base(ctx, "E", "Exception");

  const expr2tc operand = symbol2tc(symbol_type2tc("tag-E"), "e");
  const std::vector<irep_idt> legacy_ids{"E", "Exception"};
  expr2tc thr = code_cpp_throw2tc(operand, legacy_ids, locationt());
  const expr2tc original = thr;

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(thr);

  REQUIRE(thr == original);
}

TEST_CASE(
  "python_adjust derives void_ptr for an untypeable raise operand",
  "[python-adjust]")
{
  // The attribute-raise fallback (`raise pkg.Error(...)`) types the operand
  // any_type() = pointer(empty); legacy convert_exception_id derives
  // "void_ptr" for it. An empty list here would re-create the
  // remove_exceptions front() crash the arm exists to prevent.
  contextt ctx;
  const expr2tc operand =
    symbol2tc(pointer_type2tc(get_empty_type()), "untyped_exc");
  expr2tc thr =
    code_cpp_throw2tc(operand, std::vector<irep_idt>{}, locationt());

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(thr);

  const code_cpp_throw2t &t = to_code_cpp_throw2t(thr);
  REQUIRE(t.exception_list.size() == 1);
  REQUIRE(t.exception_list[0] == "void_ptr");
}

TEST_CASE(
  "python_adjust gives a non-class throw operand the never-empty fallback",
  "[python-adjust]")
{
  // A shape the converter never emits (int operand) still must not leave an
  // empty list — remove_exceptions dereferences front(). The synthetic
  // type-id never matches a real throw, mirroring legacy's fallback.
  contextt ctx;
  expr2tc thr = code_cpp_throw2tc(
    gen_zero(get_int32_type()), std::vector<irep_idt>{}, locationt());

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(thr);

  const code_cpp_throw2t &t = to_code_cpp_throw2t(thr);
  REQUIRE(t.exception_list.size() == 1);
  REQUIRE(t.exception_list[0] == "signedbv");
}

TEST_CASE("python_adjust leaves a bare re-raise untouched", "[python-adjust]")
{
  // A bare `raise` lowers to a nil-operand throw with an empty list; legacy
  // skips it (adjust_side_effect_throw returns early) and so must this arm.
  contextt ctx;
  expr2tc thr =
    code_cpp_throw2tc(expr2tc(), std::vector<irep_idt>{}, locationt());
  const expr2tc original = thr;

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(thr);

  REQUIRE(thr == original);
}

TEST_CASE(
  "python_adjust derives all direct bases of a multi-base class",
  "[python-adjust]")
{
  // register_chain treats the tail as flat direct bases of front(); the
  // derivation must emit every direct base in declaration order.
  contextt ctx;
  struct_typet legacy;
  legacy.tag("M");
  exprt &bases = static_cast<exprt &>(legacy.add("bases"));
  bases.get_sub().emplace_back(irept("tag-A"));
  bases.get_sub().emplace_back(irept("tag-B"));
  symbolt tag;
  tag.id = tag.name = "tag-M";
  tag.mode = "Python";
  tag.is_type = true;
  tag.set_type(legacy);
  ctx.add(tag);

  const expr2tc operand = symbol2tc(symbol_type2tc("tag-M"), "m");
  expr2tc thr =
    code_cpp_throw2tc(operand, std::vector<irep_idt>{}, locationt());

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(thr);

  const code_cpp_throw2t &t = to_code_cpp_throw2t(thr);
  REQUIRE(t.exception_list.size() == 3);
  REQUIRE(t.exception_list[0] == "M");
  REQUIRE(t.exception_list[1] == "A");
  REQUIRE(t.exception_list[2] == "B");
}

TEST_CASE(
  "python_adjust pre-pass write-back preserves bases for the throw chain",
  "[python-adjust]")
{
  // The M2 hazard: the type-symbol pre-pass rewrites a tag (padding fires),
  // and set_type(type2tc) would drop the legacy-only "bases" sub-irep the
  // throw derivation reads. The write-back must re-attach it, so a body
  // throw still derives the full chain after the tag was rewritten.
  config.ansi_c.set_data_model(configt::LP64);

  contextt ctx;
  struct_typet legacy;
  legacy.tag("E");
  {
    struct_union_typet::componentt c;
    c.id("component");
    c.type() = migrate_type_back(signedbv_type2tc(8));
    c.set_name("c");
    c.pretty_name("c");
    legacy.components().push_back(c);
    struct_union_typet::componentt i;
    i.id("component");
    i.type() = migrate_type_back(get_int32_type());
    i.set_name("i");
    i.pretty_name("i");
    legacy.components().push_back(i);
  }
  exprt &bases = static_cast<exprt &>(legacy.add("bases"));
  bases.get_sub().emplace_back(irept("tag-Exception"));
  symbolt tag;
  tag.id = tag.name = "tag-E";
  tag.mode = "Python";
  tag.is_type = true;
  tag.set_type(legacy);
  ctx.add(tag);

  const expr2tc operand = symbol2tc(symbol_type2tc("tag-E"), "e");
  const expr2tc thr =
    code_cpp_throw2tc(operand, std::vector<irep_idt>{}, locationt());
  const expr2tc body =
    code_block2tc(std::vector<expr2tc>{code_expression2tc(thr)});
  add_code_symbol(ctx, "py_adjust_throw_prepass_sym", body);

  python_adjust adjuster(ctx);
  REQUIRE_FALSE(adjuster.adjust());

  // The pre-pass padded (rewrote) the tag...
  const symbolt *out_tag = ctx.find_symbol("tag-E");
  REQUIRE(is_struct_type(out_tag->get_type2()));
  REQUIRE(to_struct_type(out_tag->get_type2()).members.size() == 3);
  // ...its legacy view kept the bases...
  REQUIRE(out_tag->get_type().find("bases").is_not_nil());
  // ...and the body throw derived the full chain through it.
  const expr2tc &stored =
    ctx.find_symbol("py_adjust_throw_prepass_sym")->get_value2();
  const expr2tc &stmt = to_code_block2t(stored).operands.at(0);
  const expr2tc &out_thr = to_code_expression2t(stmt).operand;
  const code_cpp_throw2t &t = to_code_cpp_throw2t(out_thr);
  REQUIRE(t.exception_list.size() == 2);
  REQUIRE(t.exception_list[0] == "E");
  REQUIRE(t.exception_list[1] == "Exception");
}

TEST_CASE(
  "python_adjust derives throw ids through an S2-resolved literal operand",
  "[python-adjust]")
{
  // The OM raise shape end-to-end: the throw operand is a by-name struct
  // literal; operand recursion (S2) retypes it to the resolved struct first,
  // and the throw arm must derive the chain from that resolved shape.
  contextt ctx;
  add_class_with_base(ctx, "E", "Exception");

  const expr2tc lit =
    constant_struct2tc(symbol_type2tc("tag-E"), std::vector<expr2tc>{});
  expr2tc thr = code_cpp_throw2tc(lit, std::vector<irep_idt>{}, locationt());

  python_adjust adjuster(ctx);
  adjuster.adjust_expr(thr);

  const code_cpp_throw2t &t = to_code_cpp_throw2t(thr);
  REQUIRE(is_constant_struct2t(t.operand));
  REQUIRE(is_struct_type(t.operand->type));
  REQUIRE(t.exception_list.size() == 2);
  REQUIRE(t.exception_list[0] == "E");
  REQUIRE(t.exception_list[1] == "Exception");
}

TEST_CASE(
  "python_adjust S1 adjust_expr completes a node's own type",
  "[python-adjust]")
{
  // adjust_expr's leading type-completion (mirroring the legacy adjust_expr's
  // adjust_type(expr.type())): a symbol whose type is a macro alias is
  // retyped to the concrete type via with_type.
  contextt ctx;
  symbolt alias;
  alias.id = alias.name = "py_expr_alias";
  alias.mode = "Python";
  alias.is_type = true;
  alias.is_macro = true;
  alias.set_type(get_int32_type());
  ctx.add(alias);

  expr2tc e = symbol2tc(symbol_type2tc("py_expr_alias"), "x");
  python_adjust adjuster(ctx);
  adjuster.adjust_expr(e);

  REQUIRE(is_symbol2t(e));
  REQUIRE(e->type == get_int32_type());
}
