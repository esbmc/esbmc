#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/python_adjust.h>
#include <irep2/irep2_utils.h>
#include <util/config.h>
#include <util/context.h>
#include <util/c_types.h>

// Part V Phase V.1k (b), sub-phases B.0/B.1 (docs/irep2-migration.md): the
// IREP2-native Python adjuster. These tests pin its contracts so each later
// sub-phase extends a verified pass:
//   B.0 — adjust_expr visits every node; adjust() skips is_type symbols and is a
//         no-op when nothing needs resolving.
//   B.1 — a member2t/index2t source carrying an unresolved symbol_type2t is
//         followed to its struct/array, recursively (resolve X before X.a).

namespace
{
// Exposes the protected recursion and counts every non-nil node descended into.
struct test_adjust : public python_adjust
{
  using python_adjust::python_adjust;
  unsigned visited = 0;

  void adjust_expr(expr2tc &expr) override
  {
    if (!is_nil_expr(expr))
      ++visited;
    python_adjust::adjust_expr(expr);
  }
};

void add_type_symbol(contextt &ctx, const irep_idt &id, const type2tc &t)
{
  symbolt s;
  s.id = id;
  s.name = id;
  s.mode = "Python";
  s.is_type = true;
  s.set_type(t);
  ctx.add(s);
}

// A one-field struct named `name` with member `memb` of type `mt`.
type2tc make_struct(const irep_idt &name, const irep_idt &memb, const type2tc &mt)
{
  return struct_type2tc(
    std::vector<type2tc>{mt},
    std::vector<irep_idt>{memb},
    std::vector<irep_idt>{memb},
    name,
    false);
}
} // namespace

TEST_CASE(
  "python_adjust B.0 walker visits every IREP2 node once",
  "[python-frontend][irep2][python-adjust]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  // add( mul(1, 2), 3 ) — 2 internal arith nodes + 3 leaf constants = 5 nodes.
  const type2tc t = get_uint_type(config.ansi_c.word_size);
  expr2tc tree = add2tc(t, mul2tc(t, gen_ulong(1), gen_ulong(2)), gen_ulong(3));

  contextt context;
  test_adjust walker(context);
  walker.adjust_expr(tree);

  REQUIRE(walker.visited == 5);
}

TEST_CASE(
  "python_adjust B.0 adjust() is a no-op over the symbol table",
  "[python-frontend][irep2][python-adjust]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  const type2tc t = get_uint_type(config.ansi_c.word_size);
  const expr2tc value =
    add2tc(t, mul2tc(t, gen_ulong(7), gen_ulong(11)), gen_ulong(13));

  contextt context;
  symbolt symbol;
  symbol.id = "py:test@F@adjust_noop";
  symbol.name = "adjust_noop";
  symbol.mode = "Python";
  symbol.set_value(value);
  symbol.set_type(t);
  context.add(symbol);

  // A type symbol carrying a value: adjust() must skip it (is_type branch) and
  // leave it byte-identical, just as clang_c_adjust::adjust() does.
  symbolt type_symbol;
  type_symbol.id = "py:test@T@tag-Skipped";
  type_symbol.name = "tag-Skipped";
  type_symbol.mode = "Python";
  type_symbol.is_type = true;
  type_symbol.set_value(value);
  type_symbol.set_type(t);
  context.add(type_symbol);

  REQUIRE_FALSE(python_adjust(context).adjust());

  const symbolt *out = context.find_symbol("py:test@F@adjust_noop");
  REQUIRE(out != nullptr);
  REQUIRE(out->get_value2() == value);

  const symbolt *type_out = context.find_symbol("py:test@T@tag-Skipped");
  REQUIRE(type_out != nullptr);
  REQUIRE(type_out->get_value2() == value);
}

TEST_CASE(
  "python_adjust B.1 follows a symbol_type2t member source to its struct",
  "[python-frontend][irep2][python-adjust]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  const type2tc intt = get_int_type(config.ansi_c.int_width);

  contextt context;
  add_type_symbol(context, "tag-Point", make_struct("tag-Point", "x", intt));

  // member( symbol<tag-Point> p, "x" ) — source carries the unresolved by-name
  // type, the transient state the relaxed construction assert permits.
  expr2tc src = symbol2tc(symbol_type2tc("tag-Point"), "p");
  expr2tc member = member2tc(intt, src, "x");
  REQUIRE(is_symbol_type(to_member2t(member).source_value->type));

  test_adjust(context).adjust_expr(member);

  const type2tc &resolved = to_member2t(member).source_value->type;
  REQUIRE(is_struct_type(resolved));
  REQUIRE(to_struct_type(resolved).name == irep_idt("tag-Point"));
}

TEST_CASE(
  "python_adjust B.1 follows a symbol_type2t index source to its array",
  "[python-frontend][irep2][python-adjust]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  const type2tc intt = get_int_type(config.ansi_c.int_width);
  const type2tc arr = array_type2tc(intt, gen_ulong(4), false);

  contextt context;
  add_type_symbol(context, "tag-IntArr", arr);

  expr2tc src = symbol2tc(symbol_type2tc("tag-IntArr"), "a");
  expr2tc index = index2tc(intt, src, gen_ulong(0));
  REQUIRE(is_symbol_type(to_index2t(index).source_value->type));

  test_adjust(context).adjust_expr(index);

  REQUIRE(is_array_type(to_index2t(index).source_value->type));
}

TEST_CASE(
  "python_adjust B.1 resolves nested member sources recursively",
  "[python-frontend][irep2][python-adjust]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  const type2tc intt = get_int_type(config.ansi_c.int_width);

  contextt context;
  // tag-A { inner: <tag-B> };  tag-B { x: int }
  add_type_symbol(
    context, "tag-A", make_struct("tag-A", "inner", symbol_type2tc("tag-B")));
  add_type_symbol(context, "tag-B", make_struct("tag-B", "x", intt));

  // (a.inner).x — both member sources are unresolved by-name types.
  expr2tc a = symbol2tc(symbol_type2tc("tag-A"), "a");
  expr2tc inner = member2tc(symbol_type2tc("tag-B"), a, "inner");
  expr2tc outer = member2tc(intt, inner, "x");

  test_adjust(context).adjust_expr(outer);

  // Outer source (a.inner) is followed to struct tag-B...
  const expr2tc &outer_src = to_member2t(outer).source_value;
  REQUIRE(is_struct_type(outer_src->type));
  REQUIRE(to_struct_type(outer_src->type).name == irep_idt("tag-B"));

  // ...and its own source (a) was resolved first to struct tag-A.
  const type2tc &inner_src = to_member2t(outer_src).source_value->type;
  REQUIRE(is_struct_type(inner_src));
  REQUIRE(to_struct_type(inner_src).name == irep_idt("tag-A"));
}

TEST_CASE(
  "python_adjust B.1 leaves an unregistered-tag source untouched",
  "[python-frontend][irep2][python-adjust]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  const type2tc intt = get_int_type(config.ansi_c.int_width);

  // No type symbol registered for tag-Missing: follow would crash, so the pass
  // must leave the source as-is for the post-adjust verifier (B.4) to flag.
  contextt context;
  expr2tc src = symbol2tc(symbol_type2tc("tag-Missing"), "m");
  expr2tc member = member2tc(intt, src, "x");

  test_adjust(context).adjust_expr(member);

  REQUIRE(is_symbol_type(to_member2t(member).source_value->type));
}

TEST_CASE(
  "python_adjust B.1 resolves member sources through adjust()",
  "[python-frontend][irep2][python-adjust]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  const type2tc intt = get_int_type(config.ansi_c.int_width);

  contextt context;
  add_type_symbol(context, "tag-Point", make_struct("tag-Point", "x", intt));

  // A value symbol whose body is an unresolved member: adjust() must walk the
  // symbol table, resolve the body, and store the resolved form back.
  symbolt symbol;
  symbol.id = "py:test@F@uses_point";
  symbol.name = "uses_point";
  symbol.mode = "Python";
  symbol.set_value(
    member2tc(intt, symbol2tc(symbol_type2tc("tag-Point"), "p"), "x"));
  symbol.set_type(intt);
  context.add(symbol);

  REQUIRE_FALSE(python_adjust(context).adjust());

  const symbolt *out = context.find_symbol("py:test@F@uses_point");
  REQUIRE(out != nullptr);
  REQUIRE(is_struct_type(to_member2t(out->get_value2()).source_value->type));
}

TEST_CASE(
  "python_adjust B.2 adjust() skips non-Python-mode symbols",
  "[python-frontend][irep2][python-adjust]")
{
  cmdlinet cmdline;
  REQUIRE_FALSE(config.set(cmdline));

  const type2tc intt = get_int_type(config.ansi_c.int_width);

  contextt context;
  add_type_symbol(context, "tag-Point", make_struct("tag-Point", "x", intt));

  // A C-mode symbol carrying an unresolved member: the Python-only pass must
  // leave it untouched (V.1k RV-adj4 — the C OM bodies stay on the legacy path).
  const expr2tc body =
    member2tc(intt, symbol2tc(symbol_type2tc("tag-Point"), "p"), "x");
  symbolt symbol;
  symbol.id = "c:@F@uses_point";
  symbol.name = "uses_point";
  symbol.mode = "C";
  symbol.set_value(body);
  symbol.set_type(intt);
  context.add(symbol);

  REQUIRE_FALSE(python_adjust(context).adjust());

  const symbolt *out = context.find_symbol("c:@F@uses_point");
  REQUIRE(out != nullptr);
  REQUIRE(is_symbol_type(to_member2t(out->get_value2()).source_value->type));
}
