// The descriptor-to-field-path inverse of member_offset_bits and of the index
// arm of value_sett::get_reference_set_rec.
//
// Every rule the two directions must agree on -- padding counted as explicit
// members, unions overlaid at zero, bits for members and bytes for elements,
// ns.follow before measuring -- was held in step by a comment while the inverse
// lived in value_set.cpp as a file-static, so the only evidence for it was an
// end-to-end MPOR race (docs/roadmap/goto-symex-verification-plan.md, R31/R32).
// Two of the defects that produced it were that disagreement made concrete: the
// inverse not existing at all, and the forward walk resolving symbol types
// where the inverse did not.

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <util/expr/type_byte_size.h>
#include <util/config/config.h>
#include <util/lang/c_types.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>
#include <util/symtab/symbol.h>
#include <irep2/irep2_utils.h>

#include <algorithm>

// type_byte_size measures against config.ansi_c, which is zero-initialised bar
// int_128_width. Pin a model in main(): `config` lives in another translation
// unit, so a static initialiser here would race its constructor.
int main(int argc, char *argv[])
{
  config.ansi_c.set_data_model(configt::LP64);
  return Catch::Session().run(argc, argv);
}

namespace
{
type2tc int_ptr()
{
  return pointer_type2tc(get_int32_type());
}

type2tc struct_of(
  std::vector<type2tc> members,
  std::vector<irep_idt> names,
  const std::string &tag)
{
  return struct_type2tc(members, names, names, tag, false);
}

type2tc union_of(
  std::vector<type2tc> members,
  std::vector<irep_idt> names,
  const std::string &tag)
{
  return union_type2tc(members, names, names, tag, false);
}

// Register `tag-<name>` = @p type so ns.follow resolves a symbol_type2t to it.
type2tc
add_type_symbol(contextt &ctx, const std::string &name, const type2tc &type)
{
  symbolt sym;
  sym.id = "tag-" + name;
  sym.name = "tag-" + name;
  sym.mode = "C";
  sym.is_type = true;
  sym.set_type(type);
  ctx.add(sym);
  return symbol_type2tc("tag-" + name);
}

bool has(const std::vector<std::string> &paths, const std::string &p)
{
  return std::find(paths.begin(), paths.end(), p) != paths.end();
}
} // namespace

TEST_CASE(
  "member_paths_at_offset inverts member_offset on every member",
  "[core][util][type_byte_size]")
{
  contextt ctx;
  namespacet ns(ctx);

  // `struct { char c1; int *p; char c2; }` as the frontend hands it over:
  // member_offset_bits sums the widths of the preceding members and makes no
  // alignment adjustment of its own, so the padding an LP64 layout needs is
  // present as explicit members. The inverse counts them the same way.
  const std::vector<type2tc> members{
    signedbv_type2tc(8),
    unsignedbv_type2tc(56),
    int_ptr(),
    signedbv_type2tc(8),
    unsignedbv_type2tc(56)};
  const std::vector<irep_idt> names{
    "c1", "anon_pad$0", "p", "c2", "anon_pad$1"};
  const type2tc s = struct_of(members, names, "S");

  for (size_t i = 0; i < members.size(); i++)
  {
    const BigInt offset = member_offset(s, names[i], &ns);
    const std::vector<std::string> paths =
      member_paths_at_offset(s, offset, members[i], ns);
    INFO("member " << names[i].as_string() << " at byte " << offset);
    REQUIRE(has(paths, "." + names[i].as_string()));
  }

  // The forward direction is the one that decides those offsets, so pin them:
  // an inverse that agrees with a wrong forward walk proves nothing.
  REQUIRE(member_offset(s, "p", &ns) == 8);
  REQUIRE(member_offset(s, "c2", &ns) == 16);
  REQUIRE(type_byte_size(s, &ns) == 24);

  // Anti-vacuity: the match is on the target type exactly, so asking for a
  // pointer at the offset of a char finds nothing.
  REQUIRE(member_paths_at_offset(s, 0, int_ptr(), ns).empty());
}

TEST_CASE(
  "member_paths_at_offset descends nested aggregates",
  "[core][util][type_byte_size]")
{
  contextt ctx;
  namespacet ns(ctx);

  const type2tc inner =
    struct_of({get_int64_type(), int_ptr()}, {"pad", "p"}, "Inner");
  const type2tc outer =
    struct_of({get_int64_type(), inner}, {"pad", "in"}, "S");

  REQUIRE(has(member_paths_at_offset(outer, 16, int_ptr(), ns), ".in.p"));
  REQUIRE(member_paths_at_offset(outer, 8, int_ptr(), ns).empty());
}

TEST_CASE(
  "member_paths_at_offset composes a member offset with an element offset",
  "[core][util][type_byte_size]")
{
  contextt ctx;
  namespacet ns(ctx);

  // R33: `struct S { long pad; int *v[2]; }` reached at byte 16 is element 1 of
  // a member based at 8. Each half worked alone and only the composition
  // failed, which is what the two arms disagreeing looks like from the forward
  // side. Both elements map back to the same "[]" suffix, since a value-set
  // path does not distinguish them.
  const type2tc arr = array_type2tc(int_ptr(), gen_ulong(2), false);
  const type2tc s = struct_of({get_int64_type(), arr}, {"pad", "v"}, "S");

  REQUIRE(has(member_paths_at_offset(s, 8, int_ptr(), ns), ".v[]"));
  REQUIRE(has(member_paths_at_offset(s, 16, int_ptr(), ns), ".v[]"));
  // Byte 4 is inside `pad`, not a pointer.
  REQUIRE(member_paths_at_offset(s, 4, int_ptr(), ns).empty());
  // Byte 12 is the interior of element 0's pointer. This is what makes the
  // walk an inverse of member_offset_bits rather than a "holds a target
  // somewhere below" query, which is the distinction R31/R32 turn on.
  REQUIRE(member_paths_at_offset(s, 12, int_ptr(), ns).empty());

  // One past the end of `v` is the start of the member after it, not element 0
  // reached by wrapping: the member bound is exclusive, and `offset % esize`
  // would otherwise hand back a path the object does not have. Only reachable
  // with the array somewhere other than last.
  const type2tc s6 = struct_of({arr, int_ptr()}, {"v", "b"}, "S6");
  const std::vector<std::string> past_end =
    member_paths_at_offset(s6, 16, int_ptr(), ns);
  REQUIRE(past_end.size() == 1);
  REQUIRE(has(past_end, ".b"));
}

TEST_CASE(
  "member_paths_at_offset does not divide by a zero element size",
  "[core][util][type_byte_size]")
{
  contextt ctx;
  namespacet ns(ctx);

  // An empty aggregate measures 0, and `offset % esize` on it is a BigInt
  // modulo by zero -- an abort, not a wrong answer. A zero-sized *member* has
  // an empty offset range and so never reaches the array arm, which is why the
  // array type has to be passed directly to exercise the guard.
  const type2tc empty = struct_of({}, {}, "Empty");
  REQUIRE(type_byte_size(empty, &ns) == 0);

  const type2tc zarr = array_type2tc(empty, gen_ulong(4), false);
  REQUIRE(member_paths_at_offset(zarr, 8, int_ptr(), ns).empty());
}

TEST_CASE(
  "member_paths_at_offset overlays every union member at the same offset",
  "[core][util][type_byte_size]")
{
  contextt ctx;
  namespacet ns(ctx);

  // An offset alone cannot say which union member is live, so both candidates
  // of the right type are returned -- the same over-approximation the member2t
  // arm of get_value_set_rec makes.
  const type2tc u = union_of({int_ptr(), int_ptr()}, {"p", "q"}, "U");
  const std::vector<std::string> paths =
    member_paths_at_offset(u, 0, int_ptr(), ns);
  REQUIRE(paths.size() == 2);
  REQUIRE(has(paths, ".p"));
  REQUIRE(has(paths, ".q"));

  // Union members start at zero rather than accumulating: a second member is
  // still reachable at offset 0 when the first is wider than it.
  const type2tc mixed =
    union_of({get_int64_type(), int_ptr()}, {"l", "p"}, "U2");
  REQUIRE(has(member_paths_at_offset(mixed, 0, int_ptr(), ns), ".p"));
}

TEST_CASE(
  "member_paths_at_offset follows symbol types before measuring",
  "[core][util][type_byte_size]")
{
  contextt ctx;
  namespacet ns(ctx);

  const type2tc inner =
    struct_of({get_int64_type(), int_ptr()}, {"pad", "p"}, "Inner");
  const type2tc inner_sym = add_type_symbol(ctx, "Inner", inner);
  const type2tc outer =
    struct_of({get_int64_type(), inner_sym}, {"pad", "in"}, "S");

  // size_bits follows before measuring, so an unfollowed inverse would both
  // mismeasure the member and fail to descend into it.
  REQUIRE(has(member_paths_at_offset(outer, 16, int_ptr(), ns), ".in.p"));
  REQUIRE(has(member_paths_at_offset(inner_sym, 8, int_ptr(), ns), ".p"));
  // The typed walk resolves them too. It measures nothing, so its reason is
  // only the descent -- but an unfollowed member is just as opaque to it.
  REQUIRE(has(member_paths_of_type(outer, int_ptr(), ns), ".in.p"));
}

TEST_CASE(
  "member_paths_of_type takes every path a descriptor without an offset leaves "
  "possible",
  "[core][util][type_byte_size]")
{
  contextt ctx;
  namespacet ns(ctx);

  const type2tc s = struct_of({int_ptr(), int_ptr()}, {"p", "q"}, "S");

  // R32: an offset selects one member; no offset selects both.
  const std::vector<std::string> at_q =
    member_paths_at_offset(s, member_offset(s, "q", &ns), int_ptr(), ns);
  REQUIRE(at_q.size() == 1);
  REQUIRE(has(at_q, ".q"));

  const std::vector<std::string> any = member_paths_of_type(s, int_ptr(), ns);
  REQUIRE(any.size() == 2);
  REQUIRE(has(any, ".p"));
  REQUIRE(has(any, ".q"));
}

TEST_CASE(
  "member_paths_of_type reaches into an element of no constant size",
  "[core][util][type_byte_size]")
{
  contextt ctx;
  namespacet ns(ctx);

  // A member the offset walk cannot measure ends its descent, keeping what it
  // already found. The typed walk consults no size at all, so it keeps going --
  // the one direction in which it is the more precise of the two.
  const type2tc vla =
    array_type2tc(int_ptr(), symbol2tc(get_uint64_type(), "n"), false);
  const type2tc s = struct_of({int_ptr(), vla}, {"p", "v"}, "S");

  const std::vector<std::string> bounded =
    member_paths_at_offset(s, 0, int_ptr(), ns);
  REQUIRE(bounded.size() == 1);
  REQUIRE(has(bounded, ".p"));

  const std::vector<std::string> any = member_paths_of_type(s, int_ptr(), ns);
  REQUIRE(has(any, ".p"));
  REQUIRE(has(any, ".v[]"));

  // It ends the descent rather than skipping the member: every later member is
  // at an offset the walk can no longer place, so continuing would attribute
  // this offset to the wrong field. Only observable with the unmeasurable
  // member somewhere other than last, which C does not allow -- see the
  // last-member argument in type_byte_size.cpp.
  const type2tc mid =
    struct_of({int_ptr(), vla, int_ptr()}, {"p", "v", "q"}, "S4");
  REQUIRE(member_paths_at_offset(mid, 8, int_ptr(), ns).empty());

  // The array arm has its own catch, reached only when the *element* is what
  // cannot be measured. Wrapped in a struct the throw lands on the member loop
  // instead, so the array type has to be passed directly.
  const type2tc vla_arr = array_type2tc(vla, gen_ulong(2), false);
  REQUIRE(member_paths_at_offset(vla_arr, 8, int_ptr(), ns).empty());
  REQUIRE(has(member_paths_of_type(vla_arr, int_ptr(), ns), "[][]"));
}

TEST_CASE(
  "both walks yield the empty path when the type is the target",
  "[core][util][type_byte_size]")
{
  contextt ctx;
  namespacet ns(ctx);

  // value_sett::offset_paths relies on this to recognise a descriptor that
  // already names the object being dereferenced, which its unrefined lookup
  // covers; returning the empty path would have the caller repeat that lookup.
  REQUIRE(
    member_paths_at_offset(int_ptr(), 0, int_ptr(), ns) ==
    std::vector<std::string>{""});
  REQUIRE(
    member_paths_of_type(int_ptr(), int_ptr(), ns) ==
    std::vector<std::string>{""});
}
