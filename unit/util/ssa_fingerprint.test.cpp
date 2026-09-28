#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <fmt/format.h>
#include <util/message/message.h>
#include <util/ssa/fingerprint.h>

namespace
{
expr2tc local(const std::string &name, unsigned offset, unsigned version)
{
  return symbol2tc(
    get_int32_type(),
    irep_idt("c:t.c@" + std::to_string(offset) + "@F@main@" + name),
    symbol_renaming_level::level2,
    0,
    version,
    0,
    0);
}

void add_step(
  symex_target_equationt::SSA_stepst &steps,
  const expr2tc &lhs,
  int literal)
{
  steps.emplace_back();
  auto &s = steps.back();
  s.type = goto_trace_stept::ASSIGNMENT;
  s.guard = gen_true_expr();
  s.cond = equality2tc(lhs, constant_int2tc(get_int32_type(), BigInt(literal)));
}
expr2tc plain(const std::string &name, unsigned version)
{
  return symbol2tc(
    get_int32_type(),
    irep_idt(name),
    symbol_renaming_level::level2,
    0,
    version,
    0,
    0);
}

void add_bare_step(
  symex_target_equationt::SSA_stepst &steps,
  goto_trace_stept::typet type)
{
  steps.emplace_back();
  steps.back().type = type;
}

void add_typed_step(
  symex_target_equationt::SSA_stepst &steps,
  const std::string &tag)
{
  const type2tc anon = struct_type2tc(
    std::vector<type2tc>{get_int32_type()},
    std::vector<irep_idt>{"f"},
    std::vector<irep_idt>{"f"},
    irep_idt(tag),
    false);

  steps.emplace_back();
  auto &s = steps.back();
  s.type = goto_trace_stept::ASSIGNMENT;
  s.guard = gen_true_expr();
  s.cond = equality2tc(
    symbol2tc(
      anon, "c:t.c@100@F@main@s", symbol_renaming_level::level2, 0, 1, 0, 0),
    symbol2tc(
      anon, "c:t.c@100@F@main@s", symbol_renaming_level::level2, 0, 0, 0, 0));
}

void add_typed_step(
  symex_target_equationt::SSA_stepst &steps,
  goto_trace_stept::typet type,
  const expr2tc &cond)
{
  steps.emplace_back();
  auto &s = steps.back();
  s.type = type;
  s.guard = gen_true_expr();
  s.cond = cond;
}

/// `s.<member> > 0`, asserted over a struct with two members whose names
/// differ only after an `_at_`.
void add_member_claim(
  symex_target_equationt::SSA_stepst &steps,
  const std::string &member)
{
  const type2tc two_members = struct_type2tc(
    std::vector<type2tc>{get_int32_type(), get_int32_type()},
    std::vector<irep_idt>{"val_at_a", "val_at_b"},
    std::vector<irep_idt>{"val_at_a", "val_at_b"},
    irep_idt("tag-struct S"),
    false);

  const expr2tc s = symbol2tc(
    two_members,
    "c:t.c@100@F@main@s",
    symbol_renaming_level::level2,
    0,
    1,
    0,
    0);

  add_typed_step(
    steps,
    goto_trace_stept::ASSERT,
    greaterthan2tc(
      member2tc(get_int32_type(), s, irep_idt(member)),
      constant_int2tc(get_int32_type(), BigInt(0))));
}

void add_renumber_step(
  symex_target_equationt::SSA_stepst &steps,
  int object_size)
{
  steps.emplace_back();
  auto &s = steps.back();
  s.type = goto_trace_stept::RENUMBER;
  s.guard = gen_true_expr();
  s.lhs = local("p", 100, 0);
  s.rhs = constant_int2tc(get_int32_type(), BigInt(object_size));
}

void add_output_step(
  symex_target_equationt::SSA_stepst &steps,
  const expr2tc &arg)
{
  steps.emplace_back();
  auto &s = steps.back();
  s.type = goto_trace_stept::OUTPUT;
  s.guard = gen_true_expr();
  auto &od = s.output_payload();
  od.format_string = "%d";
  od.output_args.push_back(arg);
}
} // namespace

TEST_CASE(
  "moving an assume across the claim changes the digest",
  "[fingerprint]")
{
  const expr2tc assumption = greaterthan2tc(
    local("x", 100, 0), constant_int2tc(get_int32_type(), BigInt(0)));
  const expr2tc claim = notequal2tc(
    local("x", 100, 0), constant_int2tc(get_int32_type(), BigInt(5)));

  symex_target_equationt::SSA_stepst assumed_first, asserted_first;
  add_typed_step(assumed_first, goto_trace_stept::ASSUME, assumption);
  add_typed_step(assumed_first, goto_trace_stept::ASSERT, claim);
  add_typed_step(asserted_first, goto_trace_stept::ASSERT, claim);
  add_typed_step(asserted_first, goto_trace_stept::ASSUME, assumption);

  // convert_internal_step encodes a claim as implies(assumpt_expr, cond),
  // where assumpt_expr holds only the assumes seen *before* it. The same steps
  // in a different order are therefore a different proof obligation, and the
  // digest must not equate them.
  REQUIRE(
    ssa_cone_digest(assumed_first, fingerprint_modet::srcloc) !=
    ssa_cone_digest(asserted_first, fingerprint_modet::srcloc));
}

TEST_CASE(
  "a symbol's source offset does not change the digest",
  "[fingerprint]")
{
  symex_target_equationt::SSA_stepst before, after;
  add_step(before, local("x", 100, 0), 7);
  // Same variable after an edit inserted text earlier in the file.
  add_step(after, local("x", 144, 0), 7);

  REQUIRE(
    ssa_cone_digest(before, fingerprint_modet::srcloc) ==
    ssa_cone_digest(after, fingerprint_modet::srcloc));
  REQUIRE(
    ssa_cone_digest(before, fingerprint_modet::raw) !=
    ssa_cone_digest(after, fingerprint_modet::raw));
}

TEST_CASE("distinct cones keep distinct digests", "[fingerprint]")
{
  symex_target_equationt::SSA_stepst one, other;
  add_step(one, local("x", 100, 0), 7);
  add_step(other, local("x", 100, 0), 8);

  REQUIRE(
    ssa_cone_digest(one, fingerprint_modet::srcloc) !=
    ssa_cone_digest(other, fingerprint_modet::srcloc));
}

TEST_CASE("ignored steps are excluded", "[fingerprint]")
{
  symex_target_equationt::SSA_stepst kept, with_ignored;
  add_step(kept, local("x", 100, 0), 7);
  add_step(with_ignored, local("x", 100, 0), 7);
  add_step(with_ignored, local("y", 200, 0), 9);
  with_ignored.back().ignore = true;

  REQUIRE(ssa_cone_size(kept) == 1);
  REQUIRE(ssa_cone_size(with_ignored) == 1);
  REQUIRE(
    ssa_cone_digest(kept, fingerprint_modet::srcloc) ==
    ssa_cone_digest(with_ignored, fingerprint_modet::srcloc));
}

TEST_CASE("a name with no source position is not merged", "[fingerprint]")
{
  // `__ESBMC_alloc` carries no `c:` prefix and `c:t.c@100` no closing `@`, so
  // there is nothing for srcloc to strip. Such names must still stay distinct
  // from one another.
  symex_target_equationt::SSA_stepst one, other;
  add_step(one, plain("__ESBMC_alloc", 3), 7);
  add_step(other, plain("c:t.c@100", 3), 7);

  REQUIRE(
    ssa_cone_digest(one, fingerprint_modet::srcloc) !=
    ssa_cone_digest(other, fingerprint_modet::srcloc));
}

TEST_CASE("a step with no guard or condition still counts", "[fingerprint]")
{
  symex_target_equationt::SSA_stepst output, skip, empty;
  add_bare_step(output, goto_trace_stept::OUTPUT);
  add_bare_step(skip, goto_trace_stept::SKIP);

  REQUIRE(ssa_cone_size(output) == 1);
  // The step type is all such a step contributes, and it must contribute it.
  REQUIRE(
    ssa_cone_digest(output, fingerprint_modet::srcloc) !=
    ssa_cone_digest(skip, fingerprint_modet::srcloc));
  REQUIRE(
    ssa_cone_digest(output, fingerprint_modet::srcloc) !=
    ssa_cone_digest(empty, fingerprint_modet::srcloc));
}

TEST_CASE(
  "an anonymous type's location does not change the digest",
  "[fingerprint]")
{
  // The shape clang_c_convert.cpp:4893 emits, `__anon_` marker and all.
  symex_target_equationt::SSA_stepst before, after;
  add_typed_step(before, "tag-struct __anon_struct_at_t.c_main_3_9");
  // The same type after an edit inserted a line above its declaration.
  add_typed_step(after, "tag-struct __anon_struct_at_t.c_main_5_9");

  REQUIRE(
    ssa_cone_digest(before, fingerprint_modet::srcloc) ==
    ssa_cone_digest(after, fingerprint_modet::srcloc));
  REQUIRE(
    ssa_cone_digest(before, fingerprint_modet::raw) !=
    ssa_cone_digest(after, fingerprint_modet::raw));
}

TEST_CASE("a tag containing _at_ is not truncated", "[fingerprint]")
{
  // Only an anonymous type's position may be cut. A user type whose name
  // happens to contain `_at_` carries no `__anon_` marker, and cutting there
  // would leave two distinct types digesting alike.
  symex_target_equationt::SSA_stepst one, other;
  add_typed_step(one, "tag-struct s_at_a");
  add_typed_step(other, "tag-struct s_at_b");

  REQUIRE(
    ssa_cone_digest(one, fingerprint_modet::srcloc) !=
    ssa_cone_digest(other, fingerprint_modet::srcloc));
}

TEST_CASE("members differing only after _at_ are not merged", "[fingerprint]")
{
  // Reading `val_at_a` where `val_at_b` was written is a different claim, and
  // an unanchored `_at_` strip digested the two alike: a proof stored for one
  // file discharged another file's violated claim (esbmc/esbmc#7143).
  symex_target_equationt::SSA_stepst reads_a, reads_b;
  add_member_claim(reads_a, "val_at_a");
  add_member_claim(reads_b, "val_at_b");

  REQUIRE(
    ssa_cone_digest(reads_a, fingerprint_modet::srcloc) !=
    ssa_cone_digest(reads_b, fingerprint_modet::srcloc));
}

TEST_CASE("a renumbered object's size reaches the digest", "[fingerprint]")
{
  // A RENUMBER step leaves `cond` nil and carries the symbol and its new
  // object size in `lhs`/`rhs`, which convert_internal_step passes to
  // renumber_symbol_address. Two cones differing only there are different
  // proof obligations.
  symex_target_equationt::SSA_stepst small, large;
  add_renumber_step(small, 8);
  add_renumber_step(large, 16);

  REQUIRE(
    ssa_cone_digest(small, fingerprint_modet::srcloc) !=
    ssa_cone_digest(large, fingerprint_modet::srcloc));
}

TEST_CASE("an output step's arguments reach the digest", "[fingerprint]")
{
  // OUTPUT arguments are converted into the formula but live in the step's
  // payload rather than in `cond`.
  symex_target_equationt::SSA_stepst prints_x, prints_y;
  add_output_step(prints_x, local("x", 100, 0));
  add_output_step(prints_y, local("y", 200, 0));

  REQUIRE(
    ssa_cone_digest(prints_x, fingerprint_modet::srcloc) !=
    ssa_cone_digest(prints_y, fingerprint_modet::srcloc));
}

TEST_CASE("the key is 32 hex digits and matches the digest", "[fingerprint]")
{
  symex_target_equationt::SSA_stepst steps;
  add_step(steps, local("x", 100, 0), 7);

  const std::string key = ssa_cone_key_string(steps, fingerprint_modet::srcloc);
  REQUIRE(key.size() == 32);
  REQUIRE(key.find_first_not_of("0123456789abcdef") == std::string::npos);
  REQUIRE(
    key.substr(0, 16) ==
    fmt::format("{:016x}", ssa_cone_digest(steps, fingerprint_modet::srcloc)));
}
