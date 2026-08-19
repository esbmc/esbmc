#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <fmt/format.h>
#include <util/message/message.h>
#include <util/ssa/fingerprint.h>

// capture_stderr redirects fd 2; the POSIX spellings live in <io.h> on MSVC.
#ifdef _WIN32
#  include <io.h>
#  define dup _dup
#  define dup2 _dup2
#  define close _close
#  define fileno _fileno
#  define STDERR_FILENO 2
#else
#  include <unistd.h>
#endif
#include <cstdio>
#include <functional>

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

std::string capture_stderr(const std::function<void()> &body)
{
  fflush(stderr);
  const int saved = dup(STDERR_FILENO);
  FILE *sink = tmpfile();
  dup2(fileno(sink), STDERR_FILENO);

  body();

  fflush(stderr);
  dup2(saved, STDERR_FILENO);
  close(saved);

  std::string text;
  rewind(sink);
  char buffer[4096];
  for (size_t n; (n = fread(buffer, 1, sizeof(buffer), sink)) > 0;)
    text.append(buffer, n);
  fclose(sink);
  return text;
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
  symex_target_equationt::SSA_stepst before, after;
  add_typed_step(before, "anon_struct_at_t.c_3_9");
  // The same type after an edit inserted a line above its declaration.
  add_typed_step(after, "anon_struct_at_t.c_5_9");

  REQUIRE(
    ssa_cone_digest(before, fingerprint_modet::srcloc) ==
    ssa_cone_digest(after, fingerprint_modet::srcloc));
  REQUIRE(
    ssa_cone_digest(before, fingerprint_modet::raw) !=
    ssa_cone_digest(after, fingerprint_modet::raw));
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
