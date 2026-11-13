/// \file
/// Tests for goto_slicert on C compiled through the clang frontend.
///
/// Run just these:
///   ctest -R goto-slicer --output-on-failure

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../testing-utils/goto_factory.h"
#include <goto-programs/goto_slicer.h>
#include <pointer-analysis/andersen.h>

namespace
{
bool ends_with(const std::string &s, const std::string &tail)
{
  return s.size() >= tail.size() &&
         s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

struct slicedt
{
  goto_functionst functions;
  slice_resultt result;

  const goto_programt &main_body() const
  {
    for (const auto &[name, f] : functions.function_map)
      if (name == "c:@F@main")
        return f.body;
    FAIL("no main");
    abort();
  }

  /// The first assignment in main to the local \p name.
  const goto_programt::instructiont &assignment(const std::string &name) const
  {
    for (const auto &i : main_body().instructions)
      if (i.is_assign())
      {
        const expr2tc &target = to_code_assign2t(i.code).target;
        if (
          is_symbol2t(target) &&
          ends_with(to_symbol2t(target).thename.as_string(), "@" + name))
          return i;
      }
    FAIL("no assignment to " + name);
    abort();
  }

  /// The first instruction in main whose code mentions a dereference.
  const goto_programt::instructiont &store() const
  {
    for (const auto &i : main_body().instructions)
      if (i.is_assign() && is_dereference2t(to_code_assign2t(i.code).target))
        return i;
    FAIL("no store");
    abort();
  }

  const goto_programt::instructiont &call(const std::string &callee) const
  {
    for (const auto &i : main_body().instructions)
      if (i.is_function_call())
      {
        const expr2tc &f = to_code_function_call2t(i.code).function;
        if (is_symbol2t(f) && to_symbol2t(f).thename == "c:@F@" + callee)
          return i;
      }
    FAIL("no call to " + callee);
    abort();
  }

  bool removed(const goto_programt::instructiont &i) const
  {
    return result.removed.count(&i) != 0;
  }
};

slicedt slice(
  std::string src,
  slice_stept step = slice_stept::base_case,
  bool pointer_check = false,
  bool keep_nondets = false)
{
  slicedt s;
  s.functions =
    goto_factory::get_goto_functions(src, goto_factory::Architecture::BIT_64)
      .functions;
  andersent points_to;
  points_to(s.functions);
  REQUIRE(!points_to.escapes());

  slice_criteriont c;
  c.step = step;
  c.pointer_check = pointer_check;
  c.no_assertions = step == slice_stept::forward_condition;
  c.keep_nondets = keep_nondets;
  s.result = goto_slicert(s.functions, points_to).slice(c);
  return s;
}
} // namespace

TEST_CASE("goto-slicer removes assignments no property reads", "[goto-slicer]")
{
  const slicedt s = slice(R"(
    int nondet_int();
    int main()
    {
      int x = nondet_int();
      int y = x * 2;
      __ESBMC_assert(x != 3, "");
      return 0;
    }
  )");
  CHECK(s.removed(s.assignment("y")));
  CHECK(!s.removed(s.assignment("x")));
}

TEST_CASE("goto-slicer keeps nondets for a witness", "[goto-slicer]")
{
  const std::string src = R"(
    int nondet_int();
    int main()
    {
      int y = nondet_int();
      __ESBMC_assert(1, "");
      return 0;
    }
  )";
  // The nondet lands in a return_value$ temporary that y copies.
  const auto nondet =
    [](const slicedt &s) -> const goto_programt::instructiont & {
    for (const auto &i : s.main_body().instructions)
      if (i.is_assign() && is_sideeffect2t(to_code_assign2t(i.code).source))
        return i;
    FAIL("no nondet");
    abort();
  };
  const slicedt plain = slice(src);
  CHECK(plain.removed(nondet(plain)));
  const slicedt witness = slice(src, slice_stept::base_case, false, true);
  CHECK(!witness.removed(nondet(witness)));
  CHECK(witness.removed(witness.assignment("y")));
}

TEST_CASE(
  "goto-slicer collapses a loop no property depends on",
  "[goto-slicer]")
{
  const slicedt s = slice(R"(
    int main()
    {
      int x = 0;
      while (1)
        x++;
      __ESBMC_assert(0, "");
      return 0;
    }
  )");
  CHECK(s.result.loops == 1);
  CHECK(s.result.collapsed_loops == 1);
  CHECK(s.removed(s.assignment("x")));
}

TEST_CASE("goto-slicer keeps a loop that feeds an assertion", "[goto-slicer]")
{
  const slicedt s = slice(R"(
    int main()
    {
      int x = 0;
      while (1)
        x++;
      __ESBMC_assert(x != 5, "");
      return 0;
    }
  )");
  CHECK(s.result.collapsed_loops == 0);
  CHECK(!s.removed(s.assignment("x")));
}

TEST_CASE(
  "goto-slicer keeps a loop whose exit test reads its body",
  "[goto-slicer]")
{
  const slicedt s = slice(R"(
    int main()
    {
      int i = 0;
      while (i < 10)
        i++;
      return 0;
    }
  )");
  CHECK(s.result.collapsed_loops == 0);
  CHECK(!s.removed(s.assignment("i")));
}

TEST_CASE(
  "goto-slicer removes and keeps calls by their effect",
  "[goto-slicer]")
{
  const std::string callee = R"(
    int g, h;
    void f() { h = 1; }
    void k() { g = 1; }
  )";
  const slicedt s = slice(callee + R"(
    int main()
    {
      f();
      k();
      __ESBMC_assert(g == 1, "");
      return 0;
    }
  )");
  CHECK(s.removed(s.call("f")));
  CHECK(!s.removed(s.call("k")));
}

TEST_CASE("goto-slicer always keeps a call into recursion", "[goto-slicer]")
{
  const slicedt s = slice(R"(
    int h;
    void r(int n) { if (n > 0) { h = n; r(n - 1); } }
    int main()
    {
      r(3);
      __ESBMC_assert(1, "");
      return 0;
    }
  )");
  CHECK(!s.removed(s.call("r")));
}

TEST_CASE("goto-slicer follows writes through pointers", "[goto-slicer]")
{
  const slicedt s = slice(R"(
    int a, b;
    int main()
    {
      int *p = &a;
      int *q = &b;
      *p = 5;
      *q = 7;
      __ESBMC_assert(a == 5, "");
      return 0;
    }
  )");
  CHECK(!s.removed(s.store()));
  CHECK(!s.removed(s.assignment("p")));
  CHECK(s.removed(s.assignment("q")));
}

TEST_CASE(
  "goto-slicer keeps a store only for its pointer check",
  "[goto-slicer]")
{
  const std::string src = R"(
    int a;
    int nondet_int();
    int main()
    {
      int *p = &a;
      int v = nondet_int() + 1;
      *p = v;
      return 0;
    }
  )";
  const slicedt checked = slice(src, slice_stept::base_case, true);
  CHECK(checked.result.check_only.count(&checked.store()) == 1);
  CHECK(!checked.removed(checked.assignment("p")));
  CHECK(checked.removed(checked.assignment("v")));

  const slicedt unchecked = slice(src);
  CHECK(unchecked.removed(unchecked.store()));
}

TEST_CASE(
  "goto-slicer forward condition drops user assertions",
  "[goto-slicer]")
{
  const std::string src = R"(
    int nondet_int();
    int main()
    {
      int x = nondet_int();
      __ESBMC_assert(x != 3, "");
      return 0;
    }
  )";
  const slicedt base = slice(src);
  CHECK(!base.removed(base.assignment("x")));
  const slicedt forward = slice(src, slice_stept::forward_condition);
  CHECK(forward.removed(forward.assignment("x")));
}

TEST_CASE("goto-slicer resolves calls through pointers", "[goto-slicer]")
{
  const auto indirect_call =
    [](const slicedt &s) -> const goto_programt::instructiont & {
    for (const auto &i : s.main_body().instructions)
      if (
        i.is_function_call() &&
        !is_symbol2t(to_code_function_call2t(i.code).function))
        return i;
    FAIL("no indirect call");
    abort();
  };
  const std::string callees = R"(
    int g, h;
    void f() { h = 1; }
    void k() { g = 1; }
  )";
  const slicedt irrelevant = slice(callees + R"(
    int main()
    {
      void (*fp)(void) = f;
      fp();
      __ESBMC_assert(g == 0, "");
      return 0;
    }
  )");
  CHECK(irrelevant.removed(indirect_call(irrelevant)));

  const slicedt relevant = slice(callees + R"(
    int main()
    {
      void (*fp)(void) = k;
      fp();
      __ESBMC_assert(g == 0, "");
      return 0;
    }
  )");
  CHECK(!relevant.removed(indirect_call(relevant)));
}

TEST_CASE("goto-slicer keeps a stored index a check reads", "[goto-slicer]")
{
  // a[1] = 5 writes no pointer, but p[a[1]] reads it as an index, so the
  // pointer check depends on it.
  const std::string src = R"(
    int buf[8];
    int main()
    {
      int a[2];
      int *p = buf;
      a[1] = 5;
      p[a[1]] = 0;
      return 0;
    }
  )";
  const auto store_to_a =
    [](const slicedt &s) -> const goto_programt::instructiont & {
    for (const auto &i : s.main_body().instructions)
      if (i.is_assign() && is_index2t(to_code_assign2t(i.code).target))
      {
        const expr2tc &source =
          to_index2t(to_code_assign2t(i.code).target).source_value;
        if (
          is_symbol2t(source) &&
          ends_with(to_symbol2t(source).thename.as_string(), "@a"))
          return i;
      }
    FAIL("no store to a");
    abort();
  };
  const slicedt checked = slice(src, slice_stept::base_case, true);
  CHECK(!checked.removed(store_to_a(checked)));
  const slicedt unchecked = slice(src);
  CHECK(unchecked.removed(store_to_a(unchecked)));
}
