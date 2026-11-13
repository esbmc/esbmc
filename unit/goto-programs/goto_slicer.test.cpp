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

slicedt slice_with(std::string src, const slice_criteriont &c)
{
  slicedt s;
  s.functions =
    goto_factory::get_goto_functions(src, goto_factory::Architecture::BIT_64)
      .functions;
  andersent points_to;
  points_to(s.functions);
  REQUIRE(!points_to.escapes());
  s.result = goto_slicert(s.functions, points_to).slice(c);
  return s;
}

slicedt slice(
  std::string src,
  slice_stept step = slice_stept::base_case,
  bool pointer_check = false,
  bool keep_nondets = false)
{
  slice_criteriont c;
  c.step = step;
  c.pointer_check = pointer_check;
  c.keep_nondets = keep_nondets;
  return slice_with(src, c);
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

namespace
{
/// The instructions of main in \p functions, in order.
std::vector<const goto_programt::instructiont *>
main_instructions(const goto_functionst &functions)
{
  std::vector<const goto_programt::instructiont *> result;
  for (const auto &i : functions.function_map.at("c:@F@main").body.instructions)
    result.push_back(&i);
  return result;
}

goto_functionst compile(std::string src)
{
  return goto_factory::get_goto_functions(
           src, goto_factory::Architecture::BIT_64)
    .functions;
}
} // namespace

TEST_CASE("goto-slicer rewrites a copy, not the original", "[goto-slicer]")
{
  goto_functionst functions = compile(R"(
    int a;
    int nondet_int();
    int main()
    {
      int x = 0;
      while (1)
        x++;
      int *p = &a;
      int v = nondet_int() + 1;
      *p = v;
      return 0;
    }
  )");
  functions.update();
  optionst options;
  andersent points_to;
  points_to(functions);
  goto_slice_cachet cache(functions, options, points_to, false, false);
  const goto_functionst &sliced = cache.program_for(slice_stept::base_case);
  REQUIRE(&sliced != &functions);

  const auto before = main_instructions(functions);
  const auto after = main_instructions(sliced);
  REQUIRE(before.size() == after.size());

  bool collapsed = false, zeroed = false;
  for (size_t i = 0; i < before.size(); ++i)
  {
    CHECK(before[i]->location_number == after[i]->location_number);
    if (before[i]->is_backwards_goto())
    {
      CHECK(after[i]->is_assume());
      collapsed = true;
    }
    if (
      before[i]->is_assign() &&
      is_dereference2t(to_code_assign2t(before[i]->code).target))
    {
      CHECK(!is_constant_int2t(to_code_assign2t(before[i]->code).source));
      const expr2tc &source = to_code_assign2t(after[i]->code).source;
      CHECK(is_constant_int2t(source));
      zeroed = true;
    }
  }
  CHECK(collapsed);
  CHECK(zeroed);
}

TEST_CASE(
  "goto-slicer keeps overwriting a pointer under leak checks",
  "[goto-slicer]")
{
  // g = 0 drops the last reference to the allocation: that is the leak.
  const std::string src = R"(
    #include <stdlib.h>
    int *g;
    int main()
    {
      g = malloc(sizeof(int));
      g = 0;
      return 0;
    }
  )";
  const auto overwrite =
    [](const slicedt &s) -> const goto_programt::instructiont & {
    const goto_programt::instructiont *last = nullptr;
    for (const auto &i : s.main_body().instructions)
      if (i.is_assign())
      {
        const expr2tc &target = to_code_assign2t(i.code).target;
        if (is_symbol2t(target) && to_symbol2t(target).thename == "c:@g")
          last = &i;
      }
    REQUIRE(last);
    return *last;
  };
  slice_criteriont c;
  c.pointer_check = false;
  c.memory_leak_check = true;
  c.no_reachable_memory_leak = true;
  const slicedt leak = slice_with(src, c);
  CHECK(!leak.removed(overwrite(leak)));
  const slicedt plain = slice(src);
  CHECK(plain.removed(overwrite(plain)));
}

TEST_CASE("goto-slicer keeps sprintf but not printf", "[goto-slicer]")
{
  // sprintf writes its buffer and symex checks that write; printf only prints.
  const auto src = [](const std::string &call) {
    return R"(
      #include <stdio.h>
      int main()
      {
        char buf[8];
        int x = 3;
        )" +
           call + R"(;
        return 0;
      }
    )";
  };
  const slicedt to_buffer = slice(src("sprintf(buf, \"%d\", x)"));
  CHECK(!to_buffer.removed(to_buffer.assignment("x")));
  const slicedt to_stdout = slice(src("printf(\"%d\", x)"));
  CHECK(to_stdout.removed(to_stdout.assignment("x")));
}

TEST_CASE(
  "goto-slicer keeps an index into a dereferenced object",
  "[goto-slicer]")
{
  // The pointer check bounds the whole offset, the array index included.
  const std::string src = R"(
    struct S { int v[2]; };
    int main()
    {
      struct S s;
      struct S *p = &s;
      int i = 1;
      p->v[i] = 0;
      return 0;
    }
  )";
  const slicedt checked = slice(src, slice_stept::base_case, true);
  CHECK(!checked.removed(checked.assignment("i")));
  const slicedt unchecked = slice(src);
  CHECK(unchecked.removed(unchecked.assignment("i")));
}

TEST_CASE(
  "goto-slicer counts an address Andersen loses as taken",
  "[goto-slicer]")
{
  // &x escapes through a double, which Andersen widens without recording x.
  const auto src = [](const char *checked) {
    return std::string(R"(
      int x, y;
      int main()
      {
        double d = (double)(unsigned long)&x;
        int *p = (int *)(unsigned long)d;
        *p = 5;
        __ESBMC_assert()") +
           checked + R"( == 0, "");
        return 0;
      }
    )";
  };
  const slicedt reached = slice(src("x"));
  CHECK(!reached.removed(reached.store()));
  const slicedt unrelated = slice(src("y"));
  CHECK(unrelated.removed(unrelated.store()));
}
