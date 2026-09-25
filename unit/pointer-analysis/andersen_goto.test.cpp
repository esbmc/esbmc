/// \file
/// Integration tests for andersent::collect_constraints.
///
/// Unlike andersen.test.cpp, which drives the solver from synthetic
/// constraints, these compile real C through the clang frontend and run the
/// whole analysis over the resulting goto-functions.
///
/// Run just these:
///   ctest -R "andersen frontend" --output-on-failure

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../testing-utils/goto_factory.h"
#include <pointer-analysis/andersen.h>
#include <irep2/irep2_utils.h>

namespace
{
bool ends_with(const std::string &s, const std::string &tail)
{
  return s.size() >= tail.size() &&
         s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

/// Locates the local \p name of function \p func, or a file-scope global when
/// \p func is empty, so the tests never hard-code a frontend mangling.  Locals
/// are scoped to one function because the linked libc models contain plenty of
/// same-named ones; globals are told apart by carrying no `@F@` scope.  The
/// expression found is the very one the analysis interned, so it is also the
/// right key to query with.
expr2tc find_symbol(
  const goto_functionst &functions,
  const std::string &func,
  const std::string &name)
{
  const bool global = func.empty();
  expr2tc found;

  std::function<void(const expr2tc &)> search = [&](const expr2tc &e) {
    if (is_nil_expr(e) || !is_nil_expr(found))
      return;
    if (is_symbol2t(e))
    {
      const std::string id = to_symbol2t(e).get_symbol_name();
      if (
        global
          ? ends_with(id, "@" + name) && id.find("@F@") == std::string::npos
          : ends_with(id, "@" + func + "@" + name))
        found = e;
      return;
    }
    e->foreach_operand([&search](const expr2tc &op) { search(op); });
  };

  forall_goto_functions (f_it, functions)
    if (global || ends_with(f_it->first.as_string(), "@" + func))
      forall_goto_program_instructions (i_it, f_it->second.body)
        search(i_it->code);

  return found;
}

/// The set of pointees the analysis reports for a local, as their symbol
/// names.  An `unknown` (TOP) target is reported as "*".
std::set<std::string> targets_of(
  andersent &a,
  const goto_functionst &functions,
  const char *func,
  const char *name)
{
  const expr2tc sym = find_symbol(functions, func, name);
  REQUIRE(!is_nil_expr(sym));

  value_setst::valuest values;
  a.get_values(goto_programt::const_targett(), sym, values);

  std::set<std::string> result;
  for (const expr2tc &v : values)
  {
    if (!is_object_descriptor2t(v))
    {
      result.insert("*");
      continue;
    }
    const expr2tc &obj = to_object_descriptor2t(v).object;
    if (is_dynamic_object2t(obj))
      result.insert("heap");
    else if (is_symbol2t(obj))
    {
      const std::string id = to_symbol2t(obj).get_symbol_name();
      result.insert(id.substr(id.rfind('@') + 1));
    }
    else
      result.insert("?");
  }
  return result;
}

goto_functionst compile(std::string src)
{
  return goto_factory::get_goto_functions(
           src, goto_factory::Architecture::BIT_64)
    .functions;
}
} // namespace

TEST_CASE("andersen frontend lowers address-of and copy", "[andersen][goto]")
{
  std::string src = R"(
    int a, b;
    int main(void)
    {
      int *p = &a;
      int *q = p;
      int *z = &b;
      return 0;
    }
  )";

  goto_functionst functions = compile(src);
  andersent andersen;
  andersen(functions);

  REQUIRE(
    targets_of(andersen, functions, "main", "p") == std::set<std::string>{"a"});
  REQUIRE(
    targets_of(andersen, functions, "main", "q") == std::set<std::string>{"a"});
  REQUIRE(
    targets_of(andersen, functions, "main", "z") == std::set<std::string>{"b"});
}

TEST_CASE("andersen frontend lowers load and store", "[andersen][goto]")
{
  // The classic mixed example, but expressed as C and lowered by the frontend:
  // s reads what p points to, and the store through r adds b to p.
  std::string src = R"(
    int a, b;
    int main(void)
    {
      int *p = &a;
      int **r = &p;
      int *s = *r;
      *r = &b;
      return 0;
    }
  )";

  goto_functionst functions = compile(src);
  andersent andersen;
  andersen(functions);

  REQUIRE(
    targets_of(andersen, functions, "main", "r") == std::set<std::string>{"p"});
  REQUIRE(
    targets_of(andersen, functions, "main", "p") ==
    std::set<std::string>{"a", "b"});
  // Flow insensitivity: the store is visible to the earlier load.
  REQUIRE(
    targets_of(andersen, functions, "main", "s") ==
    std::set<std::string>{"a", "b"});
}

TEST_CASE(
  "andersen frontend interns one node per allocation site",
  "[andersen][goto]")
{
  std::string src = R"(
    void *malloc(unsigned long);
    int main(void)
    {
      int *p = (int *)malloc(sizeof(int));
      int *q = p;
      return 0;
    }
  )";

  goto_functionst functions = compile(src);
  andersent andersen;
  andersen(functions);

  REQUIRE(
    targets_of(andersen, functions, "main", "p") ==
    std::set<std::string>{"heap"});
  REQUIRE(
    targets_of(andersen, functions, "main", "q") ==
    std::set<std::string>{"heap"});
}

TEST_CASE("andersen frontend binds arguments and returns", "[andersen][goto]")
{
  std::string src = R"(
    int a;
    int *id(int *x) { return x; }
    int main(void)
    {
      int *p = &a;
      int *q = id(p);
      return 0;
    }
  )";

  goto_functionst functions = compile(src);
  andersent andersen;
  andersen(functions);

  REQUIRE(
    targets_of(andersen, functions, "id", "x") == std::set<std::string>{"a"});
  REQUIRE(
    targets_of(andersen, functions, "main", "q") == std::set<std::string>{"a"});
}

TEST_CASE(
  "andersen frontend widens an unmodelled callee to TOP",
  "[andersen][goto]")
{
  // The soundness case: nothing in the program says what ext() returns or what
  // it does through the pointer it is handed.  Both must widen to TOP rather
  // than stay empty, or a consumer would wrongly conclude "nothing aliases".
  std::string src = R"(
    int a;
    int *ext(int **out);
    int main(void)
    {
      int *p = &a;
      int **r = &p;
      int *e = ext(r);
      return 0;
    }
  )";

  goto_functionst functions = compile(src);
  andersent andersen;
  andersen(functions);

  REQUIRE(
    targets_of(andersen, functions, "main", "e") == std::set<std::string>{"*"});
  // ext() may write through r, so p may now point anywhere too.
  REQUIRE(targets_of(andersen, functions, "main", "p").count("*") == 1);
}

TEST_CASE(
  "andersen frontend routes a pointer laundered through an integer to TOP",
  "[andersen][goto]")
{
  // Nothing tracks the arithmetic an integer holding an address may undergo,
  // so the value cast back out of x names any object.  Dropping the store into
  // x would leave it empty and q would inherit that empty set.
  std::string src = R"(
    int a;
    int main(void)
    {
      int *p = &a;
      unsigned long x = (unsigned long)p;
      int *q = (int *)x;
      return 0;
    }
  )";

  goto_functionst functions = compile(src);
  andersent andersen;
  andersen(functions);

  REQUIRE(
    targets_of(andersen, functions, "main", "q") == std::set<std::string>{"*"});
  // The laundering must not cost precision on the pointer itself.
  REQUIRE(
    targets_of(andersen, functions, "main", "p") == std::set<std::string>{"a"});
}

TEST_CASE(
  "andersen frontend widens a nondet pointer to TOP",
  "[andersen][goto]")
{
  // A call to a bodyless function is lowered to a nondet side effect: the
  // resulting pointer is an unconstrained value symbolic execution may equate
  // with the address of any object, so an empty set would under-approximate.
  std::string src = R"(
    int *ext(void);
    int main(void)
    {
      int *n = ext();
      return 0;
    }
  )";

  goto_functionst functions = compile(src);
  andersent andersen;
  andersen(functions);

  REQUIRE(
    targets_of(andersen, functions, "main", "n") == std::set<std::string>{"*"});
}

TEST_CASE(
  "andersen frontend widens what an OTHER statement writes through",
  "[andersen][goto]")
{
  // The printf family is lowered to an OTHER instruction rather than a call,
  // so skipping those would claim s still holds the null it was initialised
  // with -- while asprintf() stores a freshly allocated buffer into it.
  std::string src = R"(
    int asprintf(char **, const char *, ...);
    int main(void)
    {
      char *s = 0;
      asprintf(&s, "%d", 1);
      return 0;
    }
  )";

  goto_functionst functions = compile(src);
  andersent andersen;
  andersen(functions);

  REQUIRE(
    targets_of(andersen, functions, "main", "s") == std::set<std::string>{"*"});
}

TEST_CASE(
  "andersen frontend is field- and index-insensitive",
  "[andersen][goto]")
{
  std::string src = R"(
    int a, b;
    struct s { int *f; int *g; };
    int main(void)
    {
      struct s v;
      int *arr[4];
      v.f = &a;
      v.g = &b;
      arr[2] = &a;
      int *p = v.f;
      int *q = arr[0];
      return 0;
    }
  )";

  goto_functionst functions = compile(src);
  andersent andersen;
  andersen(functions);

  // Both fields fold onto v, so reading either yields the union.
  REQUIRE(
    targets_of(andersen, functions, "main", "p") ==
    std::set<std::string>{"a", "b"});
  REQUIRE(
    targets_of(andersen, functions, "main", "q") == std::set<std::string>{"a"});
}

TEST_CASE(
  "andersen frontend resolves calls through function pointers",
  "[andersen][goto]")
{
  // fp reaches both foo and bar, so the argument must flow into both formals
  // and out through g.  Leaving the formals empty would claim g aliases
  // nothing, which is an under-approximation: at run time g == &a.
  std::string src = R"(
    int a;
    int *g;
    void foo(int *x) { g = x; }
    void bar(int *y) { g = y; }
    int main(void)
    {
      void (*fp)(int *) = foo;
      if (a) fp = bar;
      fp(&a);
      return 0;
    }
  )";

  goto_functionst functions = compile(src);
  andersent andersen;
  andersen(functions);

  REQUIRE(
    targets_of(andersen, functions, "main", "fp") ==
    std::set<std::string>{"foo", "bar"});
  REQUIRE(
    targets_of(andersen, functions, "foo", "x") == std::set<std::string>{"a"});
  REQUIRE(
    targets_of(andersen, functions, "bar", "y") == std::set<std::string>{"a"});
  REQUIRE(
    targets_of(andersen, functions, "", "g") == std::set<std::string>{"a"});
}

TEST_CASE(
  "andersen frontend widens an unknown function pointer",
  "[andersen][goto]")
{
  // Nothing says what fp points to, so every callee is possible: the result
  // must be TOP rather than an empty set.
  std::string src = R"(
    int a;
    int *g;
    void (*volatile ext_fp)(int *);
    int main(void)
    {
      void (*fp)(int *) = ext_fp;
      int *p = &a;
      fp(p);
      return 0;
    }
  )";

  goto_functionst functions = compile(src);
  andersent andersen;
  andersen(functions);

  // p's pointee must be assumed clobbered by the unknown callee.
  REQUIRE(targets_of(andersen, functions, "", "a").count("*") == 1);
}
