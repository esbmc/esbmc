/*******************************************************************
 Module: Call-frame lifecycle on the real engine

 Tier B of docs/roadmap/goto-symex-verification-plan.md (H-A7, milestone M1).

 R7 concerns `goto_symex_statet::previous_frame()`, which evaluates
 `*(--(--call_stack.end()))` with no size check. `call_stackt` is a
 `std::vector<framet>`, so at size 1 that forms `begin() - 1` — undefined by
 [expr.add], not merely a bad read.

 It has exactly one call site,
 `goto_symext::symex_function_call_code` (symex_function.cpp), which does
 `new_frame(...)` on the line before, so the precondition holds by
 construction — but the only thing stating it is an `assert` that is a no-op in
 the shipped binary (R1). These tests exercise the paths that reach it, and pin
 the frame-balance property that a violation would have to break first.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <set>
#include <string>

#include <goto-symex/reachability_tree.h>
#include <irep2/irep2_expr.h>
#include <util/symtab/namespace.h>

#include "ssa_validator.h"
#include "../testing-utils/goto_factory.h"

namespace
{
/** A real engine over `src`, kept alive so its call stack can be inspected. */
class engine
{
public:
  explicit engine(std::string src)
    : source(std::move(src)),
      prog(goto_factory::get_goto_functions(
        source,
        goto_factory::Architecture::BIT_64)),
      ns(prog.context),
      opts(goto_factory::get_default_options(
        goto_factory::get_default_cmdline("test.c"))),
      rt(
        prog.functions,
        ns,
        opts,
        std::make_shared<symex_target_equationt>(ns),
        prog.context)
  {
    opts.set_option("unwind", "4");
    rt.setup_for_new_explore();
  }

  size_t call_stack_depth()
  {
    return rt.get_cur_state().get_active_state().call_stack.size();
  }

  /** Frames left standing when symex stops: __ESBMC_main's and main's, whose
   *  END_FUNCTION is the last instruction executed. Everything nested inside
   *  has been popped, so this is a constant of the entry sequence and not a
   *  function of how deeply the program calls. */
  static constexpr size_t residual_depth = 2;

  std::shared_ptr<symex_target_equationt> run()
  {
    auto eq = std::dynamic_pointer_cast<symex_target_equationt>(
      rt.get_next_formula().target);
    REQUIRE(eq != nullptr);
    // §7.2: every equation a Tier-B test builds is another I1/I10/P11 sample.
    symex_ssa::require_well_formed(*eq);
    return eq;
  }

private:
  std::string source;
  program prog;
  namespacet ns;
  optionst opts;
  reachability_treet rt;
};

/** L1 activation numbers seen for definitions of `base_name`. */
std::set<unsigned>
l1_activations(const symex_target_equationt &eq, const std::string &base_name)
{
  std::set<unsigned> seen;
  for (const auto &step : eq.SSA_steps)
  {
    if (!step.is_assignment() || !is_symbol2t(step.lhs))
      continue;
    const symbol2t &sym = to_symbol2t(step.lhs);
    if (sym.thename.as_string().find(base_name) != std::string::npos)
      seen.insert(sym.level1_num);
  }
  return seen;
}
} // namespace

TEST_CASE("nested calls leave the frame stack balanced", "[symex][frame]")
{
  engine e(R"(
int leaf(int a) { return a + 1; }
int mid(int a) { return leaf(a) + leaf(a + 1); }
int main(void) { return mid(2); }
)");

  // previous_frame()'s precondition is `size() >= 2`, established by the
  // new_frame() immediately preceding it. Depth 1 before any call is what makes
  // that new_frame the second frame rather than the first.
  REQUIRE(e.call_stack_depth() == 1);

  auto eq = e.run();
  REQUIRE(eq->SSA_steps.size() > 0);
  REQUIRE(e.call_stack_depth() == engine::residual_depth);
}

TEST_CASE("recursion gives each activation its own L1 number", "[symex][frame]")
{
  engine e(R"(
int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }
int main(void) { return fact(4); }
)");

  auto eq = e.run();

  // This is what previous_frame() exists for: symex_function_call_code seeds
  // the new frame's L1 map from the caller's, so a recursive local gets a
  // fresh activation rather than aliasing the outer one.
  REQUIRE(l1_activations(*eq, "fact").size() > 1);
  REQUIRE(e.call_stack_depth() == engine::residual_depth);
}

TEST_CASE("calls through a function pointer balance frames", "[symex][frame]")
{
  // run_next_function_ptr_target re-enters symex_function_call_code once per
  // candidate target, so this drives previous_frame() on the path where the
  // frame stack is manipulated most.
  engine e(R"(
int add_one(int a) { return a + 1; }
int add_two(int a) { return a + 2; }
int nondet_int(void);
int main(void)
{
  int (*f)(int) = nondet_int() > 0 ? add_one : add_two;
  return f(1);
}
)");

  auto eq = e.run();
  REQUIRE(eq->SSA_steps.size() > 0);
  REQUIRE(e.call_stack_depth() == engine::residual_depth);
}

TEST_CASE("calls inside an unwound loop balance frames", "[symex][frame]")
{
  engine e(R"(
int step(int a) { return a * 2 + 1; }
int main(void)
{
  int x = 1;
  for (int i = 0; i < 3; i++)
    x = step(x);
  return x;
}
)");

  auto eq = e.run();
  REQUIRE(l1_activations(*eq, "step").size() >= 1);
  REQUIRE(e.call_stack_depth() == engine::residual_depth);
}

TEST_CASE(
  "the residual frame depth does not grow with call nesting",
  "[symex][frame]")
{
  // The discriminating case for frame balance: if pop_frame did not match
  // new_frame, the deeper program would end with more frames standing. Both
  // recursions are fully unwound at --unwind 4, so they differ only in depth.
  auto depth_after = [](const char *body) {
    engine e(body);
    e.run();
    return e.call_stack_depth();
  };

  const size_t shallow = depth_after(R"(
int down(int n) { return n <= 0 ? 0 : down(n - 1); }
int main(void) { return down(1); }
)");
  const size_t deep = depth_after(R"(
int down(int n) { return n <= 0 ? 0 : down(n - 1); }
int main(void) { return down(3); }
)");

  REQUIRE(shallow == deep);
  REQUIRE(deep == engine::residual_depth);
}
