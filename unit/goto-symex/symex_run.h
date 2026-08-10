/*******************************************************************
 Module: One symex run, owning the context its equation refers to

 Shared by the Tier-B tests that need a produced equation and nothing else
 (docs/roadmap/goto-symex-verification-plan.md §7.2).

 `symex_target_equationt` holds its `namespacet` **by reference**, so the whole
 parse/context/options bundle has to outlive the equation. Tests that compare or
 rewrite two equations therefore need one of these per equation, which the
 `engine` fixture in the other Tier-B files cannot express: it exposes the live
 symex state and so owns a single run.

 Not merged with that `engine` yet, deliberately: `invariant`, `renaming` and
 `frame_lifecycle` read the live state *before* `run()` and so need
 `setup_for_new_explore()` in the constructor, while `unwind` defers it so its
 option setters take effect first. Reconciling those is a separate change
 -- see the cleanup note in §15 M4 (H-B2).

 \*******************************************************************/

#pragma once

// Include after the consumer's own `#define CATCH_CONFIG_MAIN`, per
// ssa_validator.h.
#include <catch2/catch.hpp>

#include <memory>
#include <string>

#include <goto-programs/goto_k_induction.h>
#include <goto-programs/remove_no_op.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/symex_target_equation.h>
#include <util/symtab/namespace.h>

#include "../testing-utils/goto_factory.h"

namespace symex_run
{
class equation
{
public:
  explicit equation(const std::string &src, const char *unwind = "4")
    : source(src),
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
    opts.set_option("unwind", unwind);
    rt.setup_for_new_explore();
    produced = std::dynamic_pointer_cast<symex_target_equationt>(
      rt.get_next_formula().target);
    REQUIRE(produced != nullptr);
  }

  symex_target_equationt &get()
  {
    return *produced;
  }

  const symex_target_equationt &get() const
  {
    return *produced;
  }

  const optionst &options() const
  {
    return opts;
  }

private:
  std::string source;
  program prog;
  namespacet ns;
  optionst opts;
  reachability_treet rt;
  std::shared_ptr<symex_target_equationt> produced;
};

/// The same run, with the k-induction goto transform applied and the
/// inductive step selected. Separate from `equation` because it is the only
/// way a Tier-B harness can reach code guarded on
/// `pc->inductive_step_instruction`: nothing but `goto_k_induction` sets that
/// flag, so a plain `goto_factory` program never carries it.
class inductive_step_equation
{
public:
  explicit inductive_step_equation(
    const std::string &src,
    const char *unwind = "4")
    : source(src),
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
    // The order `process_goto_program` uses: the pass expects the SKIPs gone.
    remove_no_op(prog.functions);
    goto_k_induction(prog.functions, ns);

    opts.set_option("unwind", unwind);
    opts.set_option("inductive-step", true);
    opts.set_option("add-symex-value-sets", true);

    rt.setup_for_new_explore();
    produced = std::dynamic_pointer_cast<symex_target_equationt>(
      rt.get_next_formula().target);
    REQUIRE(produced != nullptr);
  }

  const symex_target_equationt &get() const
  {
    return *produced;
  }

private:
  std::string source;
  program prog;
  namespacet ns;
  optionst opts;
  reachability_treet rt;
  std::shared_ptr<symex_target_equationt> produced;
};
} // namespace symex_run
