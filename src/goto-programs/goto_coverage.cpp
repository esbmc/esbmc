#include <goto-programs/goto_coverage.h>
#include <goto-programs/k_path_spanning.h>
#include <irep2/irep2_utils.h>

#include <algorithm>
#include <cassert>
#include <deque>
#include <vector>

size_t goto_coveraget::total_assert = 0;
size_t goto_coveraget::total_assert_ins = 0;
std::set<std::pair<std::string, std::string>> goto_coveraget::total_cond;
size_t goto_coveraget::total_branch = 0;
size_t goto_coveraget::total_func_branch = 0;
size_t goto_coveraget::total_kpath = 0;
size_t goto_coveraget::total_kpath_spanning = 0;
std::set<std::pair<std::string, std::string>>
  goto_coveraget::k_path_spanning_redundant;
std::set<std::pair<std::string, std::string>> goto_coveraget::all_claims;

std::string goto_coveraget::get_filename_from_path(std::string path)
{
  if (path.find_last_of('/') != std::string::npos)
    return path.substr(path.find_last_of('/') + 1);

  return path;
}

/*
  replace the old_condition of all assertions
  to the new condition(guard)
*/
void goto_coveraget::replace_all_asserts_to_guard(
  const expr2tc &guard,
  bool is_instrumentation)
{
  std::unordered_set<std::string> location_pool = {};
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      Forall_goto_program_instructions (it, goto_program)
      {
        std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->is_assert())
          replace_assert_to_guard(guard, it, is_instrumentation);
      }
    }
}

/*
  replace the old_condition of a specific assertion
  to the new condition(guard)
*/
void goto_coveraget::replace_assert_to_guard(
  const expr2tc &guard,
  goto_programt::instructiont::targett &it,
  bool is_instrumentation)
{
  const expr2tc old_guard = it->guard;
  it->guard = guard;
  if (is_instrumentation)
    it->location.property("instrumented assertion");
  else
    it->location.property("replaced assertion");
  it->location.comment(from_expr(ns, "", old_guard));
  it->location.user_provided(true);
}

/*
  convert assert(cond) to assume(cond)
  preserving the original condition as a path constraint
*/
void goto_coveraget::replace_assert_to_assume(
  goto_programt::instructiont::targett &it)
{
  const expr2tc guard = it->guard;
  it->make_assumption(guard);
  it->location.property("replaced assertion");
  it->location.user_provided(true);
}

/*
  convert all assertions to assumptions
*/
void goto_coveraget::replace_all_asserts_to_assume()
{
  std::unordered_set<std::string> location_pool = {};
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      Forall_goto_program_instructions (it, goto_program)
      {
        std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->is_assert())
          replace_assert_to_assume(it);
      }
    }
}

/*
Algo:
- convert all assertions to false and enable multi-property
*/
void goto_coveraget::assertion_coverage()
{
  replace_all_asserts_to_guard(gen_false_expr(), true);
  total_assert = get_total_instrument();
  total_assert_ins = get_total_assert_instance();
  all_claims = get_total_cond_assert();
}

/*
Branch coverage applies to any control structure that can alter the flow of execution, including:
- if-else
- switch-case
- Loops (for, while, do-while)
- try-catch-finally (not in c)
- Early exits (return, break, continue)
The goal of branch coverage is to ensure that all possible execution paths in the program are tested.

The CBMC extends it to the entry of the function. So we will do the same.


Algo:
  1. convert assertions to true
  2. add false assertion add the beginning of the function and the branch()
*/
void goto_coveraget::branch_function_coverage()
{
  log_progress("Adding false assertions...");
  total_func_branch = 0;

  std::unordered_set<std::string> location_pool = {};
  // cmdline.arg[0]
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  std::unordered_set<int> catch_tgt_list;
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      bool flg = true;

      Forall_goto_program_instructions (it, goto_program)
      {
        std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        // skip if it's not the verifying files
        // probably a library
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (flg)
        {
          // add a false assert in the beginning
          // to check if the function is entered.
          insert_assert(
            goto_program,
            it,
            gen_false_expr(),
            "function entry: " + id2string(f_it->first));
          flg = false;
        }

        if (it->location.property().as_string() == "skipped")
          // this stands for the auxiliary condition/branch we added.
          continue;

        // convert assertions to true (or assume)
        if (
          it->is_assert() &&
          it->location.property().as_string() != "replaced assertion" &&
          it->location.property().as_string() != "instrumented assertion")
        {
          if (cov_assume_asserts)
            replace_assert_to_assume(it);
          else
            replace_assert_to_guard(gen_true_expr(), it, false);
        }

        // e.g. IF !(a > 1) THEN GOTO 3
        else if (it->is_goto() && !is_true(it->guard))
        {
          if (it->is_target())
            target_num = it->target_number;
          // assert(!(a > 1));
          // assert(a > 1);
          insert_assert(goto_program, it, it->guard);
          insert_assert(goto_program, it, gen_not_expr(it->guard));
        }
      }

      flg = true;
    }

  // fix for branch coverage with kind/incr
  // It seems in kind/incr, the goto_functions used during the BMC is simplified and incomplete
  total_func_branch = get_total_instrument();
  all_claims = get_total_cond_assert();

  // avoid Assertion `call_stack.back().goto_state_map.size() == 0' failed
  goto_functions.update();
}

void goto_coveraget::branch_coverage()
{
  log_progress("Adding false assertions...");
  total_branch = 0;

  std::unordered_set<std::string> location_pool = {};
  // cmdline.arg[0]
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  std::unordered_set<int> catch_tgt_list;
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      Forall_goto_program_instructions (it, goto_program)
      {
        std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        // skip if it's not the verifying files
        // probably a library
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->location.property().as_string() == "skipped")
          // this stands for the auxiliary condition/branch we added.
          continue;

        // convert assertions to true (or assume)
        if (
          it->is_assert() &&
          it->location.property().as_string() != "replaced assertion" &&
          it->location.property().as_string() != "instrumented assertion")
        {
          if (cov_assume_asserts)
            replace_assert_to_assume(it);
          else
            replace_assert_to_guard(gen_true_expr(), it, false);
        }

        // e.g. IF !(a > 1) THEN GOTO 3
        else if (it->is_goto() && !is_true(it->guard))
        {
          if (it->is_target())
            target_num = it->target_number;
          // assert(!(a > 1));
          // assert(a > 1);
          insert_assert(goto_program, it, it->guard);
          insert_assert(goto_program, it, gen_not_expr(it->guard));
        }
      }
    }

  total_branch = get_total_instrument();
  all_claims = get_total_cond_assert();

  // avoid Assertion `call_stack.back().goto_state_map.size() == 0' failed
  goto_functions.update();
}

// Post-simplification depth of an expression tree, capped early once the
// caller's threshold is exceeded. Used to gate emission of the structural
// witness (issue #4325).
static size_t expr_depth(const expr2tc &e, size_t cap)
{
  if (is_nil_expr(e))
    return 0;
  size_t n = e->get_num_sub_exprs();
  if (n == 0)
    return 1;
  size_t d = 0;
  for (size_t i = 0; i < n; ++i)
  {
    const expr2tc *sub = e->get_sub_expr(i);
    if (sub == nullptr)
      continue;
    d = std::max(d, expr_depth(*sub, cap));
    if (d > cap)
      return d + 1;
  }
  return 1 + d;
}

/*
k-path coverage (Phase 1 — see GitHub issue #4325).

For each branching `IF g GOTO L`, emit one coverage goal per combination of
the last (n-1) prior branch directions × the two outcomes of the current
branch. Each goal is `assert(!witness)` where `witness = d_1 ∧ … ∧ d_k`
with each d_i either a prior branch guard or its negation; multi_property
marks a goal as reached when the assertion is falsifiable, i.e. when the
corresponding path is feasible. This mirrors the existing branch_coverage
inversion convention.

Bounded by the textual order of branches within a function (cheap and
deterministic). Joins make this an *over-approximation* of true path
coverage — some witnesses may be infeasible and stay uncovered, which is
correct under the spanning-set scoring proposed in #4325.

Goal count per branch: 2^min(prefix_size+1, n). Capped per function by
`k_path_max_goals`; on overflow the instrumentation aborts with an
actionable error rather than silently truncating (decision locked in #4325).
*/
void goto_coveraget::k_path_coverage()
{
  log_progress("Adding k-path coverage assertions (n={})...", k_path_n);
  total_kpath = 0;
  total_kpath_spanning = 0;
  k_path_spanning_redundant.clear();
  k_path_spanning_sett spanning;

  // Defense-in-depth: parseoptions rejects N==0 and N>30 at the CLI, but
  // re-check here in case the method is invoked via another code path.
  // 30 keeps `1 << pdepth` well below the size_t shift limit and below
  // any reasonable goal cap.
  static constexpr size_t K_PATH_N_MAX = 30;
  if (k_path_n == 0 || k_path_n > K_PATH_N_MAX)
  {
    log_error(
      "--k-path-coverage requires 1 <= N <= {} (got {})",
      K_PATH_N_MAX,
      k_path_n);
    abort();
  }

  std::unordered_set<std::string> location_pool = {};
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available || f_it->first == "__ESBMC_main")
      continue;

    goto_programt &goto_program = f_it->second.body;
    if (filter(f_it->first, goto_program))
      continue;

    // Sliding window of the last (n-1) prior branch guards in textual order.
    // Reset per function: each function is its own k-path scope (#4325).
    std::deque<expr2tc> prefix;
    size_t function_goals = 0;

    Forall_goto_program_instructions (it, goto_program)
    {
      std::string cur_filename =
        get_filename_from_path(it->location.file().as_string());
      if (location_pool.count(cur_filename) == 0)
        continue;

      if (it->location.property().as_string() == "skipped")
        continue;

      // Mirror branch_coverage: neutralise existing assertions so they don't
      // confuse multi_property_check.
      if (
        it->is_assert() &&
        it->location.property().as_string() != "replaced assertion" &&
        it->location.property().as_string() != "instrumented assertion")
      {
        if (cov_assume_asserts)
          replace_assert_to_assume(it);
        else
          replace_assert_to_guard(gen_true_expr(), it, false);
        continue;
      }

      // Conditional forward branch. Backward unconditional gotos (loop
      // back-edges) carry guard=true and are skipped here; iteration
      // semantics are picked up later by ESBMC's --unwind unrolling.
      if (it->is_goto() && !is_true(it->guard))
      {
        if (it->is_target())
          target_num = it->target_number;

        const expr2tc current_guard = it->guard;
        // pdepth is bounded by k_path_n - 1 <= K_PATH_N_MAX - 1 = 29
        // (enforced above), so the shift below cannot overflow. Assert as
        // a tripwire — silent overflow would be unsound.
        const size_t pdepth = std::min(prefix.size(), k_path_n - 1);
        assert(pdepth < 30 && "pdepth bounded by parseoptions cap");
        const size_t pcombos = size_t(1) << pdepth;
        const size_t branch_goals = 2 * pcombos;

        if (
          branch_goals > k_path_max_goals ||
          function_goals > k_path_max_goals - branch_goals)
        {
          log_error(
            "k-path coverage: per-function goal count would exceed "
            "--k-path-max-goals={} in '{}'. Lower --k-path-coverage=N "
            "(currently {}) or raise --k-path-max-goals.",
            k_path_max_goals,
            id2string(f_it->first),
            k_path_n);
          abort();
        }

        // The deque is trimmed to ≤ (n-1) entries at the bottom of every
        // branch iteration, so its current contents are exactly the active
        // prefix.
        std::vector<expr2tc> active(prefix.begin(), prefix.end());

        for (size_t mask = 0; mask < pcombos; ++mask)
        {
          // Build the prefix witness for this direction mask, while
          // tracking (stored-guard, polarity) pairs so we can drop mask
          // combinations that are unsat by construction.
          //
          // ESBMC's `simplify` recognises 2-term `p ∧ ¬p` but does not
          // fold chained forms like `p ∧ q ∧ ¬p` to FALSE — it would
          // instrument a tautological `assert(¬(p ∧ q ∧ ¬p))` that can
          // never be falsified, permanently inflating the denominator.
          // Catching this at construction time is sound (we only drop
          // witnesses we can prove unsat by syntactic structure) and
          // preserves the single-term behaviour of the simplifier.
          //
          // Phase-1 limitation: this only catches *syntactic* same-atom
          // contradictions (same stored guard with opposing polarities).
          // Semantically contradictory pairs across different stored
          // expressions — e.g. `(x == 1) ∧ (x == 2)` from successive
          // switch-case branches — require comparison-domain reasoning
          // and are out of scope for this PR.
          expr2tc pwit;
          std::vector<std::pair<expr2tc, bool>> atoms;
          atoms.reserve(pdepth);
          bool contradictory = false;
          for (size_t i = 0; i < pdepth; ++i)
          {
            const bool pol = (mask & (size_t(1) << i)) != 0;
            for (const auto &[h, p] : atoms)
            {
              if (h == active[i] && p != pol)
              {
                contradictory = true;
                break;
              }
            }
            if (contradictory)
              break;
            atoms.emplace_back(active[i], pol);
            expr2tc d = pol ? active[i] : gen_not_expr(active[i]);
            pwit = is_nil_expr(pwit) ? d : gen_and_expr(pwit, d);
          }
          if (contradictory)
            continue;

          // Emit one goal per current direction. Skip the direction if
          // it would contradict an atom already in the prefix.
          const expr2tc current_neg = gen_not_expr(current_guard);
          for (size_t cd = 0; cd < 2; ++cd)
          {
            const bool cdir_pol = (cd == 0);
            bool cdir_conflict = false;
            for (const auto &[h, p] : atoms)
            {
              if (h == current_guard && p != cdir_pol)
              {
                cdir_conflict = true;
                break;
              }
            }
            if (cdir_conflict)
              continue;

            const expr2tc &cdir = cdir_pol ? current_guard : current_neg;
            expr2tc full = is_nil_expr(pwit) ? cdir : gen_and_expr(pwit, cdir);
            simplify(full);

            if (is_false(full))
              continue;
            if (is_true(full))
              continue;

            if (expr_depth(full, k_path_witness_depth) > k_path_witness_depth)
            {
              // Phase 1: drop witnesses past the depth cap. The hashed
              // ghost-flag fallback for deep prefixes is Phase 2 (#4325).
              continue;
            }

            expr2tc neg_full = gen_not_expr(full);
            simplify(neg_full);

            std::string idf = from_expr(ns, "", full);
            insert_assert(goto_program, it, neg_full, idf);

            // Record the goal's full atom multiset (prefix + current
            // direction) so the spanning-set analysis can drop subsumed
            // emissions from the coverage denominator.
            std::vector<std::pair<expr2tc, bool>> goal_atoms = atoms;
            goal_atoms.emplace_back(current_guard, cdir_pol);
            spanning.add_goal(
              std::move(goal_atoms), idf, it->location.as_string());

            ++function_goals;
          }
        }

        prefix.push_back(current_guard);
        if (prefix.size() > k_path_n - 1)
          prefix.pop_front();
      }
    }
  }

  total_kpath = get_total_instrument();
  all_claims = get_total_cond_assert();

  // Soundness invariant: each insert_assert call above paired with
  // exactly one spanning.add_goal call, so the number of goals tracked
  // in the spanning analysis must equal the number of instrumented
  // assertions counted in the goto programs. A divergence means the
  // emission path diverged from the spanning bookkeeping (e.g. a future
  // edit added an insert_assert without the matching add_goal, or vice
  // versa) and the spanning-set denominator would be silently wrong.
  // ESBMC is a verifier — we abort rather than report an unsound
  // coverage percentage.
  if (spanning.total() != static_cast<size_t>(total_kpath))
  {
    log_error(
      "k-path coverage: internal invariant violated — spanning.total()={} "
      "but get_total_instrument()={}. Each instrumented assertion must "
      "have a matching spanning.add_goal entry. Aborting rather than "
      "report an unsound coverage percentage.",
      spanning.total(),
      total_kpath);
    abort();
  }

  // Compute the spanning-set after every goal has been collected. The
  // resulting size is the Phase-2 denominator; redundant_claims feeds the
  // JSON `feasibility` field.
  //
  // Secondary invariant: the simplifier never collapses two semantically
  // distinct witnesses to the same idf string, so spanning_size_ is
  // bounded above by all_claims.size() + |redundant|, which is what
  // allows the bmc.cpp coverage cap to make sense. Any future change
  // that reuses an idf across distinct witnesses or alters from_expr()
  // formatting must preserve this 1:1 mapping or the percentage will
  // silently deflate.
  spanning.finalize();
  total_kpath_spanning = spanning.spanning_size();
  for (const auto &claim : all_claims)
    if (spanning.is_redundant(claim.first, claim.second))
      k_path_spanning_redundant.insert(claim);

  goto_functions.update();
}

void goto_coveraget::insert_assert(
  goto_programt &goto_program,
  goto_programt::targett &it,
  const expr2tc &guard)
{
  insert_assert(goto_program, it, guard, from_expr(ns, "", guard));
}

/*
  convert
    1: DECL x   <--- it
    ASSIGN X 1
  to
    1: ASSERT(guard);
    DECL x      <--- it
    ASSIGN X 1  
*/
void goto_coveraget::insert_assert(
  goto_programt &goto_program,
  goto_programt::targett &it,
  const expr2tc &guard,
  const std::string &idf)
{
  goto_programt::instructiont instruction;
  instruction.make_assertion(guard);
  instruction.location = it->location;
  instruction.function = it->function;
  instruction.location.property("instrumented assertion");
  instruction.location.comment(idf);
  instruction.location.user_provided(true);
  goto_program.insert_swap(it++, instruction);
  it--;
}

int goto_coveraget::get_total_instrument() const
{
  int total_instrument = 0;
  forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      const goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      forall_goto_program_instructions (it, goto_program)
      {
        if (
          it->is_assert() &&
          it->location.property().as_string() == "instrumented assertion" &&
          it->location.user_provided() == true)
        {
          total_instrument++;
        }
      }
    }
  return total_instrument;
}

// Count the total assertion instances in goto level via goto-unwind api
// run the algorithm on the copy of the original goto program
int goto_coveraget::get_total_assert_instance() const
{
  // 1. execute goto unwind
  bounded_loop_unroller unwind_loops;
  unwind_loops.run(goto_functions);
  // 2. calculate the number of assertion instance
  return get_total_instrument();
}

std::set<std::pair<std::string, std::string>>
goto_coveraget::get_total_cond_assert() const
{
  std::set<std::pair<std::string, std::string>> total_cond_assert = {};
  forall_goto_functions (f_it, goto_functions)
  {
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      const goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      forall_goto_program_instructions (it, goto_program)
      {
        if (
          it->is_assert() &&
          it->location.property().as_string() == "instrumented assertion" &&
          it->location.user_provided() == true)
        {
          std::pair<std::string, std::string> claim_pair = std::make_pair(
            it->location.comment().as_string(), it->location.as_string());
          total_cond_assert.insert(claim_pair);
        }
      }
    }
  }
  return total_cond_assert;
}

/*
  Condition Coverage: fault injection
  1. find condition statements, this includes the converted for_loop/while
  2. insert assertion instances before that statement.
  e.g.
    if (a >1)
  =>
    assert(!(a>1))
    assert(a>1)
    if(a>1)
  then run multi-property
*/
void goto_coveraget::condition_coverage()
{
  // we need to skip the conditions within the built-in library
  // while keeping the file manually included by user
  // this filter, however, is unsound.. E.g. if the src filename is the same as the builtin library name
  total_cond = {{}};

  std::unordered_set<std::string> location_pool = {};
  // cmdline.arg[0]
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      Forall_goto_program_instructions (it, goto_program)
      {
        std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->location.property().as_string() == "skipped")
          // this stands for the auxiliary condition/branch we added.
          continue;

        /* 
          Places that could contains condition
          1. GOTO:          if (x == 1);
          2. ASSIGN:        int x = y && z;
          3. ASSERT
          4. ASSUME
          5. FUNCTION_CALL  test((signed int)(x != y));
          6. RETURN         return x && y;
          7. Other          1?2?3:4
          The issue is that, the side-effects have been removed 
          thus the condition might have been split or modified.

          For assert, assume and goto, we know it contains GUARD
          For others, we need to convert the code back to expr and
          check there operands.
        */

        // Skip ASSUME instructions: __VERIFIER_assume / __ESBMC_assume
        // express path constraints, not program logic, so their guards
        // must not contribute to condition-coverage claims (issue #4291).
        if (it->is_assume())
          continue;

        // e.g. assert(a == 1);
        if (
          it->is_assert() &&
          it->location.property().as_string() != "replaced assertion")
        {
          if (!is_nil_expr(it->guard))
          {
            expr2tc guard = handle_single_guard(it->guard, true);
            gen_cond_cov_assert(guard, expr2tc(), goto_program, it);
            // after adding the instrumentation, we neutralize the original assert
            if (cov_assume_asserts)
              replace_assert_to_assume(it);
            else
              replace_assert_to_guard(gen_true_expr(), it, false);
          }
        }

        // e.g. IF !(a > 1) THEN GOTO 3
        else if (it->is_goto() && !is_true(it->guard))
        {
          // e.g.
          //    GOTO 2;
          //    2: IF(...);
          if (it->is_target())
            target_num = it->target_number;

          // preprocessing: if(true) ==> if(true == true)
          expr2tc guard = handle_single_guard(it->guard, true);
          gen_cond_cov_assert(guard, expr2tc(), goto_program, it);
        }

        // e.g. bool x = (a>b);
        else if (it->is_assign())
        {
          const expr2tc &rhs = to_code_assign2t(it->code).source;
          if (!is_nil_expr(rhs))
            handle_operands_guard(rhs, goto_program, it);
        }

        // a>b;
        else if (it->is_other())
        {
          if (is_code_expression2t(it->code))
          {
            const expr2tc &other = to_code_expression2t(it->code).operand;
            if (!is_nil_expr(other))
              handle_operands_guard(other, goto_program, it);
          }
        }

        // e.g. RETURN a>b;
        else if (it->is_return())
        {
          const expr2tc &ret = to_code_return2t(it->code).operand;
          if (!is_nil_expr(ret))
            handle_operands_guard(ret, goto_program, it);
        }

        // e.g. func(a>b);
        else if (it->is_function_call())
        {
          for (const expr2tc &op : to_code_function_call2t(it->code).operands)
            if (!is_nil_expr(op))
              handle_operands_guard(op, goto_program, it);
        }

        // reset target number
        target_num = -1;
      }
    }

  total_cond = get_total_cond_assert();
  all_claims = total_cond;

  // recalculate line number/ target number
  goto_functions.update();
}

/*
  algo:
  if(b==0 && c > 90)
  => assert(b==0)
  => assert(!(b==0));
  => assert(!(b==0 && c>90))
  => assert(!(b==0 && !(c>90)))

  if(b==0 || c > 90)
  => assert(b==0)
  => assert((b==0));
  => assert(!(!b==0 && c>90))
  => assert(!(!(b==0) && !(c>90)))
*/
/// Recurse into all sub-expressions of @p ptr, calling
/// gen_cond_cov_assert on each.
void goto_coveraget::gen_cond_cov_assert(
  const expr2tc &ptr,
  const expr2tc &pre_cond,
  goto_programt &goto_program,
  goto_programt::instructiont::targett &it)
{
  if (is_nil_expr(ptr))
    return;
  const std::size_t n = ptr->get_num_sub_exprs();
  if (n == 0)
    return; // atom

  auto recurse_all = [&]() {
    for (std::size_t i = 0; i < n; ++i)
      gen_cond_cov_assert(*ptr->get_sub_expr(i), pre_cond, goto_program, it);
  };

  if (n == 1)
  {
    // (a!=0)++, !a, -a, (_Bool)(int)a
    recurse_all();
  }
  else if (n == 2)
  {
    if (is_comparison_expr(ptr))
    {
      recurse_all();
      add_cond_cov_assert(ptr, pre_cond, goto_program, it);
    }
    else if (is_and2t(ptr))
    {
      const expr2tc &lhs = *ptr->get_sub_expr(0);
      const expr2tc &rhs = *ptr->get_sub_expr(1);
      gen_cond_cov_assert(lhs, pre_cond, goto_program, it);

      // update pre-condition: pre_cond && lhs
      expr2tc new_pre =
        is_nil_expr(pre_cond) ? lhs : gen_and_expr(pre_cond, lhs);
      gen_cond_cov_assert(rhs, new_pre, goto_program, it);
    }
    else if (is_or2t(ptr))
    {
      const expr2tc &lhs = *ptr->get_sub_expr(0);
      const expr2tc &rhs = *ptr->get_sub_expr(1);
      gen_cond_cov_assert(lhs, pre_cond, goto_program, it);

      // update pre-condition: !(pre_cond && lhs)
      expr2tc new_pre =
        is_nil_expr(pre_cond) ? lhs : gen_and_expr(pre_cond, lhs);
      new_pre = gen_not_expr(new_pre);
      gen_cond_cov_assert(rhs, new_pre, goto_program, it);
    }
    else
    {
      // a+=b; a>>(b!=0);
      recurse_all();
    }
  }
  else if (n == 3)
  {
    // ternary if
    const expr2tc &cond = *ptr->get_sub_expr(0);
    const expr2tc &t_val = *ptr->get_sub_expr(1);
    const expr2tc &f_val = *ptr->get_sub_expr(2);

    gen_cond_cov_assert(cond, pre_cond, goto_program, it);

    // update pre-condition: pre_cond && cond
    expr2tc pre_cond_1 =
      is_nil_expr(pre_cond) ? cond : gen_and_expr(pre_cond, cond);
    gen_cond_cov_assert(t_val, pre_cond_1, goto_program, it);

    // update pre-condition: pre_cond && !cond
    expr2tc not_cond = gen_not_expr(cond);
    expr2tc pre_cond_2 =
      is_nil_expr(pre_cond) ? not_cond : gen_and_expr(pre_cond, not_cond);
    gen_cond_cov_assert(f_val, pre_cond_2, goto_program, it);
  }
  else
  {
    log_error("unexpected operand size");
    abort();
  }
}

void goto_coveraget::add_cond_cov_assert(
  const expr2tc &expr,
  const expr2tc &pre_cond,
  goto_programt &goto_program,
  goto_programt::instructiont::targett &it)
{
  expr2tc cond = is_nil_expr(pre_cond) ? expr : gen_and_expr(pre_cond, expr);

  // e.g. assert(!(a==1));  // a==1
  // the idf is used as the claim_msg
  // note that it's different from the actual guard.
  std::string idf = from_expr(ns, "", expr);
  expr2tc guard = gen_not_expr(cond);
  insert_assert(goto_program, it, guard, idf);

  // reversal
  expr2tc not_expr = gen_not_expr(expr);
  cond = is_nil_expr(pre_cond) ? not_expr : gen_and_expr(pre_cond, not_expr);
  idf = from_expr(ns, "", not_expr);
  guard = gen_not_expr(cond);
  insert_assert(goto_program, it, guard, idf);
}

expr2tc goto_coveraget::gen_not_eq_expr(const expr2tc &lhs, const expr2tc &rhs)
{
  expr2tc _lhs = (lhs->type == rhs->type) ? lhs : typecast2tc(rhs->type, lhs);
  return notequal2tc(_lhs, rhs);
}

expr2tc goto_coveraget::gen_and_expr(const expr2tc &lhs, const expr2tc &rhs)
{
  type2tc bt = get_bool_type();
  expr2tc _lhs = is_bool_type(lhs->type) ? lhs : typecast2tc(bt, lhs);
  expr2tc _rhs = is_bool_type(rhs->type) ? rhs : typecast2tc(bt, rhs);
  return and2tc(_lhs, _rhs);
}

expr2tc goto_coveraget::gen_not_expr(const expr2tc &guard)
{
  if (is_not2t(guard))
    return to_not2t(guard).value;
  return not2tc(guard);
}

/*
  This function convert single guard to a non_equal_to_false expression
  e.g. if(true) ==> if(true!=false)
  rule:
  1. No-op: Do nothing. This means it's a symbol or constant
  2. Binary OP: for boolean expreession, e.g. a>b, a==b, do nothing
  3. Binary OP: for and/or expresson, add on both side, if possible. Do not add if it's already a binary boolean expression in 2. 
    e.g. if(x==1 && a++) => if(x==1 && a++ !=0)
  4. Others: for any other expresison, including unary, binary and teranry, traverse its op with handle_single_guard recursivly. convert it to not equal in the top level only.
    e.g. if((bool)a+b+c) => if((bool)(a+b+c)!=0)
    typecast <--- add not equal here
    - +
      - a
      - + 
        - b
        - c
  e.g. if(a) => if(a!=0); if(true) => if(true != 0); if(a?b:c:d) => if((a?b:c:d)!=0)
  if(a==b) => if(a==b); if(a&&b) => if(a != 0 && b!=0 )
*/
/// Recursively maps each operand of @p expr through handle_single_guard
/// (with the supplied @p sub_top_level), in place. Foreach_operand detaches
/// the irep_container before mutating, so this is safe even when @p expr
/// shares storage with its caller.
static void replace_operands(
  expr2tc &expr,
  bool sub_top_level,
  const std::function<expr2tc(const expr2tc &, bool)> &recurse)
{
  expr->Foreach_operand([&](expr2tc &op) { op = recurse(op, sub_top_level); });
}

expr2tc goto_coveraget::handle_single_guard(
  const expr2tc &expr,
  bool top_level /* = true */)
{
  if (is_nil_expr(expr))
    return expr;
  const std::size_t n = expr->get_num_sub_exprs();
  auto recurse = [this](const expr2tc &e, bool tl) {
    return handle_single_guard(e, tl);
  };

  // --- Rule 1: Atomic expressions ---
  // If the expression has no operands (a symbol or constant),
  // then if it's Boolean and we're at the outer guard, wrap it with
  // "!= false".
  if (n == 0)
  {
    if (top_level && is_bool_type(expr->type))
      return gen_not_eq_expr(expr, gen_false_expr());
    return expr;
  }

  // --- Special-case for "not" nodes ---
  // For a "not" operator, process its operand with top_level = true so that
  // even nested atomic expressions (like x in !(!(x))) get wrapped.
  if (is_not2t(expr))
  {
    expr2tc result = expr;
    replace_operands(result, /*sub_top_level=*/true, recurse);
    return result;
  }

  // --- Special-case for typecasts to bool ---
  // If we have (bool)(X) and X is not already a recognized guard
  // (comparison or logical AND/OR), unwrap the typecast and wrap X.
  if (is_typecast2t(expr) && is_bool_type(expr->type))
  {
    expr2tc inner = handle_single_guard(to_typecast2t(expr).from, top_level);
    if (!(is_comparison_expr(inner) || is_and2t(inner) || is_or2t(inner)))
      return gen_not_eq_expr(inner, gen_false_expr());
    return inner;
  }

  // --- Process Binary Operators (exactly 2 operands) ---
  if (n == 2)
  {
    expr2tc result = expr;
    if (is_and2t(expr) || is_or2t(expr))
    {
      // Process each operand as an independent guard (top_level = true).
      replace_operands(result, /*sub_top_level=*/true, recurse);
      return result;
    }
    if (is_comparison_expr(expr))
    {
      replace_operands(result, /*sub_top_level=*/false, recurse);
      return result;
    }
    // Other binary operators (e.g. arithmetic '+').
    replace_operands(result, /*sub_top_level=*/false, recurse);
    if (top_level)
      return gen_not_eq_expr(result, gen_false_expr());
    return result;
  }

  // --- Process Non-Binary Operators (Unary, Ternary, etc.) ---
  expr2tc result = expr;
  replace_operands(result, /*sub_top_level=*/false, recurse);

  // For any other expression producing a Boolean value, if at the outer
  // guard (top_level true) and its kind is not among our no-wrap set, then
  // wrap it with "!= false". This catches cases like member accesses.
  if (
    top_level && is_bool_type(result->type) && !is_and2t(result) &&
    !is_or2t(result) && !is_not2t(result) && !is_comparison_expr(result))
    return gen_not_eq_expr(result, gen_false_expr());
  return result;
}

/*
  add condition instrumentation for OTHER, ASSIGN, FUNCTION_CALL..
  whose operands might contain conditions
  we handle guards for each boolean sub-operand.
*/
void goto_coveraget::handle_operands_guard(
  const expr2tc &expr,
  goto_programt &goto_program,
  goto_programt::instructiont::targett &it)
{
  if (is_nil_expr(expr))
    return;
  const std::size_t n = expr->get_num_sub_exprs();
  if (n == 0)
    return;

  expr2tc pre_cond; // nil

  if (n == 1)
  {
    // e.g. RETURN ++(x&&y);
    handle_operands_guard(*expr->get_sub_expr(0), goto_program, it);
  }
  else if (n == 2)
  {
    expr2tc target = expr;
    if (is_and2t(expr) || is_or2t(expr))
    {
      // we do not need to add a !=false at top level
      // e.g. return x?1:0 != return (x?1:0)!=false
      target->Foreach_operand(
        [this](expr2tc &op) { op = handle_single_guard(op, false); });
    }
    gen_cond_cov_assert(target, pre_cond, goto_program, it);
  }
  else
  {
    // this could only be ternary boolean
    expr2tc rewrapped = handle_single_guard(expr, false);
    gen_cond_cov_assert(rewrapped, pre_cond, goto_program, it);
  }
}

// set the target function from "--function"
void goto_coveraget::set_target(const std::string &_tgt)
{
  target_function = _tgt;
}

// check if it's the target function
bool goto_coveraget::is_target_func(
  const irep_idt &f,
  const std::string &tgt_name) const
{
  const symbolt *sym = ns.lookup(f);
  if (sym == nullptr)
  {
    log_error("Cannot find target function");
    abort();
  }

  exprt symbol = symbol_expr(*ns.lookup(f));
  std::string sym_name = symbol.name().as_string();
  if (sym_name == tgt_name)
    return true;

  // For Solidity: modifier expansion renames functions from "func" to
  // "func_modifierName". Support prefix matching so that --function func
  // matches func_modifierName.
  if (
    config.language.lid == language_idt::SOLIDITY &&
    sym_name.size() > tgt_name.size() &&
    sym_name.substr(0, tgt_name.size()) == tgt_name &&
    sym_name[tgt_name.size()] == '_')
    return true;

  return false;
}

// negate the condition inside the assertion
// The idea is that, if the claim is verified safe, and its negated claim is also verified safe, then we say this claim is unreachable
void goto_coveraget::negating_asserts(const std::string &tgt_fname)
{
  std::string old = target_function;
  target_function = tgt_fname;

  std::unordered_set<std::string> location_pool = {};
  location_pool.insert(get_filename_from_path(filename));
  for (auto const &inc : config.ansi_c.include_files)
    location_pool.insert(get_filename_from_path(inc));

  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      if (filter(f_it->first, goto_program))
        continue;

      Forall_goto_program_instructions (it, goto_program)
      {
        std::string cur_filename =
          get_filename_from_path(it->location.file().as_string());
        if (location_pool.count(cur_filename) == 0)
          continue;

        if (it->is_assert())
        {
          expr2tc guard = it->guard;
          replace_assert_to_guard(gen_not_expr(guard), it, false);
        }
      }
    }
  target_function = old;
}

// return true if this function is skipped
bool goto_coveraget::filter(
  const irep_idt &func_name,
  const goto_programt &goto_program) const
{
  // "--function" mode
  if (target_function != "" && !is_target_func(func_name, target_function))
    return true;

  // Skip the function that is labelled with "__ESBMC_HIDE"
  // Extended to support Python in addition to Solidity
  if (
    goto_program.hide && (config.language.lid == language_idt::SOLIDITY ||
                          config.language.lid == language_idt::PYTHON))
    return true;
  return false;
}
