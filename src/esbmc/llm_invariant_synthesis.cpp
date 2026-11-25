#include <esbmc/bmc.h>
#include <esbmc/llm_invariant_synthesis.h>
#include <goto-programs/goto_loop_invariant.h>
#include <llm/goto/invariants.h>
#include <llm/goto/parse_expr.h>
#include <llm/goto/render.h>
#include <util/message/message.h>

#include <algorithm>
#include <map>
#include <optional>
#include <sstream>

namespace
{
struct candidatet
{
  expr2tc expr;
  std::string text;
};

struct loopt
{
  irep_idt function;
  std::string name; // `<function> L<n>`, as the model writes it
  size_t head_index;
};

struct functiont
{
  irep_idt id;
  std::string name;
  llm::rendered_functiont rendered;
};

using candidate_mapt = std::map<size_t, std::vector<candidatet>>;

/// goto_loop_invariant collects the markers of one loop only within
/// kMaxInvariantSearchBack instructions of its head.
constexpr size_t max_conjuncts =
  goto_loop_invariantt::kMaxInvariantSearchBack - 2;

/// Which loop and conjunct a base-case or inductive-step claim checks.
struct claim_ownert
{
  size_t loop;
  expr2tc expr;
};

struct check_resultt
{
  /// Conjuncts some claim refuted, with why.
  std::vector<std::pair<claim_ownert, std::string>> refuted;
  /// Claims of the program itself not proved, as feedback text.
  std::vector<std::string> unproved;
  /// A program claim violated without help from the abstraction.
  bool real_violation = false;
};

bool user_invariant_present(const goto_functionst &goto_functions)
{
  forall_goto_functions (it, goto_functions)
    forall_goto_program_instructions (i, it->second.body)
      if (i->is_loop_invariant())
        return true;
  return false;
}

void split_conjuncts(const expr2tc &e, std::vector<expr2tc> &out)
{
  if (is_and2t(e))
  {
    split_conjuncts(to_and2t(e).side_1, out);
    split_conjuncts(to_and2t(e).side_2, out);
  }
  else
    out.push_back(e);
}

size_t count(const candidate_mapt &candidates)
{
  size_t n = 0;
  for (const auto &[loop, invs] : candidates)
    n += invs.size();
  return n;
}

/// Clamps the log to errors for the check's lifetime, even if it throws.
struct quiett
{
  const VerbosityLevel saved = messaget::state.verbosity;
  quiett()
  {
    messaget::state.verbosity = std::min(saved, VerbosityLevel::Error);
  }
  ~quiett()
  {
    messaget::state.verbosity = saved;
  }
};

class llm_cegist
{
public:
  llm_cegist(
    goto_functionst &goto_functions,
    contextt &context,
    const optionst &options,
    bool use_frame_rule)
    : goto_functions(goto_functions),
      context(context),
      ns(context),
      run_options(options),
      use_frame_rule(use_frame_rule)
  {
    // The check is a guide, not the verdict, so it runs quietly and writes
    // nothing; the caller's run re-checks whatever is kept.
    for (const char *output :
         {"witness-output-yaml",
          "witness-output-graphml",
          "sarif-output",
          "cex-output"})
      run_options.set_option(output, "");
    for (const char *unwanted :
         {"generate-testcase",
          "generate-json-report",
          "smt-formula-only",
          "smt-formula-too",
          "dead-code-check",
          "forward-condition"})
      run_options.set_option(unwanted, false);
    run_options.set_option("multi-property", true);
    run_options.set_option("base-case", true);
    // shortcut: every labelled loop is cut, but a loop the renderer does not
    // label (one in a hidden body, or recursion) runs one iteration here, so
    // a conjunct can pass the check and fail in the caller's run, which then
    // reports FAILED. Upgrade by cutting or fully unwinding those loops.
    if (run_options.get_option("unwind").empty())
      run_options.set_option("unwind", "1");
    run_options.set_option("no-unwinding-assertions", true);

    Forall_goto_functions (it, goto_functions)
    {
      if (!it->second.body_available || it->second.body.hide)
        continue;
      const std::string name = id2string(ns.lookup(it->first)->name);
      if (name.rfind("__ESBMC", 0) == 0)
        continue;
      functiont f{
        it->first, name, llm::render_function(it->first, it->second, ns)};
      const auto begin = it->second.body.instructions.cbegin();
      for (const auto &[label, head] : f.rendered.loops)
        loops.push_back(
          {it->first, name + " " + label, size_t(std::distance(begin, head))});
      functions.push_back(std::move(f));
    }
  }

  bool has_loops() const
  {
    return !loops.empty();
  }

  std::string program_text() const
  {
    std::ostringstream s;
    s << "Program:\n```c\n";
    for (const functiont &f : functions)
      s << f.rendered.text << "\n";
    s << "```\nPropose invariants for these loops:";
    for (const loopt &l : loops)
      s << " " << l.name;
    s << ".";
    return s.str();
  }

  /// Parse @p answer into @p into, and describe what was rejected.
  std::string
  parse_answer(const std::string &answer, candidate_mapt &into) const
  {
    std::ostringstream rejected;
    const auto lines = llm::parse_invariant_answer(answer);
    if (lines.empty())
      rejected << "- Your answer had no <invariants> block with a "
                  "`<function> <loop>: <expression>` line.\n";
    for (const llm::invariant_linet &line : lines)
    {
      const std::string name = line.function + " " + line.loop;
      const auto loop =
        std::find_if(loops.begin(), loops.end(), [&](const loopt &l) {
          return l.name == name;
        });
      if (loop == loops.end())
      {
        rejected << "- `" << name << "` is not a loop of the program.\n";
        continue;
      }
      const functiont &f = *std::find_if(
        functions.begin(), functions.end(), [&](const functiont &g) {
          return g.id == loop->function;
        });
      std::optional<expr2tc> e =
        llm::parse_expr(line.expression, f.rendered.scope, ns);
      if (!e)
      {
        rejected << "- " << name << ": `" << line.expression
                 << "` does not parse or type-check as a C expression over "
                    "the printed variables.\n";
        continue;
      }
      if (!is_bool_type((*e)->type))
        *e = typecast2tc(get_bool_type(), *e);
      std::vector<expr2tc> conjuncts;
      split_conjuncts(*e, conjuncts);
      auto &kept = into[loop - loops.begin()];
      for (const expr2tc &c : conjuncts)
        if (kept.size() >= max_conjuncts)
          rejected << "- " << name << ": only " << max_conjuncts
                   << " invariants per loop are checked; `" << render(*loop, c)
                   << "` was ignored.\n";
        else if (std::none_of(
                   kept.begin(), kept.end(), [&](const candidatet &k) {
                     return k.expr == c;
                   }))
          kept.push_back(
            {c, conjuncts.size() == 1 ? line.expression : render(*loop, c)});
    }
    return rejected.str();
  }

  /// Which candidate each base-case and inductive-step claim of @p program
  /// checks, keyed by the claim's location number.
  std::map<unsigned, std::vector<claim_ownert>> claim_owners(
    const goto_functionst &program,
    const candidate_mapt &candidates) const
  {
    std::map<unsigned, std::vector<claim_ownert>> owners;
    for (const auto &[index, invs] : candidates)
      for (const auto &i :
           program.function_map.at(loops[index].function).body.instructions)
      {
        const irep_idt &property = i.location.property();
        if (
          !i.is_assert() || (property != "invariant-base-case" &&
                             property != "invariant-inductive-step"))
          continue;
        for (const candidatet &c : invs)
          if (
            i.guard == c.expr ||
            (is_or2t(i.guard) && to_or2t(i.guard).side_2 == c.expr))
            owners[i.location_number].push_back({index, c.expr});
      }
    return owners;
  }

  /// A kept conjunct that is not inductive fails its claim in the caller's
  /// run and turns the verdict into FAILED, so whatever the check cannot
  /// vouch for is dropped.
  static check_resultt refute_all(const candidate_mapt &candidates)
  {
    check_resultt result;
    for (const auto &[index, invs] : candidates)
      for (const candidatet &c : invs)
        result.refuted.push_back({{index, c.expr}, "could not be checked"});
    return result;
  }

  check_resultt check(const candidate_mapt &candidates)
  {
    goto_functionst copy = goto_functions;
    std::map<unsigned, std::vector<claim_ownert>> owners;
    std::string error;
    {
      const quiett quiet;
      attach(copy, candidates, true);
      goto_loop_invariant(copy, context, use_frame_rule);
      owners = claim_owners(copy, candidates);
      goto_functionst::property_verdicts.clear();
      try
      {
        bmct bmc(copy, run_options, context);
        bmc.start_bmc();
      }
      catch (const std::exception &e)
      {
        error = e.what();
      }
      catch (...)
      {
        error = "unknown error";
      }
    }
    const auto rows = goto_functionst::property_verdicts.snapshot();
    goto_functionst::property_verdicts.clear();
    goto_functionst::reported_verdicts.clear();

    if (!error.empty())
    {
      log_warning("LLM loop invariants: the check failed: {}", error);
      return refute_all(candidates);
    }
    check_resultt result;
    for (const auto &[key, row] : rows)
    {
      if (row.verdict == property_verdictt::Passed)
        continue;
      const bool base = row.loc.description == "loop invariant base case";
      if (base || row.loc.description == "loop invariant inductive step")
      {
        const auto owned = owners.find(row.loc.instruction);
        if (owned == owners.end())
          return refute_all(candidates);
        for (const claim_ownert &o : owned->second)
          result.refuted.push_back(
            {o,
             base ? "does not hold when the loop is first reached"
                  : "is not preserved by one iteration of the loop body"});
        continue;
      }
      if (row.verdict == property_verdictt::Failed)
        result.real_violation = true;
      std::ostringstream s;
      s << "`" << row.loc.description << "` in " << row.loc.function
        << " at line " << row.loc.line;
      if (!row.note.empty())
        s << " (" << row.note << ")";
      result.unproved.push_back(s.str());
    }
    return result;
  }

  /// Check @p candidates, dropping refuted conjuncts until the rest are
  /// inductive (Houdini), and describe each drop in @p feedback.
  check_resultt
  keep_inductive(candidate_mapt &candidates, std::ostream &feedback)
  {
    for (;;)
    {
      check_resultt result = check(candidates);
      bool dropped = false;
      for (const auto &[owner, why] : result.refuted)
      {
        auto &invs = candidates[owner.loop];
        const auto it =
          std::find_if(invs.begin(), invs.end(), [&](const candidatet &c) {
            return c.expr == owner.expr;
          });
        if (it == invs.end())
          continue;
        feedback << "- " << loops[owner.loop].name << ": `" << it->text << "` "
                 << why << ".\n";
        invs.erase(it);
        dropped = true;
      }
      if (!dropped)
        return result;
    }
  }

  /// The next prompt: what was kept, and what is still unproved with it.
  void describe(
    const candidate_mapt &kept,
    const check_resultt &result,
    std::ostream &feedback) const
  {
    if (count(kept))
    {
      feedback << "These invariants are inductive and are kept:\n";
      for (const auto &[loop, invs] : kept)
        for (const candidatet &c : invs)
          feedback << "- " << loops[loop].name << ": `" << c.text << "`\n";
    }
    feedback << "With them, the model checker cannot prove:\n";
    for (const std::string &u : result.unproved)
      feedback << "- " << u << "\n";
    feedback << "Think step by step about what is missing or wrong, then "
                "answer with a new <invariants> block of additional "
                "invariants.";
  }

  /// Attach @p candidates as synthesised markers; @p cut_all gives every
  /// other loop the invariant `true`, cutting it rather than unwinding it.
  void attach(
    goto_functionst &program,
    const candidate_mapt &candidates,
    bool cut_all) const
  {
    std::vector<size_t> order(loops.size());
    for (size_t i = 0; i < order.size(); i++)
      order[i] = i;
    // Back to front, so inserting a marker leaves the other indices valid.
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return loops[a].head_index > loops[b].head_index;
    });
    for (size_t index : order)
    {
      const auto found = candidates.find(index);
      const bool has = found != candidates.end() && !found->second.empty();
      if (!has && !cut_all)
        continue;
      const loopt &loop = loops[index];
      goto_programt &body = program.function_map.at(loop.function).body;
      const auto head = std::next(body.instructions.begin(), loop.head_index);

      // One marker per conjunct, so each gets base-case and inductive-step
      // claims of its own that check() can attribute.
      auto add_marker = [&](const expr2tc &e) {
        goto_programt::instructiont marker;
        marker.type = LOOP_INVARIANT;
        marker.location = head->location;
        marker.function = head->function;
        marker.location.property(kSynthesisedInvariantProperty);
        marker.add_loop_invariant(e);
        body.instructions.insert(head, marker);
      };
      if (has)
        for (const candidatet &c : found->second)
          add_marker(c.expr);
      else
        add_marker(gen_true_expr());
    }
    program.update();
  }

  std::string render(const loopt &loop, const expr2tc &e) const
  {
    const functiont &f = *std::find_if(
      functions.begin(), functions.end(), [&](const functiont &g) {
        return g.id == loop.function;
      });
    return llm::render_expr(e, f.rendered.scope, ns);
  }

private:
  goto_functionst &goto_functions;
  contextt &context;
  namespacet ns;
  optionst run_options;
  bool use_frame_rule;
  std::vector<functiont> functions;
  std::vector<loopt> loops;
};
} // namespace

void llm_synthesise_loop_invariants(
  goto_functionst &goto_functions,
  contextt &context,
  const optionst &options,
  llm::clientt &client,
  unsigned max_rounds,
  bool use_frame_rule)
{
  if (spawns_threads(goto_functions))
  {
    log_warning(
      "--llm-invariants: the program creates threads, and "
      "cutting a loop deletes the interleaving points its body carried. "
      "Synthesising nothing");
    return;
  }
  // shortcut: declines outright rather than port the affine pass's per-loop
  // rules for user-written invariants; upgrade if mixing the two is wanted.
  if (user_invariant_present(goto_functions))
  {
    log_warning(
      "--llm-invariants: the program already carries loop "
      "invariants, which are authoritative. Synthesising nothing");
    return;
  }

  llm_cegist cegis(goto_functions, context, options, use_frame_rule);
  if (!cegis.has_loops())
  {
    log_warning("--llm-invariants added no invariant: no loop was found");
    return;
  }

  std::vector<llm::message_t> conversation{
    {"system", llm::invariant_system_prompt}, {"user", cegis.program_text()}};
  candidate_mapt kept;

  for (unsigned round = 1; round <= max_rounds; round++)
  {
    log_status("LLM loop invariants: round {}, asking the model", round);
    std::string answer;
    try
    {
      answer = client.complete(conversation);
    }
    catch (const std::runtime_error &e)
    {
      log_warning("LLM loop invariants: no answer: {}", e.what());
      continue;
    }
    log_debug("llm", "model answer:\n{}", answer);
    conversation.push_back({"assistant", answer});

    candidate_mapt candidates = kept;
    std::ostringstream feedback;
    feedback << cegis.parse_answer(answer, candidates);
    const size_t proposed = count(candidates);
    const check_resultt result = cegis.keep_inductive(candidates, feedback);
    kept = candidates;
    log_status(
      "LLM loop invariants: {} of {} candidate conjunct(s) are inductive",
      count(kept),
      proposed);

    if (result.unproved.empty() || result.real_violation)
      break;
    cegis.describe(kept, result, feedback);
    conversation.push_back({"user", feedback.str()});
  }

  cegis.attach(goto_functions, kept, false);
  size_t loops = 0;
  for (const auto &[loop, invs] : kept)
    loops += !invs.empty();
  if (loops)
    log_status(
      "Synthesised loop invariants for {} loop{}",
      loops,
      loops == 1 ? "" : "s");
  else
    log_warning(
      "--llm-invariants added no invariant: the model "
      "proposed none that is inductive; the run proceeds unchanged");
}
