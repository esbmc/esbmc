#include <solvers/neurosym/neurosym_conv.h>
#include <solvers/smtlib/oneshot_process.h>
#include <util/arith/mp_arith.h>
#include <util/message/message.h>

#include <algorithm>
#include <cctype>
#include <cstdio>

static std::string prog_command(const optionst &options)
{
  std::string cmd = options.get_option("neurosym-prog");
  return cmd.empty() ? "python main.py %f" : cmd;
}

static void skip_ws(const std::string &s, size_t &pos)
{
  while (pos < s.size() && std::isspace((unsigned char)s[pos]))
    pos++;
}

/* Skip one balanced-parens group starting at s[pos] == '(', leaving pos just
 * past its matching ')'. No-op (pos unchanged) if s[pos] is not '('. */
static void skip_paren_group(const std::string &s, size_t &pos)
{
  if (pos >= s.size() || s[pos] != '(')
    return;
  int depth = 0;
  do
  {
    if (s[pos] == '(')
      depth++;
    else if (s[pos] == ')')
      depth--;
    pos++;
  } while (pos < s.size() && depth > 0);
}

/* One bare token: runs until whitespace or a paren. */
static std::string read_token(const std::string &s, size_t &pos)
{
  size_t start = pos;
  while (pos < s.size() && !std::isspace((unsigned char)s[pos]) &&
         s[pos] != '(' && s[pos] != ')')
    pos++;
  return s.substr(start, pos - start);
}

void neurosym_convt::parse_model_block(const std::string &output)
{
  /* Purpose-built for exactly the format NeuroSym's own format_output()
   * emits (gansat/ns_solver.py, main.py):
   *   (model
   *     (define-fun NAME () SORT VALUE)
   *     ...
   *   )
   * SORT is "Int", "Bool", or "(_ BitVec N)"; VALUE is a bare decimal
   * numeral, a #x hex literal, or true/false. NAME may be pipe-quoted.
   * Anything this does not recognize is simply skipped for that one
   * variable -- it falls through to the --neurosym-model-prog fallback
   * exactly as if local parsing had not been attempted, so a NeuroSym
   * output format this cannot fully make sense of degrades to the old
   * behaviour rather than building a wrong counterexample. */
  size_t pos = output.find("(model");
  if (pos == std::string::npos)
    return;
  pos += 6;
  const size_t n = output.size();

  while (true)
  {
    skip_ws(output, pos);
    if (pos >= n || output[pos] == ')')
      break;
    if (output[pos] != '(')
      break; // unrecognized content where another define-fun was expected

    size_t entry_open = pos;
    pos++; // past '('
    skip_ws(output, pos);
    std::string keyword = read_token(output, pos);
    if (keyword != "define-fun")
    {
      // Not a form we understand (e.g. a comment leaked through as a list) --
      // skip this whole parenthesized entry and move on to the next.
      pos = entry_open;
      skip_paren_group(output, pos);
      continue;
    }

    skip_ws(output, pos);
    std::string name;
    if (pos < n && output[pos] == '|')
    {
      size_t close = output.find('|', pos + 1);
      if (close == std::string::npos)
        break;
      name = output.substr(pos + 1, close - pos - 1);
      pos = close + 1;
    }
    else
      name = read_token(output, pos);

    skip_ws(output, pos); // the empty parameter list "()"
    skip_paren_group(output, pos);

    skip_ws(output, pos); // the sort -- "(_ BitVec N)" or a bare "Int"/"Bool"
    if (pos < n && output[pos] == '(')
      skip_paren_group(output, pos);
    else
      read_token(output, pos);

    skip_ws(output, pos);
    std::string value;
    if (pos < n && output[pos] == '(')
    {
      /* A parenthesized value, e.g. SMT-LIB2's "(- 5)" for a negative
       * numeral -- not currently interpreted; leave value empty so this one
       * variable falls back to --neurosym-model-prog instead of guessing. */
      skip_paren_group(output, pos);
    }
    else
      value = read_token(output, pos);

    if (!name.empty() && !value.empty())
      local_model[name] = value;

    skip_ws(output, pos);
    if (pos < n && output[pos] == ')')
      pos++; // close this define-fun
  }
}

/* Mirrors smtlib_convt's own interp_numeric() (smtlib_conv.cpp, file-local
 * there) closely enough to interpret local_model's raw value text the same
 * way a real solver's (get-value) response would be. */
static bool numeric_value(const std::string &v, bool is_signed, BigInt &out)
{
  if (v.size() > 2 && v[0] == '#' && v[1] == 'x')
  {
    out = string2integer(v.substr(2), 16);
    return true;
  }
  if (v.size() > 2 && v[0] == '#' && v[1] == 'b')
  {
    out = binary2integer(v.substr(2), is_signed);
    return true;
  }
  size_t digits_from = (!v.empty() && v[0] == '-') ? 1 : 0;
  if (v.size() > digits_from &&
      std::all_of(v.begin() + digits_from, v.end(), [](unsigned char c) {
        return std::isdigit(c);
      }))
  {
    out = string2integer(v);
    return true;
  }
  return false;
}

smt_solver_baset *create_new_neurosym_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api [[maybe_unused]],
  array_iface **array_api,
  fp_convt **fp_api [[maybe_unused]])
{
  /* NeuroSym solves a single formula per invocation; strategies that reuse
   * one persistent solver context across repeated or incremental checks
   * cannot be served by it. --multi-property is NOT in this list: without
   * --smt-during-symex, bmct::multi_property_check() (bmc.cpp) allocates a
   * fresh create_solver() instance per claim rather than reusing one across
   * claims, which is exactly NeuroSym's one-shot-per-invocation model — so
   * --multi-property (and the coverage modes built on it, e.g.
   * --branch-coverage) work correctly, just at the cost of one NeuroSym
   * subprocess per claim. Only --smt-during-symex, which explicitly shares
   * one persistent solver across every claim, is genuinely incompatible. */
  static const char *incompatible[] = {
    "incremental-bmc",
    "falsification",
    "k-induction",
    "k-induction-parallel",
    "termination",
    "smt-during-symex",
    "parallel-solving"};
  for (const char *opt : incompatible)
    if (options.get_bool_option(opt))
    {
      log_error(
        "the neurosym backend runs NeuroSym in one-shot batch mode and does "
        "not support --{}; use a linked solver (e.g. --bitwuzla) for "
        "incremental strategies",
        opt);
      abort();
    }

  /* NeuroSym has no native tuple or floating-point support, so those two
   * interfaces stay unset -- create_solver() installs the flatteners that
   * lower structs and floating-point to pure bit-vectors before they reach
   * the serializer, same as before.
   *
   * Arrays are different: neurosym_convt inherits array_iface from its
   * smtlib_convt base (smtlib_conv.h), which already knows how to serialize
   * native SMT-LIB2 (Array ...) / select / store syntax -- that support was
   * simply never wired up for this backend. NeuroSym's own bit-blaster now
   * has a real array-theory encoding (read-over-write + weak consistency
   * axioms), so setting *array_api lets ESBMC pass arrays through natively
   * instead of pre-flattening every access into per-index bit-vector
   * equality/implication chains -- for an array-heavy program that
   * flattening can blow a modest formula up into a CNF with well over a
   * million boolean variables (measured), most of which read-over-write
   * resolves away for free instead of ever materializing as clauses. */
  auto *conv    = new neurosym_convt(ns, options);
  *array_api    = static_cast<array_iface *>(conv);
  return conv;
}

neurosym_convt::neurosym_convt(const namespacet &ns, const optionst &options)
  : neurosym_convt(
      ns,
      options,
      oneshot_process::choose_formula_path(options, "neurosym"))
{
}

neurosym_convt::neurosym_convt(
  const namespacet &ns,
  const optionst &options,
  const std::string &_formula_path)
  : smtlib_convt(
      ns,
      options,
      oneshot_process::model_prog(options, "neurosym"),
      _formula_path,
      "QF_ABV"),   // QF_ABV, not QF_BV: array_api is now enabled above, so
                   // the emitted header must declare the logic that
                   // actually matches -- QF_ABV is a superset of QF_BV, so
                   // this is correct for array-free formulas too.
    formula_path(_formula_path)
{
}

neurosym_convt::~neurosym_convt()
{
  if (oneshot_process::uses_temp_formula(options))
    remove(formula_path.c_str());
}

std::string neurosym_convt::dump_smt()
{
  /* Under --smt-formula-only no solve follows; complete the dump with the
   * (check-sat) like the base class. Under --smt-formula-too our dec_solve()
   * emits the (check-sat) itself: appending one here as well would hand
   * NeuroSym a formula containing two. The base implementation also reports
   * the destination from the --output option, which this backend redirects
   * to the formula file. */
  if (options.get_bool_option("smt-formula-only"))
    return smtlib_convt::dump_smt();
  log_status("SMT formula written to {}", formula_path);
  return "SMT formula dumped successfully";
}

smt_resultt neurosym_convt::dec_solve()
{
  if (solved)
  {
    log_error(
      "the neurosym backend supports a single (check-sat) query per run; "
      "incremental strategies are not supported");
    abort();
  }
  solved = true;

  pre_solve();

  /* The (check-sat) goes to both sinks: the formula file for NeuroSym, and
   * the local model solver's pipe (if configured), which starts solving in
   * parallel and only gets waited for when a model is actually needed. The
   * model solver only produces counterexamples, so if it has died (e.g. it
   * failed to start), disable it and let NeuroSym decide: an unsat proof
   * needs no model, and a sat result reports the missing-model error below
   * rather than crashing on an uncaught exception. */
  try
  {
    emit_check_sat();
  }
  catch (const external_process_died &)
  {
    log_warning(
      "neurosym: the local model solver '{}' terminated unexpectedly (e.g. "
      "it failed to start); continuing without counterexample support",
      options.get_option("neurosym-model-prog"));
    emit_proc.terminate();
    flush(); // complete the formula file for NeuroSym now that the pipe is gone
  }

  std::string captured_output;
  smt_resultt res = oneshot_process::run_solver(
    prog_command(options), formula_path, "neurosym", &captured_output);
  if (res != P_SATISFIABLE)
  {
    /* No model will be read; stop the local solver we fed in parallel rather
     * than let it keep solving until this object is destroyed. */
    emit_proc.terminate();
    return res;
  }

  /* NeuroSym's own batch stdout already carries a sort-correct model on a
   * sat verdict (see the class comment): parse it directly rather than
   * unconditionally waiting on the local model solver's answer here. When it
   * covers everything the trace ends up asking for -- the common case, since
   * NeuroSym's model output normally lists every free variable in the
   * formula -- --neurosym-model-prog's solve is never waited for at all,
   * however long it takes on this formula. A variable that is not in
   * local_model (parsing failure, or a value form local parsing does not
   * understand) still falls back to it, lazily, the first time get_bv() /
   * l_get() below actually needs it. */
  parse_model_block(captured_output);
  if (!local_model.empty())
    return P_SATISFIABLE;

  /* Local parsing found nothing usable (NeuroSym's output did not match the
   * expected model format, or this particular formula has no free
   * variables) -- fall back to the original --neurosym-model-prog path in
   * full, right away, exactly as before this change. */
  ensure_model_prog_ready();
  return P_SATISFIABLE;
}

void neurosym_convt::ensure_model_prog_ready()
{
  if (model_prog_response_read)
    return;
  model_prog_response_read = true;

  /* A satisfiable formula with nothing usable in local_model needs a live
   * model solver to turn into a counterexample. It is absent either because
   * the model solver died earlier (a command was given) or was never
   * configured. Under --result-only no counterexample is ever built
   * (bmc.cpp skips trace construction) and get_bv()/l_get() are normally
   * never even called -- but dec_solve() itself still calls this eagerly as
   * its own fallback when local_model comes up empty, regardless of
   * --result-only, so that path still needs handling here explicitly: stay
   * silent and let the (never-built) trace go on rather than erroring over
   * a model nothing will read. */
  if (!emit_proc)
  {
    if (options.get_bool_option("result-only"))
      return;
    if (options.get_option("neurosym-model-prog").empty())
      log_error(
        "neurosym: formula is satisfiable, but building the counterexample "
        "requires a local interactive SMT-LIB2 solver; re-run with "
        "--neurosym-model-prog <cmd> (e.g. \"z3 -in\") or with "
        "--result-only");
    else
      log_error(
        "neurosym: the local model solver is unavailable; cannot build a "
        "counterexample");
    abort();
  }

  smt_resultt model_res;
  try
  {
    model_res = read_check_sat_response();
  }
  catch (const external_process_died &)
  {
    log_error(
      "neurosym: the local model solver is unavailable; cannot build a "
      "counterexample");
    abort();
  }
  if (model_res != P_SATISFIABLE)
  {
    log_error(
      "neurosym: NeuroSym reported sat but the local model solver did not; "
      "refusing to build a counterexample from a diverging model");
    abort();
  }
}

tvt neurosym_convt::get_bool(smt_astt a)
{
  return l_get(a);
}

tvt neurosym_convt::l_get(smt_astt a)
{
  const std::string &symname = to_solver_smt_ast<smtlib_smt_ast>(a)->symname;
  auto it = local_model.find(symname);
  if (it != local_model.end())
  {
    const std::string &v = it->second;
    if (v == "true")
      return tvt(true);
    if (v == "false")
      return tvt(false);
    BigInt m;
    if (numeric_value(v, false, m))
      return tvt(m != 0);
    // Fall through: an entry exists but this parser could not make sense of
    // its value (e.g. the "(- N)" form) -- treat it the same as a miss.
  }
  ensure_model_prog_ready();
  return smtlib_convt::l_get(a);
}

BigInt neurosym_convt::get_bv(smt_astt a, bool is_signed)
{
  const std::string &symname = to_solver_smt_ast<smtlib_smt_ast>(a)->symname;
  auto it = local_model.find(symname);
  if (it != local_model.end())
  {
    BigInt m;
    if (numeric_value(it->second, is_signed, m))
      return m;
  }
  ensure_model_prog_ready();
  return smtlib_convt::get_bv(a, is_signed);
}

const std::string neurosym_convt::solver_text()
{
  return "NeuroSym '" + prog_command(options) + "'";
}
