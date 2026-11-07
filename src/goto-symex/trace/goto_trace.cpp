#include <cassert>
#include <cstring>
#include <goto-symex/trace/goto_trace.h>
#include <goto-symex/trace/printf_formatter.h>
#include <goto-symex/witness/witnesses.h>

#include <regex>
#include <unordered_map>
#include <langapi/language_util.h>
#include <langapi/languages.h>
#include <util/arith/arith_tools.h>
#include <util/base/cwe_mapping.h>
#include <util/irep/std_types.h>
#include <util/symtab/namespace.h>
#include <ostream>

void goto_tracet::output(const class namespacet &ns, std::ostream &out) const
{
  for (const auto &step : steps)
    step.output(ns, out);
}

void goto_trace_stept::dump() const
{
  std::ostringstream oss;
  output(*migrate_namespace_lookup, oss);
  log_debug("goto-trace", "{}", oss.str());
}

void goto_trace_stept::output(const namespacet &ns, std::ostream &out) const
{
  switch (type)
  {
  case symex_targett::ASSERT:
    out << "ASSERT";
    break;

  case symex_targett::ASSUME:
    out << "ASSUME";
    break;

  case symex_targett::ASSIGNMENT:
    out << "ASSIGNMENT";
    break;

  default:
    assert(false);
  }

  if (type == symex_targett::ASSERT || type == symex_targett::ASSUME)
    out << " (" << guard << ")";

  out << "\n";

  if (!pc->location.is_nil())
    out << pc->location << "\n";

  if (pc->is_goto())
    out << "GOTO   ";
  else if (pc->is_assume())
    out << "ASSUME ";
  else if (pc->is_assert())
    out << "ASSERT ";
  else if (pc->is_other())
    out << "OTHER  ";
  else if (pc->is_assign())
    out << "ASSIGN ";
  else if (pc->is_function_call())
    out << "CALL   ";
  else
    out << "(?)    ";

  out << "\n";

  if (pc->is_other() || pc->is_assign())
  {
    irep_idt identifier;

    if (!is_nil_expr(original_lhs))
      identifier = to_symbol2t(original_lhs).get_symbol_name();
    else
      identifier = to_symbol2t(lhs).get_symbol_name();

    out << "  " << identifier << " = " << from_expr(ns, identifier, value)
        << "\n";
  }
  else if (pc->is_assert())
  {
    if (!guard)
    {
      out << "Violated property:"
          << "\n";
      if (pc->location.is_nil())
        out << "  " << pc->location << "\n";

      if (!comment.empty())
      {
        out << "  " << comment << "\n";
        std::string cwes = format_cwe_list(cwe_for(comment));
        if (!cwes.empty())
          out << "  CWE: " << cwes << "\n";
      }
      out << "  " << from_expr(ns, "", pc->guard) << "\n";
      out << "\n";
    }
  }

  out << "\n";
}

void counterexample_value(
  std::ostream &out,
  const namespacet &ns,
  const expr2tc &lhs,
  const expr2tc &value)
{
  out << "  " << from_expr(ns, "", lhs);
  if (is_nil_expr(value))
    out << "(assignment removed)";
  else
  {
    out << " = " << from_expr(ns, "", value);

    // Don't print the bit-vector if we're running on integer/real mode
    if (is_constant_expr(value) && !config.options.get_bool_option("ir"))
    {
      std::string binary_value = "";
      if (is_bv_type(value))
      {
        binary_value = integer2binary(
          to_constant_int2t(value).value, value->type->get_width());
      }
      else if (is_fixedbv_type(value))
      {
        binary_value =
          to_constant_fixedbv2t(value).value.to_expr().get_value().as_string();
      }
      else if (is_floatbv_type(value))
      {
        binary_value =
          to_constant_floatbv2t(value).value.to_expr().get_value().as_string();
      }

      if (!binary_value.empty())
      {
        out << " (";

        std::string::size_type i = 0;
        for (const auto c : binary_value)
        {
          out << c;
          if (++i % 8 == 0 && binary_value.size() != i)
            out << ' ';
        }

        out << ")";
      }
    }

    out << "\n";
  }
}

void show_goto_trace_gui(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  locationt previous_location;

  for (const auto &step : goto_trace.steps)
  {
    const locationt &location = step.pc->location;

    if ((step.type == symex_targett::ASSERT) && !step.guard)
    {
      out << "FAILED"
          << "\n"
          << step.comment << "\n" // value
          << "\n"                 // PC
          << location.file() << "\n"
          << location.line() << "\n"
          << location.column() << "\n";
    }
    else if (step.type == symex_targett::ASSIGNMENT)
    {
      irep_idt identifier;

      if (!is_nil_expr(step.original_lhs))
        identifier = to_symbol2t(step.original_lhs).get_symbol_name();
      else
        identifier = to_symbol2t(step.lhs).get_symbol_name();

      std::string value_string = from_expr(ns, identifier, step.value);

      const symbolt *symbol = ns.lookup(identifier);
      irep_idt base_name;
      if (symbol)
        base_name = symbol->name;

      out << "TRACE"
          << "\n";

      out << identifier << "," << base_name << ","
          << get_type_id(step.value->type) << "," << value_string << "\n"
          << step.step_nr << "\n"
          << step.pc->location.file() << "\n"
          << step.pc->location.line() << "\n"
          << step.pc->location.column() << "\n";
    }
    else if (location != previous_location)
    {
      // just the location

      if (!location.file().empty())
      {
        out << "TRACE"
            << "\n";

        out << "," // identifier
            << "," // base_name
            << "," // type
            << ""
            << "\n" // value
            << step.step_nr << "\n"
            << location.file() << "\n"
            << location.line() << "\n"
            << location.column() << "\n";
      }
    }

    previous_location = location;
  }
}

/*
   Return true if
   - the location's file_name matches one of the user's input files or one of
     the user-supplied include files
   - the location is explicitly labeled as user_provided
   - the location is empty
   - the location is the esbmc_intrinsics.h exception
*/
bool input_file_check(const locationt &l)
{
  // probably esbmc internally converted stuff
  if (l.as_string() == "" || l.location().user_provided())
    return true;
  const irep_idt &f_name = l.get_file();
  if (f_name.empty())
    return true;
  /* config.args, not options["input-file"]: the latter retains only the last
     positional argument, so under multiple input files every step from an
     earlier one would look foreign. */
  for (const std::string &arg : config.args)
    if (f_name == arg)
      return true;
  for (const auto &inc : config.ansi_c.include_files)
  {
    if (f_name == inc)
      return true;
  }

  // exception
  if (f_name == "esbmc_intrinsics.h")
    return true;

  return false;
}

/* The step's own file, when the witness does not already describe it.
   A consumer resolves a bare line number against the program file, so a step
   from a header or an operational model has to name where it came from.
   The reference is the verified file, because that is the only case in which
   witness_start_line expresses the line in the program file. */
static std::string
witness_origin_file(const locationt &loc, const std::string &verified_file)
{
  const std::string file = loc.get_file().as_string();
  return file == verified_file ? "" : file;
}

static BigInt witness_start_line(
  const locationt &loc,
  const std::string &verified_file,
  optionst &options)
{
  BigInt line = std::atoi(loc.get_line().c_str());
  /* get_line_number() remaps by matching the text of verified_file's line into
     the program file, which says nothing about a line of some other file. */
  const irep_idt &file = loc.get_file();
  if (!file.empty() && file.as_string() != verified_file)
    return line;
  return get_line_number(verified_file, line, options);
}

static std::string
witness_waypoint_file(const locationt &loc, const std::string &verified_file)
{
  const irep_idt &file = loc.get_file();
  return file.empty() ? verified_file : file.as_string();
}

static waypoint witness_waypoint(
  const locationt &loc,
  const std::string &verified_file,
  optionst &options)
{
  waypoint wp;
  wp.file = witness_waypoint_file(loc, verified_file);
  wp.line = witness_start_line(loc, verified_file, options);
  wp.column = loc.get_column().c_str();
  wp.function = loc.function().c_str();
  return wp;
}

/* README-YAML.md requires a waypoint's file_name to be one of the task's input
   files, so a step inside a header or an operational model is reported at its
   innermost call site that is one. Null when no frame qualifies. */
static const locationt *witness_input_location(
  goto_programt::const_targett pc,
  const std::vector<stack_framet> &stack_trace)
{
  if (input_file_check(pc->location))
    return &pc->location;

  for (const auto &frame : stack_trace)
    if (
      frame.src != nullptr && frame.src->is_set &&
      input_file_check(frame.src->pc->location))
      return &frame.src->pc->location;

  return nullptr;
}

/* A violation witness whose target waypoint is missing guides a validator
   nowhere, so an unhoistable target is attributed to the verified file rather
   than dropped: a file_name the validator was never given makes the witness
   ill-formed, while a line it cannot match only costs confirmation. */
static waypoint witness_target_waypoint(
  goto_programt::const_targett pc,
  const std::vector<stack_framet> &stack_trace,
  const std::string &verified_file,
  optionst &options)
{
  const locationt *loc = witness_input_location(pc, stack_trace);
  waypoint wp = witness_waypoint(
    loc ? full_expression_start(*loc) : pc->location, verified_file, options);
  wp.type = waypoint::target;
  if (loc == nullptr)
    wp.file = verified_file;
  return wp;
}

void show_state_header(
  std::ostream &out,
  const goto_trace_stept &state,
  const locationt &location,
  unsigned step_nr,
  const bool simplify_trace)
{
  out << "\n";
  if (simplify_trace)
  {
    show_simplified_location(out, location);
    out << "------------------------"
        << "\n";
  }
  else
  {
    out << "State " << step_nr;
    out << " " << location << " thread " << state.thread_nr << "\n";

    out << "----------------------------------------------------"
        << "\n";
  }
}

/* The values a violated claim reads through a pointer into an object the
 * program never wrote. No assignment step carries them, so without this the
 * witness names every local but not what made the property fail, and a
 * validator cannot replay it (#7858). */
static std::string
format_nondet_reads(const namespacet &ns, const goto_trace_stept &step)
{
  std::string assumptions;
  for (const auto &[lvalue, value] : step.nondet_reads)
  {
    std::string one =
      from_expr(ns, "", lvalue, presentationt::WITNESS) +
      " == " + from_expr(ns, "", value, presentationt::WITNESS) + ";";
    std::replace(one.begin(), one.end(), '$', '_');
    check_replace_invalid_assignment(one);
    if (!one.empty())
      assumptions += assumptions.empty() ? one : " " + one;
  }
  return assumptions;
}

void violation_graphml_goto_trace(
  optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace,
  const std::string &output_path_override)
{
  grapht graph(grapht::VIOLATION);
  graph.verified_file = options.get_option("input-file");

  log_progress(
    "Generating Violation Graphml Witness for: {}", graph.verified_file);

  edget *first_edge = &graph.edges.at(0);
  nodet *prev_node = first_edge->to_node;

  for (const auto &step : goto_trace.steps)
  {
    switch (step.type)
    {
    case symex_targett::ASSERT:
      if (!step.guard)
      {
        graph.check_create_new_thread(step.thread_nr, prev_node);
        prev_node = graph.edges.back().to_node;

        nodet *violation_node = new nodet();
        violation_node->violation = true;
        violation_node->cwe = format_cwe_list(cwe_for(step.comment));

        edget violation_edge(prev_node, violation_node);
        violation_edge.thread_id = std::to_string(step.thread_nr);
        violation_edge.assumption = format_nondet_reads(ns, step);
        violation_edge.origin_file =
          witness_origin_file(step.pc->location, graph.verified_file);
        violation_edge.start_line =
          witness_start_line(step.pc->location, graph.verified_file, options);

        graph.edges.push_back(violation_edge);

        /* having printed a property violation, don't print more steps. */

        graph.generate_graphml(options, output_path_override);
        return;
      }
      break;

    case symex_targett::ASSIGNMENT:
      if (
        step.pc->is_assign() || step.pc->is_return() ||
        (step.pc->is_other() && is_nil_expr(step.lhs)) ||
        step.pc->is_function_call())
      {
        /* The argv setup in __ESBMC_main has no line; an edge for it would
           carry startline 0 (#9677). */
        if (step.pc->location.get_line().empty())
          break;

        std::string assignment = get_formated_assignment(ns, step);

        graph.check_create_new_thread(step.thread_nr, prev_node);
        prev_node = graph.edges.back().to_node;

        edget new_edge;
        new_edge.thread_id = std::to_string(step.thread_nr);
        new_edge.assumption = assignment;
        new_edge.origin_file =
          witness_origin_file(step.pc->location, graph.verified_file);
        new_edge.start_line =
          witness_start_line(step.pc->location, graph.verified_file, options);

        nodet *new_node = new nodet();
        new_edge.from_node = prev_node;
        new_edge.to_node = new_node;
        prev_node = new_node;
        graph.edges.push_back(new_edge);
      }
      break;

    default:
      continue;
    }
  }
}

/* A function_return waypoint for each nondet call made up to @p step. It sits
   at the call's ')' and names the value the call returned, not the value of
   the expression the call is part of (#9881). */
static void push_nondet_return_waypoints(
  const namespacet &ns,
  const goto_trace_stept &step,
  yamlt &yml,
  optionst &options)
{
  for (const auto &[rparen, value] : step.nondet_returns)
  {
    if (!input_file_check(rparen))
      continue;

    std::string result = from_expr(ns, "", value, presentationt::WITNESS);
    check_replace_invalid_assignment(result);
    if (result.empty())
      continue;

    // `\result == NAN` holds for no value, NaN included, so a NaN result
    // gets no waypoint.
    if (
      is_constant_floatbv2t(value) &&
      to_constant_floatbv2t(value).value.is_NaN())
      continue;

    waypoint wp = witness_waypoint(rparen, yml.verified_file, options);
    wp.type = waypoint::function_return;
    wp.value = "\\result == " + result;
    yml.segments.push_back(wp);
  }
}

/* Branches in a recursive trace number in the hundreds of thousands (#9689),
   and a witness that size cannot be loaded by the validator. Dropping a
   branching waypoint only weakens guidance. */
static constexpr size_t max_branching_waypoints = 1000;

/* Whether @p entry can call itself, directly or through other functions. */
static bool
recursive(const goto_functionst &goto_functions, const irep_idt &entry)
{
  std::unordered_set<irep_idt, irep_id_hash> seen;
  std::vector<irep_idt> todo{entry};
  while (!todo.empty())
  {
    const auto it = goto_functions.function_map.find(todo.back());
    todo.pop_back();
    if (it == goto_functions.function_map.end())
      continue;
    for (const auto &i : it->second.body.instructions)
    {
      if (!i.is_function_call())
        continue;
      const expr2tc &callee = to_code_function_call2t(i.code).function;
      if (!is_symbol2t(callee) || to_symbol2t(callee).thename == entry)
        return true;
      if (seen.insert(to_symbol2t(callee).thename).second)
        todo.push_back(to_symbol2t(callee).thename);
    }
  }
  return false;
}

/* The entry function, whose `}` locates the end of __ESBMC_main. */
static const goto_functionst::function_mapt::value_type *entry_function(
  const goto_functionst &goto_functions,
  const goto_trace_stept &violation)
{
  for (const auto &f : goto_functions.function_map)
    if (
      f.first != violation.pc->function &&
      !f.second.body.instructions.empty() &&
      f.second.body.instructions.back().location == violation.pc->location)
      return &f;
  return nullptr;
}

/* Where the path resumes in @p body after its last step there. */
static goto_programt::const_targett resume_point(
  const goto_tracet &goto_trace,
  const goto_trace_stept &violation,
  const irep_idt &function,
  const goto_programt &body)
{
  const goto_trace_stept *last = nullptr;
  for (const auto &step : goto_trace.steps)
  {
    if (&step == &violation)
      break;
    if (step.thread_nr == violation.thread_nr && step.pc->function == function)
      last = &step;
  }
  if (last == nullptr)
    return body.instructions.begin();
  const goto_programt::const_targett pc = last->pc;
  if (last->type == symex_targett::BREANCHING && pc->is_goto())
    return last->guard != pc->flipped_guard ? pc->targets.front()
                                            : std::next(pc);
  return pc->is_goto() || pc->is_return() || pc->is_end_function()
           ? pc
           : std::next(pc);
}

/* The `return` every path from @p pc leaves the function by, or null when
   paths can leave by different ones or fall off the end. */
static const goto_programt::instructiont *
sole_return(goto_programt::const_targett pc)
{
  const goto_programt::instructiont *ret = nullptr;
  for (; !pc->is_end_function(); ++pc)
  {
    if (pc->is_return())
    {
      if (ret != nullptr)
        return nullptr;
      ret = &*pc;
    }
    if (
      pc->is_goto() && !is_false(pc->guard) &&
      (pc->targets.front()->location_number <= pc->location_number ||
       pc->targets.front()->is_end_function()))
      return nullptr;
  }
  return ret != nullptr && std::prev(pc)->is_return() ? ret : nullptr;
}

/* A valid-memtrack violation at program exit is checked at the end of
   __ESBMC_main, located at the entry function's `}`. The witness format makes
   its target the `return` the path left the entry function by (#9939). The
   path is replayed from the branch directions the trace records; from a branch
   it does not record, the `return` must be the only one ahead. Branching steps
   carry no stack trace, so a recursive entry function is not replayed. */
static const goto_programt::instructiont *memtrack_return(
  const goto_tracet &goto_trace,
  const goto_functionst &goto_functions,
  const goto_trace_stept &violation)
{
  if (
    !violation.pc->is_end_function() ||
    violation.pc->function != "__ESBMC_main" ||
    !has_prefix(violation.comment, "dereference failure: forgotten memory"))
    return nullptr;

  const auto *entry = entry_function(goto_functions, violation);
  if (entry == nullptr || recursive(goto_functions, entry->first))
    return nullptr;

  auto pc =
    resume_point(goto_trace, violation, entry->first, entry->second.body);
  for (; !pc->is_return(); ++pc)
  {
    if (pc->is_end_function())
      return nullptr;
    if (!pc->is_goto() || is_false(pc->guard))
      continue;
    if (
      !is_true(pc->guard) ||
      pc->targets.front()->location_number <= pc->location_number)
      return sole_return(pc);
    pc = std::prev(pc->targets.front());
  }
  return &*pc;
}

/* The failed assertion's access, and for a data race the access it raced
   with. */
static void push_violation_targets(
  yamlt &yml,
  const goto_tracet &goto_trace,
  const goto_functionst &goto_functions,
  const goto_trace_stept &step,
  optionst &options)
{
  const goto_programt::instructiont *ret =
    memtrack_return(goto_trace, goto_functions, step);
  if (ret != nullptr && input_file_check(ret->location))
  {
    waypoint wp = witness_waypoint(ret->location, yml.verified_file, options);
    wp.type = waypoint::target;
    yml.segments.push_back(wp);
  }
  else
    yml.segments.push_back(witness_target_waypoint(
      step.pc, step.stack_trace, yml.verified_file, options));
  if (step.racing_access)
    yml.racing_target = witness_target_waypoint(
      step.racing_access->pc,
      step.racing_access->stack_trace,
      yml.verified_file,
      options);
}

/* A validator binds a branching waypoint to the next visit of its source
   location, not of its instruction: --goto-unwind copies a loop body. */
static std::string branch_location(const goto_trace_stept &step)
{
  const locationt &l = step.pc->location;
  return id2string(l.get_file()) + ":" + id2string(l.get_line()) + ":" +
         id2string(l.get_column());
}

/* The nondet-dependent visits of each branch location up to the violation. */
static std::unordered_map<std::string, size_t>
nondet_branch_visits(const goto_tracet &goto_trace)
{
  std::unordered_map<std::string, size_t> visits;
  for (const auto &step : goto_trace.steps)
  {
    if (step.type == symex_targett::ASSERT && !step.guard)
      break;
    if (
      step.type == symex_targett::BREANCHING && step.pc->is_goto() &&
      step.nondet_dependent)
      ++visits[branch_location(step)];
  }
  return visits;
}

void violation_yaml_goto_trace(
  optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace,
  const goto_functionst &goto_functions,
  const std::string &output_path_override)
{
  yamlt yml(yamlt::VIOLATION);
  yml.verified_file = options.get_option("input-file");
  log_progress("Generating Violation Yaml Witness for: {}", yml.verified_file);

  auto nondet_visits_ahead = nondet_branch_visits(goto_trace);
  size_t branching_waypoints = 0;
  for (const auto &step : goto_trace.steps)
  {
    push_nondet_return_waypoints(ns, step, yml, options);

    switch (step.type)
    {
    case symex_targett::ASSERT:
      if (!step.guard)
      {
        push_violation_targets(yml, goto_trace, goto_functions, step, options);

        /* having printed a property violation, don't print more steps. */

        yml.generate_yaml(options, output_path_override);
        return;
      }
      break;

    case symex_targett::BREANCHING:
      if (step.pc->is_goto())
      {
        size_t &ahead = nondet_visits_ahead[branch_location(step)];
        if (step.nondet_dependent)
          --ahead;

        /* A validator cannot match a follow waypoint against a file it was
           not given; dropping one only weakens guidance. witnesslint also
           rejects line 0, e.g. the argv setup in __ESBMC_main (#9650). A
           branch fixed by the program text gets none, as on deep recursion
           they number in the hundreds of thousands (#9689), unless a
           nondet-dependent visit of its location follows: a validator would
           bind that visit's waypoint to this one. The ones kept can fill the
           cap before a later nondet waypoint. An operand of && or || gets none:
           the waypoint's value is the whole condition's, and a validator would
           apply it to another operand. */
        if (
          branching_waypoints >= max_branching_waypoints ||
          !input_file_check(step.pc->location) ||
          step.pc->location.get_line().empty() ||
          (!step.nondet_dependent && ahead == 0) || step.condition_operand)
          break;

        /* A switch would take one waypoint naming the case it enters, but
           CPAchecker cannot parse any switch waypoint and the format does not
           require one, so emit none (#9882). */
        if (!step.pc->switch_case_ids.empty())
          break;

        ++branching_waypoints;
        waypoint wp =
          witness_waypoint(step.pc->location, yml.verified_file, options);
        wp.type = waypoint::branching;
        wp.value = !step.guard ? "true" : "false";
        yml.segments.push_back(wp);
      }
      break;

    default:
      continue;
    }
  }
}

void non_termination_yaml_goto_trace(optionst &options, const locationt &loc)
{
  yamlt yml(yamlt::VIOLATION);
  yml.verified_file = options.get_option("input-file");
  log_progress(
    "Generating Non-Termination Yaml Witness for: {}", yml.verified_file);

  waypoint wp = witness_waypoint(loc, yml.verified_file, options);
  wp.type = waypoint::assumption;
  wp.action = waypoint::cycle;
  wp.value = "1";
  yml.segments.push_back(wp);
  yml.generate_yaml(options);
}

void correctness_graphml_goto_trace(
  optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  grapht graph(grapht::CORRECTNESS);
  graph.verified_file = options.get_option("input-file");
  log_progress(
    "Generating Correctness Graphml Witness for: {}", graph.verified_file);

  edget *first_edge = &graph.edges.at(0);
  nodet *prev_node = first_edge->to_node;

  for (const auto &step : goto_trace.steps)
  {
    /* checking restrictions for correctness GraphML */
    if (
      (!(is_valid_witness_step(ns, step))) ||
      (!(step.is_assume() || step.is_assert())))
      continue;

    std::string invariant = get_invariant(
      graph.verified_file,
      std::atoi(step.pc->location.get_line().c_str()),
      options);

    if (invariant.empty())
      continue; /* we don't have to consider this invariant */

    nodet *new_node = new nodet();
    edget *new_edge = new edget();
    std::string function = step.pc->location.get_function().c_str();
    new_edge->origin_file =
      witness_origin_file(step.pc->location, graph.verified_file);
    new_edge->start_line =
      witness_start_line(step.pc->location, graph.verified_file, options);
    new_node->invariant = invariant;
    new_node->invariant_scope = function;

    new_edge->from_node = prev_node;
    new_edge->to_node = new_node;
    prev_node = new_node;
    graph.edges.push_back(*new_edge);
  }

  graph.generate_graphml(options);
}

void correctness_yaml_goto_trace(
  optionst &options,
  const namespacet &ns [[maybe_unused]],
  const goto_tracet &goto_trace [[maybe_unused]],
  const std::vector<std::pair<locationt, std::string>> &transition_invariants)
{
  yamlt yml(yamlt::CORRECTNESS);
  yml.verified_file = options.get_option("input-file");
  log_progress(
    "Generating Correctness Yaml Witness for: {}", yml.verified_file);

  for (const auto &[loc, value] : transition_invariants)
  {
    const waypoint at = witness_waypoint(loc, yml.verified_file, options);
    invariant inv;
    inv.type = invariant::loop_transition_invariant;
    inv.file = at.file;
    inv.line = at.line;
    inv.column = at.column;
    inv.function = at.function;
    inv.value = value;
    inv.format = "ext_c_expression";
    yml.invariants.push_back(inv);
  }

  yml.generate_yaml(options);
}

void write_correctness_witnesses(
  optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace,
  const std::vector<std::pair<locationt, std::string>> &transition_invariants)
{
  if (options.get_bool_option("result-only"))
    return;

  if (!options.get_option("witness-output-graphml").empty())
    correctness_graphml_goto_trace(options, ns, goto_trace);

  if (!options.get_option("witness-output-yaml").empty())
    correctness_yaml_goto_trace(options, ns, goto_trace, transition_invariants);
}

void appendInfo(
  std::string &dest,
  const std::string &label,
  const std::string &value)
{
  if (!value.empty())
  {
    if (!dest.empty())
      dest += " ";
    // Append in place rather than `label + " " + id2string(value)`, which
    // builds two throwaway temporaries per trace label (this scales with
    // counterexample size).
    dest += label;
    dest += ' ';
    dest += value;
  }
}

void show_simplified_location(std::ostream &out, const locationt &location)
{
  std::string dest;
  const irep_idt &file = location.get_file();
  const irep_idt &line = location.get_line();
  const irep_idt &function = location.get_function();

  if (file != "")
    appendInfo(dest, "file", id2string(file));
  if (line != "")
    appendInfo(dest, "line", id2string(line));
  if (function != "")
    appendInfo(dest, "function", id2string(function));
  out << dest << "\n";
}

static const irep_idt &
frame_function_name(const namespacet &ns, const irep_idt &id)
{
  const symbolt *fn = ns.lookup(id);
  return fn ? fn->name : id;
}

/// A call binds its arguments at the callee's first instruction, whatever
/// that instruction is, so such a step cannot be recognised by its pc.
static bool binds_parameter(const namespacet &ns, const goto_trace_stept &step)
{
  if (!is_symbol2t(step.original_lhs))
    return false;
  const symbolt *s = ns.lookup(to_symbol2t(step.original_lhs).thename);
  return s != nullptr && s->is_parameter;
}

static void show_stack_frame(
  std::ostream &out,
  const irep_idt &name,
  const stack_framet &frame)
{
  out << "  " << name;
  if (frame.src != nullptr && frame.src->pc->location.is_not_nil())
    out << " at " << frame.src->pc->location;
  out << std::endl;
}

static void show_function_call_trace(
  std::ostream &out,
  const goto_tracet &goto_trace,
  const goto_trace_stept &last_step)
{
  // Print chronological function-call trace.
  // Each step carries the call stack at that point (innermost-first,
  // outermost-last). Diff each step's stack against the previous to
  // detect newly pushed frames and emit them in call order.
  out << "Function call trace:" << std::endl;
  std::vector<stack_framet> prev;
  for (const auto &s : goto_trace.steps)
  {
    const auto &cur = s.stack_trace;
    // Skip steps captured outside any function context (global
    // initializers, per-tx reseed scaffolding, etc.). Their
    // stack_trace is empty even though the program clearly has
    // frames on the call stack at that moment; treating that as
    // a reset would falsely re-emit the entire chain on the
    // next non-empty step.
    if (cur.empty())
    {
      if (&s == &last_step)
        break;
      continue;
    }
    // Common suffix length (outermost end of both vectors).
    size_t common = 0;
    while (common < prev.size() && common < cur.size() &&
           prev[prev.size() - 1 - common] == cur[cur.size() - 1 - common])
      ++common;
    // New frames live at the front of `cur` (innermost end);
    // emit outermost-of-new first to preserve call order.
    size_t n_new = cur.size() - common;
    for (size_t k = n_new; k-- > 0;)
      show_stack_frame(out, cur[k].function, cur[k]);
    prev = cur;
    if (&s == &last_step)
      break;
  }
}

static void show_failed_property(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace,
  const goto_trace_stept &step,
  bool reachability_trace,
  bool simplify_trace)
{
  show_state_header(out, step, step.pc->location, step.step_nr, simplify_trace);
  out << (reachability_trace ? "Reached:" : "Violated property:") << "\n";
  if (!step.pc->location.is_nil())
  {
    if (simplify_trace)
    {
      out << "  ";
      show_simplified_location(out, step.pc->location);
    }
    else
      out << "  " << step.pc->location << "\n";
  }
  if (config.options.get_bool_option("show-stacktrace"))
  {
    // Print stack trace
    out << "Stack trace:" << std::endl;
    for (const auto &frame : step.stack_trace)
      show_stack_frame(out, frame_function_name(ns, frame.function), frame);
  }
  if (config.options.get_bool_option("show-funccall-trace"))
    show_function_call_trace(out, goto_trace, step);

  out << "  " << step.comment << "\n";
  // A reached coverage probe carries no weakness: its guard is the
  // constant `false` the instrumentation put there, not the program's
  // own condition, so neither a CWE nor the guard text says anything
  // about the program.
  if (!reachability_trace)
  {
    std::string cwes = format_cwe_list(cwe_for(step.comment));
    if (!cwes.empty())
      out << "  CWE: " << cwes << "\n";

    if (step.pc->is_assert())
      out << "  " << from_expr(ns, "", step.pc->guard) << "\n";
  }
}

void show_goto_trace(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace,
  bool reachability_trace)
{
  unsigned prev_step_nr = 0;
  bool first_step = true;
  bool cex_only = config.options.get_bool_option("cex-only");
  bool simplify_trace = config.options.get_bool_option("simplify-trace");

  for (const auto &step : goto_trace.steps)
  {
    // we only care about the counter example, which is only triggered by assert
    // steps. Ignore all other steps.
    if (cex_only && step.type != symex_targett::ASSERT)
      continue;
    switch (step.type)
    {
    case symex_targett::ASSERT:
      if (!step.guard)
      {
        show_failed_property(
          out, ns, goto_trace, step, reachability_trace, simplify_trace);

        // The trace ends at the assert that terminated it.
        return;
      }
      break;

    case symex_targett::ASSIGNMENT:
      if (
        step.pc->is_assign() || step.pc->is_return() ||
        (step.pc->is_other() && is_nil_expr(step.lhs)) ||
        step.pc->is_function_call() || binds_parameter(ns, step))
      {
        if (simplify_trace)
        {
          // if the file is empty then it's probably internally created and
          // should not print out
          if (!input_file_check(step.pc->location))
            break;
        }
        if (prev_step_nr != step.step_nr || first_step)
        {
          first_step = false;
          prev_step_nr = step.step_nr;
          show_state_header(
            out, step, step.pc->location, step.step_nr, simplify_trace);
        }
        counterexample_value(out, ns, step.lhs, step.value);
      }
      break;

    case symex_targett::OUTPUT:
    {
      printf_formattert printf_formatter;
      printf_formatter(step.format_string, step.output_args);
      printf_formatter.print(out);
      out << "\n";
      break;
    }

    case symex_targett::RENUMBER:
      out << "Renumbered pointer to ";
      counterexample_value(out, ns, step.lhs, step.value);
      break;

    case symex_targett::ASSUME:
    case symex_targett::SKIP:
    case symex_targett::BREANCHING:
      // Something deliberately ignored
      break;

    default:
      assert(false);
    }
  }
}
