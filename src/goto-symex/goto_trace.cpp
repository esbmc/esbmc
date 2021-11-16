#include <cassert>
#include <cstring>
#include <goto-symex/goto_trace.h>
#include <goto-symex/printf_formatter.h>
#include <goto-symex/witnesses.h>

#include <regex>
#include <langapi/language_util.h>
#include <langapi/languages.h>
#include <util/arith_tools.h>
#include <util/std_types.h>
#include <ostream>

void goto_tracet::output(const class namespacet &ns, std::ostream &out) const
{
  for(const auto &step : steps)
    step.output(ns, out);
}

void goto_trace_stept::dump() const
{
  std::ostringstream oss;
  output(*migrate_namespace_lookup, oss);
  log_debug("{}", oss.str());
}

void goto_trace_stept::output(const namespacet &ns, std::ostream &out) const
{
  switch(type)
  {
  case goto_trace_stept::ASSERT:
    out << "ASSERT";
    break;

  case goto_trace_stept::ASSUME:
    out << "ASSUME";
    break;

  case goto_trace_stept::ASSIGNMENT:
    out << "ASSIGNMENT";
    break;

  default:
    assert(false);
  }

  if(type == ASSERT || type == ASSUME)
    out << " (" << guard << ")";

  out << "\n";

  if(!pc->location.is_nil())
    out << pc->location << "\n";

  if(pc->is_goto())
    out << "GOTO   ";
  else if(pc->is_assume())
    out << "ASSUME ";
  else if(pc->is_assert())
    out << "ASSERT ";
  else if(pc->is_other())
    out << "OTHER  ";
  else if(pc->is_assign())
    out << "ASSIGN ";
  else if(pc->is_function_call())
    out << "CALL   ";
  else
    out << "(?)    ";

  out << "\n";

  if(pc->is_other() || pc->is_assign())
  {
    irep_idt identifier;

    if(!is_nil_expr(original_lhs))
      identifier = to_symbol2t(original_lhs).get_symbol_name();
    else
      identifier = to_symbol2t(lhs).get_symbol_name();

    out << "  " << identifier << " = " << from_expr(ns, identifier, value)
        << "\n";
  }
  else if(pc->is_assert())
  {
    if(!guard)
    {
      out << "Violated property:"
          << "\n";
      if(pc->location.is_nil())
        out << "  " << pc->location << "\n";

      if(!comment.empty())
        out << "  " << comment << "\n";
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
  if(is_nil_expr(value))
    out << "(assignment removed)";
  else
  {
    out << " = " << from_expr(ns, "", value);

    // Don't print the bit-vector if we're running on integer/real mode
    if(is_constant_expr(value) && !config.options.get_bool_option("ir"))
    {
      std::string binary_value = "";
      if(is_bv_type(value))
      {
        binary_value = integer2binary(
          to_constant_int2t(value).value, value->type->get_width());
      }
      else if(is_fixedbv_type(value))
      {
        binary_value =
          to_constant_fixedbv2t(value).value.to_expr().get_value().as_string();
      }
      else if(is_floatbv_type(value))
      {
        binary_value =
          to_constant_floatbv2t(value).value.to_expr().get_value().as_string();
      }

      if(!binary_value.empty())
      {
        out << " (";

        std::string::size_type i = 0;
        for(const auto c : binary_value)
        {
          out << c;
          if(++i % 8 == 0 && binary_value.size() != i)
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

  for(const auto &step : goto_trace.steps)
  {
    const locationt &location = step.pc->location;

    if((step.type == goto_trace_stept::ASSERT) && !step.guard)
    {
      out << "FAILED"
          << "\n"
          << step.comment << "\n" // value
          << "\n"                 // PC
          << location.file() << "\n"
          << location.line() << "\n"
          << location.column() << "\n";
    }
    else if(step.type == goto_trace_stept::ASSIGNMENT)
    {
      irep_idt identifier;

      if(!is_nil_expr(step.original_lhs))
        identifier = to_symbol2t(step.original_lhs).get_symbol_name();
      else
        identifier = to_symbol2t(step.lhs).get_symbol_name();

      std::string value_string = from_expr(ns, identifier, step.value);

      const symbolt *symbol = ns.lookup(identifier);
      irep_idt base_name;
      if(symbol)
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
    else if(location != previous_location)
    {
      // just the location

      if(!location.file().empty())
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

void show_state_header(
  std::ostream &out,
  const goto_trace_stept &state,
  const locationt &location,
  unsigned step_nr)
{
  out << "\n";
  out << "State " << step_nr;
  out << " " << location << " thread " << state.thread_nr << "\n";
  out << "----------------------------------------------------"
      << "\n";
}

void violation_graphml_goto_trace(
  optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  grapht graph(grapht::VIOLATION);
  graph.verified_file = options.get_option("input-file");

  log_status("Generating Violation Witness for: {}", graph.verified_file);

  std::string prev_assignment;
  std::string program_file = options.get_option("input-file");
  size_t prev_stack_size = 0;
  std::string prev_scope;

  // Our first node always has to start with calling main (after the globals),
  // however, esbmc could've sliced the initial calls in main and will start
  // by calling other fuctions, this flag prevents that by always forcing
  // the creation of a node calling main after the globals
  bool saw_main = false;

  // Create the entry node
  nodet *first_node = new nodet();
  first_node->entry = true;
  nodet *prev_node = first_node;

  for(const auto &step : goto_trace.steps)
  {
    // We only care about assignments in the file under verification
    if(
      program_file.find(step.pc->location.get_file().as_string()) ==
      std::string::npos)
      continue;

    // Stack size changed, either we entered a function or we exited it
    if(step.stack_trace.size() != prev_stack_size)
    {
      // If we didn't see main yet, force the creation of the main node
      bool duplicate_main = false;
      if(!saw_main)
      {
        assert(step.stack_trace.size() > prev_stack_size);
        graph.create_initial_edge(prev_node);
        saw_main = true;
        prev_node = graph.edges.back().to_node;
        prev_scope = "main";
        duplicate_main = true;
      }

      std::string cur_scope = step.pc->location.function().as_string();
      if(!(cur_scope == "main" && duplicate_main))
      {
        nodet *func_node = new nodet();
        edget func_edge(prev_node, func_node);

        // We entered a function, create a enter_function node
        if(step.stack_trace.size() > prev_stack_size)
          func_edge.enter_function = cur_scope;
        else
          // We exited a function, create a return_from_function node
          func_edge.return_from_function = prev_scope;

        graph.edges.push_back(func_edge);

        prev_scope = cur_scope;
        prev_stack_size = step.stack_trace.size();
        prev_node = graph.edges.back().to_node;
      }
    }

    switch(step.type)
    {
    case goto_trace_stept::ASSERT:
      if(!step.guard)
      {
        graph.check_create_new_thread(step.thread_nr, prev_node);
        prev_node = graph.edges.back().to_node;

        nodet *violation_node = new nodet();
        violation_node->violation = true;

        edget violation_edge(prev_node, violation_node);
        violation_edge.thread_id = std::to_string(step.thread_nr);
        violation_edge.start_line = get_line_number(
          graph.verified_file,
          std::atoi(step.pc->location.get_line().c_str()),
          options);

        graph.edges.push_back(violation_edge);

        /* having printed a property violation, don't print more steps. */
        graph.generate_graphml(options);
        return;
      }
      break;

    case goto_trace_stept::ASSIGNMENT:
      if(
        step.pc->is_assign() || step.pc->is_return() ||
        (step.pc->is_other() && is_nil_expr(step.lhs)))
      {
        std::string assignment = get_formated_assignment(ns, step);

        // Let's not repeat ourselves
        if(assignment == prev_assignment)
          continue;
        prev_assignment = assignment;

        graph.check_create_new_thread(step.thread_nr, prev_node);
        prev_node = graph.edges.back().to_node;

        edget new_edge;
        new_edge.thread_id = std::to_string(step.thread_nr);
        new_edge.assumption = assignment;
        new_edge.start_line = get_line_number(
          graph.verified_file,
          std::atoi(step.pc->location.get_line().c_str()),
          options);

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

void correctness_graphml_goto_trace(
  optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  grapht graph(grapht::CORRECTNESS);
  graph.verified_file = options.get_option("input-file");
  log_status("Generating Correctness Witness for: {}", graph.verified_file);

  edget *first_edge = &graph.edges.at(0);
  nodet *prev_node = first_edge->to_node;

  for(const auto &step : goto_trace.steps)
  {
    /* checking restrictions for correctness GraphML */
    if(
      (!(is_valid_witness_step(ns, step))) ||
      (!(step.is_assume() || step.is_assert())))
      continue;

    std::string invariant = get_invariant(
      graph.verified_file,
      std::atoi(step.pc->location.get_line().c_str()),
      options);

    if(invariant.empty())
      continue; /* we don't have to consider this invariant */

    nodet *new_node = new nodet();
    edget *new_edge = new edget();
    std::string function = step.pc->location.get_function().c_str();
    new_edge->start_line = get_line_number(
      graph.verified_file,
      std::atoi(step.pc->location.get_line().c_str()),
      options);
    new_node->invariant = invariant;
    new_node->invariant_scope = function;

    new_edge->from_node = prev_node;
    new_edge->to_node = new_node;
    prev_node = new_node;
    graph.edges.push_back(*new_edge);
  }

  graph.generate_graphml(options);
}

void show_goto_trace(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  unsigned prev_step_nr = 0;
  bool first_step = true;

  for(const auto &step : goto_trace.steps)
  {
    switch(step.type)
    {
    case goto_trace_stept::ASSERT:
      if(!step.guard)
      {
        show_state_header(out, step, step.pc->location, step.step_nr);
        out << "Violated property:"
            << "\n";
        if(!step.pc->location.is_nil())
          out << "  " << step.pc->location << "\n";
        out << "  " << step.comment << "\n";

        if(step.pc->is_assert())
          out << "  " << from_expr(ns, "", step.pc->guard) << "\n";

        // Having printed a property violation, don't print more steps.
        return;
      }
      break;

    case goto_trace_stept::ASSIGNMENT:
      if(
        step.pc->is_assign() || step.pc->is_return() ||
        (step.pc->is_other() && is_nil_expr(step.lhs)))
      {
        if(prev_step_nr != step.step_nr || first_step)
        {
          first_step = false;
          prev_step_nr = step.step_nr;
          show_state_header(out, step, step.pc->location, step.step_nr);
        }
        counterexample_value(out, ns, step.lhs, step.value);
      }
      break;

    case goto_trace_stept::OUTPUT:
    {
      printf_formattert printf_formatter;
      printf_formatter(step.format_string, step.output_args);
      printf_formatter.print(out);
      out << "\n";
      break;
    }

    case goto_trace_stept::RENUMBER:
      out << "Renumbered pointer to ";
      counterexample_value(out, ns, step.lhs, step.value);
      break;

    case goto_trace_stept::ASSUME:
    case goto_trace_stept::SKIP:
      // Something deliberately ignored
      break;

    default:
      assert(false);
    }
  }
}
