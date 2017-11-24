/*******************************************************************\

   Module: Traces of GOTO Programs

   Author: Daniel Kroening

   Date: July 2005

\*******************************************************************/

#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/version.hpp>
#include <cassert>
#include <cstring>
#include <goto-symex/goto_trace.h>
#include <goto-symex/printf_formatter.h>
#include <goto-symex/witnesses.h>
#include <iostream>
#include <regex>
#include <langapi/language_util.h>
#include <langapi/languages.h>
#include <util/arith_tools.h>
#include <util/std_types.h>
#include <boost/graph/graphml.hpp>

extern std::string verification_file;

void goto_tracet::output(const class namespacet &ns, std::ostream &out) const
{
  for(const auto & step : steps)
    step.output(ns, out);
}

void goto_trace_stept::output(const namespacet &ns, std::ostream &out) const
{
  out << "*** ";

  switch (type)
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

  out << std::endl;

  if(!pc->location.is_nil())
    out << pc->location << std::endl;

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

  out << std::endl;

  if(pc->is_other() || pc->is_assign())
  {
    irep_idt identifier;

    if(!is_nil_expr(original_lhs))
      identifier = to_symbol2t(original_lhs).get_symbol_name();
    else
      identifier = to_symbol2t(lhs).get_symbol_name();

    out << "  " << identifier << " = " << from_expr(ns, identifier, value)
        << std::endl;
  }
  else if(pc->is_assert())
  {
    if(!guard)
    {
      out << "Violated property:" << std::endl;
      if(pc->location.is_nil())
        out << "  " << pc->location << std::endl;

      if(!comment.empty())
        out << "  " << comment << std::endl;
      out << "  " << from_expr(ns, "", pc->guard) << std::endl;
      out << std::endl;
    }
  }

  out << std::endl;
}

void counterexample_value(
  std::ostream &out,
  const namespacet &ns,
  const expr2tc &lhs,
  const expr2tc &value)
{
  std::string value_string;

  if(is_nil_expr(value))
    value_string = "(assignment removed)";
  else
  {
    value_string = from_expr(ns, "", value);

    // Don't print the bit-vector if we're running on integer/real mode
    if (is_constant_expr(value) && !config.options.get_bool_option("ir"))
    {
      if(is_bv_type(value))
      {
        value_string +=
          " (" + integer2binary(to_constant_int2t(value).value, value->type->get_width()) + ")";
      }
      else if(is_fixedbv_type(value))
      {
        value_string +=
          " (" + to_constant_fixedbv2t(value).value.to_expr().get_value().as_string() + ")";
      }
      else if(is_floatbv_type(value))
      {
        value_string +=
          " (" + to_constant_floatbv2t(value).value.to_expr().get_value().as_string() + ")";
      }
    }
  }

  out << "  " << from_expr(ns, "", lhs) << " = " << value_string << std::endl;
}

void show_goto_trace_gui(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  locationt previous_location;

  for(const auto & step : goto_trace.steps)
  {
    const locationt &location = step.pc->location;

    if((step.type == goto_trace_stept::ASSERT) && !step.guard)
    {
      out << "FAILED" << std::endl << step.comment
          << std::endl // value
          << std::endl // PC
          << location.file() << std::endl << location.line() << std::endl
          << location.column() << std::endl;
    }
    else if(step.type == goto_trace_stept::ASSIGNMENT)
    {
      irep_idt identifier;

      if(!is_nil_expr(step.original_lhs))
        identifier = to_symbol2t(step.original_lhs).get_symbol_name();
      else
        identifier = to_symbol2t(step.lhs).get_symbol_name();

      std::string value_string = from_expr(ns, identifier, step.value);

      const symbolt *symbol;
      irep_idt base_name;
      if(!ns.lookup(identifier, symbol))
        base_name = symbol->base_name;

      out << "TRACE" << std::endl;

      out << identifier << "," << base_name << ","
          << get_type_id(step.value->type) << "," << value_string << std::endl
          << step.step_nr << std::endl << step.pc->location.file() << std::endl
          << step.pc->location.line() << std::endl << step.pc->location.column()
          << std::endl;
    }
    else if(location != previous_location)
    {
      // just the location

      if(!location.file().empty())
      {
        out << "TRACE" << std::endl;

        out
            << ","             // identifier
            << ","             // base_name
            << ","             // type
            << ""
            << std::endl // value
            << step.step_nr << std::endl << location.file() << std::endl
            << location.line() << std::endl << location.column() << std::endl;
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
  out << std::endl;

  if(step_nr == 0)
    out << "Initial State";
  else
    out << "State " << step_nr;

  out << " " << location << " thread " << state.thread_nr << std::endl;

  // Print stack trace

  for(const auto & it : state.stack_trace)
  {
    if(it.src == nullptr)
      out << it.function.as_string() << std::endl;
    else
      out << it.function.as_string() << " at "
          << it.src->pc->location.get_file().as_string() << " line "
          << it.src->pc->location.get_line().as_string() << std::endl;
  }

  out << "----------------------------------------------------" << std::endl;
}

void violation_graphml_goto_trace(
  optionst & options,
  const namespacet & ns,
  const goto_tracet & goto_trace)
{
  grapht graph(grapht::VIOLATION);
  graph.verified_file = verification_file;

  edget * first_edge = &graph.edges.at(0);
  nodet * prev_node = first_edge->to_node;

  for(const auto & step : goto_trace.steps)
  {
    switch (step.type)
    {
      case goto_trace_stept::ASSERT:
        if(!step.guard)
        {
          nodet * violation_node = new nodet();
          violation_node->violation = true;

          edget violation_edge(prev_node, violation_node);
          violation_edge.thread_id = std::to_string(step.thread_nr);
          violation_edge.start_line =
              get_line_number(
                verification_file,
                std::atoi(step.pc->location.get_line().c_str()),
                options);

          graph.edges.push_back(violation_edge);

          /* having printed a property violation, don't print more steps. */

          graph.generate_graphml(options);
          return;
        }
        break;

      case goto_trace_stept::ASSIGNMENT:
        if(step.pc->is_assign() || step.pc->is_return()
           || (step.pc->is_other() && is_nil_expr(step.lhs)))
        {

          std::string assignment = get_formated_assignment(ns, step);

          graph.check_create_new_thread(step.thread_nr, prev_node);
          prev_node = graph.edges.back().to_node;

          edget new_edge;
          new_edge.thread_id = std::to_string(step.thread_nr);
          new_edge.assumption = assignment;
          new_edge.start_line =
            get_line_number(
              verification_file,
              std::atoi(step.pc->location.get_line().c_str()),
              options);

          nodet * new_node = new nodet();
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
  optionst & options,
  const namespacet & ns __attribute__((unused)),
  const goto_tracet & goto_trace __attribute__((unused)) )
{
  grapht graph(grapht::CORRECTNESS);
  graph.verified_file = verification_file;

  edget * first_edge = &graph.edges.at(0);
  nodet * prev_node = first_edge->to_node;

  for(const auto & step : goto_trace.steps)
  {
    /* checking restrictions for correctness GraphML */
    if ((!(is_valid_witness_step(ns, step))) ||
        (!(step.is_assume() || step.is_assert())))
      continue;

    std::string invariant = get_invariant(
      verification_file,
      std::atoi(step.pc->location.get_line().c_str()),
      options);

    if (invariant.empty())
      continue; /* we don't have to consider this invariant */

    nodet * new_node = new nodet();
    edget * new_edge = new edget();
    std::string function = step.pc->location.get_function().c_str();
    new_edge->start_line = get_line_number(
      verification_file,
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

  for(const auto & step : goto_trace.steps)
  {
    switch (step.type)
    {
      case goto_trace_stept::ASSERT:
        if(!step.guard)
        {
          show_state_header(out, step, step.pc->location, step.step_nr);
          out << "Violated property:" << std::endl;
          if(!step.pc->location.is_nil())
            out << "  " << step.pc->location << std::endl;
          out << "  " << step.comment << std::endl;

          if(step.pc->is_assert())
            out << "  " << from_expr(ns, "", step.pc->guard) << std::endl;

          // Having printed a property violation, don't print more steps.
          return;
        }
        break;

      case goto_trace_stept::ASSIGNMENT:
        if(step.pc->is_assign() || step.pc->is_return()
            || (step.pc->is_other() && is_nil_expr(step.lhs)))
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
        out << std::endl;
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
