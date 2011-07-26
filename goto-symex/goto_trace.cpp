/*******************************************************************\

Module: Traces of GOTO Programs

Author: Daniel Kroening

  Date: July 2005

\*******************************************************************/

#include <assert.h>

#include <ansi-c/printf_formatter.h>
#include <langapi/language_util.h>
#include <arith_tools.h>

#include "goto_trace.h"

/*******************************************************************\

Function: goto_tracet::output

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_tracet::output(
  const class namespacet &ns,
  std::ostream &out) const
{
  for(stepst::const_iterator it=steps.begin();
      it!=steps.end();
      it++)
    it->output(ns, out);
}

/*******************************************************************\

Function: goto_tracet::output

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_trace_stept::output(
  const namespacet &ns,
  std::ostream &out) const
{
  out << "*** ";

  switch(type)
  {
  case goto_trace_stept::ASSERT: out << "ASSERT"; break;
  case goto_trace_stept::ASSUME: out << "ASSUME"; break;
  case goto_trace_stept::LOCATION: out << "LOCATION"; break;
  case goto_trace_stept::ASSIGNMENT: out << "ASSIGNMENT"; break;
  default: assert(false);
  }

  if(type==ASSERT || type==ASSUME)
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
  else if(pc->is_sync())
    out << "SYNC   ";
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

    if(original_lhs.is_not_nil())
      identifier=original_lhs.identifier();
    else
      identifier=lhs.identifier();

    out << "  " << identifier
        << " = " << from_expr(ns, identifier, value)
        << std::endl;
  }
  else if(pc->is_assert())
  {
    if(!guard)
    {
      out << "Violated property:" << std::endl;
      if(pc->location.is_nil())
        out << "  " << pc->location << std::endl;

      if(comment!="")
        out << "  " << comment << std::endl;
      out << "  " << from_expr(ns, "", pc->guard) << std::endl;
      out << std::endl;
    }
  }

  out << std::endl;
}

/*******************************************************************\

Function: counterexample_value

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void counterexample_value(
  std::ostream &out,
  const namespacet &ns,
  const exprt &lhs,
  const exprt &value,
  const pretty_namest &pretty_names)
{
  const irep_idt &identifier=lhs.identifier();
  std::string value_string;

  if(value.is_nil())
    value_string="(assignment removed)";
  else
  {
    value_string=from_expr(ns, identifier, value);
    if(value.is_constant())
    {
      if(value.type().id()==typet::t_signedbv ||
	 value.type().id()==typet::t_unsignedbv ||
    	 value.type().id()==typet::t_fixedbv ||
    	 value.type().id()==typet::t_floatbv)
        value_string+= " ("+value.get_string(exprt::a_value)+")";
    }
#if 0
    else if (lhs.type().id()==typet::t_pointer)
    {
      size_t found;
      found=value_string.find_first_of("@");
      if (found!=std::string::npos)
        value_string.erase(found,value_string.size());
      value_string="&"+value_string;
    }
#endif
  }

  #if 1
  std::string name=id2string(identifier);

  const symbolt *symbol;
  if(!ns.lookup(identifier, symbol))
    if(symbol->pretty_name!="")
      name=id2string(symbol->pretty_name);

  #else
  std::string name=pretty_names.pretty_name(identifier)
  #endif

  out << "  " << name << "=" << value_string
      << std::endl;
}

/*******************************************************************\

Function: show_goto_trace_gui

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void show_goto_trace_gui(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  locationt previous_location;

  for(goto_tracet::stepst::const_iterator
      it=goto_trace.steps.begin();
      it!=goto_trace.steps.end();
      it++)
  {
    const locationt &location=it->pc->location;

    if(it->type==goto_trace_stept::ASSERT &&
       !it->guard)
    {
      out << "FAILED" << std::endl
          << it->comment << std::endl // value
          << std::endl // PC
          << location.get("file") << std::endl
          << location.get("line") << std::endl
          << location.get("column") << std::endl;
    }
    else if(it->type==goto_trace_stept::ASSIGNMENT)
    {
      irep_idt identifier;

      if(it->original_lhs.is_not_nil())
        identifier=it->original_lhs.identifier();
      else
        identifier=it->lhs.identifier();

      std::string value_string=from_expr(ns, identifier, it->value);

      const symbolt *symbol;
      irep_idt base_name;
      if(!ns.lookup(identifier, symbol))
        base_name=symbol->base_name;

      out << "TRACE" << std::endl;

      out << identifier << ","
          << base_name << ","
          << it->value.type().to_string() << ","
          << value_string << std::endl
          << it->step_nr << std::endl
          << it->pc->location.get("file") << std::endl
          << it->pc->location.get("line") << std::endl
          << it->pc->location.get("column") << std::endl;
    }
    else if(location!=previous_location)
    {
      // just the location

      if(location.get("file")!="")
      {
        out << "TRACE" << std::endl;

        out << ","             // identifier
            << ","             // base_name
            << ","             // type
            << "" << std::endl // value
            << it->step_nr << std::endl
            << location.get("file") << std::endl
            << location.get("line") << std::endl
            << location.get("column") << std::endl;
      }
    }

    previous_location=location;
  }
}

/*******************************************************************\

Function: show_state_header

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void show_state_header(
  std::ostream &out,
  const goto_trace_stept &state,
  const locationt &location,
  unsigned step_nr)
{
  out << std::endl;

  if(step_nr==0)
    out << "Initial State";
  else
    out << "State " << step_nr;

  out << " " << location
      << " thread " << state.thread_nr << std::endl;
  out << "----------------------------------------------------" << std::endl;
}

/*******************************************************************\

Function: show_goto_trace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void show_goto_trace(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace)
{
  pretty_namest pretty_names;

  {
    pretty_namest::symbolst pretty_symbols;

    forall_symbols(it, ns.get_context().symbols)
      pretty_symbols.insert(it->first);

    pretty_names.get_pretty_names(pretty_symbols, ns);
  }

  show_goto_trace(out, ns, pretty_names, goto_trace);
}

/*******************************************************************\

Function: show_goto_trace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void show_goto_trace(
  std::ostream &out,
  const namespacet &ns,
  const pretty_namest &pretty_names,
  const goto_tracet &goto_trace)
{
  unsigned prev_step_nr=0;
  bool first_step=true;

  for(goto_tracet::stepst::const_iterator
      it=goto_trace.steps.begin();
      it!=goto_trace.steps.end();
      it++)
  {
    switch(it->type)
    {
    case goto_trace_stept::ASSERT:
      if(!it->guard)
      {
        out << std::endl;
        out << "Violated property:" << std::endl;
        if(!it->pc->location.is_nil())
          out << "  " << it->pc->location << std::endl;
        out << "  " << it->comment << std::endl;

        if(it->pc->is_assert())
          out << "  " << from_expr(ns, "", it->pc->guard) << std::endl;

        out << std::endl;
      }
      break;

    case goto_trace_stept::ASSUME:
      break;

    case goto_trace_stept::LOCATION:
      if(it->pc->is_sync())
      {
        show_state_header(out, *it, it->pc->location, it->step_nr);

        out << "SYNC " << it->pc->event << std::endl;
      }
      break;

    case goto_trace_stept::ASSIGNMENT:
      if(it->pc->is_assign() ||
         (it->pc->is_other() && it->lhs.is_not_nil()))
      {
        if(prev_step_nr!=it->step_nr || first_step)
        {
          first_step=false;
          prev_step_nr=it->step_nr;
          show_state_header(out, *it, it->pc->location, it->step_nr);
        }
        counterexample_value(out, ns, it->original_lhs,
                             it->value, pretty_names);
      }
      break;

    case goto_trace_stept::OUTPUT:
      {
        printf_formattert printf_formatter;
        printf_formatter(it->format_string, it->output_args);
        printf_formatter.print(out);
        out << std::endl;
      }
      break;

    default:
      assert(false);
    }
  }
}
