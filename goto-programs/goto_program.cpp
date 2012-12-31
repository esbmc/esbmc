/*******************************************************************\

Module: Program Transformation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <langapi/language_util.h>

#include "goto_program.h"

/*******************************************************************\

Function: goto_programt::output_instruction

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::ostream& goto_programt::output_instruction(
  const class namespacet &ns,
  const irep_idt &identifier,
  std::ostream& out,
  instructionst::const_iterator it,
  bool show_location,
  bool show_variables) const
{
  if (show_location) {
  out << "        // " << it->location_number << " ";

  if(!it->location.is_nil())
    out << it->location.as_string();
  else
    out << "no location";

  out << "\n";
  }

  if(show_variables && !it->local_variables.empty())
  {
    out << "        // Variables:";
    for(local_variablest::const_iterator
        l_it=it->local_variables.begin();
        l_it!=it->local_variables.end();
        l_it++)
      out << " " << *l_it;

    out << std::endl;
  }

  if(!it->labels.empty())
  {
    out << "        // Labels:";
    for(instructiont::labelst::const_iterator
        l_it=it->labels.begin();
        l_it!=it->labels.end();
        l_it++)
    {
      out << " " << *l_it;
    }

    out << std::endl;
  }

  if(it->is_target())
    out << std::setw(6) << it->target_number << ": ";
  else
    out << "        ";

  switch(it->type)
  {
  case NO_INSTRUCTION_TYPE:
    out << "NO INSTRUCTION TYPE SET" << std::endl;
    break;

  case GOTO:
    if (!is_constant_bool2t(it->guard) ||
        !to_constant_bool2t(it->guard).constant_value)
    {
      out << "IF "
          << from_expr(ns, identifier, it->guard)
          << " THEN ";
    }

    out << "GOTO ";

    for(instructiont::targetst::const_iterator
        gt_it=it->targets.begin();
        gt_it!=it->targets.end();
        gt_it++)
    {
      if(gt_it!=it->targets.begin()) out << ", ";
      out << (*gt_it)->target_number;
    }

    out << std::endl;
    break;

  case RETURN:
  case OTHER:
  case FUNCTION_CALL:
  case ASSIGN:

#if 0
    if(it->code.statement()!="typeid")
    {
#endif
      out << from_expr(ns, identifier, it->code) << std::endl;
#if 0
    }
    else
    {
      // Get the identifier
      out << "  return_value = ";
      out << "typeid(" << it->code.op0().identifier() << ").name() ";
      out << std::endl << std::endl;
    }
#endif
    break;

  case ASSUME:
  case ASSERT:
    if(it->is_assume())
      out << "ASSUME ";
    else
      out << "ASSERT ";

    {
      out << from_expr(ns, identifier, it->guard);

      const irep_idt &comment=it->location.comment();
      if(comment!="") out << " // " << comment;
    }

    out << std::endl;
    break;

  case SKIP:
    out << "SKIP" << std::endl;
    break;

  case END_FUNCTION:
    out << "END_FUNCTION" << std::endl;
    break;

  case LOCATION:
    out << "LOCATION" << std::endl;
    break;

  case THROW:
    out << "THROW";

    {
      const code_cpp_throw2t &throw_ref = to_code_cpp_throw2t(it->code);
      forall_names(it, throw_ref.exception_list) {
      	if(it != throw_ref.exception_list.begin()) out << ",";
        out << " " << *it;
      }

      if (!is_nil_expr(throw_ref.operand))
        out << ": " << from_expr(ns, identifier, throw_ref.operand);
    }

    out << std::endl;
    break;

  case CATCH:
    out << "CATCH ";

    {
      unsigned i=0;
      const code_cpp_catch2t &catch_ref = to_code_cpp_catch2t(it->code);
      assert(it->targets.size() == catch_ref.exception_list.size());

      for(instructiont::targetst::const_iterator
          gt_it=it->targets.begin();
          gt_it!=it->targets.end();
          gt_it++,
          i++)
      {
        if(gt_it!=it->targets.begin()) out << ", ";
        out << catch_ref.exception_list[i] << "->"
            << (*gt_it)->target_number;
      }
    }

    out << std::endl;
    break;

  case ATOMIC_BEGIN:
    out << "ATOMIC_BEGIN" << std::endl;
    break;

  case ATOMIC_END:
    out << "ATOMIC_END" << std::endl;
    break;

  case THROW_DECL:
    out << "THROW_DECL (";

    {
      const code_cpp_throw_decl2t &ref = to_code_cpp_throw_decl2t(it->code);

      for(unsigned int i=0; i < ref.exception_list.size(); ++i)
      {
        if(i) out << ", ";
        out << ref.exception_list[i];
      }
      out << ")";
    }

    out << std::endl;
    break;

  default:
    throw "unknown statement";
  }

  return out;
}

/*******************************************************************\

Function: operator<

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool operator<(const goto_programt::const_targett i1,
               const goto_programt::const_targett i2)
{
  const goto_programt::instructiont &_i1=*i1;
  const goto_programt::instructiont &_i2=*i2;
  return &_i1<&_i2;
}
