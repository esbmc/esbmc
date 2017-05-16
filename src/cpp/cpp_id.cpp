/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_id.h>
#include <cpp/cpp_scope.h>

/*******************************************************************\

Function: cpp_idt::cpp_idt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

cpp_idt::cpp_idt():
  is_member(false),
  is_method(false),
  is_static_member(false),
  is_scope(false),
  is_constructor(false),
  id_class(UNKNOWN),
  this_expr(static_cast<const exprt &>(get_nil_irep())),
  compound_counter(0),
  use_parent(false),
  original_scope(NULL)
{
}

/*******************************************************************\

Function: cpp_idt::print

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_idt::print(std::ostream &out, unsigned indent) const
{
  print_fields(out, indent);

  if(!sub.empty())
  {
    for(cpp_id_mapt::const_iterator it=sub.begin();
        it!=sub.end();
        it++)
      it->second.print(out, indent+2);

    out << std::endl;
  }
}

/*******************************************************************\

Function: cpp_idt::print_fields

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_idt::print_fields(std::ostream &out, unsigned indent) const
{
  for(unsigned i=0; i<indent; i++) out << ' ';
  out << "**identifier=" << identifier << std::endl;

  for(unsigned i=0; i<indent; i++) out << ' ';
  out << "  prefix=" << prefix << std::endl;

  for(unsigned i=0; i<indent; i++) out << ' ';
  out << "  base_name=" << base_name << std::endl;

  for(unsigned i=0; i<indent; i++) out << ' ';
  out << "  method=" << is_method << std::endl;

  if(original_scope!=NULL)
  {
    for(unsigned i=0; i<indent; i++) out << ' ';
    out << "  original_scope=" << original_scope->identifier << std::endl;
  }

  for(unsigned i=0; i<indent; i++) out << ' ';
  out << "  is_static_member=" << is_static_member << std::endl;

  for(unsigned i=0; i<indent; i++) out << ' ';
  out << "  is_scope=" << is_scope << std::endl;

  for(unsigned i=0; i<indent; i++) out << ' ';
  out << "  is_constructor=" << is_constructor << std::endl;

  for(unsigned i=0; i<indent; i++) out << ' ';
  out << "  class_identifier=" << class_identifier << std::endl;

  for(unsigned i=0; i<indent; i++) out << ' ';
  out << "  id_class=" << id_class << std::endl;
}

/*******************************************************************\

Function: operator<<

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::ostream &operator<<(std::ostream &out, const cpp_idt &cpp_id)
{
  cpp_id.print(out, 0);
  return out;
}

/*******************************************************************\

Function: operator<<

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::ostream &operator<<(std::ostream &out, const cpp_idt::id_classt &id_class)
{
  switch(id_class)
  {
   case cpp_idt::UNKNOWN:           out << "UNKNOWN"; break;
   case cpp_idt::SYMBOL:            out << "SYMBOL"; break;
   case cpp_idt::TYPEDEF:           out << "TYPEDEF"; break;
   case cpp_idt::CLASS:             out << "CLASS"; break;
   case cpp_idt::TEMPLATE:          out << "TEMPLATE"; break;
   case cpp_idt::TEMPLATE_ARGUMENT: out << "TEMPLATE_ARGUMENT"; break;
   case cpp_idt::ROOT_SCOPE:        out << "ROOT_SCOPE"; break;
   case cpp_idt::BLOCK_SCOPE:       out << "BLOCK_SCOPE"; break;
   case cpp_idt::NAMESPACE:         out << "NAMESPACE"; break;

   default:
    out << "(OTHER)";
  }

  return out;
}
