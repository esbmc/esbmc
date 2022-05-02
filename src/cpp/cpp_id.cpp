/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_id.h>
#include <cpp/cpp_scope.h>

cpp_idt::cpp_idt()
  : is_member(false),
    is_method(false),
    is_static_member(false),
    is_scope(false),
    is_constructor(false),
    id_class(UNKNOWN),
    this_expr(static_cast<const exprt &>(get_nil_irep())),
    compound_counter(0),
    use_parent(false),
    original_scope(nullptr)
{
}

void cpp_idt::print(std::ostream &out, unsigned indent) const
{
  print_fields(out, indent);

  if(!sub.empty())
  {
    for(const auto &it : sub)
      it.second.print(out, indent + 2);

    out << std::endl;
  }
}

void cpp_idt::print_fields(std::ostream &out, unsigned indent) const
{
  for(unsigned i = 0; i < indent; i++)
    out << ' ';
  out << "**identifier=" << identifier << std::endl;

  for(unsigned i = 0; i < indent; i++)
    out << ' ';
  out << "  prefix=" << prefix << std::endl;

  for(unsigned i = 0; i < indent; i++)
    out << ' ';
  out << "  base_name=" << base_name << std::endl;

  for(unsigned i = 0; i < indent; i++)
    out << ' ';
  out << "  method=" << is_method << std::endl;

  if(original_scope != nullptr)
  {
    for(unsigned i = 0; i < indent; i++)
      out << ' ';
    out << "  original_scope=" << original_scope->identifier << std::endl;
  }

  for(unsigned i = 0; i < indent; i++)
    out << ' ';
  out << "  is_static_member=" << is_static_member << std::endl;

  for(unsigned i = 0; i < indent; i++)
    out << ' ';
  out << "  is_scope=" << is_scope << std::endl;

  for(unsigned i = 0; i < indent; i++)
    out << ' ';
  out << "  is_constructor=" << is_constructor << std::endl;

  for(unsigned i = 0; i < indent; i++)
    out << ' ';
  out << "  class_identifier=" << class_identifier << std::endl;

  for(unsigned i = 0; i < indent; i++)
    out << ' ';
  out << "  id_class=" << id_class << std::endl;
}

std::ostream &operator<<(std::ostream &out, const cpp_idt &cpp_id)
{
  cpp_id.print(out, 0);
  return out;
}

std::ostream &operator<<(std::ostream &out, const cpp_idt::id_classt &id_class)
{
  switch(id_class)
  {
  case cpp_idt::UNKNOWN:
    out << "UNKNOWN";
    break;
  case cpp_idt::SYMBOL:
    out << "SYMBOL";
    break;
  case cpp_idt::TYPEDEF:
    out << "TYPEDEF";
    break;
  case cpp_idt::CLASS:
    out << "CLASS";
    break;
  case cpp_idt::TEMPLATE:
    out << "TEMPLATE";
    break;
  case cpp_idt::TEMPLATE_ARGUMENT:
    out << "TEMPLATE_ARGUMENT";
    break;
  case cpp_idt::ROOT_SCOPE:
    out << "ROOT_SCOPE";
    break;
  case cpp_idt::BLOCK_SCOPE:
    out << "BLOCK_SCOPE";
    break;
  case cpp_idt::NAMESPACE:
    out << "NAMESPACE";
    break;

  default:
    out << "(OTHER)";
  }

  return out;
}
