/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cassert>
#include <cpp/cpp_name.h>
#include <sstream>

void cpp_namet::convert(
  std::string &identifier,
  std::string &base_name) const
{
  forall_irep(it, get_sub())
  {
    const irep_idt id=it->id();

    std::string name_component;

    if(id=="name")
      name_component=it->get_string("identifier");
    else if(id=="template_args")
    {
      std::stringstream ss;
      ss << location() << std::endl;
      ss << "no template arguments allowed here";
      throw ss.str();
    }
    else
      name_component=it->id_string();

    identifier+=name_component;

    if(id=="::")
      base_name="";
    else
      base_name+=name_component;
  }
}

std::string cpp_namet::to_string() const
{
  std::string str;

  forall_irep(it, get_sub())
  {
    if(it->id()=="::")
      str += it->id_string();
    else if(it->id()=="template_args")
      str += "<...>";
    else
      str+=it->get_string("identifier");
  }

  return str;
}
