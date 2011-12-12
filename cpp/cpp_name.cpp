/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <assert.h>

#include "cpp_name.h"
#include <sstream>
/*******************************************************************\

Function: cpp_namet::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_namet::convert(
  std::string &identifier,
  std::string &base_name) const
{
  forall_irep(it, get_sub())
  {
    const std::string id=it->id_string();

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
