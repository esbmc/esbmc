/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_item.h>
#include <cpp/cpp_namespace_spec.h>

/*******************************************************************\

Function: cpp_namespace_spect::output

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_namespace_spect::output(std::ostream &out) const
{
  out << "  namespace: " << get_namespace() << std::endl;
}
