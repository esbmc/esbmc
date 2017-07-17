/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_typecheck.h>

void cpp_typecheckt::convert(cpp_linkage_spect &linkage_spec)
{
  irep_idt old_mode=current_mode;

  current_mode=linkage_spec.linkage().get("value");

  // there is a linkage spec "C++", which we know as "cpp"
  if(current_mode=="C++")
    current_mode="cpp";

  // do the declarations
  for(auto & it : linkage_spec.items())
    convert(it);

  // back to previous linkage spec
  current_mode=old_mode;
}
