/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_USING_H
#define CPROVER_CPP_USING_H

#include "cpp_name.h"

class cpp_usingt:public irept
{
public:
  cpp_usingt():irept("cpp-using")
  {
  }
  
  cpp_namet &name()
  {
    return (cpp_namet &)add("name");
  }

  const cpp_namet &name() const 
  {
    return (cpp_namet &)find("name");
  }
};

#endif
