/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_USING_H
#define CPROVER_CPP_USING_H

#include <cpp/cpp_name.h>

class cpp_usingt : public irept
{
public:
  cpp_usingt() : irept("cpp-using")
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

  bool get_namespace() const
  {
    return get_bool("namespace");
  }

  void set_namespace(bool value)
  {
    set("namespace", value);
  }
};

#endif
