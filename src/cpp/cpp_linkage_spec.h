/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_LINKAGE_SPEC_H
#define CPROVER_CPP_LINKAGE_SPEC_H

class cpp_linkage_spect : public exprt
{
public:
  cpp_linkage_spect() : exprt("cpp-linkage-spec")
  {
  }

  typedef std::vector<class cpp_itemt> itemst;

  const itemst &items() const
  {
    return (const itemst &)operands();
  }

  itemst &items()
  {
    return (itemst &)operands();
  }

  irept &linkage()
  {
    return add("linkage");
  }

  const irept &linkage() const
  {
    return find("linkage");
  }
};

#endif
