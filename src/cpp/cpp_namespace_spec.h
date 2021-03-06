/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_NAMESPACE_SPEC_H
#define CPROVER_CPP_NAMESPACE_SPEC_H

#include <cpp/cpp_name.h>
#include <util/expr.h>

class cpp_namespace_spect : public exprt
{
public:
  inline cpp_namespace_spect() : exprt("cpp-namespace-spec")
  {
    add("alias").make_nil();
  }

  typedef std::vector<class cpp_itemt> itemst;

  inline const itemst &items() const
  {
    return (const itemst &)operands();
  }

  inline itemst &items()
  {
    return (itemst &)operands();
  }

  inline const irep_idt &get_namespace() const
  {
    return get("namespace");
  }

  inline void set_namespace(const irep_idt &_namespace)
  {
    set("namespace", _namespace);
  }

  inline cpp_namet &alias()
  {
    return static_cast<cpp_namet &>(add("alias"));
  }

  inline const cpp_namet &alias() const
  {
    return static_cast<const cpp_namet &>(find("alias"));
  }

  void output(std::ostream &out) const;

  inline void set_is_inline(bool value)
  {
    set("is_inline", value);
  }

  inline bool get_is_inline() const
  {
    return get_bool("is_inline");
  }
};

#endif
