/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_SCOPE_H
#define CPROVER_CPP_SCOPE_H

#include <set>

#include "cpp_id.h"

class cpp_scopet:public cpp_idt
{
public:
  cpp_scopet()
  {
    is_scope=true;
  }

  typedef std::set<cpp_idt *> id_sett;

  void lookup(
    const irep_idt &base_name,
    id_sett &id_set);

  void lookup(
    const irep_idt &base_name,
    cpp_idt::id_classt id_class,
    id_sett &id_set);

  void recursive_lookup(
    const irep_idt &base_name,
    id_sett &id_set);

  void recursive_lookup(
    const irep_idt &base_name,
    cpp_idt::id_classt id_class,
    id_sett &id_set);

  void lookup_id(
    const irep_idt &identifier,
    cpp_idt::id_classt id_class,
    id_sett &id_set);

  bool contains(const irep_idt& base_name);

  bool is_root_scope() const
  {
    return id_class==ROOT_SCOPE;
  }

  bool is_global_scope() const
  {
    return id_class==ROOT_SCOPE ||
           id_class==NAMESPACE;
  }

  cpp_scopet &get_parent(unsigned i = 0) const
  {
    return (cpp_scopet &)cpp_idt::get_parent(i);
  }

  class cpp_scopet &new_scope(const irep_idt &new_scope_name);
};

class cpp_root_scopet:public cpp_scopet
{
public:
  cpp_root_scopet()
  {
    id_class=ROOT_SCOPE;
  }
};

#endif
