/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_ID_H
#define CPROVER_CPP_ID_H

#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <string>
#include <util/expr.h>
#include <util/std_types.h>

typedef std::multimap<irep_idt, class cpp_idt> cpp_id_mapt;

class cpp_idt
{
public:
  cpp_idt();

  typedef enum
  {
    UNKNOWN, SYMBOL, TYPEDEF, CLASS, ENUM, TEMPLATE,
    TEMPLATE_ARGUMENT, NAMESPACE, BLOCK_SCOPE,
    TEMPLATE_SCOPE, ROOT_SCOPE
  } id_classt;

  bool is_member, is_method, is_static_member,
       is_scope, is_constructor;

  id_classt id_class;

  inline bool is_class() const
  {
    return id_class==CLASS;
  }

  inline bool is_enum() const
  {
    return id_class==ENUM;
  }

  inline bool is_namespace() const
  {
    return id_class==NAMESPACE;
  }

  inline bool is_typedef() const
  {
    return id_class==TYPEDEF;
  }

  irep_idt identifier, base_name;

  // if it is a member or method, what class is it in?
  irep_idt class_identifier;
  exprt this_expr;

  // scope data
  std::string prefix;
  unsigned compound_counter;
  bool use_parent;

  // the scope this one originally belonged in
  class cpp_scopet *original_scope;

  cpp_idt &insert(const irep_idt &base_name)
  {
    if(use_parent)
    {
      assert(!parents.empty());
      cpp_idt &new_id=get_parent().insert(base_name);
      new_id.original_scope=(cpp_scopet *)(this);
      return new_id;
    }

    cpp_id_mapt::iterator it=
      sub.insert(std::pair<irep_idt, cpp_idt>
        (base_name, cpp_idt()));

    it->second.base_name=base_name;
    it->second.add_parent(*this);
    it->second.original_scope=nullptr;

    return it->second;
  }

  cpp_idt &get_parent(unsigned i=0) const
  {
    assert(i<parents_size());
    assert(parents[i]!=nullptr);
    return *parents[i];
  }

  inline void add_parent(cpp_idt &cpp_id)
  {
    parents.push_back(&cpp_id);
  }

  inline unsigned parents_size() const
  {
    return parents.size();
  }

  inline void clear()
  {
    *this=cpp_idt();
  }

  void print(std::ostream &out = std::cout, unsigned indent=0) const;
  void print_fields(std::ostream &out = std::cout, unsigned indent=0) const;

  friend class cpp_scopet;

public:
  std::set<cpp_idt*> using_set;

protected:
  cpp_id_mapt sub;

private:
  typedef std::vector<cpp_idt *> parentst;
  parentst parents;
};

std::ostream &operator<<(std::ostream &out, const cpp_idt &cpp_id);
std::ostream &operator<<(std::ostream &out, const cpp_idt::id_classt &id_class);

#endif
