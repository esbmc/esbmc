/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ANSI_C_C_QUALIFIERS_H
#define CPROVER_ANSI_C_C_QUALIFIERS_H

#include <expr.h>

class c_qualifierst
{
public:
  c_qualifierst()
  {
    clear();
  }
  
  c_qualifierst(const typet &src)
  {
    clear();
    read(src);
  }
  
  void clear()
  {
    is_constant=false;
    is_volatile=false;
    is_restricted=false;
  }

  bool is_constant, is_volatile, is_restricted;
  
  std::string as_string() const;
  void read(const typet &src);
  void write(typet &src) const;
  
  static void clear(typet &dest);
  
  bool is_subset_of(const c_qualifierst &q) const
  {
    return (!is_constant || q.is_constant) &&
           (!is_volatile || q.is_volatile) &&
           (!is_restricted || q.is_restricted);
  }
  
  friend bool operator == (
    const c_qualifierst &a,
    const c_qualifierst &b)
  {
    return a.is_constant==b.is_constant &&
           a.is_volatile==b.is_volatile &&
           a.is_restricted==b.is_restricted;
  }

  friend bool operator != (
    const c_qualifierst &a,
    const c_qualifierst &b)
  {
    return !(a==b);
  }
  
  friend unsigned count(const c_qualifierst &q)
  {
    return q.is_constant+q.is_volatile+q.is_restricted;
  }
  
  bool is_empty() const
  {
    return !is_constant && !is_volatile && !is_restricted;
  }
};

#endif
