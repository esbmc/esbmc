/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_TEMPLATE_MAP_H
#define CPROVER_CPP_TEMPLATE_MAP_H

#include <map>
#include <iostream>

#include <expr.h>

class template_mapt
{
public:
  typedef std::map<irep_idt, typet> type_mapt;
  typedef std::map<irep_idt, exprt> expr_mapt;
  type_mapt type_map;
  expr_mapt expr_map;

  void apply(exprt &dest) const;
  void apply(typet &dest) const;

  void swap(template_mapt &template_map)
  {
    type_map.swap(template_map.type_map);
    expr_map.swap(template_map.expr_map);
  }

  exprt lookup(const irep_idt &identifier) const;

  void print(std::ostream &out) const;

  void clear()
  {
    type_map.clear();
    expr_map.clear();
  }
};

struct cpp_saved_template_mapt
{
    cpp_saved_template_mapt(template_mapt& map):
        old_map(map),map(map){}
    ~cpp_saved_template_mapt()
    { map.swap(old_map); }

    void restore()
    { map = old_map;}

    private:
    template_mapt old_map;
    template_mapt& map;
};

#endif
