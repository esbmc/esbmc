/*******************************************************************\

Module: Cube Subsumption

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef __CUBE_SET_H
#define __CUBE_SET_H

#include "cubes.h"

class cube_sett:public cubest
{
 public:
  cube_sett():elements(0), all(false), _no_insertions(0)
  {
  }
 
  void insert(const bitvt &stars, const bitvt &bits)
  {
    _no_insertions++;
    _insert(stars, bits);
  }
   
  void _insert(const bitvt &stars, const bitvt &bits);

  unsigned size() { return elements; }
  
  unsigned no_insertions() { return _no_insertions; }
  
  void clear()
  {
    star_map.clear();
    elements=0;
    all=false;
    _no_insertions=0;
  }
   
  void swap(cube_sett &b)
  {
    star_map.swap(b.star_map);
    std::swap(elements, b.elements);
    std::swap(all, b.all);
  }
   
  bool is_all() const { return all; }

 protected:
  unsigned elements;
  bool all;
  unsigned _no_insertions;
};

#endif
