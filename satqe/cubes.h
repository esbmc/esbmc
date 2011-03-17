/*******************************************************************\

Module: Cubes

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef __CUBES_H
#define __CUBES_H

#include <vector>
#include <set>
#include <iostream>
#include <map>

#include <solvers/prop/literal.h>

typedef std::vector<literalt> bvt;

class cubest
{
 public:
  cubest()
  {
  }
 
  typedef std::vector<bool> bitvt;
  typedef std::set<bitvt> bitssett;
  
  typedef std::map<bitvt, bitssett> star_mapt;
  star_mapt star_map;
  
  void make_vector(
    const std::vector<unsigned> &map,
    const bitvt &stars,
    const bitvt &bits,
    bvt &dest);

  void make_vector(
    const bitvt &stars,
    const bitvt &bits, std::set<unsigned> &dest);
                   
  void swap(cubest &cubes)
  {
    star_map.swap(cubes.star_map);
  }
   
  void clear()
  {
    star_map.clear();
  }
   
  bool empty() const
  {
    return star_map.empty();
  }
   
  bool all_stars(const bitvt &starmap) const
  {
    for(unsigned i=0; i<starmap.size(); i++)
      if(!starmap[i]) return false;

    return true;
  }
   
  friend std::ostream &operator<<(std::ostream &out, const cubest &cubes);
};

#endif
