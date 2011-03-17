/*******************************************************************\

Module: Cubes

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>
#include "cubes.h"

/*******************************************************************\

Function: cubest::make_vector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cubest::make_vector(
  const std::vector<unsigned> &map,
  const bitvt &stars,
  const bitvt &bits,
  bvt &dest)
{
  dest.resize(bits.size());

  unsigned bit=0;

  for(unsigned i=0; i<stars.size(); i++)
    if(!stars[i])
    {
      assert(bit<bits.size());
      dest[bit].set(map[i], bits[bit]);
      bit++;
    }

  assert(bit==bits.size());
}

/*******************************************************************\

Function: cubest::make_vector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cubest::make_vector(
  const bitvt &stars,
  const bitvt &bits,
  std::set<unsigned> &dest)
{
  for(unsigned i=0; i<stars.size(); i++)
    if(!stars[i])
      dest.insert(i);
}

/*******************************************************************\

Function: operator <<

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::ostream &operator<<(std::ostream &out, const cubest &cubes)
{
  for(cubest::star_mapt::const_iterator it=cubes.star_map.begin();
      it!=cubes.star_map.end(); it++)
  {
    for(cubest::bitssett::const_iterator it2=it->second.begin();
        it2!=it->second.end(); it2++)
    {
      unsigned bit=0;

      for(unsigned i=0; i<it->first.size(); i++)
      {
        if(it->first[i])
          out << "*";
        else
        {
          out << ((*it2)[bit]?'1':'0');
          bit++;
        }
      }

      out << std::endl;
    }
  }
  
  return out;
}
