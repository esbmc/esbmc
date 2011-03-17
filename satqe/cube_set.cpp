/*******************************************************************\

Module: Cube Subsumption

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>
#include <iostream>

#include "cube_set.h"

#define ENABLE_SUBSUMPTION

/*******************************************************************\

Function: cube_sett::insert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cube_sett::_insert(const bitvt &stars, const bitvt &bits)
{
  bitssett &star=star_map[stars];

  if(bits.size()==0)
  {
    all=true;
    star.insert(bits);
    return;
  }

  if(star.find(bits)!=star.end())
  {
    //assert(0);
    return;
  }

  #if 0
  std::cout << "SIZE: " << star.size() << std::endl;
  #endif

  bitvt tmp_bits(bits);
  bool subsumed=false;

  #ifdef ENABLE_SUBSUMPTION
  for(unsigned i=0; i<tmp_bits.size(); i++)
  {
    bool old_bit=tmp_bits[i];
    tmp_bits[i]=!old_bit;

    #if 0
    std::cout << "FIND ";
    for(unsigned x=0; x<tmp_bits.size(); x++)
      std::cout << " " << tmp_bits[x];
    std::cout << "\n";
    #endif

    bitssett::iterator it=star.find(tmp_bits);

    if(it!=star.end())
    {
      #if 0
      std::cout << "FOUND\n";
      #endif

      // found it!
      subsumed=1;
      star.erase(it);
      elements--;
      bitvt tmp_stars(stars);

      unsigned count=i+1, j;
      for(j=0; j<tmp_stars.size(); j++)
      {
        if(!tmp_stars[j]) count--;

        if(count==0)
        {
          tmp_stars[j]=1;
          tmp_bits.erase(tmp_bits.begin()+i);

          #if 0
          std::cout << "Subsuming: " << tmp_bits.size() << std::endl;
          #endif

          _insert(tmp_stars, tmp_bits);
          break;
        }
      }

      assert(j<tmp_stars.size());
    }

    tmp_bits[i]=old_bit;
  }
  #endif

  if(!subsumed)
  {
    star.insert(bits);
    elements++;

    #if 0
    std::cout << "STARS: ";
    for(unsigned i=0; i<stars.size(); i++)
      std::cout << " " << stars[i];
    std::cout << "\n";
    #endif

    #if 0
    std::cout << "INSERTING ";
    for(unsigned i=0; i<bits.size(); i++)
      std::cout << " " << bits[i];
    std::cout << "\n";
    #endif
  }
}

