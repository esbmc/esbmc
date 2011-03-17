/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BOOLBV_MAP_H
#define CPROVER_BOOLBV_MAP_H

#include <vector>

#include <hash_cont.h>
#include <type.h>

#include <solvers/prop/prop.h>

#include "boolbv_type.h"

class boolbv_mapt
{
public:
  boolbv_mapt(propt &_prop):prop(_prop)
  {
  }

  struct map_bitt
  {
    map_bitt():is_set(false) { }
    bool is_set;
    literalt l;
  };

  typedef std::vector<map_bitt> literal_mapt;

  class map_entryt
  {
  public:
    map_entryt():width(0), bvtype(IS_UNKNOWN)
    {
    }

    unsigned width;
    bvtypet bvtype;
    typet type;
    literal_mapt literal_map;
  };
  
  typedef hash_map_cont<irep_idt, map_entryt, irep_id_hash> mappingt;  
  mappingt mapping;

  void show() const;

  map_entryt &get_map_entry(
    const irep_idt &identifier,
    const typet &type);

  literalt get_literal(
    const irep_idt &identifier,
    const unsigned bit,
    const typet &type);

  void set_literal(
    const irep_idt &identifier,
    const unsigned bit,
    const typet &type,
    literalt literal);
    
protected:
  propt &prop;
};

#endif
