/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "boolbv_map.h"
#include "boolbv_width.h"

//#define DEBUG

/*******************************************************************\

Function: boolbv_mapt::get_map_entry

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

boolbv_mapt::map_entryt &boolbv_mapt::get_map_entry(
  const irep_idt &identifier,
  const typet &type)
{
  std::pair<mappingt::iterator, bool> result=
    mapping.insert(std::pair<irep_idt, map_entryt>(
      identifier, map_entryt()));

  map_entryt &map_entry=result.first->second;

  if(result.second)
  { // actually inserted
    map_entry.type=type;

    if(boolbv_get_width(type, map_entry.width))
      throw "failed to get size";
      
    map_entry.bvtype=get_bvtype(type);
    map_entry.literal_map.resize(map_entry.width);
  }

  assert(map_entry.literal_map.size()==map_entry.width);

  return map_entry;
}

/*******************************************************************\

Function: boolbv_mapt::show

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbv_mapt::show() const
{
  for(mappingt::const_iterator it=mapping.begin();
      it!=mapping.end();
      it++)
  {
  }
}

/*******************************************************************\

Function: boolbv_mapt::get_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt boolbv_mapt::get_literal(
  const irep_idt &identifier,
  const unsigned bit,
  const typet &type)
{
  map_entryt &map_entry=get_map_entry(identifier, type);

  assert(bit<map_entry.literal_map.size());

  if(map_entry.literal_map[bit].is_set)
    return map_entry.literal_map[bit].l;

  literalt l=prop.new_variable();

  map_entry.literal_map[bit].is_set=true;
  map_entry.literal_map[bit].l=l;

  #ifdef DEBUG
  std::cout << "NEW: " << identifier << ":" << bit
            << "=" << l << std::endl;
  #endif

  return l;
}

/*******************************************************************\

Function: boolbv_mapt::set_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void boolbv_mapt::set_literal(
  const irep_idt &identifier,
  const unsigned bit,
  const typet &type,
  literalt literal)
{
  assert(literal.is_constant() ||
         literal.var_no()<prop.no_variables());

  map_entryt &map_entry=get_map_entry(identifier, type);
  assert(bit<map_entry.literal_map.size());

  if(map_entry.literal_map[bit].is_set)
  {
    prop.set_equal(map_entry.literal_map[bit].l, literal);
    return;
  }

  map_entry.literal_map[bit].is_set=true;
  map_entry.literal_map[bit].l=literal;
}

