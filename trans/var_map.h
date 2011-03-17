/*******************************************************************\

Module: Variable Mapping

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_TRANS_VAR_MAP_H
#define CPROVER_TRANS_VAR_MAP_H

#include <map>
#include <set>
#include <vector>
#include <string>

#include <context.h>
#include <solvers/prop/prop.h>

#include "bv_varid.h"

class var_mapt
{
public:
  struct vart
  {
    typedef enum { VAR_UNDEF, VAR_LATCH, VAR_INPUT,
                   VAR_OUTPUT, VAR_WIRE } vartypet;
    vartypet vartype;
    typet type;
    irep_idt mode;
    
    bool is_latch() const { return vartype==VAR_LATCH; }
    bool is_input() const { return vartype==VAR_INPUT; }
    bool is_wire() const { return vartype==VAR_WIRE; }
    
    struct bitt
    {
      unsigned var_no;
    };
     
    typedef std::vector<bitt> bitst;
    bitst bits;
    
    vart():vartype(VAR_UNDEF)
    {
    }
  };
  
  vart::vartypet get_type(const irep_idt &id) const;

  typedef hash_map_cont<irep_idt, vart, irep_id_hash> mapt;
  mapt map;
  
  typedef std::vector<bv_varidt> reverse_mapt;
  reverse_mapt reverse_map;

  void output(std::ostream &out) const;
  
  void map_vars(
    const contextt &context,
    const irep_idt &module);

  unsigned get(const irep_idt &id, unsigned bit_nr) const;

  unsigned get(const bv_varidt &varid) const
  {
    return get(varid.id, varid.bit_nr);
  }
  
  typedef std::set<unsigned> var_sett;

  var_sett latches, inputs, outputs, wires;
  
  var_mapt()
  {
  }
  
  var_mapt(const contextt &context, const irep_idt &module)
  {
    map_vars(context, module);
  }
  
  unsigned get_no_vars() const
  {
    return reverse_map.size();
  }
  
  unsigned add_var(const irep_idt &id, unsigned bit_no)
  {
    unsigned s=reverse_map.size();
    reverse_map.push_back(bv_varidt(id, bit_no));
    return s;
  }
  
  unsigned add_var(
    const irep_idt &id,
    unsigned bit_no,
    vart::vartypet vartype);

  void swap(var_mapt &other)
  {
    std::swap(other.reverse_map, reverse_map);
    other.latches.swap(latches);
    other.inputs.swap(inputs);
    other.outputs.swap(outputs);
    other.wires.swap(wires);
    other.map.swap(map);
  }
  
  void clear()
  {
    reverse_map.clear();
    latches.clear();
    inputs.clear();
    outputs.clear();
    wires.clear();
    map.clear();
  }
};
 
#endif
