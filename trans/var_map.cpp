/*******************************************************************\

Module: 

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <solvers/flattening/boolbv_width.h>

#include "var_map.h"

/*******************************************************************\

Function: var_mapt::map_vars

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void var_mapt::map_vars(
  const contextt &context,
  const irep_idt &module)
{
  forall_symbol_module_map(it, context.symbol_module_map,
                           module)
  {
    symbolst::const_iterator s_it=
      context.symbols.find(it->second);

    if(s_it==context.symbols.end())
      continue;

    const symbolt &symbol=s_it->second;

    vart::vartypet vartype;

    if(symbol.theorem)
      continue; // ignore theorems
    else if(symbol.type.id()=="module" ||
            symbol.type.id()=="module_instance")
      continue; // ignore modules
    else if(symbol.is_input)
      vartype=vart::VAR_INPUT;
    else if(symbol.is_statevar)
      vartype=vart::VAR_LATCH;
    else
      vartype=vart::VAR_WIRE;

    unsigned size;
    
    if(boolbv_get_width(symbol.type, size) ||
       size==0)
      continue;

    vart &var=map[symbol.name];
    var.vartype=vartype;
    var.type=symbol.type;
    var.mode=symbol.mode;
    var.bits.resize(size);

    for(unsigned bit_nr=0; bit_nr<size; bit_nr++)
    {
      vart::bitt &bit=var.bits[bit_nr];
      bit.var_no=add_var(symbol.name, bit_nr, vartype);
    }
  }
}

/*******************************************************************\

Function: var_mapt::add

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

unsigned var_mapt::add_var(
  const irep_idt &id,
  unsigned bit_no,
  vart::vartypet vartype)
{
  unsigned v=add_var(id, bit_no);

  switch(vartype)
  {
  case vart::VAR_LATCH:
    latches.insert(v);
    break;
            
  case vart::VAR_INPUT:
    inputs.insert(v);
    break;
    
  case vart::VAR_OUTPUT:
    outputs.insert(v);
    break;
    
  case vart::VAR_WIRE:
    wires.insert(v);
    break;
    
  default:;
    break;
  }
  
  return v;
}

/*******************************************************************\

Function: var_mapt::get

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

unsigned var_mapt::get(
  const irep_idt &id,
  unsigned bit_nr) const
{
  mapt::const_iterator it=map.find(id);

  if(it==map.end())
  {
    std::cerr << "failed to find identifier " 
              << id << "[" << bit_nr << "]" << std::endl;
    assert(false);
  }

  assert(it->second.bits.size()!=0);

  if(bit_nr>=it->second.bits.size())
  {
    std::cerr << "index out of range for " 
              << id << "[" << bit_nr << "]" << std::endl;
    std::cerr << "available range: 0.."
              << it->second.bits.size()-1 << std::endl;
    assert(false);
  }

  const vart::bitt &bit=it->second.bits[bit_nr];

  return bit.var_no;
}

/*******************************************************************\

Function: var_mapt::get_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

var_mapt::vart::vartypet var_mapt::get_type(
  const irep_idt &id) const
{
  mapt::const_iterator it=map.find(id);

  if(it==map.end())
  {
    std::cerr << "failed to find identifier " 
              << id << std::endl;
    assert(false);
  }

  return it->second.vartype;
}

/*******************************************************************\

Function: var_mapt::output

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void var_mapt::output(std::ostream &out) const
{
  out << "Variable map:" << std::endl;

  for(mapt::const_iterator it=map.begin();
      it!=map.end(); it++)
  {
    const vart &var=it->second;

    for(unsigned i=0; i<var.bits.size(); i++)
    {
      out << "  " << it->first
          << "[" << i << "]=" << var.bits[i].var_no
          << " ";

      switch(var.vartype)
      {
       case vart::VAR_INPUT: out << "(input)"; break;
       case vart::VAR_LATCH: out << "(latch)"; break;
       case vart::VAR_WIRE:  out << "(wire)"; break;
       case vart::VAR_OUTPUT:out << "(output)"; break;
       case vart::VAR_UNDEF: out << "(?)"; break;
      }

      out << std::endl;
    }
  }

  out << std::endl
      << "Total no. of variable bits: " << reverse_map.size()
      << std::endl
      << "Total no. of latch bits: " << latches.size()
      << std::endl
      << "Total no. of input bits: " << inputs.size()
      << std::endl;
}
