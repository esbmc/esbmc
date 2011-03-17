/*******************************************************************\

Module: Transition System represented by a Netlist

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <namespace.h>
#include <arith_tools.h>

#include <solvers/prop/aig_prop.h>
#include <solvers/flattening/boolbv_width.h>

#include "netlist_trans.h"
#include "get_trans.h"
#include "instantiate.h"

/*******************************************************************\

Function: netlist_transt::print

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void netlist_transt::print(std::ostream &out) const
{
  netlistt::print(out);

  out << std::endl;

  out << "Initial state: ";
  print(out, initial);
  out << std::endl;

  out << "Transition constraint: ";
  print(out, transition);
  out << std::endl;

  for(unsigned i=0; i<properties.size(); i++)
  {
    out << "Property " << (i+1) << ": ";
    print(out, properties[i]);
    out << std::endl;
  }
}

/*******************************************************************\

Function: netlist_transt::output_dot

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void netlist_transt::output_dot(std::ostream &out) const
{
  netlistt::output_dot(out);
}

/*******************************************************************\

   Class: convert_trans_to_netlistt

 Purpose:

\*******************************************************************/

class convert_trans_to_netlistt
{
public:
  convert_trans_to_netlistt(
    const namespacet &_ns,
    messaget &_message):
    ns(_ns),
    message(_message)
  {
  }

  void operator()(
    const transt &_trans,
    const std::list<exprt> &properties,
    netlist_transt &dest);
  
protected:
  const namespacet &ns;
  messaget &message;
  bmc_mapt bmc_map;

  class rhs_entryt
  {
  public:
    bool converted;
    exprt expr;
    bvt bv;
    unsigned width;
    
    rhs_entryt():converted(false)
    {
    }

    rhs_entryt(const exprt &_expr):converted(false), expr(_expr)
    {
    }
  };

  typedef std::list<rhs_entryt> rhs_listt;
  rhs_listt rhs_list;
  
  typedef std::list<exprt> constraint_listt;
  constraint_listt constraint_list;
  bvt transition_constraints;
  
  class rhst
  {
  public:
    rhs_entryt *entry;
    unsigned bit_number;
    
    rhst():entry(0)
    {
    }

    rhst(rhs_entryt &_entry, unsigned _nr):entry(&_entry), bit_number(_nr)
    {
    }
  };
  
  class lhs_entryt
  {
  public:
    std::list<rhst> equal_to;
    bool converted, is_latch, in_progress;
    literalt l;
    
    lhs_entryt():converted(false), is_latch(false), in_progress(false)
    {
    }
  };

  // index is variable number
  typedef std::vector<lhs_entryt> lhs_mapt;
  lhs_mapt lhs_map;

  void add_constraint(const exprt &src);
  void add_equality(const equality_exprt &src);

  void add_equality_rec(
    const equality_exprt &src,
    const exprt &lhs,
    unsigned lhs_from, unsigned lhs_to,
    rhs_entryt &rhs_entry);

  literalt convert_rhs(const rhst &rhs, propt &prop);

  void convert_lhs(unsigned var_no, propt &prop);    

  void convert_lhs_rec(
    const exprt &expr,
    unsigned from,
    unsigned to,
    propt &prop);

  literalt convert_constraints(propt &prop);
};

/*******************************************************************\

Function: convert_trans_to_netlistt::operator()

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_trans_to_netlistt::operator()(
  const transt &_trans,
  const std::list<exprt> &properties,
  netlist_transt &dest)
{
  // setup
  bmc_map.clear();
  bmc_map.var_map=dest.var_map;

  lhs_map.clear();
  rhs_list.clear();
  constraint_list.clear();
  
  lhs_map.resize(bmc_map.var_map.get_no_vars());

  for(var_mapt::var_sett::const_iterator
      it=bmc_map.var_map.latches.begin();
      it!=bmc_map.var_map.latches.end();
      it++)
    lhs_map[*it].is_latch=true;

  // extract constraints from transition relation
  add_constraint(_trans.invar());
  add_constraint(_trans.trans());

  // build the net-list
  aig_propt aig_prop(dest);
  bmc_map.map_timeframes(aig_prop, 2);

  // do recursive conversion for LHSs
  for(unsigned v=0; v<lhs_map.size(); v++)
    convert_lhs(v, aig_prop);

  // do the remaining transition constraints
  dest.transition=convert_constraints(aig_prop);
  
  // initial state
  dest.initial=instantiate_convert(
    aig_prop, bmc_map, _trans.init(), 0, 1, ns, message);

  // properties
  dest.properties.reserve(properties.size());

  for(std::list<exprt>::const_iterator
      it=properties.begin();
      it!=properties.end();
      it++)
  {
    exprt property(*it);
    literalt l;

    if(property.is_true())
      l=const_literal(true);
    else if(property.is_false())
      l=const_literal(false);
    else
    {
      if(property.id()!="AG" ||
         property.operands().size()!=1)
      {
        message.error("unsupported property - only AGp implemented");
        throw 0;
      }

      const exprt &p=property.op0();

      l=instantiate_convert(
        aig_prop, bmc_map, p, 0, 1, ns, message);
    }

    dest.properties.push_back(l);
  }

  // remember the node numbers
  for(unsigned v=0; v<dest.var_map.get_no_vars(); v++)
  {
    dest.set_current(v, bmc_map.timeframe_map[0][v]);
    dest.set_next(v, bmc_map.timeframe_map[1][v]);
  }  
}

/*******************************************************************\

Function: convert_trans_to_netlistt::convert_constraints

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt convert_trans_to_netlistt::convert_constraints(propt &prop)
{
  transition_constraints.reserve(
    transition_constraints.size()+constraint_list.size());

  for(constraint_listt::const_iterator
      it=constraint_list.begin();
      it!=constraint_list.end();
      it++)
  {
    literalt l=
      instantiate_convert(prop, bmc_map, *it, 0, 1, ns, message);
    transition_constraints.push_back(l);
  }

  if(transition_constraints.empty())
    return const_literal(true);

  return prop.land(transition_constraints);
}

/*******************************************************************\

Function: convert_trans_to_netlistt::convert_lhs

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_trans_to_netlistt::convert_lhs(
  unsigned var_no,
  propt &prop)
{
  assert(var_no<lhs_map.size());

  lhs_entryt &lhs=lhs_map[var_no];

  if(lhs.converted) return;
  
  assert(bmc_map.timeframe_map.size()==2);
  
  lhs.l=bmc_map.get(lhs.is_latch?1:0, var_no);
  
  if(lhs.in_progress) // cycle found?
    return;

  if(lhs.equal_to.empty()) // no def.?
    return;
    
  // do first one
  
  lhs.in_progress=true;
  lhs.l=convert_rhs(lhs.equal_to.front(), prop);
  bmc_map.set(lhs.is_latch?1:0, var_no, lhs.l);

  lhs.converted=true;
  lhs.in_progress=false;

  // do any additional constraints

  for(std::list<rhst>::const_iterator
      it=lhs.equal_to.begin();
      it!=lhs.equal_to.end();
      it++)
  {
    // first one? -- already done
    if(it==lhs.equal_to.begin()) continue;
    
    literalt l_rhs=convert_rhs(*it, prop);
    transition_constraints.push_back(
      prop.lequal(lhs.l, l_rhs));
  }
}

/*******************************************************************\

Function: convert_trans_to_netlistt::convert_lhs_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_trans_to_netlistt::convert_lhs_rec(
  const exprt &expr,
  unsigned from, unsigned to,
  propt &prop)
{
  assert(from<=to);

  if(expr.id()=="symbol")
  { 
    const irep_idt &identifier=expr.get("identifier");

    var_mapt::mapt::const_iterator it=
      bmc_map.var_map.map.find(identifier);

    if(it==bmc_map.var_map.map.end())
      throw "failed to find `"+id2string(identifier)+"' in var_map";
      
    assert(bmc_map.timeframe_map.size()==2);

    const var_mapt::vart &var=it->second;

    if(!var.is_wire()) return;
    
    for(unsigned bit_nr=from; bit_nr<=to; bit_nr++)
    {
      assert(bit_nr<it->second.bits.size());
      unsigned var_no=it->second.bits[bit_nr].var_no;
      assert(var_no<bmc_map.timeframe_map[1].size());
      convert_lhs(var_no, prop);
    }

    return;
  }
  else if(expr.id()=="extractbit")
  {
    assert(expr.operands().size()==2);

    mp_integer i;
    if(!to_integer(expr.op1(), i)) // constant?
    {
      from=integer2long(i);
      convert_lhs_rec(expr.op0(), from, from, prop);
      return;
    }
  }
  else if(expr.id()=="extractbits")
  {
    mp_integer new_from, new_to;

    assert(expr.operands().size()==3);

    if(!to_integer(expr.op1(), new_from) &&
       !to_integer(expr.op2(), new_to))
    {
      if(new_from>new_to) std::swap(new_from, new_to);
    
      assert(new_from<=new_to);
    
      from=integer2long(new_from);
      to=integer2long(new_to);
    
      convert_lhs_rec(expr.op0(), from, to, prop);
      return;
    }
  }

  // default
  forall_operands(it, expr)
  {
    unsigned width;
    if(boolbv_get_width(it->type(), width))
      continue;

    convert_lhs_rec(*it, 0, width-1, prop);
  }
}

/*******************************************************************\

Function: convert_trans_to_netlistt::convert_rhs

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

literalt convert_trans_to_netlistt::convert_rhs(
  const rhst &rhs,
  propt &prop)
{
  rhs_entryt &rhs_entry=*rhs.entry;
  
  // done already?
  if(!rhs_entry.converted)
  {
    // get all lhs symbols this depends on
    convert_lhs_rec(rhs_entry.expr, 0, rhs_entry.width-1, prop);

    rhs_entry.converted=true;

    // now we can convert
    instantiate_convert(
      prop, bmc_map, rhs_entry.expr, 0, 1, ns, message, rhs_entry.bv);
      
    assert(rhs_entry.bv.size()==rhs_entry.width);
  }

  assert(rhs.bit_number<rhs_entry.bv.size());
  return rhs_entry.bv[rhs.bit_number];
}

/*******************************************************************\

Function: convert_trans_to_netlistt::add_equality

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_trans_to_netlistt::add_equality(const equality_exprt &src)
{
  const exprt &lhs=src.lhs();
  const exprt &rhs=src.rhs();

  rhs_list.push_back(rhs_entryt(rhs));
  rhs_entryt &rhs_entry=rhs_list.back();
  
  if(boolbv_get_width(rhs.type(), rhs_entry.width))
  {
    constraint_list.push_back(src);
    return;
  }
  
  assert(rhs_entry.width!=0);

  unsigned lhs_width;

  if(boolbv_get_width(lhs.type(), lhs_width))
    assert(false);

  assert(lhs_width==rhs_entry.width);

  add_equality_rec(src, lhs, 0, lhs_width-1, rhs_entry);
}

/*******************************************************************\

Function: convert_trans_to_netlistt::add_equality_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_trans_to_netlistt::add_equality_rec(
  const equality_exprt &src,
  const exprt &lhs,
  unsigned lhs_from, unsigned lhs_to,
  rhs_entryt &rhs_entry)
{
  assert(lhs_from<=lhs_to);
  
  if(lhs.id()=="next_symbol" ||
     lhs.id()=="symbol")
  { 
    bool next=lhs.id()=="next_symbol";
   
    const irep_idt &identifier=lhs.get("identifier");

    var_mapt::mapt::const_iterator it=
      bmc_map.var_map.map.find(identifier);

    if(it==bmc_map.var_map.map.end())
      throw "failed to find `"+id2string(identifier)+"' in var_map";
      
    const var_mapt::vart &var=it->second;
    
    if((next && !var.is_latch()) ||
       (!next && !var.is_wire()))
    {
      // give up
      constraint_list.push_back(src);
      return;
    }

    unsigned width=lhs_to-lhs_from+1;
    
    for(unsigned bit_nr=0; bit_nr<width; bit_nr++)
    {
      unsigned lhs_bit_nr=bit_nr+lhs_from;
      assert(lhs_bit_nr<it->second.bits.size());
      unsigned var_no=it->second.bits[lhs_bit_nr].var_no;
      assert(var_no<bmc_map.var_map.get_no_vars());
      
      lhs_entryt &lhs_entry=lhs_map[var_no];
      lhs_entry.equal_to.push_back(rhst(rhs_entry, bit_nr));
    }
  }
  else if(lhs.id()=="extractbit")
  {
    assert(lhs.operands().size()==2);
    assert(lhs_to==lhs_from);

    mp_integer i;
    if(to_integer(lhs.op1(), i))
      assert(false);

    lhs_from=lhs_from+integer2long(i);
    add_equality_rec(src, lhs.op0(), lhs_from, lhs_from, rhs_entry);
  }
  else if(lhs.id()=="extractbits")
  {
    mp_integer new_lhs_from, new_lhs_to;

    assert(lhs.operands().size()==3);

    if(to_integer(lhs.op1(), new_lhs_from))
      assert(false);
    
    if(to_integer(lhs.op2(), new_lhs_to))
      assert(false);
    
    if(new_lhs_from>new_lhs_to) std::swap(new_lhs_from, new_lhs_to);

    assert(new_lhs_from<=new_lhs_to);
    
    lhs_from=lhs_from+integer2long(new_lhs_from);
    lhs_to=lhs_from+integer2long(new_lhs_to);
    
    add_equality_rec(src, lhs.op0(), lhs_from, lhs_to, rhs_entry);
  }
  else
    constraint_list.push_back(src);
}

/*******************************************************************\

Function: convert_trans_to_netlistt::add_constraint

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_trans_to_netlistt::add_constraint(const exprt &src)
{
  if(src.id()=="=")
  {
    add_equality(to_equality_expr(src));
  }
  else if(src.id()=="and")
  {
    forall_operands(it, src)
      add_constraint(*it);
  }
  else
    constraint_list.push_back(src);
}

/*******************************************************************\

Function: convert_trans_to_netlist

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_trans_to_netlist(
  const namespacet &ns,
  const transt &_trans,
  const std::list<exprt> &properties,
  netlist_transt &dest,
  messaget &message)
{
  convert_trans_to_netlistt c(ns, message);

  c(_trans, properties, dest);
}

/*******************************************************************\

Function: convert_trans_to_netlist

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_trans_to_netlist(
  const contextt &context,
  const irep_idt &module,
  const std::list<exprt> &properties,
  netlist_transt &dest,
  messaget &message)
{
  dest.var_map.map_vars(context, module);

  namespacet ns(context);

  convert_trans_to_netlist(
    ns, get_trans(ns, module), properties, dest, message);
}
