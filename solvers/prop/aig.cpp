/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <iostream>

#include <i2string.h>

#include "aig.h"

/*******************************************************************\

Function: aig_variable_labelingt::operator()

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string aig_variable_labelingt::operator()(unsigned v) const
{
  return "var("+i2string(v)+")";
}

/*******************************************************************\

Function: aigt::get_terminals

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void aigt::get_terminals(terminalst &terminals) const
{
  for(unsigned n=0; n<nodes.size(); n++)
    get_terminals_rec(n, terminals);
}

/*******************************************************************\

Function: aigt::get_terminals_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const aigt::terminal_sett &aigt::get_terminals_rec(
  unsigned n,
  terminalst &terminals) const
{
  terminalst::iterator it=terminals.find(n);
  
  if(it!=terminals.end())
    return it->second; // already done
  
  assert(n<nodes.size());
  const aig_nodet &node=nodes[n];
  
  terminal_sett &t=terminals[n];

  if(node.is_and())
  {
    if(!node.a.is_constant())
    {
      const std::set<unsigned> &ta=get_terminals_rec(node.a.var_no(), terminals);
      t.insert(ta.begin(), ta.end());
    }

    if(!node.b.is_constant())
    {
      const std::set<unsigned> &tb=get_terminals_rec(node.b.var_no(), terminals);
      t.insert(tb.begin(), tb.end());
    }
  }
  else // this is a terminal
  {
    t.insert(n);
  }
    
  return t;
}

/*******************************************************************\

Function: aigt::print

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void aigt::print(
  std::ostream& out,
  literalt a,
  const aig_variable_labelingt &labeling) const
{
  if(a==const_literal(false))
  {
    out << "FALSE";
    return;
  }
  else if(a==const_literal(true))
  {
    out << "TRUE";
    return;
  }

  unsigned node_nr=a.var_no();

  {
    const aig_nodet &node=nodes[node_nr];

    switch(node.type)
    {
    case aig_nodet::AND:
      if(a.sign()) out << "!(";
      print(out, node.a, labeling);
      out << "&";
      print(out, node.b, labeling);
      if(a.sign()) out << ")";
      break;
      
    case aig_nodet::VAR:
      if(a.sign()) out << "!";
      out << labeling(node.var_no());
      break;
      
    case aig_nodet::NEXT_VAR:
      if(a.sign()) out << "!";
      out << "next(" << labeling(node.var_no()) << ")";
      break;
      
    default:
      out << "unknown";
    }
  }
}

/*******************************************************************\

Function: aigt::output_dot_node

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void aigt::output_dot_node(
  std::ostream& out,
  unsigned v,
  const aig_variable_labelingt &labeling) const
{
  const aig_nodet &node=nodes[v];

  if(node.is_and())
  {
    out << v << " [label=\"" << v << "\"]" << std::endl;
    output_dot_edge(out, v, node.a);
    output_dot_edge(out, v, node.b);
  }
  else // the node is a terminal
  {
    out << v << " [label=\"" << labeling.dot_label(node.var_no()) << "\""
        << ",shape=box]" << std::endl;
  }
}

/*******************************************************************\

Function: aigt::output_dot_edge

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void aigt::output_dot_edge(
  std::ostream& out,
  unsigned v,
  literalt l) const
{
  if(l.is_true())
  {
    out << "TRUE -> " << v;
  }
  else if(l.is_false())
  {
    out << "TRUE -> " << v;
    out << " [arrowhead=odiamond]";
  }
  else
  {
    out << l.var_no() << " -> " << v;
    if(l.sign()) out << " [arrowhead=odiamond]";
  }

  out << std::endl;
}

/*******************************************************************\

Function: aigt::output_dot

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void aigt::output_dot(
  std::ostream& out, 
  const aig_variable_labelingt &labeling) const
{
  // constant TRUE
  out << "TRUE [label=\"TRUE\", shape=box]" << std::endl;

  // now the nodes
  for(unsigned n=0; n<number_of_nodes(); n++)
    output_dot_node(out, n, labeling);
}

/*******************************************************************\

Function: aigt::print

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void aigt::print(
  std::ostream &out,
  const aig_variable_labelingt &labeling) const
{
  for(unsigned n=0; n<number_of_nodes(); n++)
  {
    out << "n" << n << " = ";
    literalt l;
    l.set(n, false);
    print(out, l, labeling);
    out << std::endl;
  }
}

/*******************************************************************\

Function: aigt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void aigt::convert(
  propt &prop,
  std::vector<literalt> &literals) const
{
  literals.resize(number_of_nodes());

  for(unsigned i=0; i<literals.size(); i++)
  {
    const aig_nodet &n=nodes[i];

    if(n.is_and())
    {
      literalt a, b;

      if(n.a.is_constant())
        a=n.a;
      else
      {
        unsigned v_a=n.a.var_no();
        assert(v_a<i);
        a=literals[v_a].cond_negation(n.a.sign());
      }

      if(n.b.is_constant())
        b=n.b;
      else
      {
        unsigned v_b=n.b.var_no();
        assert(v_b<i);
        b=literals[v_b].cond_negation(n.b.sign());
      }

      literals[i]=prop.land(a, b);
    }
    else
      literals[i].set(n.var_no(), false);
  }
}

/*******************************************************************\

Function: aigt::add_variables

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void aigt::add_variables(propt &prop) const
{
  for(unsigned i=0; i<nodes.size(); i++)
  {
    const aig_nodet &n=nodes[i];

    if(n.is_var())
      while(prop.no_variables()<=n.var_no())
        prop.new_variable();
  }
}

/*******************************************************************\

Function: operator <<

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::ostream &operator << (std::ostream &out, const aigt &aig)
{
  aig.print(out, aig_variable_labelingt());
  return out;
}
