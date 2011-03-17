/*******************************************************************\

Module: Approximation of CT

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <algorithm>

#include <i2string.h>
#include <arith_tools.h>
#include <graph.h>

#include "compute_ct.h"

struct component_graph_nodet:public graph_nodet<>
{
public:
  mp_integer weight;
  
  component_graph_nodet():weight(1)
  {
  }
  
  typedef std::set<unsigned> latchest;
  latchest latches;
};

class component_grapht:public graph<component_graph_nodet>
{
public:
  component_grapht()
  {
  }
  
  void compute(const ldgt &ldg);
  
  void transform();

  void print(std::ostream &out) const;

  unsigned ct();
  
protected:
  void collapse_scc(
    const std::vector<unsigned> &scc_nr,
    unsigned i);

  mp_integer longest_path(unsigned start);
};

/*******************************************************************\

Function: component_grapht::print

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void component_grapht::print(std::ostream &out) const
{
  for(unsigned v=0;
      v!=nodes.size();
      v++)
  {
    const nodet &node=nodes[v];
    
    if(node.latches.empty()) continue;

    out << "Node " << v << ":" << std::endl;
    out << "  W: " << node.weight << std::endl;

    out << " ->: ";
 
    for(component_graph_nodet::edgest::const_iterator
        it=node.out.begin();
        it!=node.out.end();
        it++)
      out << " " << it->first;
      
    out << std::endl;

    out << "  L: ";
 
    for(component_graph_nodet::latchest::const_iterator
        it=node.latches.begin();
        it!=node.latches.end();
        it++)
      out << " " << *it;
      
    out << std::endl << std::endl;
  }
}

/*******************************************************************\

Function: component_grapht::compute

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void component_grapht::compute(const ldgt &ldg)
{
  // we start with a node for each latch
  nodes.clear();
  nodes.resize(ldg.size());

  // replicate edges  
  for(ldgt::latchest::const_iterator
      it=ldg.latches.begin();
      it!=ldg.latches.end();
      it++)
  {
    unsigned v=*it;
    const ldg_nodet &ldg_node=ldg[v];
    
    nodes[v].latches.insert(v);

    for(ldg_nodet::edgest::const_iterator
        it=ldg_node.out.begin();
        it!=ldg_node.out.end();
        it++)
      add_edge(v, it->first);
  }
}

/*******************************************************************\

Function: component_grapht::transform

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void component_grapht::transform()
{
  // find SCCs
  
  std::vector<unsigned> scc_nr;
  unsigned number_of_sccs=SCCs(scc_nr);

  #if 0
  for(unsigned n=0; n<nodes.size(); n++)
    std::cout << "SCC: " << scc_nr[n] << std::endl;
  #endif

  // we collapse all SCCs
  for(unsigned i=0; i<number_of_sccs; i++)
    collapse_scc(scc_nr, i);
}

/*******************************************************************\

Function: component_grapht::collapse_scc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void component_grapht::collapse_scc(
  const std::vector<unsigned> &scc_nr,
  unsigned i)
{
  std::set<unsigned> latches;

  for(unsigned n=0; n<nodes.size(); n++)
    if(!nodes[n].latches.empty())
      if(scc_nr[n]==i)
        latches.insert(n);
  
  // really an SCC?
  if(latches.empty()) return;
  
  unsigned representative=*latches.begin();
  
  if(latches.size()==1 && !has_edge(representative, representative))
    return;
  
  nodes[representative].latches=latches;
  
  for(std::set<unsigned>::const_iterator
      it=latches.begin();
      it!=latches.end();
      it++)
  {
    unsigned n=*it;
    if(n!=representative)
    {
      nodet &node=nodes[n];
    
      // add new incoming edges
      for(edgest::const_iterator
          it=node.in.begin();
          it!=node.in.end();
          it++)
        add_edge(it->first, representative);
    
      // add new outgoing edges
      for(edgest::const_iterator
          it=node.out.begin();
          it!=node.out.end();
          it++)
        add_edge(representative, it->first);
        
      // clear latches
      node.latches.clear();
    }
  }

  for(std::set<unsigned>::const_iterator
      it=latches.begin();
      it!=latches.end();
      it++)
    if(*it!=representative)
      remove_edges(*it);
  
  // update weight
  nodes[representative].weight=power(2, latches.size());

  // make sure there is no self-loop on the representative
  remove_edge(representative, representative);
}

/*******************************************************************\

Function: component_grapht::ct

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

mp_integer component_grapht::longest_path(unsigned start)
{
  mp_integer max_val=0;

  std::vector<bool> visited;
  std::vector<mp_integer> distance;
  
  visited.resize(nodes.size(), false);
  distance.resize(nodes.size(), false);

  std::stack<unsigned> dfs_stack;
  dfs_stack.push(start);
  distance[start]=nodes[start].weight;

  // this only works if the graph is cycle-free
  
  while(!dfs_stack.empty())
  {
    unsigned v=dfs_stack.top();
    dfs_stack.pop();
    
    const nodet &n=nodes[v];
    visited[v]=true;
    
    if(distance[v]>max_val) max_val=distance[v];
  
    for(edgest::const_iterator o_it=n.out.begin();
        o_it!=n.out.end();
        o_it++)
    {                                                  
      unsigned w=o_it->first;
      if(!visited[w] || distance[w]<distance[v]+nodes[w].weight)
      {
        dfs_stack.push(w);
        distance[w]=distance[v]+nodes[w].weight;
      }
    }
  }
  
  return max_val;
}

/*******************************************************************\

Function: component_grapht::ct

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

unsigned component_grapht::ct()
{
  mp_integer max_val=0;
  
  // find longest weighted path through the CG

  for(unsigned n=0; n<nodes.size(); n++)
    if(!nodes[n].latches.empty())
    {
      mp_integer d=longest_path(n);
      if(d>max_val) max_val=d;
    }
    
  if(max_val>MAX_CT)
    return MAX_CT;
    
  return integer2long(max_val);
}

/*******************************************************************\

Function: compute_ct

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

unsigned compute_ct(const ldgt &ldg)
{
  component_grapht cg;

  cg.compute(ldg);

  #if 0
  cg.print(std::cout);
  #endif

  cg.transform();

  #if 0
  std::cout << "TRANSFORMED\n\n";  
  cg.print(std::cout);
  #endif

  return cg.ct();
}
