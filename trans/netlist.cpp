/*******************************************************************\

Module: Graph representing Netlist

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ctype.h>

#include <i2string.h>

#include "netlist.h"

/*******************************************************************\

Function: netlistt::print

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void netlistt::print(std::ostream &out) const
{
  var_map.output(out);

  out << "Next state functions:\n";

  for(var_mapt::mapt::const_iterator
      it=var_map.map.begin();
      it!=var_map.map.end(); it++)
  {
    const var_mapt::vart &var=it->second;

    for(unsigned i=0; i<var.bits.size(); i++)
    {
      if(var.vartype==var_mapt::vart::VAR_LATCH)
      {
        out << "  NEXT(" << it->first
            << "[" << i << "])=";

        print(out, next_state[var.bits[i].var_no]);

        out << std::endl;
      }
    }
  }
}

/*******************************************************************\

Function: netlistt::print

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void netlistt::print(std::ostream &out, literalt a) const
{
  aigt::print(out, a, var_map_labelingt(var_map));
}

/*******************************************************************\

Function: dot_id

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static std::string dot_id(const std::string &id)
{
  std::string in=id;

  std::string::size_type pos;

  pos=in.rfind("::");

  if(pos!=std::string::npos)
    in=std::string(in, pos+2, std::string::npos);

  pos=in.rfind(".");

  if(pos!=std::string::npos)
    in=std::string(in, pos+1, std::string::npos);

  std::string result;

  result.reserve(in.size());

  for(unsigned i=0; i<in.size(); i++)
  {
    char ch=in[i];
    if(isalnum(ch) || ch=='(' || ch==')' || ch==' ' || ch=='.')
      result+=ch;
    else
      result+='_';
  }

  return result;
}

/*******************************************************************\

Function: var_map_labelingt::dot_label

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string var_map_labelingt::dot_label(unsigned v) const
{
  return dot_id((*this)(v));
}

/*******************************************************************\

Function: netlistt::output_dot

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void netlistt::output_dot(std::ostream &out) const
{
  aigt::output_dot(out, var_map_labelingt(var_map));

  // add the sinks
  for(var_mapt::mapt::const_iterator
      it=var_map.map.begin();
      it!=var_map.map.end();
      it++)
  {
    const var_mapt::vart &var=it->second;

    if(var.vartype==var_mapt::vart::VAR_LATCH)
    {
      assert(var.bits.size()==1);
      unsigned v=var.bits.front().var_no;
      literalt l=next_state[v];

      out << "next" << v << " [shape=box,label=\""
          << dot_id(id2string(it->first)) << "'\"]" << std::endl;

      if(l.is_constant())
        out << "TRUE";
      else
        out << l.var_no();

      out << " -> next" << v;
      if(l.sign()) out << " [arrowhead=odiamond]";
      out << std::endl;
    }
  }
}

