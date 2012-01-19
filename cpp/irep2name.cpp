/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <ctype.h>

#include <i2string.h>
#include <std_types.h>

#include "irep2name.h"

/*******************************************************************\

Function: do_prefix

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string do_prefix(const std::string &s)
{
  if(s.find(',')!=std::string::npos ||
     (s!="" && isdigit(s[0])))
    return i2string((unsigned long)s.size())+"_"+s;

  return s;
}

/*******************************************************************\

Function: irep2name

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void irep2name(const irept &irep, std::string &result)
{
  result="";

  if(is_reference(static_cast<const typet&>(irep)))
    result+="reference";
  if(irep.id()!="")
    result+=do_prefix(irep.id_string());

  if(irep.get_named_sub().empty() &&
     irep.get_sub().empty() &&
     irep.get_comments().empty())
    return;

  result+="(";
  bool first=true;

  forall_named_irep(it, irep.get_named_sub())
  {
    if(first) first=false; else result+=",";

    result+=do_prefix(name2string(it->first));

    result+="=";
    std::string tmp;
    irep2name(it->second, tmp);
    result+=tmp;
  }

  forall_named_irep(it, irep.get_comments())
    if(it->first=="#constant" ||
       it->first=="#volatile" ||
       it->first=="#restricted")
    {
      if(first) first=false; else result+=",";
      result+=do_prefix(name2string(it->first));
      result+="=";
      std::string tmp;
      irep2name(it->second, tmp);
      result+=tmp;
    }

  forall_irep(it, irep.get_sub())
  {
    if(first) first=false; else result+=",";
    std::string tmp;
    irep2name(*it, tmp);
    result+=tmp;
  }

  result+=")";
}

