/*******************************************************************\

Module: Find module symbol using name

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "get_module.h"
#include "message_stream.h"

/*******************************************************************\

Function: get_module

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const symbolt &get_module(
  const contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  symbolptr_listt symbolptr_list;
  message_streamt message_stream(message_handler);

  forall_symbol_base_map(it, context.symbol_base_map, module)
  {
    symbolst::const_iterator it2=context.symbols.find(it->second);

    if(it2==context.symbols.end())
      continue;

    const symbolt &s=it2->second;

    if(s.is_type || s.free_var || s.type.id()!="module")
      continue;
    
    symbolptr_list.push_back(&s);
  }

  if(symbolptr_list.empty())
  {
    message_stream.str << "module `" << module << "' not found";
    message_stream.error();
    throw 0;
  }
  else if(symbolptr_list.size()>=2)
  {
    message_stream.str << "module `" << module << "' does not uniquely resolve:" << std::endl;

    forall_symbolptr_list(it, symbolptr_list)
      message_stream.str << "  " << (*it)->name << std::endl;

    message_stream.error();
    throw 0;
  }

  // symbolptr_list has exactly one element

  return *symbolptr_list.front();
}
