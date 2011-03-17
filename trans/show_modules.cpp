/*******************************************************************\

Module: Show Modules

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <xml.h>
#include <i2string.h>
#include <xml_irep.h>

#include "show_modules.h"

/*******************************************************************\

Function: show_modules

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void show_modules(
  const contextt &context,
  ui_message_handlert::uit ui)
{
  unsigned count=0;

  forall_symbols(it, context.symbols)
  {
    const symbolt &symbol=it->second;
  
    if(symbol.type.id()=="module")
    {
      count++;

      switch(ui)
      {
      case ui_message_handlert::XML_UI:
        {
          xmlt xml("module");
          xml.new_element("number").data=i2string(count);
        
          xmlt &l=xml.new_element();
          convert(symbol.location, l);
          l.name="location";
        
          xml.new_element("identifier").data=
            xmlt::escape(id2string(symbol.name));
          
          xml.new_element("mode").data=
            xmlt::escape(id2string(symbol.mode));
          
          xml.new_element("name").data=
            xmlt::escape(id2string(symbol.display_name()));
          
          std::cout << xml << std::endl;
        }
  
        break;
      
      case ui_message_handlert::PLAIN:
        std::cout << "Module " << count << ":" << std::endl;

        std::cout << "  Location:   " << symbol.location << std::endl;
        std::cout << "  Mode:       " << symbol.mode << std::endl;
        std::cout << "  Identifier: " << symbol.name << std::endl;
        std::cout << "  Name:       " << symbol.display_name() << std::endl
                  << std::endl;
        break;
      
      default:
        assert(false);
      }
    }
  }
}
