/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_UI_LANGUAGE_H
#define CPROVER_UI_LANGUAGE_H

#include <message.h>

#include <iostream>

class ui_message_handlert:public message_handlert
{
public:
  typedef enum { PLAIN, OLD_GUI, XML_UI, GRAPHML } uit;
  
  ui_message_handlert(uit __ui):_ui(__ui)
  {
    switch(__ui)
    {
    case OLD_GUI:
      break;
      
    case XML_UI:
      std::cout << "<cprover>" << std::endl << std::endl;
      break;
      
    case PLAIN:
      break;
      
    default:;
    }
  }
   
  virtual ~ui_message_handlert()
  {
    if(get_ui()==XML_UI)
      std::cout << "</cprover>" << std::endl;
  }

  uit get_ui() const
  {
    return _ui;
  }

protected:
  uit _ui;
 
  // overloading
  virtual void print(
    unsigned level,
    const std::string &message);

  // overloading
  virtual void print(
    unsigned level,
    const std::string &message,
    const locationt &location);

  virtual void old_gui_msg(
    const std::string &type,
    const std::string &msg1,
    const locationt &location);

  virtual void xml_ui_msg(
    const std::string &type,
    const std::string &msg1,
    const locationt &location);

  virtual void ui_msg(
    const std::string &type,
    const std::string &msg1,
    const locationt &location);

  const char *level_string(unsigned level);
};

#endif
