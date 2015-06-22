/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <fstream>

#include <i2string.h>
#include <xml.h>
#include <xml_irep.h>

#include "ui_message.h"

const char *ui_message_handlert::level_string(unsigned level)
{
  if(level==1)
    return "ERROR";
  else if(level==2)
    return "WARNING";
  else
    return "STATUS-MESSAGE";
}

void ui_message_handlert::print(
  unsigned level,
  const std::string &message)
{
  if(get_ui()==OLD_GUI || get_ui()==XML_UI)
  {
    locationt location;
    location.make_nil();
    print(level, message, location);
  }
  else
  {
    if(level==1)
      std::cerr << message << std::endl;
    else
      std::cout << message << std::endl;
  }
}

void ui_message_handlert::print(
  unsigned level,
  const std::string &message,
  const locationt &location)
{
  if(get_ui()==OLD_GUI || get_ui()==XML_UI)
  {
    std::string tmp_message(message);

    if(tmp_message.size()!=0 && tmp_message[tmp_message.size()-1]=='\n')
      tmp_message.resize(tmp_message.size()-1);
  
    const char *type=level_string(level);
    
    ui_msg(type, tmp_message, location);
  }
  else
  {
    message_handlert::print(level, message, location);
  }
}

void ui_message_handlert::old_gui_msg(
  const std::string &type,
  const std::string &msg1,
  const locationt &location)
{
  std::cout << type   << std::endl
            << msg1   << std::endl
            << location.get_file() << std::endl
            << location.get_line() << std::endl
            << location.get_column() << std::endl;
}

void ui_message_handlert::ui_msg(
  const std::string &type,
  const std::string &msg1,
  const locationt &location)
{
  if(get_ui()==OLD_GUI)
    old_gui_msg(type, msg1, location);
  else
    xml_ui_msg(type, msg1, location);
}

void ui_message_handlert::xml_ui_msg(
  const std::string &type,
  const std::string &msg1,
  const locationt &location)
{
  xmlt xml;
  xml.name="message";

  if(location.is_not_nil() && location.get_file()!="")
  {
    xmlt &l=xml.new_element();
    convert(location, l);
    l.name="location";
  }

  xml.new_element("text").data=xmlt::escape(msg1);
  xml.set_attribute("type", xmlt::escape_attribute(type));
  
  std::cout << xml;
  std::cout << std::endl;
}

