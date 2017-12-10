/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ERROR_HANDLER_H
#define CPROVER_ERROR_HANDLER_H

#include <sstream>
#include <util/expr.h>
#include <util/message.h>

class message_streamt
{
public:
  message_streamt(message_handlert &_message_handler)
    : message_handler(_message_handler),
      error_found(false),
      saved_error_location(static_cast<const locationt &>(get_nil_irep()))
  {
  }

  virtual ~message_streamt() = default;

  // overload to use language specific syntax
  virtual std::string to_string(const exprt &expr)
  {
    return expr.to_string();
  }
  virtual std::string to_string(const typet &type)
  {
    return type.to_string();
  }

  void err_location(const exprt &expr)
  {
    saved_error_location = expr.find_location();
  }
  void err_location(const typet &type)
  {
    saved_error_location = type.location();
  }
  void err_location(const irept &irep)
  {
    saved_error_location = (const locationt &)irep.cmt_location();
  }
  void err_location(const locationt &_location)
  {
    saved_error_location = _location;
  }

  void error(const std::string &message)
  {
    send_msg(1, message);
  }

  void warning(const std::string &message)
  {
    send_msg(2, message);
  }

  void error()
  {
    send_msg(1, str.str());
    clear_err();
  }

  void warning()
  {
    send_msg(2, str.str());
    clear_err();
  }

  void status()
  {
    send_msg(6, str.str());
    clear_err();
  }

  std::ostringstream str;

  bool get_error_found() const
  {
    return error_found;
  }

  message_handlert &get_message_handler()
  {
    return message_handler;
  }

  void error_parse(unsigned level)
  {
    error_parse(level, str.str());
    clear_err();
  }

  void clear_err()
  {
    str.clear();
    str.str("");
  }

protected:
  message_handlert &message_handler;
  bool error_found;
  locationt saved_error_location;

  void send_msg(unsigned level, const std::string &message)
  {
    if(message == "")
      return;
    if(level <= 1)
      error_found = true;
    message_handler.print(level, message, saved_error_location);
    saved_error_location.make_nil();
  }

  void error_parse_line(unsigned level, const std::string &line);

  void error_parse(unsigned level, const std::string &error);
};

#endif
