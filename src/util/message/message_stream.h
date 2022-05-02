/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ERROR_HANDLER_H
#define CPROVER_ERROR_HANDLER_H

#include <sstream>
#include <util/expr.h>
#include <util/message/message_handler.h>
#include <util/message/message.h>
class message_streamt
{
public:
  message_streamt(const messaget &_message_handler)
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
    send_msg(VerbosityLevel::Error, message);
  }

  void warning(const std::string &message)
  {
    send_msg(VerbosityLevel::Warning, message);
  }

  void error()
  {
    send_msg(VerbosityLevel::Error, str.str());
    clear_err();
  }

  void warning()
  {
    send_msg(VerbosityLevel::Warning, str.str());
    clear_err();
  }

  void status()
  {
    send_msg(VerbosityLevel::Status, str.str());
    clear_err();
  }

  std::ostringstream str;

  bool get_error_found() const
  {
    return error_found;
  }

  const messaget &get_message_handler()
  {
    return message_handler;
  }

  void error_parse(VerbosityLevel level)
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
  const messaget &message_handler;
  bool error_found;
  locationt saved_error_location;

  void send_msg(VerbosityLevel level, const std::string &message)
  {
    if(message == "")
      return;
    if((char)level <= (char)VerbosityLevel::Error)
      error_found = true;
    message_handler.print(level, message, saved_error_location);
    saved_error_location.make_nil();
  }

  void error_parse_line(VerbosityLevel level, const std::string &line);

  void error_parse(VerbosityLevel level, const std::string &error);
};

#endif
