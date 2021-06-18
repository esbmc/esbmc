/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_MESSAGE_H

#define CPROVER_MESSAGE_H

#include <iostream>
#include <string>
#include <sstream>
#include <util/location.h>

// Levels:
//
//  0 none
//  1 only errors
//  2 + warnings
//  4 + results
//  6 + phase information
//  8 + statistical information
//  9 + progress information
// 10 + debug info

enum message_verbosity_level
{
  VERBOSITY_NONE = 0,
  VERBOSITY_ERRORS = 1,
  VERBOSITY_WARNINGS = 2,
  VERBOSITY_RESULTS = 4,
  VERBOSITY_PHASE = 6,
  VERBOSITY_STATISTICS = 8,
  VERBOSITY_PROGRESS = 9,
  VERBOSITY_DEBUG = 10
};
class message_handlert
{
public:
  virtual void print(unsigned level, const std::string &message) = 0;

  virtual void
  print(unsigned level, const std::string &message, const locationt &location);

  virtual ~message_handlert() = default;
};

/**
 * @brief This class will send print messages into
 * output streams
 * 
 */
class stream_message_handlert : public message_handlert
{
public:
  virtual void print(unsigned level, const std::string &message);
  stream_message_handlert() = default;
  stream_message_handlert(std::ostream &print, std::ostream &error)
    : default_output(print), error_output(error)
  {
  }

protected:
  std::ostream &default_output;
  std::ostream &error_output;
};

class messaget
{
public:
  virtual void print(const std::string &message)
  {
    print(VERBOSITY_ERRORS, message);
  }

  void status(const std::string &message)
  {
    print(VERBOSITY_PHASE, message);
  }

  void result(const std::string &message)
  {
    print(VERBOSITY_RESULTS, message);
  }

  void warning(const std::string &message)
  {
    print(VERBOSITY_WARNINGS, message);
  }

  void status(const std::string &message, const std::string &file)
  {
    locationt location;
    location.set_file(file);
    print(VERBOSITY_PHASE, message, location);
  }

  void error(const std::string &message)
  {
    print(VERBOSITY_ERRORS, message);
  }

  void error(const std::string &message, const std::string &file)
  {
    locationt location;
    location.set_file(file);
    print(VERBOSITY_ERRORS, message, location);
  }

  virtual void print(unsigned level, const std::string &message);

  virtual void
  print(unsigned level, const std::string &message, const locationt &location);

  virtual void set_message_handler(message_handlert *_message_handler);
  virtual void set_verbosity(unsigned _verbosity)
  {
    verbosity = _verbosity;
  }

  virtual unsigned get_verbosity() const
  {
    return verbosity;
  }

  messaget()
  {
    message_handler = (message_handlert *)nullptr;
    verbosity = 10;
  }

  messaget(message_handlert &_message_handler)
  {
    message_handler = &_message_handler;
    verbosity = 10;
  }

  virtual ~messaget() = default;

  message_handlert *get_message_handler()
  {
    return message_handler;
  }

protected:
  unsigned verbosity;
  message_handlert *message_handler;
};

namespace esbmc::global
{
extern messaget _msg; // use this if you know what you are doing
}
// Magic definitions to help the use of messages during the program

/* in time the implementation can be replaced with <format> */
#define _TO_MSG(X)                                                             \
  std::stringstream _convert_ss_to_str;                                        \
  _convert_ss_to_str << X;

#define ERROR(X)                                                               \
  {                                                                            \
    _TO_MSG(X);                                                                \
    esbmc::global::_msg.error(_convert_ss_to_str.str());                       \
  }
#define PRINT(X)                                                               \
  {                                                                            \
    _TO_MSG(X);                                                                \
    esbmc::global::_msg.print(VERBOSITY_DEBUG, _convert_ss_to_str.str());      \
  }

#endif
