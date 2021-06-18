/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_MESSAGE_H

#define CPROVER_MESSAGE_H

#include <iostream>
#include <string>
#include <util/location.h>

/**
 * @brief Message_handler is an interface for low-level print
 * operations 
 */
class message_handlert
{
public:

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
  enum VERBOSITY {
    NONE = 0,
    ERROR = 1,
    WARNING = 2,
    RESULT = 4,
    PHASE = 6,
    STATISTICAL = 8,
    PROGRESS = 9,
    DEBUG = 10
  };

  virtual void print(unsigned level, const std::string &message) = 0;

  virtual void
  print(unsigned level, const std::string &message, const locationt &location);

  virtual ~message_handlert() = default;
};

class messaget
{
public:
  virtual void print(const std::string &message)
  {
    print(message_handlert::ERROR, message);
  }

  void status(const std::string &message)
  {
    print(message_handlert::PHASE, message);
  }

  void result(const std::string &message)
  {
    print(message_handlert::RESULT, message);
  }

  void warning(const std::string &message)
  {
    print(message_handlert::WARNING, message);
  }

  void status(const std::string &message, const std::string &file)
  {
    locationt location;
    location.set_file(file);
    print(message_handlert::PHASE, message, location);
  }

  void error(const std::string &message)
  {
    print(message_handlert::ERROR, message);
  }

  void error(const std::string &message, const std::string &file)
  {
    locationt location;
    location.set_file(file);
    print(message_handlert::ERROR, message, location);
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

#endif
