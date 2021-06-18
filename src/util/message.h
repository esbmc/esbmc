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
 * operations, if you need to redirect esbmc output this
 * is where you should specialize
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
    STATUS = 6,
    VERBOSE = 8,
    DEBUG = 9
  };

  virtual void print(unsigned level, const std::string &message);

  virtual void
  print(unsigned level, const std::string &message, const locationt &location);

  virtual ~message_handlert() = default;

  void set_file_output(FILE* file_out) { 
    stdout_output = file_out;
    stderr_output = file_out;
  }

  protected:
  FILE *stdout_output = stdout;
  FILE *stderr_output = stderr;
};

/**
 * @brief messaget is used to send messages that
 * can be implemented by any kind of frontend
 * 
 * It may be from colorful output to full GUI modules
 * 
 */
class messaget
{
public:
  void status(const std::string &message)
  {
    print(message_handlert::STATUS, message);
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
    print(message_handlert::STATUS, message, location);
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

  messaget() = default;
  explicit messaget(message_handlert &_message_handler) : message_handler(_message_handler) {}

  virtual ~messaget() = default;

  message_handlert *get_message_handler()
  {
    return &message_handler;
  }

protected:
  unsigned verbosity = 10;
  message_handlert message_handler;
};

#endif
