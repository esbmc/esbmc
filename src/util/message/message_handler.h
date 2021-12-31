/*******************************************************************\

Module: Message Handler System. This system is responsible for all IO
  operations regarding the message system of ESBMC

Author: Daniel Kroening, kroening@kroening.com

Maintainers:
- @2021: Rafael Sá Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/

#ifndef CPROVER_MESSAGE_H

#define CPROVER_MESSAGE_H

#include <string>
#include <sstream>
#include <util/location.h>
#include <util/message/verbosity.h>

/**
 * @brief Message_handler is an interface for low-level print
 * operations, if you need to redirect ESBMC output this
 * is the class which you should specialize.
 * 
 */
class message_handlert
{
public:
  /**
   * @brief print the messsage
   * 
   * @param level verbosity level of message
   * @param message string with the mensage to be printed
   */
  virtual void
  print(VerbosityLevel level, const std::string &message) const = 0;

  /**
   * @brief print the message alongisde its location
   * 
   * @param level verbosity level of message
   * @param message string with the message to be printed
   * @param location add the message location
   */
  virtual void print(
    VerbosityLevel level,
    const std::string &message,
    const locationt &location) const;

  virtual ~message_handlert() = default;
};

/**
 * @brief This specialization will send print statements into a
 * FILE*
 * 
 * NOTE: The memory management for FILE* is not handled by this class
 */
class file_message_handler : public message_handlert
{
public:
  virtual void set_file(VerbosityLevel v, FILE *f)
  {
    files[v] = f;
  }

protected:
  std::unordered_map<VerbosityLevel, FILE *> files;
};

#endif
