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
#include <util/location.h>

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
   * @brief Verbosity refers to the max level
   * of which inputs are going to be printed out
   * 
   * The level adds up to the greater level which means
   * that if the level is set to 3 all messages of value
   * 0,1,2,3 are going to be printed but 4+ will not be printed
   */
  enum VERBOSITY : int
  {
    NONE = 0,     // No message output
    ERROR = 1,    // fatal errors are printed
    WARNING = 2,  // warnings are printend
    RESULT = 3,   // results of the analysis (including CE)
    PROGRESS = 4, // progress notifications
    STATUS = 5,   // all kinds of esbmc is doing that may be useful to the user
    // ALWAYS set DEBUG as last.
    DEBUG = 6 // messages that are only useful if you need to debug.
  };

  /**
   * @brief print the messsage
   * 
   * @param level verbosity level of message
   * @param message string with the mensage to be printed
   */
  virtual void print(VERBOSITY level, const std::string &message) const = 0;

  /**
   * @brief print the message alongisde its location
   * 
   * @param level verbosity level of message
   * @param message string with the message to be printed
   * @param location add the message location
   */
  virtual void print(
    VERBOSITY level,
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
  virtual void set_file(message_handlert::VERBOSITY v, FILE *f)
  {
    files[v] = f;
  }

protected:
  std::unordered_map<message_handlert::VERBOSITY, FILE *> files;
};

#endif
