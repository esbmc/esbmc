/*******************************************************************\

Module: Message System. This system is used to send messages through
    ESBMC execution.
Author: Daniel Kroening, kroening@kroening.com

Maintainers:
- @2021: Rafael Sá Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/
#pragma once

#include <memory>
#include <util/message/message_handler.h>
#include <util/location.h>

/**
 * @brief messaget is used to send messages that
 * can be implemented by any kind of frontend
 * 
 * It may be from colorful output to full GUI modules,
 * this can also hold a number of of message_handlers
 * 
 */
class messaget
{
public:
  // PRINT Functions

  /**
   * @brief Prints an error message
   * 
   * @param message string to be printed
   * @param location location where the message happened
   */
  virtual void
  error(const std::string &message, const std::string location = "") const
  {
    print(VerbosityLevel::Error, message, location);
  }
  /**
   * @brief Prints a warning message
   * 
   * @param message string to be printed
   * @param location location where the message happened
   */
  virtual void
  warning(const std::string &message, const std::string location = "") const
  {
    print(VerbosityLevel::Warning, message, location);
  }
  /**
   * @brief Prints a result message
   * 
   * @param message string to be printed
   * @param location location where the message happened
   */
  virtual void
  result(const std::string &message, const std::string location = "") const
  {
    print(VerbosityLevel::Result, message, location);
  }
  /**
   * @brief Prints a progress message
   * 
   * @param message string to be printed
   * @param progress: float from 0-1 that represents the current progress of the latest task. (-1 means infinite)
   * @param location location where the message happened
   * @return an index representing this progress
   */
  virtual unsigned progress(
    const std::string &message,
    double progress [[gnu::unused]] = -1,
    const std::string location = "") const
  {
    print(VerbosityLevel::Progress, message, location);
    return 0;
  }

  /**
   * @brief Updates progress of specific task
   * 
   * @param index of task
   * @param progress new progress status
   */
  virtual void update_progress(
    unsigned index [[gnu::unused]],
    double progress [[gnu::unused]]) const
  {
  }
  /**
   * @brief Prints a status message
   * 
   * @param message string to be printed
   * @param location location where the message happened
   */
  virtual void
  status(const std::string &message, const std::string location = "") const
  {
    print(VerbosityLevel::Status, message, location);
  }
  /**
   * @brief Prints a debug message
   * 
   * @param message string to be printed
   * @param location location where the message happened
   */
  virtual void
  debug(const std::string &message, const std::string location = "") const
  {
    print(VerbosityLevel::Debug, message, location);
  }

  /**
   * @brief Set the verbosity level
   * 
   * @param _verbosity new verbosity leveel
   */
  virtual void set_verbosity(VerbosityLevel _verbosity)
  {
    verbosity = _verbosity;
  }

  /**
   * @brief Get the verbosity level
   * 
   * @return VerbosityLevel level
   */
  virtual VerbosityLevel get_verbosity() const
  {
    return verbosity;
  }

  virtual ~messaget() = default;
  virtual void print(
    VerbosityLevel level,
    const std::string &message,
    const std::string file = "") const
  {
    // Check if the message should be printed
    if((int)level > (int)verbosity)
      return;

    // Send the message to all handlers
    for(const auto &x : handlers)
    {
      if(file != "")
      {
        locationt l;
        l.set_file(file);
        x->print(level, message, l);
      }
      else
        x->print(level, message);
    }
  }

  virtual void print(
    VerbosityLevel level,
    const std::string &message,
    const locationt l) const
  {
    // Check if the message should be printed
    if(level > verbosity)
      return;

    for(const auto &x : handlers)
      x->print(level, message, l);
  }

  /**
   * @brief Append a message_handler to be used
   * 
   * @param handler to be appended
   */
  void add_message_handler(std::shared_ptr<message_handlert> &handler)
  {
    handlers.push_back(handler);
  }

  /**
   * @brief Returns a temporary FILE* descriptor for message outputs.
   *
   * @return
   */
  static FILE *get_temp_file();

  /**
   * @brief Insert all contents of the file into all message handlers
   * and then closes the file.
   *
   * @param l verbosity level of the file
   * @param f file pointer with the file contents.
   */
  void insert_and_close_file_contents(VerbosityLevel l, FILE *f) const;

protected:
  // Current verbosity level
  VerbosityLevel verbosity = VerbosityLevel::Debug;
  // All message_handlers used
  std::vector<std::shared_ptr<message_handlert>> handlers;
};