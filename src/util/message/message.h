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
    print(message_handlert::ERROR, message, location);
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
    print(message_handlert::WARNING, message, location);
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
    print(message_handlert::RESULT, message, location);
  }
  /**
   * @brief Prints a progress message
   * 
   * @param message string to be printed
   * @param progress: float from 0-1 that represents the current progress of the latest task. (-1 means infinite)
   * @param location location where the message happened
   */
  virtual void progress(
    const std::string &message,
    double progress = -1,
    const std::string location = "") const
  {
    print(message_handlert::PROGRESS, message, location);
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
    print(message_handlert::STATUS, message, location);
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
    print(message_handlert::DEBUG, message, location);
  }

  /**
   * @brief Set the verbosity level
   * 
   * @param _verbosity new verbosity leveel
   */
  virtual void set_verbosity(message_handlert::VERBOSITY _verbosity)
  {
    verbosity = _verbosity;
  }

  /**
     * @brief Converts all parameter into a string of the specified method
     *  this should be used in order to replace stream calls. Similar of how
     * printf is used
     * 
     * This format is specified through the formatter class specification(C++20)
     * https://en.cppreference.com/w/cpp/utility/format/formatter
     * 
     * NOTE: The implementation for this must support C++14.
     * 
     * Example: format_to_string("{} {} {}!", "Hello", "World", 3);
     *        > "Hello World 3!"
     * 
     * @tparam TS any kind of object that can be converted into a string
     * @param format string with the format example "{} {} {.3d}"
     * @param ts list of strings that are going to be put inside the format
     * @return std::string with the formated string
     */
  template <typename... TS>
  std::string format_to_string(const std::string &format, const TS &...ts);

  /**
   * @brief Get the verbosity level
   * 
   * @return message_handlert::VERBOSITY level
   */
  virtual message_handlert::VERBOSITY get_verbosity() const
  {
    return verbosity;
  }

  virtual ~messaget() = default;

  // REVAMP THIS
  message_handlert *get_message_handler(unsigned index)
  {
    return handlers.at(index).get();
  }

  virtual void print(
    message_handlert::VERBOSITY level,
    const std::string &message,
    const std::string file = "") const
  {
    // Check if the message should be printed
    if(level > verbosity)
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
    message_handlert::VERBOSITY level,
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
   * @brief Appends a message_handler to be used in this message
   * 
   * @param handler to be appended
   * @return index of the vector where the handler was appended
   */
  // TODO: REVAMP THIS
  unsigned add_message_handler(std::shared_ptr<message_handlert> &handler)
  {
    handlers.push_back(handler);
    return handlers.size() - 1;
  }

protected:
  // Current verbosity level
  message_handlert::VERBOSITY verbosity = message_handlert::DEBUG;
  // All message_handlers used
  std::vector<std::shared_ptr<message_handlert>> handlers;
};