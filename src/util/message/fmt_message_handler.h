/*******************************************************************\
Module: FMT Message Handler specialization.

Author: Rafael Sá Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/
#pragma once
#include <util/message/message_handler.h>

/**
 * @brief This is a specialization that by using
 * the fmt library prints messages into FILE* objects
 * 
 */
class fmt_message_handler : public file_message_handler
{
public:
  fmt_message_handler();
  virtual void print(
    message_handlert::VERBOSITY level,
    const std::string &message) const override;
};