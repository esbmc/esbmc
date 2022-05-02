/*******************************************************************\

Module: This defines the verbosity LEVELS to be used throghout the program

Author: Rafael Menezes, rafael.sa.menezes@outlook.com

Maintainers:
\*******************************************************************/
#pragma once

/**
   * @brief Verbosity refers to the max level
   * of which inputs are going to be printed out
   *
   * The level adds up to the greater level which means
   * that if the level is set to 3 all messages of value
   * 0,1,2,3 are going to be printed but 4+ will not be printed
   *
   * The number is where it appeared in the definition, in the
   * implementation below DEBUG is the highest value
   */
enum class VerbosityLevel : char
{
  None,     // No message output
  Error,    // fatal errors are printed
  Warning,  // warnings are printend
  Result,   // results of the analysis (including CE)
  Progress, // progress notifications
  Status,   // all kinds of esbmc is doing that may be useful to the user
  Debug     // messages that are only useful if you need to debug.
};