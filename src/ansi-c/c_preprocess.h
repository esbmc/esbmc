/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_C_PREPROCESS_H
#define CPROVER_C_PREPROCESS_H

#include <iostream>
#include <string>
#include <util/message.h>

bool c_preprocess(
  const std::string &path,
  std::ostream &outstream,
  bool is_cpp);

#endif
