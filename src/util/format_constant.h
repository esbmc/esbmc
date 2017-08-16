/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_FORMAT_CONSTANT_H
#define CPROVER_FORMAT_CONSTANT_H

#include <util/irep2.h>

class format_constantt:public format_spect
{
public:
  std::string operator()(const expr2tc &expr);
};

#endif
