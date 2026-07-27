#ifndef CPROVER_FORMAT_CONSTANT_H
#define CPROVER_FORMAT_CONSTANT_H

#include <irep2/irep2.h>

class format_constantt : public format_spect
{
public:
  std::string operator()(const expr2tc &expr);
};

#endif
