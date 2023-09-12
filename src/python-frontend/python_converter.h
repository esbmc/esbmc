#pragma once

#include "util/context.h"

class python_converter
{
public:
  python_converter(contextt& _context);
  bool convert();

private:
  contextt &context;
};
