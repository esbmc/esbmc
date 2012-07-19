#include <global.h>
#include <stdio.h>
#include <string>
#include <global.h>
#include <cstdarg>

#include "z3_conv.h"
#include "z3_capi.h"

void
abortf(const char* message)
{
  fprintf(stderr, "%s\n", message);
  abort();
}
