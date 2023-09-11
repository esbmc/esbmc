#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

struct D
{
  int dummy;
  char fmt[16];
};

int main()
{
  unsigned n = nondet_uint() % 1024;
  n++;
  unsigned i = nondet_uint() % n;
  struct D data[1];
#ifndef _WIN32
  int r = printf("%02X", data[0].fmt);
  assert(r < 2);
#else
  assert(0);
#endif
}