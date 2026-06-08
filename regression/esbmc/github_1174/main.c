#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct D
{
  int dummy;
  char fmt[16];
};

int main()
{
  struct D data[1];
  int r = printf("%02X", data[0].fmt);
  assert(r < 2);
}
