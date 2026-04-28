#include <assert.h>

struct D {
  int dummy;
  char fmt[16];
};

int main()
{
  unsigned n = nondet_uint() % 1024;
  struct D data[n];
  assert(data->fmt);
}
