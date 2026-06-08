#include <assert.h>

struct D {
  int dummy;
  char fmt[16];
};

int main()
{
  unsigned n = nondet_uint() % 1024 + 1;
  struct D data[n];
  data->dummy = 42;
  assert(data->dummy == 42);
}
