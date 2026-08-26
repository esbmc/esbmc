#include <assert.h>

int main(void)
{
  int c = 0;
  double d = c ? -0.0 : 0.0;
  assert(__builtin_signbit(d) != 0);
  return 0;
}
