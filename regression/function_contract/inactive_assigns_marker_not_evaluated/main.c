/* memcpy's frame is not processed (only f is replaced), so its marker must not
   run: evaluating __ESBMC_object(dst) into an array member raises a spurious
   dereference failure. */
#include <string.h>
int f(int x)
{
  __ESBMC_ensures(__ESBMC_return_value == x);
  return x;
}
struct S { unsigned short buf[4]; };
int main()
{
  struct S s0, *s = &s0;
  unsigned short seed[4] = {1, 2, 3, 4};
  memcpy(s->buf, seed, sizeof seed);
  __ESBMC_assert(s->buf[3] == 4, "copied");
  return f(1) - 1;
}
