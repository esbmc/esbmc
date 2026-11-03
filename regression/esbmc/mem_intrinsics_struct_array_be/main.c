#include <assert.h>
#include <string.h>

/* A symbolic-length copy out of an array of structs, with a byte-wide element
   array beside it, on a big-endian target. */
struct P
{
  unsigned short x, y;
};

struct B
{
  unsigned char c;
};

struct P pairs[2] = {{0, 0}, {0x0506, 0x0708}};
struct B wrapped[4] = {{1}, {2}, {3}, {4}};

int main()
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 4);
  unsigned char d[4] = {0, 0, 0, 0};
  memcpy(d, &pairs[1], n);
  assert(n < 4 || (d[0] == 5 && d[1] == 6 && d[2] == 7 && d[3] == 8));
  assert(memchr(wrapped, 3, 4) == (unsigned char *)&wrapped[2]);
  return 0;
}
