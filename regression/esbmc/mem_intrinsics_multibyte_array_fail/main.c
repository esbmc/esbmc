#include <assert.h>
#include <string.h>

/* As mem_intrinsics_multibyte_array, but the copied word is compared with
   the bytes in the wrong order. */
struct P
{
  unsigned short x, y;
};

unsigned words[4] = {0, 0x04030201u, 0, 0};
struct P pairs[2] = {{0, 0}, {0x0605, 0x0807}};

int main()
{
  const unsigned char bytes[4] = {1, 2, 3, 4};
  assert(memcmp(bytes, &words[1], 4) == 0);
  assert(memcmp(&words[1], bytes, 4) == 0);
  assert(memcmp(bytes, &pairs[1], 2) != 0);
  assert(memchr(&words[1], 3, 4) == (unsigned char *)&words[1] + 2);

  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 4);
  unsigned char d[4] = {0, 0, 0, 0};
  memcpy(d, &pairs[1], n);
  assert(n < 3 || d[2] == 7);
  assert(memcmp(d, &pairs[1], n) == 0);

  memcpy(&words[2], bytes, n);
  assert(n < 4 || words[2] == 0x01020304u);
  return 0;
}
