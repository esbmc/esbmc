#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);

int main()
{
  unsigned char src[128] = {7}, dst[128];
  unsigned off = nondet_uint(), n = nondet_uint();
  __ESBMC_assume(off <= 64 && n >= 1 && n <= 64);
  memcpy(dst + off, src, n);
  assert(dst[off] == 7);

  unsigned char a[16];
  for (int i = 0; i < 16; i++)
    a[i] = i;
  unsigned d = nondet_uint(), s = nondet_uint(), m = nondet_uint();
  __ESBMC_assume(d < 8 && s < 8 && m <= 8);
  memmove(a + d, a + s, m);
  unsigned k = nondet_uint(), u = nondet_uint();
  __ESBMC_assume(k < m && u < 16 && (u < d || u >= d + m));
  assert(a[d + k] == s + k);
  assert(a[u] == u);

  int w[8] = {1, 2, 3, 4, 5, 6, 7, 8}, v[8] = {0};
  unsigned i = nondet_uint(), c = nondet_uint();
  __ESBMC_assume(i < 8 && c <= 4);
  memcpy((char *)v + 4 * i, (char *)w + 4 * i, c);
  assert(v[i] == 0 || v[i] == w[i]);
  return 0;
}
