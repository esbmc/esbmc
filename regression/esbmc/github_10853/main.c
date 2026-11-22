#include <stddef.h>
extern unsigned char __VERIFIER_nondet_uchar(void);
extern unsigned int __VERIFIER_nondet_uint(void);
int copy_bytes(const unsigned char *in, size_t n, unsigned char out[4]) {
  if (n > 4) return -1;
  size_t i = 0;
  __ESBMC_loop_invariant(i <= n && n <= 4);
  for (; i < n; i++)
    out[i] = in[i];
  return (int)n;
}
int main(void) {
  unsigned char in[5], out[4];
  int i = 0;
  __ESBMC_loop_invariant(0 <= i && i <= 5);
  for (; i < 5; i++) in[i] = __VERIFIER_nondet_uchar();
  size_t n = __VERIFIER_nondet_uint();
  copy_bytes(in, n, out);
  return 0;
}
