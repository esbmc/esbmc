// A counter decremented before the write reads it, wrapping below zero.
#define N 100000
unsigned a[N];
unsigned nondet_uint(void);
int main() {
  unsigned j = 7;
  for (unsigned i = 0; i < N; i++) { j -= 2; a[i] = j; }
  unsigned x = nondet_uint();
  __ESBMC_assume(x < N);
  __ESBMC_assert(a[x] == 7 - 2 * (x + 1), "");
}
