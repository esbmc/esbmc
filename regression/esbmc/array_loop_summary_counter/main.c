// A second counter read by the write before its own update.
#define N 100000
int a[N];
unsigned nondet_uint(void);
int main() {
  int j = 0;
  for (int i = 0; i < N; i++) { a[i] = j; j += 3; }
  unsigned x = nondet_uint();
  __ESBMC_assume(x < N);
  __ESBMC_assert(a[x] == 3 * (int)x, "");
  __ESBMC_assert(j == 3 * N, "");
}
