// A counter stepping by 2 writes every other element.
#define N 100000
int a[N];
unsigned nondet_uint(void);
int main() {
  unsigned i;
  for (i = 1; i < N; i += 2) a[i] = 1;
  unsigned x = nondet_uint();
  __ESBMC_assume(x < N);
  __ESBMC_assert(a[x] == (x % 2), "");
  __ESBMC_assert(i == N + 1, "");
}
