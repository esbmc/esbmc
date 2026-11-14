// hi widens an unsigned short, so i += 2 cannot wrap past it.
unsigned short nondet_ushort(void);
int main() {
  unsigned short N = nondet_ushort();
  unsigned x = 0;
  while (x < N) x += 2;
  __ESBMC_assert(x % 2 == 0, "");
  __ESBMC_assert(x == N || x == N + 1, "");
}
