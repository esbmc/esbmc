// Another counter's update follows the loop counter's.
unsigned nondet_uint(void);
int main() {
  unsigned x = nondet_uint();
  unsigned y = x;
  while (x < 100) { x++; y++; }
  __ESBMC_assert(x == y + 1, "");
}
