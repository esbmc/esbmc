// With n near UINT_MAX, i += 2 wraps before reaching n, so no closed form
// holds: the loop must stay.
unsigned nondet_uint(void);
int main() {
  unsigned n = nondet_uint();
  unsigned i;
  for (i = 0; i < n; i += 2)
    ;
  __ESBMC_assert(i >= n, "");
}
