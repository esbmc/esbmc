unsigned nondet_uint(void);
unsigned id(unsigned v) { return v; }
int main(void) {
  unsigned c = nondet_uint();
  unsigned s = s + 1;
  unsigned a = id(a);
  unsigned t = c ? id(c) : 0;
  unsigned u = ({ unsigned k = 0; while (k < c) k = k + 1; k; });
  __ESBMC_assert(t <= c, "t");
  return 0;
}
