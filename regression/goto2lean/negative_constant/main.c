int nondet_int(void);
enum { NEG = -3 };
int main(void) {
  int x = nondet_int();
  int k = NEG;
  switch (x) { case -1: k = 1; break; default: break; }
  __ESBMC_assert(k != 0, "k");
  return 0;
}
