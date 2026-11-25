unsigned nondet_uint(void);
int main(void) {
  unsigned i = 0;
  _Bool b = __ESBMC_forall(&i, i <= 4294967295u);
  return 0;
}
