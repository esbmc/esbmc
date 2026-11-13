int nondet_int(void);
int main(void) {
  int x = nondet_int();
  long y = x;
  _Bool b = x;
  int z = b;
  __ESBMC_assert(y == x && z <= 1 && x * x >= 0, "casts are identities");
  return 0;
}
