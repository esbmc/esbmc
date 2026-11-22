void g(unsigned);
void h(void) { }
int main(void) {
  g(1);
  h();
  h();
  __ESBMC_assert(1, "trivial");
  __ESBMC_assume(1);
  return 0;
}
