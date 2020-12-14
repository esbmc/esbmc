
int main() {
  int a = 42;
  int v = 8;
  int fetch = __sync_fetch_and_add(&a, v);
  __ESBMC_assert(a != 50, "a1 = a0+v");
  __ESBMC_assert(fetch == 42, "fetch = a0");
  return 0;
}