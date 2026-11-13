unsigned f(unsigned n) { return n == 0 ? 0 : f(n - 1); }
int main(void) { __ESBMC_assert(f(3) == 0, "f"); return 0; }
