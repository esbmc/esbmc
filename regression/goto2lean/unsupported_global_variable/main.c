unsigned g;
int main(void) { g = 1; __ESBMC_assert(g == 1, "g"); return 0; }
