int main(void) { int a[2] = {1, 2}; int *p = a; __ESBMC_assert(p[1] == 2, "x"); return 0; }
