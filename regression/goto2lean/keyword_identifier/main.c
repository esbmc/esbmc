unsigned nondet_uint(void);
int main(void){ unsigned _ = nondet_uint(); __ESBMC_assert(_ == _, "k"); return 0; }
