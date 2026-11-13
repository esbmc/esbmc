int nondet_int(void);
int main(void) { int x = nondet_int(); __ESBMC_assert((x & 1) <= 1, "low bit"); return 0; }
