__ESBMC_logic int bad(int x) { return bad(x) + 1; }
int main() { int v = bad(3); __ESBMC_assert(v == 0, "anything"); return 0; }
