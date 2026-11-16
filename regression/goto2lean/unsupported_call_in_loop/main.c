unsigned nondet_uint(void);
unsigned inc(unsigned v) { return v + 1; }
unsigned find(unsigned n) {
  unsigned i = 0;
  while (i < n) {
    if (i == 7) return i;
    i = inc(i);
    for (unsigned j = 0; j < 3; j++) {
      if (j == i) break;
      if (j == 2) continue;
      i = i + 0;
    }
  }
  return n;
}
int main(void) {
  unsigned n = nondet_uint();
  unsigned r = find(n);
  __ESBMC_assert(r <= n || r == 7, "r");
  return 0;
}
