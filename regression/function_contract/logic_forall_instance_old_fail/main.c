__ESBMC_logic int id(long j) { return j; }
static long kk;
void f(char *d, int n) {
  __ESBMC_requires(n > 0 && __ESBMC_is_fresh(d, n));
  int u = 0;
  __ESBMC_loop_invariant(0 <= u && u <= n &&
    __ESBMC_forall(&kk, !(0 <= kk && kk < n) || d[kk] == (kk < u ? (char)id(kk) : 0)));
  while (u < n) { d[u] = (char)u; u++; }
}
int main() { char b[4]; f(b, 4); }
