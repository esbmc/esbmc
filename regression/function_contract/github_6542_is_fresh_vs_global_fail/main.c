/* github_6542_is_fresh_vs_global_fail:
 * __ESBMC_is_fresh(p, n) says p addresses a fresh object, so it is separate
 * from everything the caller can reach -- not only from the other pointer
 * arguments. A global the contract names is reachable, so passing its address
 * as the fresh parameter has to be rejected. The enforce harness grants that
 * separation by backing p on its own, so a replace site owes it. */
typedef struct { int x; } P;
P g;
/* p 用 is_fresh 声明了 extent, 但全局 g 没有被 is_fresh 提到 */
void callee(P *p) {
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(P)));
  __ESBMC_assigns(p->x);
  __ESBMC_ensures(p->x == 1);
  __ESBMC_ensures(g.x == __ESBMC_old(g.x));
  p->x = 1;
}
int main(void){ g.x = 5; callee(&g);
  __ESBMC_assert(g.x == 5, "false: p aliases g"); return 0; }
