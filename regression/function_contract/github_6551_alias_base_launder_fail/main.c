/* github_6551_alias_base_launder_fail:
 * The frame exemption guards a value snapshotted before the call, so its
 * aliasing test reads the assigns-target base in the pre-state. Otherwise a
 * callee free to assign a global base could point it at the checked object
 * on the way out and launder a write that was outside the frame. */
typedef struct { int coeffs[4]; } P;
extern P *g;
void f(P *b) {
  __ESBMC_requires(b != 0);
  __ESBMC_requires(__ESBMC_is_fresh(g, sizeof(P)));
  __ESBMC_assigns(g);
  __ESBMC_assigns(g->coeffs);
  __ESBMC_ensures(1);
  b->coeffs[0] = 1;   /* out of frame when b != g on entry */
  g = b;              /* used to launder it */
}
int main(void){ return 0; }
