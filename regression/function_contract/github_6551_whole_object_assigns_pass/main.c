/* github_6551_whole_object_assigns_pass:
 * __ESBMC_assigns(*r) covers the whole pointee, so a parameter aliasing r is
 * writable through it and the frame check must exempt every field, not only
 * the p->field target shape. */
typedef struct { int x; } S;
void f(S *r, S *b) {
  __ESBMC_requires(r != 0);
  __ESBMC_requires(b != 0);
  /* Guarded: b has to stay free to alias r for the frame check to be tested. */
  __ESBMC_requires(r == 0 || __ESBMC_is_fresh(r, sizeof(S)));
  __ESBMC_requires(b == 0 || __ESBMC_is_fresh(b, sizeof(S)));
  __ESBMC_assigns(*r);
  __ESBMC_ensures(r->x == 1);
  r->x = 1;
}
int main(void){ return 0; }
