/* github_6797_is_fresh_no_precondition_fail:
 * The control for github_6797_is_fresh_initial_version_pass. With no requires
 * establishing t->p, nothing makes the ensures hold on the path where the
 * branch is not taken, so it must still be rejected. Giving the object an
 * initial version must fix the lost precondition, not invent one. */
typedef struct { int m; int p; } T;

void f(T *t, int v)
{
  __ESBMC_requires(__ESBMC_is_fresh(t, sizeof(T)));
  __ESBMC_requires(v == t->m);
  __ESBMC_assigns(t->p);
  __ESBMC_ensures(t->p == 0);
  if (v != t->m) { t->p = 1; }
}
int main(void) { return 0; }
