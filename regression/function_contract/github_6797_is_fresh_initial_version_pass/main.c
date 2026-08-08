/* github_6797_is_fresh_initial_version_pass:
 * __ESBMC_is_fresh allocated the object without ever writing through the
 * pointer, so it had no level-2 version. symex skips the phi at a join for
 * such an object, leaving the branch's version current on both sides, so the
 * conditional write below escaped its guard and the precondition on t->p was
 * lost. The branch cannot be taken under v == t->m, so t->p keeps the value
 * the requires clause gave it and the ensures holds. */
typedef struct { int m; int p; } T;

void f(T *t, int v)
{
  __ESBMC_requires(__ESBMC_is_fresh(t, sizeof(T)));
  __ESBMC_requires(t->p == 0);
  __ESBMC_requires(v == t->m);
  __ESBMC_assigns(t->p);
  __ESBMC_ensures(t->p == 0);
  if (v != t->m) { t->p = 1; }
}
int main(void) { return 0; }
