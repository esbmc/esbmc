// A user function whose mangled id ends in __ESBMC_old_raw is not
// __ESBMC_old_raw. Suffix-matching the id lifted this call into a snapshot of
// g, silently replacing the clause the user wrote with a different one and
// reporting a verdict on that. The base name is compared instead, so the call
// is left alone and the unmodellable call is reported as such.
#define N 4

int g[N];

void *my__ESBMC_old_raw(void *p);

void f(void)
{
  unsigned j;
  __ESBMC_ensures(__ESBMC_forall(
    &j, !(j < N) || (*(int *)my__ESBMC_old_raw((void *)(&g[j])) == 0)));

  for (unsigned i = 0; i < N; i++)
    g[i] = 0;
}

int main(void)
{
  return 0;
}
