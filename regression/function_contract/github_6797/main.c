typedef struct
{
  int m;
  int p;
} T;

void f(T *t, int v)
{
  __ESBMC_requires(__ESBMC_is_fresh(t, sizeof(T)));
  __ESBMC_requires(t->p == 0);
  __ESBMC_requires(v == t->m);
  __ESBMC_assigns(t->p);
  __ESBMC_ensures(t->p == 0);

  /* Unreachable under the requires clauses, so t->p keeps the value they
     gave it. The join must not drop that value on the not-taken arm. */
  if (v != t->m)
  {
    t->p = 1;
  }
}

int main(void)
{
  return 0;
}
