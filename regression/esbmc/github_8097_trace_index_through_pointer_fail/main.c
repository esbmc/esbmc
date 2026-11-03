/* #8097: building the counterexample asked the solver for the index
 * s->a[i], which still holds a dereference, and aborted. */
typedef struct
{
  unsigned a[2];
  unsigned b[2];
} S;

unsigned nondet_uint(void);

void f(S *s)
{
  for (int i = 0; i < 2; i++)
  {
    s->a[i] = nondet_uint() % 2;
    s->b[s->a[i]] = i + 1;
  }
}

int main(void)
{
  S s = {{0, 0}, {0, 0}};
  f(&s);
  __ESBMC_assert(s.b[0] + s.b[1] != 3, "b holds 1 and 2");
  return 0;
}
