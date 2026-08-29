struct S
{
  int buf[4];
  unsigned n;
};

int main()
{
  struct S s;
  struct S *p = &s;

  p->n = nondet_uint();
  __ESBMC_assume(p->n <= 4);

  /* n == 4 writes onto `n' itself, which stays inside the enclosing struct:
     only the array's own bound rejects it. */
  p->buf[p->n] = 1;
  return 0;
}
