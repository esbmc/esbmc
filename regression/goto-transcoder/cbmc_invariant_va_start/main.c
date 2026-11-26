/* A va_start call under && cannot be lifted, so the invariant is dropped. */
int a[4];

void fill(int n, ...)
{
  __builtin_va_list ap;
  for (unsigned i = 0; i < 4; i++)
    __CPROVER_loop_invariant(i <= 4 && (__builtin_va_start(ap, n), 1))
  {
    a[i] = 2;
  }
}

int main(void)
{
  fill(0);
  __CPROVER_assert(a[3] == 2, "loop wrote a[3]");
  return 0;
}
