/* The invariant claims a global is dynamic, so its base case fails. */
int a[4];

int main(void)
{
  int *p = a;
  for (unsigned i = 0; i < 4; i++)
    __CPROVER_loop_invariant(i <= 4 && __CPROVER_DYNAMIC_OBJECT(p))
  {
    a[i] = 0;
  }
  return 0;
}
