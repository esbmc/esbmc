/* The loop writes x through havoc_object, so k-induction must not take it as
 * loop-invariant: x reaches 100 after 100 iterations. */
void __CPROVER_havoc_object(void *);

int main(void)
{
  int x = 0;
  int *px = &x;
  while (1)
  {
    __CPROVER_assert(x < 100, "x stays below 100");
    int old = x;
    __CPROVER_havoc_object(px);
    __CPROVER_assume(x == old + 1);
  }
  return 0;
}
