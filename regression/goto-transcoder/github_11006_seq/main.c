/* #11006: the first loop's invariant must not attach to the second loop. */
int main()
{
  unsigned i = 0, j = 0;
  while (i < 3)
    __CPROVER_loop_invariant(i <= 3)
  {
    i++;
  }
  while (j < 5)
  {
    j++;
  }
  __CPROVER_assert(j == 5, "second loop runs to the end");
  return 0;
}
