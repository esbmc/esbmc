/* An inner loop with its own head keeps its back edge: the outer contract is
 * not checked there (#11085). */
int main(void)
{
  unsigned i = 0, j = 0;
  while (i < 5)
    __CPROVER_loop_invariant(i <= 4 && (j == 0 || j == 3))
    {
      j = 0;
      while (j < 3)
        j++;
      i++;
    }
  __CPROVER_assert(i == 5, "i is 5");
  return 0;
}
