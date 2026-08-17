/* The loop frame rule runs the same element-wise machinery in ASSUME mode, a
 * path no loop-invariant test reached: every element but the one the clause
 * names is assumed equal to its snapshot across the havoc. */
int global[8];

int main()
{
  int i = 0;
  global[3] = 7;
  global[5] = 9;

  __ESBMC_loop_assigns(i, global[3]);
  __ESBMC_loop_invariant(i >= 0 && i <= 4);
  while (i < 4)
  {
    global[3] = i;
    i++;
  }

  __ESBMC_assert(global[5] == 9, "an element the clause does not name is held");
  return 0;
}
