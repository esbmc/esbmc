/* An index that the loop itself advances is ambiguous: `global[i]` excuses the
 * element at the value `i` holds when the check runs, which is the value
 * *after* the iteration, while the write used the value before it. So this
 * correct body is reported, and reported one element late -- the write is to
 * global[0] and the complaint names global[1].
 *
 * Naming a fixed index (loop_assigns_array_elem_pass) avoids this; excusing a
 * moving one needs the index snapshotted at entry to the iteration. */
int global[8];

int main()
{
  int i = 0;
  global[7] = 42;

  __ESBMC_loop_assigns(i, global[i]);
  __ESBMC_loop_invariant(i >= 0 && i <= 7);
  while (i < 7)
  {
    global[i] = i;
    i++;
  }

  __ESBMC_assert(global[7] == 42, "element outside the loop's writes is held");
  return 0;
}
