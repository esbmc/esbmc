/* What withholding the frame hypothesis costs. Past the element-wise cap the
 * loop rule has no sound way to say "every element but global[3] is
 * unchanged", so it says nothing about the array at all, and this correct
 * program can no longer be proved.
 *
 * That is the deliberate trade: the fallback it replaces was an assumption the
 * clause contradicts, which proved assertions that were false
 * (loop_assigns_large_array_no_false_proof). Below the cap this shape works
 * (loop_assigns_array_elem_pass); lifting the cap needs the spared element as
 * a quantifier rather than an assertion per element. */
int global[512];

int main()
{
  int i = 0;
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
