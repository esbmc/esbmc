/* The loop rule cannot use the store form: it runs in ASSUME mode, and
 * Bitwuzla -- the default solver -- rejects equality over constant arrays
 * ("not fully supported yet"), aborting the run rather than answering. So a
 * loop over a global past the element-wise cap keeps the whole-array
 * assumption and the solver fails outright.
 *
 * Below the cap this shape works (loop_assigns_array_elem_pass), and the same
 * program answers under --z3. */
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
