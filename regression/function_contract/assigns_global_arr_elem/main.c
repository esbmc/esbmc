/*
 * Phase 2B (array element assigns) recognises an array element only as
 * `add2t(pointer_symbol, index)`, the pointer arithmetic a *parameter* decays
 * to. A global array element is `index2t(symbol, index)`, which reached
 * neither that nor the per-field machinery, so the frame enforcer snapshotted
 * `global` as a whole and asserted `global == snapshot` -- which the write the
 * clause itself names falsifies.
 *
 * Pinned as a KNOWNBUG since the frame rule landed (#3702) and lifted here:
 * the indices a clause names are recorded, and every other element is held to
 * its pre-state instead of the array as a whole.
 */
int global[10];

void write_global_elem(int i, int v)
{
  __ESBMC_requires(i >= 0 && i < 10);
  __ESBMC_assigns(global[i]);
  __ESBMC_ensures(1);
  global[i] = v; /* only touches global[i], which is in assigns */
}

int main()
{
  return 0;
}
