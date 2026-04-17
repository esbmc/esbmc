/*
 * assigns_global_arr_elem_knownbug:
 *   Phase 2B (array element assigns) does not apply to global arrays.
 *   The function only writes global[i] which IS listed in __ESBMC_assigns,
 *   but ESBMC's frame-enforcer snapshots 'global' as a whole-array scalar
 *   and emits ASSERT(global == snapshot) — which always fails on any write.
 *
 *   Known limitation: global array element assigns are unsupported;
 *   only pointer-param array elements (Phase 2B) are supported.
 *
 *   Expected (correct): VERIFICATION SUCCESSFUL
 *   Current (bug):      VERIFICATION FAILED — false positive
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
