// Exercises the cpp_new round-trip under --irep2-bodies (V.4.4, esbmc#4715).
//
// Two distinct migrate paths were dropping data before the fix:
//   * new T[n]  — the array size lives in the "size" field at frontend time
//                 (mirrored into "#size" only later); migrating before that
//                 mirror dropped the whole size operand, leaving (T*)(nil)*sz.
//   * new T(v)  — the initializer lives in the "initializer" sub, which was
//                 never carried through the round-trip, so `new int(7)` was
//                 silently lowered to `new int` (default-init to 0).
int main()
{
  unsigned n = 4;
  int *a = new int[n];
  for (unsigned i = 0; i < n; ++i)
    a[i] = (int)i;
  __ESBMC_assert(a[3] == 3, "array element write/read survives round-trip");
  delete[] a;

  int *p = new int(7);
  __ESBMC_assert(*p == 7, "scalar new initializer survives round-trip");
  delete p;

  return 0;
}
