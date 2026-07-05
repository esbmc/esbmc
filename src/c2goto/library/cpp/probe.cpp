// Minimal C++ operational-model probe used to exercise compiling C++ sources
// into the embedded clib (esbmc/esbmc#5298). Bodies are emitted by the C++
// frontend and must survive serialization + goto_convert re-lowering.

extern "C" int __esbmc_probe_sum(int n)
{
  int total = 0;
  for (int i = 0; i < n; i++)
    total += i;
  return total;
}

extern "C" int __esbmc_probe_range()
{
  int a[3] = {1, 2, 3};
  int total = 0;
  for (int x : a)
    total += x;
  return total;
}
