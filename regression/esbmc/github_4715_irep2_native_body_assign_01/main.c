// Exercises the --irep2-native-body IREP2-native assign dispatcher (W1-loc
// spike Phase C, esbmc/esbmc#4715): the decl-free bodies of set()/set2()/
// store() are all side-effect-free, non-atomic assignments, so goto_convert
// consumes each code_assign2t natively (stored directly, no legacy round-trip)
// while main() (locals + asserts) falls back to goto_convert_rec. The verdict
// and GOTO must match a run without the flag.
#include <assert.h>

void set(int *p, int v)
{
  *p = v;
}

void set2(int *a, int x)
{
  a[0] = x;
  a[1] = x + 1;
}

int g;

void store(int x)
{
  g = x;
}

int main(void)
{
  int z = 0;
  set(&z, 7);
  assert(z == 7);

  int arr[2];
  set2(arr, 4);
  assert(arr[0] == 4 && arr[1] == 5);

  store(9);
  assert(g == 9);
  return 0;
}
