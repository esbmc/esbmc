// _fail sibling of github_4715_irep2_native_body_assign_01 (W1-loc spike Phase
// C, esbmc/esbmc#4715). Pins that consuming the decl-free assignment bodies
// natively does not corrupt the assigned values or suppress bug detection: the
// value stored by the native code_assign2t in set() must reach main(), so the
// wrong-value assertion is a reachable violation reported as VERIFICATION
// FAILED under --irep2-native-body.
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
  assert(z == 8);

  int arr[2];
  set2(arr, 4);
  assert(arr[0] == 4 && arr[1] == 5);

  store(9);
  assert(g == 9);
  return 0;
}
