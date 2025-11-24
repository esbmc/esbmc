#include <assert.h>

int main() {
  int x = nondet_int(), y = nondet_int();

  while(1)
  {
    outer:
      if (x == y) goto success;
      x++;

    inner:
      if (x == 3) goto fail;
      goto outer;

    success:
      assert(x == y); // This should hold when we reach success
      return 0;

   fail:
      assert(1);
  }
  return 0;
}

