#include <assert.h>

// Companion failing test: a negative-amount shift is present (declined by the
// fold guard), while an in-range shift still folds. The assertion is wrong on
// purpose so the expected verdict is FAILED.
int main()
{
  int amt = 2 - 3;           // -1
  volatile int r = 1 << amt; // out-of-range amount: no internal UB fold
  (void)r;

  assert((4 << 1) == 9); // 4 << 1 == 8, so this must fail
  return 0;
}
