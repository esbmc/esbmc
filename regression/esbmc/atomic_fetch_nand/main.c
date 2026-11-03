#include <assert.h>

int main()
{
  int i = 6;
  assert(__atomic_fetch_nand(&i, 3, __ATOMIC_SEQ_CST) == 6 && i == ~2);
  long l = 0xff;
  assert(__sync_fetch_and_nand(&l, 0x0f) == 0xff && l == ~0x0fL);
  return 0;
}
