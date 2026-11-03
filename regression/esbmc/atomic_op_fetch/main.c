#include <assert.h>

int main()
{
  int i = 5;
  assert(__atomic_add_fetch(&i, 3, __ATOMIC_SEQ_CST) == 8 && i == 8);
  assert(__atomic_sub_fetch(&i, 10, __ATOMIC_SEQ_CST) == -2 && i == -2);
  assert(__atomic_and_fetch(&i, 7, __ATOMIC_SEQ_CST) == 6 && i == 6);
  assert(__atomic_or_fetch(&i, 9, __ATOMIC_SEQ_CST) == 15 && i == 15);
  assert(__atomic_xor_fetch(&i, 5, __ATOMIC_SEQ_CST) == 10 && i == 10);
  assert(__atomic_nand_fetch(&i, 6, __ATOMIC_SEQ_CST) == ~2 && i == ~2);

  int j = 5;
  assert(__sync_add_and_fetch(&j, 3) == 8 && j == 8);
  assert(__sync_sub_and_fetch(&j, 10) == -2 && j == -2);
  assert(__sync_and_and_fetch(&j, 7) == 6 && j == 6);
  assert(__sync_or_and_fetch(&j, 9) == 15 && j == 15);
  assert(__sync_xor_and_fetch(&j, 5) == 10 && j == 10);
  assert(__sync_nand_and_fetch(&j, 6) == ~2 && j == ~2);

  unsigned char c = 250;
  unsigned char r = __atomic_add_fetch(&c, 10, __ATOMIC_SEQ_CST);
  assert(r == 4 && c == 4);

  long l = 1;
  assert(__atomic_add_fetch(&l, 1L << 40, __ATOMIC_RELAXED) == (1L << 40) + 1);
  return 0;
}
