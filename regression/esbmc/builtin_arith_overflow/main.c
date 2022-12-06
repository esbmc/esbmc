#include <assert.h>

int main() {
  int x1 = 0x7fffffff;
  int y1;
  _Bool c1 = __builtin_sadd_overflow(2, x1, &y1);
  assert(c1);
  
  unsigned int x2 = 0x7fffffff;
  unsigned int y2;
  _Bool c2 = __builtin_uadd_overflow(2, x2, &y2);
  assert(!c2);
  
  int x3 = -1;
  int y3;
  _Bool c3 = __builtin_ssub_overflow(0x7fffffff, x3, &y3);
  assert(c3);
  
  unsigned int x4 = 0;
  unsigned int y4;
  _Bool c4 = __builtin_usub_overflow(2, x4, &y4);
  assert(!c4);

  int x5 = 0x7fffffff;
  int y5;
  _Bool c5 = __builtin_smul_overflow(0x7fffffff, x5, &y5);
  assert(c5);
  
  unsigned int x6 = 0x7fffffff;
  unsigned int y6;
  _Bool c6 = __builtin_umul_overflow(2, x6, &y6);
  assert(!c6);
}
