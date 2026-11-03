// #8743: enumerator values clang gives an unsigned type
#include <assert.h>
enum F { X = 0x7fffffff, Xn };
enum G { M = -1, N = 0xffffffffu };
int main()
{
  assert(Xn > 0);
  long n = N;
  assert(n == 4294967295L);
  return 0;
}
