// The index is one-based, so ffsl of the lowest bit is 1 rather than 0. Keeps
// the model honest against an off-by-one that would still satisfy every
// zero-argument case. #183
#include <assert.h>

int ffsl(long);

int main(void)
{
  assert(ffsl(1L) == 0);
  return 0;
}
