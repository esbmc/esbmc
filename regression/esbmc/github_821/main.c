// #821: without simplification, an eight-int shuffle stored into a vector of
// 32 chars is its bytes.
#include <assert.h>

typedef int v4si __attribute__((__vector_size__(16)));
typedef char v32c __attribute__((__vector_size__(32)));

int main()
{
  v4si v1 = (v4si){5, 6, 7, 8};
  v4si v2 = (v4si){10, 11, 13, 15};
  v32c r;
  r = __builtin_shufflevector(v1, v2, 0, 1, 2, 3, 4, 5, 6, 7);
  assert(r[0] == 5 && r[1] == 0 && r[4] == 6 && r[16] == 10);
  return 0;
}
