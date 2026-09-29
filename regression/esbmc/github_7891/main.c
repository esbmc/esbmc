// #7891: clang's SSE2 headers typedef vectors of __bf16 at file scope.
#include <immintrin.h>

int main()
{
  return 0;
}
