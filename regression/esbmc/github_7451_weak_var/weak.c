// #7451: a strong definition in a later translation unit overrides the weak one
#include <assert.h>

int global __attribute__((weak)) = 999;

int main(void)
{
  assert(global == EXPECTED);
  return 0;
}
