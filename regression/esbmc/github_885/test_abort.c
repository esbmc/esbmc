//test_abort.c:
#include "test_abort.h"

void func1(int y);
void func1(int y)
{
  func2(y);
}
