#include <stdlib.h>
#include <assert.h>

int main()
{
  int result = atoi("2") * 2;
  assert(result == 4);

  result = atoi("testing");
  assert(result == 0);

  result = atoi("-567");
  assert(result == -567);

  result = atoi("   142");
  assert(result == 142);

  result = atoi("234cde");
  assert(result == 234);

  result = atoi("   +123  ");
  assert(result == 123);

  result = atoi("2147483647");
  assert(result == 2147483647);

  result = atoi("");
  assert(result == 0);

  return 0;
}
