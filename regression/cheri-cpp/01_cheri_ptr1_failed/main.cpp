#include <stdio.h>
#include <string.h>
#include <cheri/cheric.h>

char *buffer = "hello";
char *secret = "secret";

class t1 {
public:
  int i;

  t1(): i(0)
  {
  }

  int foo();
};

int t1::foo()
{
  char *__capability cap_ptr = cheri_ptr(buffer, 6);
  /* Overflow buffer, leaking the secret in a traditional system */
  for(int i = 0; i < strlen(buffer) + 7; i++)
  {
    printf("ptr[%d] = '%c'\n", i, cap_ptr[i]);
  }
  return 0;
}

int main(int argc, char **argv)
{
  t1 instance1;
  t1 &r = instance1;
  assert(r.i == 0); // pass

  r.foo(); // access the CHERI API function from a lvalue reference

  r.i = 1;
  assert(r.i == 1); // fail

  return 0;
}
