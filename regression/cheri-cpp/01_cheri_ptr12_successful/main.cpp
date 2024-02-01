// TC description:
//  This is a modified version of 01_cheri_ptr12_successful with a non-trivial destructor

#include <stdio.h>
#include <string.h>
#include <cheri/cheric.h>

char *buffer = "hello";
char *secret = "secret";

int ii = 1;

class t2
{
public:
  int i;

  t2() : i(2)
  {
  }

  ~t2() { ii = 0; }

  int foo();
};

int t2::foo()
{
  // zero-length capability is allowed
  char *__capability cap_ptr = cheri_ptr(buffer, 3);
  assert(cap_ptr[2] == 'l');
  return 0;
}

int main(int argc, char **argv)
{
  t2 *p = new t2;
  assert(p->i == 2);

  t2.foo();

  delete p;
  assert(ii == 0);

  return 0;
}
