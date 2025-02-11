// TC description:
//  This is a modified version of 01_cheri_clear_tag_leak_failed with non-trivial class ctor

#include <cheri/cheric.h>
#include <assert.h>

char *buffer = "hello";

class t1
{
public:
  int i;

  t1()
  {
    i = 1;
  }

  int foo();
};



int t1::foo() {
    /* create valid capability: */
    char *__capability valid_cap = cheri_ptr(buffer, 6);
    /* create a copy with tag cleared: */
    char *__capability cleared_cap = cheri_cleartag(valid_cap);

    /* this does not fail, leaking the capability base */
    assert(cheri_getbase(cleared_cap));

    return 0;
}

int main(int argc, char **argv) {

  t1 instance1;
  /* test data member in object */
  assert(instance1.i == 1);

  instance1.foo();

  return 0;
}
