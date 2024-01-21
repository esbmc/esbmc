// TC description :
//  This is a modified version of 01_cheri_clear_tag_failed with trivial class ctor
#include <cheri/cheric.h>
#include <assert.h>

char *buffer = "hello";
char *secret = "secret";

class t1
{
public:
  t1() { }

  int foo();
};

int t1::foo() {
    /* create valid capability: */
    char *__capability valid_cap = cheri_ptr(buffer, 6);
    /* create a copy with tag cleared: */
    char *__capability cleared_cap = cheri_cleartag(valid_cap);

    /* now try to dereference the capability with cleared tag */
    assert(cleared_cap[0]);  // this fails with tag violation exception

    return 0;
}

int main(int argc, char **argv) {
  t1 instance1;
  instance1.foo();

  return 0;
}
