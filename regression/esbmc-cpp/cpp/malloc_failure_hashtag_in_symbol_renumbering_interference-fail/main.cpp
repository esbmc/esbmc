#include <assert.h>
extern void *malloc(unsigned long);
float _M_array = 0;

extern "C"
{
  void foo_extern_c()
  {
    long result = (&_M_array - &_M_array + 1) * 4;
    assert(result == 4);
    int a = *(int *)(malloc(result));
  }
}

void foo()
{
  long result = (&_M_array - &_M_array + 1) * 4;
  assert(result == 4);
  int a = *(int *)(malloc(
    result)); // fails because `--force-malloc-success` is not enabled.
}

int main(void)
{
  foo(); // this fails, because the id contains `#` confusing renumbering/renaming/symbol logic in esbmc.
  // foo_extern_c(); // this works, because the id does not contain `#` in C mode.
}
