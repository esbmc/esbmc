#include <cassert>

extern "C" void *__VERIFIER_nondet_pointer(void);

int main()
{
  void *p = __VERIFIER_nondet_pointer();
  assert(p != nullptr);
  return 0;
}
