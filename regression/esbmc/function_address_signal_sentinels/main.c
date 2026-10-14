// C11 7.14p3: SIG_ERR and SIG_IGN never equal a function's address.
#include <assert.h>

void handler(int sig)
{
  (void)sig;
}

int main(void)
{
  void (*f)(int) = handler;
  assert(f != (void (*)(int))1);
  assert(f != (void (*)(int))-1);
  return 0;
}
