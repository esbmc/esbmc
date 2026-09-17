// Counterpart of function_address_signal_sentinels: an integer no libc reserves
// as a handler sentinel may still convert to a function's address.
#include <assert.h>

void handler(int sig)
{
  (void)sig;
}

int main(void)
{
  void (*f)(int) = handler;
  assert(f != (void (*)(int))16);
  return 0;
}
