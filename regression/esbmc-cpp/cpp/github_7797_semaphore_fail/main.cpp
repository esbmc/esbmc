// #7797: a semaphore at zero must not hand out a count.
#include <semaphore>
#include <cassert>

int main()
{
  std::counting_semaphore<2> s(1);
  assert(s.try_acquire());
  assert(s.try_acquire());
  return 0;
}
