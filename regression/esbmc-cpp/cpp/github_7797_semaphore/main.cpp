// #7797: <semaphore> over ESBMC's pthread model. The worker blocks on acquire
// until the main thread releases, so the data is visible afterwards.
//
// --no-unwinding-assertions as in github_6319_condvar; the _fail counterpart
// runs under the same flags.
#include <semaphore>
#include <thread>
#include <cassert>

std::binary_semaphore handoff(0);
int data = 0;

void worker()
{
  handoff.acquire();
  assert(data == 7);
}

int main()
{
  std::binary_semaphore b(0);
  assert(!b.try_acquire());
  b.release();
  assert(b.try_acquire());
  assert(!b.try_acquire());

  std::counting_semaphore<4> c(2);
  assert(std::counting_semaphore<4>::max() == 4);
  assert(c.try_acquire());
  assert(c.try_acquire());
  assert(!c.try_acquire());
  // release(n) must return n counts, not one.
  c.release(2);
  assert(c.try_acquire());
  assert(c.try_acquire());
  assert(!c.try_acquire());

  std::thread t(worker);
  data = 7;
  handoff.release();
  t.join();
  return 0;
}
