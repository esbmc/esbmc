#include <atomic>
#include <pthread.h>
#include <cassert>

std::atomic<int> counter{0};

void *worker(void *)
{
  // load and store are individually atomic but the increment is not
  // indivisible: a lost update is possible.
  counter.store(counter.load() + 1);
  return nullptr;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, nullptr, worker, nullptr);
  pthread_create(&t2, nullptr, worker, nullptr);
  pthread_join(t1, nullptr);
  pthread_join(t2, nullptr);
  assert(counter.load() == 2); // must fail: lost update reachable
  return 0;
}
