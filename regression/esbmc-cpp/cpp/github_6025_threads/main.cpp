#include <atomic>
#include <pthread.h>
#include <cassert>

std::atomic<int> counter{0};

void *worker(void *)
{
  counter.fetch_add(1);
  return nullptr;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, nullptr, worker, nullptr);
  pthread_create(&t2, nullptr, worker, nullptr);
  pthread_join(t1, nullptr);
  pthread_join(t2, nullptr);
  assert(counter.load() == 2); // atomicity: no lost updates
  return 0;
}
