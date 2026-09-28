#include <cassert>
#include <pthread.h>
#include <shared_mutex>

std::shared_mutex m;
int counter = 0;

void *bump(void *arg)
{
  m.lock();
  int local = counter;
  counter = local + 1;
  m.unlock();
  return 0;
}

int main()
{
  pthread_t t;
  pthread_create(&t, 0, bump, 0);
  bump(0);
  pthread_join(t, 0);

  std::shared_lock<std::shared_mutex> reader(m);
  assert(reader.owns_lock());
  assert(counter == 2);
  return 0;
}
