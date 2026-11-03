#include <mutex>
#include <pthread.h>

// #8189: t1 can take the lock once main releases it, while t2 writes.
int global;
std::mutex m;
pthread_t id1, id2;

void *t1(void *)
{
  m.lock();
  global++;
  m.unlock();
  return nullptr;
}

void *t2(void *)
{
  global++;
  return nullptr;
}

int main()
{
  pthread_create(&id1, nullptr, t1, nullptr);
  m.lock();
  pthread_create(&id2, nullptr, t2, nullptr);
  m.unlock();
  pthread_join(id2, nullptr);
  return 0;
}
