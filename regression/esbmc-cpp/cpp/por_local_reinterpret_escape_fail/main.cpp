// main's local escapes only through &reinterpret_cast<int &>, so MPOR must
// see through the cast to key it, or no schedule lets the worker read after
// the write.
#include <cassert>
#include <pthread.h>

void *worker(void *arg)
{
  assert(*static_cast<int *>(arg) == 0);
  return nullptr;
}

int main()
{
  unsigned done = 0;
  pthread_t t;
  int *p = &reinterpret_cast<int &>(done);
  pthread_create(&t, nullptr, worker, p);
  done = 1;
  pthread_join(t, nullptr);
  return 0;
}
