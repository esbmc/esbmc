// As por_local_reinterpret_escape_fail, with the assertion admitting both schedules.
#include <cassert>
#include <pthread.h>

void *worker(void *arg)
{
  assert(*static_cast<int *>(arg) == 0 || *static_cast<int *>(arg) == 1);
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
