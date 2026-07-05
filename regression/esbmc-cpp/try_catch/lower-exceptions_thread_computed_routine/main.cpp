// Same computed (function-pointer) thread start routine as the _fail companion,
// but here the routine catches its own exception, so nothing escapes the thread
// and std::terminate is not called — VERIFICATION SUCCESSFUL. Confirms the
// over-approximation for unresolved start routines does not spuriously terminate
// a thread that handles its exception locally.
#include <pthread.h>
#include <cassert>

struct E
{
};

void *worker(void *)
{
  try
  {
    throw E();
  }
  catch (E &)
  {
    return 0;
  }
  assert(0);
  return 0;
}

int main()
{
  pthread_t t;
  void *(*fp)(void *) = worker;
  pthread_create(&t, 0, fp, 0);
  pthread_join(t, 0);
  return 0;
}
