// github #6319: <thread> is modelled over ESBMC's pthread model, so a
// std::thread really interleaves rather than being rejected at parse time.
// One live thread at a time: each extra concurrent thread multiplies the
// interleavings, and CI caps a test at 120s.
#include <thread>
#include <cassert>

int g = 0;
int v = 0;

void w()
{
  g = 42;
}

void *wa(void *p)
{
  *(int *)p = 7;
  return 0;
}

int main()
{
  std::thread e; // a default-constructed thread is not joinable
  assert(!e.joinable());

  std::thread t(w);
  assert(t.joinable());
  t.join();
  assert(!t.joinable());
  assert(g == 42); // the spawned function really ran

  std::thread u(wa, &v); // argument through pthread_create's void* slot
  u.join();
  assert(v == 7);

  return 0;
}
