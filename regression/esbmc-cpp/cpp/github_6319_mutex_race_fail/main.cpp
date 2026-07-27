// github #6319, negative direction: with the lock_guard removed the increment
// interleaves and the update is lost, so the mutex in github_6319_mutex is
// doing real work rather than being a no-op.
#include <thread>
#include <cassert>

int x = 0;

void w()
{
  int t = x;
  x = t + 1;
}

int main()
{
  std::thread a(w);
  w();
  a.join();
  assert(x == 2);
  return 0;
}
