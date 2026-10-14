// github #6319: <mutex> is modelled over ESBMC's pthread mutex, so it really
// excludes concurrent threads rather than being rejected at parse time.
// Without the lock_guard the increment interleaves and x can end up 1 --
// that is github_6319_mutex_race_fail, run under the same context bound so the
// pair differs only in the lock. The bound keeps this under the 120s CI cap
// (unbounded it enumerates ~31k interleavings); the single-threaded API surface
// is exercised separately in github_6319_mutex_api.
#include <mutex>
#include <thread>
#include <cassert>

std::mutex m;
int x = 0;

void w()
{
  std::lock_guard<std::mutex> g(m);
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
