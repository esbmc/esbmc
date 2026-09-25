// github #6319: single-threaded API surface of the <mutex> model -- lock,
// try_lock, unlock, lock_guard and unique_lock's ownership tracking. The
// mutual-exclusion property is checked in github_6319_mutex.
#include <mutex>
#include <cassert>

int main()
{
  std::mutex n;
  n.lock();
  n.unlock();

  {
    std::lock_guard<std::mutex> g(n);
  }

  {
    std::unique_lock<std::mutex> u(n);
    assert(u.owns_lock());
    u.unlock();
    assert(!u.owns_lock());
    u.lock();
    assert(u.owns_lock());
  }

  {
    std::unique_lock<std::mutex> u(n, std::defer_lock);
    assert(!u.owns_lock());
    assert(u.try_lock());
    u.unlock();
  }

  assert(n.try_lock());
  n.unlock();

  return 0;
}
