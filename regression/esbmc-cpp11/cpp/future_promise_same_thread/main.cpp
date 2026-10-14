// github #6319: promise/future used within a single thread -- get_future(),
// valid(), and a set_value() that has already happened, so no waiting occurs.
#include <future>
#include <cassert>
int main()
{
  std::future<int> e;
  assert(!e.valid());
  std::promise<int> p;
  std::future<int> f = p.get_future();
  assert(f.valid());
  p.set_value(7); // same thread: no waiting needed
  assert(f.get() == 7);
  return 0;
}
