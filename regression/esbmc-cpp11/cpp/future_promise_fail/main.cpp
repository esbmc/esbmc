// github #6319, negative direction: future_promise with the expected value
// changed, under identical flags. It must FAIL, so the positive test is not
// vacuously discharged by the truncated wait loop.
#include <future>
#include <thread>
#include <cassert>
std::promise<int> p;
void producer()
{
  p.set_value(42);
}
int main()
{
  std::future<int> f = p.get_future();
  assert(f.valid());
  std::thread t(producer);
  int v = f.get();
  assert(v == 43);
  t.join();
  return 0;
}
