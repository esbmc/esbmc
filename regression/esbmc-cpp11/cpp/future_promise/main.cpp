// github #6319: the promise/future half of <future>, over ESBMC's pthread
// model. The producer thread publishes under the state's mutex and broadcasts;
// future::get() waits on the condvar, so the value is observed only after the
// producer sets it -- a model that did not synchronise would read the
// uninitialised state and fail this assertion.
//
// --no-unwinding-assertions follows the repo convention for a condvar wait
// (see regression/esbmc-unix/01_cond_10): ESBMC's condvar model admits an
// unbounded run of spurious wake-ups, so the wait loop has no finite unwind
// bound. future_promise_fail runs the same program under the same flags and
// must FAIL, which is what shows the assertion is still reached.
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
  assert(v == 42);
  t.join();
  return 0;
}
