// github #6319: <condition_variable> over ESBMC's pthread condvar. The waiter
// blocks until the producer publishes, so `data` is visible after the wait.
//
// --no-unwinding-assertions is the repo's convention for this shape (see
// regression/esbmc-unix/01_cond_10): ESBMC's condvar model admits an unbounded
// run of spurious wake-ups, so `while (!ready) wait(lk);` has no finite unwind
// bound. github_6319_condvar_fail runs the same program under the same flags
// and must FAIL, which is what shows the truncated loop still reaches and
// evaluates the assertion.
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cassert>

std::mutex m;
std::condition_variable cv;
bool ready = false;
int data = 0;

void producer()
{
  std::unique_lock<std::mutex> lk(m);
  data = 42;
  ready = true;
  cv.notify_one();
}

int main()
{
  std::thread t(producer);
  {
    std::unique_lock<std::mutex> lk(m);
    while (!ready)
      cv.wait(lk);
    assert(data == 42);
  }
  t.join();
  return 0;
}
