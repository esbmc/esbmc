// github #6319, negative direction: github_6319_condvar with the assertion
// negated, run under identical flags. It must FAIL, which is what shows the
// truncated wait loop still reaches and evaluates the assertion -- otherwise
// --no-unwinding-assertions could be hiding a vacuous SUCCESSFUL there.
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
    assert(data == 0);
  }
  t.join();
  return 0;
}
