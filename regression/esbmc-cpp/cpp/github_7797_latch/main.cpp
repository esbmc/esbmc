// #7797: <latch> over ESBMC's pthread model. The worker blocks until the main
// thread has counted the latch down, so the data is visible after the wait.
//
// --no-unwinding-assertions follows github_6319_condvar: the condvar model
// admits an unbounded run of spurious wake-ups, so the wait loop has no finite
// unwind bound. github_7797_latch_fail runs under the same flags and must FAIL,
// which shows the truncated loop still reaches the assertion.
#include <latch>
#include <thread>
#include <cassert>

std::latch gate(2);
int data = 0;

void worker()
{
  gate.wait();
  assert(data == 42);
}

int main()
{
  std::latch counted(2);
  assert(!counted.try_wait());
  counted.count_down();
  assert(!counted.try_wait());
  counted.count_down();
  assert(counted.try_wait());
  counted.wait();

  // count_down(n) must count n, not one.
  std::latch bulk(3);
  bulk.count_down(2);
  assert(!bulk.try_wait());
  bulk.count_down(1);
  assert(bulk.try_wait());

  std::latch arrive(1);
  arrive.arrive_and_wait();
  assert(arrive.try_wait());

  assert(std::latch::max() > 0);

  std::thread t(worker);
  data = 42;
  gate.count_down(2);
  t.join();
  return 0;
}
