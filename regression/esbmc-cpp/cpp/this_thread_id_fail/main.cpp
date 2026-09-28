#include <cassert>
#include <thread>

std::thread::id worker_id;

void worker()
{
  worker_id = std::this_thread::get_id();
  std::this_thread::yield();
}

int main()
{
  std::thread t(worker);
  // [thread.thread.id]/1: distinct running threads compare unequal.
  assert(std::this_thread::get_id() == t.get_id());
  t.join();
  return 0;
}
