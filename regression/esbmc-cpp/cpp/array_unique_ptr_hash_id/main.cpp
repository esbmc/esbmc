#include <array>
#include <cassert>
#include <memory>
#include <thread>

void worker()
{
}

int main()
{
  std::array<std::unique_ptr<int>, 3> slots{};
  assert(slots[0] == nullptr);
  slots[0] = std::unique_ptr<int>(new int(7));
  assert(*slots[0] == 7);

  std::hash<std::thread::id> h;
  std::thread t(worker);
  std::thread::id running = t.get_id();
  assert(h(running) == h(running));
  assert(h(running) != h(std::thread::id()));
  t.join();
  return 0;
}
