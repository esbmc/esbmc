// A namespace-scope std::mutex pulled the host <mutex> in, which reached the
// OM <stdexcept>/<ios> through <system_error> and failed to parse against the
// host basic_ios<char> (github #3387).
#include <mutex>
#include <cassert>

std::mutex mtx;

int main()
{
  mtx.lock();
  int i = 0;
  mtx.unlock();
  assert(i == 0);
  return 0;
}
