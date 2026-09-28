// github #6319: the predicate overload of condition_variable::wait, which is a
// member template of a non-template class -- unlike a free-function template,
// its body IS converted in an OM header, so the loop really runs. A lambda
// predicate works because it is used as a functor; note that a lambda is not
// usable as a std::thread callable, where the conversion to a function pointer
// is needed and is currently bodyless in the frontend.
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
  cv.notify_all();
}

int main()
{
  std::thread t(producer);
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [] { return ready; });
    assert(data == 42);
  }
  t.join();
  return 0;
}
