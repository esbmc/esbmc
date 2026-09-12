/* Counterpart of github_7644: a genuine std::thread makes the constructor
 * reachable, so the unresolved start routine is real and the program must still
 * be declined. This is what keeps the reachability gate from being widened into
 * skipping the check altogether. */
#include <thread>

static int x = 0;

static void worker()
{
  x = 1;
}

int main()
{
  std::thread t(worker);
  t.join();
  try
  {
    throw 1;
  }
  catch (int)
  {
    return 0;
  }
  return 1;
}
