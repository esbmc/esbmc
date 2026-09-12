/* std::thread's operational model hands pthread_create its own `f` parameter,
 * which is a function-pointer variable rather than a direct &worker. Exception
 * lowering scanned every function body, so that unreachable constructor made it
 * report "a thread with an unresolved start routine" and decline any program
 * that merely included <thread> and used exceptions -- no thread required.
 * Reported against immer, whose headers pull <thread> in transitively (#7644).
 * See github_7644_thread_fail for a program that does start a thread, which
 * must still be declined. */
#include <thread>

int main()
{
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
