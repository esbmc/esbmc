// github #6319: [thread.thread.destr]/1 -- destroying a still-joinable thread
// calls terminate(); the model reports it as a property violation.
#include <thread>

int g = 0;

void w()
{
  g = 42;
}

int main()
{
  std::thread t(w);
  return 0;
}
