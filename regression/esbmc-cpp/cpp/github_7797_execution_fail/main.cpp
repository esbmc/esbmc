// #7797: a plain type is not an execution policy.
#include <execution>
#include <cassert>

int main()
{
  assert(std::is_execution_policy<int>::value);
  return 0;
}
