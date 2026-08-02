// Including <functional> must parse under -std=c++11: the bundled std::invoke
// model once used decltype(auto), a C++14 construct, breaking every such TU.
#include <functional>

int main()
{
  return 0;
}
