// The last shared owner must really free: a raw pointer outliving it is a
// use-after-free, and a model that never freed would report SUCCESSFUL.
#include <memory>
#include <cassert>

int main()
{
  int *raw;
  {
    std::shared_ptr<int> a(new int(7));
    raw = a.get();
  }
  assert(*raw == 7);
  return 0;
}
