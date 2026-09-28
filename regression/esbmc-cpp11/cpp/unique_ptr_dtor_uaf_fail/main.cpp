// Without a unique_ptr destructor the model never freed, so a genuine
// use-after-free through a raw pointer outliving the unique_ptr was reported
// SUCCESSFUL -- the unsound half of the same defect.
#include <memory>
#include <cassert>

int main()
{
  int *raw;
  {
    std::unique_ptr<int> p(new int(7));
    raw = p.get();
  }
  assert(*raw == 7);
  return 0;
}
