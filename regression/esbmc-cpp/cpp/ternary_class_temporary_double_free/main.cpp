// The same over-destruction used to become a double free once the class owns a
// resource. That is why ~unique_ptr is still disabled in src/cpp/library/memory
// ("TODO: fix remove goto sideeffect"): before this fix, re-enabling it made
// ESBMC report a false `invalid pointer freed` on
//   std::unique_ptr<T> p = cond ? std::unique_ptr<T>(new T) : nullptr;
// which is the aws-sdk-cpp shape the nullptr_t constructor there exists for.
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <cassert>
#include <cstddef>

struct T
{
  int *p;
  explicit T(int *q) : p(q)
  {
  }
  T(std::nullptr_t) : p(nullptr)
  {
  }
  T(T &&o) : p(o.p)
  {
    o.p = nullptr;
  }
  T(const T &) = delete;
  ~T()
  {
    delete p;
  }
};

int main()
{
  int n = 2;
  T a = (n > 0) ? T(new int(5)) : nullptr;
  assert(*a.p == 5);
  return 0;
}
