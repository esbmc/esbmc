#include <cassert>
#include <cstdlib>
#include <memory>
#include <type_traits>

struct Plain
{
  int x;
};
struct Poly
{
  virtual ~Poly()
  {
  }
};
struct NonTrivialDtor
{
  ~NonTrivialDtor()
  {
  }
};

int main()
{
  static_assert(std::is_class<Plain>::value, "");
  static_assert(!std::is_class<int>::value, "");
  static_assert(std::is_polymorphic<Poly>::value, "");
  static_assert(!std::is_polymorphic<Plain>::value, "");
  static_assert(std::is_trivially_default_constructible<Plain>::value, "");
  static_assert(!std::is_trivially_default_constructible<Poly>::value, "");
  static_assert(std::is_trivially_destructible<Plain>::value, "");
  static_assert(!std::is_trivially_destructible<NonTrivialDtor>::value, "");

  std::size_t n = 2;

  std::unique_ptr<int> p(new int(7));
  assert(p.get() != nullptr);
  p = nullptr;
  assert(p.get() == nullptr);
  assert(!p);

  std::unique_ptr<int> q = nullptr;
  assert(q.get() == nullptr);

  // The aws-sdk-cpp shape: the nullptr_t ctor must be non-explicit for the
  // conditional operator to find a common type.
  std::unique_ptr<int[]> arr =
    (n > 0) ? std::unique_ptr<int[]>(new int[n]) : nullptr;
  assert(arr.get() != nullptr);
  arr[0] = 5;
  assert(arr[0] == 5);
  arr = nullptr;
  assert(arr.get() == nullptr);

  return 0;
}
