#include <memory>
#include <type_traits>

// An allocator that supplies no rebind member, as fmt::detail::allocator does.
template <class T>
struct plain_alloc
{
  typedef T value_type;
  T *allocate(std::size_t n);
  void deallocate(T *p, std::size_t n);
};

int main()
{
  typedef std::allocator_traits<plain_alloc<int>>::rebind_alloc<char> rebound;
  static_assert(
    std::is_same<rebound, plain_alloc<char>>::value,
    "[allocator.traits.types]: rebind_alloc rewrites the first argument");
  // An allocator that does supply rebind must keep working.
  static_assert(
    std::is_same<
      std::allocator_traits<std::allocator<int>>::rebind_alloc<char>,
      std::allocator<char>>::value,
    "std::allocator's own rebind still wins");
  return 0;
}
