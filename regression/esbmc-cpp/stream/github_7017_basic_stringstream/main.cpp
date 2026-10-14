#include <sstream>
#include <cstring>
#include <cassert>

template <class T>
struct MyAlloc
{
  typedef T value_type;
  MyAlloc()
  {
  }
  template <class U>
  MyAlloc(const MyAlloc<U> &)
  {
  }
  T *allocate(size_t n)
  {
    return (T *)::operator new(n * sizeof(T));
  }
  void deallocate(T *p, size_t)
  {
    ::operator delete(p);
  }
};

typedef std::basic_stringstream<char, std::char_traits<char>, MyAlloc<char> >
  Stream;

int main()
{
  Stream ss;
  ss << "n=" << 42;
  assert(strcmp(ss.str().c_str(), "n=42") == 0);
  return 0;
}
