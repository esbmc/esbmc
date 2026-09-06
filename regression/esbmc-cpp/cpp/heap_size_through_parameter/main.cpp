// KNOWNBUG. The allocation size reaching ::operator new through a function
// PARAMETER loses the heap object's extent, so an in-bounds write is reported
// as out of bounds. g++ runs this program with all assertions holding.
//
//   ::operator new(16) at the use site          -> SUCCESSFUL
//   a callee whose size is a literal            -> SUCCESSFUL  (heap_size_literal)
//   a callee whose size is its own parameter    -> FAILED      (this test)
//
// It is the parameter, not the call: a virtual callee with a literal size
// verifies, while a plain free function taking the size does not. Any
// allocator that forwards a byte count -- which is every allocator interface,
// std::pmr::memory_resource::do_allocate included -- hits this.
#include <new>
#include <cassert>

void *alloc(unsigned long n)
{
  return ::operator new(n);
}

int main()
{
  int *p = static_cast<int *>(alloc(16));
  p[0] = 11;
  p[3] = 22;
  assert(p[0] == 11 && p[3] == 22);
  ::operator delete(p);
  return 0;
}
