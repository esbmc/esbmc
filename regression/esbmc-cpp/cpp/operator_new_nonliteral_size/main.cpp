// KNOWNBUG. ::operator new loses the heap object's extent when its size
// argument is not a literal, so even the first in-bounds write is reported out
// of bounds. g++ runs this program with the assertion holding.
//
// It is ::operator new specifically, and it is the literal, not a call
// boundary -- every one of these verifies:
//
//   ::operator new(16)                        literal, same statement
//   callee returning ::operator new(16)       literal inside a callee
//   new int[n]                                array new, variable extent
//   malloc(n) --force-malloc-success          malloc, variable size
//
// while ::operator new(n) with n a plain local does not. Any allocator that
// forwards a byte count hits this, std::pmr::memory_resource::do_allocate
// included.
#include <new>
#include <cassert>

int main()
{
  unsigned long n = 16;
  int *p = static_cast<int *>(::operator new(n));
  p[0] = 11;
  assert(p[0] == 11);
  ::operator delete(p);
  return 0;
}
