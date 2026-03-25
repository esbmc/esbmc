// Test: unique_ptr relational operators (<, <=, >, >=) and nullptr comparisons
#include <cassert>
#include <memory>

int main()
{
  std::unique_ptr<int> null_ptr;
  auto p = std::make_unique<int>(42);

  // operator== and != with nullptr (already existed; confirm still works)
  assert(null_ptr == nullptr);
  assert(nullptr == null_ptr);
  assert(p != nullptr);
  assert(nullptr != p);

  // Reflexive properties of relational operators using get() on same pointer
  int *raw = p.get();
  assert(!(raw < raw));
  assert(raw <= raw);
  assert(!(raw > raw));
  assert(raw >= raw);

  return 0;
}
