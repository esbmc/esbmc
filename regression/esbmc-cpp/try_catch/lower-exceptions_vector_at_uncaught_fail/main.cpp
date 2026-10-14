// vector::at with an out-of-range index throws std::out_of_range. Here only an
// unrelated handler is present (std::runtime_error is a sibling of
// std::out_of_range, not a base), so the exception escapes main and calls
// std::terminate -> VERIFICATION FAILED. Counterpart to try-catch_vector_02_bug,
// where the same throw is caught (SUCCESSFUL).
#include <vector>
#include <stdexcept>

int main()
{
  std::vector<int> v(10);
  try
  {
    v.at(20) = 1; // throws std::out_of_range
  }
  catch (std::runtime_error &) // does NOT match out_of_range -> uncaught
  {
  }
  return 0;
}
