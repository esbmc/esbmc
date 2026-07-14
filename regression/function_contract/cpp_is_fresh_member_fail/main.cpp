/* cpp_is_fresh_member_fail:
 * Negative variant of cpp_is_fresh_member_pass (issue #6). The is_fresh member
 * pointer is allocated correctly, but the requires bound is too weak
 * (id <= size_ permits id == size_), so the body's inst_[id] reads one element
 * past the size_-element allocation.
 *
 * Confirms the member is_fresh allocation is genuinely bounded — the array
 * access is checked, not vacuously valid.
 *
 * Regression for: https://github.com/Yiannis128/esbmc/issues/6
 *
 * Expected: VERIFICATION FAILED
 */
#include <cstddef>

class Prog
{
public:
  __ESBMC_contract
  int byteAt(int id)
  {
    __ESBMC_requires(id >= 0 && id <= size_); /* BUG: <= allows id == size_ */
    __ESBMC_requires(__ESBMC_is_fresh(inst_, (size_t)size_ * sizeof(int)));
    return inst_[id];
  }

private:
  int *inst_;
  int size_;
};

int main(void)
{
  return 0;
}
