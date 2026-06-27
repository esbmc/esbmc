/* cpp_is_fresh_member_pass:
 * __ESBMC_is_fresh on a private data-member pointer of a C++ method, under
 * --enforce-contract.
 *
 * The is_fresh allocation writes through the receiver (this->inst_). It must be
 * emitted AFTER the harness sets up `this` with NONDET(struct); otherwise the
 * NONDET havoc clobbers inst_ and the body's in-bounds dereference fails with
 * "dereference failure: invalid pointer". The identical clause on a parameter
 * pointer already worked.
 *
 * Regression for: https://github.com/Yiannis128/esbmc/issues/6
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <cstddef>

class Prog
{
public:
  __ESBMC_contract
  int byteAt(int id)
  {
    __ESBMC_requires(id >= 0 && id < size_);
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
