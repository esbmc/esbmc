#ifndef ESBMC_UTIL_STACK_BUDGET_H
#define ESBMC_UTIL_STACK_BUDGET_H

#include <cstddef>
#include <cstdint>

/** Bound the stack a recursive tree walk may consume, so a pathologically
 *  deep expression or type yields a diagnostic instead of SIGSEGV
 *  (esbmc/esbmc#5048).
 *
 *  Counts **bytes, not frames**: a frame count tuned on one build is wrong on
 *  another -- a Debug frame is several times an optimised one -- whereas a
 *  byte budget holds whatever the frame costs.
 *
 *  Declare one instance at the top of the recursive function, then ask
 *  `exceeded()`. Reporting is left to the caller so the accounting can be
 *  exercised by a test without taking the process down with it.
 *
 *  `Tag` gives each walk its own base pointer and depth, so nested walks of
 *  different kinds do not measure each other. State is thread-local: several
 *  threads may be walking at once, on stacks that are nowhere near each other.
 */
template <class Tag>
class stack_budget_guardt
{
public:
  stack_budget_guardt()
  {
    if (depth++ == 0)
      base = here();
  }

  ~stack_budget_guardt()
  {
    if (--depth == 0)
      base = 0;
  }

  stack_budget_guardt(const stack_budget_guardt &) = delete;
  stack_budget_guardt &operator=(const stack_budget_guardt &) = delete;

  /** Stack consumed since the outermost level. Zero at that level itself. */
  std::ptrdiff_t bytes_used() const
  {
    // Direction is platform business, not ours -- take the magnitude.
    const std::uintptr_t now = here();
    return static_cast<std::ptrdiff_t>(now > base ? now - base : base - now);
  }

  bool exceeded(std::ptrdiff_t budget) const
  {
    return bytes_used() > budget;
  }

private:
  /* An integer, not a pointer: the address is only ever differenced, never
   * dereferenced, and storing it as a pointer to a local trips GCC's
   * -Wdangling-pointer even though the guard clears it on the way out. */
  std::uintptr_t here() const
  {
    return reinterpret_cast<std::uintptr_t>(&marker);
  }

  char marker;
  static thread_local std::uintptr_t base;
  static thread_local unsigned depth;
};

template <class Tag>
thread_local std::uintptr_t stack_budget_guardt<Tag>::base = 0;
template <class Tag>
thread_local unsigned stack_budget_guardt<Tag>::depth = 0;

/** Default ceiling for a walk. main() runs ESBMC on a 512 MiB stack (#6617);
 *  this leaves room to unwind and report rather than dying on the way out. */
constexpr std::ptrdiff_t default_stack_budget = 384L * 1024 * 1024;

#endif
