#include <exception>

namespace std
{
namespace
{
void default_terminate_handler()
{
  __ESBMC_assert(0, "std::terminate called");
  __ESBMC_assume(0);
}

terminate_handler current_terminate_handler = default_terminate_handler;

#if __cplusplus < 201703L
typedef void (*unexpected_handler)();
unexpected_handler current_unexpected_handler = 0;
#endif
} // namespace

terminate_handler set_terminate(terminate_handler handler) _ESBMC_NOEXCEPT
{
  terminate_handler old_handler = current_terminate_handler;
  current_terminate_handler = handler ? handler : default_terminate_handler;
  return old_handler;
}

terminate_handler get_terminate() _ESBMC_NOEXCEPT
{
  return current_terminate_handler;
}

void terminate() _ESBMC_NOEXCEPT
{
  terminate_handler handler = get_terminate();

  try
  {
    (*handler)();
  }
  catch (...)
  {
    __ESBMC_assert(0, "std::terminate handler threw an exception");
    __ESBMC_assume(0);
  }

  __ESBMC_assert(0, "std::terminate handler returned");
  __ESBMC_assume(0);
}

#if __cplusplus < 201703L
unexpected_handler set_unexpected(unexpected_handler handler) _ESBMC_NOEXCEPT
{
  unexpected_handler old_handler = current_unexpected_handler;
  current_unexpected_handler = handler;
  return old_handler;
}

unexpected_handler get_unexpected() _ESBMC_NOEXCEPT
{
  return current_unexpected_handler;
}

void unexpected()
{
  unexpected_handler handler = get_unexpected();

  if (handler)
    (*handler)();

  terminate();
}
#endif

bool uncaught_exception() _ESBMC_NOEXCEPT
{
  return nondet_bool();
}

int uncaught_exceptions() _ESBMC_NOEXCEPT
{
  int count = nondet_int();
  __ESBMC_assume(count >= 0);
  return count;
}

#if __cplusplus >= 201103L
exception_ptr current_exception() _ESBMC_NOEXCEPT
{
  return exception_ptr();
}

void rethrow_exception(exception_ptr ptr)
{
  if (!ptr)
    terminate();

  __ESBMC_assert(0, "std::rethrow_exception is not modelled yet");
  __ESBMC_assume(0);
}
#endif

} // namespace std
