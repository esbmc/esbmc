#if __cplusplus >= 201103L
#  define _ESBMC_NOEXCEPT noexcept
#else
#  define _ESBMC_NOEXCEPT throw()
#endif

// Per-thread count of exceptions thrown/rethrown but not yet entered into their
// handler, maintained by the GOTO exception lowering (exception_globals,
// remove_exceptions.cpp). Backs std::uncaught_exception(s), [except.uncaught].
extern "C" unsigned long __ESBMC_exc_uncaught_count;

namespace std
{

typedef void (*terminate_handler)();
typedef void (*unexpected_handler)();
void terminate() _ESBMC_NOEXCEPT;

} // namespace std

namespace __ESBMC_exception_detail
{

inline void __ESBMC_default_terminate_handler()
{
  __ESBMC_assert(0, "terminate called after throwing an exception");
  __ESBMC_assume(0);
}

inline void __ESBMC_default_unexpected_handler()
{
  std::terminate();
}

inline std::terminate_handler &__ESBMC_terminate_handler_ref()
{
  static std::terminate_handler handler = __ESBMC_default_terminate_handler;
  return handler;
}

inline std::unexpected_handler &__ESBMC_unexpected_handler_ref()
{
  static std::unexpected_handler handler = __ESBMC_default_unexpected_handler;
  return handler;
}

} // namespace __ESBMC_exception_detail

extern "C" void __ESBMC_run_unexpected()
{
  std::unexpected_handler handler =
    __ESBMC_exception_detail::__ESBMC_unexpected_handler_ref();
  (*handler)();
}

namespace std
{

terminate_handler set_terminate(terminate_handler f) _ESBMC_NOEXCEPT
{
  terminate_handler old =
    __ESBMC_exception_detail::__ESBMC_terminate_handler_ref();
  __ESBMC_exception_detail::__ESBMC_terminate_handler_ref() =
    f ? f : __ESBMC_exception_detail::__ESBMC_default_terminate_handler;
  return old;
}

terminate_handler get_terminate() _ESBMC_NOEXCEPT
{
  return __ESBMC_exception_detail::__ESBMC_terminate_handler_ref();
}

void terminate() _ESBMC_NOEXCEPT
{
  terminate_handler handler =
    __ESBMC_exception_detail::__ESBMC_terminate_handler_ref();

  try
  {
    (*handler)();
  }
  catch (...)
  {
    __ESBMC_assert(0, "std::terminate handler threw an exception");
    __ESBMC_assume(0);
  }

  __ESBMC_assert(0, "std::terminate handler unexpectedly returned");
  __ESBMC_assume(0);
}

unexpected_handler set_unexpected(unexpected_handler f) _ESBMC_NOEXCEPT
{
  unexpected_handler old =
    __ESBMC_exception_detail::__ESBMC_unexpected_handler_ref();
  __ESBMC_exception_detail::__ESBMC_unexpected_handler_ref() =
    f ? f : __ESBMC_exception_detail::__ESBMC_default_unexpected_handler;
  return old;
}

unexpected_handler get_unexpected() _ESBMC_NOEXCEPT
{
  return __ESBMC_exception_detail::__ESBMC_unexpected_handler_ref();
}

void unexpected()
{
  try
  {
    __ESBMC_run_unexpected();
  }
  catch (...)
  {
    terminate();
  }

  terminate();
}

// Both forms are defined unconditionally: the OM library is compiled at a single
// fixed standard (no --std flag), so a __cplusplus guard could omit a body that a
// user program built at a different standard still references — the same
// on-demand-linking gap that the set_unexpected/unexpected models hit. Providing
// the entry points always is harmless: conformant code at the wrong standard
// cannot name the form the standard removed, so it stays dead.
bool uncaught_exception() _ESBMC_NOEXCEPT
{
  return __ESBMC_exc_uncaught_count != 0;
}

int uncaught_exceptions() _ESBMC_NOEXCEPT
{
  return (int)__ESBMC_exc_uncaught_count;
}

} // namespace std
