#if __cplusplus >= 201103L
#  define _ESBMC_NOEXCEPT noexcept
#else
#  define _ESBMC_NOEXCEPT throw()
#endif

extern bool nondet_bool();
extern int nondet_int();

namespace std
{

typedef void (*terminate_handler)();

#if __cplusplus < 201703L
typedef void (*unexpected_handler)();
#endif

} // namespace std

namespace __ESBMC_exception_detail
{

inline void __ESBMC_default_terminate_handler()
{
  __ESBMC_assert(0, "terminate called after throwing an exception");
  __ESBMC_assume(0);
}

#if __cplusplus < 201703L
inline void __ESBMC_default_unexpected_handler()
{
  std::terminate();
}
#endif

inline std::terminate_handler &__ESBMC_terminate_handler_ref()
{
  static std::terminate_handler handler = __ESBMC_default_terminate_handler;
  return handler;
}

#if __cplusplus < 201703L
inline std::unexpected_handler &__ESBMC_unexpected_handler_ref()
{
  static std::unexpected_handler handler = __ESBMC_default_unexpected_handler;
  return handler;
}
#endif

} // namespace __ESBMC_exception_detail

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

#if __cplusplus < 201703L
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
  unexpected_handler handler =
    __ESBMC_exception_detail::__ESBMC_unexpected_handler_ref();

  try
  {
    (*handler)();
  }
  catch (...)
  {
    terminate();
  }

  terminate();
}
#endif

#if __cplusplus < 202002L
bool uncaught_exception() _ESBMC_NOEXCEPT
{
  return nondet_bool();
}
#endif

#if __cplusplus >= 201703L
int uncaught_exceptions() _ESBMC_NOEXCEPT
{
  int count = nondet_int();
  __ESBMC_assume(count >= 0);
  return count;
}
#endif

} // namespace std
