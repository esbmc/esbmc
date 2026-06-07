#if __cplusplus >= 201103L
#  define _ESBMC_NOEXCEPT noexcept
#else
#  define _ESBMC_NOEXCEPT throw()
#endif

// Per-thread count of exceptions thrown/rethrown but not yet entered into their
// handler, defined in library/exception_globals.c and maintained by the GOTO
// exception lowering. Backs std::uncaught_exception(s), [except.uncaught].
extern "C" __thread bool __ESBMC_exc_thrown;
extern "C" __thread __SIZE_TYPE__ __ESBMC_exc_typeid;
extern "C" __thread void *__ESBMC_exc_value;
extern "C" __thread __SIZE_TYPE__ __ESBMC_exc_uncaught_count;
extern "C" __thread __SIZE_TYPE__ __ESBMC_exc_terminate_reason;

namespace __ESBMC_exception_detail
{
enum __ESBMC_terminate_reason
{
  __ESBMC_terminate_reason_generic = 0,
  __ESBMC_terminate_reason_uncaught = 1,
  __ESBMC_terminate_reason_noexcept = 2,
  __ESBMC_terminate_reason_exception_spec = 3,
  __ESBMC_terminate_reason_no_active = 4,
};

struct __ESBMC_exception_slot
{
  __SIZE_TYPE__ typeid_value;
  void *object;
  bool valid;
};

struct __ESBMC_handled_exception
{
  __SIZE_TYPE__ typeid_value;
  void *object;
};

static __ESBMC_exception_slot __ESBMC_exception_slots[256];
static __SIZE_TYPE__ __ESBMC_exception_slot_count = 0;
static __ESBMC_handled_exception __ESBMC_handled_exceptions[256];
static __SIZE_TYPE__ __ESBMC_handled_exception_depth = 0;

} // namespace __ESBMC_exception_detail

namespace std
{
typedef void (*terminate_handler)();
typedef void (*unexpected_handler)();
void terminate() _ESBMC_NOEXCEPT;

} // namespace std

namespace __ESBMC_exception_detail
{
void __ESBMC_default_terminate_handler()
{
__ESBMC_HIDE:;
  switch (__ESBMC_exc_terminate_reason)
  {
  case __ESBMC_terminate_reason_uncaught:
    __ESBMC_assert(0, "uncaught exception");
    break;
  case __ESBMC_terminate_reason_noexcept:
    __ESBMC_assert(0, "noexcept specification violated");
    break;
  case __ESBMC_terminate_reason_exception_spec:
    __ESBMC_assert(0, "exception specification violated");
    break;
  case __ESBMC_terminate_reason_no_active:
    __ESBMC_assert(0, "throw with no active exception");
    break;
  default:
    __ESBMC_assert(0, "terminate called after throwing an exception");
    break;
  }
  __ESBMC_assume(0);
}

void __ESBMC_default_unexpected_handler()
{
__ESBMC_HIDE:;
  std::terminate();
}

inline std::terminate_handler &__ESBMC_terminate_handler_ref()
{
__ESBMC_HIDE:;
  static std::terminate_handler handler = 0;
  return handler;
}

inline std::unexpected_handler &__ESBMC_unexpected_handler_ref()
{
__ESBMC_HIDE:;
  static std::unexpected_handler handler = 0;
  return handler;
}

} // namespace __ESBMC_exception_detail

extern "C" void __ESBMC_run_unexpected()
{
__ESBMC_HIDE:;
  std::unexpected_handler handler =
    __ESBMC_exception_detail::__ESBMC_unexpected_handler_ref();
  if (!handler)
    handler = __ESBMC_exception_detail::__ESBMC_default_unexpected_handler;
  (*handler)();
}

extern "C" void *__ESBMC_current_exception_raw()
{
__ESBMC_HIDE:;
  __SIZE_TYPE__ typeid_value = __ESBMC_exc_typeid;
  void *object = __ESBMC_exc_value;

  if (__ESBMC_exception_detail::__ESBMC_handled_exception_depth != 0)
  {
    __SIZE_TYPE__ top =
      __ESBMC_exception_detail::__ESBMC_handled_exception_depth - 1;
    typeid_value =
      __ESBMC_exception_detail::__ESBMC_handled_exceptions[top].typeid_value;
    object = __ESBMC_exception_detail::__ESBMC_handled_exceptions[top].object;
  }

  if (typeid_value == 0)
    return 0;

  __ESBMC_assert(
    __ESBMC_exception_detail::__ESBMC_exception_slot_count < 256,
    "exception_ptr slot table exhausted");

  __SIZE_TYPE__ idx = __ESBMC_exception_detail::__ESBMC_exception_slot_count++;
  __ESBMC_exception_detail::__ESBMC_exception_slots[idx].typeid_value =
    typeid_value;
  __ESBMC_exception_detail::__ESBMC_exception_slots[idx].object = object;
  __ESBMC_exception_detail::__ESBMC_exception_slots[idx].valid = 1;
  return &__ESBMC_exception_detail::__ESBMC_exception_slots[idx];
}

extern "C" void __ESBMC_push_handled_exception()
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    __ESBMC_exception_detail::__ESBMC_handled_exception_depth < 256,
    "handled exception stack exhausted");

  __SIZE_TYPE__ idx =
    __ESBMC_exception_detail::__ESBMC_handled_exception_depth++;
  __ESBMC_exception_detail::__ESBMC_handled_exceptions[idx].typeid_value =
    __ESBMC_exc_typeid;
  __ESBMC_exception_detail::__ESBMC_handled_exceptions[idx].object =
    __ESBMC_exc_value;
}

extern "C" void __ESBMC_pop_handled_exception()
{
__ESBMC_HIDE:;
  if (__ESBMC_exception_detail::__ESBMC_handled_exception_depth == 0)
    return;

  __ESBMC_exception_detail::__ESBMC_handled_exception_depth--;
  if (__ESBMC_exc_thrown)
    return;

  if (__ESBMC_exception_detail::__ESBMC_handled_exception_depth == 0)
  {
    __ESBMC_exc_typeid = 0;
    __ESBMC_exc_value = 0;
    return;
  }

  __SIZE_TYPE__ top =
    __ESBMC_exception_detail::__ESBMC_handled_exception_depth - 1;
  __ESBMC_exc_typeid =
    __ESBMC_exception_detail::__ESBMC_handled_exceptions[top].typeid_value;
  __ESBMC_exc_value =
    __ESBMC_exception_detail::__ESBMC_handled_exceptions[top].object;
}

extern "C" void __ESBMC_rethrow_current_exception()
{
__ESBMC_HIDE:;
  if (__ESBMC_exception_detail::__ESBMC_handled_exception_depth == 0)
  {
    __ESBMC_exc_terminate_reason =
      __ESBMC_exception_detail::__ESBMC_terminate_reason_no_active;
    std::terminate();
  }

  __SIZE_TYPE__ top =
    __ESBMC_exception_detail::__ESBMC_handled_exception_depth - 1;
  __ESBMC_exc_thrown = 1;
  __ESBMC_exc_typeid =
    __ESBMC_exception_detail::__ESBMC_handled_exceptions[top].typeid_value;
  __ESBMC_exc_value =
    __ESBMC_exception_detail::__ESBMC_handled_exceptions[top].object;
  __ESBMC_exc_uncaught_count = __ESBMC_exc_uncaught_count + 1;
}

extern "C" void __ESBMC_rethrow_exception_raw(void *ptr)
{
__ESBMC_HIDE:;
  if (!ptr)
    std::terminate();

  __ESBMC_exception_detail::__ESBMC_exception_slot *slot =
    (__ESBMC_exception_detail::__ESBMC_exception_slot *)ptr;
  if (!slot->valid)
    std::terminate();

  __ESBMC_exc_thrown = 1;
  __ESBMC_exc_typeid = slot->typeid_value;
  __ESBMC_exc_value = slot->object;
  __ESBMC_exc_uncaught_count = __ESBMC_exc_uncaught_count + 1;
}

namespace std
{
terminate_handler set_terminate(terminate_handler f) _ESBMC_NOEXCEPT
{
__ESBMC_HIDE:;
  terminate_handler &slot =
    __ESBMC_exception_detail::__ESBMC_terminate_handler_ref();
  terminate_handler old =
    slot ? slot : __ESBMC_exception_detail::__ESBMC_default_terminate_handler;
  slot = f;
  return old;
}

terminate_handler get_terminate() _ESBMC_NOEXCEPT
{
__ESBMC_HIDE:;
  terminate_handler handler =
    __ESBMC_exception_detail::__ESBMC_terminate_handler_ref();
  return handler ? handler
                 : __ESBMC_exception_detail::__ESBMC_default_terminate_handler;
}

void terminate() _ESBMC_NOEXCEPT
{
__ESBMC_HIDE:;
  terminate_handler handler =
    __ESBMC_exception_detail::__ESBMC_terminate_handler_ref();

  try
  {
    if (handler)
      (*handler)();
    else
      __ESBMC_exception_detail::__ESBMC_default_terminate_handler();
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
__ESBMC_HIDE:;
  unexpected_handler &slot =
    __ESBMC_exception_detail::__ESBMC_unexpected_handler_ref();
  unexpected_handler old =
    slot ? slot : __ESBMC_exception_detail::__ESBMC_default_unexpected_handler;
  slot = f;
  return old;
}

unexpected_handler get_unexpected() _ESBMC_NOEXCEPT
{
__ESBMC_HIDE:;
  unexpected_handler handler =
    __ESBMC_exception_detail::__ESBMC_unexpected_handler_ref();
  return handler ? handler
                 : __ESBMC_exception_detail::__ESBMC_default_unexpected_handler;
}

void unexpected()
{
__ESBMC_HIDE:;
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
__ESBMC_HIDE:;
  return __ESBMC_exc_uncaught_count != 0;
}

int uncaught_exceptions() _ESBMC_NOEXCEPT
{
__ESBMC_HIDE:;
  return (int)__ESBMC_exc_uncaught_count;
}

} // namespace std
