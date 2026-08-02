#include <cstdint>
#include <esbmc/esbmc_parseoptions.h>
#include <langapi/mode.h>

#include <exception>
#include <typeinfo>

#include <irep2/irep2.h>
#include <util/config/config.h>
#include <util/message/message.h>

#if __has_include(<cxxabi.h>)
#  include <cxxabi.h>
#  define ESBMC_HAVE_CXXABI 1
#endif

// Name the exception being handled, for the catch-all arm. A `catch (const
// std::exception &)` does not match when the throwing library was built against
// a different C++ runtime than ESBMC -- the typeinfo pointers differ even
// though the type derives from std::exception -- and that is exactly the case
// the catch-all exists to report on. The Itanium ABI entry point still names
// the type there. Names stay mangled: demangling is compiler-specific, and the
// mangling identifies the type uniquely.
static const char *current_exception_name()
{
#ifdef ESBMC_HAVE_CXXABI
  if (const std::type_info *t = abi::__cxa_current_exception_type())
    return t->name();
#endif
  return "unknown type";
}

int main(int argc, const char **argv)
{
  try
  {
    esbmc_parseoptionst parseoptions(argc, argv);
    return parseoptions.main();
  }
  // Without this, an escaping exception reaches std::terminate, which aborts
  // with no ESBMC context -- on a CI runner that abort is the only evidence
  // there is (esbmc/esbmc#5189).
  catch (const std::exception &e)
  {
    log_error("uncaught exception [{}]: {}", typeid(e).name(), e.what());
  }
  catch (...)
  {
    log_error("uncaught exception [{}]", current_exception_name());
  }

  return 70; /* sysexits.h EX_SOFTWARE: internal error */
}
