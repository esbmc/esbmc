#include <cstdint>
#include <esbmc/esbmc_parseoptions.h>
#include <esbmc/globals.h>
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

#ifdef __GLIBC__
#  include <cstring>
#  include <malloc.h>

/* glibc gives a non-main thread its own arena, so running the work on the
 * thread below leaves symbolic execution, encoding and the solver allocating
 * from a secondary one. Capping the process at a single arena puts them back on
 * the main arena and is worth ~5% of wall clock, but only together with the
 * malloc_trim() once GOTO creation is done (esbmc/esbmc#6831): the two are a
 * pair, and the trim alone is a 3% regression.
 *
 * --parallel-solving is the exception, since it allocates from several threads
 * at once and one arena would serialise them. It has to be read out of argv
 * because a thread's arena is fixed at its first allocation, long before the
 * options are parsed. */
static bool wants_parallel_solving(int argc, const char **argv)
{
  for (int i = 1; i < argc; i++)
    if (strcmp(argv[i], "--parallel-solving") == 0)
      return true;
  return false;
}
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

/* Called from wherever the run happens so the handlers travel with the work.
 * An exception escaping the worker thread below would otherwise reach
 * std::terminate without passing through main at all. */
static int run_esbmc(int argc, const char **argv)
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

#ifndef _WIN32
#  include <pthread.h>

/* Conversion and symbolic execution both recurse over expression structure, so
 * generated inputs with deeply nested expressions exhaust the default 8MB main
 * thread stack (#6617). A secondary thread can be given a much larger one,
 * which is what clang and LLVM do for the same reason. Raising RLIMIT_STACK
 * instead only works on Linux: macOS accepts the call and reports the new soft
 * limit, but still does not let the main thread's stack grow. */
static const size_t esbmc_stack_size = 512UL * 1024 * 1024;

namespace
{
struct main_argst
{
  int argc;
  const char **argv;
  int result;
};
} // namespace

static void *run_main(void *arg)
{
  main_argst *args = static_cast<main_argst *>(arg);
  /* esbmc_parseoptionst is constructed in here rather than by the caller:
   * language_uit's constructor initialises migrate_namespace_lookup, which is
   * thread_local, so building it on one thread and running it on another
   * leaves it null. */
  args->result = run_esbmc(args->argc, args->argv);
  return nullptr;
}
#endif

int main(int argc, const char **argv)
{
#ifdef __GLIBC__
  if (!wants_parallel_solving(argc, argv))
    mallopt(M_ARENA_MAX, 1);
#endif

  register_bundled_files();

#ifndef _WIN32
  pthread_attr_t attr;
  if (pthread_attr_init(&attr) == 0)
  {
    main_argst args{argc, argv, 0};
    pthread_t thread;
    bool started = pthread_attr_setstacksize(&attr, esbmc_stack_size) == 0 &&
                   pthread_create(&thread, &attr, run_main, &args) == 0;
    pthread_attr_destroy(&attr);
    if (started)
    {
      pthread_join(thread, nullptr);
      return args.result;
    }
  }
  /* Falling through means the larger stack was refused, which is not a reason
   * to give up: everything short of very deep recursion still works. */
#endif

  return run_esbmc(argc, argv);
}
