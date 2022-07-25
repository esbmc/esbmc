#if defined(_WIN32)
#define EX_OK 0
#define EX_USAGE 1
#else
#include <sysexits.h>
#endif

#include <util/cmdline.h>
#include <util/parseoptions.h>
#include <util/signal_catcher.h>
#include <boost/program_options.hpp>
#include <boost/program_options.hpp>

parseoptions_baset::parseoptions_baset(
  const struct group_opt_templ *opts,
  int argc,
  const char **argv)
  : executable_path(argv[0])
{
  exception_occured = cmdline.parse(argc, argv, opts);
}

void parseoptions_baset::help()
{
}

int parseoptions_baset::main()
{
  if(exception_occured)
  {
    return EX_USAGE;
  }
  if(cmdline.isset("help") || cmdline.isset("explain"))
  {
    help();
    return EX_OK;
  }
  // install signal catcher
  install_signal_catcher();
  return doit();
}
