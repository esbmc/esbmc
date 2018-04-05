/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#if defined(_WIN32)
#define EX_OK 0
#define EX_USAGE 1
#else
#include <sysexits.h>
#endif

#include <iostream>
#include <util/cmdline.h>
#include <util/parseoptions.h>
#include <util/signal_catcher.h>

parseoptions_baset::parseoptions_baset(
  const struct opt_templ *opts,
  int argc,
  const char **argv)
{
  parse_result = cmdline.parse(argc, argv, opts);
}

void parseoptions_baset::help()
{
}

int parseoptions_baset::main()
{
  if(parse_result)
  {
    std::cerr << "Unrecognized option \"" << cmdline.failing_option << "\"";
    std::cerr << std::endl;
    return EX_USAGE;
  }

  if(cmdline.isset('?') || cmdline.isset('h') || cmdline.isset("help"))
  {
    help();
    return EX_OK;
  }

  // install signal catcher
  install_signal_catcher();

  return doit();
}
