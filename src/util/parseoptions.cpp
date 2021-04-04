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
#include <boost/program_options.hpp>
#include <ac_config.h>
parseoptions_baset::parseoptions_baset(
  int argc,
  const char **argv)
{
  exception_message = "";
  try
  {

    cmdline.parse(argc, argv);
 
  }
  catch(std::exception &e)
  {
    exception_message = e.what();
  }
}

void parseoptions_baset::help()
{
}

int parseoptions_baset::main()
{
  if(exception_message != "")
  {
    std::cerr << "esbmc error: " << exception_message;
    std::cerr << std::endl;
    return EX_USAGE;
  }
  if(cmdline.isset("help"))

  {
    std::cout << "\n* * *           ESBMC " ESBMC_VERSION "          * * *\n";

    std::cout << cmdline.cmdline_options << "\n";
    return EX_OK;
  }

  // install signal catcher
  install_signal_catcher();

  return doit();
}
