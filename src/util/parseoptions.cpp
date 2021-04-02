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

using namespace boost::program_options;
#include <ac_config.h>
parseoptions_baset::parseoptions_baset(
  const struct opt_templ *opts,
  int argc,
  const char **argv)
{
  exception_message = "";
  try
  {
    //  if( argc > 1 )
    // {
    //     std::cout << "there are " << argc-1 << " (more) arguments, they are:\n" ;

    //     std::copy( argv+1, argv+argc, std::ostream_iterator<const char*>( std::cout, "\n" ) ) ;
    // }
    cmdline.parse(argc, argv);
    //     if(cmdline.vm.count("input-file"))
    //     {
    // auto src = cmdline.vm["input-file"].as<std::vector<std::string>>();
    //   std::cout<<"Values for input-file are: \n"<<src<<"\n";
    //     }
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
  if(cmdline.isset("help") || cmdline.isset("h"))
  // if(cmdline.isset('?') || cmdline.isset('h') || cmdline.isset("help"))
  {
    std::cout << "\n* * *           ESBMC " ESBMC_VERSION "          * * *\n";

    std::cout << cmdline.cmdline_options << "\n";
    return EX_OK;
  }

  // install signal catcher
  install_signal_catcher();

  return doit();
}
