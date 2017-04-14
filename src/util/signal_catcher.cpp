/*******************************************************************\

Module:

Author:

Date:

\*******************************************************************/

#if defined(_WIN32)

#else
#include <csignal>
#include <cstdlib>
#endif

#include <util/signal_catcher.h>

void install_signal_catcher()
{
  #if defined(_WIN32)
  #else
  // declare act to deal with action on signal set
  static struct sigaction act;
	
  act.sa_handler=signal_catcher;
  act.sa_flags=0;
  sigfillset(&(act.sa_mask));

  // install signal handler
  sigaction(SIGTERM, &act, NULL);
  #endif
}

void signal_catcher(int sig)
{
  #if defined(_WIN32)
  #else
  // kill any children by killing group
  killpg(0, sig);

  exit(sig);
  #endif
}
