#pragma once

static void breakpoint()
{
#ifndef _WIN32
#  if defined(__i386__) || defined(__x86_64__)
  __asm__("int $3");
#  else
  log_error("Can't trap on this platform, sorry");
  abort();
#  endif
#else
  log_error("Can't trap on windows, sorry");
  abort();
#endif
}
