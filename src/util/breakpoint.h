#pragma once

static void breakpoint()
{
#ifndef _WIN32
#  if !(defined(__arm__) || defined(__aarch64__))
  __asm__("int $3");
#  else
  log_error("Can't trap on ARM, sorry");
  abort();
#  endif
#else
  log_error("Can't trap on windows, sorry");
  abort();
#endif
}
