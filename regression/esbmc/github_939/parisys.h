/* Copyright (C) 2000  The PARI group.

This file is part of the PARI/GP package.

PARI/GP is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY WHATSOEVER.

Check the License for details. You should have received a copy of it, along
with the package; see the file 'COPYING'. If not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

/* This files contains macros depending on system and compiler    */

#ifdef __cplusplus
#  define BEGINEXTERN extern "C" {
#  define ENDEXTERN }
#else
#  define BEGINEXTERN
#  define ENDEXTERN
#endif

#ifdef DISABLE_INLINE
#  undef ASMINLINE
#else
#  ifdef __cplusplus
#    define INLINE inline static
#  elif defined(__GNUC__)
#    define INLINE __inline__ static
#  endif
#endif

#ifndef DISABLE_VOLATILE
#  ifdef __GNUC__
#    define VOLATILE volatile
#  endif
#endif

#ifndef VOLATILE
#  define VOLATILE
#endif
#ifndef INLINE
#  define INLINE static
#endif
#ifdef ENABLE_TLS
#  define THREAD __thread
#else
#  define THREAD
#endif

#if defined(_WIN32) || defined(__CYGWIN32__)
/* ANSI C does not allow to longjmp() out of a signal handler, in particular,
 * the SIGINT handler. On Win32, the handler is executed in another thread, and
 * longjmp'ing into another thread's stack will utterly confuse the system.
 * Instead, we check whether win32ctrlc is set in new_chunk(). */
BEGINEXTERN
  extern int win32ctrlc, win32alrm;
  void dowin32ctrlc(void);
ENDEXTERN
#define CHECK_CTRLC if (win32ctrlc) dowin32ctrlc();
#else
#define CHECK_CTRLC
#endif
