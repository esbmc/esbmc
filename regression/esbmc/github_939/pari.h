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

#ifndef __GENPARI__
#define __GENPARI__
#include "paricfg.h"

#include <stdlib.h>   /* malloc, free, atoi */
#ifdef UNIX
#  define _INCLUDE_POSIX_SOURCE /* for HPUX */
#  include <sys/types.h> /* size_t */
#endif

#include <signal.h>
#include <stdio.h>
#include <stdarg.h>
#include <setjmp.h>
#include <string.h>
#if !defined(_WIN32)
#  include <unistd.h>
#else
#  include <io.h>
#endif
#include <math.h>
#include <memory.h>
#include <ctype.h>

#include "parisys.h"
#include "parigen.h"
#include "paricast.h"
#include "paristio.h"
#include "paricom.h"
#include "parierr.h"
#include "paridbglvl.h"
BEGINEXTERN
#include "paridecl.h"
#include "paritune.h"
#include "parimt.h"
#ifndef PARI_NO_MPINL_H
#  include "mpinl.h"
#endif
#ifndef PARI_NO_PARIINL_H
#  include "pariinl.h"
#endif
ENDEXTERN
#include "pariold.h"
#endif
