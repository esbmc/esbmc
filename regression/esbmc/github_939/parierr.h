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

enum err_list {
/* Force errors into non-0 */
  e_SYNTAX = 1, e_BUG,

  e_ALARM, e_FILE,

  e_MISC, e_FLAG, e_IMPL, e_ARCH, e_PACKAGE, e_NOTFUNC,

  e_PREC, e_TYPE, e_DIM, e_VAR, e_PRIORITY, e_USER,

  e_STACK, e_STACKTHREAD, e_OVERFLOW, e_DOMAIN, e_COMPONENT,

  e_MAXPRIME,

  e_CONSTPOL, e_IRREDPOL, e_COPRIME, e_PRIME, e_MODULUS, e_ROOTS0,

  e_OP, e_TYPE2, e_INV,

  e_MEM,

  e_SQRTN,

  e_FILEDESC,
/* NO ERROR */
  e_NONE
};

enum { warner, warnprec, warnfile, warnmem, warnuser, warnstack, warnstackthread };
