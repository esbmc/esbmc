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

#define mael2(m,x1,x2)          (((GEN*)   (m))[x1][x2])
#define mael3(m,x1,x2,x3)       (((GEN**)  (m))[x1][x2][x3])
#define mael4(m,x1,x2,x3,x4)    (((GEN***) (m))[x1][x2][x3][x4])
#define mael5(m,x1,x2,x3,x4,x5) (((GEN****)(m))[x1][x2][x3][x4][x5])
#define mael mael2

#define gmael1(m,x1)             (((GEN*)    (m))[x1])
#define gmael2(m,x1,x2)          (((GEN**)   (m))[x1][x2])
#define gmael3(m,x1,x2,x3)       (((GEN***)  (m))[x1][x2][x3])
#define gmael4(m,x1,x2,x3,x4)    (((GEN****) (m))[x1][x2][x3][x4])
#define gmael5(m,x1,x2,x3,x4,x5) (((GEN*****)(m))[x1][x2][x3][x4][x5])
#define gmael(m,x,y) gmael2(m,x,y)
#define gel(m,x)     gmael1(m,x)

#define gcoeff(a,i,j) (((GEN**)(a))[j][i])
#define coeff(a,i,j) (((GEN*)(a))[j][i])

#define GSTR(x) ((char*) (((GEN) (x)) + 1 ))
