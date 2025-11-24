/* Copyright (C) 2000, 2012  The PARI group.

This file is part of the PARI/GP package.

PARI/GP is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY WHATSOEVER.

Check the License for details. You should have received a copy of it, along
with the package; see the file 'COPYING'. If not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

/********************************************************************/
/**                                                                **/
/**                         LINEAR ALGEBRA                         **/
/**                          (first part)                          **/
/**                                                                **/
/********************************************************************/
#include "pari.h"
#include "paripriv.h"

#define DEBUGLEVEL DEBUGLEVEL_mat

/*******************************************************************/
/*                                                                 */
/*                         GEREPILE                                */
/*                                                                 */
/*******************************************************************/

static void
gerepile_mat(pari_sp av, pari_sp tetpil, GEN x, long k, long m, long n, long t)
{
  pari_sp A, bot = pari_mainstack->bot;
  long u, i;
  size_t dec;

  (void)gerepile(av,tetpil,NULL); dec = av-tetpil;

  for (u=t+1; u<=m; u++)
  {
    A = (pari_sp)coeff(x,u,k);
    if (A < av && A >= bot) coeff(x,u,k) += dec;
  }
  for (i=k+1; i<=n; i++)
    for (u=1; u<=m; u++)
    {
      A = (pari_sp)coeff(x,u,i);
      if (A < av && A >= bot) coeff(x,u,i) += dec;
    }
}

static void
gen_gerepile_gauss_ker(GEN x, long k, long t, pari_sp av, void *E, GEN (*copy)(void*, GEN))
{
  pari_sp tetpil = avma;
  long u,i, n = lg(x)-1, m = n? nbrows(x): 0;

  if (DEBUGMEM > 1) pari_warn(warnmem,"gauss_pivot_ker. k=%ld, n=%ld",k,n);
  for (u=t+1; u<=m; u++) gcoeff(x,u,k) = copy(E,gcoeff(x,u,k));
  for (i=k+1; i<=n; i++)
    for (u=1; u<=m; u++) gcoeff(x,u,i) = copy(E,gcoeff(x,u,i));
  gerepile_mat(av,tetpil,x,k,m,n,t);
}

/* special gerepile for huge matrices */

#define COPY(x) {\
  GEN _t = (x); if (!is_universal_constant(_t)) x = gcopy(_t); \
}

INLINE GEN
_copy(void *E, GEN x)
{
  (void) E; COPY(x);
  return x;
}

static void
gerepile_gauss_ker(GEN x, long k, long t, pari_sp av)
{
  gen_gerepile_gauss_ker(x, k, t, av, NULL, &_copy);
}

static void
gerepile_gauss(GEN x,long k,long t,pari_sp av, long j, GEN c)
{
  pari_sp tetpil = avma, A, bot;
  long u,i, n = lg(x)-1, m = n? nbrows(x): 0;
  size_t dec;

  if (DEBUGMEM > 1) pari_warn(warnmem,"gauss_pivot. k=%ld, n=%ld",k,n);
  for (u=t+1; u<=m; u++)
    if (u==j || !c[u]) COPY(gcoeff(x,u,k));
  for (u=1; u<=m; u++)
    if (u==j || !c[u])
      for (i=k+1; i<=n; i++) COPY(gcoeff(x,u,i));

  (void)gerepile(av,tetpil,NULL); dec = av-tetpil;
  bot = pari_mainstack->bot;
  for (u=t+1; u<=m; u++)
    if (u==j || !c[u])
    {
      A=(pari_sp)coeff(x,u,k);
      if (A<av && A>=bot) coeff(x,u,k)+=dec;
    }
  for (u=1; u<=m; u++)
    if (u==j || !c[u])
      for (i=k+1; i<=n; i++)
      {
        A=(pari_sp)coeff(x,u,i);
        if (A<av && A>=bot) coeff(x,u,i)+=dec;
      }
}

/*******************************************************************/
/*                                                                 */
/*                         GENERIC                                 */
/*                                                                 */
/*******************************************************************/
GEN
gen_ker(GEN x, long deplin, void *E, const struct bb_field *ff)
{
  pari_sp av0 = avma, av, tetpil;
  GEN y, c, d;
  long i, j, k, r, t, n, m;

  n=lg(x)-1; if (!n) return cgetg(1,t_MAT);
  m=nbrows(x); r=0;
  x = RgM_shallowcopy(x);
  c = zero_zv(m);
  d=new_chunk(n+1);
  av=avma;
  for (k=1; k<=n; k++)
  {
    for (j=1; j<=m; j++)
      if (!c[j])
      {
        gcoeff(x,j,k) = ff->red(E, gcoeff(x,j,k));
        if (!ff->equal0(gcoeff(x,j,k))) break;
      }
    if (j>m)
    {
      if (deplin)
      {
        GEN c = cgetg(n+1, t_COL), g0 = ff->s(E,0), g1=ff->s(E,1);
        for (i=1; i<k; i++) gel(c,i) = ff->red(E, gcoeff(x,d[i],k));
        gel(c,k) = g1; for (i=k+1; i<=n; i++) gel(c,i) = g0;
        return gerepileupto(av0, c);
      }
      r++; d[k]=0;
      for(j=1; j<k; j++)
        if (d[j]) gcoeff(x,d[j],k) = gclone(gcoeff(x,d[j],k));
    }
    else
    {
      GEN piv = ff->neg(E,ff->inv(E,gcoeff(x,j,k)));
      c[j] = k; d[k] = j;
      gcoeff(x,j,k) = ff->s(E,-1);
      for (i=k+1; i<=n; i++) gcoeff(x,j,i) = ff->red(E,ff->mul(E,piv,gcoeff(x,j,i)));
      for (t=1; t<=m; t++)
      {
        if (t==j) continue;

        piv = ff->red(E,gcoeff(x,t,k));
        if (ff->equal0(piv)) continue;

        gcoeff(x,t,k) = ff->s(E,0);
        for (i=k+1; i<=n; i++)
           gcoeff(x,t,i) = ff->red(E, ff->add(E, gcoeff(x,t,i),
                                      ff->mul(E,piv,gcoeff(x,j,i))));
        if (gc_needed(av,1))
          gen_gerepile_gauss_ker(x,k,t,av,E,ff->red);
      }
    }
  }
  if (deplin) return gc_NULL(av0);

  tetpil=avma; y=cgetg(r+1,t_MAT);
  for (j=k=1; j<=r; j++,k++)
  {
    GEN C = cgetg(n+1,t_COL);
    GEN g0 = ff->s(E,0), g1 = ff->s(E,1);
    gel(y,j) = C; while (d[k]) k++;
    for (i=1; i<k; i++)
      if (d[i])
      {
        GEN p1=gcoeff(x,d[i],k);
        gel(C,i) = ff->red(E,p1); gunclone(p1);
      }
      else
        gel(C,i) = g0;
    gel(C,k) = g1; for (i=k+1; i<=n; i++) gel(C,i) = g0;
  }
  return gerepile(av0,tetpil,y);
}

GEN
gen_Gauss_pivot(GEN x, long *rr, void *E, const struct bb_field *ff)
{
  pari_sp av;
  GEN c, d;
  long i, j, k, r, t, m, n = lg(x)-1;

  if (!n) { *rr = 0; return NULL; }

  m=nbrows(x); r=0;
  d = cgetg(n+1, t_VECSMALL);
  x = RgM_shallowcopy(x);
  c = zero_zv(m);
  av=avma;
  for (k=1; k<=n; k++)
  {
    for (j=1; j<=m; j++)
      if (!c[j])
      {
        gcoeff(x,j,k) = ff->red(E,gcoeff(x,j,k));
        if (!ff->equal0(gcoeff(x,j,k))) break;
      }
    if (j>m) { r++; d[k]=0; }
    else
    {
      GEN piv = ff->neg(E,ff->inv(E,gcoeff(x,j,k)));
      GEN g0 = ff->s(E,0);
      c[j] = k; d[k] = j;
      for (i=k+1; i<=n; i++) gcoeff(x,j,i) = ff->red(E,ff->mul(E,piv,gcoeff(x,j,i)));
      for (t=1; t<=m; t++)
      {
        if (c[t]) continue; /* already a pivot on that line */

        piv = ff->red(E,gcoeff(x,t,k));
        if (ff->equal0(piv)) continue;
        gcoeff(x,t,k) = g0;
        for (i=k+1; i<=n; i++)
          gcoeff(x,t,i) = ff->red(E, ff->add(E,gcoeff(x,t,i), ff->mul(E,piv,gcoeff(x,j,i))));
        if (gc_needed(av,1))
          gerepile_gauss(x,k,t,av,j,c);
      }
      for (i=k; i<=n; i++) gcoeff(x,j,i) = g0; /* dummy */
    }
  }
  *rr = r; return gc_const((pari_sp)d, d);
}

GEN
gen_det(GEN a, void *E, const struct bb_field *ff)
{
  pari_sp av = avma;
  long i,j,k, s = 1, nbco = lg(a)-1;
  GEN x = ff->s(E,1);
  if (!nbco) return x;
  a = RgM_shallowcopy(a);
  for (i=1; i<nbco; i++)
  {
    GEN q;
    for(k=i; k<=nbco; k++)
    {
      gcoeff(a,k,i) = ff->red(E,gcoeff(a,k,i));
      if (!ff->equal0(gcoeff(a,k,i))) break;
    }
    if (k > nbco) return gerepileupto(av, gcoeff(a,i,i));
    if (k != i)
    { /* exchange the lines s.t. k = i */
      for (j=i; j<=nbco; j++) swap(gcoeff(a,i,j), gcoeff(a,k,j));
      s = -s;
    }
    q = gcoeff(a,i,i);
    x = ff->red(E,ff->mul(E,x,q));
    q = ff->inv(E,q);
    for (k=i+1; k<=nbco; k++)
    {
      GEN m = ff->red(E,gcoeff(a,i,k));
      if (ff->equal0(m)) continue;
      m = ff->neg(E, ff->red(E,ff->mul(E,m, q)));
      for (j=i+1; j<=nbco; j++)
        gcoeff(a,j,k) = ff->red(E, ff->add(E, gcoeff(a,j,k),
                                   ff->mul(E, m, gcoeff(a,j,i))));
    }
    if (gc_needed(av,2))
    {
      if(DEBUGMEM>1) pari_warn(warnmem,"det. col = %ld",i);
      gerepileall(av,2, &a,&x);
    }
  }
  if (s < 0) x = ff->neg(E,x);
  return gerepileupto(av, ff->red(E,ff->mul(E, x, gcoeff(a,nbco,nbco))));
}

INLINE void
_gen_addmul(GEN b, long k, long i, GEN m, void *E, const struct bb_field *ff)
{
  gel(b,i) = ff->red(E,gel(b,i));
  gel(b,k) = ff->add(E,gel(b,k), ff->mul(E,m, gel(b,i)));
}

static GEN
_gen_get_col(GEN a, GEN b, long li, void *E, const struct bb_field *ff)
{
  GEN u = cgetg(li+1,t_COL);
  pari_sp av = avma;
  long i, j;

  gel(u,li) = gerepileupto(av, ff->red(E,ff->mul(E,gel(b,li), gcoeff(a,li,li))));
  for (i=li-1; i>0; i--)
  {
    pari_sp av = avma;
    GEN m = gel(b,i);
    for (j=i+1; j<=li; j++) m = ff->add(E,m, ff->neg(E,ff->mul(E,gcoeff(a,i,j), gel(u,j))));
    m = ff->red(E, m);
    gel(u,i) = gerepileupto(av, ff->red(E,ff->mul(E,m, gcoeff(a,i,i))));
  }
  return u;
}

GEN
gen_Gauss(GEN a, GEN b, void *E, const struct bb_field *ff)
{
  long i, j, k, li, bco, aco;
  GEN u, g0 = ff->s(E,0);
  pari_sp av = avma;
  a = RgM_shallowcopy(a);
  b = RgM_shallowcopy(b);
  aco = lg(a)-1; bco = lg(b)-1; li = nbrows(a);
  for (i=1; i<=aco; i++)
  {
    GEN invpiv;
    for (k = i; k <= li; k++)
    {
      GEN piv = ff->red(E,gcoeff(a,k,i));
      if (!ff->equal0(piv)) { gcoeff(a,k,i) = ff->inv(E,piv); break; }
      gcoeff(a,k,i) = g0;
    }
    /* found a pivot on line k */
    if (k > li) return NULL;
    if (k != i)
    { /* swap lines so that k = i */
      for (j=i; j<=aco; j++) swap(gcoeff(a,i,j), gcoeff(a,k,j));
      for (j=1; j<=bco; j++) swap(gcoeff(b,i,j), gcoeff(b,k,j));
    }
    if (i == aco) break;

    invpiv = gcoeff(a,i,i); /* 1/piv mod p */
    for (k=i+1; k<=li; k++)
    {
      GEN m = ff->red(E,gcoeff(a,k,i)); gcoeff(a,k,i) = g0;
      if (ff->equal0(m)) continue;

      m = ff->red(E,ff->neg(E,ff->mul(E,m, invpiv)));
      for (j=i+1; j<=aco; j++) _gen_addmul(gel(a,j),k,i,m,E,ff);
      for (j=1  ; j<=bco; j++) _gen_addmul(gel(b,j),k,i,m,E,ff);
    }
    if (gc_needed(av,1))
    {
      if(DEBUGMEM>1) pari_warn(warnmem,"gen_Gauss. i=%ld",i);
      gerepileall(av,2, &a,&b);
    }
  }

  if(DEBUGLEVEL>4) err_printf("Solving the triangular system\n");
  u = cgetg(bco+1,t_MAT);
  for (j=1; j<=bco; j++) gel(u,j) = _gen_get_col(a, gel(b,j), aco, E, ff);
  return u;
}

/* compatible t_MAT * t_COL, lgA = lg(A) = lg(B) > 1, l = lgcols(A) */
static GEN
gen_matcolmul_i(GEN A, GEN B, ulong lgA, ulong l,
                void *E, const struct bb_field *ff)
{
  GEN C = cgetg(l, t_COL);
  ulong i;
  for (i = 1; i < l; i++) {
    pari_sp av = avma;
    GEN e = ff->mul(E, gcoeff(A, i, 1), gel(B, 1));
    ulong k;
    for(k = 2; k < lgA; k++)
      e = ff->add(E, e, ff->mul(E, gcoeff(A, i, k), gel(B, k)));
    gel(C, i) = gerepileupto(av, ff->red(E, e));
  }
  return C;
}

GEN
gen_matcolmul(GEN A, GEN B, void *E, const struct bb_field *ff)
{
  ulong lgA = lg(A);
  if (lgA != (ulong)lg(B))
    pari_err_OP("operation 'gen_matcolmul'", A, B);
  if (lgA == 1)
    return cgetg(1, t_COL);
  return gen_matcolmul_i(A, B, lgA, lgcols(A), E, ff);
}

static GEN
gen_matmul_classical(GEN A, GEN B, long l, long la, long lb,
                     void *E, const struct bb_field *ff)
{
  long j;
  GEN C = cgetg(lb, t_MAT);
  for(j = 1; j < lb; j++)
    gel(C, j) = gen_matcolmul_i(A, gel(B, j), la, l, E, ff);
  return C;
}

/* Strassen-Winograd algorithm */

/*
  Return A[ma+1..ma+da, na+1..na+ea] - B[mb+1..mb+db, nb+1..nb+eb]
  as an (m x n)-matrix, padding the input with zeroes as necessary.
*/
static GEN
add_slices(long m, long n,
           GEN A, long ma, long da, long na, long ea,
           GEN B, long mb, long db, long nb, long eb,
           void *E, const struct bb_field *ff)
{
  long min_d = minss(da, db), min_e = minss(ea, eb), i, j;
  GEN M = cgetg(n + 1, t_MAT), C;

  for (j = 1; j <= min_e; j++) {
    gel(M, j) = C = cgetg(m + 1, t_COL);
    for (i = 1; i <= min_d; i++)
      gel(C, i) = ff->add(E, gcoeff(A, ma + i, na + j),
                          gcoeff(B, mb + i, nb + j));
    for (; i <= da; i++)
      gel(C, i) = gcoeff(A, ma + i, na + j);
    for (; i <= db; i++)
      gel(C, i) = gcoeff(B, mb + i, nb + j);
    for (; i <= m; i++)
      gel(C, i) = ff->s(E, 0);
  }
  for (; j <= ea; j++) {
    gel(M, j) = C = cgetg(m + 1, t_COL);
    for (i = 1; i <= da; i++)
      gel(C, i) = gcoeff(A, ma + i, na + j);
    for (; i <= m; i++)
      gel(C, i) = ff->s(E, 0);
  }
  for (; j <= eb; j++) {
    gel(M, j) = C = cgetg(m + 1, t_COL);
    for (i = 1; i <= db; i++)
      gel(C, i) = gcoeff(B, mb + i, nb + j);
    for (; i <= m; i++)
      gel(C, i) = ff->s(E, 0);
  }
  for (; j <= n; j++) {
    gel(M, j) = C = cgetg(m + 1, t_COL);
    for (i = 1; i <= m; i++)
      gel(C, i) = ff->s(E, 0);
  }
  return M;
}

/*
  Return A[ma+1..ma+da, na+1..na+ea] - B[mb+1..mb+db, nb+1..nb+eb]
  as an (m x n)-matrix, padding the input with zeroes as necessary.
*/
static GEN
subtract_slices(long m, long n,
                GEN A, long ma, long da, long na, long ea,
                GEN B, long mb, long db, long nb, long eb,
                void *E, const struct bb_field *ff)
{
  long min_d = minss(da, db), min_e = minss(ea, eb), i, j;
  GEN M = cgetg(n + 1, t_MAT), C;

  for (j = 1; j <= min_e; j++) {
    gel(M, j) = C = cgetg(m + 1, t_COL);
    for (i = 1; i <= min_d; i++)
      gel(C, i) = ff->add(E, gcoeff(A, ma + i, na + j),
                          ff->neg(E, gcoeff(B, mb + i, nb + j)));
    for (; i <= da; i++)
      gel(C, i) = gcoeff(A, ma + i, na + j);
    for (; i <= db; i++)
      gel(C, i) = ff->neg(E, gcoeff(B, mb + i, nb + j));
    for (; i <= m; i++)
      gel(C, i) = ff->s(E, 0);
  }
  for (; j <= ea; j++) {
    gel(M, j) = C = cgetg(m + 1, t_COL);
    for (i = 1; i <= da; i++)
      gel(C, i) = gcoeff(A, ma + i, na + j);
    for (; i <= m; i++)
      gel(C, i) = ff->s(E, 0);
  }
  for (; j <= eb; j++) {
    gel(M, j) = C = cgetg(m + 1, t_COL);
    for (i = 1; i <= db; i++)
      gel(C, i) = ff->neg(E, gcoeff(B, mb + i, nb + j));
    for (; i <= m; i++)
      gel(C, i) = ff->s(E, 0);
  }
  for (; j <= n; j++) {
    gel(M, j) = C = cgetg(m + 1, t_COL);
    for (i = 1; i <= m; i++)
      gel(C, i) = ff->s(E, 0);
  }
  return M;
}

static GEN gen_matmul_i(GEN A, GEN B, long l, long la, long lb,
                        void *E, const struct bb_field *ff);

static GEN
gen_matmul_sw(GEN A, GEN B, long m, long n, long p,
              void *E, const struct bb_field *ff)
{
  pari_sp av = avma;
  long m1 = (m + 1)/2, m2 = m/2,
    n1 = (n + 1)/2, n2 = n/2,
    p1 = (p + 1)/2, p2 = p/2;
  GEN A11, A12, A22, B11, B21, B22,
    S1, S2, S3, S4, T1, T2, T3, T4,
    M1, M2, M3, M4, M5, M6, M7,
    V1, V2, V3, C11, C12, C21, C22, C;

  T2 = subtract_slices(n1, p2, B, 0, n1, p1, p2, B, n1, n2, p1, p2, E, ff);
  S1 = subtract_slices(m2, n1, A, m1, m2, 0, n1, A, 0, m2, 0, n1, E, ff);
  M2 = gen_matmul_i(S1, T2, m2 + 1, n1 + 1, p2 + 1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 2, &T2, &M2);  /* destroy S1 */
  T3 = subtract_slices(n1, p1, T2, 0, n1, 0, p2, B, 0, n1, 0, p1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 2, &M2, &T3);  /* destroy T2 */
  S2 = add_slices(m2, n1, A, m1, m2, 0, n1, A, m1, m2, n1, n2, E, ff);
  T1 = subtract_slices(n1, p1, B, 0, n1, p1, p2, B, 0, n1, 0, p2, E, ff);
  M3 = gen_matmul_i(S2, T1, m2 + 1, n1 + 1, p2 + 1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 4, &M2, &T3, &S2, &M3);  /* destroy T1 */
  S3 = subtract_slices(m1, n1, S2, 0, m2, 0, n1, A, 0, m1, 0, n1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 4, &M2, &T3, &M3, &S3);  /* destroy S2 */
  A11 = matslice(A, 1, m1, 1, n1);
  B11 = matslice(B, 1, n1, 1, p1);
  M1 = gen_matmul_i(A11, B11, m1 + 1, n1 + 1, p1 + 1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 5, &M2, &T3, &M3, &S3, &M1);  /* destroy A11, B11 */
  A12 = matslice(A, 1, m1, n1 + 1, n);
  B21 = matslice(B, n1 + 1, n, 1, p1);
  M4 = gen_matmul_i(A12, B21, m1 + 1, n2 + 1, p1 + 1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 6, &M2, &T3, &M3, &S3, &M1, &M4);  /* destroy A12, B21 */
  C11 = add_slices(m1, p1, M1, 0, m1, 0, p1, M4, 0, m1, 0, p1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 6, &M2, &T3, &M3, &S3, &M1, &C11);  /* destroy M4 */
  M5 = gen_matmul_i(S3, T3, m1 + 1, n1 + 1, p1 + 1, E, ff);
  S4 = subtract_slices(m1, n2, A, 0, m1, n1, n2, S3, 0, m1, 0, n2, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 7, &M2, &T3, &M3, &M1, &C11, &M5, &S4);  /* destroy S3 */
  T4 = add_slices(n2, p1, B, n1, n2, 0, p1, T3, 0, n2, 0, p1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 7, &M2, &M3, &M1, &C11, &M5, &S4, &T4);  /* destroy T3 */
  V1 = subtract_slices(m1, p1, M1, 0, m1, 0, p1, M5, 0, m1, 0, p1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 6, &M2, &M3, &S4, &T4, &C11, &V1);  /* destroy M1, M5 */
  B22 = matslice(B, n1 + 1, n, p1 + 1, p);
  M6 = gen_matmul_i(S4, B22, m1 + 1, n2 + 1, p2 + 1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 6, &M2, &M3, &T4, &C11, &V1, &M6);  /* destroy S4, B22 */
  A22 = matslice(A, m1 + 1, m, n1 + 1, n);
  M7 = gen_matmul_i(A22, T4, m2 + 1, n2 + 1, p1 + 1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 6, &M2, &M3, &C11, &V1, &M6, &M7);  /* destroy A22, T4 */
  V3 = add_slices(m1, p2, V1, 0, m1, 0, p2, M3, 0, m2, 0, p2, E, ff);
  C12 = add_slices(m1, p2, V3, 0, m1, 0, p2, M6, 0, m1, 0, p2, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 6, &M2, &M3, &C11, &V1, &M7, &C12);  /* destroy V3, M6 */
  V2 = add_slices(m2, p1, V1, 0, m2, 0, p1, M2, 0, m2, 0, p2, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 5, &M3, &C11, &M7, &C12, &V2);  /* destroy V1, M2 */
  C21 = add_slices(m2, p1, V2, 0, m2, 0, p1, M7, 0, m2, 0, p1, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 5, &M3, &C11, &C12, &V2, &C21);  /* destroy M7 */
  C22 = add_slices(m2, p2, V2, 0, m2, 0, p2, M3, 0, m2, 0, p2, E, ff);
  if (gc_needed(av, 1))
    gerepileall(av, 4, &C11, &C12, &C21, &C22);  /* destroy V2, M3 */
  C = mkmat2(mkcol2(C11, C21), mkcol2(C12, C22));
  return gerepileupto(av, matconcat(C));
}

/* Strassen-Winograd used for dim >= gen_matmul_sw_bound */
static const long gen_matmul_sw_bound = 24;

static GEN
gen_matmul_i(GEN A, GEN B, long l, long la, long lb,
             void *E, const struct bb_field *ff)
{
  if (l <= gen_matmul_sw_bound
      || la <= gen_matmul_sw_bound
      || lb <= gen_matmul_sw_bound)
    return gen_matmul_classical(A, B, l, la, lb, E, ff);
  else
    return gen_matmul_sw(A, B, l - 1, la - 1, lb - 1, E, ff);
}

GEN
gen_matmul(GEN A, GEN B, void *E, const struct bb_field *ff)
{
  ulong lgA, lgB = lg(B);
  if (lgB == 1)
    return cgetg(1, t_MAT);
  lgA = lg(A);
  if (lgA != (ulong)lgcols(B))
    pari_err_OP("operation 'gen_matmul'", A, B);
  if (lgA == 1)
    return zeromat(0, lgB - 1);
  return gen_matmul_i(A, B, lgcols(A), lgA, lgB, E, ff);
}

static GEN
gen_colneg(GEN A, void *E, const struct bb_field *ff)
{
  long i, l;
  GEN B = cgetg_copy(A, &l);
  for (i = 1; i < l; i++)
    gel(B, i) = ff->neg(E, gel(A, i));
  return B;
}

static GEN
gen_matneg(GEN A, void *E, const struct bb_field *ff)
{
  long i, l;
  GEN B = cgetg_copy(A, &l);
  for (i = 1; i < l; i++)
    gel(B, i) = gen_colneg(gel(A, i), E, ff);
  return B;
}

static GEN
gen_colscalmul(GEN A, GEN b, void *E, const struct bb_field *ff)
{
  long i, l;
  GEN B = cgetg_copy(A, &l);
  for (i = 1; i < l; i++)
    gel(B, i) = ff->red(E, ff->mul(E, gel(A, i), b));
  return B;
}

static GEN
gen_matscalmul(GEN A, GEN b, void *E, const struct bb_field *ff)
{
  long i, l;
  GEN B = cgetg_copy(A, &l);
  for (i = 1; i < l; i++)
    gel(B, i) = gen_colscalmul(gel(A, i), b, E, ff);
  return B;
}

static GEN
gen_colsub(GEN A, GEN C, void *E, const struct bb_field *ff)
{
  long i, l;
  GEN B = cgetg_copy(A, &l);
  for (i = 1; i < l; i++)
    gel(B, i) = ff->add(E, gel(A, i), ff->neg(E, gel(C, i)));
  return B;
}

static GEN
gen_matsub(GEN A, GEN C, void *E, const struct bb_field *ff)
{
  long i, l;
  GEN B = cgetg_copy(A, &l);
  for (i = 1; i < l; i++)
    gel(B, i) = gen_colsub(gel(A, i), gel(C, i), E, ff);
  return B;
}

static GEN
gen_zerocol(long n, void* data, const struct bb_field *R)
{
  GEN C = cgetg(n+1,t_COL), zero = R->s(data, 0);
  long i;
  for (i=1; i<=n; i++) gel(C,i) = zero;
  return C;
}

static GEN
gen_zeromat(long m, long n, void* data, const struct bb_field *R)
{
  GEN M = cgetg(n+1,t_MAT);
  long i;
  for (i=1; i<=n; i++) gel(M,i) = gen_zerocol(m, data, R);
  return M;
}

static GEN
gen_colei(long n, long i, void *E, const struct bb_field *S)
{
  GEN y = cgetg(n+1,t_COL), _0, _1;
  long j;
  if (n < 0) pari_err_DOMAIN("gen_colei", "dimension","<",gen_0,stoi(n));
  _0 = S->s(E,0);
  _1 = S->s(E,1);
  for (j=1; j<=n; j++)
    gel(y, j) = i==j ? _1: _0;
  return y;
}

/* assume dim A >= 1, A invertible + upper triangular  */
static GEN
gen_matinv_upper_ind(GEN A, long index, void *E, const struct bb_field *ff)
{
  long n = lg(A) - 1, i, j;
  GEN u = cgetg(n + 1, t_COL);
  for (i = n; i > index; i--)
    gel(u, i) = ff->s(E, 0);
  gel(u, i) = ff->inv(E, gcoeff(A, i, i));
  for (i--; i > 0; i--) {
    pari_sp av = avma;
    GEN m = ff->neg(E, ff->mul(E, gcoeff(A, i, i + 1), gel(u, i + 1)));
    for (j = i + 2; j <= n; j++)
      m = ff->add(E, m, ff->neg(E, ff->mul(E, gcoeff(A, i, j), gel(u, j))));
    gel(u, i) = gerepileupto(av, ff->red(E, ff->mul(E, m, ff->inv(E, gcoeff(A, i, i)))));
  }
  return u;
}

static GEN
gen_matinv_upper(GEN A, void *E, const struct bb_field *ff)
{
  long i, l;
  GEN B = cgetg_copy(A, &l);
  for (i = 1; i < l; i++)
    gel(B,i) = gen_matinv_upper_ind(A, i, E, ff);
  return B;
}

/* find z such that A z = y. Return NULL if no solution */
GEN
gen_matcolinvimage(GEN A, GEN y, void *E, const struct bb_field *ff)
{
  pari_sp av = avma;
  long i, l = lg(A);
  GEN M, x, t;

  M = gen_ker(shallowconcat(A, y), 0, E, ff);
  i = lg(M) - 1;
  if (!i) return gc_NULL(av);

  x = gel(M, i);
  t = gel(x, l);
  if (ff->equal0(t)) return gc_NULL(av);

  t = ff->neg(E, ff->inv(E, t));
  setlg(x, l);
  for (i = 1; i < l; i++)
    gel(x, i) = ff->red(E, ff->mul(E, t, gel(x, i)));
  return gerepilecopy(av, x);
}

/* find Z such that A Z = B. Return NULL if no solution */
GEN
gen_matinvimage(GEN A, GEN B, void *E, const struct bb_field *ff)
{
  pari_sp av = avma;
  GEN d, x, X, Y;
  long i, j, nY, nA, nB;
  x = gen_ker(shallowconcat(gen_matneg(A, E, ff), B), 0, E, ff);
  /* AX = BY, Y in strict upper echelon form with pivots = 1.
   * We must find T such that Y T = Id_nB then X T = Z. This exists
   * iff Y has at least nB columns and full rank. */
  nY = lg(x) - 1;
  nB = lg(B) - 1;
  if (nY < nB) return gc_NULL(av);
  nA = lg(A) - 1;
  Y = rowslice(x, nA + 1, nA + nB); /* nB rows */
  d = cgetg(nB + 1, t_VECSMALL);
  for (i = nB, j = nY; i >= 1; i--, j--) {
    for (; j >= 1; j--)
      if (!ff->equal0(gcoeff(Y, i, j))) { d[i] = j; break; }
    if (!j) return gc_NULL(av);
  }
  /* reduce to the case Y square, upper triangular with 1s on diagonal */
  Y = vecpermute(Y, d);
  x = vecpermute(x, d);
  X = rowslice(x, 1, nA);
  return gerepileupto(av, gen_matmul(X, gen_matinv_upper(Y, E, ff), E, ff));
}

static GEN
image_from_pivot(GEN x, GEN d, long r)
{
  GEN y;
  long j, k;

  if (!d) return gcopy(x);
  /* d left on stack for efficiency */
  r = lg(x)-1 - r; /* = dim Im(x) */
  y = cgetg(r+1,t_MAT);
  for (j=k=1; j<=r; k++)
    if (d[k]) gel(y,j++) = gcopy(gel(x,k));
  return y;
}

/* r = dim Ker x, n = nbrows(x) */
static GEN
get_suppl(GEN x, GEN d, long n, long r, GEN(*ei)(long,long))
{
  pari_sp av;
  GEN y, c;
  long j, k, rx = lg(x)-1; /* != 0 due to init_suppl() */

  if (rx == n && r == 0) return gcopy(x);
  y = cgetg(n+1, t_MAT);
  av = avma; c = zero_zv(n);
  /* c = lines containing pivots (could get it from gauss_pivot, but cheap)
   * In theory r = 0 and d[j] > 0 for all j, but why take chances? */
  for (k = j = 1; j<=rx; j++)
    if (d[j]) { c[ d[j] ] = 1; gel(y,k++) = gel(x,j); }
  for (j=1; j<=n; j++)
    if (!c[j]) gel(y,k++) = (GEN)j; /* HACK */
  set_avma(av);

  rx -= r;
  for (j=1; j<=rx; j++) gel(y,j) = gcopy(gel(y,j));
  for (   ; j<=n; j++)  gel(y,j) = ei(n, y[j]);
  return y;
}

/* n = dim x, r = dim Ker(x), d from gauss_pivot */
static GEN
indexrank0(long n, long r, GEN d)
{
  GEN p1, p2, res = cgetg(3,t_VEC);
  long i, j;

  r = n - r; /* now r = dim Im(x) */
  p1 = cgetg(r+1,t_VECSMALL); gel(res,1) = p1;
  p2 = cgetg(r+1,t_VECSMALL); gel(res,2) = p2;
  if (d)
  {
    for (i=0,j=1; j<=n; j++)
      if (d[j]) { i++; p1[i] = d[j]; p2[i] = j; }
    vecsmall_sort(p1);
  }
  return res;
}

/*******************************************************************/
/*                                                                 */
/*                Echelon form and CUP decomposition               */
/*                                                                 */
/*******************************************************************/

/* By Peter Bruin, based on
  C.-P. Jeannerod, C. Pernet and A. Storjohann, Rank-profile revealing
  Gaussian elimination and the CUP matrix decomposition.  J. Symbolic
  Comput. 56 (2013), 46-68.

  Decompose an m x n-matrix A of rank r as C*U*P, with
  - C: m x r-matrix in column echelon form (not necessarily reduced)
       with all pivots equal to 1
  - U: upper-triangular r x n-matrix
  - P: permutation matrix
  The pivots of C and the known zeroes in C and U are not necessarily
  filled in; instead, we also return the vector R of pivot rows.
  Instead of the matrix P, we return the permutation p of [1..n]
  (t_VECSMALL) such that P[i,j] = 1 if and only if j = p[i].
*/

/* complement of a strictly increasing subsequence of (1, 2, ..., n) */
static GEN
indexcompl(GEN v, long n)
{
  long i, j, k, m = lg(v) - 1;
  GEN w = cgetg(n - m + 1, t_VECSMALL);
  for (i = j = k = 1; i <= n; i++)
    if (j <= m && v[j] == i) j++; else w[k++] = i;
  return w;
}

static GEN
gen_solve_upper_1(GEN U, GEN B, void *E, const struct bb_field *ff)
{ return gen_matscalmul(B, ff->inv(E, gcoeff(U, 1, 1)), E, ff); }

static GEN
gen_rsolve_upper_2(GEN U, GEN B, void *E, const struct bb_field *ff)
{
  GEN a = gcoeff(U, 1, 1), b = gcoeff(U, 1, 2), d = gcoeff(U, 2, 2);
  GEN D = ff->red(E, ff->mul(E, a, d)), Dinv = ff->inv(E, D);
  GEN ainv = ff->red(E, ff->mul(E, d, Dinv));
  GEN dinv = ff->red(E, ff->mul(E, a, Dinv));
  GEN B1 = rowslice(B, 1, 1);
  GEN B2 = rowslice(B, 2, 2);
  GEN X2 = gen_matscalmul(B2, dinv, E, ff);
  GEN X1 = gen_matscalmul(gen_matsub(B1, gen_matscalmul(X2, b, E, ff), E, ff),
                          ainv, E, ff);
  return vconcat(X1, X2);
}

/* solve U*X = B,  U upper triangular and invertible */
static GEN
gen_rsolve_upper(GEN U, GEN B, void *E, const struct bb_field *ff,
                 GEN (*mul)(void *E, GEN a, GEN))
{
  long n = lg(U) - 1, n1;
  GEN U2, U11, U12, U22, B1, B2, X1, X2, X;
  pari_sp av = avma;

  if (n == 0) return B;
  if (n == 1) return gen_solve_upper_1(U, B, E, ff);
  if (n == 2) return gen_rsolve_upper_2(U, B, E, ff);
  n1 = (n + 1)/2;
  U2 = vecslice(U, n1 + 1, n);
  U11 = matslice(U, 1,n1, 1,n1);
  U12 = rowslice(U2, 1, n1);
  U22 = rowslice(U2, n1 + 1, n);
  B1 = rowslice(B, 1, n1);
  B2 = rowslice(B, n1 + 1, n);
  X2 = gen_rsolve_upper(U22, B2, E, ff, mul);
  B1 = gen_matsub(B1, mul(E, U12, X2), E, ff);
  if (gc_needed(av, 1)) gerepileall(av, 3, &B1, &U11, &X2);
  X1 = gen_rsolve_upper(U11, B1, E, ff, mul);
  X = vconcat(X1, X2);
  if (gc_needed(av, 1)) X = gerepilecopy(av, X);
  return X;
}

static GEN
gen_lsolve_upper_2(GEN U, GEN B, void *E, const struct bb_field *ff)
{
  GEN a = gcoeff(U, 1, 1), b = gcoeff(U, 1, 2), d = gcoeff(U, 2, 2);
  GEN D = ff->red(E, ff->mul(E, a, d)), Dinv = ff->inv(E, D);
  GEN ainv = ff->red(E, ff->mul(E, d, Dinv)), dinv = ff->red(E, ff->mul(E, a, Dinv));
  GEN B1 = vecslice(B, 1, 1);
  GEN B2 = vecslice(B, 2, 2);
  GEN X1 = gen_matscalmul(B1, ainv, E, ff);
  GEN X2 = gen_matscalmul(gen_matsub(B2, gen_matscalmul(X1, b, E, ff), E, ff), dinv, E, ff);
  return shallowconcat(X1, X2);
}

/* solve X*U = B,  U upper triangular and invertible */
static GEN
gen_lsolve_upper(GEN U, GEN B, void *E, const struct bb_field *ff,
                 GEN (*mul)(void *E, GEN a, GEN))
{
  long n = lg(U) - 1, n1;
  GEN U2, U11, U12, U22, B1, B2, X1, X2, X;
  pari_sp av = avma;

  if (n == 0) return B;
  if (n == 1) return gen_solve_upper_1(U, B, E, ff);
  if (n == 2) return gen_lsolve_upper_2(U, B, E, ff);
  n1 = (n + 1)/2;
  U2 = vecslice(U, n1 + 1, n);
  U11 = matslice(U, 1,n1, 1,n1);
  U12 = rowslice(U2, 1, n1);
  U22 = rowslice(U2, n1 + 1, n);
  B1 = vecslice(B, 1, n1);
  B2 = vecslice(B, n1 + 1, n);
  X1 = gen_lsolve_upper(U11, B1, E, ff, mul);
  B2 = gen_matsub(B2, mul(E, X1, U12), E, ff);
  if (gc_needed(av, 1)) gerepileall(av, 3, &B2, &U22, &X1);
  X2 = gen_lsolve_upper(U22, B2, E, ff, mul);
  X = shallowconcat(X1, X2);
  if (gc_needed(av, 1)) X = gerepilecopy(av, X);
  return X;
}

static GEN
gen_rsolve_lower_unit_2(GEN L, GEN A, void *E, const struct bb_field *ff)
{
  GEN X1 = rowslice(A, 1, 1);
  GEN X2 = gen_matsub(rowslice(A, 2, 2), gen_matscalmul(X1, gcoeff(L, 2, 1), E, ff), E, ff);
  return vconcat(X1, X2);
}

/* solve L*X = A,  L lower triangular with ones on the diagonal
 * (at least as many rows as columns) */
static GEN
gen_rsolve_lower_unit(GEN L, GEN A, void *E, const struct bb_field *ff,
                      GEN (*mul)(void *E, GEN a, GEN))
{
  long m = lg(L) - 1, m1, n;
  GEN L1, L11, L21, L22, A1, A2, X1, X2, X;
  pari_sp av = avma;

  if (m == 0) return zeromat(0, lg(A) - 1);
  if (m == 1) return rowslice(A, 1, 1);
  if (m == 2) return gen_rsolve_lower_unit_2(L, A, E, ff);
  m1 = (m + 1)/2;
  n = nbrows(L);
  L1 = vecslice(L, 1, m1);
  L11 = rowslice(L1, 1, m1);
  L21 = rowslice(L1, m1 + 1, n);
  A1 = rowslice(A, 1, m1);
  X1 = gen_rsolve_lower_unit(L11, A1, E, ff, mul);
  A2 = rowslice(A, m1 + 1, n);
  A2 = gen_matsub(A2, mul(E, L21, X1), E, ff);
  if (gc_needed(av, 1)) gerepileall(av, 2, &A2, &X1);
  L22 = matslice(L, m1+1,n, m1+1,m);
  X2 = gen_rsolve_lower_unit(L22, A2, E, ff, mul);
  X = vconcat(X1, X2);
  if (gc_needed(av, 1)) X = gerepilecopy(av, X);
  return X;
}

static GEN
gen_lsolve_lower_unit_2(GEN L, GEN A, void *E, const struct bb_field *ff)
{
  GEN X2 = vecslice(A, 2, 2);
  GEN X1 = gen_matsub(vecslice(A, 1, 1),
                    gen_matscalmul(X2, gcoeff(L, 2, 1), E, ff), E, ff);
  return shallowconcat(X1, X2);
}

/* solve L*X = A,  L lower triangular with ones on the diagonal
 * (at least as many rows as columns) */
static GEN
gen_lsolve_lower_unit(GEN L, GEN A, void *E, const struct bb_field *ff,
                      GEN (*mul)(void *E, GEN a, GEN))
{
  long m = lg(L) - 1, m1;
  GEN L1, L2, L11, L21, L22, A1, A2, X1, X2, X;
  pari_sp av = avma;

  if (m <= 1) return A;
  if (m == 2) return gen_lsolve_lower_unit_2(L, A, E, ff);
  m1 = (m + 1)/2;
  L2 = vecslice(L, m1 + 1, m);
  L22 = rowslice(L2, m1 + 1, m);
  A2 = vecslice(A, m1 + 1, m);
  X2 = gen_lsolve_lower_unit(L22, A2, E, ff, mul);
  if (gc_needed(av, 1)) X2 = gerepilecopy(av, X2);
  L1 = vecslice(L, 1, m1);
  L21 = rowslice(L1, m1 + 1, m);
  A1 = vecslice(A, 1, m1);
  A1 = gen_matsub(A1, mul(E, X2, L21), E, ff);
  L11 = rowslice(L1, 1, m1);
  if (gc_needed(av, 1)) gerepileall(av, 3, &A1, &L11, &X2);
  X1 = gen_lsolve_lower_unit(L11, A1, E, ff, mul);
  X = shallowconcat(X1, X2);
  if (gc_needed(av, 1)) X = gerepilecopy(av, X);
  return X;
}

/* destroy A */
static long
gen_CUP_basecase(GEN A, GEN *R, GEN *C, GEN *U, GEN *P, void *E, const struct bb_field *ff)
{
  long i, j, k, m = nbrows(A), n = lg(A) - 1, pr, pc;
  pari_sp av;
  GEN u, v;

  if (P) *P = identity_perm(n);
  *R = cgetg(m + 1, t_VECSMALL);
  av = avma;
  for (j = 1, pr = 0; j <= n; j++)
  {
    for (pr++, pc = 0; pr <= m; pr++)
    {
      for (k = j; k <= n; k++)
      {
        v = ff->red(E, gcoeff(A, pr, k));
        gcoeff(A, pr, k) = v;
        if (!pc && !ff->equal0(v)) pc = k;
      }
      if (pc) break;
    }
    if (!pc) break;
    (*R)[j] = pr;
    if (pc != j)
    {
      swap(gel(A, j), gel(A, pc));
      if (P) lswap((*P)[j], (*P)[pc]);
    }
    u = ff->inv(E, gcoeff(A, pr, j));
    for (i = pr + 1; i <= m; i++)
    {
      v = ff->red(E, ff->mul(E, gcoeff(A, i, j), u));
      gcoeff(A, i, j) = v;
      v = ff->neg(E, v);
      for (k = j + 1; k <= n; k++)
        gcoeff(A, i, k) = ff->add(E, gcoeff(A, i, k),
                                  ff->red(E, ff->mul(E, gcoeff(A, pr, k), v)));
    }
    if (gc_needed(av, 2)) A = gerepilecopy(av, A);
  }
  setlg(*R, j);
  *C = vecslice(A, 1, j - 1);
  if (U) *U = rowpermute(A, *R);
  return j - 1;
}

static const long gen_CUP_LIMIT = 5;

static long
gen_CUP(GEN A, GEN *R, GEN *C, GEN *U, GEN *P, void *E, const struct bb_field *ff,
        GEN (*mul)(void *E, GEN a, GEN))
{
  long m = nbrows(A), m1, n = lg(A) - 1, i, r1, r2, r;
  GEN R1, C1, U1, P1, R2, C2, U2, P2;
  GEN A1, A2, B2, C21, U11, U12, T21, T22;
  pari_sp av = avma;

  if (m < gen_CUP_LIMIT || n < gen_CUP_LIMIT)
    /* destroy A; not called at the outermost recursion level */
    return gen_CUP_basecase(A, R, C, U, P, E, ff);
  m1 = (minss(m, n) + 1)/2;
  A1 = rowslice(A, 1, m1);
  A2 = rowslice(A, m1 + 1, m);
  r1 = gen_CUP(A1, &R1, &C1, &U1, &P1, E, ff, mul);
  if (r1 == 0)
  {
    r2 = gen_CUP(A2, &R2, &C2, &U2, &P2, E, ff, mul);
    *R = cgetg(r2 + 1, t_VECSMALL);
    for (i = 1; i <= r2; i++) (*R)[i] = R2[i] + m1;
    *C = vconcat(gen_zeromat(m1, r2, E, ff), C2);
    *U = U2;
    *P = P2;
    r = r2;
  }
  else
  {
    U11 = vecslice(U1, 1, r1);
    U12 = vecslice(U1, r1 + 1, n);
    T21 = vecslicepermute(A2, P1, 1, r1);
    T22 = vecslicepermute(A2, P1, r1 + 1, n);
    C21 = gen_lsolve_upper(U11, T21, E, ff, mul);
    if (gc_needed(av, 1))
      gerepileall(av, 7, &R1, &C1, &P1, &U11, &U12, &T22, &C21);
    B2 = gen_matsub(T22, mul(E, C21, U12), E, ff);
    r2 = gen_CUP(B2, &R2, &C2, &U2, &P2, E, ff, mul);
    r = r1 + r2;
    *R = cgetg(r + 1, t_VECSMALL);
    for (i = 1; i <= r1; i++) (*R)[i] = R1[i];
    for (     ; i <= r; i++)  (*R)[i] = R2[i - r1] + m1;
    *C = shallowconcat(vconcat(C1, C21),
                       vconcat(gen_zeromat(m1, r2, E, ff), C2));
    *U = shallowconcat(vconcat(U11, gen_zeromat(r2, r1, E, ff)),
                       vconcat(vecpermute(U12, P2), U2));

    *P = cgetg(n + 1, t_VECSMALL);
    for (i = 1; i <= r1; i++) (*P)[i] = P1[i];
    for (     ; i <= n; i++)  (*P)[i] = P1[P2[i - r1] + r1];
  }
  if (gc_needed(av, 1)) gerepileall(av, 4, R, C, U, P);
  return r;
}

/* column echelon form */
static long
gen_echelon(GEN A, GEN *R, GEN *C, void *E, const struct bb_field *ff,
            GEN (*mul)(void*, GEN, GEN))
{
  long j, j1, j2, m = nbrows(A), n = lg(A) - 1, n1, r, r1, r2;
  GEN A1, A2, R1, R1c, C1, R2, C2;
  GEN A12, A22, B2, C11, C21, M12;
  pari_sp av = avma;

  if (m < gen_CUP_LIMIT || n < gen_CUP_LIMIT)
    return gen_CUP_basecase(shallowcopy(A), R, C, NULL, NULL, E, ff);

  n1 = (n + 1)/2;
  A1 = vecslice(A, 1, n1);
  A2 = vecslice(A, n1 + 1, n);
  r1 = gen_echelon(A1, &R1, &C1, E, ff, mul);
  if (!r1) return gen_echelon(A2, R, C, E, ff, mul);
  if (r1 == m) { *R = R1; *C = C1; return r1; }
  R1c = indexcompl(R1, m);
  C11 = rowpermute(C1, R1);
  C21 = rowpermute(C1, R1c);
  A12 = rowpermute(A2, R1);
  A22 = rowpermute(A2, R1c);
  M12 = gen_rsolve_lower_unit(C11, A12, E, ff, mul);
  B2 = gen_matsub(A22, mul(E, C21, M12), E, ff);
  r2 = gen_echelon(B2, &R2, &C2, E, ff, mul);
  if (!r2) { *R = R1; *C = C1; r = r1; }
  else
  {
    R2 = perm_mul(R1c, R2);
    C2 = rowpermute(vconcat(gen_zeromat(r1, r2, E, ff), C2),
                    perm_inv(vecsmall_concat(R1, R1c)));
    r = r1 + r2;
    *R = cgetg(r + 1, t_VECSMALL);
    *C = cgetg(r + 1, t_MAT);
    for (j = j1 = j2 = 1; j <= r; j++)
      if (j2 > r2 || (j1 <= r1 && R1[j1] < R2[j2]))
      {
        gel(*C, j) = gel(C1, j1);
        (*R)[j] = R1[j1++];
      }
      else
      {
        gel(*C, j) = gel(C2, j2);
        (*R)[j] = R2[j2++];
      }
  }
  if (gc_needed(av, 1)) gerepileall(av, 2, R, C);
  return r;
}

static GEN
gen_pivots_CUP(GEN x, long *rr, void *E, const struct bb_field *ff,
               GEN (*mul)(void*, GEN, GEN))
{
  pari_sp av;
  long i, n = lg(x) - 1, r;
  GEN R, C, U, P, d = zero_zv(n);
  av = avma;
  r = gen_CUP(x, &R, &C, &U, &P, E, ff, mul);
  for(i = 1; i <= r; i++)
    d[P[i]] = R[i];
  set_avma(av);
  *rr = n - r;
  return d;
}

static GEN
gen_det_CUP(GEN a, void *E, const struct bb_field *ff,
            GEN (*mul)(void*, GEN, GEN))
{
  pari_sp av = avma;
  GEN R, C, U, P, d;
  long i, n = lg(a) - 1, r;
  r = gen_CUP(a, &R, &C, &U, &P, E, ff, mul);
  if (r < n)
    d = ff->s(E, 0);
  else {
    d = ff->s(E, perm_sign(P) == 1 ? 1: - 1);
    for (i = 1; i <= n; i++)
      d = ff->red(E, ff->mul(E, d, gcoeff(U, i, i)));
  }
  return gerepileupto(av, d);
}

static long
gen_matrank(GEN x, void *E, const struct bb_field *ff,
            GEN (*mul)(void*, GEN, GEN))
{
  pari_sp av = avma;
  long r;
  if (lg(x) - 1 >= gen_CUP_LIMIT && nbrows(x) >= gen_CUP_LIMIT)
  {
    GEN R, C;
    return gc_long(av, gen_echelon(x, &R, &C, E, ff, mul));
  }
  (void) gen_Gauss_pivot(x, &r, E, ff);
  return gc_long(av, lg(x)-1 - r);
}

static GEN
gen_invimage_CUP(GEN A, GEN B, void *E, const struct bb_field *ff,
                 GEN (*mul)(void*, GEN, GEN))
{
  pari_sp av = avma;
  GEN R, Rc, C, U, P, B1, B2, C1, C2, X, Y, Z;
  long r = gen_CUP(A, &R, &C, &U, &P, E, ff, mul);
  Rc = indexcompl(R, nbrows(B));
  C1 = rowpermute(C, R);
  C2 = rowpermute(C, Rc);
  B1 = rowpermute(B, R);
  B2 = rowpermute(B, Rc);
  Z = gen_rsolve_lower_unit(C1, B1, E, ff, mul);
  if (!gequal(mul(E, C2, Z), B2))
    return NULL;
  Y = vconcat(gen_rsolve_upper(vecslice(U, 1, r), Z, E, ff, mul),
              gen_zeromat(lg(A) - 1 - r, lg(B) - 1, E, ff));
  X = rowpermute(Y, perm_inv(P));
  return gerepilecopy(av, X);
}

static GEN
gen_ker_echelon(GEN x, void *E, const struct bb_field *ff,
                GEN (*mul)(void*, GEN, GEN))
{
  pari_sp av = avma;
  GEN R, Rc, C, C1, C2, S, K;
  long n = lg(x) - 1, r;
  r = gen_echelon(shallowtrans(x), &R, &C, E, ff, mul);
  Rc = indexcompl(R, n);
  C1 = rowpermute(C, R);
  C2 = rowpermute(C, Rc);
  S = gen_lsolve_lower_unit(C1, C2, E, ff, mul);
  K = vecpermute(shallowconcat(gen_matneg(S, E, ff), gen_matid(n - r, E, ff)),
                 perm_inv(vecsmall_concat(R, Rc)));
  K = shallowtrans(K);
  return gerepilecopy(av, K);
}

static GEN
gen_deplin_echelon(GEN x, void *E, const struct bb_field *ff,
                   GEN (*mul)(void*, GEN, GEN))
{
  pari_sp av = avma;
  GEN R, Rc, C, C1, C2, s, v;
  long i, n = lg(x) - 1, r;
  r = gen_echelon(shallowtrans(x), &R, &C, E, ff, mul);
  if (r == n) return gc_NULL(av);
  Rc = indexcompl(R, n);
  i = Rc[1];
  C1 = rowpermute(C, R);
  C2 = rowslice(C, i, i);
  s = row(gen_lsolve_lower_unit(C1, C2, E, ff, mul), 1);
  settyp(s, t_COL);
  v = vecpermute(shallowconcat(gen_colneg(s, E, ff), gen_colei(n - r, 1, E, ff)),
                 perm_inv(vecsmall_concat(R, Rc)));
  return gerepilecopy(av, v);
}

static GEN
gen_gauss_CUP(GEN a, GEN b, void *E, const struct bb_field *ff,
              GEN (*mul)(void*, GEN, GEN))
{
  GEN R, C, U, P, Y;
  long n = lg(a) - 1, r;
  if (nbrows(a) < n || (r = gen_CUP(a, &R, &C, &U, &P, E, ff, mul)) < n)
    return NULL;
  Y = gen_rsolve_lower_unit(rowpermute(C, R), rowpermute(b, R), E, ff, mul);
  return rowpermute(gen_rsolve_upper(U, Y, E, ff, mul), perm_inv(P));
}

static GEN
gen_gauss(GEN a, GEN b, void *E, const struct bb_field *ff,
          GEN (*mul)(void*, GEN, GEN))
{
  if (lg(a) - 1 >= gen_CUP_LIMIT)
    return gen_gauss_CUP(a, b, E, ff, mul);
  return gen_Gauss(a, b, E, ff);
}

static GEN
gen_ker_i(GEN x, long deplin, void *E, const struct bb_field *ff,
          GEN (*mul)(void*, GEN, GEN)) {
  if (lg(x) - 1 >= gen_CUP_LIMIT && nbrows(x) >= gen_CUP_LIMIT)
    return deplin? gen_deplin_echelon(x, E, ff, mul): gen_ker_echelon(x, E, ff, mul);
  return gen_ker(x, deplin, E, ff);
}

static GEN
gen_invimage(GEN A, GEN B, void *E, const struct bb_field *ff,
             GEN (*mul)(void*, GEN, GEN))
{
  long nA = lg(A)-1, nB = lg(B)-1;

  if (!nB) return cgetg(1, t_MAT);
  if (nA + nB >= gen_CUP_LIMIT && nbrows(B) >= gen_CUP_LIMIT)
    return gen_invimage_CUP(A, B, E, ff, mul);
  return gen_matinvimage(A, B, E, ff);
}

/* find z such that A z = y. Return NULL if no solution */
static GEN
gen_matcolinvimage_i(GEN A, GEN y, void *E, const struct bb_field *ff,
                     GEN (*mul)(void*, GEN, GEN))
{
  pari_sp av = avma;
  long i, l = lg(A);
  GEN M, x, t;

  M = gen_ker_i(shallowconcat(A, y), 0, E, ff, mul);
  i = lg(M) - 1;
  if (!i) return gc_NULL(av);

  x = gel(M, i);
  t = gel(x, l);
  if (ff->equal0(t)) return gc_NULL(av);

  t = ff->neg(E, ff->inv(E, t));
  setlg(x, l);
  for (i = 1; i < l; i++)
    gel(x, i) = ff->red(E, ff->mul(E, t, gel(x, i)));
  return gerepilecopy(av, x);
}

static GEN
gen_det_i(GEN a, void *E, const struct bb_field *ff,
          GEN (*mul)(void*, GEN, GEN))
{
  if (lg(a) - 1 >= gen_CUP_LIMIT)
    return gen_det_CUP(a, E, ff, mul);
  else
    return gen_det(a, E, ff);
}

static GEN
gen_pivots(GEN x, long *rr, void *E, const struct bb_field *ff,
           GEN (*mul)(void*, GEN, GEN))
{
  if (lg(x) - 1 >= gen_CUP_LIMIT && nbrows(x) >= gen_CUP_LIMIT)
    return gen_pivots_CUP(x, rr, E, ff, mul);
  return gen_Gauss_pivot(x, rr, E, ff);
}

/* r = dim Ker x, n = nbrows(x) */
static GEN
gen_get_suppl(GEN x, GEN d, long n, long r, void *E, const struct bb_field *ff)
{
  GEN y, c;
  long j, k, rx = lg(x)-1; /* != 0 due to init_suppl() */

  if (rx == n && r == 0) return gcopy(x);
  c = zero_zv(n);
  y = cgetg(n+1, t_MAT);
  /* c = lines containing pivots (could get it from gauss_pivot, but cheap)
   * In theory r = 0 and d[j] > 0 for all j, but why take chances? */
  for (k = j = 1; j<=rx; j++)
    if (d[j]) { c[ d[j] ] = 1; gel(y,k++) = gcopy(gel(x,j)); }
  for (j=1; j<=n; j++)
    if (!c[j]) gel(y,k++) = gen_colei(n, j, E, ff);
  return y;
}

static GEN
gen_suppl(GEN x, void *E, const struct bb_field *ff,
          GEN (*mul)(void*, GEN, GEN))
{
  GEN d;
  long n = nbrows(x), r;

  if (lg(x) == 1) pari_err_IMPL("suppl [empty matrix]");
  d = gen_pivots(x, &r, E, ff, mul);
  return gen_get_suppl(x, d, n, r, E, ff);
}

/*******************************************************************/
/*                                                                 */
/*                MATRIX MULTIPLICATION MODULO P                   */
/*                                                                 */
/*******************************************************************/

GEN
F2xqM_F2xqC_mul(GEN A, GEN B, GEN T) {
  void *E;
  const struct bb_field *ff = get_F2xq_field(&E, T);
  return gen_matcolmul(A, B, E, ff);
}

GEN
FlxqM_FlxqC_mul(GEN A, GEN B, GEN T, ulong p) {
  void *E;
  const struct bb_field *ff = get_Flxq_field(&E, T, p);
  return gen_matcolmul(A, B, E, ff);
}

GEN
FqM_FqC_mul(GEN A, GEN B, GEN T, GEN p) {
  void *E;
  const struct bb_field *ff = get_Fq_field(&E, T, p);
  return gen_matcolmul(A, B, E, ff);
}

GEN
F2xqM_mul(GEN A, GEN B, GEN T) {
  void *E;
  const struct bb_field *ff = get_F2xq_field(&E, T);
  return gen_matmul(A, B, E, ff);
}

GEN
FlxqM_mul(GEN A, GEN B, GEN T, ulong p) {
  void *E;
  const struct bb_field *ff;
  long n = lg(A) - 1;

  if (n == 0)
    return cgetg(1, t_MAT);
  if (n > 1)
    return FlxqM_mul_Kronecker(A, B, T, p);
  ff = get_Flxq_field(&E, T, p);
  return gen_matmul(A, B, E, ff);
}

GEN
FqM_mul(GEN A, GEN B, GEN T, GEN p) {
  void *E;
  long n = lg(A) - 1;
  const struct bb_field *ff;
  if (n == 0)
    return cgetg(1, t_MAT);
  if (n > 1)
    return FqM_mul_Kronecker(A, B, T, p);
  ff = get_Fq_field(&E, T, p);
  return gen_matmul(A, B, E, ff);
}

/*******************************************************************/
/*                                                                 */
/*                    LINEAR ALGEBRA MODULO P                      */
/*                                                                 */
/*******************************************************************/

static GEN
_F2xqM_mul(void *E, GEN A, GEN B)
{ return F2xqM_mul(A, B, (GEN) E); }

struct _Flxq {
  GEN aut;
  GEN T;
  ulong p;
};

static GEN
_FlxqM_mul(void *E, GEN A, GEN B)
{
  struct _Flxq *D = (struct _Flxq*)E;
  return FlxqM_mul(A, B, D->T, D->p);
}

static GEN
_FpM_mul(void *E, GEN A, GEN B)
{ return FpM_mul(A, B, (GEN) E); }

struct _Fq_field
{
  GEN T, p;
};

static GEN
_FqM_mul(void *E, GEN A, GEN B)
{
  struct _Fq_field *D = (struct _Fq_field*) E;
  return FqM_mul(A, B, D->T, D->p);
}

static GEN
FpM_init(GEN a, GEN p, ulong *pp)
{
  if (lgefint(p) == 3)
  {
    *pp = uel(p,2);
    return (*pp==2)? ZM_to_F2m(a): ZM_to_Flm(a, *pp);
  }
  *pp = 0; return a;
}
static GEN
FpM_init3(GEN a, GEN p, ulong *pp)
{
  if (lgefint(p) == 3)
  {
    *pp = uel(p,2);
    switch(*pp)
    {
      case 2: return ZM_to_F2m(a);
      case 3: return ZM_to_F3m(a);
      default:return ZM_to_Flm(a, *pp);
    }
  }
  *pp = 0; return a;
}
GEN
RgM_Fp_init(GEN a, GEN p, ulong *pp)
{
  if (lgefint(p) == 3)
  {
    *pp = uel(p,2);
    return (*pp==2)? RgM_to_F2m(a): RgM_to_Flm(a, *pp);
  }
  *pp = 0; return RgM_to_FpM(a,p);
}
static GEN
RgM_Fp_init3(GEN a, GEN p, ulong *pp)
{
  if (lgefint(p) == 3)
  {
    *pp = uel(p,2);
    switch(*pp)
    {
      case 2: return RgM_to_F2m(a);
      case 3: return RgM_to_F3m(a);
      default:return RgM_to_Flm(a, *pp);
    }
  }
  *pp = 0; return RgM_to_FpM(a,p);
}

static GEN
FpM_det_gen(GEN a, GEN p)
{
  void *E;
  const struct bb_field *S = get_Fp_field(&E,p);
  return gen_det_i(a, E, S, _FpM_mul);
}
GEN
FpM_det(GEN a, GEN p)
{
  pari_sp av = avma;
  ulong pp, d;
  a = FpM_init(a, p, &pp);
  switch(pp)
  {
  case 0: return FpM_det_gen(a, p);
  case 2: d = F2m_det_sp(a); break;
  default:d = Flm_det_sp(a,pp); break;
  }
  set_avma(av); return utoi(d);
}

GEN
F2xqM_det(GEN a, GEN T)
{
  void *E;
  const struct bb_field *S = get_F2xq_field(&E, T);
  return gen_det_i(a, E, S, _F2xqM_mul);
}

GEN
FlxqM_det(GEN a, GEN T, ulong p) {
  void *E;
  const struct bb_field *S = get_Flxq_field(&E, T, p);
  return gen_det_i(a, E, S, _FlxqM_mul);
}

GEN
FqM_det(GEN x, GEN T, GEN p)
{
  void *E;
  const struct bb_field *S = get_Fq_field(&E,T,p);
  return gen_det_i(x, E, S, _FqM_mul);
}

static GEN
FpM_gauss_pivot_gen(GEN x, GEN p, long *rr)
{
  void *E;
  const struct bb_field *S = get_Fp_field(&E,p);
  return gen_pivots(x, rr, E, S, _FpM_mul);
}

static GEN
FpM_gauss_pivot(GEN x, GEN p, long *rr)
{
  ulong pp;
  if (lg(x)==1) { *rr = 0; return NULL; }
  x = FpM_init(x, p, &pp);
  switch(pp)
  {
  case 0: return FpM_gauss_pivot_gen(x, p, rr);
  case 2: return F2m_gauss_pivot(x, rr);
  default:return Flm_pivots(x, pp, rr, 1);
  }
}

static GEN
F2xqM_gauss_pivot(GEN x, GEN T, long *rr)
{
  void *E;
  const struct bb_field *S = get_F2xq_field(&E,T);
  return gen_pivots(x, rr, E, S, _F2xqM_mul);
}

static GEN
FlxqM_gauss_pivot(GEN x, GEN T, ulong p, long *rr) {
  void *E;
  const struct bb_field *S = get_Flxq_field(&E, T, p);
  return gen_pivots(x, rr, E, S, _FlxqM_mul);
}

static GEN
FqM_gauss_pivot_gen(GEN x, GEN T, GEN p, long *rr)
{
  void *E;
  const struct bb_field *S = get_Fq_field(&E,T,p);
  return gen_pivots(x, rr, E, S, _FqM_mul);
}
static GEN
FqM_gauss_pivot(GEN x, GEN T, GEN p, long *rr)
{
  if (lg(x)==1) { *rr = 0; return NULL; }
  if (!T) return FpM_gauss_pivot(x, p, rr);
  if (lgefint(p) == 3)
  {
    pari_sp av = avma;
    ulong pp = uel(p,2);
    GEN Tp = ZXT_to_FlxT(T, pp);
    GEN d = FlxqM_gauss_pivot(ZXM_to_FlxM(x, pp, get_Flx_var(Tp)), Tp, pp, rr);
    return d ? gerepileuptoleaf(av, d): d;
  }
  return FqM_gauss_pivot_gen(x, T, p, rr);
}

GEN
FpM_image(GEN x, GEN p)
{
  long r;
  GEN d = FpM_gauss_pivot(x,p,&r); /* d left on stack for efficiency */
  return image_from_pivot(x,d,r);
}

GEN
Flm_image(GEN x, ulong p)
{
  long r;
  GEN d = Flm_pivots(x, p, &r, 0); /* d left on stack for efficiency */
  return image_from_pivot(x,d,r);
}

GEN
F2m_image(GEN x)
{
  long r;
  GEN d = F2m_gauss_pivot(F2m_copy(x),&r); /* d left on stack for efficiency */
  return image_from_pivot(x,d,r);
}

GEN
F2xqM_image(GEN x, GEN T)
{
  long r;
  GEN d = F2xqM_gauss_pivot(x,T,&r); /* d left on stack for efficiency */
  return image_from_pivot(x,d,r);
}

GEN
FlxqM_image(GEN x, GEN T, ulong p)
{
  long r;
  GEN d = FlxqM_gauss_pivot(x, T, p, &r); /* d left on stack for efficiency */
  return image_from_pivot(x,d,r);
}

GEN
FqM_image(GEN x, GEN T, GEN p)
{
  long r;
  GEN d = FqM_gauss_pivot(x,T,p,&r); /* d left on stack for efficiency */
  return image_from_pivot(x,d,r);
}

long
FpM_rank(GEN x, GEN p)
{
  pari_sp av = avma;
  long r;
  (void)FpM_gauss_pivot(x,p,&r);
  return gc_long(av, lg(x)-1 - r);
}

long
F2xqM_rank(GEN x, GEN T)
{
  pari_sp av = avma;
  long r;
  (void)F2xqM_gauss_pivot(x,T,&r);
  return gc_long(av, lg(x)-1 - r);
}

long
FlxqM_rank(GEN x, GEN T, ulong p)
{
  void *E;
  const struct bb_field *S = get_Flxq_field(&E, T, p);
  return gen_matrank(x, E, S, _FlxqM_mul);
}

long
FqM_rank(GEN x, GEN T, GEN p)
{
  pari_sp av = avma;
  long r;
  (void)FqM_gauss_pivot(x,T,p,&r);
  return gc_long(av, lg(x)-1 - r);
}

static GEN
FpM_invimage_gen(GEN A, GEN B, GEN p)
{
  void *E;
  const struct bb_field *ff = get_Fp_field(&E, p);
  return gen_invimage(A, B, E, ff, _FpM_mul);
}

GEN
FpM_invimage(GEN A, GEN B, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  GEN y;

  A = FpM_init(A, p, &pp);
  switch(pp)
  {
  case 0: return FpM_invimage_gen(A, B, p);
  case 2:
    y = F2m_invimage(A, ZM_to_F2m(B));
    if (!y) return gc_NULL(av);
    y = F2m_to_ZM(y);
    return gerepileupto(av, y);
  default:
    y = Flm_invimage_i(A, ZM_to_Flm(B, pp), pp);
    if (!y) return gc_NULL(av);
    y = Flm_to_ZM(y);
    return gerepileupto(av, y);
  }
}

GEN
F2xqM_invimage(GEN A, GEN B, GEN T) {
  void *E;
  const struct bb_field *ff = get_F2xq_field(&E, T);
  return gen_invimage(A, B, E, ff, _F2xqM_mul);
}

GEN
FlxqM_invimage(GEN A, GEN B, GEN T, ulong p) {
  void *E;
  const struct bb_field *ff = get_Flxq_field(&E, T, p);
  return gen_invimage(A, B, E, ff, _FlxqM_mul);
}

GEN
FqM_invimage(GEN A, GEN B, GEN T, GEN p) {
  void *E;
  const struct bb_field *ff = get_Fq_field(&E, T, p);
  return gen_invimage(A, B, E, ff, _FqM_mul);
}

static GEN
FpM_FpC_invimage_gen(GEN A, GEN y, GEN p)
{
  void *E;
  const struct bb_field *ff = get_Fp_field(&E, p);
  return gen_matcolinvimage_i(A, y, E, ff, _FpM_mul);
}

GEN
FpM_FpC_invimage(GEN A, GEN x, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  GEN y;

  A = FpM_init(A, p, &pp);
  switch(pp)
  {
  case 0: return FpM_FpC_invimage_gen(A, x, p);
  case 2:
    y = F2m_F2c_invimage(A, ZV_to_F2v(x));
    if (!y) return y;
    y = F2c_to_ZC(y);
    return gerepileupto(av, y);
  default:
    y = Flm_Flc_invimage(A, ZV_to_Flv(x, pp), pp);
    if (!y) return y;
    y = Flc_to_ZC(y);
    return gerepileupto(av, y);
  }
}

GEN
F2xqM_F2xqC_invimage(GEN A, GEN B, GEN T) {
  void *E;
  const struct bb_field *ff = get_F2xq_field(&E, T);
  return gen_matcolinvimage_i(A, B, E, ff, _F2xqM_mul);
}

GEN
FlxqM_FlxqC_invimage(GEN A, GEN B, GEN T, ulong p) {
  void *E;
  const struct bb_field *ff = get_Flxq_field(&E, T, p);
  return gen_matcolinvimage_i(A, B, E, ff, _FlxqM_mul);
}

GEN
FqM_FqC_invimage(GEN A, GEN B, GEN T, GEN p) {
  void *E;
  const struct bb_field *ff = get_Fq_field(&E, T, p);
  return gen_matcolinvimage_i(A, B, E, ff, _FqM_mul);
}

static GEN
FpM_ker_gen(GEN x, GEN p, long deplin)
{
  void *E;
  const struct bb_field *S = get_Fp_field(&E,p);
  return gen_ker_i(x, deplin, E, S, _FpM_mul);
}
static GEN
FpM_ker_i(GEN x, GEN p, long deplin)
{
  pari_sp av = avma;
  ulong pp;
  GEN y;

  if (lg(x)==1) return cgetg(1,t_MAT);
  x = FpM_init3(x, p, &pp);
  switch(pp)
  {
  case 0: return FpM_ker_gen(x,p,deplin);
  case 2:
    y = F2m_ker_sp(x, deplin);
    if (!y) return gc_NULL(av);
    y = deplin? F2c_to_ZC(y): F2m_to_ZM(y);
    return gerepileupto(av, y);
  case 3:
    y = F3m_ker_sp(x, deplin);
    if (!y) return gc_NULL(av);
    y = deplin? F3c_to_ZC(y): F3m_to_ZM(y);
    return gerepileupto(av, y);
  default:
    y = Flm_ker_sp(x, pp, deplin);
    if (!y) return gc_NULL(av);
    y = deplin? Flc_to_ZC(y): Flm_to_ZM(y);
    return gerepileupto(av, y);
  }
}

GEN
FpM_ker(GEN x, GEN p) { return FpM_ker_i(x,p,0); }

static GEN
F2xqM_ker_i(GEN x, GEN T, long deplin)
{
  const struct bb_field *ff;
  void *E;

  if (lg(x)==1) return cgetg(1,t_MAT);
  ff = get_F2xq_field(&E,T);
  return gen_ker_i(x,deplin, E, ff, _F2xqM_mul);
}

GEN
F2xqM_ker(GEN x, GEN T)
{
  return F2xqM_ker_i(x, T, 0);
}

static GEN
FlxqM_ker_i(GEN x, GEN T, ulong p, long deplin) {
  void *E;
  const struct bb_field *S = get_Flxq_field(&E, T, p);
  return gen_ker_i(x, deplin, E, S, _FlxqM_mul);
}

GEN
FlxqM_ker(GEN x, GEN T, ulong p)
{
  return FlxqM_ker_i(x, T, p, 0);
}

static GEN
FqM_ker_gen(GEN x, GEN T, GEN p, long deplin)
{
  void *E;
  const struct bb_field *S = get_Fq_field(&E,T,p);
  return gen_ker_i(x,deplin,E,S,_FqM_mul);
}
static GEN
FqM_ker_i(GEN x, GEN T, GEN p, long deplin)
{
  if (!T) return FpM_ker_i(x,p,deplin);
  if (lg(x)==1) return cgetg(1,t_MAT);

  if (lgefint(p)==3)
  {
    pari_sp ltop=avma;
    ulong l= p[2];
    GEN Tl = ZXT_to_FlxT(T,l);
    GEN Ml = ZXM_to_FlxM(x, l, get_Flx_var(Tl));
    GEN p1 = FlxqM_ker_i(Ml, Tl, l, deplin);
    if (deplin)
    {
      if (!p1) return gc_NULL(ltop);
      return gerepileupto(ltop, FlxC_to_ZXC(p1));
    }
    else return gerepileupto(ltop, FlxM_to_ZXM(p1));
  }
  return FqM_ker_gen(x, T, p, deplin);
}

GEN
FqM_ker(GEN x, GEN T, GEN p) { return FqM_ker_i(x,T,p,0); }

GEN
FpM_deplin(GEN x, GEN p) { return FpM_ker_i(x,p,1); }

GEN
F2xqM_deplin(GEN x, GEN T)
{
  return F2xqM_ker_i(x, T, 1);
}

GEN
FlxqM_deplin(GEN x, GEN T, ulong p)
{
  return FlxqM_ker_i(x, T, p, 1);
}

GEN
FqM_deplin(GEN x, GEN T, GEN p) { return FqM_ker_i(x,T,p,1); }

static GEN
FpM_gauss_gen(GEN a, GEN b, GEN p)
{
  void *E;
  const struct bb_field *S = get_Fp_field(&E,p);
  return gen_gauss(a,b, E, S, _FpM_mul);
}
/* a an FpM, lg(a)>1; b an FpM or NULL (replace by identity) */
static GEN
FpM_gauss_i(GEN a, GEN b, GEN p, ulong *pp)
{
  long n = nbrows(a);
  a = FpM_init(a,p,pp);
  switch(*pp)
  {
  case 0:
    if (!b) b = matid(n);
    return FpM_gauss_gen(a,b,p);
  case 2:
    if (b) b = ZM_to_F2m(b); else b = matid_F2m(n);
    return F2m_gauss_sp(a,b);
  default:
    if (b) b = ZM_to_Flm(b, *pp); else b = matid_Flm(n);
    return Flm_gauss_sp(a,b, NULL, *pp);
  }
}
GEN
FpM_gauss(GEN a, GEN b, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  GEN u;
  if (lg(a) == 1 || lg(b)==1) return cgetg(1, t_MAT);
  u = FpM_gauss_i(a, b, p, &pp);
  if (!u) return gc_NULL(av);
  switch(pp)
  {
  case 0: return gerepilecopy(av, u);
  case 2:  u = F2m_to_ZM(u); break;
  default: u = Flm_to_ZM(u); break;
  }
  return gerepileupto(av, u);
}

static GEN
F2xqM_gauss_gen(GEN a, GEN b, GEN T)
{
  void *E;
  const struct bb_field *S = get_F2xq_field(&E, T);
  return gen_gauss(a, b, E, S, _F2xqM_mul);
}

GEN
F2xqM_gauss(GEN a, GEN b, GEN T)
{
  pari_sp av = avma;
  long n = lg(a)-1;
  GEN u;
  if (!n || lg(b)==1) { set_avma(av); return cgetg(1, t_MAT); }
  u = F2xqM_gauss_gen(a, b, T);
  if (!u) return gc_NULL(av);
  return gerepilecopy(av, u);
}

static GEN
FlxqM_gauss_i(GEN a, GEN b, GEN T, ulong p) {
  void *E;
  const struct bb_field *S = get_Flxq_field(&E, T, p);
  return gen_gauss(a, b, E, S, _FlxqM_mul);
}

GEN
FlxqM_gauss(GEN a, GEN b, GEN T, ulong p)
{
  pari_sp av = avma;
  long n = lg(a)-1;
  GEN u;
  if (!n || lg(b)==1) { set_avma(av); return cgetg(1, t_MAT); }
  u = FlxqM_gauss_i(a, b, T, p);
  if (!u) return gc_NULL(av);
  return gerepilecopy(av, u);
}

static GEN
FqM_gauss_gen(GEN a, GEN b, GEN T, GEN p)
{
  void *E;
  const struct bb_field *S = get_Fq_field(&E,T,p);
  return gen_gauss(a,b,E,S,_FqM_mul);
}
GEN
FqM_gauss(GEN a, GEN b, GEN T, GEN p)
{
  pari_sp av = avma;
  GEN u;
  long n;
  if (!T) return FpM_gauss(a,b,p);
  n = lg(a)-1; if (!n || lg(b)==1) return cgetg(1, t_MAT);
  u = FqM_gauss_gen(a,b,T,p);
  if (!u) return gc_NULL(av);
  return gerepilecopy(av, u);
}

GEN
FpM_FpC_gauss(GEN a, GEN b, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  GEN u;
  if (lg(a) == 1) return cgetg(1, t_COL);
  u = FpM_gauss_i(a, mkmat(b), p, &pp);
  if (!u) return gc_NULL(av);
  switch(pp)
  {
  case 0: return gerepilecopy(av, gel(u,1));
  case 2:  u = F2c_to_ZC(gel(u,1)); break;
  default: u = Flc_to_ZC(gel(u,1)); break;
  }
  return gerepileupto(av, u);
}

GEN
F2xqM_F2xqC_gauss(GEN a, GEN b, GEN T)
{
  pari_sp av = avma;
  GEN u;
  if (lg(a) == 1) return cgetg(1, t_COL);
  u = F2xqM_gauss_gen(a, mkmat(b), T);
  if (!u) return gc_NULL(av);
  return gerepilecopy(av, gel(u,1));
}

GEN
FlxqM_FlxqC_gauss(GEN a, GEN b, GEN T, ulong p)
{
  pari_sp av = avma;
  GEN u;
  if (lg(a) == 1) return cgetg(1, t_COL);
  u = FlxqM_gauss_i(a, mkmat(b), T, p);
  if (!u) return gc_NULL(av);
  return gerepilecopy(av, gel(u,1));
}

GEN
FqM_FqC_gauss(GEN a, GEN b, GEN T, GEN p)
{
  pari_sp av = avma;
  GEN u;
  if (!T) return FpM_FpC_gauss(a,b,p);
  if (lg(a) == 1) return cgetg(1, t_COL);
  u = FqM_gauss_gen(a,mkmat(b),T,p);
  if (!u) return gc_NULL(av);
  return gerepilecopy(av, gel(u,1));
}

GEN
FpM_inv(GEN a, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  GEN u;
  if (lg(a) == 1) return cgetg(1, t_MAT);
  u = FpM_gauss_i(a, NULL, p, &pp);
  if (!u) return gc_NULL(av);
  switch(pp)
  {
  case 0: return gerepilecopy(av, u);
  case 2:  u = F2m_to_ZM(u); break;
  default: u = Flm_to_ZM(u); break;
  }
  return gerepileupto(av, u);
}

GEN
F2xqM_inv(GEN a, GEN T)
{
  pari_sp av = avma;
  GEN u;
  if (lg(a) == 1) { set_avma(av); return cgetg(1, t_MAT); }
  u = F2xqM_gauss_gen(a, matid_F2xqM(nbrows(a),T), T);
  if (!u) return gc_NULL(av);
  return gerepilecopy(av, u);
}

GEN
FlxqM_inv(GEN a, GEN T, ulong p)
{
  pari_sp av = avma;
  GEN u;
  if (lg(a) == 1) { set_avma(av); return cgetg(1, t_MAT); }
  u = FlxqM_gauss_i(a, matid_FlxqM(nbrows(a),T,p), T,p);
  if (!u) return gc_NULL(av);
  return gerepilecopy(av, u);
}

GEN
FqM_inv(GEN a, GEN T, GEN p)
{
  pari_sp av = avma;
  GEN u;
  if (!T) return FpM_inv(a,p);
  if (lg(a) == 1) return cgetg(1, t_MAT);
  u = FqM_gauss_gen(a,matid(nbrows(a)),T,p);
  if (!u) return gc_NULL(av);
  return gerepilecopy(av, u);
}

GEN
FpM_intersect_i(GEN x, GEN y, GEN p)
{
  long j, lx = lg(x);
  GEN z;

  if (lx == 1 || lg(y) == 1) return cgetg(1,t_MAT);
  if (lgefint(p) == 3)
  {
    ulong pp = p[2];
    return Flm_to_ZM(Flm_intersect_i(ZM_to_Flm(x,pp), ZM_to_Flm(y,pp), pp));
  }
  z = FpM_ker(shallowconcat(x,y), p);
  for (j=lg(z)-1; j; j--) setlg(gel(z,j),lx);
  return FpM_mul(x,z,p);
}
GEN
FpM_intersect(GEN x, GEN y, GEN p)
{
  pari_sp av = avma;
  GEN z;
  if (lgefint(p) == 3)
  {
    ulong pp = p[2];
    z = Flm_image(Flm_intersect_i(ZM_to_Flm(x,pp), ZM_to_Flm(y,pp), pp), pp);
  }
  else
    z = FpM_image(FpM_intersect_i(x,y,p), p);
  return gerepileupto(av, z);
}

static void
init_suppl(GEN x)
{
  if (lg(x) == 1) pari_err_IMPL("suppl [empty matrix]");
  /* HACK: avoid overwriting d from gauss_pivot() after set_avma(av) */
  (void)new_chunk(lgcols(x) * 2);
}

GEN
FpM_suppl(GEN x, GEN p)
{
  GEN d;
  long r;
  init_suppl(x); d = FpM_gauss_pivot(x,p, &r);
  return get_suppl(x,d,nbrows(x),r,&col_ei);
}

GEN
F2m_suppl(GEN x)
{
  GEN d;
  long r;
  init_suppl(x); d = F2m_gauss_pivot(F2m_copy(x), &r);
  return get_suppl(x,d,mael(x,1,1),r,&F2v_ei);
}

GEN
Flm_suppl(GEN x, ulong p)
{
  GEN d;
  long r;
  init_suppl(x); d = Flm_pivots(x, p, &r, 0);
  return get_suppl(x,d,nbrows(x),r,&vecsmall_ei);
}

GEN
F2xqM_suppl(GEN x, GEN T)
{
  void *E;
  const struct bb_field *S = get_F2xq_field(&E, T);
  return gen_suppl(x, E, S, _F2xqM_mul);
}

GEN
FlxqM_suppl(GEN x, GEN T, ulong p)
{
  void *E;
  const struct bb_field *S = get_Flxq_field(&E, T, p);
  return gen_suppl(x, E, S, _FlxqM_mul);
}

GEN
FqM_suppl(GEN x, GEN T, GEN p)
{
  pari_sp av = avma;
  GEN d;
  long r;

  if (!T) return FpM_suppl(x,p);
  init_suppl(x);
  d = FqM_gauss_pivot(x,T,p,&r);
  set_avma(av); return get_suppl(x,d,nbrows(x),r,&col_ei);
}

static void
init_indexrank(GEN x) {
  (void)new_chunk(3 + 2*lg(x)); /* HACK */
}

GEN
FpM_indexrank(GEN x, GEN p) {
  pari_sp av = avma;
  long r;
  GEN d;
  init_indexrank(x);
  d = FpM_gauss_pivot(x,p,&r);
  set_avma(av); return indexrank0(lg(x)-1, r, d);
}

GEN
Flm_indexrank(GEN x, ulong p) {
  pari_sp av = avma;
  long r;
  GEN d;
  init_indexrank(x);
  d = Flm_pivots(x, p, &r, 0);
  set_avma(av); return indexrank0(lg(x)-1, r, d);
}

GEN
F2m_indexrank(GEN x) {
  pari_sp av = avma;
  long r;
  GEN d;
  init_indexrank(x);
  d = F2m_gauss_pivot(F2m_copy(x),&r);
  set_avma(av); return indexrank0(lg(x)-1, r, d);
}

GEN
F2xqM_indexrank(GEN x, GEN T) {
  pari_sp av = avma;
  long r;
  GEN d;
  init_indexrank(x);
  d = F2xqM_gauss_pivot(x, T, &r);
  set_avma(av); return indexrank0(lg(x) - 1, r, d);
}

GEN
FlxqM_indexrank(GEN x, GEN T, ulong p) {
  pari_sp av = avma;
  long r;
  GEN d;
  init_indexrank(x);
  d = FlxqM_gauss_pivot(x, T, p, &r);
  set_avma(av); return indexrank0(lg(x) - 1, r, d);
}

GEN
FqM_indexrank(GEN x, GEN T, GEN p) {
  pari_sp av = avma;
  long r;
  GEN d;
  init_indexrank(x);
  d = FqM_gauss_pivot(x, T, p, &r);
  set_avma(av); return indexrank0(lg(x) - 1, r, d);
}

/*******************************************************************/
/*                                                                 */
/*                       Solve A*X=B (Gauss pivot)                 */
/*                                                                 */
/*******************************************************************/
/* x a column, x0 same column in the original input matrix (for reference),
 * c list of pivots so far */
static long
gauss_get_pivot_max(GEN X, GEN X0, long ix, GEN c)
{
  GEN p, r, x = gel(X,ix), x0 = gel(X0,ix);
  long i, k = 0, ex = - (long)HIGHEXPOBIT, lx = lg(x);
  if (c)
  {
    for (i=1; i<lx; i++)
      if (!c[i])
      {
        long e = gexpo(gel(x,i));
        if (e > ex) { ex = e; k = i; }
      }
  }
  else
  {
    for (i=ix; i<lx; i++)
    {
      long e = gexpo(gel(x,i));
      if (e > ex) { ex = e; k = i; }
    }
  }
  if (!k) return lx;
  p = gel(x,k);
  r = gel(x0,k); if (isrationalzero(r)) r = x0;
  return cx_approx0(p, r)? lx: k;
}
static long
gauss_get_pivot_padic(GEN X, GEN p, long ix, GEN c)
{
  GEN x = gel(X, ix);
  long i, k = 0, ex = (long)HIGHVALPBIT, lx = lg(x);
  if (c)
  {
    for (i=1; i<lx; i++)
      if (!c[i] && !gequal0(gel(x,i)))
      {
        long e = gvaluation(gel(x,i), p);
        if (e < ex) { ex = e; k = i; }
      }
  }
  else
  {
    for (i=ix; i<lx; i++)
      if (!gequal0(gel(x,i)))
      {
        long e = gvaluation(gel(x,i), p);
        if (e < ex) { ex = e; k = i; }
      }
  }
  return k? k: lx;
}
static long
gauss_get_pivot_NZ(GEN X, GEN x0/*unused*/, long ix, GEN c)
{
  GEN x = gel(X, ix);
  long i, lx = lg(x);
  (void)x0;
  if (c)
  {
    for (i=1; i<lx; i++)
      if (!c[i] && !gequal0(gel(x,i))) return i;
  }
  else
  {
    for (i=ix; i<lx; i++)
      if (!gequal0(gel(x,i))) return i;
  }
  return lx;
}

/* Return pivot seeking function appropriate for the domain of the RgM x
 * (first non zero pivot, maximal pivot...)
 * x0 is a reference point used when guessing whether x[i,j] ~ 0
 * (iff x[i,j] << x0[i,j]); typical case: mateigen, Gauss pivot on x - vp.Id,
 * but use original x when deciding whether a prospective pivot is nonzero */
static pivot_fun
get_pivot_fun(GEN x, GEN x0, GEN *data)
{
  long i, j, hx, lx = lg(x);
  int res = t_INT;
  GEN p = NULL;

  *data = NULL;
  if (lx == 1) return &gauss_get_pivot_NZ;
  hx = lgcols(x);
  for (j=1; j<lx; j++)
  {
    GEN xj = gel(x,j);
    for (i=1; i<hx; i++)
    {
      GEN c = gel(xj,i);
      switch(typ(c))
      {
        case t_REAL:
          res = t_REAL;
          break;
        case t_COMPLEX:
          if (typ(gel(c,1)) == t_REAL || typ(gel(c,2)) == t_REAL) res = t_REAL;
          break;
        case t_INT: case t_INTMOD: case t_FRAC: case t_FFELT: case t_QUAD:
        case t_POLMOD: /* exact types */
          break;
        case t_PADIC:
          p = gel(c,2);
          res = t_PADIC;
          break;
        default: return &gauss_get_pivot_NZ;
      }
    }
  }
  switch(res)
  {
    case t_REAL: *data = x0; return &gauss_get_pivot_max;
    case t_PADIC: *data = p; return &gauss_get_pivot_padic;
    default: return &gauss_get_pivot_NZ;
  }
}

static GEN
get_col(GEN a, GEN b, GEN p, long li)
{
  GEN u = cgetg(li+1,t_COL);
  long i, j;

  gel(u,li) = gdiv(gel(b,li), p);
  for (i=li-1; i>0; i--)
  {
    pari_sp av = avma;
    GEN m = gel(b,i);
    for (j=i+1; j<=li; j++) m = gsub(m, gmul(gcoeff(a,i,j), gel(u,j)));
    gel(u,i) = gerepileupto(av, gdiv(m, gcoeff(a,i,i)));
  }
  return u;
}

/* bk -= m * bi */
static void
_submul(GEN b, long k, long i, GEN m)
{
  gel(b,k) = gsub(gel(b,k), gmul(m, gel(b,i)));
}
static int
init_gauss(GEN a, GEN *b, long *aco, long *li, int *iscol)
{
  *iscol = *b ? (typ(*b) == t_COL): 0;
  *aco = lg(a) - 1;
  if (!*aco) /* a empty */
  {
    if (*b && lg(*b) != 1) pari_err_DIM("gauss");
    *li = 0; return 0;
  }
  *li = nbrows(a);
  if (*li < *aco) pari_err_INV("gauss [no left inverse]", a);
  if (*b)
  {
    switch(typ(*b))
    {
      case t_MAT:
        if (lg(*b) == 1) return 0;
        *b = RgM_shallowcopy(*b);
        break;
      case t_COL:
        *b = mkmat( leafcopy(*b) );
        break;
      default: pari_err_TYPE("gauss",*b);
    }
    if (nbrows(*b) != *li) pari_err_DIM("gauss");
  }
  else
    *b = matid(*li);
  return 1;
}

static GEN
RgM_inv_FpM(GEN a, GEN p)
{
  ulong pp;
  a = RgM_Fp_init(a, p, &pp);
  switch(pp)
  {
  case 0:
    a = FpM_inv(a,p);
    if (a) a = FpM_to_mod(a, p);
    break;
  case 2:
    a = F2m_inv(a);
    if (a) a = F2m_to_mod(a);
    break;
  default:
    a = Flm_inv_sp(a, NULL, pp);
    if (a) a = Flm_to_mod(a, pp);
  }
  return a;
}

static GEN
RgM_inv_FqM(GEN x, GEN pol, GEN p)
{
  pari_sp av = avma;
  GEN b, T = RgX_to_FpX(pol, p);
  if (signe(T) == 0) pari_err_OP("^",x,gen_m1);
  b = FqM_inv(RgM_to_FqM(x, T, p), T, p);
  if (!b) return gc_NULL(av);
  return gerepileupto(av, FqM_to_mod(b, T, p));
}

#define code(t1,t2) ((t1 << 6) | t2)
static GEN
RgM_inv_fast(GEN x)
{
  GEN p, pol;
  long pa;
  long t = RgM_type(x, &p,&pol,&pa);
  switch(t)
  {
    case t_INT:    /* Fall back */
    case t_FRAC:   return QM_inv(x);
    case t_FFELT:  return FFM_inv(x, pol);
    case t_INTMOD: return RgM_inv_FpM(x, p);
    case code(t_POLMOD, t_INTMOD):
                   return RgM_inv_FqM(x, pol, p);
    default:       return gen_0;
  }
}
#undef code

static GEN
RgM_RgC_solve_FpC(GEN a, GEN b, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  a = RgM_Fp_init(a, p, &pp);
  switch(pp)
  {
  case 0:
    b = RgC_to_FpC(b, p);
    a = FpM_FpC_gauss(a,b,p);
    return a ? gerepileupto(av, FpC_to_mod(a, p)): NULL;
  case 2:
    b = RgV_to_F2v(b);
    a = F2m_F2c_gauss(a,b);
    return a ? gerepileupto(av, F2c_to_mod(a)): NULL;
  default:
    b = RgV_to_Flv(b, pp);
    a = Flm_Flc_gauss(a, b, pp);
    return a ? gerepileupto(av, Flc_to_mod(a, pp)): NULL;
  }
}

static GEN
RgM_solve_FpM(GEN a, GEN b, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  a = RgM_Fp_init(a, p, &pp);
  switch(pp)
  {
  case 0:
    b = RgM_to_FpM(b, p);
    a = FpM_gauss(a,b,p);
    return a ? gerepileupto(av, FpM_to_mod(a, p)): NULL;
  case 2:
    b = RgM_to_F2m(b);
    a = F2m_gauss(a,b);
    return a ? gerepileupto(av, F2m_to_mod(a)): NULL;
  default:
    b = RgM_to_Flm(b, pp);
    a = Flm_gauss(a,b,pp);
    return a ? gerepileupto(av, Flm_to_mod(a, pp)): NULL;
  }
}

/* Gaussan Elimination. If a is square, return a^(-1)*b;
 * if a has more rows than columns and b is NULL, return c such that c a = Id.
 * a is a (not necessarily square) matrix
 * b is a matrix or column vector, NULL meaning: take the identity matrix,
 *   effectively returning the inverse of a
 * If a and b are empty, the result is the empty matrix.
 *
 * li: number of rows of a and b
 * aco: number of columns of a
 * bco: number of columns of b (if matrix)
 */
static GEN
RgM_solve_basecase(GEN a, GEN b)
{
  pari_sp av = avma;
  long i, j, k, li, bco, aco;
  int iscol;
  pivot_fun pivot;
  GEN p, u, data;

  set_avma(av);

  if (lg(a)-1 == 2 && nbrows(a) == 2) {
    /* 2x2 matrix, start by inverting a */
    GEN u = gcoeff(a,1,1), v = gcoeff(a,1,2);
    GEN w = gcoeff(a,2,1), x = gcoeff(a,2,2);
    GEN D = gsub(gmul(u,x), gmul(v,w)), ainv;
    if (gequal0(D)) return NULL;
    ainv = mkmat2(mkcol2(x, gneg(w)), mkcol2(gneg(v), u));
    ainv = gmul(ainv, ginv(D));
    if (b) ainv = gmul(ainv, b);
    return gerepileupto(av, ainv);
  }

  if (!init_gauss(a, &b, &aco, &li, &iscol)) return cgetg(1, iscol?t_COL:t_MAT);
  pivot = get_pivot_fun(a, a, &data);
  a = RgM_shallowcopy(a);
  bco = lg(b)-1;
  if(DEBUGLEVEL>4) err_printf("Entering gauss\n");

  p = NULL; /* gcc -Wall */
  for (i=1; i<=aco; i++)
  {
    /* k is the line where we find the pivot */
    k = pivot(a, data, i, NULL);
    if (k > li) return NULL;
    if (k != i)
    { /* exchange the lines s.t. k = i */
      for (j=i; j<=aco; j++) swap(gcoeff(a,i,j), gcoeff(a,k,j));
      for (j=1; j<=bco; j++) swap(gcoeff(b,i,j), gcoeff(b,k,j));
    }
    p = gcoeff(a,i,i);
    if (i == aco) break;

    for (k=i+1; k<=li; k++)
    {
      GEN m = gcoeff(a,k,i);
      if (!gequal0(m))
      {
        m = gdiv(m,p);
        for (j=i+1; j<=aco; j++) _submul(gel(a,j),k,i,m);
        for (j=1;   j<=bco; j++) _submul(gel(b,j),k,i,m);
      }
    }
    if (gc_needed(av,1))
    {
      if(DEBUGMEM>1) pari_warn(warnmem,"gauss. i=%ld",i);
      gerepileall(av,2, &a,&b);
    }
  }

  if(DEBUGLEVEL>4) err_printf("Solving the triangular system\n");
  u = cgetg(bco+1,t_MAT);
  for (j=1; j<=bco; j++) gel(u,j) = get_col(a,gel(b,j),p,aco);
  return gerepilecopy(av, iscol? gel(u,1): u);
}

static GEN
RgM_RgC_solve_fast(GEN x, GEN y)
{
  GEN p, pol;
  long pa;
  long t = RgM_RgC_type(x, y, &p,&pol,&pa);
  switch(t)
  {
    case t_INT:    return ZM_gauss(x, y);
    case t_FRAC:   return QM_gauss(x, y);
    case t_INTMOD: return RgM_RgC_solve_FpC(x, y, p);
    case t_FFELT:  return FFM_FFC_gauss(x, y, pol);
    default:       return gen_0;
  }
}

static GEN
RgM_solve_fast(GEN x, GEN y)
{
  GEN p, pol;
  long pa;
  long t = RgM_type2(x, y, &p,&pol,&pa);
  switch(t)
  {
    case t_INT:    return ZM_gauss(x, y);
    case t_FRAC:   return QM_gauss(x, y);
    case t_INTMOD: return RgM_solve_FpM(x, y, p);
    case t_FFELT:  return FFM_gauss(x, y, pol);
    default:       return gen_0;
  }
}

GEN
RgM_solve(GEN a, GEN b)
{
  pari_sp av = avma;
  GEN u;
  if (!b) return RgM_inv(a);
  u = typ(b)==t_MAT ? RgM_solve_fast(a, b): RgM_RgC_solve_fast(a, b);
  if (!u) return gc_NULL(av);
  if (u != gen_0) return u;
  return RgM_solve_basecase(a, b);
}

GEN
RgM_inv(GEN a)
{
  GEN b = RgM_inv_fast(a);
  return b==gen_0? RgM_solve_basecase(a, NULL): b;
}

/* assume dim A >= 1, A invertible + upper triangular  */
static GEN
RgM_inv_upper_ind(GEN A, long index)
{
  long n = lg(A)-1, i = index, j;
  GEN u = zerocol(n);
  gel(u,i) = ginv(gcoeff(A,i,i));
  for (i--; i>0; i--)
  {
    pari_sp av = avma;
    GEN m = gneg(gmul(gcoeff(A,i,i+1),gel(u,i+1))); /* j = i+1 */
    for (j=i+2; j<=n; j++) m = gsub(m, gmul(gcoeff(A,i,j),gel(u,j)));
    gel(u,i) = gerepileupto(av, gdiv(m, gcoeff(A,i,i)));
  }
  return u;
}
GEN
RgM_inv_upper(GEN A)
{
  long i, l;
  GEN B = cgetg_copy(A, &l);
  for (i = 1; i < l; i++) gel(B,i) = RgM_inv_upper_ind(A, i);
  return B;
}

static GEN
split_realimag_col(GEN z, long r1, long r2)
{
  long i, ru = r1+r2;
  GEN x = cgetg(ru+r2+1,t_COL), y = x + r2;
  for (i=1; i<=r1; i++) {
    GEN a = gel(z,i);
    if (typ(a) == t_COMPLEX) a = gel(a,1); /* paranoia: a should be real */
    gel(x,i) = a;
  }
  for (   ; i<=ru; i++) {
    GEN b, a = gel(z,i);
    if (typ(a) == t_COMPLEX) { b = gel(a,2); a = gel(a,1); } else b = gen_0;
    gel(x,i) = a;
    gel(y,i) = b;
  }
  return x;
}
GEN
split_realimag(GEN x, long r1, long r2)
{
  long i,l; GEN y;
  if (typ(x) == t_COL) return split_realimag_col(x,r1,r2);
  y = cgetg_copy(x, &l);
  for (i=1; i<l; i++) gel(y,i) = split_realimag_col(gel(x,i), r1, r2);
  return y;
}

/* assume M = (r1+r2) x (r1+2r2) matrix and y compatible vector or matrix
 * r1 first lines of M,y are real. Solve the system obtained by splitting
 * real and imaginary parts. */
GEN
RgM_solve_realimag(GEN M, GEN y)
{
  long l = lg(M), r2 = l - lgcols(M), r1 = l-1 - 2*r2;
  return RgM_solve(split_realimag(M, r1,r2),
                   split_realimag(y, r1,r2));
}

GEN
gauss(GEN a, GEN b)
{
  GEN z;
  long t = typ(b);
  if (typ(a)!=t_MAT) pari_err_TYPE("gauss",a);
  if (t!=t_COL && t!=t_MAT) pari_err_TYPE("gauss",b);
  z = RgM_solve(a,b);
  if (!z) pari_err_INV("gauss",a);
  return z;
}

static GEN
ZlM_gauss_ratlift(GEN a, GEN b, ulong p, long e, GEN C)
{
  pari_sp av = avma, av2;
  GEN bb, xi, xb, pi, q, B, r;
  long i, f, k;
  ulong mask;
  if (!C) {
    C = Flm_inv(ZM_to_Flm(a, p), p);
    if (!C) pari_err_INV("ZlM_gauss", a);
  }
  k = f = ZM_max_lg(a)-1;
  mask = quadratic_prec_mask((e+f-1)/f);
  pi = q = powuu(p, f);
  bb = b;
  C = ZpM_invlift(FpM_red(a, q), Flm_to_ZM(C), utoipos(p), f);
  av2 = avma;
  xb = xi = FpM_mul(C, b, q);
  for (i = f; i <= e; i+=f)
  {
    if (i==k)
    {
      k = (mask&1UL) ? 2*k-f: 2*k;
      mask >>= 1;
      B = sqrti(shifti(pi,-1));
      r = FpM_ratlift(xb, pi, B, B, NULL);
      if (r)
      {
        GEN dr, nr = Q_remove_denom(r,&dr);
        if (ZM_equal(ZM_mul(a,nr), dr? ZM_Z_mul(b,dr): b))
        {
          if (DEBUGLEVEL>=4)
            err_printf("ZlM_gauss: early solution: %ld/%ld\n",i,e);
          return gerepilecopy(av, r);
        }
      }
    }
    bb = ZM_Z_divexact(ZM_sub(bb, ZM_mul(a, xi)), q);
    if (gc_needed(av,2))
    {
      if(DEBUGMEM>1) pari_warn(warnmem,"ZlM_gauss. i=%ld/%ld",i,e);
      gerepileall(av2,3, &pi,&bb,&xb);
    }
    xi = FpM_mul(C, bb, q);
    xb = ZM_add(xb, ZM_Z_mul(xi, pi));
    pi = mulii(pi, q);
  }
  B = sqrti(shifti(pi,-1));
  return gerepileupto(av, FpM_ratlift(xb, pi, B, B, NULL));
}

/* Dixon p-adic lifting algorithm.
 * Numer. Math. 40, 137-141 (1982), DOI: 10.1007/BF01459082 */
GEN
ZM_gauss(GEN a, GEN b)
{
  pari_sp av = avma, av2;
  int iscol;
  long n, ncol, i, m, elim;
  ulong p;
  GEN C, delta, nb, nmin, res;
  forprime_t S;

  if (!init_gauss(a, &b, &n, &ncol, &iscol)) return cgetg(1, iscol?t_COL:t_MAT);
  if (n < ncol)
  {
    GEN y = ZM_indexrank(a), y1 = gel(y,1), y2 = gel(y,2);
    if (lg(y2)-1 != n) return NULL;
    a = rowpermute(a, y1);
    b = rowpermute(b, y1);
  }
  /* a is square and invertible */
  nb = gen_0; ncol = lg(b);
  for (i = 1; i < ncol; i++)
  {
    GEN ni = gnorml2(gel(b, i));
    if (cmpii(nb, ni) < 0) nb = ni;
  }
  if (!signe(nb)) {set_avma(av); return iscol? zerocol(n): zeromat(n,lg(b)-1);}
  delta = gen_1; nmin = nb;
  for (i = 1; i <= n; i++)
  {
    GEN ni = gnorml2(gel(a, i));
    if (cmpii(ni, nmin) < 0)
    {
      delta = mulii(delta, nmin); nmin = ni;
    }
    else
      delta = mulii(delta, ni);
  }
  if (!signe(nmin)) return NULL;
  elim = expi(delta)+1;
  av2 = avma;
  init_modular_big(&S);
  for(;;)
  {
    p = u_forprime_next(&S);
    C = Flm_inv_sp(ZM_to_Flm(a, p), NULL, p);
    if (C) break;
    elim -= expu(p);
    if (elim < 0) return NULL;
    set_avma(av2);
  }
  /* N.B. Our delta/lambda are SQUARES of those in the paper
   * log(delta lambda) / log p, where lambda is 3+sqrt(5) / 2,
   * whose log is < 1, hence + 1 (to cater for rounding errors) */
  m = (long)ceil((dbllog2(delta)*M_LN2 + 1) / log((double)p));
  res = ZlM_gauss_ratlift(a, b, p, m, C);
  if (iscol) return gerepilecopy(av, gel(res, 1));
  return gerepileupto(av, res);
}

/* #C = n, C[z[i]] = K[i], complete by 0s */
static GEN
RgC_inflate(GEN K, GEN z, long n)
{
  GEN c = zerocol(n);
  long j, l = lg(K);
  for (j = 1; j < l; j++) gel(c, z[j]) = gel(K, j);
  return c;
}
/* in place: C[i] *= cB / v[i] */
static void
QC_normalize(GEN C, GEN v, GEN cB)
{
  long l = lg(C), i;
  for (i = 1; i < l; i++)
  {
    GEN c = cB, k = gel(C,i), d = gel(v,i);
    if (d)
    {
      if (isintzero(d)) { gel(C,i) = gen_0; continue; }
      c = div_content(c, d);
    }
    gel(C,i) = c? gmul(k,c): k;
  }
}

/* same as above, M rational; if flag = 1, call indexrank and return 1 sol */
GEN
QM_gauss_i(GEN M, GEN B, long flag)
{
  pari_sp av = avma;
  long i, l, n;
  int col = typ(B) == t_COL;
  GEN K, cB, N = cgetg_copy(M, &l), v = cgetg(l, t_VEC), z2 = NULL;

  for (i = 1; i < l; i++)
    gel(N,i) = Q_primitive_part(gel(M,i), &gel(v,i));
  if (flag)
  {
    GEN z = ZM_indexrank(N), z1 = gel(z,1);
    z2 = gel(z,2);
    N = shallowmatextract(N, z1, z2);
    B = col? vecpermute(B,z1): rowpermute(B,z1);
    if (lg(z2) == l) z2 = NULL; else v = vecpermute(v, z2);
  }
  B = Q_primitive_part(B, &cB);
  K = ZM_gauss(N, B); if (!K) return gc_NULL(av);
  n = l - 1;
  if (col)
  {
    QC_normalize(K, v, cB);
    if (z2) K = RgC_inflate(K, z2, n);
  }
  else
  {
    long lK = lg(K);
    for (i = 1; i < lK; i++)
    {
      QC_normalize(gel(K,i), v, cB);
      if (z2) gel(K,i) = RgC_inflate(gel(K,i), z2, n);
    }
  }
  return gerepilecopy(av, K);
}
GEN
QM_gauss(GEN M, GEN B) { return QM_gauss_i(M, B, 0); }

static GEN
ZM_inv_slice(GEN A, GEN P, GEN *mod)
{
  pari_sp av = avma;
  long i, n = lg(P)-1;
  GEN H, T;
  if (n == 1)
  {
    ulong p = uel(P,1);
    GEN Hp, a = ZM_to_Flm(A, p);
    Hp = Flm_adjoint(a, p);
    Hp = gerepileupto(av, Flm_to_ZM(Hp));
    *mod = utoipos(p); return Hp;
  }
  T = ZV_producttree(P);
  A = ZM_nv_mod_tree(A, P, T);
  H = cgetg(n+1, t_VEC);
  for(i=1; i <= n; i++)
    gel(H,i) = Flm_adjoint(gel(A, i), uel(P,i));
  H = nmV_chinese_center_tree_seq(H, P, T, ZV_chinesetree(P,T));
  *mod = gmael(T, lg(T)-1, 1); return gc_all(av, 2, &H, mod);
}

static GEN
RgM_true_Hadamard(GEN a)
{
  pari_sp av = avma;
  long n = lg(a)-1, i;
  GEN B;
  if (n == 0) return gen_1;
  a = RgM_gtofp(a, LOWDEFAULTPREC);
  B = gnorml2(gel(a,1));
  for (i = 2; i <= n; i++) B = gmul(B, gnorml2(gel(a,i)));
  return gerepileuptoint(av, ceil_safe(sqrtr(B)));
}

GEN
ZM_inv_worker(GEN P, GEN A)
{
  GEN V = cgetg(3, t_VEC);
  gel(V,1) = ZM_inv_slice(A, P, &gel(V,2));
  return V;
}

static GEN
ZM_inv0(GEN A, GEN *pden)
{
  if (pden) *pden = gen_1;
  (void)A; return cgetg(1, t_MAT);
}
static GEN
ZM_inv1(GEN A, GEN *pden)
{
  GEN a = gcoeff(A,1,1);
  long s = signe(a);
  if (!s) return NULL;
  if (pden) *pden = absi(a);
  retmkmat(mkcol(s == 1? gen_1: gen_m1));
}
static GEN
ZM_inv2(GEN A, GEN *pden)
{
  GEN a, b, c, d, D, cA;
  long s;
  A = Q_primitive_part(A, &cA);
  a = gcoeff(A,1,1); b = gcoeff(A,1,2);
  c = gcoeff(A,2,1); d = gcoeff(A,2,2);
  D = subii(mulii(a,d), mulii(b,c)); /* left on stack */
  s = signe(D);
  if (!s) return NULL;
  if (s < 0) D = negi(D);
  if (pden) *pden = mul_denom(D, cA);
  if (s > 0)
    retmkmat2(mkcol2(icopy(d), negi(c)), mkcol2(negi(b), icopy(a)));
  else
    retmkmat2(mkcol2(negi(d), icopy(c)), mkcol2(icopy(b), negi(a)));
}

/* to be used when denom(M^(-1)) << det(M) and a sharp multiple is
 * not available. Return H primitive such that M*H = den*Id */
GEN
ZM_inv_ratlift(GEN M, GEN *pden)
{
  pari_sp av2, av = avma;
  GEN Hp, q, H;
  ulong p;
  long m = lg(M)-1;
  forprime_t S;
  pari_timer ti;

  if (m == 0) return ZM_inv0(M,pden);
  if (m == 1 && nbrows(M)==1) return ZM_inv1(M,pden);
  if (m == 2 && nbrows(M)==2) return ZM_inv2(M,pden);

  if (DEBUGLEVEL>5) timer_start(&ti);
  init_modular_big(&S);
  av2 = avma;
  H = NULL;
  while ((p = u_forprime_next(&S)))
  {
    GEN Mp, B, Hr;
    Mp = ZM_to_Flm(M,p);
    Hp = Flm_inv_sp(Mp, NULL, p);
    if (!Hp) continue;
    if (!H)
    {
      H = ZM_init_CRT(Hp, p);
      q = utoipos(p);
    }
    else
      ZM_incremental_CRT(&H, Hp, &q, p);
    B = sqrti(shifti(q,-1));
    Hr = FpM_ratlift(H,q,B,B,NULL);
    if (DEBUGLEVEL>5)
      timer_printf(&ti,"ZM_inv mod %lu (ratlift=%ld)", p,!!Hr);
    if (Hr) {/* DONE ? */
      GEN Hl = Q_remove_denom(Hr, pden);
      if (ZM_isscalar(ZM_mul(Hl, M), *pden)) { H = Hl; break; }
    }

    if (gc_needed(av,2))
    {
      if (DEBUGMEM>1) pari_warn(warnmem,"ZM_inv_ratlift");
      gerepileall(av2, 2, &H, &q);
    }
  }
  if (!*pden) *pden = gen_1;
  return gc_all(av, 2, &H, pden);
}

GEN
FpM_ratlift_worker(GEN A, GEN mod, GEN B)
{
  long l, i;
  GEN H = cgetg_copy(A, &l);
  for (i = 1; i < l; i++)
  {
     GEN c = FpC_ratlift(gel(A,i), mod, B, B, NULL);
     gel(H,i) = c? c: gen_0;
  }
  return H;
}
static int
can_ratlift(GEN x, GEN mod, GEN B)
{
  pari_sp av = avma;
  GEN a, b;
  return gc_bool(av, Fp_ratlift(x, mod, B, B, &a,&b));
}
static GEN
FpM_ratlift_parallel(GEN A, GEN mod, GEN B)
{
  pari_sp av = avma;
  GEN worker;
  long i, l = lg(A), m = mt_nbthreads();
  int test = !!B;

  if (l == 1 || lgcols(A) == 1) return gcopy(A);
  if (!B) B = sqrti(shifti(mod,-1));
  if (m == 1 || l == 2 || lgcols(A) < 10)
  {
    A = FpM_ratlift(A, mod, B, B, NULL);
    return A? A: gc_NULL(av);
  }
  /* test one coefficient first */
  if (test && !can_ratlift(gcoeff(A,1,1), mod, B)) return gc_NULL(av);
  worker = snm_closure(is_entry("_FpM_ratlift_worker"), mkvec2(mod,B));
  A = gen_parapply_slice(worker, A, m);
  for (i = 1; i < l; i++) if (typ(gel(A,i)) != t_COL) return gc_NULL(av);
  return A;
}

static GEN
ZM_adj_ratlift(GEN A, GEN H, GEN mod, GEN T)
{
  pari_sp av = avma;
  GEN B, D, g;
  D = ZMrow_ZC_mul(H, gel(A,1), 1);
  if (T) D = mulii(T, D);
  g = gcdii(D, mod);
  if (!equali1(g))
  {
    mod = diviiexact(mod, g);
    H = FpM_red(H, mod);
  }
  D = Fp_inv(Fp_red(D, mod), mod);
  /* test 1 coeff first */
  B = sqrti(shifti(mod,-1));
  if (!can_ratlift(Fp_mul(D, gcoeff(A,1,1), mod), mod, B)) return gc_NULL(av);
  H = FpM_Fp_mul(H, D, mod);
  H = FpM_ratlift_parallel(H, mod, B);
  return H? H: gc_NULL(av);
}

/* if (T) return T A^(-1) in Mn(Q), else B in Mn(Z) such that A B = den*Id */
static GEN
ZM_inv_i(GEN A, GEN *pden, GEN T)
{
  pari_sp av = avma;
  long m = lg(A)-1, n, k1 = 1, k2;
  GEN H = NULL, D, H1 = NULL, mod1 = NULL, worker;
  ulong bnd, mask;
  forprime_t S;
  pari_timer ti;

  if (m == 0) return ZM_inv0(A,pden);
  if (pden) *pden = gen_1;
  if (nbrows(A) < m) return NULL;
  if (m == 1 && nbrows(A)==1 && !T) return ZM_inv1(A,pden);
  if (m == 2 && nbrows(A)==2 && !T) return ZM_inv2(A,pden);

  if (DEBUGLEVEL>=5) timer_start(&ti);
  init_modular_big(&S);
  bnd = expi(RgM_true_Hadamard(A));
  worker = snm_closure(is_entry("_ZM_inv_worker"), mkvec(A));
  gen_inccrt("ZM_inv_r", worker, NULL, k1, 0, &S, &H1, &mod1, nmV_chinese_center, FpM_center);
  n = (bnd+1)/expu(S.p)+1;
  if (DEBUGLEVEL>=5) timer_printf(&ti,"inv (%ld/%ld primes)", k1, n);
  mask = quadratic_prec_mask(n);
  for (k2 = 0;;)
  {
    GEN Hr;
    if (k2 > 0)
    {
      gen_inccrt("ZM_inv_r", worker, NULL, k2, 0, &S, &H1, &mod1,nmV_chinese_center,FpM_center);
      k1 += k2;
      if (DEBUGLEVEL>=5) timer_printf(&ti,"CRT (%ld/%ld primes)", k1, n);
    }
    if (mask == 1) break;
    k2 = (mask&1UL) ? k1-1: k1;
    mask >>= 1;

    Hr = ZM_adj_ratlift(A, H1, mod1, T);
    if (DEBUGLEVEL>=5) timer_printf(&ti,"ratlift (%ld/%ld primes)", k1, n);
    if (Hr) {/* DONE ? */
      GEN Hl = Q_primpart(Hr), R = ZM_mul(Hl, A), d = gcoeff(R,1,1);
      if (gsigne(d) < 0) { d = gneg(d); Hl = ZM_neg(Hl); }
      if (DEBUGLEVEL>=5) timer_printf(&ti,"mult (%ld/%ld primes)", k1, n);
      if (equali1(d))
      {
        if (ZM_isidentity(R)) { H = Hl; break; }
      }
      else if (ZM_isscalar(R, d))
      {
        if (T) T = gdiv(T,d);
        else if (pden) *pden = d;
        H = Hl; break;
      }
    }
  }
  if (!H)
  {
    GEN d;
    H = H1;
    D = ZMrow_ZC_mul(H, gel(A,1), 1);
    if (signe(D)==0) pari_err_INV("ZM_inv", A);
    if (T) T = gdiv(T, D);
    else
    {
      d = gcdii(Q_content_safe(H), D);
      if (signe(D) < 0) d = negi(d);
      if (!equali1(d))
      {
        H = ZM_Z_divexact(H, d);
        D = diviiexact(D, d);
      }
      if (pden) *pden = D;
    }
  }
  if (T && !isint1(T)) H = ZM_Q_mul(H, T);
  return gc_all(av, pden? 2: 1, &H, pden);
}
GEN
ZM_inv(GEN A, GEN *pden) { return ZM_inv_i(A, pden, NULL); }

/* same as above, M rational */
GEN
QM_inv(GEN M)
{
  pari_sp av = avma;
  GEN den, dM, K;
  M = Q_remove_denom(M, &dM);
  K = ZM_inv_i(M, &den, dM);
  if (!K) return gc_NULL(av);
  if (den && !equali1(den)) K = ZM_Q_mul(K, ginv(den));
  return gerepileupto(av, K);
}

static GEN
ZM_ker_filter(GEN A, GEN P)
{
  long i, j, l = lg(A), n = 1, d = lg(gmael(A,1,1));
  GEN B, Q, D = gmael(A,1,2);
  for (i=2; i<l; i++)
  {
    GEN Di = gmael(A,i,2);
    long di = lg(gmael(A,i,1));
    int c = vecsmall_lexcmp(D, Di);
    if (di==d && c==0) n++;
    else if (d > di || (di==d && c>0))
    { n = 1; d = di; D = Di; }
  }
  B = cgetg(n+1, t_VEC);
  Q = cgetg(n+1, typ(P));
  for (i=1, j=1; i<l; i++)
  {
    if (lg(gmael(A,i,1))==d &&  vecsmall_lexcmp(D, gmael(A,i,2))==0)
    {
      gel(B,j) = gmael(A,i,1);
      Q[j] = P[i];
      j++;
    }
  }
  return mkvec3(B,Q,D);
}

static GEN
ZM_ker_chinese(GEN A, GEN P, GEN *mod)
{
  GEN BQD = ZM_ker_filter(A, P);
  return mkvec2(nmV_chinese_center(gel(BQD,1), gel(BQD,2), mod), gel(BQD,3));
}

static GEN
ZM_ker_slice(GEN A, GEN P, GEN *mod)
{
  pari_sp av = avma;
  long i, n = lg(P)-1;
  GEN BQD, D, H, T, Q;
  if (n == 1)
  {
    ulong p = uel(P,1);
    GEN K = Flm_ker_sp(ZM_to_Flm(A, p), p, 2);
    *mod = utoipos(p); return mkvec2(Flm_to_ZM(gel(K,1)), gel(K,2));
  }
  T = ZV_producttree(P);
  A = ZM_nv_mod_tree(A, P, T);
  H = cgetg(n+1, t_VEC);
  for(i=1 ; i <= n; i++)
    gel(H,i) = Flm_ker_sp(gel(A, i), P[i], 2);
  BQD = ZM_ker_filter(H, P); Q = gel(BQD,2);
  if (lg(Q) != lg(P)) T = ZV_producttree(Q);
  H = nmV_chinese_center_tree_seq(gel(BQD,1), Q, T, ZV_chinesetree(Q,T));
  *mod = gmael(T, lg(T)-1, 1);
  D = gel(BQD, 3);
  gerepileall(av, 3, &H, &D, mod);
  return mkvec2(H,D);
}

GEN
ZM_ker_worker(GEN P, GEN A)
{
  GEN V = cgetg(3, t_VEC);
  gel(V,1) = ZM_ker_slice(A, P, &gel(V,2));
  return V;
}

/* assume lg(A) > 1 */
static GEN
ZM_ker_i(GEN A)
{
  pari_sp av;
  long k, m = lg(A)-1;
  GEN HD = NULL, mod = gen_1, worker;
  forprime_t S;

  if (m >= 2*nbrows(A))
  {
    GEN v = ZM_indexrank(A), y = gel(v,2), z = indexcompl(y, m);
    GEN B, A1, A1i, d;
    A = rowpermute(A, gel(v,1)); /* same kernel */
    A1 = vecpermute(A, y); /* maximal rank submatrix */
    B = vecpermute(A, z);
    A1i = ZM_inv(A1, &d);
    if (!d) d = gen_1;
    B = vconcat(ZM_mul(ZM_neg(A1i), B), scalarmat_shallow(d, lg(B)-1));
    if (!gequal(y, identity_perm(lg(y)-1)))
      B = rowpermute(B, perm_inv(shallowconcat(y,z)));
    return vec_Q_primpart(B);
  }
  init_modular_big(&S);
  worker = snm_closure(is_entry("_ZM_ker_worker"), mkvec(A));
  av = avma;
  for (k = 1;; k <<= 1)
  {
    pari_timer ti;
    GEN H, Hr;
    gen_inccrt_i("ZM_ker", worker, NULL, (k+1)>>1, 0,
                 &S, &HD, &mod, ZM_ker_chinese, NULL);
    gerepileall(av, 2, &HD, &mod);
    H = gel(HD, 1); if (lg(H) == 1) return H;
    if (DEBUGLEVEL >= 4) timer_start(&ti);
    Hr = FpM_ratlift_parallel(H, mod, NULL);
    if (DEBUGLEVEL >= 4) timer_printf(&ti,"ZM_ker: ratlift (%ld)",!!Hr);
    if (Hr)
    {
      GEN MH;
      Hr = vec_Q_primpart(Hr);
      MH = ZM_mul(A, Hr);
      if (DEBUGLEVEL >= 4) timer_printf(&ti,"ZM_ker: QM_mul");
      if (ZM_equal0(MH)) return Hr;
    }
  }
}

GEN
ZM_ker(GEN M)
{
  pari_sp av = avma;
  long l = lg(M)-1;
  if (l==0) return cgetg(1, t_MAT);
  if (lgcols(M)==1) return matid(l);
  return gerepilecopy(av, ZM_ker_i(M));
}

GEN
QM_ker(GEN M)
{
  pari_sp av = avma;
  long l = lg(M)-1;
  if (l==0) return cgetg(1, t_MAT);
  if (lgcols(M)==1) return matid(l);
  return gerepilecopy(av, ZM_ker_i(row_Q_primpart(M)));
}

/* x a ZM. Return a multiple of the determinant of the lattice generated by
 * the columns of x. From Algorithm 2.2.6 in GTM138 */
GEN
detint(GEN A)
{
  if (typ(A) != t_MAT) pari_err_TYPE("detint",A);
  RgM_check_ZM(A, "detint");
  return ZM_detmult(A);
}
GEN
ZM_detmult(GEN A)
{
  pari_sp av1, av = avma;
  GEN B, c, v, piv;
  long rg, i, j, k, m, n = lg(A) - 1;

  if (!n) return gen_1;
  m = nbrows(A);
  if (n < m) return gen_0;
  c = zero_zv(m);
  av1 = avma;
  B = zeromatcopy(m,m);
  v = cgetg(m+1, t_COL);
  piv = gen_1; rg = 0;
  for (k=1; k<=n; k++)
  {
    GEN pivprec = piv;
    long t = 0;
    for (i=1; i<=m; i++)
    {
      pari_sp av2 = avma;
      GEN vi;
      if (c[i]) continue;

      vi = mulii(piv, gcoeff(A,i,k));
      for (j=1; j<=m; j++)
        if (c[j]) vi = addii(vi, mulii(gcoeff(B,j,i),gcoeff(A,j,k)));
      if (!t && signe(vi)) t = i;
      gel(v,i) = gerepileuptoint(av2, vi);
    }
    if (!t) continue;
    /* at this point c[t] = 0 */

    if (++rg >= m) { /* full rank; mostly done */
      GEN det = gel(v,t); /* last on stack */
      if (++k > n)
        det = absi(det);
      else
      {
        /* improve further; at this point c[i] is set for all i != t */
        gcoeff(B,t,t) = piv; v = centermod(gel(B,t), det);
        for ( ; k<=n; k++)
          det = gcdii(det, ZV_dotproduct(v, gel(A,k)));
      }
      return gerepileuptoint(av, det);
    }

    piv = gel(v,t);
    for (i=1; i<=m; i++)
    {
      GEN mvi;
      if (c[i] || i == t) continue;

      gcoeff(B,t,i) = mvi = negi(gel(v,i));
      for (j=1; j<=m; j++)
        if (c[j]) /* implies j != t */
        {
          pari_sp av2 = avma;
          GEN z = addii(mulii(gcoeff(B,j,i), piv), mulii(gcoeff(B,j,t), mvi));
          if (rg > 1) z = diviiexact(z, pivprec);
          gcoeff(B,j,i) = gerepileuptoint(av2, z);
        }
    }
    c[t] = k;
    if (gc_needed(av,1))
    {
      if(DEBUGMEM>1) pari_warn(warnmem,"detint. k=%ld",k);
      gerepileall(av1, 2, &piv,&B); v = zerovec(m);
    }
  }
  return gc_const(av, gen_0);
}

/* Reduce x modulo (invertible) y */
GEN
closemodinvertible(GEN x, GEN y)
{
  return gmul(y, ground(RgM_solve(y,x)));
}
GEN
reducemodinvertible(GEN x, GEN y)
{
  return gsub(x, closemodinvertible(x,y));
}
GEN
reducemodlll(GEN x,GEN y)
{
  return reducemodinvertible(x, ZM_lll(y, 0.75, LLL_INPLACE));
}

/*******************************************************************/
/*                                                                 */
/*                    KERNEL of an m x n matrix                    */
/*          return n - rk(x) linearly independent vectors          */
/*                                                                 */
/*******************************************************************/
static GEN
RgM_deplin_i(GEN x0)
{
  pari_sp av = avma, av2;
  long i, j, k, nl, nc = lg(x0)-1;
  GEN D, x, y, c, l, d, ck;

  if (!nc) return NULL;
  nl = nbrows(x0);
  c = zero_zv(nl);
  l = cgetg(nc+1, t_VECSMALL); /* not initialized */
  av2 = avma;
  x = RgM_shallowcopy(x0);
  d = const_vec(nl, gen_1); /* pivot list */
  ck = NULL; /* gcc -Wall */
  for (k=1; k<=nc; k++)
  {
    ck = gel(x,k);
    for (j=1; j<k; j++)
    {
      GEN cj = gel(x,j), piv = gel(d,j), q = gel(ck,l[j]);
      for (i=1; i<=nl; i++)
        if (i!=l[j]) gel(ck,i) = gsub(gmul(piv, gel(ck,i)), gmul(q, gel(cj,i)));
    }

    i = gauss_get_pivot_NZ(x, NULL, k, c);
    if (i > nl) break;
    if (gc_needed(av,1))
    {
      if (DEBUGMEM>1) pari_warn(warnmem,"deplin k = %ld/%ld",k,nc);
      gerepileall(av2, 2, &x, &d);
      ck = gel(x,k);
    }
    gel(d,k) = gel(ck,i);
    c[i] = k; l[k] = i; /* pivot d[k] in x[i,k] */
  }
  if (k > nc) return gc_NULL(av);
  if (k == 1) { set_avma(av); return scalarcol_shallow(gen_1,nc); }
  y = cgetg(nc+1,t_COL);
  gel(y,1) = gcopy(gel(ck, l[1]));
  for (D=gel(d,1),j=2; j<k; j++)
  {
    gel(y,j) = gmul(gel(ck, l[j]), D);
    D = gmul(D, gel(d,j));
  }
  gel(y,j) = gneg(D);
  for (j++; j<=nc; j++) gel(y,j) = gen_0;
  y = primitive_part(y, &c);
  return c? gerepileupto(av, y): gerepilecopy(av, y);
}
static GEN
RgV_deplin(GEN v)
{
  pari_sp av = avma;
  long n = lg(v)-1;
  GEN y, p = NULL;
  if (n <= 1)
  {
    if (n == 1 && gequal0(gel(v,1))) return mkcol(gen_1);
    return cgetg(1, t_COL);
  }
  if (gequal0(gel(v,1))) return scalarcol_shallow(gen_1, n);
  v = primpart(mkvec2(gel(v,1),gel(v,2)));
  if (RgV_is_FpV(v, &p) && p) v = centerlift(v);
  y = zerocol(n);
  gel(y,1) = gneg(gel(v,2));
  gel(y,2) = gcopy(gel(v,1));
  return gerepileupto(av, y);

}

static GEN
RgM_deplin_FpM(GEN x, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  x = RgM_Fp_init3(x, p, &pp);
  switch(pp)
  {
  case 0:
    x = FpM_ker_gen(x,p,1);
    if (!x) return gc_NULL(av);
    x = FpC_center(x,p,shifti(p,-1));
    break;
  case 2:
    x = F2m_ker_sp(x,1);
    if (!x) return gc_NULL(av);
    x = F2c_to_ZC(x); break;
  case 3:
    x = F3m_ker_sp(x,1);
    if (!x) return gc_NULL(av);
    x = F3c_to_ZC(x); break;
  default:
    x = Flm_ker_sp(x,pp,1);
    if (!x) return gc_NULL(av);
    x = Flv_center(x, pp, pp>>1);
    x = zc_to_ZC(x);
    break;
  }
  return gerepileupto(av, x);
}

/* FIXME: implement direct modular ZM_deplin ? */
static GEN
QM_deplin(GEN M)
{
  pari_sp av = avma;
  long l = lg(M)-1;
  GEN k;
  if (l==0) return NULL;
  if (lgcols(M)==1) return col_ei(l, 1);
  k = ZM_ker_i(row_Q_primpart(M));
  if (lg(k)== 1) return gc_NULL(av);
  return gerepilecopy(av, gel(k,1));
}

static GEN
RgM_deplin_FqM(GEN x, GEN pol, GEN p)
{
  pari_sp av = avma;
  GEN b, T = RgX_to_FpX(pol, p);
  if (signe(T) == 0) pari_err_OP("deplin",x,pol);
  b = FqM_deplin(RgM_to_FqM(x, T, p), T, p);
  if (!b) return gc_NULL(av);
  return gerepileupto(av, b);
}

#define code(t1,t2) ((t1 << 6) | t2)
static GEN
RgM_deplin_fast(GEN x)
{
  GEN p, pol;
  long pa;
  long t = RgM_type(x, &p,&pol,&pa);
  switch(t)
  {
    case t_INT:    /* fall through */
    case t_FRAC:   return QM_deplin(x);
    case t_FFELT:  return FFM_deplin(x, pol);
    case t_INTMOD: return RgM_deplin_FpM(x, p);
    case code(t_POLMOD, t_INTMOD):
                   return RgM_deplin_FqM(x, pol, p);
    default:       return gen_0;
  }
}
#undef code

static GEN
RgM_deplin(GEN x)
{
  GEN z = RgM_deplin_fast(x);
  if (z!= gen_0) return z;
  return RgM_deplin_i(x);
}

GEN
deplin(GEN x)
{
  switch(typ(x))
  {
    case t_MAT:
    {
      GEN z = RgM_deplin(x);
      if (z) return z;
      return cgetg(1, t_COL);
    }
    case t_VEC: return RgV_deplin(x);
    default: pari_err_TYPE("deplin",x);
  }
  return NULL;/*LCOV_EXCL_LINE*/
}

/*******************************************************************/
/*                                                                 */
/*         GAUSS REDUCTION OF MATRICES  (m lines x n cols)         */
/*           (kernel, image, complementary image, rank)            */
/*                                                                 */
/*******************************************************************/
/* return the transform of x under a standard Gauss pivot.
 * x0 is a reference point when guessing whether x[i,j] ~ 0
 * (iff x[i,j] << x0[i,j])
 * Set r = dim ker(x). d[k] contains the index of the first nonzero pivot
 * in column k */
static GEN
gauss_pivot_ker(GEN x, GEN x0, GEN *dd, long *rr)
{
  GEN c, d, p, data;
  pari_sp av;
  long i, j, k, r, t, n, m;
  pivot_fun pivot;

  n=lg(x)-1; if (!n) { *dd=NULL; *rr=0; return cgetg(1,t_MAT); }
  m=nbrows(x); r=0;
  pivot = get_pivot_fun(x, x0, &data);
  x = RgM_shallowcopy(x);
  c = zero_zv(m);
  d = cgetg(n+1,t_VECSMALL);
  av=avma;
  for (k=1; k<=n; k++)
  {
    j = pivot(x, data, k, c);
    if (j > m)
    {
      r++; d[k]=0;
      for(j=1; j<k; j++)
        if (d[j]) gcoeff(x,d[j],k) = gclone(gcoeff(x,d[j],k));
    }
    else
    { /* pivot for column k on row j */
      c[j]=k; d[k]=j; p = gdiv(gen_m1,gcoeff(x,j,k));
      gcoeff(x,j,k) = gen_m1;
      /* x[j,] /= - x[j,k] */
      for (i=k+1; i<=n; i++) gcoeff(x,j,i) = gmul(p,gcoeff(x,j,i));
      for (t=1; t<=m; t++)
        if (t!=j)
        { /* x[t,] -= 1 / x[j,k] x[j,] */
          p = gcoeff(x,t,k); gcoeff(x,t,k) = gen_0;
          if (gequal0(p)) continue;
          for (i=k+1; i<=n; i++)
            gcoeff(x,t,i) = gadd(gcoeff(x,t,i),gmul(p,gcoeff(x,j,i)));
          if (gc_needed(av,1)) gerepile_gauss_ker(x,k,t,av);
        }
    }
  }
  *dd=d; *rr=r; return x;
}

/* r = dim ker(x).
 * Returns d:
 *   d[k] != 0 contains the index of a nonzero pivot in column k
 *   d[k] == 0 if column k is a linear combination of the (k-1) first ones */
GEN
RgM_pivots(GEN x0, GEN data, long *rr, pivot_fun pivot)
{
  GEN x, c, d, p;
  long i, j, k, r, t, m, n = lg(x0)-1;
  pari_sp av;

  if (RgM_is_ZM(x0)) return ZM_pivots(x0, rr);
  if (!n) { *rr = 0; return NULL; }

  d = cgetg(n+1, t_VECSMALL);
  x = RgM_shallowcopy(x0);
  m = nbrows(x); r = 0;
  c = zero_zv(m);
  av = avma;
  for (k=1; k<=n; k++)
  {
    j = pivot(x, data, k, c);
    if (j > m) { r++; d[k] = 0; }
    else
    {
      c[j] = k; d[k] = j; p = gdiv(gen_m1, gcoeff(x,j,k));
      for (i=k+1; i<=n; i++) gcoeff(x,j,i) = gmul(p,gcoeff(x,j,i));

      for (t=1; t<=m; t++)
        if (!c[t]) /* no pivot on that line yet */
        {
          p = gcoeff(x,t,k); gcoeff(x,t,k) = gen_0;
          for (i=k+1; i<=n; i++)
            gcoeff(x,t,i) = gadd(gcoeff(x,t,i), gmul(p, gcoeff(x,j,i)));
          if (gc_needed(av,1)) gerepile_gauss(x,k,t,av,j,c);
        }
      for (i=k; i<=n; i++) gcoeff(x,j,i) = gen_0; /* dummy */
    }
  }
  *rr = r; return gc_const((pari_sp)d, d);
}

static long
ZM_count_0_cols(GEN M)
{
  long i, l = lg(M), n = 0;
  for (i = 1; i < l; i++)
    if (ZV_equal0(gel(M,i))) n++;
  return n;
}

static void indexrank_all(long m, long n, long r, GEN d, GEN *prow, GEN *pcol);
/* As RgM_pivots, integer entries. Set *rr = dim Ker M0 */
GEN
ZM_pivots(GEN M0, long *rr)
{
  GEN d, dbest = NULL;
  long m, mm, n, nn, i, imax, rmin, rbest, zc;
  int beenthere = 0;
  pari_sp av, av0 = avma;
  forprime_t S;

  rbest = n = lg(M0)-1;
  if (n == 0) { *rr = 0; return NULL; }
  zc = ZM_count_0_cols(M0);
  if (n == zc) { *rr = zc; return zero_zv(n); }

  m = nbrows(M0);
  rmin = maxss(zc, n-m);
  init_modular_small(&S);
  if (n <= m) { nn = n; mm = m; } else { nn = m; mm = n; }
  imax = (nn < 16)? 1: (nn < 64)? 2: 3; /* heuristic */

  for(;;)
  {
    GEN row, col, M, KM, IM, RHS, X, cX;
    long rk;
    for (av = avma, i = 0;; set_avma(av), i++)
    {
      ulong p = u_forprime_next(&S);
      long rp;
      if (!p) pari_err_OVERFLOW("ZM_pivots [ran out of primes]");
      d = Flm_pivots(ZM_to_Flm(M0, p), p, &rp, 1);
      if (rp == rmin) { rbest = rp; goto END; } /* maximal rank, return */
      if (rp < rbest) { /* save best r so far */
        rbest = rp;
        guncloneNULL(dbest);
        dbest = gclone(d);
        if (beenthere) break;
      }
      if (!beenthere && i >= imax) break;
    }
    beenthere = 1;
    /* Dubious case: there is (probably) a non trivial kernel */
    indexrank_all(m,n, rbest, dbest, &row, &col);
    M = rowpermute(vecpermute(M0, col), row);
    rk = n - rbest; /* (probable) dimension of image */
    if (n > m) M = shallowtrans(M);
    IM = vecslice(M,1,rk);
    KM = vecslice(M,rk+1, nn);
    M = rowslice(IM, 1,rk); /* square maximal rank */
    X = ZM_gauss(M, rowslice(KM, 1,rk));
    RHS = rowslice(KM,rk+1,mm);
    M = rowslice(IM,rk+1,mm);
    X = Q_remove_denom(X, &cX);
    if (cX) RHS = ZM_Z_mul(RHS, cX);
    if (ZM_equal(ZM_mul(M, X), RHS)) { d = vecsmall_copy(dbest); goto END; }
    set_avma(av);
  }
END:
  *rr = rbest; guncloneNULL(dbest);
  return gerepileuptoleaf(av0, d);
}

/* set *pr = dim Ker x */
static GEN
gauss_pivot(GEN x, long *pr) {
  GEN data;
  pivot_fun pivot = get_pivot_fun(x, x, &data);
  return RgM_pivots(x, data, pr, pivot);
}

/* compute ker(x), x0 is a reference point when guessing whether x[i,j] ~ 0
 * (iff x[i,j] << x0[i,j]) */
static GEN
ker_aux(GEN x, GEN x0)
{
  pari_sp av = avma;
  GEN d,y;
  long i,j,k,r,n;

  x = gauss_pivot_ker(x,x0,&d,&r);
  if (!r) { set_avma(av); return cgetg(1,t_MAT); }
  n = lg(x)-1; y=cgetg(r+1,t_MAT);
  for (j=k=1; j<=r; j++,k++)
  {
    GEN p = cgetg(n+1,t_COL);

    gel(y,j) = p; while (d[k]) k++;
    for (i=1; i<k; i++)
      if (d[i])
      {
        GEN p1=gcoeff(x,d[i],k);
        gel(p,i) = gcopy(p1); gunclone(p1);
      }
      else
        gel(p,i) = gen_0;
    gel(p,k) = gen_1; for (i=k+1; i<=n; i++) gel(p,i) = gen_0;
  }
  return gerepileupto(av,y);
}

static GEN
RgM_ker_FpM(GEN x, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  x = RgM_Fp_init3(x, p, &pp);
  switch(pp)
  {
    case 0: x = FpM_to_mod(FpM_ker_gen(x,p,0),p); break;
    case 2: x = F2m_to_mod(F2m_ker_sp(x,0)); break;
    case 3: x = F3m_to_mod(F3m_ker_sp(x,0)); break;
    default:x = Flm_to_mod(Flm_ker_sp(x,pp,0), pp); break;
  }
  return gerepileupto(av, x);
}

static GEN
RgM_ker_FqM(GEN x, GEN pol, GEN p)
{
  pari_sp av = avma;
  GEN b, T = RgX_to_FpX(pol, p);
  if (signe(T) == 0) pari_err_OP("ker",x,pol);
  b = FqM_ker(RgM_to_FqM(x, T, p), T, p);
  return gerepileupto(av, FqM_to_mod(b, T, p));
}

#define code(t1,t2) ((t1 << 6) | t2)
static GEN
RgM_ker_fast(GEN x)
{
  GEN p, pol;
  long pa;
  long t = RgM_type(x, &p,&pol,&pa);
  switch(t)
  {
    case t_INT:    /* fall through */
    case t_FRAC:   return QM_ker(x);
    case t_FFELT:  return FFM_ker(x, pol);
    case t_INTMOD: return RgM_ker_FpM(x, p);
    case code(t_POLMOD, t_INTMOD):
                   return RgM_ker_FqM(x, pol, p);
    default:       return NULL;
  }
}
#undef code

GEN
ker(GEN x)
{
  GEN b = RgM_ker_fast(x);
  if (b) return b;
  return ker_aux(x,x);
}

GEN
matker0(GEN x,long flag)
{
  if (typ(x)!=t_MAT) pari_err_TYPE("matker",x);
  if (!flag) return ker(x);
  RgM_check_ZM(x, "matker");
  return ZM_ker(x);
}

static GEN
RgM_image_FpM(GEN x, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  x = RgM_Fp_init(x, p, &pp);
  switch(pp)
  {
    case 0: x = FpM_to_mod(FpM_image(x,p),p); break;
    case 2: x = F2m_to_mod(F2m_image(x)); break;
    default:x = Flm_to_mod(Flm_image(x,pp), pp); break;
  }
  return gerepileupto(av, x);
}

static GEN
RgM_image_FqM(GEN x, GEN pol, GEN p)
{
  pari_sp av = avma;
  GEN b, T = RgX_to_FpX(pol, p);
  if (signe(T) == 0) pari_err_OP("image",x,pol);
  b = FqM_image(RgM_to_FqM(x, T, p), T, p);
  return gerepileupto(av, FqM_to_mod(b, T, p));
}

GEN
QM_image_shallow(GEN A)
{
  A = vec_Q_primpart(A);
  return vecpermute(A, ZM_indeximage(A));
}
GEN
QM_image(GEN A)
{
  pari_sp av = avma;
  return gerepilecopy(av, QM_image_shallow(A));
}

#define code(t1,t2) ((t1 << 6) | t2)
static GEN
RgM_image_fast(GEN x)
{
  GEN p, pol;
  long pa;
  long t = RgM_type(x, &p,&pol,&pa);
  switch(t)
  {
    case t_INT:    /* fall through */
    case t_FRAC:   return QM_image(x);
    case t_FFELT:  return FFM_image(x, pol);
    case t_INTMOD: return RgM_image_FpM(x, p);
    case code(t_POLMOD, t_INTMOD):
                   return RgM_image_FqM(x, pol, p);
    default:       return NULL;
  }
}
#undef code

GEN
image(GEN x)
{
  GEN d, M;
  long r;

  if (typ(x)!=t_MAT) pari_err_TYPE("matimage",x);
  M = RgM_image_fast(x);
  if (M) return M;
  d = gauss_pivot(x,&r); /* d left on stack for efficiency */
  return image_from_pivot(x,d,r);
}

static GEN
imagecompl_aux(GEN x, GEN(*PIVOT)(GEN,long*))
{
  pari_sp av = avma;
  GEN d,y;
  long j,i,r;

  if (typ(x)!=t_MAT) pari_err_TYPE("imagecompl",x);
  (void)new_chunk(lg(x) * 4 + 1); /* HACK */
  d = PIVOT(x,&r); /* if (!d) then r = 0 */
  set_avma(av); y = cgetg(r+1,t_VECSMALL);
  for (i=j=1; j<=r; i++)
    if (!d[i]) y[j++] = i;
  return y;
}
GEN
imagecompl(GEN x) { return imagecompl_aux(x, &gauss_pivot); }
GEN
ZM_imagecompl(GEN x) { return imagecompl_aux(x, &ZM_pivots); }

static GEN
RgM_RgC_invimage_FpC(GEN A, GEN y, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  GEN x;
  A = RgM_Fp_init(A,p,&pp);
  switch(pp)
  {
  case 0:
    y = RgC_to_FpC(y,p);
    x = FpM_FpC_invimage(A, y, p);
    return x ? gerepileupto(av, FpC_to_mod(x,p)): NULL;
  case 2:
    y = RgV_to_F2v(y);
    x = F2m_F2c_invimage(A, y);
    return x ? gerepileupto(av, F2c_to_mod(x)): NULL;
  default:
    y = RgV_to_Flv(y,pp);
    x = Flm_Flc_invimage(A, y, pp);
    return x ? gerepileupto(av, Flc_to_mod(x,pp)): NULL;
  }
}

static GEN
RgM_RgC_invimage_fast(GEN x, GEN y)
{
  GEN p, pol;
  long pa;
  long t = RgM_RgC_type(x, y, &p,&pol,&pa);
  switch(t)
  {
    case t_INTMOD: return RgM_RgC_invimage_FpC(x, y, p);
    case t_FFELT:  return FFM_FFC_invimage(x, y, pol);
    default:       return gen_0;
  }
}

GEN
RgM_RgC_invimage(GEN A, GEN y)
{
  pari_sp av = avma;
  long i, l = lg(A);
  GEN M, x, t;
  if (l==1) return NULL;
  if (lg(y) != lgcols(A)) pari_err_DIM("inverseimage");
  M = RgM_RgC_invimage_fast(A, y);
  if (!M) return gc_NULL(av);
  if (M != gen_0) return M;
  M = ker(shallowconcat(A, y));
  i = lg(M)-1;
  if (!i) return gc_NULL(av);

  x = gel(M,i); t = gel(x,l);
  if (gequal0(t)) return gc_NULL(av);

  t = gneg_i(t); setlg(x,l);
  return gerepileupto(av, RgC_Rg_div(x, t));
}

/* Return X such that m X = v (t_COL or t_MAT), resp. an empty t_COL / t_MAT
 * if no solution exist */
GEN
inverseimage(GEN m, GEN v)
{
  GEN y;
  if (typ(m)!=t_MAT) pari_err_TYPE("inverseimage",m);
  switch(typ(v))
  {
    case t_COL:
      y = RgM_RgC_invimage(m,v);
      return y? y: cgetg(1,t_COL);
    case t_MAT:
      y = RgM_invimage(m, v);
      return y? y: cgetg(1,t_MAT);
  }
  pari_err_TYPE("inverseimage",v);
  return NULL;/*LCOV_EXCL_LINE*/
}

static GEN
RgM_invimage_FpM(GEN A, GEN B, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  GEN x;
  A = RgM_Fp_init(A,p,&pp);
  switch(pp)
  {
  case 0:
    B = RgM_to_FpM(B,p);
    x = FpM_invimage_gen(A, B, p);
    return x ? gerepileupto(av, FpM_to_mod(x, p)): x;
  case 2:
    B = RgM_to_F2m(B);
    x = F2m_invimage_i(A, B);
    return x ? gerepileupto(av, F2m_to_mod(x)): x;
  default:
    B = RgM_to_Flm(B,pp);
    x = Flm_invimage_i(A, B, pp);
    return x ? gerepileupto(av, Flm_to_mod(x, pp)): x;
  }
}

static GEN
RgM_invimage_fast(GEN x, GEN y)
{
  GEN p, pol;
  long pa;
  long t = RgM_type2(x, y, &p,&pol,&pa);
  switch(t)
  {
    case t_INTMOD: return RgM_invimage_FpM(x, y, p);
    case t_FFELT:  return FFM_invimage(x, y, pol);
    default:       return gen_0;
  }
}

/* find Z such that A Z = B. Return NULL if no solution */
GEN
RgM_invimage(GEN A, GEN B)
{
  pari_sp av = avma;
  GEN d, x, X, Y;
  long i, j, nY, nA = lg(A)-1, nB = lg(B)-1;
  X = RgM_invimage_fast(A, B);
  if (!X) return gc_NULL(av);
  if (X != gen_0) return X;
  x = ker(shallowconcat(RgM_neg(A), B));
  /* AX = BY, Y in strict upper echelon form with pivots = 1.
   * We must find T such that Y T = Id_nB then X T = Z. This exists iff
   * Y has at least nB columns and full rank */
  nY = lg(x)-1;
  if (nY < nB) return gc_NULL(av);
  Y = rowslice(x, nA+1, nA+nB); /* nB rows */
  d = cgetg(nB+1, t_VECSMALL);
  for (i = nB, j = nY; i >= 1; i--, j--)
  {
    for (; j>=1; j--)
      if (!gequal0(gcoeff(Y,i,j))) { d[i] = j; break; }
    if (!j) return gc_NULL(av);
  }
  /* reduce to the case Y square, upper triangular with 1s on diagonal */
  Y = vecpermute(Y, d);
  x = vecpermute(x, d);
  X = rowslice(x, 1, nA);
  return gerepileupto(av, RgM_mul(X, RgM_inv_upper(Y)));
}

static GEN
RgM_suppl_FpM(GEN x, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  x = RgM_Fp_init(x, p, &pp);
  switch(pp)
  {
  case 0: x = FpM_to_mod(FpM_suppl(x,p), p); break;
  case 2: x = F2m_to_mod(F2m_suppl(x)); break;
  default:x = Flm_to_mod(Flm_suppl(x,pp), pp); break;
  }
  return gerepileupto(av, x);
}

static GEN
RgM_suppl_fast(GEN x)
{
  GEN p, pol;
  long pa;
  long t = RgM_type(x,&p,&pol,&pa);
  switch(t)
  {
    case t_INTMOD: return RgM_suppl_FpM(x, p);
    case t_FFELT:  return FFM_suppl(x, pol);
    default:       return NULL;
  }
}

/* x is an n x k matrix, rank(x) = k <= n. Return an invertible n x n matrix
 * whose first k columns are given by x. If rank(x) < k, undefined result. */
GEN
suppl(GEN x)
{
  pari_sp av = avma;
  GEN d, M;
  long r;
  if (typ(x)!=t_MAT) pari_err_TYPE("suppl",x);
  M = RgM_suppl_fast(x);
  if (M) return M;
  init_suppl(x);
  d = gauss_pivot(x,&r);
  set_avma(av); return get_suppl(x,d,nbrows(x),r,&col_ei);
}

GEN
image2(GEN x)
{
  pari_sp av = avma;
  long k, n, i;
  GEN A, B;

  if (typ(x)!=t_MAT) pari_err_TYPE("image2",x);
  if (lg(x) == 1) return cgetg(1,t_MAT);
  A = ker(x); k = lg(A)-1;
  if (!k) { set_avma(av); return gcopy(x); }
  A = suppl(A); n = lg(A)-1;
  B = cgetg(n-k+1, t_MAT);
  for (i = k+1; i <= n; i++) gel(B,i-k) = RgM_RgC_mul(x, gel(A,i));
  return gerepileupto(av, B);
}

GEN
matimage0(GEN x,long flag)
{
  switch(flag)
  {
    case 0: return image(x);
    case 1: return image2(x);
    default: pari_err_FLAG("matimage");
  }
  return NULL; /* LCOV_EXCL_LINE */
}

static long
RgM_rank_FpM(GEN x, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  long r;
  x = RgM_Fp_init(x,p,&pp);
  switch(pp)
  {
  case 0: r = FpM_rank(x,p); break;
  case 2: r = F2m_rank(x); break;
  default:r = Flm_rank(x,pp); break;
  }
  return gc_long(av, r);
}

static long
RgM_rank_FqM(GEN x, GEN pol, GEN p)
{
  pari_sp av = avma;
  long r;
  GEN T = RgX_to_FpX(pol, p);
  if (signe(T) == 0) pari_err_OP("rank",x,pol);
  r = FqM_rank(RgM_to_FqM(x, T, p), T, p);
  return gc_long(av,r);
}

#define code(t1,t2) ((t1 << 6) | t2)
static long
RgM_rank_fast(GEN x)
{
  GEN p, pol;
  long pa;
  long t = RgM_type(x,&p,&pol,&pa);
  switch(t)
  {
    case t_INT:    return ZM_rank(x);
    case t_FRAC:   return QM_rank(x);
    case t_INTMOD: return RgM_rank_FpM(x, p);
    case t_FFELT:  return FFM_rank(x, pol);
    case code(t_POLMOD, t_INTMOD):
                   return RgM_rank_FqM(x, pol, p);
    default:       return -1;
  }
}
#undef code

long
rank(GEN x)
{
  pari_sp av = avma;
  long r;

  if (typ(x)!=t_MAT) pari_err_TYPE("rank",x);
  r = RgM_rank_fast(x);
  if (r >= 0) return r;
  (void)gauss_pivot(x, &r);
  return gc_long(av, lg(x)-1 - r);
}

/* d a t_VECSMALL of integers in 1..n. Return the vector of the d[i]
 * followed by the missing indices */
static GEN
perm_complete(GEN d, long n)
{
  GEN y = cgetg(n+1, t_VECSMALL);
  long i, j = 1, k = n, l = lg(d);
  pari_sp av = avma;
  char *T = stack_calloc(n+1);
  for (i = 1; i < l; i++) T[d[i]] = 1;
  for (i = 1; i <= n; i++)
    if (T[i]) y[j++] = i; else y[k--] = i;
  return gc_const(av, y);
}

/* n = dim x, r = dim Ker(x), d from gauss_pivot */
static GEN
indeximage0(long n, long r, GEN d)
{
  long i, j;
  GEN v;

  r = n - r; /* now r = dim Im(x) */
  v = cgetg(r+1,t_VECSMALL);
  if (d) for (i=j=1; j<=n; j++)
    if (d[j]) v[i++] = j;
  return v;
}
/* x an m x n t_MAT, n > 0, r = dim Ker(x), d from gauss_pivot */
static void
indexrank_all(long m, long n, long r, GEN d, GEN *prow, GEN *pcol)
{
  GEN IR = indexrank0(n, r, d);
  *prow = perm_complete(gel(IR,1), m);
  *pcol = perm_complete(gel(IR,2), n);
}

static GEN
RgM_indexrank_FpM(GEN x, GEN p)
{
  pari_sp av = avma;
  ulong pp;
  GEN r;
  x = RgM_Fp_init(x,p,&pp);
  switch(pp)
  {
  case 0:  r = FpM_indexrank(x,p); break;
  case 2:  r = F2m_indexrank(x); break;
  default: r = Flm_indexrank(x,pp); break;
  }
  return gerepileupto(av, r);
}

static GEN
RgM_indexrank_FqM(GEN x, GEN pol, GEN p)
{
  pari_sp av = avma;
  GEN r, T = RgX_to_FpX(pol, p);
  if (signe(T) == 0) pari_err_OP("indexrank",x,pol);
  r = FqM_indexrank(RgM_to_FqM(x, T, p), T, p);
  return gerepileupto(av, r);
}

#define code(t1,t2) ((t1 << 6) | t2)
static GEN
RgM_indexrank_fast(GEN x)
{
  GEN p, pol;
  long pa;
  long t = RgM_type(x,&p,&pol,&pa);
  switch(t)
  {
    case t_INT:    return ZM_indexrank(x);
    case t_FRAC:   return QM_indexrank(x);
    case t_INTMOD: return RgM_indexrank_FpM(x, p);
    case t_FFELT:  return FFM_indexrank(x, pol);
    case code(t_POLMOD, t_INTMOD):
                   return RgM_indexrank_FqM(x, pol, p);
    default:       return NULL;
  }
}
#undef code

GEN
indexrank(GEN x)
{
  pari_sp av;
  long r;
  GEN d;
  if (typ(x)!=t_MAT) pari_err_TYPE("indexrank",x);
  d = RgM_indexrank_fast(x);
  if (d) return d;
  av = avma;
  init_indexrank(x);
  d = gauss_pivot(x, &r);
  set_avma(av); return indexrank0(lg(x)-1, r, d);
}

GEN
ZM_indeximage(GEN x) {
  pari_sp av = avma;
  long r;
  GEN d;
  init_indexrank(x);
  d = ZM_pivots(x,&r);
  set_avma(av); return indeximage0(lg(x)-1, r, d);
}
long
ZM_rank(GEN x) {
  pari_sp av = avma;
  long r;
  (void)ZM_pivots(x,&r);
  return gc_long(av, lg(x)-1-r);
}
GEN
ZM_indexrank(GEN x) {
  pari_sp av = avma;
  long r;
  GEN d;
  init_indexrank(x);
  d = ZM_pivots(x,&r);
  set_avma(av); return indexrank0(lg(x)-1, r, d);
}

long
QM_rank(GEN x)
{
  pari_sp av = avma;
  long r = ZM_rank(Q_primpart(x));
  set_avma(av);
  return r;
}

GEN
QM_indexrank(GEN x)
{
  pari_sp av = avma;
  GEN r = ZM_indexrank(Q_primpart(x));
  return gerepileupto(av, r);
}

/*******************************************************************/
/*                                                                 */
/*                             ZabM                                */
/*                                                                 */
/*******************************************************************/

static GEN
FpXM_ratlift(GEN a, GEN q)
{
  GEN B, y;
  long i, j, l = lg(a), n;
  B = sqrti(shifti(q,-1));
  y = cgetg(l, t_MAT);
  if (l==1) return y;
  n = lgcols(a);
  for (i=1; i<l; i++)
  {
    GEN yi = cgetg(n, t_COL);
    for (j=1; j<n; j++)
    {
      GEN v = FpX_ratlift(gmael(a,i,j), q, B, B, NULL);
      if (!v) return NULL;
      gel(yi, j) = RgX_renormalize(v);
    }
    gel(y,i) = yi;
  }
  return y;
}

static GEN
FlmV_recover_pre(GEN a, GEN M, ulong p, ulong pi, long sv)
{
  GEN a1 = gel(a,1);
  long i, j, k, l = lg(a1), n, lM = lg(M);
  GEN v = cgetg(lM, t_VECSMALL);
  GEN y = cgetg(l, t_MAT);
  if (l==1) return y;
  n = lgcols(a1);
  for (i=1; i<l; i++)
  {
    GEN yi = cgetg(n, t_COL);
    for (j=1; j<n; j++)
    {
      for (k=1; k<lM; k++) uel(v,k) = umael(gel(a,k),i,j);
      gel(yi, j) = Flm_Flc_mul_pre_Flx(M, v, p, pi, sv);
    }
    gel(y,i) = yi;
  }
  return y;
}

static GEN
FlkM_inv(GEN M, GEN P, ulong p)
{
  ulong PI = get_Fl_red(p), pi = SMALL_ULONG(p)? 0: PI;
  GEN R = Flx_roots_pre(P, p, pi);
  long l = lg(R), i;
  GEN W = Flv_invVandermonde(R, 1UL, p);
  GEN V = cgetg(l, t_VEC);
  for(i=1; i<l; i++)
  {
    GEN pows = Fl_powers_pre(uel(R,i), degpol(P), p, PI);
    GEN H = Flm_inv_sp(FlxM_eval_powers_pre(M, pows, p, pi), NULL, p);
    if (!H) return NULL;
    gel(V, i) = H;
  }
  return FlmV_recover_pre(V, W, p, pi, P[1]);
}

static GEN
FlkM_adjoint(GEN M, GEN P, ulong p)
{
  ulong PI = get_Fl_red(p), pi = SMALL_ULONG(p)? 0: PI;
  GEN R = Flx_roots_pre(P, p, pi);
  long l = lg(R), i;
  GEN W = Flv_invVandermonde(R, 1UL, p);
  GEN V = cgetg(l, t_VEC);
  for(i=1; i<l; i++)
  {
    GEN pows = Fl_powers_pre(uel(R,i), degpol(P), p, PI);
    gel(V, i) = Flm_adjoint(FlxM_eval_powers_pre(M, pows, p, pi), p);
  }
  return FlmV_recover_pre(V, W, p, pi, P[1]);
}

static GEN
ZabM_inv_slice(GEN A, GEN Q, GEN P, GEN *mod)
{
  pari_sp av = avma;
  long i, n = lg(P)-1, w = varn(Q);
  GEN H, T;
  if (n == 1)
  {
    ulong p = uel(P,1);
    GEN Qp = ZX_to_Flx(Q, p);
    GEN Ap = ZXM_to_FlxM(A, p, get_Flx_var(Qp));
    GEN Hp = FlkM_adjoint(Ap, Qp, p);
    Hp = gerepileupto(av, FlxM_to_ZXM(Hp));
    *mod = utoipos(p); return Hp;
  }
  T = ZV_producttree(P);
  A = ZXM_nv_mod_tree(A, P, T, w);
  Q = ZX_nv_mod_tree(Q, P, T);
  H = cgetg(n+1, t_VEC);
  for(i=1; i <= n; i++)
  {
    ulong p = P[i];
    GEN a = gel(A,i), q = gel(Q, i);
    gel(H,i) = FlkM_adjoint(a, q, p);
  }
  H = nxMV_chinese_center_tree_seq(H, P, T, ZV_chinesetree(P,T));
  *mod = gmael(T, lg(T)-1, 1); return gc_all(av, 2, &H, mod);
}

GEN
ZabM_inv_worker(GEN P, GEN A, GEN Q)
{
  GEN V = cgetg(3, t_VEC);
  gel(V,1) = ZabM_inv_slice(A, Q, P, &gel(V,2));
  return V;
}

static GEN
vecnorml1(GEN a)
{
  long i, l;
  GEN g = cgetg_copy(a, &l);
  for (i=1; i<l; i++)
    gel(g, i) = gnorml1_fake(gel(a,i));
  return g;
}

static GEN
ZabM_true_Hadamard(GEN a)
{
  pari_sp av = avma;
  long n = lg(a)-1, i;
  GEN B;
  if (n == 0) return gen_1;
  if (n == 1) return gnorml1_fake(gcoeff(a,1,1));
  B = gen_1;
  for (i = 1; i <= n; i++)
    B = gmul(B, gnorml2(RgC_gtofp(vecnorml1(gel(a,i)),DEFAULTPREC)));
  return gerepileuptoint(av, ceil_safe(sqrtr_abs(B)));
}

GEN
ZabM_inv(GEN A, GEN Q, long n, GEN *pt_den)
{
  pari_sp av = avma;
  forprime_t S;
  GEN bnd, H, D, d, mod, worker;
  if (lg(A) == 1)
  {
    if (pt_den) *pt_den = gen_1;
    return cgetg(1, t_MAT);
  }
  bnd = ZabM_true_Hadamard(A);
  worker = snm_closure(is_entry("_ZabM_inv_worker"), mkvec2(A, Q));
  u_forprime_arith_init(&S, HIGHBIT+1, ULONG_MAX, 1, n);
  H = gen_crt("ZabM_inv", worker, &S, NULL, expi(bnd), 0, &mod,
              nxMV_chinese_center, FpXM_center);
  D = RgMrow_RgC_mul(H, gel(A,1), 1);
  D = ZX_rem(D, Q);
  d = Z_content(mkvec2(H, D));
  if (d)
  {
    D = ZX_Z_divexact(D, d);
    H = Q_div_to_int(H, d);
  }
  if (!pt_den) return gerepileupto(av, H);
  *pt_den = D; return gc_all(av, 2, &H, pt_den);
}

GEN
ZabM_inv_ratlift(GEN M, GEN P, long n, GEN *pden)
{
  pari_sp av2, av = avma;
  GEN q, H;
  ulong m = LONG_MAX>>1;
  ulong p= 1 + m - (m % n);
  long lM = lg(M);
  if (lM == 1) { *pden = gen_1; return cgetg(1,t_MAT); }

  av2 = avma;
  H = NULL;
  for(;;)
  {
    GEN Hp, Pp, Mp, Hr;
    do p += n; while(!uisprime(p));
    Pp = ZX_to_Flx(P, p);
    Mp = ZXM_to_FlxM(M, p, get_Flx_var(Pp));
    Hp = FlkM_inv(Mp, Pp, p);
    if (!Hp) continue;
    if (!H)
    {
      H = ZXM_init_CRT(Hp, degpol(P)-1, p);
      q = utoipos(p);
    }
    else
      ZXM_incremental_CRT(&H, Hp, &q, p);
    Hr = FpXM_ratlift(H, q);
    if (DEBUGLEVEL>5) err_printf("ZabM_inv mod %ld (ratlift=%ld)\n", p,!!Hr);
    if (Hr) {/* DONE ? */
      GEN Hl = Q_remove_denom(Hr, pden);
      GEN MH = ZXQM_mul(Hl, M, P);
      if (*pden)
      { if (RgM_isscalar(MH, *pden)) { H = Hl; break; }}
      else
      { if (RgM_isidentity(MH)) { H = Hl; *pden = gen_1; break; } }
    }

    if (gc_needed(av,2))
    {
      if (DEBUGMEM>1) pari_warn(warnmem,"ZabM_inv");
      gerepileall(av2, 2, &H, &q);
    }
  }
  return gc_all(av, 2, &H, pden);
}

static GEN
FlkM_ker(GEN M, GEN P, ulong p)
{
  ulong PI = get_Fl_red(p), pi = SMALL_ULONG(p)? 0: PI;
  GEN R = Flx_roots_pre(P, p, pi);
  long l = lg(R), i, dP = degpol(P), r;
  GEN M1, K, D;
  GEN W = Flv_invVandermonde(R, 1UL, p);
  GEN V = cgetg(l, t_VEC);
  M1 = FlxM_eval_powers_pre(M, Fl_powers_pre(uel(R,1), dP, p, PI), p, pi);
  K = Flm_ker_sp(M1, p, 2);
  r = lg(gel(K,1)); D = gel(K,2);
  gel(V, 1) = gel(K,1);
  for(i=2; i<l; i++)
  {
    GEN Mi = FlxM_eval_powers_pre(M, Fl_powers_pre(uel(R,i), dP, p, PI), p, pi);
    GEN K = Flm_ker_sp(Mi, p, 2);
    if (lg(gel(K,1)) != r || !zv_equal(D, gel(K,2))) return NULL;
    gel(V, i) = gel(K,1);
  }
  return mkvec2(FlmV_recover_pre(V, W, p, pi, P[1]), D);
}

static int
ZabM_ker_check(GEN M, GEN H, ulong p, GEN P, long n)
{
  GEN pow;
  long j, l = lg(H);
  ulong pi, r;
  do p += n; while(!uisprime(p));
  pi = get_Fl_red(p);
  P = ZX_to_Flx(P, p);
  r = Flx_oneroot_pre(P, p, pi);
  pow = Fl_powers_pre(r, degpol(P),p, (p & HIGHMASK)? pi: 0);
  M = ZXM_to_FlxM(M, p, P[1]); M = FlxM_eval_powers_pre(M, pow, p, pi);
  H = ZXM_to_FlxM(H, p, P[1]); H = FlxM_eval_powers_pre(H, pow, p, pi);
  for (j = 1; j < l; j++)
    if (!zv_equal0(Flm_Flc_mul_pre(M, gel(H,j), p, pi))) return 0;
  return 1;
}

GEN
ZabM_ker(GEN M, GEN P, long n)
{
  pari_sp av = avma;
  pari_timer ti;
  GEN q, H = NULL, D = NULL;
  ulong m = LONG_MAX>>1;
  ulong p = 1 + m - (m % n);

  if (DEBUGLEVEL>5) timer_start(&ti);
  for(;;)
  {
    GEN Kp, Hp, Dp, Pp, Mp, Hr;
    do p += n; while(!uisprime(p));
    Pp = ZX_to_Flx(P, p);
    Mp = ZXM_to_FlxM(M, p, get_Flx_var(Pp));
    Kp = FlkM_ker(Mp, Pp, p);
    if (!Kp) continue;
    Hp = gel(Kp,1); Dp = gel(Kp,2);
    if (H && (lg(Hp)>lg(H) || (lg(Hp)==lg(H) && vecsmall_lexcmp(Dp,D)>0))) continue;
    if (!H || (lg(Hp)<lg(H) || vecsmall_lexcmp(Dp,D)<0))
    {
      H = ZXM_init_CRT(Hp, degpol(P)-1, p); D = Dp;
      q = utoipos(p);
    }
    else
      ZXM_incremental_CRT(&H, Hp, &q, p);
    Hr = FpXM_ratlift(H, q);
    if (DEBUGLEVEL>5) timer_printf(&ti,"ZabM_ker mod %ld (ratlift=%ld)", p,!!Hr);
    if (Hr) {/* DONE ? */
      GEN Hl = vec_Q_primpart(Hr);
      if (ZabM_ker_check(M, Hl, p, P, n)) { H = Hl;  break; }
    }

    if (gc_needed(av,2))
    {
      if (DEBUGMEM>1) pari_warn(warnmem,"ZabM_ker");
      gerepileall(av, 3, &H, &D, &q);
    }
  }
  return gerepilecopy(av, H);
}

GEN
ZabM_indexrank(GEN M, GEN P, long n)
{
  pari_sp av = avma;
  ulong m = LONG_MAX>>1;
  ulong p = 1+m-(m%n), D = degpol(P);
  long lM = lg(M), lmax = 0, c = 0;
  GEN v;
  for(;;)
  {
    GEN R, Pp, Mp, K;
    ulong pi;
    long l;
    do p += n; while (!uisprime(p));
    pi = (p & HIGHMASK)? get_Fl_red(p): 0;
    Pp = ZX_to_Flx(P, p);
    R = Flx_roots_pre(Pp, p, pi);
    Mp = ZXM_to_FlxM(M, p, get_Flx_var(Pp));
    K = FlxM_eval_powers_pre(Mp, Fl_powers_pre(uel(R,1), D,p,pi), p,pi);
    v = Flm_indexrank(K, p);
    l = lg(gel(v,2));
    if (l == lM) break;
    if (lmax >= 0 && l > lmax) { lmax = l; c = 0; } else c++;
    if (c > 2)
    { /* probably not maximal rank, expensive check */
      lM -= lg(ZabM_ker(M, P, n))-1; /* actual rank (+1) */
      if (lmax == lM) break;
      lmax = -1; /* disable check */
    }
  }
  return gerepileupto(av, v);
}

#if 0
GEN
ZabM_gauss(GEN M, GEN P, long n, GEN *den)
{
  pari_sp av = avma;
  GEN v, S, W;
  v = ZabM_indexrank(M, P, n);
  S = shallowmatextract(M,gel(v,1),gel(v,2));
  W = ZabM_inv(S, P, n, den);
  return gc_all(av,2,&W,den);
}
#endif

GEN
ZabM_pseudoinv(GEN M, GEN P, long n, GEN *pv, GEN *den)
{
  GEN v = ZabM_indexrank(M, P, n);
  if (pv) *pv = v;
  M = shallowmatextract(M,gel(v,1),gel(v,2));
  return ZabM_inv(M, P, n, den);
}
GEN
ZM_pseudoinv(GEN M, GEN *pv, GEN *den)
{
  GEN v = ZM_indexrank(M);
  if (pv) *pv = v;
  M = shallowmatextract(M,gel(v,1),gel(v,2));
  return ZM_inv(M, den);
}

/*******************************************************************/
/*                                                                 */
/*                   Structured Elimination                        */
/*                                                                 */
/*******************************************************************/

static void
rem_col(GEN c, long i, GEN iscol, GEN Wrow, long *rcol, long *rrow)
{
  long lc = lg(c), k;
  iscol[i] = 0; (*rcol)--;
  for (k = 1; k < lc; ++k)
  {
    Wrow[c[k]]--;
    if (Wrow[c[k]]==0) (*rrow)--;
  }
}

static void
rem_singleton(GEN M, GEN iscol, GEN Wrow, long idx, long *rcol, long *rrow)
{
  long i, j;
  long nbcol = lg(iscol)-1, last;
  do
  {
    last = 0;
    for (i = 1; i <= nbcol; ++i)
      if (iscol[i])
      {
        GEN c = idx ? gmael(M, i, idx): gel(M,i);
        long lc = lg(c);
        for (j = 1; j < lc; ++j)
          if (Wrow[c[j]] == 1)
          {
            rem_col(c, i, iscol, Wrow, rcol, rrow);
            last=1; break;
          }
      }
  } while (last);
}

static GEN
fill_wcol(GEN M, GEN iscol, GEN Wrow, long *w, GEN wcol)
{
  long nbcol = lg(iscol)-1;
  long i, j, m, last;
  GEN per;
  for (m = 2, last=0; !last ; m++)
  {
    for (i = 1; i <= nbcol; ++i)
    {
      wcol[i] = 0;
      if (iscol[i])
      {
        GEN c = gmael(M, i, 1);
        long lc = lg(c);
        for (j = 1; j < lc; ++j)
          if (Wrow[c[j]] == m) {  wcol[i]++; last = 1; }
      }
    }
  }
  per = vecsmall_indexsort(wcol);
  *w = wcol[per[nbcol]];
  return per;
}

/* M is a RgMs with nbrow rows, A a list of row indices.
   Eliminate rows of M with a single entry that do not belong to A,
   and the corresponding columns. Also eliminate columns until #colums=#rows.
   Return pcol and prow:
   pcol is a map from the new columns indices to the old one.
   prow is a map from the old rows indices to the new one (0 if removed).
*/

void
RgMs_structelim_col(GEN M, long nbcol, long nbrow, GEN A, GEN *p_col, GEN *p_row)
{
  long i, j, k, lA = lg(A);
  GEN prow = cgetg(nbrow+1, t_VECSMALL);
  GEN pcol = zero_zv(nbcol);
  pari_sp av = avma;
  long rcol = nbcol, rrow = 0, imin = nbcol - usqrt(nbcol);
  GEN iscol = const_vecsmall(nbcol, 1);
  GEN Wrow  = zero_zv(nbrow);
  GEN wcol = cgetg(nbcol+1, t_VECSMALL);
  pari_sp av2 = avma;
  for (i = 1; i <= nbcol; ++i)
  {
    GEN F = gmael(M, i, 1);
    long l = lg(F)-1;
    for (j = 1; j <= l; ++j) Wrow[F[j]]++;
  }
  for (j = 1; j < lA; ++j)
  {
    if (Wrow[A[j]] == 0) { *p_col=NULL; return; }
    Wrow[A[j]] = -1;
  }
  for (i = 1; i <= nbrow; ++i)
    if (Wrow[i]) rrow++;
  rem_singleton(M, iscol, Wrow, 1, &rcol, &rrow);
  if (rcol < rrow) pari_err_BUG("RgMs_structelim, rcol<rrow");
  while (rcol > rrow)
  {
    long w;
    GEN per = fill_wcol(M, iscol, Wrow, &w, wcol);
    for (i = nbcol; i>=imin && wcol[per[i]]>=w && rcol>rrow; i--)
      rem_col(gmael(M, per[i], 1), per[i], iscol, Wrow, &rcol, &rrow);
    rem_singleton(M, iscol, Wrow, 1, &rcol, &rrow); set_avma(av2);
  }
  for (j = 1, i = 1; i <= nbcol; ++i)
    if (iscol[i]) pcol[j++] = i;
  setlg(pcol,j);
  for (k = 1, i = 1; i <= nbrow; ++i) prow[i] = Wrow[i]? k++: 0;
  *p_col = pcol; *p_row = prow; set_avma(av);
}

void
RgMs_structelim(GEN M, long nbrow, GEN A, GEN *p_col, GEN *p_row)
{ RgMs_structelim_col(M, lg(M)-1, nbrow, A, p_col, p_row); }

GEN
F2Ms_colelim(GEN M, long nbrow)
{
  long i,j, nbcol = lg(M)-1, rcol = nbcol, rrow = 0;
  GEN pcol = zero_zv(nbcol);
  pari_sp av = avma;
  GEN iscol = const_vecsmall(nbcol, 1), Wrow  = zero_zv(nbrow);
  for (i = 1; i <= nbcol; ++i)
  {
    GEN F = gel(M, i);
    long l = lg(F)-1;
    for (j = 1; j <= l; ++j) Wrow[F[j]]++;
  }
  rem_singleton(M, iscol, Wrow, 0, &rcol, &rrow);
  for (j = 1, i = 1; i <= nbcol; ++i)
    if (iscol[i]) pcol[j++] = i;
  fixlg(pcol,j); return gc_const(av, pcol);
}

/*******************************************************************/
/*                                                                 */
/*                        EIGENVECTORS                             */
/*   (independent eigenvectors, sorted by increasing eigenvalue)   */
/*                                                                 */
/*******************************************************************/
/* assume x is square of dimension > 0 */
static int
RgM_is_symmetric_cx(GEN x, long bit)
{
  pari_sp av = avma;
  long i, j, l = lg(x);
  for (i = 1; i < l; i++)
    for (j = 1; j < i; j++)
    {
      GEN a = gcoeff(x,i,j), b = gcoeff(x,j,i), c = gsub(a,b);
      if (!gequal0(c) && gexpo(c) - gexpo(a) > -bit) return gc_long(av,0);
    }
  return gc_long(av,1);
}
static GEN
eigen_err(int exact, GEN x, long flag, long prec)
{
  pari_sp av = avma;
  if (RgM_is_symmetric_cx(x, prec2nbits(prec) - 10))
  { /* approximately symmetric: recover */
    x = jacobi(x, prec); if (flag) return x;
    return gerepilecopy(av, gel(x,2));
  }
  if (exact)
  {
    GEN y = mateigen(x, flag, precdbl(prec));
    return gerepilecopy(av, gprec_wtrunc(y, prec));
  }
  pari_err_PREC("mateigen");
  return NULL; /* LCOV_EXCL_LINE */
}
GEN
mateigen(GEN x, long flag, long prec)
{
  GEN y, R, T;
  long k, l, ex, n = lg(x);
  int exact;
  pari_sp av = avma;

  if (typ(x)!=t_MAT) pari_err_TYPE("eigen",x);
  if (n != 1 && n != lgcols(x)) pari_err_DIM("eigen");
  if (flag < 0 || flag > 1) pari_err_FLAG("mateigen");
  if (n == 1)
  {
    if (flag) retmkvec2(cgetg(1,t_VEC), cgetg(1,t_MAT));
    return cgetg(1,t_VEC);
  }
  if (n == 2)
  {
    if (flag) retmkvec2(mkveccopy(gcoeff(x,1,1)), matid(1));
    return matid(1);
  }

  ex = 16 - prec2nbits(prec);
  T = charpoly(x,0);
  exact = RgX_is_QX(T);
  if (exact)
  {
    T = ZX_radical( Q_primpart(T) );
    R = nfrootsQ(T);
    if (lg(R)-1 < degpol(T))
    { /* add missing complex roots */
      GEN r = cleanroots(RgX_div(T, roots_to_pol(R, 0)), prec);
      settyp(r, t_VEC);
      R = shallowconcat(R, r);
    }
  }
  else
  {
    GEN r1, v = vectrunc_init(lg(T));
    long e;
    R = cleanroots(T,prec);
    r1 = NULL;
    for (k = 1; k < lg(R); k++)
    {
      GEN r2 = gel(R,k), r = grndtoi(r2, &e);
      if (e < ex) r2 = r;
      if (r1)
      {
        r = gsub(r1,r2);
        if (gequal0(r) || gexpo(r) < ex) continue;
      }
      vectrunc_append(v, r2);
      r1 = r2;
    }
    R = v;
  }
  /* R = distinct complex roots of charpoly(x) */
  l = lg(R); y = cgetg(l, t_VEC);
  for (k = 1; k < l; k++)
  {
    GEN F = ker_aux(RgM_Rg_sub_shallow(x, gel(R,k)), x);
    long d = lg(F)-1;
    if (!d) { set_avma(av); return eigen_err(exact, x, flag, prec); }
    gel(y,k) = F;
    if (flag) gel(R,k) = const_vec(d, gel(R,k));
  }
  y = shallowconcat1(y);
  if (lg(y) > n) { set_avma(av); return eigen_err(exact, x, flag, prec); }
  /* lg(y) < n if x is not diagonalizable */
  if (flag) y = mkvec2(shallowconcat1(R), y);
  return gerepilecopy(av,y);
}
GEN
eigen(GEN x, long prec) { return mateigen(x, 0, prec); }

/*******************************************************************/
/*                                                                 */
/*                           DETERMINANT                           */
/*                                                                 */
/*******************************************************************/

GEN
det0(GEN a,long flag)
{
  switch(flag)
  {
    case 0: return det(a);
    case 1: return det2(a);
    default: pari_err_FLAG("matdet");
  }
  return NULL; /* LCOV_EXCL_LINE */
}

/* M a 2x2 matrix, returns det(M) */
static GEN
RgM_det2(GEN M)
{
  pari_sp av = avma;
  GEN a = gcoeff(M,1,1), b = gcoeff(M,1,2);
  GEN c = gcoeff(M,2,1), d = gcoeff(M,2,2);
  return gerepileupto(av, gsub(gmul(a,d), gmul(b,c)));
}
/* M a 2x2 ZM, returns det(M) */
static GEN
ZM_det2(GEN M)
{
  pari_sp av = avma;
  GEN a = gcoeff(M,1,1), b = gcoeff(M,1,2);
  GEN c = gcoeff(M,2,1), d = gcoeff(M,2,2);
  return gerepileuptoint(av, subii(mulii(a,d), mulii(b, c)));
}
/* M a 3x3 ZM, return det(M) */
static GEN
ZM_det3(GEN M)
{
  pari_sp av = avma;
  GEN a = gcoeff(M,1,1), b = gcoeff(M,1,2), c = gcoeff(M,1,3);
  GEN d = gcoeff(M,2,1), e = gcoeff(M,2,2), f = gcoeff(M,2,3);
  GEN g = gcoeff(M,3,1), h = gcoeff(M,3,2), i = gcoeff(M,3,3);
  GEN t, D = signe(i)? mulii(subii(mulii(a,e), mulii(b,d)), i): gen_0;
  if (signe(g))
  {
    t = mulii(subii(mulii(b,f), mulii(c,e)), g);
    D = addii(D, t);
  }
  if (signe(h))
  {
    t = mulii(subii(mulii(c,d), mulii(a,f)), h);
    D = addii(D, t);
  }
  return gerepileuptoint(av, D);
}

static GEN
det_simple_gauss(GEN a, GEN data, pivot_fun pivot)
{
  pari_sp av = avma;
  long i,j,k, s = 1, nbco = lg(a)-1;
  GEN p, x = gen_1;

  a = RgM_shallowcopy(a);
  for (i=1; i<nbco; i++)
  {
    k = pivot(a, data, i, NULL);
    if (k > nbco) return gerepilecopy(av, gcoeff(a,i,i));
    if (k != i)
    { /* exchange the lines s.t. k = i */
      for (j=i; j<=nbco; j++) swap(gcoeff(a,i,j), gcoeff(a,k,j));
      s = -s;
    }
    p = gcoeff(a,i,i);

    x = gmul(x,p);
    for (k=i+1; k<=nbco; k++)
    {
      GEN m = gcoeff(a,i,k);
      if (gequal0(m)) continue;

      m = gdiv(m,p);
      for (j=i+1; j<=nbco; j++)
        gcoeff(a,j,k) = gsub(gcoeff(a,j,k), gmul(m,gcoeff(a,j,i)));
    }
    if (gc_needed(av,2))
    {
      if(DEBUGMEM>1) pari_warn(warnmem,"det. col = %ld",i);
      gerepileall(av,2, &a,&x);
    }
  }
  if (s < 0) x = gneg_i(x);
  return gerepileupto(av, gmul(x, gcoeff(a,nbco,nbco)));
}

GEN
det2(GEN a)
{
  GEN data;
  pivot_fun pivot;
  long n = lg(a)-1;
  if (typ(a)!=t_MAT) pari_err_TYPE("det2",a);
  if (!n) return gen_1;
  if (n != nbrows(a)) pari_err_DIM("det2");
  if (n == 1) return gcopy(gcoeff(a,1,1));
  if (n == 2) return RgM_det2(a);
  pivot = get_pivot_fun(a, a, &data);
  return det_simple_gauss(a, data, pivot);
}

/* Assumes a a square t_MAT of dimension n > 0. Returns det(a) using
 * Gauss-Bareiss. */
static GEN
det_bareiss(GEN a)
{
  pari_sp av = avma;
  long nbco = lg(a)-1,i,j,k,s = 1;
  GEN p, pprec;

  a = RgM_shallowcopy(a);
  for (pprec=gen_1,i=1; i<nbco; i++,pprec=p)
  {
    int diveuc = (gequal1(pprec)==0);
    GEN ci;

    p = gcoeff(a,i,i);
    if (gequal0(p))
    {
      k=i+1; while (k<=nbco && gequal0(gcoeff(a,i,k))) k++;
      if (k>nbco) return gerepilecopy(av, p);
      swap(gel(a,k), gel(a,i)); s = -s;
      p = gcoeff(a,i,i);
    }
    ci = gel(a,i);
    for (k=i+1; k<=nbco; k++)
    {
      GEN ck = gel(a,k), m = gel(ck,i);
      if (gequal0(m))
      {
        if (gequal1(p))
        {
          if (diveuc)
            gel(a,k) = gdiv(gel(a,k), pprec);
        }
        else
          for (j=i+1; j<=nbco; j++)
          {
            GEN p1 = gmul(p, gel(ck,j));
            if (diveuc) p1 = gdiv(p1,pprec);
            gel(ck,j) = p1;
          }
      }
      else
        for (j=i+1; j<=nbco; j++)
        {
          pari_sp av2 = avma;
          GEN p1 = gsub(gmul(p,gel(ck,j)), gmul(m,gel(ci,j)));
          if (diveuc) p1 = gdiv(p1,pprec);
          gel(ck,j) = gerepileupto(av2, p1);
        }
      if (gc_needed(av,2))
      {
        if(DEBUGMEM>1) pari_warn(warnmem,"det. col = %ld",i);
        gerepileall(av,2, &a,&pprec);
        ci = gel(a,i);
        p = gcoeff(a,i,i);
      }
    }
  }
  p = gcoeff(a,nbco,nbco);
  p = (s < 0)? gneg(p): gcopy(p);
  return gerepileupto(av, p);
}

/* count nonzero entries in col j, at most 'max' of them.
 * Return their indices */
static GEN
col_count_non_zero(GEN a, long j, long max)
{
  GEN v = cgetg(max+1, t_VECSMALL);
  GEN c = gel(a,j);
  long i, l = lg(a), k = 1;
  for (i = 1; i < l; i++)
    if (!gequal0(gel(c,i)))
    {
      if (k > max) return NULL; /* fail */
      v[k++] = i;
    }
  setlg(v, k); return v;
}
/* count nonzero entries in row i, at most 'max' of them.
 * Return their indices */
static GEN
row_count_non_zero(GEN a, long i, long max)
{
  GEN v = cgetg(max+1, t_VECSMALL);
  long j, l = lg(a), k = 1;
  for (j = 1; j < l; j++)
    if (!gequal0(gcoeff(a,i,j)))
    {
      if (k > max) return NULL; /* fail */
      v[k++] = j;
    }
  setlg(v, k); return v;
}

static GEN det_develop(GEN a, long max, double bound);
/* (-1)^(i+j) a[i,j] * det RgM_minor(a,i,j) */
static GEN
coeff_det(GEN a, long i, long j, long max, double bound)
{
  GEN c = gcoeff(a, i, j);
  c = gmul(c, det_develop(RgM_minor(a, i,j), max, bound));
  if (odd(i+j)) c = gneg(c);
  return c;
}
/* a square t_MAT, 'bound' a rough upper bound for the number of
 * multiplications we are willing to pay while developing rows/columns before
 * switching to Gaussian elimination */
static GEN
det_develop(GEN M, long max, double bound)
{
  pari_sp av = avma;
  long i,j, n = lg(M)-1, lbest = max+2, best_col = 0, best_row = 0;
  GEN best = NULL;

  if (bound < 1.) return det_bareiss(M); /* too costly now */

  switch(n)
  {
    case 0: return gen_1;
    case 1: return gcopy(gcoeff(M,1,1));
    case 2: return RgM_det2(M);
  }
  if (max > ((n+2)>>1)) max = (n+2)>>1;
  for (j = 1; j <= n; j++)
  {
    pari_sp av2 = avma;
    GEN v = col_count_non_zero(M, j, max);
    long lv;
    if (!v || (lv = lg(v)) >= lbest) { set_avma(av2); continue; }
    if (lv == 1) { set_avma(av); return gen_0; }
    if (lv == 2) {
      set_avma(av);
      return gerepileupto(av, coeff_det(M,v[1],j,max,bound));
    }
    best = v; lbest = lv; best_col = j;
  }
  for (i = 1; i <= n; i++)
  {
    pari_sp av2 = avma;
    GEN v = row_count_non_zero(M, i, max);
    long lv;
    if (!v || (lv = lg(v)) >= lbest) { set_avma(av2); continue; }
    if (lv == 1) { set_avma(av); return gen_0; }
    if (lv == 2) {
      set_avma(av);
      return gerepileupto(av, coeff_det(M,i,v[1],max,bound));
    }
    best = v; lbest = lv; best_row = i;
  }
  if (best_row)
  {
    double d = lbest-1;
    GEN s = NULL;
    long k;
    bound /= d*d*d;
    for (k = 1; k < lbest; k++)
    {
      GEN c = coeff_det(M, best_row, best[k], max, bound);
      s = s? gadd(s, c): c;
    }
    return gerepileupto(av, s);
  }
  if (best_col)
  {
    double d = lbest-1;
    GEN s = NULL;
    long k;
    bound /= d*d*d;
    for (k = 1; k < lbest; k++)
    {
      GEN c = coeff_det(M, best[k], best_col, max, bound);
      s = s? gadd(s, c): c;
    }
    return gerepileupto(av, s);
  }
  return det_bareiss(M);
}

/* area of parallelogram bounded by (v1,v2) */
static GEN
parallelogramarea(GEN v1, GEN v2)
{ return gsub(gmul(gnorml2(v1), gnorml2(v2)), gsqr(RgV_dotproduct(v1, v2))); }

/* Square of Hadamard bound for det(a), a square matrix.
 * Slight improvement: instead of using the column norms, use the area of
 * the parallelogram formed by pairs of consecutive vectors */
GEN
RgM_Hadamard(GEN a)
{
  pari_sp av = avma;
  long n = lg(a)-1, i;
  GEN B;
  if (n == 0) return gen_1;
  if (n == 1) return gsqr(gcoeff(a,1,1));
  a = RgM_gtofp(a, LOWDEFAULTPREC);
  B = gen_1;
  for (i = 1; i <= n/2; i++)
    B = gmul(B, parallelogramarea(gel(a,2*i-1), gel(a,2*i)));
  if (odd(n)) B = gmul(B, gnorml2(gel(a, n)));
  return gerepileuptoint(av, ceil_safe(B));
}

/* If B=NULL, assume B=A' */
static GEN
ZM_det_slice(GEN A, GEN P, GEN *mod)
{
  pari_sp av = avma;
  long i, n = lg(P)-1;
  GEN H, T;
  if (n == 1)
  {
    ulong Hp, p = uel(P,1);
    GEN a = ZM_to_Flm(A, p);
    Hp = Flm_det_sp(a, p);
    set_avma(av); *mod = utoipos(p); return utoi(Hp);
  }
  T = ZV_producttree(P);
  A = ZM_nv_mod_tree(A, P, T);
  H = cgetg(n+1, t_VECSMALL);
  for(i=1; i <= n; i++)
  {
    ulong p = P[i];
    GEN a = gel(A,i);
    H[i] = Flm_det_sp(a, p);
  }
  H = ZV_chinese_tree(H, P, T, ZV_chinesetree(P,T));
  *mod = gmael(T, lg(T)-1, 1); return gc_all(av, 2, &H, mod);
}

GEN
ZM_det_worker(GEN P, GEN A)
{
  GEN V = cgetg(3, t_VEC);
  gel(V,1) = ZM_det_slice(A, P, &gel(V,2));
  return V;
}

GEN
ZM_det(GEN M)
{
  const long DIXON_THRESHOLD = 40;
  pari_sp av, av2;
  long i, n = lg(M)-1;
  ulong p, Dp;
  forprime_t S;
  pari_timer ti;
  GEN H, D, mod, h, q, v, worker;
#ifdef LONG_IS_64BIT
  const ulong PMAX = 18446744073709551557UL;
#else
  const ulong PMAX = 4294967291UL;
#endif

  switch(n)
  {
    case 0: return gen_1;
    case 1: return icopy(gcoeff(M,1,1));
    case 2: return ZM_det2(M);
    case 3: return ZM_det3(M);
  }
  if (DEBUGLEVEL>=4) timer_start(&ti);
  av = avma; h = RgM_Hadamard(M); /* |D| <= sqrt(h) */
  if (!signe(h)) { set_avma(av); return gen_0; }
  h = sqrti(h);
  if (lgefint(h) == 3 && (ulong)h[2] <= (PMAX >> 1))
  { /* h < p/2 => direct result */
    p = PMAX;
    Dp = Flm_det_sp(ZM_to_Flm(M, p), p);
    set_avma(av);
    if (!Dp) return gen_0;
    return (Dp <= (p>>1))? utoipos(Dp): utoineg(p - Dp);
  }
  q = gen_1; Dp = 1;
  init_modular_big(&S);
  p = 0; /* -Wall */
  while (cmpii(q, h) <= 0 && (p = u_forprime_next(&S)))
  {
    av2 = avma; Dp = Flm_det_sp(ZM_to_Flm(M, p), p);
    set_avma(av2);
    if (Dp) break;
    q = muliu(q, p);
  }
  if (!p) pari_err_OVERFLOW("ZM_det [ran out of primes]");
  if (!Dp) { set_avma(av); return gen_0; }
  if (mt_nbthreads() > 1 || n <= DIXON_THRESHOLD)
    D = q; /* never competitive when bound is sharp even with 2 threads */
  else
  {
    av2 = avma;
    v = cgetg(n+1, t_COL);
    gel(v, 1) = gen_1; /* ensure content(v) = 1 */
    for (i = 2; i <= n; i++) gel(v, i) = stoi(random_Fl(15) - 7);
    D = Q_denom(ZM_gauss(M, v));
    if (expi(D) < expi(h) >> 1)
    { /* First try unlucky, try once more */
      for (i = 2; i <= n; i++) gel(v, i) = stoi(random_Fl(15) - 7);
      D = lcmii(D, Q_denom(ZM_gauss(M, v)));
    }
    D = gerepileuptoint(av2, D);
    if (q != gen_1) D = lcmii(D, q);
  }
  if (DEBUGLEVEL >=4)
    timer_printf(&ti,"ZM_det: Dixon %ld/%ld bits",expi(D),expi(h));
  /* determinant is a multiple of D */
  if (is_pm1(D)) D = NULL;
  if (D) h = diviiexact(h, D);
  worker = snm_closure(is_entry("_ZM_det_worker"), mkvec(M));
  H = gen_crt("ZM_det", worker, &S, D, expi(h)+1, 0, &mod,
              ZV_chinese, NULL);
  if (D) H = Fp_div(H, D, mod);
  H = Fp_center(H, mod, shifti(mod,-1));
  if (D) H = mulii(H, D);
  return gerepileuptoint(av, H);
}

static GEN
RgM_det_FpM(GEN a, GEN p)
{
  pari_sp av = avma;
  ulong pp, d;
  a = RgM_Fp_init(a,p,&pp);
  switch(pp)
  {
  case 0: return gerepileupto(av, Fp_to_mod(FpM_det(a,p),p)); break;
  case 2: d = F2m_det_sp(a); break;
  default:d = Flm_det_sp(a, pp); break;
  }
  set_avma(av); return mkintmodu(d, pp);
}

static GEN
RgM_det_FqM(GEN x, GEN pol, GEN p)
{
  pari_sp av = avma;
  GEN b, T = RgX_to_FpX(pol, p);
  if (signe(T) == 0) pari_err_OP("%",x,pol);
  b = FqM_det(RgM_to_FqM(x, T, p), T, p);
  if (!b) return gc_NULL(av);
  return gerepilecopy(av, mkpolmod(FpX_to_mod(b, p), FpX_to_mod(T, p)));
}

#define code(t1,t2) ((t1 << 6) | t2)
static GEN
RgM_det_fast(GEN x)
{
  GEN p, pol;
  long pa;
  long t = RgM_type(x, &p,&pol,&pa);
  switch(t)
  {
    case t_INT:    return ZM_det(x);
    case t_FRAC:   return QM_det(x);
    case t_FFELT:  return FFM_det(x, pol);
    case t_INTMOD: return RgM_det_FpM(x, p);
    case code(t_POLMOD, t_INTMOD):
                   return RgM_det_FqM(x, pol, p);
    default:       return NULL;
  }
}
#undef code

static long
det_init_max(long n)
{
  if (n > 100) return 0;
  if (n > 50) return 1;
  if (n > 30) return 4;
  return 7;
}

GEN
det(GEN a)
{
  long n = lg(a)-1;
  double B;
  GEN data, b;
  pivot_fun pivot;

  if (typ(a)!=t_MAT) pari_err_TYPE("det",a);
  if (!n) return gen_1;
  if (n != nbrows(a)) pari_err_DIM("det");
  if (n == 1) return gcopy(gcoeff(a,1,1));
  if (n == 2) return RgM_det2(a);
  b = RgM_det_fast(a);
  if (b) return b;
  pivot = get_pivot_fun(a, a, &data);
  if (pivot != gauss_get_pivot_NZ) return det_simple_gauss(a, data, pivot);
  B = (double)n;
  return det_develop(a, det_init_max(n), B*B*B);
}

GEN
QM_det(GEN M)
{
  pari_sp av = avma;
  GEN cM, pM = Q_primitive_part(M, &cM);
  GEN b = ZM_det(pM);
  if (cM) b = gmul(b, gpowgs(cM, lg(M)-1));
  return gerepileupto(av, b);
}
