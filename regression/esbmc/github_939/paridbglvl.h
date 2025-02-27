/* Copyright (C) 2021  The PARI group.

This file is part of the PARI/GP package.

PARI/GP is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY WHATSOEVER.

Check the License for details. You should have received a copy of it, along
with the package; see the file 'COPYING'. If not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

BEGINEXTERN

#define PARI_DBG_ID(s) DEBUGLEVEL_##s

#define PARI_DBG_LIST(ID) \
  ID(alg), ID(arith), \
  ID(bern), ID(bnf), ID(bnr), ID(bnrclassfield), ID(bb_group), \
  ID(compiler), \
  ID(ell), ID(ellanal), ID(ellcard), ID(ellisogeny), ID(ellrank), \
  ID(ellsea), ID(factcyclo),\
  ID(factor), ID(factorff), ID(factorint), ID(factormod), ID(fflog), \
  ID(galois), ID(gammamellininv), ID(gchar), ID(genus2red), \
  ID(hensel), ID(hgm), ID(hyperell), \
  ID(intnum), ID(io), ID(isprime), \
  ID(lfun), \
  ID(mat), ID(mathnf), ID(mf), ID(mod), ID(mpqs), ID(ms), ID(mt),\
  ID(nf), ID(nffactor), ID(nflist), ID(nfsubfields), \
  ID(padicfields), \
  ID(pol), ID(polclass), ID(polgalois), ID(polmodular), ID(polroots),\
  ID(qf), ID(qflll), ID(qfsolve), ID(qfisom), ID(quadclassunit),\
  ID(rnf), \
  ID(stark), ID(subcyclo), ID(subgrouplist), \
  ID(thue), ID(trans), \
  ID(zetamult)

#ifndef PARI_INIT
extern
#endif
ulong PARI_DBG_LIST(PARI_DBG_ID);
#ifdef PARI_INIT
#define PARI_DBG_PTR(s) &DEBUGLEVEL_##s
#define PARI_DBG_STR(s) #s

ulong * const pari_DEBUGLEVEL_ptr[] = { PARI_DBG_LIST(PARI_DBG_PTR) };

const char * pari_DEBUGLEVEL_str[] = { PARI_DBG_LIST(PARI_DBG_STR) };
#endif

ENDEXTERN
