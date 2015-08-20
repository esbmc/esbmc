#include "common.h"
typedef struct
{
  float real, imag;
}
COMPLEX;
void setup_codec(int), key_down(), int_enable(), int_disable();
int flags(int);
int encode(int, int);
void decode(int, int *);
int filtez(int *bpl, int *dlt);
void upzero(int dlt, int *dlti, int *bli);
int filtep(int rlt1, int al1, int rlt2, int al2);
int quantl(int el, int detl);
int invqxl(int il, int detl, int *code_table, int mode);
int logscl(int il, int nbl);
int scalel(int nbl, int shift_constant);
int uppol2(int al1, int al2, int plt, int plt1, int plt2);
int uppol1(int al1, int apl2, int plt, int plt1);
int invqah(int ih, int deth);
int logsch(int ih, int nbh);
void reset();
int tqmf[24];
int h[24] = {
  12, -44, -44, 212, 48, -624, 128, 1448,
  -840, -3220, 3804, 15504, 15504, 3804, -3220, -840,
  1448, 128, -624, 48, 212, -44, -44, 12
};
int xl, xh;
int accumc[11], accumd[11];
int xs, xd;
int il, szl, spl, sl, el;
int qq4_code4_table[16] = {
  0, -20456, -12896, -8968, -6288, -4240, -2584, -1200,
  20456, 12896, 8968, 6288, 4240, 2584, 1200, 0
};
int qq5_code5_table[32] = {
  -280, -280, -23352, -17560, -14120, -11664, -9752, -8184,
  -6864, -5712, -4696, -3784, -2960, -2208, -1520, -880,
  23352, 17560, 14120, 11664, 9752, 8184, 6864, 5712,
  4696, 3784, 2960, 2208, 1520, 880, 280, -280
};
int qq6_code6_table[64] = {
  -136, -136, -136, -136, -24808, -21904, -19008, -16704,
  -14984, -13512, -12280, -11192, -10232, -9360, -8576, -7856,
  -7192, -6576, -6000, -5456, -4944, -4464, -4008, -3576,
  -3168, -2776, -2400, -2032, -1688, -1360, -1040, -728,
  24808, 21904, 19008, 16704, 14984, 13512, 12280, 11192,
  10232, 9360, 8576, 7856, 7192, 6576, 6000, 5456,
  4944, 4464, 4008, 3576, 3168, 2776, 2400, 2032,
  1688, 1360, 1040, 728, 432, 136, -432, -136
};
int delay_bpl[6];
int delay_dltx[6];
int wl_code_table[16] = {
  -60, 3042, 1198, 538, 334, 172, 58, -30,
  3042, 1198, 538, 334, 172, 58, -30, -60
};
int wl_table[8] = {
  -60, -30, 58, 172, 334, 538, 1198, 3042
};
int ilb_table[32] = {
  2048, 2093, 2139, 2186, 2233, 2282, 2332, 2383,
  2435, 2489, 2543, 2599, 2656, 2714, 2774, 2834,
  2896, 2960, 3025, 3091, 3158, 3228, 3298, 3371,
  3444, 3520, 3597, 3676, 3756, 3838, 3922, 4008
};
int nbl;
int al1, al2;
int plt, plt1, plt2;
int rs;
int dlt;
int rlt, rlt1, rlt2;
int decis_levl[30] = {
  280, 576, 880, 1200, 1520, 1864, 2208, 2584,
  2960, 3376, 3784, 4240, 4696, 5200, 5712, 6288,
  6864, 7520, 8184, 8968, 9752, 10712, 11664, 12896,
  14120, 15840, 17560, 20456, 23352, 32767
};
int detl;
int quant26bt_pos[31] = {
  61, 60, 59, 58, 57, 56, 55, 54,
  53, 52, 51, 50, 49, 48, 47, 46,
  45, 44, 43, 42, 41, 40, 39, 38,
  37, 36, 35, 34, 33, 32, 32
};
int quant26bt_neg[31] = {
  63, 62, 31, 30, 29, 28, 27, 26,
  25, 24, 23, 22, 21, 20, 19, 18,
  17, 16, 15, 14, 13, 12, 11, 10,
  9, 8, 7, 6, 5, 4, 4
};
int deth;
int sh;
int eh;
int qq2_code2_table[4] = {
  -7408, -1616, 7408, 1616
};
int wh_code_table[4] = {
  798, -214, 798, -214
};
int dh, ih;
int nbh, szh;
int sph, ph, yh, rh;
int delay_dhx[6];
int delay_bph[6];
int ah1, ah2;
int ph1, ph2;
int rh1, rh2;
int ilr, yl, rl;
int dec_deth, dec_detl, dec_dlt;
int dec_del_bpl[6];
int dec_del_dltx[6];
int dec_plt, dec_plt1, dec_plt2;
int dec_szl, dec_spl, dec_sl;
int dec_rlt1, dec_rlt2, dec_rlt;
int dec_al1, dec_al2;
int dl;
int dec_nbl, dec_yh, dec_dh, dec_nbh;
int dec_del_bph[6];
int dec_del_dhx[6];
int dec_szh;
int dec_rh1, dec_rh2;
int dec_ah1, dec_ah2;
int dec_ph, dec_sph;
int dec_sh, dec_rh;
int dec_ph1, dec_ph2;
int encode(int xin1, int xin2)
{
  int i;
  int *h_ptr, *tqmf_ptr, *tqmf_ptr1;
  long int xa, xb;
  int decis;
  h_ptr = h;
  tqmf_ptr = tqmf;
  xa = (long) (*tqmf_ptr++) * (*h_ptr++);
  xb = (long) (*tqmf_ptr++) * (*h_ptr++);
  for (i = 0; i < 10; i++)
    {
      xa += (long) (*tqmf_ptr++) * (*h_ptr++);
      xb += (long) (*tqmf_ptr++) * (*h_ptr++);
    }
  xa += (long) (*tqmf_ptr++) * (*h_ptr++);
  xb += (long) (*tqmf_ptr) * (*h_ptr++);
  tqmf_ptr1 = tqmf_ptr - 2;
  for (i = 0; i < 22; i++)
    *tqmf_ptr-- = *tqmf_ptr1--;
  *tqmf_ptr-- = xin1;
  *tqmf_ptr = xin2;
  xl = (xa + xb) >> 15;
  xh = (xa - xb) >> 15;
  szl = filtez(delay_bpl, delay_dltx);
  spl = filtep(rlt1, al1, rlt2, al2);
  sl = szl + spl;
  el = xl - sl;
  il = quantl(el, detl);
  dlt = ((long) detl * qq4_code4_table[il >> 2]) >> 15;
  nbl = logscl(il, nbl);
  detl = scalel(nbl, 8);
  plt = dlt + szl;
  upzero(dlt, delay_dltx, delay_bpl);
  al2 = uppol2(al1, al2, plt, plt1, plt2);
  al1 = uppol1(al1, al2, plt, plt1);
  rlt = sl + dlt;
  rlt2 = rlt1;
  rlt1 = rlt;
  plt2 = plt1;
  plt1 = plt;
  szh = filtez(delay_bph, delay_dhx);
  sph = filtep(rh1, ah1, rh2, ah2);
  sh = sph + szh;
  eh = xh - sh;
  if(eh >= 0)
    {
      ih = 3;
    } 
  else
    {
      ih = 1;
    }
  decis = (564L * (long) deth) >> 12L;
  if(abs(eh) > decis)
    ih--;
  dh = ((long) deth * qq2_code2_table[ih]) >> 15L;
  nbh = logsch(ih, nbh);
  deth = scalel(nbh, 10);
  ph = dh + szh;
  upzero(dh, delay_dhx, delay_bph);
  ah2 = uppol2(ah1, ah2, ph, ph1, ph2);
  ah1 = uppol1(ah1, ah2, ph, ph1);
  yh = sh + dh;
  rh2 = rh1;
  rh1 = yh;
  ph2 = ph1;
  ph1 = ph;
  return (il | (ih << 6));
}
void decode(int input, int *xout)
{
  int i;
  long int xa1, xa2;
  int *h_ptr, *ac_ptr, *ac_ptr1, *ad_ptr, *ad_ptr1;
  ilr = input & 0x3f;
  ih = input >> 6;
  dec_szl = filtez(dec_del_bpl, dec_del_dltx);
  dec_spl = filtep(dec_rlt1, dec_al1, dec_rlt2, dec_al2);
  dec_sl = dec_spl + dec_szl;
  dec_dlt = ((long) dec_detl * qq4_code4_table[ilr >> 2]) >> 15;
  dl = ((long) dec_detl * qq6_code6_table[il]) >> 15;
  rl = dl + dec_sl;
  dec_nbl = logscl(ilr, dec_nbl);
  dec_detl = scalel(dec_nbl, 8);
  dec_plt = dec_dlt + dec_szl;
  upzero(dec_dlt, dec_del_dltx, dec_del_bpl);
  dec_al2 = uppol2(dec_al1, dec_al2, dec_plt, dec_plt1, dec_plt2);
  dec_al1 = uppol1(dec_al1, dec_al2, dec_plt, dec_plt1);
  dec_rlt = dec_sl + dec_dlt;
  dec_rlt2 = dec_rlt1;
  dec_rlt1 = dec_rlt;
  dec_plt2 = dec_plt1;
  dec_plt1 = dec_plt;
  dec_szh = filtez(dec_del_bph, dec_del_dhx);
  dec_sph = filtep(dec_rh1, dec_ah1, dec_rh2, dec_ah2);
  dec_sh = dec_sph + dec_szh;
  dec_dh = ((long) dec_deth * qq2_code2_table[ih]) >> 15L;
  dec_nbh = logsch(ih, dec_nbh);
  dec_deth = scalel(dec_nbh, 10);
  dec_ph = dec_dh + dec_szh;
  upzero(dec_dh, dec_del_dhx, dec_del_bph);
  dec_ah2 = uppol2(dec_ah1, dec_ah2, dec_ph, dec_ph1, dec_ph2);
  dec_ah1 = uppol1(dec_ah1, dec_ah2, dec_ph, dec_ph1);
  rh = dec_sh + dec_dh;
  dec_rh2 = dec_rh1;
  dec_rh1 = rh;
  dec_ph2 = dec_ph1;
  dec_ph1 = dec_ph;
  xd = rl - rh;
  xs = rl + rh;
  h_ptr = h;
  ac_ptr = accumc;
  ad_ptr = accumd;
  xa1 = (long) xd *(*h_ptr++);
  xa2 = (long) xs *(*h_ptr++);
  for (i = 0; i < 10; i++)
    {
      xa1 += (long) (*ac_ptr++) * (*h_ptr++);
      xa2 += (long) (*ad_ptr++) * (*h_ptr++);
    }
  xa1 += (long) (*ac_ptr) * (*h_ptr++);
  xa2 += (long) (*ad_ptr) * (*h_ptr++);
  xout[1] = xa1 >> 14;
  xout[2] = xa2 >> 14;
  ac_ptr1 = ac_ptr - 1;
  ad_ptr1 = ad_ptr - 1;
  for (i = 0; i < 10; i++)
    {
      *ac_ptr-- = *ac_ptr1--;
      *ad_ptr-- = *ad_ptr1--;
    }
  *ac_ptr = xd;
  *ad_ptr = xs;
}
void reset()
{
  int i;
  detl = dec_detl = 32;
  deth = dec_deth = 8;
  nbl = al1 = al2 = plt1 = plt2 = rlt1 = rlt2 = 0;
  nbh = ah1 = ah2 = ph1 = ph2 = rh1 = rh2 = 0;
  dec_nbl = dec_al1 = dec_al2 = dec_plt1 = dec_plt2 = dec_rlt1 = dec_rlt2 = 0;
  dec_nbh = dec_ah1 = dec_ah2 = dec_ph1 = dec_ph2 = dec_rh1 = dec_rh2 = 0;
  for (i = 0; i < 6; i++)
    {
      delay_dltx[i] = 0;
      delay_dhx[i] = 0;
      dec_del_dltx[i] = 0;
      dec_del_dhx[i] = 0;
    }
  for (i = 0; i < 6; i++)
    {
      delay_bpl[i] = 0;
      delay_bph[i] = 0;
      dec_del_bpl[i] = 0;
      dec_del_bph[i] = 0;
    }
  for (i = 0; i < 23; i++)
    tqmf[i] = 0;
  for (i = 0; i < 11; i++)
    {
      accumc[i] = 0;
      accumd[i] = 0;
    }
}
int filtez(int *bpl, int *dlt)
{
  int i;
  long int zl;
  zl = (long) (*bpl++) * (*dlt++);
  for (i = 1; i < 6; i++)
    zl += (long) (*bpl++) * (*dlt++);
  return ((int) (zl >> 14));
}
int filtep(int rlt1, int al1, int rlt2, int al2)
{
  long int pl, pl2;
  pl = 2 * rlt1;
  pl = (long) al1 *pl;
  pl2 = 2 * rlt2;
  pl += (long) al2 *pl2;
  return ((int) (pl >> 15));
}
int quantl(int el, int detl)
{
  int ril, mil;
  long int wd, decis;
  wd = abs(el);
  for (mil = 0; mil < 30; mil++)
    {
      decis = (decis_levl[mil] * (long) detl) >> 15L;
      if(wd <= decis)
	break;
    }
  if(el >= 0)
    ril = quant26bt_pos[mil];
  else
    ril = quant26bt_neg[mil];
  return (ril);
}
int invqxl(int il, int detl, int *code_table, int mode)
{
  long int dlt;
  dlt = (long) detl *code_table[il >> (mode - 1)];
  return ((int) (dlt >> 15));
}
int logscl(int il, int nbl)
{
  long int wd;
  wd = ((long) nbl * 127L) >> 7L;
  nbl = (int) wd + wl_code_table[il >> 2];
  if(nbl < 0)
    nbl = 0;
  if(nbl > 18432)
    nbl = 18432;
  return (nbl);
}
int scalel(int nbl, int shift_constant)
{
  int wd1, wd2, wd3;
  wd1 = (nbl >> 6) & 31;
  wd2 = nbl >> 11;
  wd3 = ilb_table[wd1] >> (shift_constant + 1 - wd2);
  return (wd3 << 3);
}
void upzero(int dlt, int *dlti, int *bli)
{
  int i, wd2, wd3;
  if(dlt == 0)
    {
      for (i = 0; i < 6; i++)
	{
	  bli[i] = (int) ((255L * bli[i]) >> 8L);
	}
    } 
  else
    {
      for (i = 0; i < 6; i++)
	{
	  if((long) dlt * dlti[i] >= 0)
	    wd2 = 128;
	  else
	    wd2 = -128;
	  wd3 = (int) ((255L * bli[i]) >> 8L);
	  bli[i] = wd2 + wd3;
	}
    }
  dlti[5] = dlti[4];
  dlti[4] = dlti[3];
  dlti[3] = dlti[2];
  dlti[1] = dlti[0];
  dlti[0] = dlt;
}
int uppol2(int al1, int al2, int plt, int plt1, int plt2)
{
  long int wd2, wd4;
  int apl2;
  wd2 = 4L * (long) al1;
  if((long) plt * plt1 >= 0L)
    wd2 = -wd2;
  wd2 = wd2 >> 7;
  if((long) plt * plt2 >= 0L)
    {
      wd4 = wd2 + 128;
    } 
  else
    {
      wd4 = wd2 - 128;
    }
  apl2 = wd4 + (127L * (long) al2 >> 7L);
  if(apl2 > 12288)
    apl2 = 12288;
  if(apl2 < -12288)
    apl2 = -12288;
  return (apl2);
}
int uppol1(int al1, int apl2, int plt, int plt1)
{
  long int wd2;
  int wd3, apl1;
  wd2 = ((long) al1 * 255L) >> 8L;
  if((long) plt * plt1 >= 0L)
    {
      apl1 = (int) wd2 + 192;
    } 
  else
    {
      apl1 = (int) wd2 - 192;
    }
  wd3 = 15360 - apl2;
  if(apl1 > wd3)
    apl1 = wd3;
  if(apl1 < -wd3)
    apl1 = -wd3;
  return (apl1);
}
int invqah(int ih, int deth)
{
  long int rdh;
  rdh = ((long) deth * qq2_code2_table[ih]) >> 15L;
  return ((int) (rdh));
}
int logsch(int ih, int nbh)
{
  int wd;
  wd = ((long) nbh * 127L) >> 7L;
  nbh = wd + wh_code_table[ih];
  if(nbh < 0)
    nbh = 0;
  if(nbh > 22528)
    nbh = 22528;
  return (nbh);
}
int main()
{
  int xout[3];
  int i, j, f;
  static int compressed[100], result[100];
  static int test_data[] = {
    0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
    0x44, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x41, 0x41,
    0x41, 0x41, 0x41, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f,
    0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3c, 0x3c, 0x3c, 0x3c,
    0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b,
    0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3c, 0x3c, 0x3c, 0x3c,
    0x3c, 0x3c, 0x3c, 0x3c};
  reset();
  j = 10;
  f = 100;
  for (i = 0; i < 100; i += 2)
    {
      decode(compressed[i / 2], xout);
      result[i] = xout[1];
      result[i + 1] = xout[2];
    }
  if(xout[1] == 11113 && xout[2] == -11197)
    {
      puts("adpcm: success\n");
    } 
  else
    {
      puts("adpcm: failed\n");
    }
  return 0;
}
