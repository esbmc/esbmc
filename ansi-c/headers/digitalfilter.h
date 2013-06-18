/**
 * ============================================================================
 * Name        : digitalfilter.h
 * Author      : Renato Abreu
 * Version     : 0.3
 * Copyright   : Copyright by Renato Abreu
 * Description : BMC verification of digital filters
 * ============================================================================
 */

#ifndef _DIGITALFILTER_H
#define _DIGITALFILTER_H	1

#ifdef __cplusplus

extern "C" { 

#endif

/* arithmetic under- and over-flow checking */
int check_filter_overflow(float *a, float *b, int k, int l, int Na, int Nb, float max, float min, int xsize);

/* limit cycle checking */
int check_filter_limitcycle(float *a, float *b, int k, int l, int Na, int Nb, float max, float min, int xsize);

/* timing constraints checking  */
int check_filter_timing(float *a, float *b, int k, int l, int Na, int Nb, float max, float min, int xsize);

#ifdef __cplusplus

}

#endif

#endif	/* digitalfilter.h */
