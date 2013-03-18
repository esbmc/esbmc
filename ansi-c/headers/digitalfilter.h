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

/* overflow checking */
int overflow(float *a, float *b, int k, int l, int Na, int Nb, float max, float min, int xsize);

/* limit cycle checking */
int limitCycle(float *a, float *b, int k, int l, int Na, int Nb, float max, float min, int xsize);

/* timing verification */
int timing(float *a, float *b, int k, int l, int Na, int Nb, float max, float min, int xsize);

#ifdef __cplusplus

}

#endif

#endif	/* digitalfilter.h */
