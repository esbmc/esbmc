/*
 * device_types.h
 *
 *  Created on: Feb 20, 2015
 *      Author: isabela
 */

#ifndef DEVICE_TYPES_H_
#define DEVICE_TYPES_H_

#include "host_defines.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

enum __device_builtin__ cudaRoundMode
{
  cudaRoundNearest,
  cudaRoundZero,
  cudaRoundPosInf,
  cudaRoundMinInf
};

#endif /* DEVICE_TYPES_H_ */
