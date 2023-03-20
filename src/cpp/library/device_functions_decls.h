/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__DEVICE_FUNCTIONS_DECLS_H__)
#define __DEVICE_FUNCTIONS_DECLS_H__

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate sine of the input argument.
 *
 * Calculate the fast approximate sine of the input argument \p x, measured in radians.
 *
 * \return Returns the approximate sine of \p x.
 *
 * \note_accuracy_single_intrinsic
 * \note Input and output in the denormal range is flushed to sign preserving 0.0.
 */
float __nv_fast_sinf(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate cosine of the input argument.
 *
 * Calculate the fast approximate cosine of the input argument \p x, measured in radians.
 *
 * \return Returns the approximate cosine of \p x.
 *
 * \note_accuracy_single_intrinsic
 * \note Input and output in the denormal range is flushed to sign preserving 0.0.
 */
float __nv_fast_cosf(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate base 2 logarithm of the input argument.
 *
 * Calculate the fast approximate base 2 logarithm of the input argument \p x.
 *
 * \return Returns an approximation to
 * \latexonly $\log_2(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>log</m:mi>
 *     <m:mn>2</m:mn>
 *   </m:msub>
 *   <m:mo>&#x2061;<!-- ⁡ --></m:mo>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 * \note Input and output in the denormal range is flushed to sign preserving 0.0.
 */
float __nv_fast_log2f(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate tangent of the input argument.
 *
 * Calculate the fast approximate tangent of the input argument \p x, measured in radians.
 *
 * \return Returns the approximate tangent of \p x.
 *
 * \note_accuracy_single_intrinsic
 * \note The result is computed as the fast divide of __nv_sinf()
 * by __nv_cosf(). Denormal input and output are flushed to sign-preserving
 * 0.0 at each step of the computation.
 */
float __nv_fast_tanf(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate of sine and cosine of the first input argument.
 *
 * Calculate the fast approximate of sine and cosine of the first input argument \p x (measured
 * in radians). The results for sine and cosine are written into the second
 * argument, \p sptr, and, respectively, third argument, \p zptr.
 *
 * \return
 * - none
 *
 * \note_accuracy_single_intrinsic
 * \note Denorm input/output is flushed to sign preserving 0.0.
 */
void __nv_fast_sincosf(float x, float *sptr, float *cptr);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument.
 *
 * Calculate the fast approximate base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x,
 * \latexonly $e^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return Returns an approximation to
 * \latexonly $e^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 * \note Most input and output values around denormal range are flushed to sign preserving 0.0.
 */
float __nv_fast_expf(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate base 10 exponential of the input argument.
 *
 * Calculate the fast approximate base 10 exponential of the input argument \p x,
 * \latexonly $10^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>10</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return Returns an approximation to
 * \latexonly $10^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>10</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 * \note Most input and output values around denormal range are flushed to sign preserving 0.0.
 */
float __nv_fast_exp10f(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate base 10 logarithm of the input argument.
 *
 * Calculate the fast approximate base 10 logarithm of the input argument \p x.
 *
 * \return Returns an approximation to
 * \latexonly $\log_{10}(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>log</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>10</m:mn>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo>&#x2061;<!-- ⁡ --></m:mo>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 * \note Most input and output values around denormal range are flushed to sign preserving 0.0.
 */
float __nv_fast_log10f(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  logarithm of the input argument.
 *
 * Calculate the fast approximate base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  logarithm of the input argument \p x.
 *
 * \return Returns an approximation to
 * \latexonly $\log_e(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>log</m:mi>
 *     <m:mi>e</m:mi>
 *   </m:msub>
 *   <m:mo>&#x2061;<!-- ⁡ --></m:mo>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 * \note Most input and output values around denormal range are flushed to sign preserving 0.0.
 */
float __nv_fast_logf(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate of
 * \latexonly $x^y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the fast approximate of \p x, the first input argument,
 * raised to the power of \p y, the second input argument,
 * \latexonly $x^y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return Returns an approximation to
 * \latexonly $x^y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 * \note Most input and output values around denormal range are flushed to sign preserving 0.0.
 */
float __nv_fast_powf(float x, float y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Compute average of signed input arguments, avoiding overflow
 * in the intermediate sum.
 *
 * Compute average of signed input arguments \p x and \p y
 * as ( \p x + \p y ) >> 1, avoiding overflow in the intermediate sum.
 *
 * \return Returns a signed integer value representing the signed
 * average value of the two inputs.
 */
int __nv_hadd(int x, int y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Compute rounded average of signed input arguments, avoiding
 * overflow in the intermediate sum.
 *
 * Compute average of signed input arguments \p x and \p y
 * as ( \p x + \p y + 1 ) >> 1, avoiding overflow in the intermediate
 * sum.
 *
 * \return Returns a signed integer value representing the signed
 * rounded average value of the two inputs.
 */
int __nv_rhadd(int x, int y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Compute average of unsigned input arguments, avoiding overflow
 * in the intermediate sum.
 *
 * Compute average of unsigned input arguments \p x and \p y
 * as ( \p x + \p y ) >> 1, avoiding overflow in the intermediate sum.
 *
 * \return Returns an unsigned integer value representing the unsigned
 * average value of the two inputs.
 */
unsigned int __nv_uhadd(unsigned int x, unsigned int y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Compute rounded average of unsigned input arguments, avoiding
 * overflow in the intermediate sum.
 *
 * Compute average of unsigned input arguments \p x and \p y
 * as ( \p x + \p y + 1 ) >> 1, avoiding overflow in the intermediate
 * sum.
 *
 * \return Returns an unsigned integer value representing the unsigned
 * rounded average value of the two inputs.
 */
unsigned int __nv_urhadd(unsigned int x, unsigned int y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Subtract two floating point values in round-to-nearest-even mode.
 *
 * Compute the difference of \p x and \p y in round-to-nearest-even rounding mode.
 *
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fsub_rn(float x, float y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Subtract two floating point values in round-towards-zero mode.
 *
 * Compute the difference of \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fsub_rz(float x, float y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Subtract two floating point values in round-down mode.
 *
 * Compute the difference of \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fsub_rd(float x, float y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Subtract two floating point values in round-up mode.
 *
 * Compute the difference of \p x and \p y in round-up (to positive infinity) mode.
 *
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fsub_ru(float x, float y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-to-nearest-even mode.
 *
 * Compute the reciprocal square root of \p x in round-to-nearest-even mode.
 *
 * \return Returns
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_frsqrt_rn(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Find the position of the least significant bit set to 1 in a 32 bit integer.
 *
 * Find the position of the first (least significant) bit set to 1 in \p x, where the least significant
 * bit position is 1.
 *
 * \return Returns a value between 0 and 32 inclusive representing the position of the first bit set.
 * - __nv_ffs(0) returns 0.
 */
int __nv_ffs(int x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Find the position of the least significant bit set to 1 in a 64 bit integer.
 *
 * Find the position of the first (least significant) bit set to 1 in \p x, where the least significant
 * bit position is 1.
 *
 * \return Returns a value between 0 and 64 inclusive representing the position of the first bit set.
 * - __nv_ffsll(0) returns 0.
 */
int __nv_ffsll(long long int x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded to the nearest even integer value.
 *
 * \return
 * Returns rounded integer value.
 */
float __nv_rintf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round input to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded
 * towards zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return
 * Returns rounded integer value.
 */
long long int __nv_llrintf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round the input argument to the nearest integer.
 *
 * Round argument \p x to an integer value in double precision floating-point format.
 *
 * \return
 * - __nv_nearbyintf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_nearbyintf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_nearbyintf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 *
 * \brief Return the sign bit of the input.
 *
 * Determine whether the floating-point value \p x is negative.
 *
 * \return
 * Returns a nonzero value if and only if \p x is negative.  Reports
 * the sign bit of all values including infinities, zeros, and NaNs.
 */
int __nv_signbitf(float x);

/** \ingroup CUDA_MATH_SINGLE
 * \brief Create value with given magnitude, copying sign of second value.
 *
 * Create a floating-point value with the magnitude \p x and the sign of \p y.
 *
 * \return
 * Returns a value with the magnitude of \p x and the sign of \p y.
 */
float __nv_copysignf(float x, float y);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Determine whether argument is finite
 *
 * Determine whether the floating-point value \p x is a finite value.
 *
 * \return
 * Returns a non-zero value if and only if \p x is a finite value.
 */
int __nv_finitef(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 *
 * \brief Determine whether argument is infinite.
 *
 * Determine whether the floating-point value \p x is an infinite value
 * (positive or negative).
 *
 * \return
 * Returns a nonzero value if and only if \p x is a infinite value.
 */
int __nv_isinff(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 *
 * \brief Determine whether argument is a NaN.
 *
 * Determine whether the floating-point value \p x is a NaN.
 *
 * \return
 * Returns a nonzero value if and only if \p x is a NaN value.
 */
int __nv_isnanf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Return next representable double-precision floating-point value after argument.
 *
 * Calculate the next representable double-precision floating-point value
 * following \p x in the direction of \p y. For example, if \p y is greater than \p x, ::nextafter()
 * returns the smallest representable number greater than \p x
 *
 * \return
 * - __nv_nextafterf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_nextafterf(float x, float y);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Returns "Not a Number" value.
 *
 * Return a representation of a quiet NaN. Argument \p tagp selects one of the possible representations.
 *
 * \return
 * - __nv_nanf(\p tagp) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_nanf(const signed char *tagp);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the sine of the input argument.
 *
 * Calculate the sine of the input argument \p x (measured in radians).
 *
 * \return
 * - __nv_sinf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_sinf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_sinf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the cosine of the input argument.
 *
 * Calculate the cosine of the input argument \p x (measured in radians).
 *
 * \return
 * - __nv_cosf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - __nv_cosf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_cosf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the sine and cosine of the first input argument.
 *
 * Calculate the sine and cosine of the first input argument \p x (measured
 * in radians). The results for sine and cosine are written into the
 * second argument, \p sptr, and, respectively, third argument, \p zptr.
 *
 * \return
 * - none
 *
 * See ::__nv_sinf() and ::__nv_cosf().
 * \note_accuracy_single
 */
void __nv_sincosf(float x, float *sptr, float *cptr);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the sine of the input argument
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians),
 * where \p x is the input argument.
 *
 * \return
 * - __nv_sinpif(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_sinpif(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_sinpif(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the cosine of the input argument
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the cosine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians),
 * where \p x is the input argument.
 *
 * \return
 * - __nv_cospif(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - __nv_cospif(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_cospif(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief  Calculate the sine and cosine of the first input argument
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine and cosine of the first input argument, \p x (measured in radians),
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.  The results for sine and cosine are written into the
 * second argument, \p sptr, and, respectively, third argument, \p zptr.
 *
 * \return
 * - none
 *
 * See ::__nv_sinpif() and ::__nv_cospif().
 * \note_accuracy_single
 */
void __nv_sincospif(float x, float *sptr, float *cptr);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the tangent of the input argument.
 *
 * Calculate the tangent of the input argument \p x (measured in radians).
 *
 * \return
 * - __nv_tanf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_tanf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_tanf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 2 logarithm of the input argument.
 *
 * Calculate the base 2 logarithm of the input argument \p x.
 *
 * \return
 * - __nv_log2f(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_log2f(1) returns +0.
 * - __nv_log2f(\p x) returns NaN for \p x < 0.
 * - __nv_log2f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_log2f(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument.
 *
 * Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x.
 *
 * \return Returns
 * \latexonly $e^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_expf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 10 exponential of the input argument.
 *
 * Calculate the base 10 exponential of the input argument \p x.
 *
 * \return Returns
 * \latexonly $10^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>10</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_exp10f(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the hyperbolic cosine of the input argument.
 *
 * Calculate the hyperbolic cosine of the input argument \p x.
 *
 * \return
 * - __nv_coshf(0) returns 1.
 * - __nv_coshf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_coshf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the hyperbolic sine of the input argument.
 *
 * Calculate the hyperbolic sine of the input argument \p x.
 *
 * \return
 * - __nv_sinhf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_sinhf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the hyperbolic tangent of the input argument.
 *
 * Calculate the hyperbolic tangent of the input argument \p x.
 *
 * \return
 * - __nv_tanhf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_tanhf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc tangent of the ratio of first and second input arguments.
 *
 * Calculate the principal value of the arc tangent of the ratio of first
 * and second input arguments \p x / \p y. The quadrant of the result is
 * determined by the signs of inputs \p x and \p y.
 *
 * \return
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - __nv_atan2f(0, 1) returns +0.
 *
 * \note_accuracy_single
 */
float __nv_atan2f(float x, float y);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc tangent of the input argument.
 *
 * Calculate the principal value of the arc tangent of the input argument \p x.
 *
 * \return
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2].
 * - __nv_atan(0) returns +0.
 *
 * \note_accuracy_single
 */
float __nv_atanf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc sine of the input argument.
 *
 * Calculate the principal value of the arc sine of the input argument \p x.
 *
 * \return
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2] for \p x inside [-1, +1].
 * - __nv_asinf(0) returns +0.
 * - __nv_asinf(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_single
 */
float __nv_asinf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc cosine of the input argument.
 *
 * Calculate the principal value of the arc cosine of the input argument \p x.
 *
 * \return
 * Result will be in radians, in the interval [0,
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ] for \p x inside [-1, +1].
 * - __nv_acosf(1) returns +0.
 * - __nv_acosf(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_single
 */
float __nv_acosf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  logarithm of the input argument.
 *
 * Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  logarithm of the input argument \p x.
 *
 * \return
 * - __nv_logf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_logf(1) returns +0.
 * - __nv_logf(\p x) returns NaN for \p x < 0.
 * - __nv_logf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly

 *
 * \note_accuracy_single
 */
float __nv_logf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 10 logarithm of the input argument.
 *
 * Calculate the base 10 logarithm of the input argument \p x.
 *
 * \return
 * - __nv_log10f(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_log10f(1) returns +0.
 * - __nv_log10f(\p x) returns NaN for \p x < 0.
 * - __nv_log10f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_log10f(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   of the input argument \p x.
 *
 * \return
 * - __nv_log1pf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_log1pf(-1) returns +0.
 * - __nv_log1pf(\p x) returns NaN for \p x < -1.
 * - __nv_log1pf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_log1pf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the nonnegative arc hyperbolic cosine of the input argument.
 *
 * Calculate the nonnegative arc hyperbolic cosine of the input argument \p x.
 *
 * \return
 * Result will be in the interval [0,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - __nv_acoshf(1) returns 0.
 * - __nv_acoshf(\p x) returns NaN for \p x in the interval [
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 1).
 *
 * \note_accuracy_single
 */
float __nv_acoshf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc hyperbolic sine of the input argument.
 *
 * Calculate the arc hyperbolic sine of the input argument \p x.
 *
 * \return
 * - __nv_asinh(0) returns 1.
 *
 * \note_accuracy_single
 */
float __nv_asinhf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc hyperbolic tangent of the input argument.
 *
 * Calculate the arc hyperbolic tangent of the input argument \p x.
 *
 * \return
 * - __nv_atanhf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_atanhf(
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_atanhf(\p x) returns NaN for \p x outside interval [-1, 1].
 *
 * \note_accuracy_single
 */
float __nv_atanhf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument, minus 1.
 *
 * Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x, minus 1.
 *
 * \return Returns
 * \latexonly $e^x - 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_expm1f(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the square root of the sum of squares of two arguments.
 *
 * Calculate the length of the hypotenuse of a right triangle whose two sides have lengths
 * \p x and \p y without undue overflow or underflow.
 *
 * \return Returns the length of the hypotenuse
 * \latexonly $\sqrt{x^2+y^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would overflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_single
 */
float __nv_hypotf(float x, float y);

//FIXME DOCUMENTATION
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the reciprocal square root of the sum of squares of two arguments.
 *
 * Calculate the reciprocal length of the hypotenuse of a right triangle whose two sides have lengths
 * \p x and \p y without undue overflow or underflow.
 *
 * \return Returns the length of the hypotenuse
 * \latexonly $\frac{1}{\sqrt{x^2+y^2}}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>1</m:mi>
 *     <m:mi>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would overflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_double
 */
float __nv_rhypotf(float x, float y);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the cube root of the input argument.
 *
 * Calculate the cube root of \p x,
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_cbrtf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_cbrtf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_cbrtf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate reciprocal cube root function.
 *
 * Calculate reciprocal cube root function of \p x
 *
 * \return
 * - __nv_rcbrtf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_rcbrtf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_rcbrtf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the first kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 0 for
 * the input argument \p x,
 * \latexonly $J_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 0.
 * - __nv_j0f(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_j0f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_j0f(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the first kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 1 for
 * the input argument \p x,
 * \latexonly $J_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 1.
 * - __nv_j1f(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_j1f(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_j1f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_j1f(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the second kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 0 for
 * the input argument \p x,
 * \latexonly $Y_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 0.
 * - __nv_y0f(0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_y0f(\p x) returns NaN for \p x < 0.
 * - __nv_y0f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_y0f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_y0f(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the second kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 1 for
 * the input argument \p x,
 * \latexonly $Y_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 1.
 * - __nv_y1f(0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_y1f(\p x) returns NaN for \p x < 0.
 * - __nv_y1f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_y1f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_y1f(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the second kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order \p n for
 * the input argument \p x,
 * \latexonly $Y_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order \p n.
 * - __nv_ynf(\p n, \p x) returns NaN for \p n < 0.
 * - __nv_ynf(\p n, 0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_ynf(\p n, \p x) returns NaN for \p x < 0.
 * - __nv_ynf(\p n,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_ynf(\p n, NaN) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_ynf(int n, float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the first kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order \p n for
 * the input argument \p x,
 * \latexonly $J_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order \p n.
 * - __nv_jnf(\p n, NaN) returns NaN.
 * - __nv_jnf(\p n, \p x) returns NaN for \p n < 0.
 * - __nv_jnf(\p n,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_single
 */
float __nv_jnf(int n, float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 0 for
 * the input argument \p x,
 * \latexonly $I_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the regular modified cylindrical Bessel function.
 *
 * \note_accuracy_single
 */
float __nv_cyl_bessel_i0f(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 1 for
 * the input argument \p x,
 * \latexonly $I_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the reguler modified cylindrical Bessel function.
 *
 * \note_accuracy_single
 */
float __nv_cyl_bessel_i1f(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the error function of the input argument.
 *
 * Calculate the value of the error function for the input argument \p x,
 * \latexonly $\frac{2}{\sqrt \pi} \int_0^x e^{-t^2} dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>2</m:mn>
 *     <m:msqrt>
 *       <m:mi>&#x03C0;<!-- π --></m:mi>
 *     </m:msqrt>
 *   </m:mfrac>
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *     <m:mn>0</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:msup>
 *         <m:mi>t</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_erff(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_erff(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_erff(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the inverse error function of the input argument.
 *
 * Calculate the inverse error function of the input argument \p y, for \p y in the interval [-1, 1].
 * The inverse error function finds the value \p x that satisfies the equation \p y = erf(\p x),
 * for
 * \latexonly $-1 \le y \le 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_erfinvf(1) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_erfinvf(-1) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_erfinvf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the complementary error function of the input argument.
 *
 * Calculate the complementary error function of the input argument \p x,
 * 1 - erf(\p x).
 *
 * \return
 * - __nv_erfcf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 2.
 * - __nv_erfcf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_single
 */
float __nv_erfcf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the scaled complementary error function of the input argument.
 *
 * Calculate the scaled complementary error function of the input argument \p x,
 * \latexonly $e^{x^2}\cdot \textrm{erfc}(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:msup>
 *         <m:mi>x</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mtext>erfc</m:mtext>
 *   </m:mrow>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_erfcxf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - __nv_erfcxf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0
 * - __nv_erfcxf(\p x) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 *
 * \note_accuracy_single
 */
float __nv_erfcxf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the inverse complementary error function of the input argument.
 *
 * Calculate the inverse complementary error function of the input argument \p y, for \p y in the interval [0, 2].
 * The inverse complementary error function find the value \p x that satisfies the equation \p y = erfc(\p x),
 * for
 * \latexonly $0 \le y \le 2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>0</m:mn>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_erfcinvf(0) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_erfcinvf(2) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_erfcinvf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the inverse of the standard normal cumulative distribution function.
 *
 * Calculate the inverse of the standard normal cumulative distribution function for input argument \p y,
 * \latexonly $\Phi^{-1}(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi mathvariant="normal">&#x03A6;<!-- Φ --></m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly. The function is defined for input values in the interval
 * \latexonly $(0, 1)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>0</m:mn>
 *   <m:mo>,</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_normcdfinvf(0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_normcdfinvf(1) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_normcdfinvf(\p x) returns NaN
 *  if \p x is not in the interval [0,1].
 *
 * \note_accuracy_single
 */
float __nv_normcdfinvf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the standard normal cumulative distribution function.
 *
 * Calculate the cumulative distribution function of the standard normal distribution for input argument \p y,
 * \latexonly $\Phi(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x03A6;<!-- Φ --></m:mi>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_normcdff(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1
 * - __nv_normcdff(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0
 *
 * \note_accuracy_single
 */
float __nv_normcdff(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the natural logarithm of the absolute value of the gamma function of the input argument.
 *
 * Calculate the natural logarithm of the absolute value of the gamma function of the input argument \p x, namely the value of
 * \latexonly $\log_{e}\left|\int_{0}^{\infty} e^{-t}t^{x-1}dt\right|$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>log</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mfenced open="|" close="|">
 *     <m:mrow>
 *       <m:msubsup>
 *         <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mn>0</m:mn>
 *         </m:mrow>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *         </m:mrow>
 *       </m:msubsup>
 *       <m:msup>
 *         <m:mi>e</m:mi>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mo>&#x2212;<!-- − --></m:mo>
 *           <m:mi>t</m:mi>
 *         </m:mrow>
 *       </m:msup>
 *       <m:msup>
 *         <m:mi>t</m:mi>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mi>x</m:mi>
 *           <m:mo>&#x2212;<!-- − --></m:mo>
 *           <m:mn>1</m:mn>
 *         </m:mrow>
 *       </m:msup>
 *       <m:mi>d</m:mi>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:mfenced>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *
 * \return
 * - __nv_lgammaf(1) returns +0.
 * - __nv_lgammaf(2) returns +0.
 * - __nv_lgammaf(\p x) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 * - __nv_lgammaf(\p x) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p x
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly 0.
 * - __nv_lgammaf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_lgammaf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_lgammaf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  of the input arguments \p x and \p exp.
 *
 * \return
 * - __nv_ldexpf(\p x) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 *
 * \note_accuracy_single
 */
float __nv_ldexpf(float x, int y);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Scale floating-point input by integer power of two.
 *
 * Scale \p x by
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  by efficient manipulation of the floating-point
 * exponent.
 *
 * \return
 * Returns \p x *
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_scalbnf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_scalbnf(\p x, 0) returns \p x.
 * - __nv_scalbnf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
float __nv_scalbnf(float x, int y);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Extract mantissa and exponent of a floating-point value
 *
 * Decompose the floating-point value \p x into a component \p m for the
 * normalized fraction element and another term \p n for the exponent.
 * The absolute value of \p m will be greater than or equal to  0.5 and
 * less than 1.0 or it will be equal to 0;
 * \latexonly $x = m\cdot 2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>m</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The integer exponent \p n will be stored in the location to which \p nptr points.
 *
 * \return
 * Returns the fractional component \p m.
 * - __nv_frexpf(0, \p nptr) returns 0 for the fractional component and zero for the integer component.
 * - __nv_frexpf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores zero in the location pointed to by \p nptr.
 * - __nv_frexpf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores an unspecified value in the
 * location to which \p nptr points.
 * - __nv_frexpf(NaN, \p y) returns a NaN and stores an unspecified value in the location to which \p nptr points.
 *
 * \note_accuracy_single
 */
float __nv_frexpf(float x, int *b);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Break down the input argument into fractional and integral parts.
 *
 * Break down the argument \p x into fractional and integral parts. The
 * integral part is stored in the argument \p iptr.
 * Fractional and integral parts are given the same sign as the argument \p x.
 *
 * \return
 * - __nv_modff(
 * \latexonly $\pm x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *  <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi>x</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns a result with the same sign as \p x.
 * - __nv_modff(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   in the object pointed to by \p iptr.
 * - __nv_modff(NaN, \p iptr) stores a NaN in the object pointed to by \p iptr and returns a NaN.
 *
 * \note_accuracy_single
 */
float __nv_modff(float x, float *b);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the floating-point remainder of \p x / \p y.
 *
 * Calculate the floating-point remainder of \p x / \p y.
 * The absolute value of the computed value is always less than \p y's
 * absolute value and will have the same sign as \p x.
 *
 * \return
 * - Returns the floating point remainder of \p x / \p y.
 * - __nv_fmodf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p y is not zero.
 * - __nv_fmodf(\p x, \p y) returns NaN and raised an invalid floating point exception if \p x is
 * \latexonly $\pm\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  or \p y is zero.
 * - __nv_fmodf(\p x, \p y) returns zero if \p y is zero or the result would overflow.
 * - __nv_fmodf(\p x,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x if \p x is finite.
 * - __nv_fmodf(\p x, 0) returns NaN.
 *
 * \note_accuracy_single
 */
float __nv_fmodf(float x, float y);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute double-precision floating-point remainder.
 *
 * Compute double-precision floating-point remainder \p r of dividing
 * \p x by \p y for nonzero \p y. Thus
 * \latexonly $ r = x - n y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>r</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>n</m:mi>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The value \p n is the integer value nearest
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * In the case when
 * \latexonly $ | n -\frac{x}{y} | = \frac{1}{2} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>n</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>=</m:mo>
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mn>2</m:mn>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the
 * even \p n value is chosen.
 *
 * \return
 * - __nv_remainderf(\p x, 0) returns NaN.
 * - __nv_remainderf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns NaN.
 * - __nv_remainderf(\p x,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x for finite \p x.
 *
 * \note_accuracy_single
 */
float __nv_remainderf(float x, float y);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute double-precision floating-point remainder and part of quotient.
 *
 * Compute a double-precision floating-point remainder in the same way as the
 * ::remainder() function. Argument \p quo returns part of quotient upon
 * division of \p x by \p y. Value \p quo has the same sign as
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * and may not be the exact quotient but agrees with the exact quotient
 * in the low order 3 bits.
 *
 * \return
 * Returns the remainder.
 * - __nv_remquof(\p x, 0, \p quo) returns NaN.
 * - __nv_remquof(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y, \p quo) returns NaN.
 * - __nv_remquof(\p x,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p quo) returns \p x.
 *
 * \note_accuracy_single
 */
float __nv_remquof(float x, float y, int *quo);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 *
 * Compute the value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation. After computing the value
 * to infinite precision, the value is rounded once.
 *
 * \return
 * Returns the rounded value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - __nv_fmaf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fmaf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fmaf(\p x, \p y,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_fmaf(\p x, \p y,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_fmaf(float x, float y, float z);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of first argument to the power of second argument.
 *
 * Calculate the value of \p x to the power of \p y.
 *
 * \return
 * - __nv_powif(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an integer less than 0.
 * - __nv_powif(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - __nv_powif(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y > 0 and not and odd integer.
 * - __nv_powif(-1,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - __nv_powif(+1, \p y) returns 1 for any \p y, even a NaN.
 * - __nv_powif(\p x,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1 for any \p x, even a NaN.
 * - __nv_powif(\p x, \p y) returns a NaN for finite \p x < 0 and finite non-integer \p y.
 * - __nv_powif(\p x,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powif(\p x,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powif(\p x,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powif(\p x,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powif(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns -0 for \p y an odd integer less than 0.
 * - __nv_powif(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0 and not an odd integer.
 * - __nv_powif(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - __nv_powif(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0 and not an odd integer.
 * - __nv_powif(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0.
 * - __nv_powif(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0.
 *
 * \note_accuracy_single
 */
float __nv_powif(float x, int y);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of first argument to the power of second argument.
 *
 * Calculate the value of \p x to the power of \p y
 *
 * \return
 * - __nv_powi(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an integer less than 0.
 * - __nv_powi(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - __nv_powi(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y > 0 and not and odd integer.
 * - __nv_powi(-1,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - __nv_powi(+1, \p y) returns 1 for any \p y, even a NaN.
 * - __nv_powi(\p x,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1 for any \p x, even a NaN.
 * - __nv_powi(\p x, \p y) returns a NaN for finite \p x < 0 and finite non-integer \p y.
 * - __nv_powi(\p x,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powi(\p x,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powi(\p x,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powi(\p x,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powi(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns -0 for \p y an odd integer less than 0.
 * - __nv_powi(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0 and not an odd integer.
 * - __nv_powi(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - __nv_powi(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0 and not an odd integer.
 * - __nv_powi(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0.
 * - __nv_powi(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0.
 *
 * \note_accuracy_double
 */
double __nv_powi(double x, int y);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of first argument to the power of second argument.
 *
 * Calculate the value of \p x to the power of \p y
 *
 * \return
 * - __nv_powf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an integer less than 0.
 * - __nv_powf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - __nv_powf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y > 0 and not and odd integer.
 * - __nv_powf(-1,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - __nv_powf(+1, \p y) returns 1 for any \p y, even a NaN.
 * - __nv_powf(\p x,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1 for any \p x, even a NaN.
 * - __nv_powf(\p x, \p y) returns a NaN for finite \p x < 0 and finite non-integer \p y.
 * - __nv_powf(\p x,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powf(\p x,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powf(\p x,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powf(\p x,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns -0 for \p y an odd integer less than 0.
 * - __nv_powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0 and not an odd integer.
 * - __nv_powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - __nv_powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0 and not an odd integer.
 * - __nv_powf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0.
 * - __nv_powf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0.
 *
 * \note_accuracy_single
 */
float __nv_powf(float x, float y);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the gamma function of the input argument.
 *
 * Calculate the gamma function of the input argument \p x, namely the value of
 * \latexonly $\int_{0}^{\infty} e^{-t}t^{x-1}dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>0</m:mn>
 *     </m:mrow>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *     </m:mrow>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:msup>
 *   <m:msup>
 *     <m:mi>t</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>x</m:mi>
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_tgammaf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_tgammaf(2) returns +0.
 * - __nv_tgammaf(\p x) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 * - __nv_tgammaf(\p x) returns NaN if \p x < 0.
 * - __nv_tgammaf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 * - __nv_tgammaf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_tgammaf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded away from zero.
 *
 * \return
 * Returns rounded integer value.
 *
 * \note_slow_round See ::rint().
 */
float __nv_roundf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded
 * away from zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return
 * Returns rounded integer value.
 *
 * \note_slow_round See ::llrint().
 */
long long int __nv_llroundf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute the positive difference between \p x and \p y.
 *
 * Compute the positive difference between \p x and \p y.  The positive
 * difference is \p x - \p y when \p x > \p y and +0 otherwise.
 *
 * \return
 * Returns the positive difference between \p x and \p y.
 * - __nv_fdimf(\p x, \p y) returns \p x - \p y if \p x > \p y.
 * - __nv_fdimf(\p x, \p y) returns +0 if \p x
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly \p y.
 * \note_accuracy_single
 */
float __nv_fdimf(float x, float y);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute the unbiased integer exponent of the argument.
 *
 * Calculates the unbiased integer exponent of the input argument \p x.
 *
 * \return
 * - If successful, returns the unbiased exponent of the argument.
 * - __nv_ilogbf(0) returns <tt>INT_MIN</tt>.
 * - __nv_ilogbf(NaN) returns NaN.
 * - __nv_ilogbf(\p x) returns <tt>INT_MAX</tt> if \p x is
 * \latexonly $\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *     or the correct value is greater than <tt>INT_MAX</tt>.
 * - __nv_ilogbf(\p x) return <tt>INT_MIN</tt> if the correct value is less
 *     than <tt>INT_MIN</tt>.
 *
 * \note_accuracy_single
 */
int __nv_ilogbf(float x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the floating point representation of the exponent of the input argument.
 *
 * Calculate the floating point representation of the exponent of the input argument \p x.
 *
 * \return
 * - __nv_logbf
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - __nv_logbf
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *
 * \note_accuracy_single
 */
float __nv_logbf(float x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded to the nearest even integer value.
 *
 * \return
 * Returns rounded integer value.
 */
double __nv_rint(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round input to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded
 * towards zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return
 * Returns rounded integer value.
 */
long long int __nv_llrint(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round the input argument to the nearest integer.
 *
 * Round argument \p x to an integer value in double precision floating-point format.
 *
 * \return
 * - __nv_nearbyint(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_nearbyint(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_nearbyint(double x);

/**
 * \ingroup CUDA_MATH_SINGLE
 *
 * \brief Return the sign bit of the input.
 *
 * Determine whether the floating-point value \p x is negative.
 *
 * \return
 * Returns a nonzero value if and only if \p x is negative.  Reports
 * the sign bit of all values including infinities, zeros, and NaNs.
 */
int __nv_signbitd(double x);

/**
 * \ingroup CUDA_MATH_SINGLE
 *
 * \brief Determine whether argument is finite.
 *
 * Determine whether the floating-point value \p x is a finite value
 * (zero, subnormal, or normal and not infinity or NaN).
 *
 * \return
 * Returns a nonzero value if and only if \p x is a finite value.
 */
int __nv_isfinited(double x);

/**
 * \ingroup CUDA_MATH_SINGLE
 *
 * \brief Determine whether argument is infinite.
 *
 * Determine whether the floating-point value \p x is an infinite value
 * (positive or negative).
 *
 * \return
 * Returns a nonzero value if and only if \p x is a infinite value.
 */
int __nv_isinfd(double x);

/**
 * \ingroup CUDA_MATH_SINGLE
 *
 * \brief Determine whether argument is a NaN.
 *
 * Determine whether the floating-point value \p x is a NaN.
 *
 * \return
 * Returns a nonzero value if and only if \p x is a NaN value.
 */
int __nv_isnand(double x);

/** \ingroup CUDA_MATH_DOUBLE
 * \brief Create value with given magnitude, copying sign of second value.
 *
 * Create a floating-point value with the magnitude \p x and the sign of \p y.
 *
 * \return
 * Returns a value with the magnitude of \p x and the sign of \p y.
 */
double __nv_copysign(double x, double y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the sine and cosine of the first input argument.
 *
 * Calculate the sine and cosine of the first input argument \p x (measured
 * in radians). The results for sine and cosine are written into the
 * second argument, \p sptr, and, respectively, third argument, \p zptr.
 *
 * \return
 * - none
 *
 * See ::__nv_sin() and ::__nv_cos().
 * \note_accuracy_double
 */
void __nv_sincos(double x, double *sptr, double *cptr);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief  Calculate the sine and cosine of the first input argument
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine and cosine of the first input argument, \p x (measured in radians),
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.  The results for sine and cosine are written into the
 * second argument, \p sptr, and, respectively, third argument, \p zptr.
 *
 * \return
 * - none
 *
 * See ::__nv_sinpi() and ::__nv_cospi().
 * \note_accuracy_double
 */
void __nv_sincospi(double x, double *sptr, double *cptr);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the sine of the input argument.
 *
 * Calculate the sine of the input argument \p x (measured in radians).
 *
 * \return
 * - __nv_sin(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_sin(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_sin(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the cosine of the input argument.
 *
 * Calculate the cosine of the input argument \p x (measured in radians).
 *
 * \return
 * - __nv_cos(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - __nv_cos(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_cos(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the sine of the input argument
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians),
 * where \p x is the input argument.
 *
 * \return
 * - __nv_sinpi(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_sinpi(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_sinpi(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the cosine of the input argument
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the cosine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians),
 * where \p x is the input argument.
 *
 * \return
 * - __nv_cospi(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - __nv_cospi(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_cospi(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the tangent of the input argument.
 *
 * Calculate the tangent of the input argument \p x (measured in radians).
 *
 * \return
 * - __nv_tan(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_tan(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_tan(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  logarithm of the input argument.
 *
 * Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  logarithm of the input argument \p x.
 *
 * \return
 * - __nv_log(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_log(1) returns +0.
 * - __nv_log(\p x) returns NaN for \p x < 0.
 * - __nv_log(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly

 *
 * \note_accuracy_double
 */
double __nv_log(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 2 logarithm of the input argument.
 *
 * Calculate the base 2 logarithm of the input argument \p x.
 *
 * \return
 * - __nv_log2(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_log2(1) returns +0.
 * - __nv_log2(\p x) returns NaN for \p x < 0.
 * - __nv_log2(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_log2(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 10 logarithm of the input argument.
 *
 * Calculate the base 10 logarithm of the input argument \p x.
 *
 * \return
 * - __nv_log10(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_log10(1) returns +0.
 * - __nv_log10(\p x) returns NaN for \p x < 0.
 * - __nv_log10(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_log10(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   of the input argument \p x.
 *
 * \return
 * - __nv_log1p(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_log1p(-1) returns +0.
 * - __nv_log1p(\p x) returns NaN for \p x < -1.
 * - __nv_log1p(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_log1p(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument.
 *
 * Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x.
 *
 * \return Returns
 * \latexonly $e^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_exp(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 2 exponential of the input argument.
 *
 * Calculate the base 2 exponential of the input argument \p x.
 *
 * \return Returns
 * \latexonly $2^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_exp2(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 10 exponential of the input argument.
 *
 * Calculate the base 10 exponential of the input argument \p x.
 *
 * \return Returns
 * \latexonly $10^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>10</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_exp10(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument, minus 1.
 *
 * Calculate the base
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x, minus 1.
 *
 * \return Returns
 * \latexonly $e^x - 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_expm1(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the hyperbolic cosine of the input argument.
 *
 * Calculate the hyperbolic cosine of the input argument \p x.
 *
 * \return
 * - __nv_cosh(0) returns 1.
 * - __nv_cosh(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_cosh(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the hyperbolic sine of the input argument.
 *
 * Calculate the hyperbolic sine of the input argument \p x.
 *
 * \return
 * - __nv_sinh(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_sinh(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the hyperbolic tangent of the input argument.
 *
 * Calculate the hyperbolic tangent of the input argument \p x.
 *
 * \return
 * - __nv_tanh(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_tanh(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc tangent of the ratio of first and second input arguments.
 *
 * Calculate the principal value of the arc tangent of the ratio of first
 * and second input arguments \p x / \p y. The quadrant of the result is
 * determined by the signs of inputs \p x and \p y.
 *
 * \return
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - __nv_atan2(0, 1) returns +0.
 *
 * \note_accuracy_double
 */
double __nv_atan2(double x, double y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc tangent of the input argument.
 *
 * Calculate the principal value of the arc tangent of the input argument \p x.
 *
 * \return
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2].
 * - __nv_atan(0) returns +0.
 *
 * \note_accuracy_double
 */
double __nv_atan(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc sine of the input argument.
 *
 * Calculate the principal value of the arc sine of the input argument \p x.
 *
 * \return
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2] for \p x inside [-1, +1].
 * - __nv_asin(0) returns +0.
 * - __nv_asin(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_double
 */
double __nv_asin(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc cosine of the input argument.
 *
 * Calculate the principal value of the arc cosine of the input argument \p x.
 *
 * \return
 * Result will be in radians, in the interval [0,
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- π --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ] for \p x inside [-1, +1].
 * - __nv_acos(1) returns +0.
 * - __nv_acos(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_double
 */
double __nv_acos(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the nonnegative arc hyperbolic cosine of the input argument.
 *
 * Calculate the nonnegative arc hyperbolic cosine of the input argument \p x.
 *
 * \return
 * Result will be in the interval [0,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - __nv_acosh(1) returns 0.
 * - __nv_acosh(\p x) returns NaN for \p x in the interval [
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 1).
 *
 * \note_accuracy_double
 */
double __nv_acosh(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc hyperbolic sine of the input argument.
 *
 * Calculate the arc hyperbolic sine of the input argument \p x.
 *
 * \return
 * - __nv_asinh(0) returns 1.
 *
 * \note_accuracy_double
 */
double __nv_asinh(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc hyperbolic tangent of the input argument.
 *
 * Calculate the arc hyperbolic tangent of the input argument \p x.
 *
 * \return
 * - __nv_atanh(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_atanh(
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_atanh(\p x) returns NaN for \p x outside interval [-1, 1].
 *
 * \note_accuracy_double
 */
double __nv_atanh(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the sum of squares of two arguments.
 *
 * Calculate the length of the hypotenuse of a right triangle whose two sides have lengths
 * \p x and \p y without undue overflow or underflow.
 *
 * \return Returns the length of the hypotenuse
 * \latexonly $\sqrt{x^2+y^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would overflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_double
 */
double __nv_hypot(double x, double y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the reciprocal square root of the sum of squares of two arguments.
 *
 * Calculate reciprocal of the length of the hypotenuse of a right triangle whose two sides have lengths
 * \p x and \p y without undue overflow or underflow.
 *
 * \return Returns the length of the hypotenuse
 * \latexonly $\sqrt{x^2+y^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>1</m:mi>
 *     <m:mi>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would overflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_double
 */
double __nv_rhypot(double x, double y);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the cube root of the input argument.
 *
 * Calculate the cube root of \p x,
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_cbrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_cbrt(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_cbrt(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate reciprocal cube root function.
 *
 * Calculate reciprocal cube root function of \p x
 *
 * \return
 * - __nv_rcbrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_rcbrt(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_rcbrt(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of first argument to the power of second argument.
 *
 * Calculate the value of \p x to the power of \p y
 *
 * \return
 * - __nv_pow(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an integer less than 0.
 * - __nv_pow(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - __nv_pow(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y > 0 and not and odd integer.
 * - __nv_pow(-1,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - __nv_pow(+1, \p y) returns 1 for any \p y, even a NaN.
 * - __nv_pow(\p x,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1 for any \p x, even a NaN.
 * - __nv_pow(\p x, \p y) returns a NaN for finite \p x < 0 and finite non-integer \p y.
 * - __nv_pow(\p x,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_pow(\p x,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_pow(\p x,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_pow(\p x,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns -0 for \p y an odd integer less than 0.
 * - __nv_pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0 and not an odd integer.
 * - __nv_pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - __nv_pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0 and not an odd integer.
 * - __nv_pow(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0.
 * - __nv_pow(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0.
 *
 * \note_accuracy_double
 */
double __nv_pow(double x, double y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the first kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 0 for
 * the input argument \p x,
 * \latexonly $J_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 0.
 * - __nv_j0(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_j0(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_j0(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the first kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 1 for
 * the input argument \p x,
 * \latexonly $J_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 1.
 * - __nv_j1(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_j1(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_j1(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_j1(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the second kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 0 for
 * the input argument \p x,
 * \latexonly $Y_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 0.
 * - __nv_y0(0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_y0(\p x) returns NaN for \p x < 0.
 * - __nv_y0(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_y0(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_y0(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the second kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 1 for
 * the input argument \p x,
 * \latexonly $Y_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 1.
 * - __nv_y1(0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_y1(\p x) returns NaN for \p x < 0.
 * - __nv_y1(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_y1(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_y1(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the second kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order \p n for
 * the input argument \p x,
 * \latexonly $Y_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order \p n.
 * - __nv_yn(\p n, \p x) returns NaN for \p n < 0.
 * - __nv_yn(\p n, 0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_yn(\p n, \p x) returns NaN for \p x < 0.
 * - __nv_yn(\p n,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_yn(\p n, NaN) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_yn(int n, double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the first kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order \p n for
 * the input argument \p x,
 * \latexonly $J_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order \p n.
 * - __nv_jn(\p n, NaN) returns NaN.
 * - __nv_jn(\p n, \p x) returns NaN for \p n < 0.
 * - __nv_jn(\p n,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_double
 */
double __nv_jn(int n, double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 0 for
 * the input argument \p x,
 * \latexonly $I_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the reguler modified cylindrical Bessel function of order 0.
 *
 * \note_accuracy_double
 */
double __nv_cyl_bessel_i0(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 1 for
 * the input argument \p x,
 * \latexonly $I_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the reguler modified cylindrical Bessel function of order 1.
 *
 * \note_accuracy_double
 */
double __nv_cyl_bessel_i1(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the error function of the input argument.
 *
 * Calculate the value of the error function for the input argument \p x,
 * \latexonly $\frac{2}{\sqrt \pi} \int_0^x e^{-t^2} dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>2</m:mn>
 *     <m:msqrt>
 *       <m:mi>&#x03C0;<!-- π --></m:mi>
 *     </m:msqrt>
 *   </m:mfrac>
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *     <m:mn>0</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:msup>
 *         <m:mi>t</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_erf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_erf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_erf(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the inverse error function of the input argument.
 *
 * Calculate the inverse error function of the input argument \p y, for \p y in the interval [-1, 1].
 * The inverse error function finds the value \p x that satisfies the equation \p y = erf(\p x),
 * for
 * \latexonly $-1 \le y \le 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_erfinv(1) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_erfinv(-1) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_erfinv(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the inverse complementary error function of the input argument.
 *
 * Calculate the inverse complementary error function of the input argument \p y, for \p y in the interval [0, 2].
 * The inverse complementary error function find the value \p x that satisfies the equation \p y = erfc(\p x),
 * for
 * \latexonly $0 \le y \le 2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>0</m:mn>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_erfcinv(0) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_erfcinv(2) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_erfcinv(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the inverse of the standard normal cumulative distribution function.
 *
 * Calculate the inverse of the standard normal cumulative distribution function for input argument \p y,
 * \latexonly $\Phi^{-1}(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi mathvariant="normal">&#x03A6;<!-- Φ --></m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly. The function is defined for input values in the interval
 * \latexonly $(0, 1)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>0</m:mn>
 *   <m:mo>,</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_normcdfinv(0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_normcdfinv(1) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_normcdfinv(\p x) returns NaN
 *  if \p x is not in the interval [0,1].
 *
 * \note_accuracy_double
 */
double __nv_normcdfinv(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the complementary error function of the input argument.
 *
 * Calculate the complementary error function of the input argument \p x,
 * 1 - erf(\p x).
 *
 * \return
 * - __nv_erfc(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 2.
 * - __nv_erfc(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_double
 */
double __nv_erfc(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the scaled complementary error function of the input argument.
 *
 * Calculate the scaled complementary error function of the input argument \p x,
 * \latexonly $e^{x^2}\cdot \textrm{erfc}(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:msup>
 *         <m:mi>x</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mtext>erfc</m:mtext>
 *   </m:mrow>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_erfcx(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - __nv_erfcx(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0
 * - __nv_erfcx(\p x) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 *
 * \note_accuracy_double
 */
double __nv_erfcx(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the standard normal cumulative distribution function.
 *
 * Calculate the cumulative distribution function of the standard normal distribution for input argument \p y,
 * \latexonly $\Phi(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x03A6;<!-- Φ --></m:mi>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_normcdf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1
 * - __nv_normcdf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0
 *
 * \note_accuracy_double
 */
double __nv_normcdf(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the gamma function of the input argument.
 *
 * Calculate the gamma function of the input argument \p x, namely the value of
 * \latexonly $\int_{0}^{\infty} e^{-t}t^{x-1}dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>0</m:mn>
 *     </m:mrow>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *     </m:mrow>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:msup>
 *   <m:msup>
 *     <m:mi>t</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>x</m:mi>
 *       <m:mo>&#x2212;<!-- − --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * - __nv_tgamma(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_tgamma(2) returns +0.
 * - __nv_tgamma(\p x) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 * - __nv_tgamma(\p x) returns NaN if \p x < 0.
 * - __nv_tgamma(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 * - __nv_tgamma(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_tgamma(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the natural logarithm of the absolute value of the gamma function of the input argument.
 *
 * Calculate the natural logarithm of the absolute value of the gamma function of the input argument \p x, namely the value of
 * \latexonly $\log_{e}\left|\int_{0}^{\infty} e^{-t}t^{x-1}dt\right|$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>log</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mfenced open="|" close="|">
 *     <m:mrow>
 *       <m:msubsup>
 *         <m:mo>&#x222B;<!-- ∫ --></m:mo>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mn>0</m:mn>
 *         </m:mrow>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 *         </m:mrow>
 *       </m:msubsup>
 *       <m:msup>
 *         <m:mi>e</m:mi>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mo>&#x2212;<!-- − --></m:mo>
 *           <m:mi>t</m:mi>
 *         </m:mrow>
 *       </m:msup>
 *       <m:msup>
 *         <m:mi>t</m:mi>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mi>x</m:mi>
 *           <m:mo>&#x2212;<!-- − --></m:mo>
 *           <m:mn>1</m:mn>
 *         </m:mrow>
 *       </m:msup>
 *       <m:mi>d</m:mi>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:mfenced>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *
 * \return
 * - __nv_lgamma(1) returns +0.
 * - __nv_lgamma(2) returns +0.
 * - __nv_lgamma(\p x) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 * - __nv_lgamma(\p x) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p x
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly 0.
 * - __nv_lgamma(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_lgamma(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_lgamma(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  of the input arguments \p x and \p exp.
 *
 * \return
 * - __nv_ldexp(\p x) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating point range.
 *
 * \note_accuracy_double
 */
double __nv_ldexp(double x, int y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Scale floating-point input by integer power of two.
 *
 * Scale \p x by
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  by efficient manipulation of the floating-point
 * exponent.
 *
 * \return
 * Returns \p x *
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_scalbn(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_scalbn(\p x, 0) returns \p x.
 * - __nv_scalbn(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
double __nv_scalbn(double x, int y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Extract mantissa and exponent of a floating-point value
 *
 * Decompose the floating-point value \p x into a component \p m for the
 * normalized fraction element and another term \p n for the exponent.
 * The absolute value of \p m will be greater than or equal to  0.5 and
 * less than 1.0 or it will be equal to 0;
 * \latexonly $x = m\cdot 2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>m</m:mi>
 *   <m:mo>&#x22C5;<!-- ⋅ --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The integer exponent \p n will be stored in the location to which \p nptr points.
 *
 * \return
 * Returns the fractional component \p m.
 * - __nv_frexp(0, \p nptr) returns 0 for the fractional component and zero for the integer component.
 * - __nv_frexp(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores zero in the location pointed to by \p nptr.
 * - __nv_frexp(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores an unspecified value in the
 * location to which \p nptr points.
 * - __nv_frexp(NaN, \p y) returns a NaN and stores an unspecified value in the location to which \p nptr points.
 *
 * \note_accuracy_double
 */
double __nv_frexp(double x, int *b);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Break down the input argument into fractional and integral parts.
 *
 * Break down the argument \p x into fractional and integral parts. The
 * integral part is stored in the argument \p iptr.
 * Fractional and integral parts are given the same sign as the argument \p x.
 *
 * \return
 * - __nv_modf(
 * \latexonly $\pm x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *  <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi>x</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns a result with the same sign as \p x.
 * - __nv_modf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   in the object pointed to by \p iptr.
 * - __nv_modf(NaN, \p iptr) stores a NaN in the object pointed to by \p iptr and returns a NaN.
 *
 * \note_accuracy_double
 */
double __nv_modf(double x, double *b);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the floating-point remainder of \p x / \p y.
 *
 * Calculate the floating-point remainder of \p x / \p y.
 * The absolute value of the computed value is always less than \p y's
 * absolute value and will have the same sign as \p x.
 *
 * \return
 * - Returns the floating point remainder of \p x / \p y.
 * - __nv_fmod(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p y is not zero.
 * - __nv_fmod(\p x, \p y) returns NaN and raised an invalid floating point exception if \p x is
 * \latexonly $\pm\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  or \p y is zero.
 * - __nv_fmod(\p x, \p y) returns zero if \p y is zero or the result would overflow.
 * - __nv_fmod(\p x,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x if \p x is finite.
 * - __nv_fmod(\p x, 0) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_fmod(double x, double y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute double-precision floating-point remainder.
 *
 * Compute double-precision floating-point remainder \p r of dividing
 * \p x by \p y for nonzero \p y. Thus
 * \latexonly $ r = x - n y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>r</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>n</m:mi>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The value \p n is the integer value nearest
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * In the case when
 * \latexonly $ | n -\frac{x}{y} | = \frac{1}{2} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>n</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>=</m:mo>
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mn>2</m:mn>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the
 * even \p n value is chosen.
 *
 * \return
 * - __nv_remainder(\p x, 0) returns NaN.
 * - __nv_remainder(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns NaN.
 * - __nv_remainder(\p x,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x for finite \p x.
 *
 * \note_accuracy_double
 */
double __nv_remainder(double x, double y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute double-precision floating-point remainder and part of quotient.
 *
 * Compute a double-precision floating-point remainder in the same way as the
 * ::remainder() function. Argument \p quo returns part of quotient upon
 * division of \p x by \p y. Value \p quo has the same sign as
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * and may not be the exact quotient but agrees with the exact quotient
 * in the low order 3 bits.
 *
 * \return
 * Returns the remainder.
 * - __nv_remquo(\p x, 0, \p quo) returns NaN.
 * - __nv_remquo(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y, \p quo) returns NaN.
 * - __nv_remquo(\p x,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p quo) returns \p x.
 *
 * \note_accuracy_double
 */
double __nv_remquo(double x, double y, int *c);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Return next representable double-precision floating-point value after argument.
 *
 * Calculate the next representable double-precision floating-point value
 * following \p x in the direction of \p y. For example, if \p y is greater than \p x, ::nextafter()
 * returns the smallest representable number greater than \p x
 *
 * \return
 * - __nv_nextafter(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_nextafter(double x, double y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Returns "Not a Number" value.
 *
 * Return a representation of a quiet NaN. Argument \p tagp selects one of the possible representations.
 *
 * \return
 * - __nv_nan(\p tagp) returns NaN.
 *
 * \note_accuracy_double
 */
double __nv_nan(const signed char *tagp);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded away from zero.
 *
 * \return
 * Returns rounded integer value.
 *
 * \note_slow_round See ::rint().
 */
double __nv_round(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded
 * away from zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return
 * Returns rounded integer value.
 *
 * \note_slow_round See ::llrint().
 */
long long int __nv_llround(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute the positive difference between \p x and \p y.
 *
 * Compute the positive difference between \p x and \p y.  The positive
 * difference is \p x - \p y when \p x > \p y and +0 otherwise.
 *
 * \return
 * Returns the positive difference between \p x and \p y.
 * - __nv_fdim(\p x, \p y) returns \p x - \p y if \p x > \p y.
 * - __nv_fdim(\p x, \p y) returns +0 if \p x
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly \p y.
 * \note_accuracy_single
 */
double __nv_fdim(double x, double y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute the unbiased integer exponent of the argument.
 *
 * Calculates the unbiased integer exponent of the input argument \p x.
 *
 * \return
 * - If successful, returns the unbiased exponent of the argument.
 * - __nv_ilogb(0) returns <tt>INT_MIN</tt>.
 * - __nv_ilogb(NaN) returns NaN.
 * - __nv_ilogb(\p x) returns <tt>INT_MAX</tt> if \p x is
 * \latexonly $\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *     or the correct value is greater than <tt>INT_MAX</tt>.
 * - __nv_ilogb(\p x) return <tt>INT_MIN</tt> if the correct value is less
 *     than <tt>INT_MIN</tt>.
 *
 * \note_accuracy_double
 */
int __nv_ilogb(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the floating point representation of the exponent of the input argument.
 *
 * Calculate the floating point representation of the exponent of the input argument \p x.
 *
 * \return
 * - __nv_logb
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - __nv_logb
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *
 * \note_accuracy_double
 */
double __nv_logb(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 *
 * Compute the value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation. After computing the value
 * to infinite precision, the value is rounded once.
 *
 * \return
 * Returns the rounded value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - __nv_fma(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fma(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fma(\p x, \p y,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_fma(\p x, \p y,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_fma(double x, double y, double z);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Return the number of consecutive high-order zero bits in a 32 bit integer.
 *
 * Count the number of consecutive leading zero bits, starting at the most significant bit (bit 31) of \p x.
 *
 * \return Returns a value between 0 and 32 inclusive representing the number of zero bits.
 */
int __nv_clz(int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Count the number of consecutive high-order zero bits in a 64 bit integer.
 *
 * Count the number of consecutive leading zero bits, starting at the most significant bit (bit 63) of \p x.
 *
 * \return Returns a value between 0 and 64 inclusive representing the number of zero bits.
 */
int __nv_clzll(long long x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Count the number of bits that are set to 1 in a 32 bit integer.
 *
 * Count the number of bits that are set to 1 in \p x.
 *
 * \return Returns a value between 0 and 32 inclusive representing the number of set bits.
 */
int __nv_popc(int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Count the number of bits that are set to 1 in a 64 bit integer.
 *
 * Count the number of bits that are set to 1 in \p x.
 *
 * \return Returns a value between 0 and 64 inclusive representing the number of set bits.
 */
int __nv_popcll(long long x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Return selected bytes from two 32 bit unsigned integers.
 *
 * __nv_byte_perm(x,y,s) returns a 32-bit integer consisting of four bytes from eight input bytes provided in the two
 * input integers \p x and \p y, as specified by a selector, \p s.
 *
 * The input bytes are indexed as follows:
 * <pre>
 * input[0] = x<7:0>   input[1] = x<15:8>
 * input[2] = x<23:16> input[3] = x<31:24>
 * input[4] = y<7:0>   input[5] = y<15:8>
 * input[6] = y<23:16> input[7] = y<31:24>
 * </pre>
 * The selector indices are as follows (the upper 16-bits of the selector are not used):
 * <pre>
 * selector[0] = s<2:0>  selector[1] = s<6:4>
 * selector[2] = s<10:8> selector[3] = s<14:12>
 * </pre>
 * \return The returned value r is computed to be:
 * <tt>result[n] := input[selector[n]]</tt>
 * where <tt>result[n]</tt> is the nth byte of r.
 */
unsigned int __nv_byte_perm(unsigned int x, unsigned int y, unsigned int z);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Determine the minimum value of two 32-bit signed integers.
 *
 * Determine the minimum value of the two 32-bit signed integers \p x and \p y.
 *
 * \return
 * Returns the minimum value of the two 32-bit signed integers \p x and \p y.
 */
int __nv_min(int x, int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Determine the minimum value of two 32-bit unsigned integers.
 *
 * Determine the minimum value of the two 32-bit unsigned integers \p x and \p y.
 *
 * \return
 * Returns the minimum value of the two 32-bit unsigned integers \p x and \p y.
 */
unsigned int __nv_umin(unsigned int x, unsigned int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Determine the minimum value of two 64-bit signed integers.
 *
 * Determine the minimum value of the two 64-bit signed integers \p x and \p y.
 *
 * \return
 * Returns the minimum value of the two 64-bit signed integers \p x and \p y.
 */
long long __nv_llmin(long long x, long long y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Determine the minimum value of two 64-bit unsigned integers.
 *
 * Determine the minimum value of the two 64-bit unsigned integers \p x and \p y.
 *
 * \return
 * Returns the minimum value of the two 64-bit unsigned integers \p x and \p y.
 */
unsigned long long __nv_ullmin(unsigned long long x, unsigned long long y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Determine the maximum value of two 32-bit signed integers.
 *
 * Determine the maximum value of the two 32-bit signed integers \p x and \p y.
 *
 * \return
 * Returns the maximum value of the two 32-bit signed integers \p x and \p y.
 */
int __nv_max(int x, int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Determine the maximum value of two 32-bit unsigned integers.
 *
 * Determine the maximum value of the two 32-bit unsigned integers \p x and \p y.
 *
 * \return
 * Returns the maximum value of the two 32-bit unsigned integers \p x and \p y.
 */
unsigned int __nv_umax(unsigned int x, unsigned int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Determine the maximum value of two 64-bit signed integers.
 *
 * Determine the maximum value of the two 64-bit signed integers \p x and \p y.
 *
 * \return
 * Returns the maximum value of the two 64-bit signed integers \p x and \p y.
 */
long long __nv_llmax(long long x, long long y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Determine the maximum value of two 64-bit unsigned integers.
 *
 * Determine the maximum value of the two 64-bit unsigned integers \p x and \p y.
 *
 * \return
 * Returns the maximum value of the two 64-bit unsigned integers \p x and \p y.
 */
unsigned long long __nv_ullmax(unsigned long long x, unsigned long long y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 32 bits of the product of the two 32 bit integers.
 *
 * Calculate the most significant 32 bits of the 64-bit product \p x * \p y, where \p x and \p y
 * are 32-bit integers.
 *
 * \return Returns the most significant 32 bits of the product \p x * \p y.
 */
int __nv_mulhi(int x, int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 32 bits of the product of the two 32 bit unsigned integers.
 *
 * Calculate the most significant 32 bits of the 64-bit product \p x * \p y, where \p x and \p y
 * are 32-bit unsigned integers.
 *
 * \return Returns the most significant 32 bits of the product \p x * \p y.
 */
unsigned int __nv_umulhi(unsigned int x, unsigned int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 64 bits of the product of the two 64 bit integers.
 *
 * Calculate the most significant 64 bits of the 128-bit product \p x * \p y, where \p x and \p y
 * are 64-bit integers.
 *
 * \return Returns the most significant 64 bits of the product \p x * \p y.
 */
long long __nv_mul64hi(long long x, long long y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 64 bits of the product of the two 64 unsigned bit integers.
 *
 * Calculate the most significant 64 bits of the 128-bit product \p x * \p y, where \p x and \p y
 * are 64-bit unsigned integers.
 *
 * \return Returns the most significant 64 bits of the product \p x * \p y.
 */
unsigned long long __nv_umul64hi(unsigned long long x, unsigned long long y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the least significant 32 bits of the product of the least significant 24 bits of two integers.
 *
 * Calculate the least significant 32 bits of the product of the least significant 24 bits of \p x and \p y.
 * The high order 8 bits of \p x and \p y are ignored.
 *
 * \return Returns the least significant 32 bits of the product \p x * \p y.
 */
int __nv_mul24(int x, int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the least significant 32 bits of the product of the least significant 24 bits of two unsigned integers.
 *
 * Calculate the least significant 32 bits of the product of the least significant 24 bits of \p x and \p y.
 * The high order 8 bits of  \p x and  \p y are ignored.
 *
 * \return Returns the least significant 32 bits of the product \p x * \p y.
 */
unsigned int __nv_umul24(unsigned int x, unsigned int y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Reverse the bit order of a 32 bit unsigned integer.
 *
 * Reverses the bit order of the 32 bit unsigned integer \p x.
 *
 * \return Returns the bit-reversed value of \p x. i.e. bit N of the return value corresponds to bit 31-N of \p x.
 */
unsigned int __nv_brev(unsigned int x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Reverse the bit order of a 64 bit unsigned integer.
 *
 * Reverses the bit order of the 64 bit unsigned integer \p x.
 *
 * \return Returns the bit-reversed value of \p x. i.e. bit N of the return value corresponds to bit 63-N of \p x.
 */
unsigned long long __nv_brevll(unsigned long long x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the sum of absolute difference.
 *
 * Calculate
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the 32-bit sum of the third argument \p z plus and the absolute
 * value of the difference between the first argument, \p x, and second
 * argument, \p y.
 *
 * Inputs \p x and \p y are signed 32-bit integers, input \p z is
 * a 32-bit unsigned integer.
 *
 * \return Returns
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
int __nv_sad(int x, int y, int z);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the sum of absolute difference.
 *
 * Calculate
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the 32-bit sum of the third argument \p z plus and the absolute
 * value of the difference between the first argument, \p x, and second
 * argument, \p y.
 *
 * Inputs \p x, \p y, and \p z are unsigned 32-bit integers.
 *
 * \return Returns
 * \latexonly $|x - y| + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
unsigned int __nv_usad(unsigned int x, unsigned int y, unsigned int z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Determine the absolute value of a 32-bit signed integer.
 *
 * Determine the absolute value of the 32-bit signed integer \p x.
 *
 * \return
 * Returns the absolute value of the 32-bit signed integer \p x.
 */
int __nv_abs(int x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Determine the absolute value of a 64-bit signed integer.
 *
 * Determine the absolute value of the 64-bit signed integer \p x.
 *
 * \return
 * Returns the absolute value of the 64-bit signed integer \p x.
 */
long long __nv_llabs(long long x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the largest integer less than or equal to \p x.
 *
 * Calculates the largest integer value which is less than or equal to \p x.
 *
 * \return
 * Returns
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  expressed as a floating-point number.
 * - __nv_floorf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_floorf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
float __nv_floorf(float f);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the largest integer less than or equal to \p x.
 *
 * Calculates the largest integer value which is less than or equal to \p x.
 *
 * \return
 * Returns
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  expressed as a floating-point number.
 * - __nv_floor(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_floor(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
double __nv_floor(double f);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the absolute value of the input argument.
 *
 * Calculate the absolute value of the input argument \p x.
 *
 * \return
 * Returns the absolute value of the input argument.
 * - __nv_fabsf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_fabsf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 0.
 * \note_accuracy_double
 */
float __nv_fabsf(float f);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the absolute value of the input argument.
 *
 * Calculate the absolute value of the input argument \p x.
 *
 * \return
 * Returns the absolute value of the input argument.
 * - __nv_fabs(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_fabs(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 0.
 * \note_accuracy_double
 */
double __nv_fabs(double f);

/** DOCUMENTATION MISSING */
double __nv_rcp64h(double d);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Determine the minimum numeric value of the arguments.
 *
 * Determines the minimum numeric value of the arguments \p x and \p y. Treats NaN
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the minimum numeric values of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_double
 */
float __nv_fminf(float x, float y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Determine the maximum numeric value of the arguments.
 *
 * Determines the maximum numeric value of the arguments \p x and \p y. Treats NaN
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the maximum numeric values of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_double
 */
float __nv_fmaxf(float x, float y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the reciprocal of the square root of the input argument.
 *
 * Calculate the reciprocal of the nonnegative square root of \p x,
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_rsqrtf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_rsqrtf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_rsqrtf(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_double
 */
float __nv_rsqrtf(float x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Determine the minimum numeric value of the arguments.
 *
 * Determines the minimum numeric value of the arguments \p x and \p y. Treats NaN
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the minimum numeric values of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_double
 */
double __nv_fmin(double x, double y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Determine the maximum numeric value of the arguments.
 *
 * Determines the maximum numeric value of the arguments \p x and \p y. Treats NaN
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the maximum numeric values of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_double
 */
double __nv_fmax(double x, double y);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the reciprocal of the square root of the input argument.
 *
 * Calculate the reciprocal of the nonnegative square root of \p x,
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_rsqrt(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - __nv_rsqrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_rsqrt(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_double
 */
double __nv_rsqrt(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate ceiling of the input argument.
 *
 * Compute the smallest integer value not less than \p x.
 *
 * \return
 * Returns
 * \latexonly $\lceil x \rceil$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo fence="false" stretchy="false">&#x2308;<!-- ⌈ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo fence="false" stretchy="false">&#x2309;<!-- ⌉ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 expressed as a floating-point number.
 * - __nv_ceil(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_ceil(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
double __nv_ceil(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Truncate input argument to the integral part.
 *
 * Round \p x to the nearest integer value that does not exceed \p x in
 * magnitude.
 *
 * \return
 * Returns truncated integer value.
 */
double __nv_trunc(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 2 exponential of the input argument.
 *
 * Calculate the base 2 exponential of the input argument \p x.
 *
 * \return Returns
 * \latexonly $2^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
float __nv_exp2f(float x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Truncate input argument to the integral part.
 *
 * Round \p x to the nearest integer value that does not exceed \p x in
 * magnitude.
 *
 * \return
 * Returns truncated integer value.
 */
float __nv_truncf(float x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate ceiling of the input argument.
 *
 * Compute the smallest integer value not less than \p x.
 *
 * \return
 * Returns
 * \latexonly $\lceil x \rceil$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo fence="false" stretchy="false">&#x2308;<!-- ⌈ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo fence="false" stretchy="false">&#x2309;<!-- ⌉ --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 expressed as a floating-point number.
 * - __nv_ceilf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_ceilf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
float __nv_ceilf(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Clamp the input argument to [+0.0, 1.0].
 *
 * Clamp the input argument \p x to be within the interval [+0.0, 1.0].
 * \return
 * - __nv_saturatef(\p x) returns 0 if \p x < 0.
 * - __nv_saturatef(\p x) returns 1 if \p x > 1.
 * - __nv_saturatef(\p x) returns \p x if
 * \latexonly $0 \le x \le 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>0</m:mn>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- ≤ --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_saturatef(NaN) returns 0.
 */
float __nv_saturatef(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation, in round-to-nearest-even mode.
 *
 * Computes the value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-to-nearest-even mode.
 *
 * \return Returns the rounded value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - __nv_fmaf_rn(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fmaf_rn(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fmaf_rn(\p x, \p y,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_fmaf_rn(\p x, \p y,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_fmaf_rn(float x, float y, float z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation, in round-towards-zero mode.
 *
 * Computes the value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-towards-zero mode.
 *
 * \return Returns the rounded value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - __nv_fmaf_rz(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fmaf_rz(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fmaf_rz(\p x, \p y,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_fmaf_rz(\p x, \p y,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_fmaf_rz(float x, float y, float z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation, in round-down mode.
 *
 * Computes the value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-down (to negative infinity) mode.
 *
 * \return Returns the rounded value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - __nv_fmaf_rd(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fmaf_rd(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fmaf_rd(\p x, \p y,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_fmaf_rd(\p x, \p y,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_fmaf_rd(float x, float y, float z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation, in round-up mode.
 *
 * Computes the value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-up (to positive infinity) mode.
 *
 * \return Returns the rounded value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - __nv_fmaf_ru(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fmaf_ru(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fmaf_ru(\p x, \p y,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_fmaf_ru(\p x, \p y,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_fmaf_ru(float x, float y, float z);

/** DOCUMENTATION MISSING */
float __nv_fmaf_ieee_rn(float x, float y, float z);
/** DOCUMENTATION MISSING */
float __nv_fmaf_ieee_rz(float x, float y, float z);
/** DOCUMENTATION MISSING */
float __nv_fmaf_ieee_rd(float x, float y, float z);
/** DOCUMENTATION MISSING */
float __nv_fmaf_ieee_ru(float x, float y, float z);

/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation in round-to-nearest-even mode.
 *
 * Computes the value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-to-nearest-even mode.
 *
 * \return Returns the rounded value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - __nv_fma_rn(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fma_rn(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fma_rn(\p x, \p y,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 * - __nv_fma_rn(\p x, \p y,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 *
 * \note_accuracy_double
 */
double __nv_fma_rn(double x, double y, double z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation in round-towards-zero mode.
 *
 * Computes the value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-towards-zero mode.
 *
 * \return Returns the rounded value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - __nv_fma_rz(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fma_rz(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fma_rz(\p x, \p y,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 * - __nv_fma_rz(\p x, \p y,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 *
 * \note_accuracy_double
 */
double __nv_fma_rz(double x, double y, double z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation in round-down mode.
 *
 * Computes the value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-down (to negative infinity) mode.
 *
 * \return Returns the rounded value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - __nv_fma_rd(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fma_rd(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fma_rd(\p x, \p y,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 * - __nv_fma_rd(\p x, \p y,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 *
 * \note_accuracy_double
 */
double __nv_fma_rd(double x, double y, double z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation in round-up mode.
 *
 * Computes the value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-up (to positive infinity) mode.
 *
 * \return Returns the rounded value of
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - __nv_fma_ru(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fma_ru(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ,
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - __nv_fma_ru(\p x, \p y,
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 * - __nv_fma_ru(\p x, \p y,
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- × --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- − --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 *
 * \note_accuracy_double
 */
double __nv_fma_ru(double x, double y, double z);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate division of the input arguments.
 *
 * Calculate the fast approximate division of \p x by \p y.
 *
 * \return Returns \p x / \p y.
 * - __nv_fast_fdividef(
 * \latexonly $\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns NaN for
 * \latexonly $2^{126} < y < 2^{128}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>126</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo>&lt;</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&lt;</m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>128</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_fast_fdividef(\p x, \p y) returns 0 for
 * \latexonly $2^{126} < y < 2^{128}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>126</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo>&lt;</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&lt;</m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>128</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and
 * \latexonly $x \ne \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2260;<!-- ≠ --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single_intrinsic
 */
float __nv_fast_fdividef(float x, float y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Divide two floating point values in round-to-nearest-even mode.
 *
 * Divide two floating point values \p x by \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_single
 */
float __nv_fdiv_rn(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Divide two floating point values in round-towards-zero mode.
 *
 * Divide two floating point values \p x by \p y in round-towards-zero mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_single
 */
float __nv_fdiv_rz(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Divide two floating point values in round-down mode.
 *
 * Divide two floating point values \p x by \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_single
 */
float __nv_fdiv_rd(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Divide two floating point values in round-up mode.
 *
 * Divide two floating point values \p x by \p y in round-up (to positive infinity) mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_single
 */
float __nv_fdiv_ru(float x, float y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-to-nearest-even mode.
 *
 * Compute the reciprocal of \p x in round-to-nearest-even mode.
 *
 * \return Returns
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_frcp_rn(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-towards-zero mode.
 *
 * Compute the reciprocal of \p x in round-towards-zero mode.
 *
 * \return Returns
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_frcp_rz(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-down mode.
 *
 * Compute the reciprocal of \p x in round-down (to negative infinity) mode.
 *
 * \return Returns
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_frcp_rd(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-up mode.
 *
 * Compute the reciprocal of \p x in round-up (to positive infinity) mode.
 *
 * \return Returns
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_frcp_ru(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-to-nearest-even mode.
 *
 * Compute the square root of \p x in round-to-nearest-even mode.
 *
 * \return Returns
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_fsqrt_rn(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-towards-zero mode.
 *
 * Compute the square root of \p x in round-towards-zero mode.
 *
 * \return Returns
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_fsqrt_rz(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-down mode.
 *
 * Compute the square root of \p x in round-down (to negative infinity) mode.
 *
 * \return Returns
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_fsqrt_rd(float x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-up mode.
 *
 * Compute the square root of \p x in round-up (to positive infinity) mode.
 *
 * \return Returns
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
float __nv_fsqrt_ru(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating point values in round-to-nearest-even mode.
 *
 * Divides two floating point values \p x by \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_ddiv_rn(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating point values in round-towards-zero mode.
 *
 * Divides two floating point values \p x by \p y in round-towards-zero mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_ddiv_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating point values in round-down mode.
 *
 * Divides two floating point values \p x by \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_ddiv_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating point values in round-up mode.
 *
 * Divides two floating point values \p x by \p y in round-up (to positive infinity) mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_ddiv_ru(double x, double y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-to-nearest-even mode.
 *
 * Compute the reciprocal of \p x in round-to-nearest-even mode.
 *
 * \return Returns
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_drcp_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-towards-zero mode.
 *
 * Compute the reciprocal of \p x in round-towards-zero mode.
 *
 * \return Returns
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_drcp_rz(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-down mode.
 *
 * Compute the reciprocal of \p x in round-down (to negative infinity) mode.
 *
 * \return Returns
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_drcp_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-up mode.
 *
 * Compute the reciprocal of \p x in round-up (to positive infinity) mode.
 *
 * \return Returns
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_drcp_ru(double x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-to-nearest-even mode.
 *
 * Compute the square root of \p x in round-to-nearest-even mode.
 *
 * \return Returns
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_dsqrt_rn(double x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-towards-zero mode.
 *
 * Compute the square root of \p x in round-towards-zero mode.
 *
 * \return Returns
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_dsqrt_rz(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-down mode.
 *
 * Compute the square root of \p x in round-down (to negative infinity) mode.
 *
 * \return Returns
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_dsqrt_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-up mode.
 *
 * Compute the square root of \p x in round-up (to positive infinity) mode.
 *
 * \return Returns
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
double __nv_dsqrt_ru(double x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the input argument.
 *
 * Calculate the nonnegative square root of \p x,
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_sqrtf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_sqrtf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_sqrtf(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_double
 */
float __nv_sqrtf(float x);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the input argument.
 *
 * Calculate the nonnegative square root of \p x,
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_sqrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- ± --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_sqrt(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- ∞ --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - __nv_sqrt(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_double
 */
double __nv_sqrt(double x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating point values in round-to-nearest-even mode.
 *
 * Adds two floating point values \p x and \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
double __nv_dadd_rn(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating point values in round-towards-zero mode.
 *
 * Adds two floating point values \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
double __nv_dadd_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating point values in round-down mode.
 *
 * Adds two floating point values \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
double __nv_dadd_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating point values in round-up mode.
 *
 * Adds two floating point values \p x and \p y in round-up (to positive infinity) mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
double __nv_dadd_ru(double x, double y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating point values in round-to-nearest-even mode.
 *
 * Multiplies two floating point values \p x and \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
double __nv_dmul_rn(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating point values in round-towards-zero mode.
 *
 * Multiplies two floating point values \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
double __nv_dmul_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating point values in round-down mode.
 *
 * Multiplies two floating point values \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
double __nv_dmul_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating point values in round-up mode.
 *
 * Multiplies two floating point values \p x and \p y in round-up (to positive infinity) mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
double __nv_dmul_ru(double x, double y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Add two floating point values in round-down mode.
 *
 * Compute the sum of \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fadd_rd(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Add two floating point values in round-up mode.
 *
 * Compute the sum of \p x and \p y in round-up (to positive infinity) mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fadd_ru(float x, float y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Multiply two floating point values in round-down mode.
 *
 * Compute the product of \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fmul_rd(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Multiply two floating point values in round-up mode.
 *
 * Compute the product of \p x and \p y in round-up (to positive infinity) mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fmul_ru(float x, float y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Add two floating point values in round-to-nearest-even mode.
 *
 * Compute the sum of \p x and \p y in round-to-nearest-even rounding mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fadd_rn(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Add two floating point values in round-towards-zero mode.
 *
 * Compute the sum of \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fadd_rz(float x, float y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Multiply two floating point values in round-to-nearest-even mode.
 *
 * Compute the product of \p x and \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fmul_rn(float x, float y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Multiply two floating point values in round-towards-zero mode.
 *
 * Compute the product of \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_single
 * \note_nofma
 */
float __nv_fmul_rz(float x, float y);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-to-nearest-even mode.
 *
 * Convert the double-precision floating point value \p x to a single-precision
 * floating point value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
float __nv_double2float_rn(double d);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to a single-precision
 * floating point value in round-towards-zero mode.
 * \return Returns converted value.
 */
float __nv_double2float_rz(double d);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-down mode.
 *
 * Convert the double-precision floating point value \p x to a single-precision
 * floating point value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
float __nv_double2float_rd(double d);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-up mode.
 *
 * Convert the double-precision floating point value \p x to a single-precision
 * floating point value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
float __nv_double2float_ru(double d);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
int __nv_double2int_rn(double d);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
int __nv_double2int_rz(double d);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-down mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
int __nv_double2int_rd(double d);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-up mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
int __nv_double2int_ru(double d);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
unsigned int __nv_double2uint_rn(double d);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
unsigned int __nv_double2uint_rz(double d);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-down mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
unsigned int __nv_double2uint_rd(double d);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-up mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
unsigned int __nv_double2uint_ru(double d);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed int to a double.
 *
 * Convert the signed integer value \p x to a double-precision floating point value.
 * \return Returns converted value.
 */
double __nv_int2double_rn(int i);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned int to a double.
 *
 * Convert the unsigned integer value \p x to a double-precision floating point value.
 * \return Returns converted value.
 */
double __nv_uint2double_rn(unsigned int i);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
int __nv_float2int_rn(float in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
int __nv_float2int_rz(float in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
int __nv_float2int_rd(float in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
int __nv_float2int_ru(float in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
unsigned int __nv_float2uint_rn(float in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
unsigned int __nv_float2uint_rz(float in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
unsigned int __nv_float2uint_rd(float in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
unsigned int __nv_float2uint_ru(float in);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-to-nearest-even mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
float __nv_int2float_rn(int in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-towards-zero mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
float __nv_int2float_rz(int in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-down mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
float __nv_int2float_rd(int in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-up mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
float __nv_int2float_ru(int in);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-to-nearest-even mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
float __nv_uint2float_rn(unsigned int in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-towards-zero mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
float __nv_uint2float_rz(unsigned int in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-down mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
float __nv_uint2float_rd(unsigned int in);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-up mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
float __nv_uint2float_ru(unsigned int in);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret high and low 32-bit integer values as a double.
 *
 * Reinterpret the integer value of \p hi as the high 32 bits of a
 * double-precision floating point value and the integer value of \p lo
 * as the low 32 bits of the same double-precision floating point value.
 * \return Returns reinterpreted value.
 */
double __nv_hiloint2double(int x, int y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret low 32 bits in a double as a signed integer.
 *
 * Reinterpret the low 32 bits in the double-precision floating point value \p x
 * as a signed integer.
 * \return Returns reinterpreted value.
 */
int __nv_double2loint(double d);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret high 32 bits in a double as a signed integer.
 *
 * Reinterpret the high 32 bits in the double-precision floating point value \p x
 * as a signed integer.
 * \return Returns reinterpreted value.
 */
int __nv_double2hiint(double d);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
long long __nv_float2ll_rn(float f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
long long __nv_float2ll_rz(float f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
long long __nv_float2ll_rd(float f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
long long __nv_float2ll_ru(float f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
unsigned long long __nv_float2ull_rn(float f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-towards_zero mode.
 * \return Returns converted value.
 */
unsigned long long __nv_float2ull_rz(float f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
unsigned long long __nv_float2ull_rd(float f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
unsigned long long __nv_float2ull_ru(float f);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed 64-bit integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
long long __nv_double2ll_rn(double f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed 64-bit integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
long long __nv_double2ll_rz(double f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-down mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed 64-bit integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
long long __nv_double2ll_rd(double f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-up mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed 64-bit integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
long long __nv_double2ll_ru(double f);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned 64-bit integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
unsigned long long __nv_double2ull_rn(double f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned 64-bit integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
unsigned long long __nv_double2ull_rz(double f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-down mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned 64-bit integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
unsigned long long __nv_double2ull_rd(double f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-up mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned 64-bit integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
unsigned long long __nv_double2ull_ru(double f);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit integer to a float in round-to-nearest-even mode.
 *
 * Convert the signed 64-bit integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
float __nv_ll2float_rn(long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-towards-zero mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
float __nv_ll2float_rz(long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-down mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
float __nv_ll2float_rd(long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-up mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
float __nv_ll2float_ru(long long l);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-to-nearest-even mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
float __nv_ull2float_rn(unsigned long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-towards-zero mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
float __nv_ull2float_rz(unsigned long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-down mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
float __nv_ull2float_rd(unsigned long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-up mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
float __nv_ull2float_ru(unsigned long long l);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-to-nearest-even mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating
 * point value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
double __nv_ll2double_rn(long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-towards-zero mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating
 * point value in round-towards-zero mode.
 * \return Returns converted value.
 */
double __nv_ll2double_rz(long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-down mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating
 * point value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
double __nv_ll2double_rd(long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-up mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating
 * point value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
double __nv_ll2double_ru(long long l);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-to-nearest-even mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating
 * point value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
double __nv_ull2double_rn(unsigned long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-towards-zero mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating
 * point value in round-towards-zero mode.
 * \return Returns converted value.
 */
double __nv_ull2double_rz(unsigned long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-down mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating
 * point value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
double __nv_ull2double_rd(unsigned long long l);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-up mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating
 * point value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
double __nv_ull2double_ru(unsigned long long l);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a single-precision float to a half-precision float in round-to-nearest-even mode.
 *
 * Convert the single-precision float value \p x to a half-precision floating point value
 * represented in <tt>unsigned short</tt> format, in round-to-nearest-even mode.
 * \return Returns converted value.
 */
unsigned short __nv_float2half_rn(float f);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a half-precision float to a single-precision float in round-to-nearest-even mode.
 *
 * Convert the half-precision floating point value \p x represented in
 * <tt>unsigned short</tt> format to a single-precision floating point value.
 * \return Returns converted value.
 */
float __nv_half2float(unsigned short h);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in an integer as a float.
 *
 * Reinterpret the bits in the signed integer value \p x as a single-precision
 * floating point value.
 * \return Returns reinterpreted value.
 */
float __nv_int_as_float(int x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in a float as a signed integer.
 *
 * Reinterpret the bits in the single-precision floating point value \p x
 * as a signed integer.
 * \return Returns reinterpreted value.
 */
int __nv_float_as_int(float x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in a 64-bit signed integer as a double.
 *
 * Reinterpret the bits in the 64-bit signed integer value \p x as
 * a double-precision floating point value.
 * \return Returns reinterpreted value.
 */
double __nv_longlong_as_double(long long x);

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in a double as a 64-bit signed integer.
 *
 * Reinterpret the bits in the double-precision floating point value \p x
 * as a signed 64-bit integer.
 * \return Returns reinterpreted value.
 */
long long __nv_double_as_longlong(double x);

#endif /* __DEVICE_FUNCTIONS_DECLS_H__ */
