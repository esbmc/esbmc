/*
 * device_functions.h
 *
 *  Created on: Feb 20, 2015
 *      Author: isabela
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#if defined(__cplusplus) //&& defined(__CUDACC__) //

#include "builtin_types.h"
#include "device_types.h"
#include "host_defines.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern "C"
{
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 32 bits of the product of the two 32 bit integers.
 *
 * Calculate the most significant 32 bits of the 64-bit product \p x * \p y, where \p x and \p y
 * are 32-bit integers.
 *
 * \return Returns the most significant 32 bits of the product \p x * \p y.
 */
  extern __device__ __device_builtin__ int __mulhi(int x, int y);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 32 bits of the product of the two 32 bit unsigned integers.
 *
 * Calculate the most significant 32 bits of the 64-bit product \p x * \p y, where \p x and \p y
 * are 32-bit unsigned integers.
 *
 * \return Returns the most significant 32 bits of the product \p x * \p y.
 */
  extern __device__ __device_builtin__ unsigned int
  __umulhi(unsigned int x, unsigned int y);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 64 bits of the product of the two 64 bit integers.
 *
 * Calculate the most significant 64 bits of the 128-bit product \p x * \p y, where \p x and \p y
 * are 64-bit integers.
 *
 * \return Returns the most significant 64 bits of the product \p x * \p y.
 */
  extern __device__ __device_builtin__ long long int
  __mul64hi(long long int x, long long int y);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the most significant 64 bits of the product of the two 64 unsigned bit integers.
 *
 * Calculate the most significant 64 bits of the 128-bit product \p x * \p y, where \p x and \p y
 * are 64-bit unsigned integers.
 *
 * \return Returns the most significant 64 bits of the product \p x * \p y.
 */
  extern __device__ __device_builtin__ unsigned long long int
  __umul64hi(unsigned long long int x, unsigned long long int y);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in an integer as a float.
 *
 * Reinterpret the bits in the signed integer value \p x as a single-precision
 * floating point value.
 * \return Returns reinterpreted value.
 */
  extern __device__ __device_builtin__ float __int_as_float(int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in a float as a signed integer.
 *
 * Reinterpret the bits in the single-precision floating point value \p x
 * as a signed integer.
 * \return Returns reinterpreted value.
 */
  extern __device__ __device_builtin__ int __float_as_int(float x);
  //extern __device__ __device_builtin__ void                   __syncthreads(void);
  extern __device__ __device_builtin__ void __prof_trigger(int);
  extern __device__ __device_builtin__ void __threadfence(void);
  extern __device__ __device_builtin__ void __threadfence_block(void);
  extern __device__ __device_builtin__ void __trap(void);
  extern __device__ __device_builtin__ void __brkpt(int c = 0);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Clamp the input argument to [+0.0, 1.0].
 *
 * Clamp the input argument \p x to be within the interval [+0.0, 1.0].
 * \return
 * - __saturatef(\p x) returns 0 if \p x < 0.
 * - __saturatef(\p x) returns 1 if \p x > 1.
 * - __saturatef(\p x) returns \p x if
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
 * - __saturatef(NaN) returns 0.
 */
  extern __device__ __device_builtin__ float __saturatef(float x);
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
  extern __device__ __device_builtin__ unsigned int
  __sad(int x, int y, unsigned int z);
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
  extern __device__ __device_builtin__ unsigned int
  __usad(unsigned int x, unsigned int y, unsigned int z);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the least significant 32 bits of the product of the least significant 24 bits of two integers.
 *
 * Calculate the least significant 32 bits of the product of the least significant 24 bits of \p x and \p y.
 * The high order 8 bits of \p x and \p y are ignored.
 *
 * \return Returns the least significant 32 bits of the product \p x * \p y.
 */
  extern __device__ __device_builtin__ int __mul24(int x, int y);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Calculate the least significant 32 bits of the product of the least significant 24 bits of two unsigned integers.
 *
 * Calculate the least significant 32 bits of the product of the least significant 24 bits of \p x and \p y.
 * The high order 8 bits of  \p x and  \p y are ignored.
 *
 * \return Returns the least significant 32 bits of the product \p x * \p y.
 */
  extern __device__ __device_builtin__ unsigned int
  __umul24(unsigned int x, unsigned int y);
  /**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Divide two floating point values.
 *
 * Compute \p x divided by \p y.  If <tt>--use_fast_math</tt> is specified,
 * use ::__fdividef() for higher performance, otherwise use normal division.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
  extern __device__ __device_builtin__ float fdividef(float x, float y);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate division of the input arguments.
 *
 * Calculate the fast approximate division of \p x by \p y.
 *
 * \return Returns \p x / \p y.
 * - __fdividef(
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
 * - __fdividef(\p x, \p y) returns 0 for
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
  extern __device__ __device_builtin__ float __fdividef(float x, float y);
  extern __device__ __device_builtin__ double fdivide(double x, double y);
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
  extern __device__ __device_builtin__ __cudart_builtin__ float
  __sinf(float x) __THROW;
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
  extern __device__ __device_builtin__ __cudart_builtin__ float
  __cosf(float x) __THROW;
  /**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate tangent of the input argument.
 *
 * Calculate the fast approximate tangent of the input argument \p x, measured in radians.
 *
 * \return Returns the approximate tangent of \p x.
 *
 * \note_accuracy_single_intrinsic
 * \note The result is computed as the fast divide of ::__sinf()
 * by ::__cosf(). Denormal input and output are flushed to sign-preserving
 * 0.0 at each step of the computation.
 */
  extern __device__ __device_builtin__ __cudart_builtin__ float
  __tanf(float x) __THROW;
  /**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Calculate the fast approximate of sine and cosine of the first input argument.
 *
 * Calculate the fast approximate of sine and cosine of the first input argument \p x (measured
 * in radians). The results for sine and cosine are written into the second
 * argument, \p sptr, and, respectively, third argument, \p cptr.
 *
 * \return
 * - none
 *
 * \note_accuracy_single_intrinsic
 * \note Denorm input/output is flushed to sign preserving 0.0.
 */
  extern __device__ __device_builtin__ __cudart_builtin__ void
  __sincosf(float x, float *sptr, float *cptr) __THROW;
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
  extern __device__ __device_builtin__ __cudart_builtin__ float
  __expf(float x) __THROW;
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
  extern __device__ __device_builtin__ __cudart_builtin__ float
  __exp10f(float x) __THROW;
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
  extern __device__ __device_builtin__ __cudart_builtin__ float
  __log2f(float x) __THROW;
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
  extern __device__ __device_builtin__ __cudart_builtin__ float
  __log10f(float x) __THROW;
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
  extern __device__ __device_builtin__ __cudart_builtin__ float
  __logf(float x) __THROW;
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
  extern __device__ __device_builtin__ __cudart_builtin__ float
  __powf(float x, float y) __THROW;
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ int __float2int_rn(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ int __float2int_rz(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ int __float2int_ru(float);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to a signed integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ int __float2int_rd(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ unsigned int __float2uint_rn(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ unsigned int __float2uint_rz(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ unsigned int __float2uint_ru(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ unsigned int __float2uint_rd(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-to-nearest-even mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __int2float_rn(int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-towards-zero mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __int2float_rz(int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-up mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __int2float_ru(int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-down mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __int2float_rd(int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-to-nearest-even mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __uint2float_rn(unsigned int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-towards-zero mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __uint2float_rz(unsigned int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-up mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __uint2float_ru(unsigned int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-down mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __uint2float_rd(unsigned int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ long long int __float2ll_rn(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ long long int __float2ll_rz(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ long long int __float2ll_ru(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to a signed 64-bit integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to a signed 64-bit integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ long long int __float2ll_rd(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-to-nearest-even mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ unsigned long long int
  __float2ull_rn(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-towards-zero mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-towards_zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ unsigned long long int
  __float2ull_rz(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-up mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ unsigned long long int
  __float2ull_ru(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a float to an unsigned 64-bit integer in round-down mode.
 *
 * Convert the single-precision floating point value \p x to an unsigned 64-bit integer
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ unsigned long long int
  __float2ull_rd(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit integer to a float in round-to-nearest-even mode.
 *
 * Convert the signed 64-bit integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __ll2float_rn(long long int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-towards-zero mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __ll2float_rz(long long int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-up mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __ll2float_ru(long long int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed integer to a float in round-down mode.
 *
 * Convert the signed integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __ll2float_rd(long long int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-to-nearest-even mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-to-nearest-even mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float
  __ull2float_rn(unsigned long long int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-towards-zero mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-towards-zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float
  __ull2float_rz(unsigned long long int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-up mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float
  __ull2float_ru(unsigned long long int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned integer to a float in round-down mode.
 *
 * Convert the unsigned integer value \p x to a single-precision floating point value
 * in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float
  __ull2float_rd(unsigned long long int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a single-precision float to a half-precision float in round-to-nearest-even mode.
 *
 * Convert the single-precision float value \p x to a half-precision floating point value
 * represented in <tt>unsigned short</tt> format, in round-to-nearest-even mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ unsigned short __float2half_rn(float x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a half-precision float to a single-precision float in round-to-nearest-even mode.
 *
 * Convert the half-precision floating point value \p x represented in
 * <tt>unsigned short</tt> format to a single-precision floating point value.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ float __half2float(unsigned short x);
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
  extern __device__ __device_builtin__ float __fadd_rn(float x, float y);
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
  extern __device__ __device_builtin__ float __fadd_rz(float x, float y);
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
  extern __device__ __device_builtin__ float __fadd_ru(float x, float y);
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
  extern __device__ __device_builtin__ float __fadd_rd(float x, float y);
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
  extern __device__ __device_builtin__ float __fsub_rn(float x, float y);
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
  extern __device__ __device_builtin__ float __fsub_rz(float x, float y);
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
  extern __device__ __device_builtin__ float __fsub_ru(float x, float y);
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
  extern __device__ __device_builtin__ float __fsub_rd(float x, float y);
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
  extern __device__ __device_builtin__ float __fmul_rn(float x, float y);
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
  extern __device__ __device_builtin__ float __fmul_rz(float x, float y);
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
  extern __device__ __device_builtin__ float __fmul_ru(float x, float y);
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
  extern __device__ __device_builtin__ float __fmul_rd(float x, float y);
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
 * - fmaf(
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
 * - fmaf(
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
 * - fmaf(\p x, \p y,
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
 * - fmaf(\p x, \p y,
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
  extern __device__ __device_builtin__ float
  __fmaf_rn(float x, float y, float z);
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
 * - fmaf(
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
 * - fmaf(
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
 * - fmaf(\p x, \p y,
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
 * - fmaf(\p x, \p y,
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
  extern __device__ __device_builtin__ float
  __fmaf_rz(float x, float y, float z);
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
 * - fmaf(
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
 * - fmaf(
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
 * - fmaf(\p x, \p y,
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
 * - fmaf(\p x, \p y,
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
  extern __device__ __device_builtin__ float
  __fmaf_ru(float x, float y, float z);
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
 * - fmaf(
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
 * - fmaf(
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
 * - fmaf(\p x, \p y,
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
 * - fmaf(\p x, \p y,
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
  extern __device__ __device_builtin__ float
  __fmaf_rd(float x, float y, float z);
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
  extern __device__ __device_builtin__ float __frcp_rn(float x);
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
  extern __device__ __device_builtin__ float __frcp_rz(float x);
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
  extern __device__ __device_builtin__ float __frcp_ru(float x);
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
  extern __device__ __device_builtin__ float __frcp_rd(float x);
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
  extern __device__ __device_builtin__ float __fsqrt_rn(float x);
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
  extern __device__ __device_builtin__ float __fsqrt_rz(float x);
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
  extern __device__ __device_builtin__ float __fsqrt_ru(float x);
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
  extern __device__ __device_builtin__ float __fsqrt_rd(float x);
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
  extern __device__ __device_builtin__ float __frsqrt_rn(float x);
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
  extern __device__ __device_builtin__ float __fdiv_rn(float x, float y);
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
  extern __device__ __device_builtin__ float __fdiv_rz(float x, float y);
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
  extern __device__ __device_builtin__ float __fdiv_ru(float x, float y);
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
  extern __device__ __device_builtin__ float __fdiv_rd(float x, float y);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Return the number of consecutive high-order zero bits in a 32 bit integer.
 *
 * Count the number of consecutive leading zero bits, starting at the most significant bit (bit 31) of \p x.
 *
 * \return Returns a value between 0 and 32 inclusive representing the number of zero bits.
 */
  extern __device__ __device_builtin__ int __clz(int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Find the position of the least significant bit set to 1 in a 32 bit integer.
 *
 * Find the position of the first (least significant) bit set to 1 in \p x, where the least significant
 * bit position is 1.
 *
 * \return Returns a value between 0 and 32 inclusive representing the position of the first bit set.
 * - __ffs(0) returns 0.
 */
  extern __device__ __device_builtin__ int __ffs(int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Count the number of bits that are set to 1 in a 32 bit integer.
 *
 * Count the number of bits that are set to 1 in \p x.
 *
 * \return Returns a value between 0 and 32 inclusive representing the number of set bits.
 */
  extern __device__ __device_builtin__ int __popc(unsigned int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Reverse the bit order of a 32 bit unsigned integer.
 *
 * Reverses the bit order of the 32 bit unsigned integer \p x.
 *
 * \return Returns the bit-reversed value of \p x. i.e. bit N of the return value corresponds to bit 31-N of \p x.
 */
  extern __device__ __device_builtin__ unsigned int __brev(unsigned int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Count the number of consecutive high-order zero bits in a 64 bit integer.
 *
 * Count the number of consecutive leading zero bits, starting at the most significant bit (bit 63) of \p x.
 *
 * \return Returns a value between 0 and 64 inclusive representing the number of zero bits.
 */
  extern __device__ __device_builtin__ int __clzll(long long int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Find the position of the least significant bit set to 1 in a 64 bit integer.
 *
 * Find the position of the first (least significant) bit set to 1 in \p x, where the least significant
 * bit position is 1.
 *
 * \return Returns a value between 0 and 64 inclusive representing the position of the first bit set.
 * - __ffsll(0) returns 0.
 */
  extern __device__ __device_builtin__ int __ffsll(long long int x);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Count the number of bits that are set to 1 in a 64 bit integer.
 *
 * Count the number of bits that are set to 1 in \p x.
 *
 * \return Returns a value between 0 and 64 inclusive representing the number of set bits.
 */
  extern __device__ __device_builtin__ int __popcll(unsigned long long int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Reverse the bit order of a 64 bit unsigned integer.
 *
 * Reverses the bit order of the 64 bit unsigned integer \p x.
 *
 * \return Returns the bit-reversed value of \p x. i.e. bit N of the return value corresponds to bit 63-N of \p x.
 */
  extern __device__ __device_builtin__ unsigned long long int
  __brevll(unsigned long long int x);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Return selected bytes from two 32 bit unsigned integers.
 *
 * byte_perm(x,y,s) returns a 32-bit integer consisting of four bytes from eight input bytes provided in the two
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
  extern __device__ __device_builtin__ unsigned int
  __byte_perm(unsigned int x, unsigned int y, unsigned int s);
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
  extern __device__ __device_builtin__ int __hadd(int, int);
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
  extern __device__ __device_builtin__ int __rhadd(int, int);
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
  extern __device__ __device_builtin__ unsigned int
  __uhadd(unsigned int, unsigned int);
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
  extern __device__ __device_builtin__ unsigned int
  __urhadd(unsigned int, unsigned int);

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 130
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ int __double2int_rz(double);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ unsigned int __double2uint_rz(double);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to a
 * signed 64-bit integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ long long int __double2ll_rz(double);
  /**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-towards-zero mode.
 *
 * Convert the double-precision floating point value \p x to an
 * unsigned 64-bit integer value in round-towards-zero mode.
 * \return Returns converted value.
 */
  extern __device__ __device_builtin__ unsigned long long int
  __double2ull_rz(double);
  extern __device__ __device_builtin__ unsigned int __pm0(void);
  extern __device__ __device_builtin__ unsigned int __pm1(void);
  extern __device__ __device_builtin__ unsigned int __pm2(void);
  extern __device__ __device_builtin__ unsigned int __pm3(void);
#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 130 */

  /*******************************************************************************
 *                                                                             *
 *                                SIMD functions                               *
 *                                                                             *
 *******************************************************************************/

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword absolute value.
 *
 * Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes,
 * then computes absolute value for each of parts.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int __vabs2(unsigned int a);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword absolute value with signed saturation.
 *
 * Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes,
 * then computes absolute value with signed saturation for each of parts.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int __vabsss2(unsigned int a);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed addition, with wrap-around: a + b
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes,
 * then performs unsigned addition on corresponding parts.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vadd2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword addition with signed saturation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes,
 * then performs addition with signed saturation on corresponding parts.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vaddss2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword addition with unsigned saturation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes,
 * then performs addition with unsigned saturation on corresponding parts.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vaddus2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed rounded average computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * then computes signed rounded avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vavgs2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned rounded average computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * then computes unsigned rounded avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vavgu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned average computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * then computes unsigned avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vhaddu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if they are equal, and 0000 otherwise.
 * For example __vcmpeq2(0x1234aba5, 0x1234aba6) returns 0xffff0000.
 * \return Returns 0xffff computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpeq2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison: a >= b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part >= 'b' part, and 0000 otherwise.
 * For example __vcmpges2(0x1234aba5, 0x1234aba6) returns 0xffff0000.
 * \return Returns 0xffff if a >= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpges2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison: a >= b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part >= 'b' part, and 0000 otherwise.
 * For example __vcmpgeu2(0x1234aba5, 0x1234aba6) returns 0xffff0000.
 * \return Returns 0xffff if a >= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpgeu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison: a > b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part > 'b' part, and 0000 otherwise.
 * For example __vcmpgts2(0x1234aba5, 0x1234aba6) returns 0x00000000.
 * \return Returns 0xffff if a > b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpgts2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison: a > b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part > 'b' part, and 0000 otherwise.
 * For example __vcmpgtu2(0x1234aba5, 0x1234aba6) returns 0x00000000.
 * \return Returns 0xffff if a > b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpgtu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison: a <= b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part <= 'b' part, and 0000 otherwise.
 * For example __vcmples2(0x1234aba5, 0x1234aba6) returns 0xffffffff.
 * \return Returns 0xffff if a <= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmples2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison: a <= b ? 0xffff : 0.
 *
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part <= 'b' part, and 0000 otherwise.
 * For example __vcmpleu2(0x1234aba5, 0x1234aba6) returns 0xffffffff.
 * \return Returns 0xffff if a <= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpleu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison: a < b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part < 'b' part, and 0000 otherwise.
 * For example __vcmplts2(0x1234aba5, 0x1234aba6) returns 0x0000ffff.
 * \return Returns 0xffff if a < b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmplts2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison: a < b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part < 'b' part, and 0000 otherwise.
 * For example __vcmpltu2(0x1234aba5, 0x1234aba6) returns 0x0000ffff.
 * \return Returns 0xffff if a < b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpltu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed comparison: a != b ? 0xffff : 0.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts result is ffff if 'a' part != 'b' part, and 0000 otherwise.
 * For example __vcmplts2(0x1234aba5, 0x1234aba6) returns 0x0000ffff.
 * \return Returns 0xffff if a != b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpne2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword absolute difference of unsigned integer computation: |a - b|
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes absolute difference. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vabsdiffu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed maximum computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes signed maximum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vmaxs2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned maximum computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes unsigned maximum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vmaxu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed minimum computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes signed minimum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vmins2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned minimum computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes unsigned minimum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vminu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part == 'b' part.
 * If both equalities are satisfiad, function returns 1.
 * \return Returns 1 if a = b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vseteq2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part >= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a >= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetges2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned minimum unsigned comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part >= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a >= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetgeu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part > 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a > b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetgts2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part > 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a > b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetgtu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned minimum computation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a <= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetles2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a <= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetleu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a < b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetlts2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword unsigned comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a < b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetltu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function performs comparison 'a' part != 'b' part.
 * If both conditions are satisfied, function returns 1.
 * \return Returns 1 if a != b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetne2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword sum of abs diff of unsigned.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes absolute differences, and returns
 * sum of those differences.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsadu2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed substraction, with wrap-around.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts functions performs substraction. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsub2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword (un)signed substraction, with signed saturation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts functions performs substraction with signed saturation.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsubss2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword substraction with unsigned saturation.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts functions performs substraction with unsigned saturation.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsubus2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword negation.
 *
 * Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes.
 * For each part function computes negation. Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int __vneg2(unsigned int a);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword negation with signed saturation.
 *
 * Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes.
 * For each part function computes negation. Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int __vnegss2(unsigned int a);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-halfword sum of absolute difference of signed integer.
 *
 * Splits 4 bytes of each into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes absolute difference.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vabsdiffs2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-halfword sum of absolute difference of signed.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts functions computes absolute difference and sum it up.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsads2(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte absolute value.
 *
 * Splits argument by bytes. Computes absolute value of each byte.
 * Result is stored as unsigned int.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int __vabs4(unsigned int a);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte absolute value with signed saturation.
 *
 * Splits 4 bytes of argument into 4 parts, each consisting of 1 byte,
 * then computes absolute value with signed saturation for each of parts.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int __vabsss4(unsigned int a);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte (un)signed addition.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte,
 * then performs unsigned addition on corresponding parts.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vadd4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte addition with signed saturation.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte,
 * then performs addition with signed saturation on corresponding parts.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vaddss4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unaddition with signed saturation.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte,
 * then performs addition with unsigned saturation on corresponding parts.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vaddus4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte signed rounder average.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * then computes signed rounded avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vavgs4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned rounded average.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * then computes unsigned rounded avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vavgu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte unsigned average.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * then computes unsigned avarege of corresponding parts. Result is stored as
 * unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vhaddu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if they are equal, and 00 otherwise.
 * For example __vcmpeq4(0x1234aba5, 0x1234aba6) returns 0xffffff00.
 * \return Returns 0xff if a = b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpeq4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part >= 'b' part, and 00 otherwise.
 * For example __vcmpges4(0x1234aba5, 0x1234aba6) returns 0xffffff00.
 * \return Returns 0xff if a >= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpges4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part >= 'b' part, and 00 otherwise.
 * For example __vcmpgeu4(0x1234aba5, 0x1234aba6) returns 0xffffff00.
 * \return Returns 0xff if a = b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpgeu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part > 'b' part, and 00 otherwise.
 * For example __vcmpgts4(0x1234aba5, 0x1234aba6) returns 0x00000000.
 * \return Returns 0xff if a > b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpgts4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part > 'b' part, and 00 otherwise.
 * For example __vcmpgtu4(0x1234aba5, 0x1234aba6) returns 0x00000000.
 * \return Returns 0xff if a > b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpgtu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part <= 'b' part, and 00 otherwise.
 * For example __vcmples4(0x1234aba5, 0x1234aba6) returns 0xffffffff.
 * \return Returns 0xff if a <= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmples4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part <= 'b' part, and 00 otherwise.
 * For example __vcmpleu4(0x1234aba5, 0x1234aba6) returns 0xffffffff.
 * \return Returns 0xff if a <= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpleu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part < 'b' part, and 00 otherwise.
 * For example __vcmplts4(0x1234aba5, 0x1234aba6) returns 0x000000ff.
 * \return Returns 0xff if a < b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmplts4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part < 'b' part, and 00 otherwise.
 * For example __vcmpltu4(0x1234aba5, 0x1234aba6) returns 0x000000ff.
 * \return Returns 0xff if a < b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpltu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts result is ff if 'a' part != 'b' part, and 00 otherwise.
 * For example __vcmplts4(0x1234aba5, 0x1234aba6) returns 0x000000ff.
 * \return Returns 0xff if a != b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vcmpne4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte absolute difference of unsigned integer.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes absolute difference. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vabsdiffu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte signed maximum.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes signed maximum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vmaxs4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte unsigned maximum.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes unsigned maximum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vmaxu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte signed minimum.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes signed minimum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vmins4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte unsigned minimum.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes unsigned minimum. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vminu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part == 'b' part.
 * If both equalities are satisfiad, function returns 1.
 * \return Returns 1 if a = b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vseteq4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a <= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetles4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 part, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a <= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetleu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a < b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetlts4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part <= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a < b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetltu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part >= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a >= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetges4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part >= 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a >= b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetgeu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part > 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a > b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetgts4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte unsigned comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts function performs comparison 'a' part > 'b' part.
 * If both inequalities are satisfied, function returns 1.
 * \return Returns 1 if a > b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetgtu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte (un)signed comparison.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 bytes.
 * For corresponding parts function performs comparison 'a' part != 'b' part.
 * If both conditions are satisfied, function returns 1.
 * \return Returns 1 if a != b, else returns 0.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsetne4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte sum af abs difference of unsigned.
 *
 * Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes.
 * For corresponding parts function computes absolute differences, and returns
 * sum of those differences.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsadu4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte substraction.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 bytes.
 * For corresponding parts functions performs substraction. Result is stored
 * as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsub4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte substraction with signed saturation.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts functions performs substraction with signed saturation.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsubss4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte substraction with unsigned saturation.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts functions performs substraction with unsigned saturation.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsubus4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte negation.
 *
 * Splits 4 bytes of argument into 4 parts, each consisting of 1 byte.
 * For each part function computes negation. Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int __vneg4(unsigned int a);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Performs per-byte negation with signed saturation.
 *
 * Splits 4 bytes of argument into 4 parts, each consisting of 1 byte.
 * For each part function computes negation. Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int __vnegss4(unsigned int a);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte absolute difference of signed integer.
 *
 * Splits 4 bytes of each into 4 parts, each consisting of 1 byte.
 * For corresponding parts function computes absolute difference.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vabsdiffs4(unsigned int a, unsigned int b);

  /**
 * \ingroup CUDA_MATH_INTRINSIC_SIMD
 * \brief Computes per-byte sum of abs difference of signed.
 *
 * Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte.
 * For corresponding parts functions computes absolute difference and sum it up.
 * Result is stored as unsigned int and returned.
 * \return Returns computed value.
 */
  extern __device__ __device_builtin__ unsigned int
  __vsads4(unsigned int a, unsigned int b);

  /*******************************************************************************
 *                                                                             *
 *                            END SIMD functions                               *
 *                                                                             *
 *******************************************************************************/
} /*end of extern C*/

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

static __inline__ __device__ int mulhi(int a, int b)
{
  return __mulhi(a, b);
}

static __inline__ __device__ unsigned int mulhi(unsigned int a, unsigned int b)
{
  return __umulhi(a, b);
}

static __inline__ __device__ unsigned int mulhi(int a, unsigned int b)
{
  return __umulhi((unsigned int)a, b);
}

static __inline__ __device__ unsigned int mulhi(unsigned int a, int b)
{
  return __umulhi(a, (unsigned int)b);
}

static __inline__ __device__ long long int
mul64hi(long long int a, long long int b)
{
  return __mul64hi(a, b);
}

static __inline__ __device__ unsigned long long int
mul64hi(unsigned long long int a, unsigned long long int b)
{
  return __umul64hi(a, b);
}

static __inline__ __device__ unsigned long long int
mul64hi(long long int a, unsigned long long int b)
{
  return __umul64hi((unsigned long long int)a, b);
}

static __inline__ __device__ unsigned long long int
mul64hi(unsigned long long int a, long long int b)
{
  return __umul64hi(a, (unsigned long long int)b);
}

static __inline__ __device__ int float_as_int(float a)
{
  return __float_as_int(a);
}

static __inline__ __device__ float int_as_float(int a)
{
  return __int_as_float(a);
}

static __inline__ __device__ float saturate(float a)
{
  return __saturatef(a);
}

static __inline__ __device__ int mul24(int a, int b)
{
  return __mul24(a, b);
}

static __inline__ __device__ unsigned int umul24(unsigned int a, unsigned int b)
{
  return __umul24(a, b);
}

static __inline__ __device__ void trap(void)
{
  __trap();
}

/* argument is optional, value of 0 means no value */
static __inline__ __device__ void brkpt(int c = 0)
{
  __brkpt(c);
}

static __inline__ __device__ void syncthreads(void)
{
  __syncthreads();
}

static __inline__ __device__ void prof_trigger(int e)
{
  if(e == 0)
    __prof_trigger(0);
  else if(e == 1)
    __prof_trigger(1);
  else if(e == 2)
    __prof_trigger(2);
  else if(e == 3)
    __prof_trigger(3);
  else if(e == 4)
    __prof_trigger(4);
  else if(e == 5)
    __prof_trigger(5);
  else if(e == 6)
    __prof_trigger(6);
  else if(e == 7)
    __prof_trigger(7);
  else if(e == 8)
    __prof_trigger(8);
  else if(e == 9)
    __prof_trigger(9);
  else if(e == 10)
    __prof_trigger(10);
  else if(e == 11)
    __prof_trigger(11);
  else if(e == 12)
    __prof_trigger(12);
  else if(e == 13)
    __prof_trigger(13);
  else if(e == 14)
    __prof_trigger(14);
  else if(e == 15)
    __prof_trigger(15);
}

static __inline__ __device__ void threadfence(bool global = true)
{
  global ? __threadfence() : __threadfence_block();
}

static __inline__ __device__ int
float2int(float a, enum cudaRoundMode mode = cudaRoundZero)
{
  return mode == cudaRoundNearest  ? __float2int_rn(a)
         : mode == cudaRoundPosInf ? __float2int_ru(a)
         : mode == cudaRoundMinInf ? __float2int_rd(a)
                                   : __float2int_rz(a);
}

static __inline__ __device__ unsigned int
float2uint(float a, enum cudaRoundMode mode = cudaRoundZero)
{
  return mode == cudaRoundNearest  ? __float2uint_rn(a)
         : mode == cudaRoundPosInf ? __float2uint_ru(a)
         : mode == cudaRoundMinInf ? __float2uint_rd(a)
                                   : __float2uint_rz(a);
}

static __inline__ __device__ float
int2float(int a, enum cudaRoundMode mode = cudaRoundNearest)
{
  return mode == cudaRoundZero     ? __int2float_rz(a)
         : mode == cudaRoundPosInf ? __int2float_ru(a)
         : mode == cudaRoundMinInf ? __int2float_rd(a)
                                   : __int2float_rn(a);
}

static __inline__ __device__ float
uint2float(unsigned int a, enum cudaRoundMode mode = cudaRoundNearest)
{
  return mode == cudaRoundZero     ? __uint2float_rz(a)
         : mode == cudaRoundPosInf ? __uint2float_ru(a)
         : mode == cudaRoundMinInf ? __uint2float_rd(a)
                                   : __uint2float_rn(a);
}

#elif 1 //defined(__CUDABE__)

#if 1 //defined(__CUDANVVM__)

/*******************************************************************************
*                                                                              *
* SYNCHRONIZATION FUNCTIONS                                                    *
*                                                                              *
*******************************************************************************/
static __forceinline__ int __syncthreads_count(int predicate)
{
  return __nvvm_bar0_popc(predicate);
}

static __forceinline__ int __syncthreads_and(int predicate)
{
  return __nvvm_bar0_and(predicate);
}

static __forceinline__ int __syncthreads_or(int predicate)
{
  return __nvvm_bar0_or(predicate);
}

/*******************************************************************************
*                                                                              *
* MEMORY FENCE FUNCTIONS                                                       *
*                                                                              *
*******************************************************************************/
static __forceinline__ void __threadfence_block()
{
  __nvvm_membar_cta();
}

static __forceinline__ void __threadfence()
{
  __nvvm_membar_gl();
}

static __forceinline__ void __threadfence_system()
{
  __nvvm_membar_sys();
}

/*******************************************************************************
*                                                                              *
* VOTE FUNCTIONS                                                               *
*                                                                              *
*******************************************************************************/
static __forceinline__ int __all(int a)
{
  int result;
  asm __volatile__(
    "{ \n\t"
    ".reg .pred \t%%p1; \n\t"
    ".reg .pred \t%%p2; \n\t"
    "setp.ne.u32 \t%%p1, %1, 0; \n\t"
    "vote.all.pred \t%%p2, %%p1; \n\t"
    "selp.s32 \t%0, 1, 0, %%p2; \n\t"
    "}"
    : "=r"(result)
    : "r"(a));
  return result;
}

static __forceinline__ int __any(int a)
{
  int result;
  asm __volatile__(
    "{ \n\t"
    ".reg .pred \t%%p1; \n\t"
    ".reg .pred \t%%p2; \n\t"
    "setp.ne.u32 \t%%p1, %1, 0; \n\t"
    "vote.any.pred \t%%p2, %%p1; \n\t"
    "selp.s32 \t%0, 1, 0, %%p2; \n\t"
    "}"
    : "=r"(result)
    : "r"(a));
  return result;
}

static __forceinline__ int __ballot(int a)
{
  int result;
  asm __volatile__(
    "{ \n\t"
    ".reg .pred \t%%p1; \n\t"
    "setp.ne.u32 \t%%p1, %1, 0; \n\t"
    "vote.ballot.b32 \t%0, %%p1; \n\t"
    "}"
    : "=r"(result)
    : "r"(a));
  return result;
}

/*******************************************************************************
*                                                                              *
* MISCELLANEOUS FUNCTIONS                                                      *
*                                                                              *
*******************************************************************************/
static __forceinline__ void __brkpt()
{
  asm __volatile__("brkpt;");
}

static __forceinline__ int clock()
{
  int r;
  asm __volatile__("mov.u32 \t%0, %%clock;" : "=r"(r));
  return r;
}

static __forceinline__ long long clock64()
{
  long long z;
  asm __volatile__("mov.u64 \t%0, %%clock64;" : "=l"(z));
  return z;
}

#define __prof_trigger(X) asm __volatile__("pmevent \t" #X ";")

static __forceinline__ unsigned int __pm0(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm0;" : "=r"(r));
  return r;
}

static __forceinline__ unsigned int __pm1(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm1;" : "=r"(r));
  return r;
}

static __forceinline__ unsigned int __pm2(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm2;" : "=r"(r));
  return r;
}

static __forceinline__ unsigned int __pm3(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm3;" : "=r"(r));
  return r;
}

static __forceinline__ void __trap(void)
{
  asm __volatile__("trap;");
}

static __forceinline__ void *memcpy(void *dest, const void *src, size_t n)
{
  __nvvm_memcpy(
    (unsigned char *)dest,
    (unsigned char *)src,
    n,
    /*alignment=*/1);
  return dest;
}

static __forceinline__ void *memset(void *dest, int c, size_t n)
{
  __nvvm_memset(
    (unsigned char *)dest,
    (unsigned char)c,
    n,
    /*alignment=*/1);
  return dest;
}

/*******************************************************************************
*                                                                              *
* MATH FUNCTIONS                                                               *
*                                                                              *
*******************************************************************************/
static __forceinline__ int __clz(int x)
{
  return __nv_clz(x);
}

static __forceinline__ int __clzll(long long x)
{
  return __nv_clzll(x);
}

static __forceinline__ int __popc(int x)
{
  return __nv_popc(x);
}

static __forceinline__ int __popcll(long long x)
{
  return __nv_popcll(x);
}

static __forceinline__ unsigned int
__byte_perm(unsigned int a, unsigned int b, unsigned int c)
{
  return __nv_byte_perm(a, b, c);
}

/*******************************************************************************
*                                                                              *
* INTEGER MATH FUNCTIONS                                                       *
*                                                                              *
*******************************************************************************/
static __forceinline__ int min(int x, int y)
{
  return __nv_min(x, y);
}

static __forceinline__ unsigned int umin(unsigned int x, unsigned int y)
{
  return __nv_umin(x, y);
}

static __forceinline__ long long llmin(long long x, long long y)
{
  return __nv_llmin(x, y);
}

static __forceinline__ unsigned long long
ullmin(unsigned long long x, unsigned long long y)
{
  return __nv_ullmin(x, y);
}

static __forceinline__ int max(int x, int y)
{
  return __nv_max(x, y);
}

static __forceinline__ unsigned int umax(unsigned int x, unsigned int y)
{
  return __nv_umax(x, y);
}

static __forceinline__ long long llmax(long long x, long long y)
{
  return __nv_llmax(x, y);
}

static __forceinline__ unsigned long long
ullmax(unsigned long long x, unsigned long long y)
{
  return __nv_ullmax(x, y);
}

static __forceinline__ int __mulhi(int x, int y)
{
  return __nv_mulhi(x, y);
}

static __forceinline__ unsigned int __umulhi(unsigned int x, unsigned int y)
{
  return __nv_umulhi(x, y);
}

static __forceinline__ long long __mul64hi(long long x, long long y)
{
  return __nv_mul64hi(x, y);
}

static __forceinline__ unsigned long long
__umul64hi(unsigned long long x, unsigned long long y)
{
  return __nv_umul64hi(x, y);
}

static __forceinline__ int __mul24(int x, int y)
{
  return __nv_mul24(x, y);
}

static __forceinline__ unsigned int __umul24(unsigned int x, unsigned int y)
{
  return __nv_umul24(x, y);
}

static __forceinline__ unsigned int __brev(unsigned int x)
{
  return __nv_brev(x);
}

static __forceinline__ unsigned long long __brevll(unsigned long long x)
{
  return __nv_brevll(x);
}

static __forceinline__ int __sad(int x, int y, int z)
{
  return __nv_sad(x, y, z);
}

static __forceinline__ unsigned int
__usad(unsigned int x, unsigned int y, unsigned int z)
{
  return __nv_usad(x, y, z);
}

static __forceinline__ int abs(int x)
{
  return __nv_abs(x);
}

static __forceinline__ long labs(long x)
{
#if defined(__LP64__)
  return __nv_llabs((long long)x);
#else /* __LP64__ */
  return __nv_abs((int)x);
#endif /* __LP64__ */
}

static __forceinline__ long long llabs(long long x)
{
  return __nv_llabs(x);
}

/*******************************************************************************
*                                                                              *
* FP MATH FUNCTIONS                                                            *
*                                                                              *
*******************************************************************************/
static __forceinline__ float floorf(float f)
{
  return __nv_floorf(f);
}

static __forceinline__ double floor(double f)
{
  return __nv_floor(f);
}

static __forceinline__ float fabsf(float f)
{
  return __nv_fabsf(f);
}

static __forceinline__ double fabs(double f)
{
  return __nv_fabs(f);
}

static __forceinline__ double __rcp64h(double d)
{
  return __nv_rcp64h(d);
}

static __forceinline__ float fminf(float x, float y)
{
  return __nv_fminf(x, y);
}

static __forceinline__ float fmaxf(float x, float y)
{
  return __nv_fmaxf(x, y);
}

static __forceinline__ float rsqrtf(float x)
{
  return __nv_rsqrtf(x);
}

static __forceinline__ double fmin(double x, double y)
{
  return __nv_fmin(x, y);
}

static __forceinline__ double fmax(double x, double y)
{
  return __nv_fmax(x, y);
}

static __forceinline__ double rsqrt(double x)
{
  return __nv_rsqrt(x);
}

static __forceinline__ double ceil(double x)
{
  return __nv_ceil(x);
}

static __forceinline__ double trunc(double x)
{
  return __nv_trunc(x);
}

static __forceinline__ float exp2f(float x)
{
  return __nv_exp2f(x);
}

static __forceinline__ float truncf(float x)
{
  return __nv_truncf(x);
}

static __forceinline__ float ceilf(float x)
{
  return __nv_ceilf(x);
}

static __forceinline__ float __saturatef(float x)
{
  return __nv_saturatef(x);
}

/*******************************************************************************
*                                                                              *
* FMAF                                                                         *
*                                                                              *
*******************************************************************************/
static __forceinline__ float __fmaf_rn(float x, float y, float z)
{
  return __nv_fmaf_rn(x, y, z);
}

static __forceinline__ float __fmaf_rz(float x, float y, float z)
{
  return __nv_fmaf_rz(x, y, z);
}

static __forceinline__ float __fmaf_rd(float x, float y, float z)
{
  return __nv_fmaf_rd(x, y, z);
}

static __forceinline__ float __fmaf_ru(float x, float y, float z)
{
  return __nv_fmaf_ru(x, y, z);
}

/*******************************************************************************
*                                                                              *
* FMAF_IEEE                                                                    *
*                                                                              *
*******************************************************************************/
static __forceinline__ float __fmaf_ieee_rn(float x, float y, float z)
{
  return __nv_fmaf_ieee_rn(x, y, z);
}

static __forceinline__ float __fmaf_ieee_rz(float x, float y, float z)
{
  return __nv_fmaf_ieee_rz(x, y, z);
}

static __forceinline__ float __fmaf_ieee_rd(float x, float y, float z)
{
  return __nv_fmaf_ieee_rd(x, y, z);
}

static __forceinline__ float __fmaf_ieee_ru(float x, float y, float z)
{
  return __nv_fmaf_ieee_ru(x, y, z);
}

/*******************************************************************************
*                                                                              *
* FMA                                                                          *
*                                                                              *
*******************************************************************************/
static __forceinline__ double __fma_rn(double x, double y, double z)
{
  return __nv_fma_rn(x, y, z);
}

static __forceinline__ double __fma_rz(double x, double y, double z)
{
  return __nv_fma_rz(x, y, z);
}

static __forceinline__ double __fma_rd(double x, double y, double z)
{
  return __nv_fma_rd(x, y, z);
}

static __forceinline__ double __fma_ru(double x, double y, double z)
{
  return __nv_fma_ru(x, y, z);
}

static __forceinline__ float __fdividef(float x, float y)
{
  return __nv_fast_fdividef(x, y);
}

/*******************************************************************************
*                                                                              *
* FDIV                                                                         *
*                                                                              *
*******************************************************************************/
static __forceinline__ float __fdiv_rn(float x, float y)
{
  return __nv_fdiv_rn(x, y);
}

static __forceinline__ float __fdiv_rz(float x, float y)
{
  return __nv_fdiv_rz(x, y);
}

static __forceinline__ float __fdiv_rd(float x, float y)
{
  return __nv_fdiv_rd(x, y);
}

static __forceinline__ float __fdiv_ru(float x, float y)
{
  return __nv_fdiv_ru(x, y);
}

/*******************************************************************************
*                                                                              *
* FRCP                                                                         *
*                                                                              *
*******************************************************************************/
static __forceinline__ float __frcp_rn(float x)
{
  return __nv_frcp_rn(x);
}

static __forceinline__ float __frcp_rz(float x)
{
  return __nv_frcp_rz(x);
}

static __forceinline__ float __frcp_rd(float x)
{
  return __nv_frcp_rd(x);
}

static __forceinline__ float __frcp_ru(float x)
{
  return __nv_frcp_ru(x);
}

/*******************************************************************************
*                                                                              *
* FSQRT                                                                        *
*                                                                              *
*******************************************************************************/
static __forceinline__ float __fsqrt_rn(float x)
{
  return __nv_fsqrt_rn(x);
}

static __forceinline__ float __fsqrt_rz(float x)
{
  return __nv_fsqrt_rz(x);
}

static __forceinline__ float __fsqrt_rd(float x)
{
  return __nv_fsqrt_rd(x);
}

static __forceinline__ float __fsqrt_ru(float x)
{
  return __nv_fsqrt_ru(x);
}

/*******************************************************************************
*                                                                              *
* DDIV                                                                         *
*                                                                              *
*******************************************************************************/
static __forceinline__ double __ddiv_rn(double x, double y)
{
  return __nv_ddiv_rn(x, y);
}

static __forceinline__ double __ddiv_rz(double x, double y)
{
  return __nv_ddiv_rz(x, y);
}

static __forceinline__ double __ddiv_rd(double x, double y)
{
  return __nv_ddiv_rd(x, y);
}

static __forceinline__ double __ddiv_ru(double x, double y)
{
  return __nv_ddiv_ru(x, y);
}

/*******************************************************************************
*                                                                              *
* DRCP                                                                         *
*                                                                              *
*******************************************************************************/
static __forceinline__ double __drcp_rn(double x)
{
  return __nv_drcp_rn(x);
}

static __forceinline__ double __drcp_rz(double x)
{
  return __nv_drcp_rz(x);
}

static __forceinline__ double __drcp_rd(double x)
{
  return __nv_drcp_rd(x);
}

static __forceinline__ double __drcp_ru(double x)
{
  return __nv_drcp_ru(x);
}

/*******************************************************************************
*                                                                              *
* DSQRT                                                                        *
*                                                                              *
*******************************************************************************/
static __forceinline__ double __dsqrt_rn(double x)
{
  return __nv_dsqrt_rn(x);
}

static __forceinline__ double __dsqrt_rz(double x)
{
  return __nv_dsqrt_rz(x);
}

static __forceinline__ double __dsqrt_rd(double x)
{
  return __nv_dsqrt_rd(x);
}

static __forceinline__ double __dsqrt_ru(double x)
{
  return __nv_dsqrt_ru(x);
}

static __forceinline__ float sqrtf(float x)
{
  return __nv_sqrtf(x);
}

static __forceinline__ double sqrt(double x)
{
  return __nv_sqrt(x);
}

/*******************************************************************************
*                                                                              *
* DADD                                                                         *
*                                                                              *
*******************************************************************************/
static __forceinline__ double __dadd_rn(double x, double y)
{
  return __nv_dadd_rn(x, y);
}

static __forceinline__ double __dadd_rz(double x, double y)
{
  return __nv_dadd_rz(x, y);
}

static __forceinline__ double __dadd_rd(double x, double y)
{
  return __nv_dadd_rd(x, y);
}

static __forceinline__ double __dadd_ru(double x, double y)
{
  return __nv_dadd_ru(x, y);
}

/*******************************************************************************
*                                                                              *
* DMUL                                                                         *
*                                                                              *
*******************************************************************************/
static __forceinline__ double __dmul_rn(double x, double y)
{
  return __nv_dmul_rn(x, y);
}

static __forceinline__ double __dmul_rz(double x, double y)
{
  return __nv_dmul_rz(x, y);
}

static __forceinline__ double __dmul_rd(double x, double y)
{
  return __nv_dmul_rd(x, y);
}

static __forceinline__ double __dmul_ru(double x, double y)
{
  return __nv_dmul_ru(x, y);
}

/*******************************************************************************
*                                                                              *
* FADD                                                                         *
*                                                                              *
*******************************************************************************/
static __forceinline__ float __fadd_rd(float x, float y)
{
  return __nv_fadd_rd(x, y);
}

static __forceinline__ float __fadd_ru(float x, float y)
{
  return __nv_fadd_ru(x, y);
}

static __forceinline__ float __fadd_rn(float x, float y)
{
  return __nv_fadd_rn(x, y);
}

static __forceinline__ float __fadd_rz(float x, float y)
{
  return __nv_fadd_rz(x, y);
}

/*******************************************************************************
*                                                                              *
* FMUL                                                                         *
*                                                                              *
*******************************************************************************/
static __forceinline__ float __fmul_rd(float x, float y)
{
  return __nv_fmul_rd(x, y);
}

static __forceinline__ float __fmul_ru(float x, float y)
{
  return __nv_fmul_ru(x, y);
}

static __forceinline__ float __fmul_rn(float x, float y)
{
  return __nv_fmul_rn(x, y);
}

static __forceinline__ float __fmul_rz(float x, float y)
{
  return __nv_fmul_rz(x, y);
}

/*******************************************************************************
*                                                                              *
* CONVERSION FUNCTIONS                                                         *
*                                                                              *
*******************************************************************************/
/* double to float */
static __forceinline__ float __double2float_rn(double d)
{
  return __nv_double2float_rn(d);
}

static __forceinline__ float __double2float_rz(double d)
{
  return __nv_double2float_rz(d);
}

static __forceinline__ float __double2float_rd(double d)
{
  return __nv_double2float_rd(d);
}

static __forceinline__ float __double2float_ru(double d)
{
  return __nv_double2float_ru(d);
}

/* double to int */
static __forceinline__ int __double2int_rn(double d)
{
  return __nv_double2int_rn(d);
}

static __forceinline__ int __double2int_rz(double d)
{
  return __nv_double2int_rz(d);
}

static __forceinline__ int __double2int_rd(double d)
{
  return __nv_double2int_rd(d);
}

static __forceinline__ int __double2int_ru(double d)
{
  return __nv_double2int_ru(d);
}

/* double to uint */
static __forceinline__ unsigned int __double2uint_rn(double d)
{
  return __nv_double2uint_rn(d);
}

static __forceinline__ unsigned int __double2uint_rz(double d)
{
  return __nv_double2uint_rz(d);
}

static __forceinline__ unsigned int __double2uint_rd(double d)
{
  return __nv_double2uint_rd(d);
}

static __forceinline__ unsigned int __double2uint_ru(double d)
{
  return __nv_double2uint_ru(d);
}

/* int to double */
static __forceinline__ double __int2double_rn(int i)
{
  return __nv_int2double_rn(i);
}

/* uint to double */
static __forceinline__ double __uint2double_rn(unsigned int i)
{
  return __nv_uint2double_rn(i);
}

/* float to int */
static __forceinline__ int __float2int_rn(float in)
{
  return __nv_float2int_rn(in);
}

static __forceinline__ int __float2int_rz(float in)
{
  return __nv_float2int_rz(in);
}

static __forceinline__ int __float2int_rd(float in)
{
  return __nv_float2int_rd(in);
}

static __forceinline__ int __float2int_ru(float in)
{
  return __nv_float2int_ru(in);
}

/* float to uint */
static __forceinline__ unsigned int __float2uint_rn(float in)
{
  return __nv_float2uint_rn(in);
}

static __forceinline__ unsigned int __float2uint_rz(float in)
{
  return __nv_float2uint_rz(in);
}

static __forceinline__ unsigned int __float2uint_rd(float in)
{
  return __nv_float2uint_rd(in);
}

static __forceinline__ unsigned int __float2uint_ru(float in)
{
  return __nv_float2uint_ru(in);
}

/* int to float */
static __forceinline__ float __int2float_rn(int in)
{
  return __nv_int2float_rn(in);
}

static __forceinline__ float __int2float_rz(int in)
{
  return __nv_int2float_rz(in);
}

static __forceinline__ float __int2float_rd(int in)
{
  return __nv_int2float_rd(in);
}

static __forceinline__ float __int2float_ru(int in)
{
  return __nv_int2float_ru(in);
}

/* unsigned int to float */
static __forceinline__ float __uint2float_rn(unsigned int in)
{
  return __nv_uint2float_rn(in);
}

static __forceinline__ float __uint2float_rz(unsigned int in)
{
  return __nv_uint2float_rz(in);
}

static __forceinline__ float __uint2float_rd(unsigned int in)
{
  return __nv_uint2float_rd(in);
}

static __forceinline__ float __uint2float_ru(unsigned int in)
{
  return __nv_uint2float_ru(in);
}

/* hiloint vs double */
static __forceinline__ double __hiloint2double(int a, int b)
{
  return __nv_hiloint2double(a, b);
}

static __forceinline__ int __double2loint(double d)
{
  return __nv_double2loint(d);
}

static __forceinline__ int __double2hiint(double d)
{
  return __nv_double2hiint(d);
}

/* float to long long */
static __forceinline__ long long __float2ll_rn(float f)
{
  return __nv_float2ll_rn(f);
}

static __forceinline__ long long __float2ll_rz(float f)
{
  return __nv_float2ll_rz(f);
}

static __forceinline__ long long __float2ll_rd(float f)
{
  return __nv_float2ll_rd(f);
}

static __forceinline__ long long __float2ll_ru(float f)
{
  return __nv_float2ll_ru(f);
}

/* float to unsigned long long */
static __forceinline__ unsigned long long __float2ull_rn(float f)
{
  return __nv_float2ull_rn(f);
}

static __forceinline__ unsigned long long __float2ull_rz(float f)
{
  return __nv_float2ull_rz(f);
}

static __forceinline__ unsigned long long __float2ull_rd(float f)
{
  return __nv_float2ull_rd(f);
}

static __forceinline__ unsigned long long __float2ull_ru(float f)
{
  return __nv_float2ull_ru(f);
}

/* double to long long */
static __forceinline__ long long __double2ll_rn(double f)
{
  return __nv_double2ll_rn(f);
}

static __forceinline__ long long __double2ll_rz(double f)
{
  return __nv_double2ll_rz(f);
}

static __forceinline__ long long __double2ll_rd(double f)
{
  return __nv_double2ll_rd(f);
}

static __forceinline__ long long __double2ll_ru(double f)
{
  return __nv_double2ll_ru(f);
}

/* double to unsigned long long */
static __forceinline__ unsigned long long __double2ull_rn(double f)
{
  return __nv_double2ull_rn(f);
}

static __forceinline__ unsigned long long __double2ull_rz(double f)
{
  return __nv_double2ull_rz(f);
}

static __forceinline__ unsigned long long __double2ull_rd(double f)
{
  return __nv_double2ull_rd(f);
}

static __forceinline__ unsigned long long __double2ull_ru(double f)
{
  return __nv_double2ull_ru(f);
}

/* long long to float */
static __forceinline__ float __ll2float_rn(long long l)
{
  return __nv_ll2float_rn(l);
}

static __forceinline__ float __ll2float_rz(long long l)
{
  return __nv_ll2float_rz(l);
}

static __forceinline__ float __ll2float_rd(long long l)
{
  return __nv_ll2float_rd(l);
}

static __forceinline__ float __ll2float_ru(long long l)
{
  return __nv_ll2float_ru(l);
}

/* unsigned long long to float */
static __forceinline__ float __ull2float_rn(unsigned long long l)
{
  return __nv_ull2float_rn(l);
}

static __forceinline__ float __ull2float_rz(unsigned long long l)
{
  return __nv_ull2float_rz(l);
}

static __forceinline__ float __ull2float_rd(unsigned long long l)
{
  return __nv_ull2float_rd(l);
}

static __forceinline__ float __ull2float_ru(unsigned long long l)
{
  return __nv_ull2float_ru(l);
}

/* long long to double */
static __forceinline__ double __ll2double_rn(long long l)
{
  return __nv_ll2double_rn(l);
}

static __forceinline__ double __ll2double_rz(long long l)
{
  return __nv_ll2double_rz(l);
}

static __forceinline__ double __ll2double_rd(long long l)
{
  return __nv_ll2double_rd(l);
}

static __forceinline__ double __ll2double_ru(long long l)
{
  return __nv_ll2double_ru(l);
}

/* unsigned long long to double */
static __forceinline__ double __ull2double_rn(unsigned long long l)
{
  return __nv_ull2double_rn(l);
}

static __forceinline__ double __ull2double_rz(unsigned long long l)
{
  return __nv_ull2double_rz(l);
}

static __forceinline__ double __ull2double_rd(unsigned long long l)
{
  return __nv_ull2double_rd(l);
}

static __forceinline__ double __ull2double_ru(unsigned long long l)
{
  return __nv_ull2double_ru(l);
}

static __forceinline__ unsigned short __float2half_rn(float f)
{
  return __nv_float2half_rn(f);
}

static __forceinline__ float __half2float(unsigned short h)
{
  return __nv_half2float(h);
}

static __forceinline__ float __int_as_float(int x)
{
  return __nv_int_as_float(x);
}

static __forceinline__ int __float_as_int(float x)
{
  return __nv_float_as_int(x);
}

static __forceinline__ double __longlong_as_double(long long x)
{
  return __nv_longlong_as_double(x);
}

static __forceinline__ long long __double_as_longlong(double x)
{
  return __nv_double_as_longlong(x);
}

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPERATIONS        *
*                                                                              *
*******************************************************************************/

static __forceinline__ float __sinf(float a)
{
  return __nv_fast_sinf(a);
}

static __forceinline__ float __cosf(float a)
{
  return __nv_fast_cosf(a);
}

static __forceinline__ float __log2f(float a)
{
  return __nv_fast_log2f(a);
}

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITHOUT BUILTIN NVOPENCC OPERATIONS     *
*                                                                              *
*******************************************************************************/

static __forceinline__ float __tanf(float a)
{
  return __nv_fast_tanf(a);
}

static __forceinline__ void __sincosf(float a, float *sptr, float *cptr)
{
  __nv_fast_sincosf(a, sptr, cptr);
}

static __forceinline__ float __expf(float a)
{
  return __nv_fast_expf(a);
}

static __forceinline__ float __exp10f(float a)
{
  return __nv_fast_exp10f(a);
}

static __forceinline__ float __log10f(float a)
{
  return __nv_fast_log10f(a);
}

static __forceinline__ float __logf(float a)
{
  return __nv_fast_logf(a);
}

static __forceinline__ float __powf(float a, float b)
{
  return __nv_fast_powf(a, b);
}

static __forceinline__ float fdividef(float a, float b)
{
#if defined(__USE_FAST_MATH__) && !defined(__CUDA_PREC_DIV)
  return __nv_fast_fdividef(a, b);
#else /* __USE_FAST_MATH__ && !__CUDA_PREC_DIV */
  return a / b;
#endif /* __USE_FAST_MATH__ && !__CUDA_PREC_DIV */
}

static __forceinline__ double fdivide(double a, double b)
{
  return a / b;
}

/*
  According to Boolean algebra:

       (a | b) = (a & b) + (a ^ b)
  <==> (a & b) = (a | b) - (a ^ b)

  When adding a + b, a & b represents the carry bit vector, while a ^ b
  represents the sum bit vector. Thus:

  a + b = 2 * (a & b) + (a ^ b)               (I)

        = 2 * ((a | b) - (a ^ b)) + (a ^ b)

        = 2 * (a | b) - (a ^ b)               (II)

  Compare HAKMEM #23 at http://home.pipeline.com/~hbaker1/hakmem/hakmem.html

  hadd(a, b) is (a + b) / 2 rounded to negative infinity, whereas rhadd(a, b)
  is (a + b) / 2 rounded to positive infinity. The two terms 2 * (a & b) and
  2 * (a | b) are even, so don't influence the rounding when dividing by two.
  So the rounding must be via the sum bit term. Computing (a ^ b) / 2 by right
  shifting rounds this term to negative infinity. This means we need to base
  hadd() on formula (I), but rhadd() on formula(II). This results in

  hadd(a,b)  = (a & b) + ((a ^ b) >> 1)
  rhadd(a,b) = (a | b) - ((a ^ b) >> 1)
*/
static __forceinline__ int __hadd(int a, int b)
{
  return __nv_hadd(a, b);
}

static __forceinline__ int __rhadd(int a, int b)
{
  return __nv_rhadd(a, b);
}

static __forceinline__ unsigned int __uhadd(unsigned int a, unsigned int b)
{
  return __nv_uhadd(a, b);
}

static __forceinline__ unsigned int __urhadd(unsigned int a, unsigned int b)
{
  return __nv_urhadd(a, b);
}

static __forceinline__ float __fsub_rn(float a, float b)
{
  return __nv_fsub_rn(a, b);
}

static __forceinline__ float __fsub_rz(float a, float b)
{
  return __nv_fsub_rz(a, b);
}

static __forceinline__ float __fsub_rd(float a, float b)
{
  return __nv_fsub_rd(a, b);
}

static __forceinline__ float __fsub_ru(float a, float b)
{
  return __nv_fsub_ru(a, b);
}

static __forceinline__ float __frsqrt_rn(float a)
{
  return __nv_frsqrt_rn(a);
}

static __forceinline__ int __ffs(int a)
{
  return __nv_ffs(a);
}

static __forceinline__ int __ffsll(long long int a)
{
  return __nv_ffsll(a);
}

/*******************************************************************************
*                                                                              *
* ATOMIC OPERATIONS                                                            *
*                                                                              *
*******************************************************************************/
static __forceinline__ int __iAtomicAdd(int *p, int val)
{
  return __nvvm_atom_add_gen_i((volatile int *)p, val);
}

static __forceinline__ unsigned int
__uAtomicAdd(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_add_gen_i((volatile int *)p, (int)val);
}

static __forceinline__ unsigned long long
__ullAtomicAdd(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_add_gen_ll((volatile long long *)p, (long long)val);
}

static __forceinline__ float __fAtomicAdd(float *p, float val)
{
  return __nvvm_atom_add_gen_f((volatile float *)p, val);
}

static __forceinline__ int __iAtomicExch(int *p, int val)
{
  return __nvvm_atom_xchg_gen_i((volatile int *)p, val);
}

static __forceinline__ unsigned int
__uAtomicExch(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_xchg_gen_i((volatile int *)p, (int)val);
}

static __forceinline__ unsigned long long
__ullAtomicExch(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_xchg_gen_ll((volatile long long *)p, (long long)val);
}

static __forceinline__ float __fAtomicExch(float *p, float val)
{
  int old = __nvvm_atom_xchg_gen_i((volatile int *)p, __float_as_int(val));
  return __int_as_float(old);
}

static __forceinline__ int __iAtomicMin(int *p, int val)
{
  return __nvvm_atom_min_gen_i((volatile int *)p, val);
}

static __forceinline__ long long __illAtomicMin(long long *p, long long val)
{
  return __nvvm_atom_min_gen_ll((volatile long long *)p, val);
}

static __forceinline__ unsigned int
__uAtomicMin(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_min_gen_ui((volatile unsigned int *)p, val);
}

static __forceinline__ unsigned long long
__ullAtomicMin(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_min_gen_ull((volatile unsigned long long *)p, val);
}

static __forceinline__ int __iAtomicMax(int *p, int val)
{
  return __nvvm_atom_max_gen_i((volatile int *)p, val);
}

static __forceinline__ long long __illAtomicMax(long long *p, long long val)
{
  return __nvvm_atom_max_gen_ll((volatile long long *)p, val);
}

static __forceinline__ unsigned int
__uAtomicMax(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_max_gen_ui((unsigned int *)p, val);
}

static __forceinline__ unsigned long long
__ullAtomicMax(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_max_gen_ull((volatile unsigned long long *)p, val);
}

static __forceinline__ unsigned int
__uAtomicInc(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_inc_gen_ui((unsigned int *)p, val);
}

static __forceinline__ unsigned int
__uAtomicDec(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_dec_gen_ui((unsigned int *)p, val);
}

static __forceinline__ int __iAtomicCAS(int *p, int compare, int val)
{
  return __nvvm_atom_cas_gen_i((int *)p, compare, val);
}

static __forceinline__ unsigned int
__uAtomicCAS(unsigned int *p, unsigned int compare, unsigned int val)
{
  return (unsigned int)__nvvm_atom_cas_gen_i(
    (volatile int *)p, (int)compare, (int)val);
}

static __forceinline__ unsigned long long int __ullAtomicCAS(
  unsigned long long int *p,
  unsigned long long int compare,
  unsigned long long int val)
{
  return (unsigned long long int)__nvvm_atom_cas_gen_ll(
    (volatile long long int *)p, (long long int)compare, (long long int)val);
}

static __forceinline__ int __iAtomicAnd(int *p, int val)
{
  return __nvvm_atom_and_gen_i((volatile int *)p, val);
}

static __forceinline__ long long int
__llAtomicAnd(long long int *p, long long int val)
{
  return __nvvm_atom_and_gen_ll((volatile long long int *)p, (long long)val);
}

static __forceinline__ unsigned int
__uAtomicAnd(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_and_gen_i((volatile int *)p, (int)val);
}

static __forceinline__ unsigned long long int
__ullAtomicAnd(unsigned long long int *p, unsigned long long int val)
{
  return __nvvm_atom_and_gen_ll((volatile long long int *)p, (long long)val);
}

static __forceinline__ int __iAtomicOr(int *p, int val)
{
  return __nvvm_atom_or_gen_i((volatile int *)p, val);
}

static __forceinline__ long long int
__llAtomicOr(long long int *p, long long int val)
{
  return __nvvm_atom_or_gen_ll((volatile long long int *)p, (long long)val);
}

static __forceinline__ unsigned int
__uAtomicOr(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_or_gen_i((volatile int *)p, (int)val);
}

static __forceinline__ unsigned long long int
__ullAtomicOr(unsigned long long int *p, unsigned long long int val)
{
  return __nvvm_atom_or_gen_ll((volatile long long int *)p, (long long)val);
}

static __forceinline__ int __iAtomicXor(int *p, int val)
{
  return __nvvm_atom_xor_gen_i((volatile int *)p, val);
}

static __forceinline__ long long int
__llAtomicXor(long long int *p, long long int val)
{
  return __nvvm_atom_xor_gen_ll((volatile long long int *)p, (long long)val);
}

static __forceinline__ unsigned int
__uAtomicXor(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_xor_gen_i((volatile int *)p, (int)val);
}

static __forceinline__ unsigned long long int
__ullAtomicXor(unsigned long long int *p, unsigned long long int val)
{
  return __nvvm_atom_xor_gen_ll((volatile long long int *)p, (long long)val);
}

#else /* __CUDANVVM__ */

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITH BUILTIN NVOPENCC OPERATIONS        *
*                                                                              *
*******************************************************************************/

static __forceinline__ float __sinf(float a)
{
  return __builtin_sinf(a);
}

static __forceinline__ float __cosf(float a)
{
  return __builtin_cosf(a);
}

static __forceinline__ float __log2f(float a)
{
  return __builtin_log2f(a);
}

/*******************************************************************************
*                                                                              *
* DEVICE IMPLEMENTATIONS FOR FUNCTIONS WITHOUT BUILTIN NVOPENCC OPERATIONS     *
*                                                                              *
*******************************************************************************/

static __forceinline__ float __tanf(float a)
{
  return __fdividef(__sinf(a), __cosf(a));
}

static __forceinline__ void __sincosf(float a, float *sptr, float *cptr)
{
  *sptr = __sinf(a);
  *cptr = __cosf(a);
}

static __forceinline__ float __expf(float a)
{
  return exp2f(a * CUDART_L2E_F);
}

static __forceinline__ float __exp10f(float a)
{
  return exp2f(a * CUDART_L2T_F);
}

static __forceinline__ float __log10f(float a)
{
  return CUDART_LG2_F * __log2f(a);
}

static __forceinline__ float __logf(float a)
{
  return CUDART_LN2_F * __log2f(a);
}

static __forceinline__ float __powf(float a, float b)
{
  return exp2f(b * __log2f(a));
}

static __forceinline__ float fdividef(float a, float b)
{
#if defined(__USE_FAST_MATH__) && !defined(__CUDA_PREC_DIV)
  return __fdividef(a, b);
#else /* __USE_FAST_MATH__ && !__CUDA_PREC_DIV */
  return a / b;
#endif /* __USE_FAST_MATH__ && !__CUDA_PREC_DIV */
}

#if defined(CUDA_FLOAT_MATH_FUNCTIONS)

static __forceinline__ double fdivide(double a, double b)
{
  return (double)fdividef((float)a, (float)b);
}

#endif /* CUDA_FLOAT_MATH_FUNCTIONS */

#if defined(CUDA_DOUBLE_MATH_FUNCTIONS)

static __forceinline__ double fdivide(double a, double b)
{
  return a / b;
}

#endif /* CUDA_DOUBLE_MATH_FUNCTIONS */

/*
  According to Boolean algebra:

       (a | b) = (a & b) + (a ^ b)
  <==> (a & b) = (a | b) - (a ^ b)

  When adding a + b, a & b represents the carry bit vector, while a ^ b
  represents the sum bit vector. Thus:

  a + b = 2 * (a & b) + (a ^ b)               (I)

        = 2 * ((a | b) - (a ^ b)) + (a ^ b)

        = 2 * (a | b) - (a ^ b)               (II)

  Compare HAKMEM #23 at http://home.pipeline.com/~hbaker1/hakmem/hakmem.html

  hadd(a, b) is (a + b) / 2 rounded to negative infinity, whereas rhadd(a, b)
  is (a + b) / 2 rounded to positive infinity. The two terms 2 * (a & b) and
  2 * (a | b) are even, so don't influence the rounding when dividing by two.
  So the rounding must be via the sum bit term. Computing (a ^ b) / 2 by right
  shifting rounds this term to negative infinity. This means we need to base
  hadd() on formula (I), but rhadd() on formula(II). This results in

  hadd(a,b)  = (a & b) + ((a ^ b) >> 1)
  rhadd(a,b) = (a | b) - ((a ^ b) >> 1)
*/
static __forceinline__ int __hadd(int a, int b)
{
  return (a & b) + ((a ^ b) >> 1);
}

static __forceinline__ int __rhadd(int a, int b)
{
  return (a | b) - ((a ^ b) >> 1);
}

static __forceinline__ unsigned int __uhadd(unsigned int a, unsigned int b)
{
  return (a & b) + ((a ^ b) >> 1);
}

static __forceinline__ unsigned int __urhadd(unsigned int a, unsigned int b)
{
  return (a | b) - ((a ^ b) >> 1);
}

static __forceinline__ float __fsub_rn(float a, float b)
{
  float res;
#if defined(__CUDA_FTZ)
  asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
#else
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
#endif
  return res;
}

static __forceinline__ float __fsub_rz(float a, float b)
{
  float res;
#if defined(__CUDA_FTZ)
  asm("sub.rz.ftz.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
#else
  asm("sub.rz.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
#endif
  return res;
}

static __forceinline__ float __fsub_rd(float a, float b)
{
#if __CUDA_ARCH__ >= 200
  float res;
#if defined(__CUDA_FTZ)
  asm("sub.rm.ftz.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
#else
  asm("sub.rm.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
#endif
  return res;
#else /* __CUDA_ARCH__ >= 200 */
  unsigned int expo_x, expo_y;
  unsigned int xxi, yyi, temp;

  xxi = __float_as_int(a);
  yyi = __float_as_int(b);
  yyi = yyi ^ 0x80000000;

  /* make bigger operand the augend */
  expo_y = yyi << 1;
  if(expo_y > (xxi << 1))
  {
    expo_y = xxi;
    xxi = yyi;
    yyi = expo_y;
  }

  temp = 0xff;
  expo_x = temp & (xxi >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yyi >> 23);
  expo_y = expo_y - 1;

  if((expo_x <= 0xFD) && (expo_y <= 0xFD))
  {
    expo_y = expo_x - expo_y;
    if(expo_y > 25)
    {
      expo_y = 31;
    }
    temp = xxi ^ yyi;
    xxi = xxi & ~0x7f000000;
    xxi = xxi | 0x00800000;
    yyi = yyi & ~0xff000000;
    yyi = yyi | 0x00800000;

    if((int)temp < 0)
    {
      /* signs differ, effective subtraction */
      temp = 32 - expo_y;
      temp = (expo_y) ? (yyi << temp) : 0;
      temp = (unsigned int)(-((int)temp));
      xxi = xxi - (yyi >> expo_y) - (temp ? 1 : 0);
      if(xxi & 0x00800000)
      {
        if(expo_x <= 0xFD)
        {
          xxi = xxi & ~0x00800000; /* lop off integer bit */
          xxi = (xxi + (expo_x << 23)) + 0x00800000;
          xxi += (temp && (xxi & 0x80000000));
          return __int_as_float(xxi);
        }
      }
      else
      {
        if((temp | (xxi << 1)) == 0)
        {
          /* operands cancelled, resulting in a clean zero */
          xxi = 0x80000000;
          return __int_as_float(xxi);
        }
        /* normalize result */
        yyi = xxi & 0x80000000;
        do
        {
          xxi = (xxi << 1) | (temp >> 31);
          temp <<= 1;
          expo_x--;
        } while(!(xxi & 0x00800000));
        xxi = xxi | yyi;
      }
    }
    else
    {
      /* signs are the same, effective addition */
      temp = 32 - expo_y;
      temp = (expo_y) ? (yyi << temp) : 0;
      xxi = xxi + (yyi >> expo_y);
      if(!(xxi & 0x01000000))
      {
        if(expo_x <= 0xFD)
        {
          expo_y = xxi & 1;
          xxi = xxi + (expo_x << 23);
          xxi += (temp && (xxi & 0x80000000));
          return __int_as_float(xxi);
        }
      }
      else
      {
        /* normalize result */
        temp = (xxi << 31) | (temp >> 1);
        xxi = ((xxi & 0x80000000) | (xxi >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
    if(expo_x <= 0xFD)
    {
      xxi += (temp && (xxi & 0x80000000));
      xxi = xxi + (expo_x << 23);
      return __int_as_float(xxi);
    }
    if((int)expo_x >= 254)
    {
      /* overflow: return infinity or largest normal */
      temp = xxi & 0x80000000;
      xxi = (temp ? 0xFF800000 : 0x7f7fffff);
      return __int_as_float(xxi);
    }
    /* underflow: zero */
    xxi = xxi & 0x80000000;
    return __int_as_float(xxi);
  }
  else
  {
    a = a - b;
    xxi = xxi ^ yyi;
    if((a == 0.0f) && ((int)xxi < 0))
      a = __int_as_float(0x80000000);
    return a;
  }
#endif /* __CUDA_ARCH__ >= 200 */
}

static __forceinline__ float __fsub_ru(float a, float b)
{
#if __CUDA_ARCH__ >= 200
  float res;
#if defined(__CUDA_FTZ)
  asm("sub.rp.ftz.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
#else
  asm("sub.rp.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
#endif
  return res;
#else /* __CUDA_ARCH__ >= 200 */
  unsigned int expo_x, expo_y;
  unsigned int xxi, yyi, temp;

  xxi = __float_as_int(a);
  yyi = __float_as_int(b);
  yyi = yyi ^ 0x80000000;

  /* make bigger operand the augend */
  expo_y = yyi << 1;
  if(expo_y > (xxi << 1))
  {
    expo_y = xxi;
    xxi = yyi;
    yyi = expo_y;
  }

  temp = 0xff;
  expo_x = temp & (xxi >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yyi >> 23);
  expo_y = expo_y - 1;

  if((expo_x <= 0xFD) && (expo_y <= 0xFD))
  {
    expo_y = expo_x - expo_y;
    if(expo_y > 25)
    {
      expo_y = 31;
    }
    temp = xxi ^ yyi;
    xxi = xxi & ~0x7f000000;
    xxi = xxi | 0x00800000;
    yyi = yyi & ~0xff000000;
    yyi = yyi | 0x00800000;

    if((int)temp < 0)
    {
      /* signs differ, effective subtraction */
      temp = 32 - expo_y;
      temp = (expo_y) ? (yyi << temp) : 0;
      temp = (unsigned int)(-((int)temp));
      xxi = xxi - (yyi >> expo_y) - (temp ? 1 : 0);
      if(xxi & 0x00800000)
      {
        if(expo_x <= 0xFD)
        {
          xxi = (xxi + (expo_x << 23));
          xxi += (temp && !(xxi & 0x80000000));
          return __int_as_float(xxi);
        }
      }
      else
      {
        if((temp | (xxi << 1)) == 0)
        {
          /* operands cancelled, resulting in a clean zero */
          xxi = 0;
          return __int_as_float(xxi);
        }
        /* normalize result */
        yyi = xxi & 0x80000000;
        do
        {
          xxi = (xxi << 1) | (temp >> 31);
          temp <<= 1;
          expo_x--;
        } while(!(xxi & 0x00800000));
        xxi = xxi | yyi;
      }
    }
    else
    {
      /* signs are the same, effective addition */
      temp = 32 - expo_y;
      temp = (expo_y) ? (yyi << temp) : 0;
      xxi = xxi + (yyi >> expo_y);
      if(!(xxi & 0x01000000))
      {
        if(expo_x <= 0xFD)
        {
          xxi = xxi + (expo_x << 23);
          xxi += (temp && !(xxi & 0x80000000));
          return __int_as_float(xxi);
        }
      }
      else
      {
        /* normalize result */
        temp = (xxi << 31) | (temp >> 1);
        xxi = ((xxi & 0x80000000) | (xxi >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
    if(expo_x <= 0xFD)
    {
      xxi += (temp && !(xxi & 0x80000000));
      xxi = xxi + (expo_x << 23);
      return __int_as_float(xxi);
    }
    if((int)expo_x >= 254)
    {
      /* overflow: return infinity or largest normal */
      temp = xxi & 0x80000000;
      xxi = (temp ? 0xff7fffff : 0x7F800000);
      return __int_as_float(xxi);
    }
    /* underflow: zero */
    xxi = xxi & 0x80000000;
    return __int_as_float(xxi);
  }
  else
  {
    return a - b;
  }
#endif /* __CUDA_ARCH__ >= 200 */
}

static __forceinline__ float __frsqrt_rn(float a)
{
#if __CUDA_ARCH__ >= 200
  float y, h, l, e;
  int i, t;

#if !defined(__CUDA_FTZ)
  float aa = fabsf(a);
  if(aa < 1.175494351e-38f)
    a = a * 16777216.0f; /* 2^24 */
#endif /* !defined(__CUDA_FTZ) */
  i = __float_as_int(a);
  if(((unsigned)i - 0x00800000U) < 0x7f000000U)
  {
    t = (__float_as_int(a) & 0x00ffffff) | 0x3f000000;
    a = __int_as_float(t);
    t = t - i;
#if !defined(__CUDA_FTZ)
    if(aa < 1.175494351e-38f)
      t = t + (24 << 23);
#endif /* !defined(__CUDA_FTZ) */
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a));
    h = y * y;
    l = __fmaf_rn(y, y, -h);
    e = __fmaf_rn(l, -a, __fmaf_rn(h, -a, 1.0f));
    /* Round as shown in Peter Markstein, "IA-64 and Elementary Functions" */
    y = __fmaf_rn(__fmaf_rn(0.375f, e, 0.5f), e * y, y);
    asm("shr.s32 %0, %0, 1;" : "+r"(t));
    a = __int_as_float(__float_as_int(y) + t);
  }
  else
  {
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(a) : "f"(a));
  }
#else /* __CUDA_ARCH__ >= 200 */
  unsigned int e, i;
  unsigned int s, x;
  unsigned long long int prod1, prod2;

  i = __float_as_int(a);
  if(((unsigned)i - 0x00800000U) < 0x7f000000U)
  {
    x = (i & 0x00ffffff) | 0x00800000;
    x = x << (7 - ((i >> 23) & 1));
    i = __float_as_int(rsqrtf(__int_as_float(i | 1)));
    e = (i & 0x7f800000) - 0x00800000;
    i = (i & 0x00ffffff) | 0x00800000;
    i = i << 6;
    s = __umulhi(i, i);
    s = 0x06000000 - __umulhi(x, s);
    i = __umulhi(s, i);
    s = 2 * i;
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod1) : "r"(s), "r"(x));
    prod1 = prod1 * (i + 1);
    if((long long int)(prod1 + x) < 0)
      i++;
    a = __int_as_float(i + e);
  }
  else
  {
    a = rsqrtf(a);
  }
#endif /* __CUDA_ARCH__ >= 200 */
  return a;
}

#if __CUDA_ARCH__ < 200

static __forceinline__ float __frcp_rn(float x)
{
  unsigned int expo;
  unsigned f, y;
  unsigned int argi;
  float t;

  argi = __float_as_int(x);
  expo = (argi >> 23);
  expo = expo & 0xff;
  f = expo - 1;
  if(f <= 0xFD)
  {
    y = (argi & 0x00ffffff) | 0x00800000;
    expo = (2 * 127) - expo - 2;
    t = 1.0f / x;
    argi = __float_as_int(t);
    argi = (argi & 0x00ffffff) | 0x00800000;
    if((int)expo >= 0)
    {
      /* compute remainder1 */
      f = __umul24(y, argi);
      /* remainder1 must be negative. Fix if neccessary */
      if((int)f > 0)
      {
        t = __int_as_float(__float_as_int(t) - 1);
        f -= y;
      }
      /* compute remainder2 */
      expo = f + y;
      /* round result based on which remainder is smaller in magnitude */
      f = (unsigned)(-(int)f);
      if(expo < f)
      {
        t = __int_as_float(__float_as_int(t) + 1);
      }
      return t;
    }
  }
  return 1.0f / x;
}

static __forceinline__ float __frcp_rz(float x)
{
  unsigned int expo;
  unsigned f, y;
  unsigned int argi;
  float t;

  argi = __float_as_int(x);
  expo = (argi >> 23);
  expo = expo & 0xff;
  f = expo - 1;
  if(f <= 0xFD)
  {
    y = (argi & 0x00ffffff) | 0x00800000;
    expo = (2 * 127) - expo - 2;
    t = 1.0f / x;
    argi = __float_as_int(t);
    argi = (argi & 0x00ffffff) | 0x00800000;
    if((int)expo >= 0)
    {
      f = __umul24(y, argi);
      if((int)f > 0)
      {
        t = __int_as_float(__float_as_int(t) - 1);
      }
      return t;
    }
  }
  return 1.0f / x;
}

static __forceinline__ float __frcp_rd(float x)
{
  unsigned int expo;
  unsigned f, y;
  unsigned int argi;
  float t;

  argi = __float_as_int(x);
  expo = (argi >> 23);
  expo = expo & 0xff;
  f = expo - 1;
  if(f <= 0xFD)
  {
    y = (argi & 0x00ffffff) | 0x00800000;
    expo = (2 * 127) - expo - 2;
    t = 1.0f / x;
    argi = __float_as_int(t);
    argi = (argi & 0x00ffffff) | 0x00800000;
    if((int)expo >= 0)
    {
      f = __umul24(y, argi);
      if(((int)f > 0) && (x > 0.0f))
      {
        t = __int_as_float(__float_as_int(t) - 1);
      }
      if(((int)f < 0) && (x < 0.0f))
      {
        t = __int_as_float(__float_as_int(t) + 1);
      }
      return t;
    }
  }
  return 1.0f / x;
}

static __forceinline__ float __frcp_ru(float x)
{
  unsigned int expo;
  unsigned f, y;
  unsigned int argi;
  float t;

  argi = __float_as_int(x);
  expo = (argi >> 23);
  expo = expo & 0xff;
  f = expo - 1;
  if(f <= 0xFD)
  {
    y = (argi & 0x00ffffff) | 0x00800000;
    expo = (2 * 127) - expo - 2;
    t = 1.0f / x;
    argi = __float_as_int(t);
    argi = (argi & 0x00ffffff) | 0x00800000;
    if((int)expo >= 0)
    {
      f = __umul24(y, argi);
      if(((int)f > 0) && (x < 0.0f))
      {
        t = __int_as_float(__float_as_int(t) - 1);
      }
      if(((int)f < 0) && (x > 0.0f))
      {
        t = __int_as_float(__float_as_int(t) + 1);
      }
      return t;
    }
  }
  return 1.0f / x;
}

static __forceinline__ float __fsqrt_rn(float radicand)
{
  unsigned int expo, argi;
  unsigned int f, x;

  argi = __float_as_int(radicand);
  expo = argi >> 23;
  f = expo - 1;

  if(f <= 0xFD)
  {
    x = (argi << 8) | 0x80000000;
    x = x >> (expo & 1);
    argi =
      (((__float_as_int(rsqrtf(__int_as_float(__float_as_int(radicand) | 1))) &
         0x00ffffff) |
        0x00800000)
       << 7);
    /* second NR iteration */
    f = __umulhi(argi, argi);
    f = 0x30000000 - __umulhi(x, f);
    argi = __umulhi(f, argi);
    /* compute sqrt_rn(x) as x * 1/sqrt_rn(x) */
    argi = __umulhi(x, argi);
    argi = argi >> 3;
    x = (x << 16) - (argi * argi);
    /* round to nearest based on remainder; tie case impossible */
    if((int)x > (int)argi)
      argi++;
    argi = argi + (((expo + 125) & ~0x1) << 22);
    return __int_as_float(argi);
  }
  return sqrtf(radicand);
}

static __forceinline__ float __fsqrt_rz(float radicand)
{
  unsigned int expo, argi;
  unsigned int s, f, x;

  argi = __float_as_int(radicand);
  expo = argi >> 23;
  f = expo - 1;

  if(f <= 0xFD)
  {
    x = (argi << 8) | 0x80000000;
    x = x >> (expo & 1);
    argi =
      (((__float_as_int(rsqrtf(__int_as_float(__float_as_int(radicand) | 1))) &
         0x00ffffff) |
        0x00800000)
       << 7);
    /* NR iteration */
    s = __umulhi(argi, argi);
    f = 0x30000000 - __umulhi(x, s);
    argi = __umulhi(f, argi);
    /* compute sqrt_rz(x) as x * 1/sqrt_rz(x) */
    argi = __umulhi(x, argi);
    /* compute truncated result */
    argi = (argi + 4) >> 3;
    x = (x << 16) - (argi * argi);
    if((int)x < 0)
      argi--;
    argi = argi + (((expo + 125) & ~0x1) << 22);
    return __int_as_float(argi);
  }
  return sqrtf(radicand);
}

static __forceinline__ float __fsqrt_ru(float radicand)
{
  unsigned int expo, argi;
  unsigned int s, f, x;

  argi = __float_as_int(radicand);
  expo = argi >> 23;
  f = expo - 1;

  if(f <= 0xFD)
  {
    x = (argi << 8) | 0x80000000;
    x = x >> (expo & 1);
    argi =
      (((__float_as_int(rsqrtf(__int_as_float(__float_as_int(radicand) | 1))) &
         0x00ffffff) |
        0x00800000)
       << 7);
    /* NR iteration */
    s = __umulhi(argi, argi);
    f = 0x30000000 - __umulhi(x, s);
    argi = __umulhi(f, argi);
    /* compute sqrt_ru(x) as x * 1/sqrt_ru(x) */
    argi = __umulhi(x, argi);
    argi = (argi + 4) >> 3;
    x = (x << 16) - (argi * argi);
    if((int)x > 0)
      argi++;
    argi = argi + (((expo + 125) & ~0x1) << 22);
    return __int_as_float(argi);
  }
  return sqrtf(radicand);
}

static __forceinline__ float __fsqrt_rd(float radicand)
{
  unsigned int expo, argi;
  unsigned int s, f, x;

  argi = __float_as_int(radicand);
  expo = argi >> 23;
  f = expo - 1;

  if(f <= 0xFD)
  {
    x = (argi << 8) | 0x80000000;
    x = x >> (expo & 1);
    argi =
      (((__float_as_int(rsqrtf(__int_as_float(__float_as_int(radicand) | 1))) &
         0x00ffffff) |
        0x00800000)
       << 7);
    /* NR iteration */
    s = __umulhi(argi, argi);
    f = 0x30000000 - __umulhi(x, s);
    argi = __umulhi(f, argi);
    /* compute sqrt_rd(x) as x * 1/sqrt_rd(x) */
    argi = __umulhi(x, argi);
    /* compute truncated result */
    argi = (argi + 4) >> 3;
    x = (x << 16) - (argi * argi);
    if((int)x < 0)
      argi--;
    argi = argi + (((expo + 125) & ~0x1) << 22);
    return __int_as_float(argi);
  }
  return sqrtf(radicand);
}

static __forceinline__ float __fdiv_rn(float dividend, float divisor)
{
  unsigned long long prod;
  unsigned r, f, x, y, expox, expoy, sign;
  unsigned expo_res;
  unsigned resi, cvtxi, cvtyi;
  float t;

  cvtxi = __float_as_int(dividend);
  cvtyi = __float_as_int(divisor);
  expox = (cvtxi >> 23) & 0xff;
  expoy = (cvtyi >> 23) & 0xff;
  sign = ((cvtxi ^ cvtyi) & 0x80000000);

  if(((expox - 1) <= 0xFD) && ((expoy - 1) <= 0xFD))
  {
    expo_res = expox - expoy + 127 - 1;
    /* extract mantissas */
    y = (cvtyi << 8) | 0x80000000;
    x = (cvtxi & 0x00ffffff) | 0x00800000;
    t = __int_as_float((cvtyi & 0x00ffffff) | 0x3f800001);
    r = ((__float_as_int(1.0f / t) & 0x00ffffff) | 0x00800000) << 7;
    /* NR iteration */
    f = (unsigned)-(int)__umulhi(y, r << 1);
    r = __umulhi(f, r << 1);
    /* produce quotient */
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(x), "r"(r << 1));
    /* normalize mantissa */
    if(((int)((prod >> 32) << 8)) > 0)
    {
      expo_res--;
      prod = prod + prod;
    }
    /* preliminary mantissa */
    r = (unsigned)(prod >> 32);
    y = y >> 8;
    /* result is a normal */
    if(expo_res <= 0xFD)
    {
      int rem0, rem1, inc;
      /* round mantissa to nearest even */
      asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(y), "r"(r));
      x = x << (23 + ((prod >> 32) >> 15));
      rem1 = x - (unsigned)(prod & 0xffffffff);
      rem0 = rem1 - y;
      inc = abs(rem0) < abs(rem1);
      /* merge sign, mantissa, exponent for final result */
      resi = sign | ((expo_res << 23) + r + inc);
      return __int_as_float(resi);
    }
    else if((int)expo_res >= 254)
    {
      /* overflow: return infinity */
      resi = sign | 0x7f800000;
      return __int_as_float(resi);
    }
    else
    {
      /* underflow, may still round to normal */
      int rem0, rem1, inc;
      asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(y), "r"(r));
      x = x << (23 + ((prod >> 32) >> 15));
      rem1 = x - (unsigned)(prod & 0xffffffff);
      rem0 = rem1 - y;
      inc = abs(rem0) < abs(rem1);
      resi = ((expo_res << 23) + r + inc);
      if(resi != 0x00800000)
        resi = 0;
      return __int_as_float(sign | resi);
    }
  }
  if(fabsf(divisor) > CUDART_TWO_TO_126_F)
  {
    divisor *= 0.25f;
    dividend *= 0.25f;
  }
  return __fdividef(dividend, divisor);
}

static __forceinline__ float __fdiv_rz(float dividend, float divisor)
{
  unsigned long long prod;
  unsigned r, f, x, y, expox, expoy, sign;
  unsigned expo_res;
  unsigned resi, cvtxi, cvtyi;
  float t;

  cvtxi = __float_as_int(dividend);
  cvtyi = __float_as_int(divisor);
  expox = (cvtxi >> 23) & 0xff;
  expoy = (cvtyi >> 23) & 0xff;
  sign = ((cvtxi ^ cvtyi) & 0x80000000);

  if(((expox - 1) <= 0xFD) && ((expoy - 1) <= 0xFD))
  {
    expo_res = expox - expoy + 127 - 1;
    /* extract mantissas */
    y = (cvtyi << 8) | 0x80000000;
    x = (cvtxi & 0x00ffffff) | 0x00800000;
    t = __int_as_float((cvtyi & 0x00ffffff) | 0x3f800001);
    r = ((__float_as_int(1.0f / t) & 0x00ffffff) | 0x00800000) << 7;
    /* NR iteration */
    f = (unsigned)-(int)__umulhi(y, r << 1);
    r = __umulhi(f, r << 1);
    /* produce quotient */
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(x), "r"(r << 1));
    /* normalize mantissa */
    if(((int)((prod >> 32) << 8)) > 0)
    {
      expo_res--;
      prod = prod + prod;
    }
    /* preliminary mantissa */
    prod += 0x0000000080000000ULL;
    r = (unsigned)(prod >> 32);
    y = y >> 8;
    if(expo_res <= 0xFD)
    {
      /* result is a normal */
      int rem1;
      asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(y), "r"(r));
      x = x << (23 + ((prod >> 32) >> 15));
      rem1 = x - (unsigned)(prod & 0xffffffff);
      if(rem1 < 0)
        r--;
      resi = (expo_res << 23) + r;
      if(resi == 0x7f800000)
        resi = 0x7f7fffff;
      return __int_as_float(sign | resi);
    }
    else if((int)expo_res >= 254)
    {
      /* overflow: return largest normal */
      resi = 0x7f7fffff;
      return __int_as_float(sign | resi);
    }
    else
    {
      /* underflow: result is smallest normal or zero */
      int rem1;
      asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(y), "r"(r));
      x = x << (23 + ((prod >> 32) >> 15));
      rem1 = x - (unsigned)(prod & 0xffffffff);
      if(rem1 < 0)
        r--;
      resi = ((expo_res << 23) + r);
      if(resi != 0x00800000)
        resi = 0;
      return __int_as_float(sign | resi);
    }
  }
  if(fabsf(divisor) > CUDART_TWO_TO_126_F)
  {
    divisor *= 0.25f;
    dividend *= 0.25f;
  }
  return __fdividef(dividend, divisor);
}

static __forceinline__ float __fdiv_ru(float dividend, float divisor)
{
  unsigned long long prod;
  unsigned r, f, x, y, expox, expoy, sign;
  unsigned expo_res;
  unsigned resi, cvtxi, cvtyi;
  float t;

  cvtxi = __float_as_int(dividend);
  cvtyi = __float_as_int(divisor);
  expox = (cvtxi >> 23) & 0xff;
  expoy = (cvtyi >> 23) & 0xff;
  sign = ((cvtxi ^ cvtyi) & 0x80000000);

  if(((expox - 1) <= 0xFD) && ((expoy - 1) <= 0xFD))
  {
    expo_res = expox - expoy + 127 - 1;
    /* extract mantissas */
    y = (cvtyi << 8) | 0x80000000;
    x = (cvtxi & 0x00ffffff) | 0x00800000;
    t = __int_as_float((cvtyi & 0x00ffffff) | 0x3f800001);
    r = ((__float_as_int(1.0f / t) & 0x00ffffff) | 0x00800000) << 7;
    /* NR iteration */
    f = (unsigned)-(int)__umulhi(y, r << 1);
    r = __umulhi(f, r << 1);
    /* produce quotient */
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(x), "r"(r << 1));
    /* normalize mantissa */
    if(((int)((prod >> 32) << 8)) > 0)
    {
      expo_res--;
      prod = prod + prod;
    }
    /* preliminary mantissa */
    prod += 0x0000000080000000ULL;
    r = (unsigned)(prod >> 32);
    y = y >> 8;
    if(expo_res <= 0xFD)
    {
      /* result is a normal */
      int rem1;
      asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(y), "r"(r));
      x = x << (23 + ((prod >> 32) >> 15));
      rem1 = x - (unsigned)(prod & 0xffffffff);
      if((rem1 < 0) && (sign))
        r--;
      if((rem1 > 0) && (!sign))
        r++;
      resi = (expo_res << 23) + r;
      if((resi == 0x7f800000) && (sign))
        resi = 0x7f7fffff;
      return __int_as_float(sign | resi);
    }
    else if((int)expo_res >= 254)
    {
      /* overflow: return largest normal */
      resi = sign ? 0x7f7fffff : 0x7f800000;
      return __int_as_float(sign | resi);
    }
    else
    {
      /* underflow: result is smallest normal or zero */
      int rem1;
      asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(y), "r"(r));
      x = x << (23 + ((prod >> 32) >> 15));
      rem1 = x - (unsigned)(prod & 0xffffffff);
      if((rem1 < 0) && (sign))
        r--;
      if((rem1 > 0) && (!sign))
        r++;
      resi = ((expo_res << 23) + r);
      if(resi != 0x00800000)
        resi = 0;
      return __int_as_float(sign | resi);
    }
  }
  if(fabsf(divisor) > CUDART_TWO_TO_126_F)
  {
    divisor *= 0.25f;
    dividend *= 0.25f;
  }
  return __fdividef(dividend, divisor);
}

static __forceinline__ float __fdiv_rd(float dividend, float divisor)
{
  unsigned long long prod;
  unsigned r, f, x, y, expox, expoy, sign;
  unsigned expo_res;
  unsigned resi, cvtxi, cvtyi;
  float t;

  cvtxi = __float_as_int(dividend);
  cvtyi = __float_as_int(divisor);
  expox = (cvtxi >> 23) & 0xff;
  expoy = (cvtyi >> 23) & 0xff;
  sign = ((cvtxi ^ cvtyi) & 0x80000000);

  if(((expox - 1) <= 0xFD) && ((expoy - 1) <= 0xFD))
  {
    expo_res = expox - expoy + 127 - 1;
    /* extract mantissas */
    y = (cvtyi << 8) | 0x80000000;
    x = (cvtxi & 0x00ffffff) | 0x00800000;
    t = __int_as_float((cvtyi & 0x00ffffff) | 0x3f800001);
    r = ((__float_as_int(1.0f / t) & 0x00ffffff) | 0x00800000) << 7;
    /* NR iteration */
    f = (unsigned)-(int)__umulhi(y, r << 1);
    r = __umulhi(f, r << 1);
    /* produce quotient */
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(x), "r"(r << 1));
    /* normalize mantissa */
    if(((int)((prod >> 32) << 8)) > 0)
    {
      expo_res--;
      prod = prod + prod;
    }
    /* preliminary mantissa */
    prod += 0x0000000080000000ULL;
    r = (unsigned)(prod >> 32);
    y = y >> 8;
    if(expo_res <= 0xFD)
    {
      /* result is a normal */
      int rem1;
      asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(y), "r"(r));
      x = x << (23 + ((prod >> 32) >> 15));
      rem1 = x - (unsigned)(prod & 0xffffffff);
      if((rem1 < 0) && (!sign))
        r--;
      if((rem1 > 0) && (sign))
        r++;
      resi = (expo_res << 23) + r;
      if((resi == 0x7f800000) && (!sign))
        resi = 0x7f7fffff;
      return __int_as_float(sign | resi);
    }
    else if((int)expo_res >= 254)
    {
      /* overflow: return largest normal */
      resi = sign ? 0x7f800000 : 0x7f7fffff;
      return __int_as_float(sign | resi);
    }
    else
    {
      /* underflow: result is smallest normal or zero */
      int rem1;
      asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(y), "r"(r));
      x = x << (23 + ((prod >> 32) >> 15));
      rem1 = x - (unsigned)(prod & 0xffffffff);
      if((rem1 < 0) && (!sign))
        r--;
      if((rem1 > 0) && (sign))
        r++;
      resi = ((expo_res << 23) + r);
      if(resi != 0x00800000)
        resi = 0;
      return __int_as_float(sign | resi);
    }
  }
  if(fabsf(divisor) > CUDART_TWO_TO_126_F)
  {
    divisor *= 0.25f;
    dividend *= 0.25f;
  }
  return __fdividef(dividend, divisor);
}

static __forceinline__ float __fadd_ru(float a, float b)
{
  unsigned int expo_x, expo_y;
  unsigned int xxi, yyi, temp;

  xxi = __float_as_int(a);
  yyi = __float_as_int(b);

  /* make bigger operand the augend */
  expo_y = yyi << 1;
  if(expo_y > (xxi << 1))
  {
    expo_y = xxi;
    xxi = yyi;
    yyi = expo_y;
  }

  temp = 0xff;
  expo_x = temp & (xxi >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yyi >> 23);
  expo_y = expo_y - 1;

  if((expo_x <= 0xFD) && (expo_y <= 0xFD))
  {
    expo_y = expo_x - expo_y;
    if(expo_y > 25)
    {
      expo_y = 31;
    }
    temp = xxi ^ yyi;
    xxi = xxi & ~0x7f000000;
    xxi = xxi | 0x00800000;
    yyi = yyi & ~0xff000000;
    yyi = yyi | 0x00800000;

    if((int)temp < 0)
    {
      /* signs differ, effective subtraction */
      temp = 32 - expo_y;
      temp = (expo_y) ? (yyi << temp) : 0;
      temp = (unsigned int)(-((int)temp));
      xxi = xxi - (yyi >> expo_y) - (temp ? 1 : 0);
      if(xxi & 0x00800000)
      {
        if(expo_x <= 0xFD)
        {
          xxi = (xxi + (expo_x << 23));
          xxi += (temp && !(xxi & 0x80000000));
          return __int_as_float(xxi);
        }
      }
      else
      {
        if((temp | (xxi << 1)) == 0)
        {
          /* operands cancelled, resulting in a clean zero */
          xxi = 0;
          return __int_as_float(xxi);
        }
        /* normalize result */
        yyi = xxi & 0x80000000;
        do
        {
          xxi = (xxi << 1) | (temp >> 31);
          temp <<= 1;
          expo_x--;
        } while(!(xxi & 0x00800000));
        xxi = xxi | yyi;
      }
    }
    else
    {
      /* signs are the same, effective addition */
      temp = 32 - expo_y;
      temp = (expo_y) ? (yyi << temp) : 0;
      xxi = xxi + (yyi >> expo_y);
      if(!(xxi & 0x01000000))
      {
        if(expo_x <= 0xFD)
        {
          xxi = xxi + (expo_x << 23);
          xxi += (temp && !(xxi & 0x80000000));
          return __int_as_float(xxi);
        }
      }
      else
      {
        /* normalize result */
        temp = (xxi << 31) | (temp >> 1);
        xxi = ((xxi & 0x80000000) | (xxi >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
    if(expo_x <= 0xFD)
    {
      xxi += (temp && !(xxi & 0x80000000));
      xxi = xxi + (expo_x << 23);
      return __int_as_float(xxi);
    }
    if((int)expo_x >= 254)
    {
      /* overflow: return infinity or largest normal */
      temp = xxi & 0x80000000;
      xxi = (temp ? 0xff7fffff : 0x7F800000);
      return __int_as_float(xxi);
    }
    /* underflow: zero */
    xxi = xxi & 0x80000000;
    return __int_as_float(xxi);
  }
  else
  {
    return a + b;
  }
}

static __forceinline__ float __fadd_rd(float a, float b)
{
  unsigned int expo_x, expo_y;
  unsigned int xxi, yyi, temp;

  xxi = __float_as_int(a);
  yyi = __float_as_int(b);

  /* make bigger operand the augend */
  expo_y = yyi << 1;
  if(expo_y > (xxi << 1))
  {
    expo_y = xxi;
    xxi = yyi;
    yyi = expo_y;
  }

  temp = 0xff;
  expo_x = temp & (xxi >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yyi >> 23);
  expo_y = expo_y - 1;

  if((expo_x <= 0xFD) && (expo_y <= 0xFD))
  {
    expo_y = expo_x - expo_y;
    if(expo_y > 25)
    {
      expo_y = 31;
    }
    temp = xxi ^ yyi;
    xxi = xxi & ~0x7f000000;
    xxi = xxi | 0x00800000;
    yyi = yyi & ~0xff000000;
    yyi = yyi | 0x00800000;

    if((int)temp < 0)
    {
      /* signs differ, effective subtraction */
      temp = 32 - expo_y;
      temp = (expo_y) ? (yyi << temp) : 0;
      temp = (unsigned int)(-((int)temp));
      xxi = xxi - (yyi >> expo_y) - (temp ? 1 : 0);
      if(xxi & 0x00800000)
      {
        if(expo_x <= 0xFD)
        {
          xxi = xxi & ~0x00800000; /* lop off integer bit */
          xxi = (xxi + (expo_x << 23)) + 0x00800000;
          xxi += (temp && (xxi & 0x80000000));
          return __int_as_float(xxi);
        }
      }
      else
      {
        if((temp | (xxi << 1)) == 0)
        {
          /* operands cancelled, resulting in a clean zero */
          xxi = 0x80000000;
          return __int_as_float(xxi);
        }
        /* normalize result */
        yyi = xxi & 0x80000000;
        do
        {
          xxi = (xxi << 1) | (temp >> 31);
          temp <<= 1;
          expo_x--;
        } while(!(xxi & 0x00800000));
        xxi = xxi | yyi;
      }
    }
    else
    {
      /* signs are the same, effective addition */
      temp = 32 - expo_y;
      temp = (expo_y) ? (yyi << temp) : 0;
      xxi = xxi + (yyi >> expo_y);
      if(!(xxi & 0x01000000))
      {
        if(expo_x <= 0xFD)
        {
          expo_y = xxi & 1;
          xxi = xxi + (expo_x << 23);
          xxi += (temp && (xxi & 0x80000000));
          return __int_as_float(xxi);
        }
      }
      else
      {
        /* normalize result */
        temp = (xxi << 31) | (temp >> 1);
        xxi = ((xxi & 0x80000000) | (xxi >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
    if(expo_x <= 0xFD)
    {
      xxi += (temp && (xxi & 0x80000000));
      xxi = xxi + (expo_x << 23);
      return __int_as_float(xxi);
    }
    if((int)expo_x >= 254)
    {
      /* overflow: return infinity or largest normal */
      temp = xxi & 0x80000000;
      xxi = (temp ? 0xFF800000 : 0x7f7fffff);
      return __int_as_float(xxi);
    }
    /* underflow: zero */
    xxi = xxi & 0x80000000;
    return __int_as_float(xxi);
  }
  else
  {
    a = a + b;
    xxi = xxi ^ yyi;
    if((a == 0.0f) && ((int)xxi < 0))
      a = __int_as_float(0x80000000);
    return a;
  }
}

static __forceinline__ float __fmul_ru(float a, float b)
{
  unsigned long long product;
  unsigned int expo_x, expo_y;
  unsigned int xxi, yyi;

  xxi = __float_as_int(a);
  yyi = __float_as_int(b);

  expo_y = 0xFF;
  expo_x = expo_y & (xxi >> 23);
  expo_x = expo_x - 1;
  expo_y = expo_y & (yyi >> 23);
  expo_y = expo_y - 1;

  if((expo_x <= 0xFD) && (expo_y <= 0xFD))
  {
    expo_x = expo_x + expo_y;
    expo_y = xxi ^ yyi;
    xxi = xxi & 0x00ffffff;
    yyi = yyi << 8;
    xxi = xxi | 0x00800000;
    yyi = yyi | 0x80000000;
    /* compute product */
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(product) : "r"(xxi), "r"(yyi));
    expo_x = expo_x - 127 + 2;
    expo_y = expo_y & 0x80000000;
    xxi = (unsigned int)(product >> 32);
    yyi = (unsigned int)(product & 0xffffffff);
    /* normalize mantissa */
    if(xxi < 0x00800000)
    {
      xxi = (xxi << 1) | (yyi >> 31);
      yyi = (yyi << 1);
      expo_x--;
    }
    if(expo_x <= 0xFD)
    {
      xxi = xxi | expo_y;         /* OR in sign bit */
      xxi = xxi + (expo_x << 23); /* add in exponent */
      /* round result */
      xxi += (yyi && !expo_y);
      return __int_as_float(xxi);
    }
    else if((int)expo_x >= 254)
    {
      /* overflow: return infinity or largest normal */
      xxi = (expo_y ? 0xff7fffff : 0x7F800000);
      return __int_as_float(xxi);
    }
    else
    {
      /* underflow: zero, or smallest normal */
      expo_x = ((unsigned int)-((int)expo_x));
      xxi += (yyi && !expo_y);
      xxi = (xxi >> expo_x);
      if((expo_x > 25) || (xxi != 0x00800000))
        xxi = 0;
      return __int_as_float(expo_y | xxi);
    }
  }
  else
  {
    return a * b;
  }
}

static __forceinline__ float __fmul_rd(float a, float b)
{
  unsigned long long product;
  unsigned int expo_x, expo_y;
  unsigned int xxi, yyi;

  xxi = __float_as_int(a);
  yyi = __float_as_int(b);

  expo_y = 0xFF;
  expo_x = expo_y & (xxi >> 23);
  expo_x = expo_x - 1;
  expo_y = expo_y & (yyi >> 23);
  expo_y = expo_y - 1;

  if((expo_x <= 0xFD) && (expo_y <= 0xFD))
  {
    expo_x = expo_x + expo_y;
    expo_y = xxi ^ yyi;
    xxi = xxi & 0x00ffffff;
    yyi = yyi << 8;
    xxi = xxi | 0x00800000;
    yyi = yyi | 0x80000000;
    /* compute product */
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(product) : "r"(xxi), "r"(yyi));
    expo_x = expo_x - 127 + 2;
    expo_y = expo_y & 0x80000000;
    xxi = (unsigned int)(product >> 32);
    yyi = (unsigned int)(product & 0xffffffff);
    /* normalize mantissa */
    if(xxi < 0x00800000)
    {
      xxi = (xxi << 1) | (yyi >> 31);
      yyi = (yyi << 1);
      expo_x--;
    }
    if(expo_x <= 0xFD)
    {
      xxi = xxi | expo_y;         /* OR in sign bit */
      xxi = xxi + (expo_x << 23); /* add in exponent */
      /* round result */
      xxi += (yyi && expo_y);
      return __int_as_float(xxi);
    }
    else if((int)expo_x >= 254)
    {
      /* overflow: return infinity or largest normal */
      xxi = expo_y | (expo_y ? 0x7F800000 : 0x7f7fffff);
      return __int_as_float(xxi);
    }
    else
    {
      /* underflow: zero, or smallest normal */
      expo_x = ((unsigned int)-((int)expo_x));
      xxi += (yyi && expo_y);
      xxi = (xxi >> expo_x);
      if((expo_x > 25) || (xxi != 0x00800000))
        xxi = 0;
      return __int_as_float(expo_y | xxi);
    }
  }
  else
  {
    return a * b;
  }
}

static __forceinline__ float __fmaf_rn(float a, float b, float c)
{
  unsigned long long product;
  unsigned int xx, yy, zz, ww;
  unsigned int temp, s, u;
  unsigned int expo_x, expo_y, expo_z;

  xx = __float_as_int(a);
  yy = __float_as_int(b);
  zz = __float_as_int(c);

  /* Match 'denormals are zero' behavior of the GPU */
  if((xx << 1) < 0x01000000)
    xx &= 0x80000000;
  if((yy << 1) < 0x01000000)
    yy &= 0x80000000;
  if((zz << 1) < 0x01000000)
    zz &= 0x80000000;

  temp = 0xff;
  expo_x = temp & (xx >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yy >> 23);
  expo_y = expo_y - 1;
  expo_z = temp & (zz >> 23);
  expo_z = expo_z - 1;

  if(!((expo_x <= 0xFD) && (expo_y <= 0xFD) && (expo_z <= 0xFD)))
  {
    /* fmad (nan, y, z) --> nan
       fmad (x, nan, z) --> nan
       fmad (x, y, nan) --> nan
    */
    if((yy << 1) > 0xff000000)
    {
      return rsqrtf(b);
    }
    if((zz << 1) > 0xff000000)
    {
      return rsqrtf(c);
    }
    if((xx << 1) > 0xff000000)
    {
      return rsqrtf(a);
    }
    /* fmad (0, inf, z) --> NaN
       fmad (inf, 0, z) --> NaN
       fmad (-inf,+y,+inf) --> NaN
       fmad (+x,-inf,+inf) --> NaN
       fmad (+inf,-y,+inf) --> NaN
       fmad (-x,+inf,+inf) --> NaN
       fmad (-inf,-y,-inf) --> NaN
       fmad (-x,-inf,-inf) --> NaN
       fmad (+inf,+y,-inf) --> NaN
       fmad (+x,+inf,-inf) --> NaN
    */
    if(
      (((xx << 1) == 0) && ((yy << 1) == 0xff000000)) ||
      (((yy << 1) == 0) && ((xx << 1) == 0xff000000)))
    {
      return rsqrtf(__int_as_float(0xffc00000));
    }
    if((zz << 1) == 0xff000000)
    {
      if(((yy << 1) == 0xff000000) || ((xx << 1) == 0xff000000))
      {
        if((int)(xx ^ yy ^ zz) < 0)
        {
          return rsqrtf(__int_as_float(0xffc00000));
        }
      }
    }
    /* fmad (inf, y, z) --> inf
       fmad (x, inf, z) --> inf
       fmad (x, y, inf) --> inf
    */
    if((xx << 1) == 0xff000000)
    {
      xx = xx ^ (yy & 0x80000000);
      return __int_as_float(xx);
    }
    if((yy << 1) == 0xff000000)
    {
      yy = yy ^ (xx & 0x80000000);
      return __int_as_float(yy);
    }
    if((zz << 1) == 0xff000000)
    {
      return __int_as_float(zz);
    }
    /* fmad (+0, -y, -0) --> -0
       fmad (-0, +y, -0) --> -0
       fmad (+x, -0, -0) --> -0
       fmad (-x, +0, -0) --> -0
    */
    if(zz == 0x80000000)
    {
      if(((xx << 1) == 0) || ((yy << 1) == 0))
      {
        if((int)(xx ^ yy) < 0)
        {
          return __int_as_float(zz);
        }
      }
    }
    /* fmad (0, y, 0) --> +0
       fmad (x, 0, 0) --> +0
    */
    if(((zz << 1) == 0) && (((xx << 1) == 0) || ((yy << 1) == 0)))
    {
      zz &= 0x7fffffff;
      return __int_as_float(zz);
    }
    /* fmad (0, y, z) --> z
       fmad (x, 0, z) --> z
     */
    if(((xx << 1) == 0) || ((yy << 1) == 0))
    {
      return __int_as_float(zz);
    }
    /* normalize x, if denormal */
    if(expo_x == (unsigned)-1)
    {
      temp = xx & 0x80000000;
      xx = xx << 8;
      while(!(xx & 0x80000000))
      {
        xx <<= 1;
        expo_x--;
      }
      expo_x++;
      xx = (xx >> 8) | temp;
    }
    /* normalize y, if denormal */
    if(expo_y == (unsigned)-1)
    {
      temp = yy & 0x80000000;
      yy = yy << 8;
      while(!(yy & 0x80000000))
      {
        yy <<= 1;
        expo_y--;
      }
      expo_y++;
      yy = (yy >> 8) | temp;
    }
    /* normalize z, if denormal */
    if((expo_z == (unsigned)-1) && ((zz << 1) != 0))
    {
      temp = zz & 0x80000000;
      zz = zz << 8;
      while(!(zz & 0x80000000))
      {
        zz <<= 1;
        expo_z--;
      }
      expo_z++;
      zz = (zz >> 8) | temp;
    }
  }

  expo_x = expo_x + expo_y;
  expo_y = xx ^ yy;
  xx = xx & 0x00ffffff;
  yy = yy << 8;
  xx = xx | 0x00800000;
  yy = yy | 0x80000000;

  asm("mul.wide.u32 %0, %1, %2;" : "=l"(product) : "r"(xx), "r"(yy));
  xx = (unsigned)(product >> 32);
  yy = (unsigned)(product & 0xffffffff);

  expo_x = expo_x - 127 + 2;
  expo_y = expo_y & 0x80000000;
  /* normalize mantissa */
  if(xx < 0x00800000)
  {
    xx = (xx << 1) | (yy >> 31);
    yy = (yy << 1);
    expo_x--;
  }
  temp = 0;

  if((zz << 1) != 0)
  { /* z is not zero */
    s = zz & 0x80000000;
    zz &= 0x00ffffff;
    zz |= 0x00800000;
    ww = 0;
    /* compare and swap. put augend into xx:yy */
    if((int)expo_z > (int)expo_x)
    {
      temp = expo_z;
      expo_z = expo_x;
      expo_x = temp;
      temp = zz;
      zz = xx;
      xx = temp;
      temp = ww;
      ww = yy;
      yy = temp;
      temp = expo_y;
      expo_y = s;
      s = temp;
    }
    /* augend_sign = expo_y, augend_mant = xx:yy, augend_expo = expo_x */
    /* addend_sign = s, addend_mant = zz:ww, addend_expo = expo_z */
    expo_z = expo_x - expo_z;
    u = expo_y ^ s;
    if(expo_z <= 49)
    {
      /* denormalize addend */
      temp = 0;
      while(expo_z >= 32)
      {
        temp = ww | (temp != 0);
        ww = zz;
        zz = 0;
        expo_z -= 32;
      }
      if(expo_z)
      {
        temp =
          ((temp >> expo_z) | (ww << (32 - expo_z)) |
           ((temp << (32 - expo_z)) != 0));
        ww = (ww >> expo_z) | (zz << (32 - expo_z));
        zz = (zz >> expo_z);
      }
    }
    else
    {
      temp = 1;
      ww = 0;
      zz = 0;
    }
    if((int)u < 0)
    {
      /* signs differ, effective subtraction */
      temp = (unsigned)(-(int)temp);
      s = (temp != 0);
      u = yy - s;
      s = u > yy;
      yy = u - ww;
      s += yy > u;
      xx = (xx - zz) - s;
      if(!(xx | yy | temp))
      {
        /* complete cancelation, return 0 */
        return __int_as_float(xx);
      }
      if((int)xx < 0)
      {
        /* ooops, augend had smaller mantissa. Negate mantissa and flip
           sign of result*/
        temp = ~temp;
        yy = ~yy;
        xx = ~xx;
        if(++temp == 0)
        {
          if(++yy == 0)
          {
            ++xx;
          }
        }
        expo_y ^= 0x80000000;
      }
      /* normalize mantissa, if necessary */
      while(!(xx & 0x00800000))
      {
        xx = (xx << 1) | (yy >> 31);
        yy = (yy << 1);
        expo_x--;
      }
    }
    else
    {
      /* signs are the same, effective addition */
      yy = yy + ww;
      s = yy < ww;
      xx = xx + zz + s;
      if(xx & 0x01000000)
      {
        temp = temp | (yy << 31);
        yy = (yy >> 1) | (xx << 31);
        xx = ((xx & 0x80000000) | (xx >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
  }
  temp = yy | (temp != 0);
  if(expo_x <= 0xFD)
  {
    /* normal */
    xx |= expo_y; /* or in sign bit */
    s = xx & 1;   /* mantissa lsb */
    xx += (temp == 0x80000000) ? s : (temp >> 31);
    xx = xx + (expo_x << 23); /* add in exponent */
    return __int_as_float(xx);
  }
  else if((int)expo_x >= 126)
  {
    /* overflow */
    xx = expo_y | 0x7f800000;
    return __int_as_float(xx);
  }
  /* subnormal */
  expo_x = (unsigned int)(-(int)expo_x);
  /* Match 'flush to zero' response of the GPU */
  xx += (temp >= 0x80000000);
  if(xx >= 0x01000000)
  {
    xx = xx >> 1;
    expo_x--;
  }
  if(expo_x > 0)
    xx = 0;
  xx = expo_y | xx;
  return __int_as_float(xx);
}

static __forceinline__ float __fmaf_rz(float a, float b, float c)
{
  unsigned long long product;
  unsigned int xx, yy, zz, ww;
  unsigned int temp, s, u;
  unsigned int expo_x, expo_y, expo_z;

  xx = __float_as_int(a);
  yy = __float_as_int(b);
  zz = __float_as_int(c);

  /* Match 'denormals are zero' behavior of the GPU */
  if((xx << 1) < 0x01000000)
    xx &= 0x80000000;
  if((yy << 1) < 0x01000000)
    yy &= 0x80000000;
  if((zz << 1) < 0x01000000)
    zz &= 0x80000000;

  temp = 0xff;
  expo_x = temp & (xx >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yy >> 23);
  expo_y = expo_y - 1;
  expo_z = temp & (zz >> 23);
  expo_z = expo_z - 1;

  if(!((expo_x <= 0xFD) && (expo_y <= 0xFD) && (expo_z <= 0xFD)))
  {
    /* fmad (nan, y, z) --> nan
       fmad (x, nan, z) --> nan
       fmad (x, y, nan) --> nan
    */
    if((yy << 1) > 0xff000000)
    {
      return rsqrtf(b);
    }
    if((zz << 1) > 0xff000000)
    {
      return rsqrtf(c);
    }
    if((xx << 1) > 0xff000000)
    {
      return rsqrtf(a);
    }
    /* fmad (0, inf, z) --> NaN
       fmad (inf, 0, z) --> NaN
       fmad (-inf,+y,+inf) --> NaN
       fmad (+x,-inf,+inf) --> NaN
       fmad (+inf,-y,+inf) --> NaN
       fmad (-x,+inf,+inf) --> NaN
       fmad (-inf,-y,-inf) --> NaN
       fmad (-x,-inf,-inf) --> NaN
       fmad (+inf,+y,-inf) --> NaN
       fmad (+x,+inf,-inf) --> NaN
    */
    if(
      (((xx << 1) == 0) && ((yy << 1) == 0xff000000)) ||
      (((yy << 1) == 0) && ((xx << 1) == 0xff000000)))
    {
      return rsqrtf(__int_as_float(0xffc00000));
    }
    if((zz << 1) == 0xff000000)
    {
      if(((yy << 1) == 0xff000000) || ((xx << 1) == 0xff000000))
      {
        if((int)(xx ^ yy ^ zz) < 0)
        {
          return rsqrtf(__int_as_float(0xffc00000));
        }
      }
    }
    /* fmad (inf, y, z) --> inf
       fmad (x, inf, z) --> inf
       fmad (x, y, inf) --> inf
    */
    if((xx << 1) == 0xff000000)
    {
      xx = xx ^ (yy & 0x80000000);
      return __int_as_float(xx);
    }
    if((yy << 1) == 0xff000000)
    {
      yy = yy ^ (xx & 0x80000000);
      return __int_as_float(yy);
    }
    if((zz << 1) == 0xff000000)
    {
      return __int_as_float(zz);
    }
    /* fmad (+0, -y, -0) --> -0
       fmad (-0, +y, -0) --> -0
       fmad (+x, -0, -0) --> -0
       fmad (-x, +0, -0) --> -0
    */
    if(zz == 0x80000000)
    {
      if(((xx << 1) == 0) || ((yy << 1) == 0))
      {
        if((int)(xx ^ yy) < 0)
        {
          return __int_as_float(zz);
        }
      }
    }
    /* fmad (0, y, 0) --> +0
       fmad (x, 0, 0) --> +0
    */
    if(((zz << 1) == 0) && (((xx << 1) == 0) || ((yy << 1) == 0)))
    {
      zz &= 0x7fffffff;
      return __int_as_float(zz);
    }
    /* fmad (0, y, z) --> z
       fmad (x, 0, z) --> z
     */
    if(((xx << 1) == 0) || ((yy << 1) == 0))
    {
      return __int_as_float(zz);
    }
    /* normalize x, if denormal */
    if(expo_x == (unsigned)-1)
    {
      temp = xx & 0x80000000;
      xx = xx << 8;
      while(!(xx & 0x80000000))
      {
        xx <<= 1;
        expo_x--;
      }
      expo_x++;
      xx = (xx >> 8) | temp;
    }
    /* normalize y, if denormal */
    if(expo_y == (unsigned)-1)
    {
      temp = yy & 0x80000000;
      yy = yy << 8;
      while(!(yy & 0x80000000))
      {
        yy <<= 1;
        expo_y--;
      }
      expo_y++;
      yy = (yy >> 8) | temp;
    }
    /* normalize z, if denormal */
    if((expo_z == (unsigned)-1) && ((zz << 1) != 0))
    {
      temp = zz & 0x80000000;
      zz = zz << 8;
      while(!(zz & 0x80000000))
      {
        zz <<= 1;
        expo_z--;
      }
      expo_z++;
      zz = (zz >> 8) | temp;
    }
  }

  expo_x = expo_x + expo_y;
  expo_y = xx ^ yy;
  xx = xx & 0x00ffffff;
  yy = yy << 8;
  xx = xx | 0x00800000;
  yy = yy | 0x80000000;

  asm("mul.wide.u32 %0, %1, %2;" : "=l"(product) : "r"(xx), "r"(yy));
  xx = (unsigned)(product >> 32);
  yy = (unsigned)(product & 0xffffffff);

  expo_x = expo_x - 127 + 2;
  expo_y = expo_y & 0x80000000;
  /* normalize mantissa */
  if(xx < 0x00800000)
  {
    xx = (xx << 1) | (yy >> 31);
    yy = (yy << 1);
    expo_x--;
  }
  temp = 0;

  if((zz << 1) != 0)
  { /* z is not zero */
    s = zz & 0x80000000;
    zz &= 0x00ffffff;
    zz |= 0x00800000;
    ww = 0;
    /* compare and swap. put augend into xx:yy */
    if((int)expo_z > (int)expo_x)
    {
      temp = expo_z;
      expo_z = expo_x;
      expo_x = temp;
      temp = zz;
      zz = xx;
      xx = temp;
      temp = ww;
      ww = yy;
      yy = temp;
      temp = expo_y;
      expo_y = s;
      s = temp;
    }
    /* augend_sign = expo_y, augend_mant = xx:yy, augend_expo = expo_x */
    /* addend_sign = s, addend_mant = zz:ww, addend_expo = expo_z */
    expo_z = expo_x - expo_z;
    u = expo_y ^ s;
    if(expo_z <= 49)
    {
      /* denormalize addend */
      temp = 0;
      while(expo_z >= 32)
      {
        temp = ww | (temp != 0);
        ww = zz;
        zz = 0;
        expo_z -= 32;
      }
      if(expo_z)
      {
        temp =
          ((temp >> expo_z) | (ww << (32 - expo_z)) |
           ((temp << (32 - expo_z)) != 0));
        ww = (ww >> expo_z) | (zz << (32 - expo_z));
        zz = (zz >> expo_z);
      }
    }
    else
    {
      temp = 1;
      ww = 0;
      zz = 0;
    }
    if((int)u < 0)
    {
      /* signs differ, effective subtraction */
      temp = (unsigned)(-(int)temp);
      s = (temp != 0);
      u = yy - s;
      s = u > yy;
      yy = u - ww;
      s += yy > u;
      xx = (xx - zz) - s;
      if(!(xx | yy | temp))
      {
        /* complete cancelation, return 0 */
        return __int_as_float(xx);
      }
      if((int)xx < 0)
      {
        /* ooops, augend had smaller mantissa. Negate mantissa and flip
           sign of result*/
        temp = ~temp;
        yy = ~yy;
        xx = ~xx;
        if(++temp == 0)
        {
          if(++yy == 0)
          {
            ++xx;
          }
        }
        expo_y ^= 0x80000000;
      }
      /* normalize mantissa, if necessary */
      while(!(xx & 0x00800000))
      {
        xx = (xx << 1) | (yy >> 31);
        yy = (yy << 1);
        expo_x--;
      }
    }
    else
    {
      /* signs are the same, effective addition */
      yy = yy + ww;
      s = yy < ww;
      xx = xx + zz + s;
      if(xx & 0x01000000)
      {
        temp = temp | (yy << 31);
        yy = (yy >> 1) | (xx << 31);
        xx = ((xx & 0x80000000) | (xx >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
  }
  temp = yy | (temp != 0);
  if(expo_x <= 0xFD)
  {
    /* normal */
    xx |= expo_y;             /* or in sign bit */
    xx = xx + (expo_x << 23); /* add in exponent */
    return __int_as_float(xx);
  }
  else if((int)expo_x >= 126)
  {
    /* overflow */
    xx = expo_y | 0x7f7fffff;
    return __int_as_float(xx);
  }
  /* subnormal */
  return __int_as_float(expo_y);
}

static __forceinline__ float __fmaf_ru(float a, float b, float c)
{
  unsigned long long product;
  unsigned int xx, yy, zz, ww;
  unsigned int temp, s, u;
  unsigned int expo_x, expo_y, expo_z;

  xx = __float_as_int(a);
  yy = __float_as_int(b);
  zz = __float_as_int(c);

  /* Match 'denormals are zero' behavior of the GPU */
  if((xx << 1) < 0x01000000)
    xx &= 0x80000000;
  if((yy << 1) < 0x01000000)
    yy &= 0x80000000;
  if((zz << 1) < 0x01000000)
    zz &= 0x80000000;

  temp = 0xff;
  expo_x = temp & (xx >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yy >> 23);
  expo_y = expo_y - 1;
  expo_z = temp & (zz >> 23);
  expo_z = expo_z - 1;

  if(!((expo_x <= 0xFD) && (expo_y <= 0xFD) && (expo_z <= 0xFD)))
  {
    /* fmad (nan, y, z) --> nan
       fmad (x, nan, z) --> nan
       fmad (x, y, nan) --> nan
    */
    if((yy << 1) > 0xff000000)
    {
      return rsqrtf(b);
    }
    if((zz << 1) > 0xff000000)
    {
      return rsqrtf(c);
    }
    if((xx << 1) > 0xff000000)
    {
      return rsqrtf(a);
    }
    /* fmad (0, inf, z) --> NaN
       fmad (inf, 0, z) --> NaN
       fmad (-inf,+y,+inf) --> NaN
       fmad (+x,-inf,+inf) --> NaN
       fmad (+inf,-y,+inf) --> NaN
       fmad (-x,+inf,+inf) --> NaN
       fmad (-inf,-y,-inf) --> NaN
       fmad (-x,-inf,-inf) --> NaN
       fmad (+inf,+y,-inf) --> NaN
       fmad (+x,+inf,-inf) --> NaN
    */
    if(
      (((xx << 1) == 0) && ((yy << 1) == 0xff000000)) ||
      (((yy << 1) == 0) && ((xx << 1) == 0xff000000)))
    {
      return rsqrtf(__int_as_float(0xffc00000));
    }
    if((zz << 1) == 0xff000000)
    {
      if(((yy << 1) == 0xff000000) || ((xx << 1) == 0xff000000))
      {
        if((int)(xx ^ yy ^ zz) < 0)
        {
          return rsqrtf(__int_as_float(0xffc00000));
        }
      }
    }
    /* fmad (inf, y, z) --> inf
       fmad (x, inf, z) --> inf
       fmad (x, y, inf) --> inf
    */
    if((xx << 1) == 0xff000000)
    {
      xx = xx ^ (yy & 0x80000000);
      return __int_as_float(xx);
    }
    if((yy << 1) == 0xff000000)
    {
      yy = yy ^ (xx & 0x80000000);
      return __int_as_float(yy);
    }
    if((zz << 1) == 0xff000000)
    {
      return __int_as_float(zz);
    }
    /* fmad (+0, -y, -0) --> -0
       fmad (-0, +y, -0) --> -0
       fmad (+x, -0, -0) --> -0
       fmad (-x, +0, -0) --> -0
    */
    if(zz == 0x80000000)
    {
      if(((xx << 1) == 0) || ((yy << 1) == 0))
      {
        if((int)(xx ^ yy) < 0)
        {
          return __int_as_float(zz);
        }
      }
    }
    /* fmad (0, y, 0) --> +0
       fmad (x, 0, 0) --> +0
    */
    if(((zz << 1) == 0) && (((xx << 1) == 0) || ((yy << 1) == 0)))
    {
      zz &= 0x7fffffff;
      return __int_as_float(zz);
    }
    /* fmad (0, y, z) --> z
       fmad (x, 0, z) --> z
     */
    if(((xx << 1) == 0) || ((yy << 1) == 0))
    {
      return __int_as_float(zz);
    }
    /* normalize x, if denormal */
    if(expo_x == (unsigned)-1)
    {
      temp = xx & 0x80000000;
      xx = xx << 8;
      while(!(xx & 0x80000000))
      {
        xx <<= 1;
        expo_x--;
      }
      expo_x++;
      xx = (xx >> 8) | temp;
    }
    /* normalize y, if denormal */
    if(expo_y == (unsigned)-1)
    {
      temp = yy & 0x80000000;
      yy = yy << 8;
      while(!(yy & 0x80000000))
      {
        yy <<= 1;
        expo_y--;
      }
      expo_y++;
      yy = (yy >> 8) | temp;
    }
    /* normalize z, if denormal */
    if((expo_z == (unsigned)-1) && ((zz << 1) != 0))
    {
      temp = zz & 0x80000000;
      zz = zz << 8;
      while(!(zz & 0x80000000))
      {
        zz <<= 1;
        expo_z--;
      }
      expo_z++;
      zz = (zz >> 8) | temp;
    }
  }

  expo_x = expo_x + expo_y;
  expo_y = xx ^ yy;
  xx = xx & 0x00ffffff;
  yy = yy << 8;
  xx = xx | 0x00800000;
  yy = yy | 0x80000000;

  asm("mul.wide.u32 %0, %1, %2;" : "=l"(product) : "r"(xx), "r"(yy));
  xx = (unsigned)(product >> 32);
  yy = (unsigned)(product & 0xffffffff);

  expo_x = expo_x - 127 + 2;
  expo_y = expo_y & 0x80000000;
  /* normalize mantissa */
  if(xx < 0x00800000)
  {
    xx = (xx << 1) | (yy >> 31);
    yy = (yy << 1);
    expo_x--;
  }
  temp = 0;

  if((zz << 1) != 0)
  { /* z is not zero */
    s = zz & 0x80000000;
    zz &= 0x00ffffff;
    zz |= 0x00800000;
    ww = 0;
    /* compare and swap. put augend into xx:yy */
    if((int)expo_z > (int)expo_x)
    {
      temp = expo_z;
      expo_z = expo_x;
      expo_x = temp;
      temp = zz;
      zz = xx;
      xx = temp;
      temp = ww;
      ww = yy;
      yy = temp;
      temp = expo_y;
      expo_y = s;
      s = temp;
    }
    /* augend_sign = expo_y, augend_mant = xx:yy, augend_expo = expo_x */
    /* addend_sign = s, addend_mant = zz:ww, addend_expo = expo_z */
    expo_z = expo_x - expo_z;
    u = expo_y ^ s;
    if(expo_z <= 49)
    {
      /* denormalize addend */
      temp = 0;
      while(expo_z >= 32)
      {
        temp = ww | (temp != 0);
        ww = zz;
        zz = 0;
        expo_z -= 32;
      }
      if(expo_z)
      {
        temp =
          ((temp >> expo_z) | (ww << (32 - expo_z)) |
           ((temp << (32 - expo_z)) != 0));
        ww = (ww >> expo_z) | (zz << (32 - expo_z));
        zz = (zz >> expo_z);
      }
    }
    else
    {
      temp = 1;
      ww = 0;
      zz = 0;
    }
    if((int)u < 0)
    {
      /* signs differ, effective subtraction */
      temp = (unsigned)(-(int)temp);
      s = (temp != 0);
      u = yy - s;
      s = u > yy;
      yy = u - ww;
      s += yy > u;
      xx = (xx - zz) - s;
      if(!(xx | yy | temp))
      {
        /* complete cancelation, return 0 */
        return __int_as_float(xx);
      }
      if((int)xx < 0)
      {
        /* ooops, augend had smaller mantissa. Negate mantissa and flip
           sign of result*/
        temp = ~temp;
        yy = ~yy;
        xx = ~xx;
        if(++temp == 0)
        {
          if(++yy == 0)
          {
            ++xx;
          }
        }
        expo_y ^= 0x80000000;
      }
      /* normalize mantissa, if necessary */
      while(!(xx & 0x00800000))
      {
        xx = (xx << 1) | (yy >> 31);
        yy = (yy << 1);
        expo_x--;
      }
    }
    else
    {
      /* signs are the same, effective addition */
      yy = yy + ww;
      s = yy < ww;
      xx = xx + zz + s;
      if(xx & 0x01000000)
      {
        temp = temp | (yy << 31);
        yy = (yy >> 1) | (xx << 31);
        xx = ((xx & 0x80000000) | (xx >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
  }
  temp = yy | (temp != 0);
  if(expo_x <= 0xFD)
  {
    /* normal */
    xx |= expo_y;             /* or in sign bit */
    xx += (temp && !expo_y);  /* round result */
    xx = xx + (expo_x << 23); /* add in exponent */
    return __int_as_float(xx);
  }
  else if((int)expo_x >= 126)
  {
    /* overflow */
    xx = expo_y | (expo_y ? 0x7f7fffff : 0x7F800000);
    return __int_as_float(xx);
  }
  /* subnormal */
  expo_x = ((unsigned int)-((int)expo_x));
  xx += (temp && !expo_y);
  xx = (xx >> expo_x);
  if((expo_x > 25) || (xx != 0x00800000))
    xx = 0;
  return __int_as_float(expo_y | xx);
}

static __forceinline__ float __fmaf_rd(float a, float b, float c)
{
  unsigned long long product;
  unsigned int xx, yy, zz, ww;
  unsigned int temp, s, u;
  unsigned int expo_x, expo_y, expo_z;

  xx = __float_as_int(a);
  yy = __float_as_int(b);
  zz = __float_as_int(c);

  /* Match 'denormals are zero' behavior of the GPU */
  if((xx << 1) < 0x01000000)
    xx &= 0x80000000;
  if((yy << 1) < 0x01000000)
    yy &= 0x80000000;
  if((zz << 1) < 0x01000000)
    zz &= 0x80000000;

  temp = 0xff;
  expo_x = temp & (xx >> 23);
  expo_x = expo_x - 1;
  expo_y = temp & (yy >> 23);
  expo_y = expo_y - 1;
  expo_z = temp & (zz >> 23);
  expo_z = expo_z - 1;

  if(!((expo_x <= 0xFD) && (expo_y <= 0xFD) && (expo_z <= 0xFD)))
  {
    /* fmad (nan, y, z) --> nan
       fmad (x, nan, z) --> nan
       fmad (x, y, nan) --> nan
    */
    if((yy << 1) > 0xff000000)
    {
      return rsqrtf(b);
    }
    if((zz << 1) > 0xff000000)
    {
      return rsqrtf(c);
    }
    if((xx << 1) > 0xff000000)
    {
      return rsqrtf(a);
    }
    /* fmad (0, inf, z) --> NaN
       fmad (inf, 0, z) --> NaN
       fmad (-inf,+y,+inf) --> NaN
       fmad (+x,-inf,+inf) --> NaN
       fmad (+inf,-y,+inf) --> NaN
       fmad (-x,+inf,+inf) --> NaN
       fmad (-inf,-y,-inf) --> NaN
       fmad (-x,-inf,-inf) --> NaN
       fmad (+inf,+y,-inf) --> NaN
       fmad (+x,+inf,-inf) --> NaN
    */
    if(
      (((xx << 1) == 0) && ((yy << 1) == 0xff000000)) ||
      (((yy << 1) == 0) && ((xx << 1) == 0xff000000)))
    {
      return rsqrtf(__int_as_float(0xffc00000));
    }
    if((zz << 1) == 0xff000000)
    {
      if(((yy << 1) == 0xff000000) || ((xx << 1) == 0xff000000))
      {
        if((int)(xx ^ yy ^ zz) < 0)
        {
          return rsqrtf(__int_as_float(0xffc00000));
        }
      }
    }
    /* fmad (inf, y, z) --> inf
       fmad (x, inf, z) --> inf
       fmad (x, y, inf) --> inf
    */
    if((xx << 1) == 0xff000000)
    {
      xx = xx ^ (yy & 0x80000000);
      return __int_as_float(xx);
    }
    if((yy << 1) == 0xff000000)
    {
      yy = yy ^ (xx & 0x80000000);
      return __int_as_float(yy);
    }
    if((zz << 1) == 0xff000000)
    {
      return __int_as_float(zz);
    }
    /* fmad (+0, -y, -0) --> -0
       fmad (-0, +y, -0) --> -0
       fmad (+x, -0, -0) --> -0
       fmad (-x, +0, -0) --> -0
    */
    if(zz == 0x80000000)
    {
      if(((xx << 1) == 0) || ((yy << 1) == 0))
      {
        if((int)(xx ^ yy) < 0)
        {
          return __int_as_float(zz);
        }
      }
    }
    /* fmad (0, y, 0) --> +0
       fmad (x, 0, 0) --> +0
    */
    if(((zz << 1) == 0) && (((xx << 1) == 0) || ((yy << 1) == 0)))
    {
      zz = (xx ^ yy ^ zz) & 0x80000000;
      return __int_as_float(zz);
    }
    /* fmad (0, y, z) --> z
       fmad (x, 0, z) --> z
     */
    if(((xx << 1) == 0) || ((yy << 1) == 0))
    {
      return __int_as_float(zz);
    }
    /* normalize x, if denormal */
    if(expo_x == (unsigned)-1)
    {
      temp = xx & 0x80000000;
      xx = xx << 8;
      while(!(xx & 0x80000000))
      {
        xx <<= 1;
        expo_x--;
      }
      expo_x++;
      xx = (xx >> 8) | temp;
    }
    /* normalize y, if denormal */
    if(expo_y == (unsigned)-1)
    {
      temp = yy & 0x80000000;
      yy = yy << 8;
      while(!(yy & 0x80000000))
      {
        yy <<= 1;
        expo_y--;
      }
      expo_y++;
      yy = (yy >> 8) | temp;
    }
    /* normalize z, if denormal */
    if((expo_z == (unsigned)-1) && ((zz << 1) != 0))
    {
      temp = zz & 0x80000000;
      zz = zz << 8;
      while(!(zz & 0x80000000))
      {
        zz <<= 1;
        expo_z--;
      }
      expo_z++;
      zz = (zz >> 8) | temp;
    }
  }

  expo_x = expo_x + expo_y;
  expo_y = xx ^ yy;
  xx = xx & 0x00ffffff;
  yy = yy << 8;
  xx = xx | 0x00800000;
  yy = yy | 0x80000000;

  asm("mul.wide.u32 %0, %1, %2;" : "=l"(product) : "r"(xx), "r"(yy));
  xx = (unsigned)(product >> 32);
  yy = (unsigned)(product & 0xffffffff);

  expo_x = expo_x - 127 + 2;
  expo_y = expo_y & 0x80000000;
  /* normalize mantissa */
  if(xx < 0x00800000)
  {
    xx = (xx << 1) | (yy >> 31);
    yy = (yy << 1);
    expo_x--;
  }
  temp = 0;

  if((zz << 1) != 0)
  { /* z is not zero */
    s = zz & 0x80000000;
    zz &= 0x00ffffff;
    zz |= 0x00800000;
    ww = 0;
    /* compare and swap. put augend into xx:yy */
    if((int)expo_z > (int)expo_x)
    {
      temp = expo_z;
      expo_z = expo_x;
      expo_x = temp;
      temp = zz;
      zz = xx;
      xx = temp;
      temp = ww;
      ww = yy;
      yy = temp;
      temp = expo_y;
      expo_y = s;
      s = temp;
    }
    /* augend_sign = expo_y, augend_mant = xx:yy, augend_expo = expo_x */
    /* addend_sign = s, addend_mant = zz:ww, addend_expo = expo_z */
    expo_z = expo_x - expo_z;
    u = expo_y ^ s;
    if(expo_z <= 49)
    {
      /* denormalize addend */
      temp = 0;
      while(expo_z >= 32)
      {
        temp = ww | (temp != 0);
        ww = zz;
        zz = 0;
        expo_z -= 32;
      }
      if(expo_z)
      {
        temp =
          ((temp >> expo_z) | (ww << (32 - expo_z)) |
           ((temp << (32 - expo_z)) != 0));
        ww = (ww >> expo_z) | (zz << (32 - expo_z));
        zz = (zz >> expo_z);
      }
    }
    else
    {
      temp = 1;
      ww = 0;
      zz = 0;
    }
    if((int)u < 0)
    {
      /* signs differ, effective subtraction */
      temp = (unsigned)(-(int)temp);
      s = (temp != 0);
      u = yy - s;
      s = u > yy;
      yy = u - ww;
      s += yy > u;
      xx = (xx - zz) - s;
      if(!(xx | yy | temp))
      {
        /* complete cancelation, return -0 */
        return __int_as_float(0x80000000);
      }
      if((int)xx < 0)
      {
        /* ooops, augend had smaller mantissa. Negate mantissa and flip
           sign of result*/
        temp = ~temp;
        yy = ~yy;
        xx = ~xx;
        if(++temp == 0)
        {
          if(++yy == 0)
          {
            ++xx;
          }
        }
        expo_y ^= 0x80000000;
      }
      /* normalize mantissa, if necessary */
      while(!(xx & 0x00800000))
      {
        xx = (xx << 1) | (yy >> 31);
        yy = (yy << 1);
        expo_x--;
      }
    }
    else
    {
      /* signs are the same, effective addition */
      yy = yy + ww;
      s = yy < ww;
      xx = xx + zz + s;
      if(xx & 0x01000000)
      {
        temp = temp | (yy << 31);
        yy = (yy >> 1) | (xx << 31);
        xx = ((xx & 0x80000000) | (xx >> 1)) & ~0x40000000;
        expo_x++;
      }
    }
  }
  temp = yy | (temp != 0);
  if(expo_x <= 0xFD)
  {
    /* normal */
    xx |= expo_y;             /* or in sign bit */
    xx += (temp && expo_y);   /* round result */
    xx = xx + (expo_x << 23); /* add in exponent */
    return __int_as_float(xx);
  }
  else if((int)expo_x >= 126)
  {
    /* overflow */
    xx = expo_y | (expo_y ? 0x7f800000 : 0x7F7FFFFF);
    return __int_as_float(xx);
  }
  /* subnormal */
  expo_x = ((unsigned int)-((int)expo_x));
  xx += (temp && expo_y);
  xx = (xx >> expo_x);
  if((expo_x > 25) || (xx != 0x00800000))
    xx = 0;
  return __int_as_float(expo_y | xx);
}

static __forceinline__ int __clz(int a)
{
  return (a) ? (158 - (__float_as_int(__uint2float_rz((unsigned int)a)) >> 23))
             : 32;
}

static __forceinline__ int __clzll(long long int a)
{
  int ahi = ((int)((unsigned long long)a >> 32));
  int alo = ((int)((unsigned long long)a & 0xffffffffULL));
  int res;
  if(ahi)
  {
    res = 0;
  }
  else
  {
    res = 32;
    ahi = alo;
  }
  res = res + __clz(ahi);
  return res;
}

static __forceinline__ int __popc(unsigned int a)
{
  a = a - ((a >> 1) & 0x55555555);
  a = (a & 0x33333333) + ((a >> 2) & 0x33333333);
  a = (a + (a >> 4)) & 0x0f0f0f0f;
  a = ((__umul24(a, 0x808080) << 1) + a) >> 24;
  return a;
}

static __forceinline__ int __popcll(unsigned long long int a)
{
  unsigned int ahi = ((unsigned int)(a >> 32));
  unsigned int alo = ((unsigned int)(a & 0xffffffffULL));
  alo = alo - ((alo >> 1) & 0x55555555);
  alo = (alo & 0x33333333) + ((alo >> 2) & 0x33333333);
  ahi = ahi - ((ahi >> 1) & 0x55555555);
  ahi = (ahi & 0x33333333) + ((ahi >> 2) & 0x33333333);
  alo = alo + ahi;
  alo = (alo & 0x0f0f0f0f) + ((alo >> 4) & 0x0f0f0f0f);
  alo = ((__umul24(alo, 0x808080) << 1) + alo) >> 24;
  return alo;
}

static __forceinline__ unsigned int __brev(unsigned int a)
{
  /* Use Knuth's algorithm from http://www.hackersdelight.org/revisions.pdf */
  unsigned int t;
  a = (a << 15) | (a >> 17);
  t = (a ^ (a >> 10)) & 0x003f801f;
  a = (t + (t << 10)) ^ a;
  t = (a ^ (a >> 4)) & 0x0e038421;
  a = (t + (t << 4)) ^ a;
  t = (a ^ (a >> 2)) & 0x22488842;
  a = (t + (t << 2)) ^ a;
  return a;
}

static __forceinline__ unsigned long long int __brevll(unsigned long long int a)
{
  unsigned int hi = (unsigned int)(a >> 32);
  unsigned int lo = (unsigned int)(a & 0xffffffffULL);
  unsigned int t;
  t = __brev(lo);
  lo = __brev(hi);
  return ((unsigned long long int)t << 32) + (unsigned long long int)lo;
}

static __forceinline__ unsigned int
__byte_perm(unsigned int a, unsigned int b, unsigned int slct)
{
  unsigned int i0 = (slct >> 0) & 0x7;
  unsigned int i1 = (slct >> 4) & 0x7;
  unsigned int i2 = (slct >> 8) & 0x7;
  unsigned int i3 = (slct >> 12) & 0x7;

  return (
    ((((i0 < 4) ? (a >> (i0 * 8)) : (b >> ((i0 - 4) * 8))) & 0xff) << 0) +
    ((((i1 < 4) ? (a >> (i1 * 8)) : (b >> ((i1 - 4) * 8))) & 0xff) << 8) +
    ((((i2 < 4) ? (a >> (i2 * 8)) : (b >> ((i2 - 4) * 8))) & 0xff) << 16) +
    ((((i3 < 4) ? (a >> (i3 * 8)) : (b >> ((i3 - 4) * 8))) & 0xff) << 24));
}

#endif /* __CUDA_ARCH__ < 200 */

static __forceinline__ int __ffs(int a)
{
  return 32 - __clz(a & -a);
}

static __forceinline__ int __ffsll(long long int a)
{
  return 64 - __clzll(a & -a);
}

#endif /* __CUDANVVM__ */

/*******************************************************************************
 *                                                                             *
 *                          SIMD functions                                     *
 *                                                                             *
 *******************************************************************************/

static __forceinline__ unsigned int __vabs2(unsigned int a)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int b = 0, c = 0;
  asm("vabsdiff2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#elif __CUDA_ARCH__ >= 200
  asm(
    "{                       \n\t"
    ".reg .u32 a,m,r;        \n\t"
    "mov.b32  a,%1;          \n\t"
    "prmt.b32 m,a,0,0xbb99;  \n\t" // msb ? 0xffff : 0000
    "xor.b32  r,a,m;         \n\t" // conditionally invert bits
    "and.b32  m,m,0x00010001;\n\t" // msb ? 0x1 : 0
    "add.u32  r,r,m;         \n\t" // conditionally add 1
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a));
#else /* __CUDA_ARCH__ >= 200 */
  asm(
    "{                       \n\t"
    ".reg .u32 a,m,r,s;      \n\t"
    "mov.b32  a,%1;          \n\t"
    "and.b32  m,a,0x80008000;\n\t" // extract msb
    "and.b32  r,a,0x7fff7fff;\n\t" // clear msb
    "shr.u32  s,m,15;        \n\t" // build lsb mask
    "sub.u32  m,m,s;         \n\t" //  from msb
    "xor.b32  r,r,m;         \n\t" // conditionally invert lsbs
    "add.u32  r,r,s;         \n\t" // conditionally add 1
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a));
#endif /* __CUDA_ARCH__ >= 200 */
  return r; // halfword-wise absolute value, with wrap-around
}

static __forceinline__ unsigned int __vabsss2(unsigned int a)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int b = 0, c = 0;
  asm("vabsdiff2.s32.s32.s32.sat %0,%1,%2,%3;"
      : "=r"(r)
      : "r"(a), "r"(b), "r"(c));
#elif __CUDA_ARCH__ >= 200
  asm(
    "{                       \n\t"
    ".reg .u32 a,m,r;        \n\t"
    "mov.b32  a,%1;          \n\t"
    "prmt.b32 m,a,0,0xbb99;  \n\t" // msb ? 0xffff : 0000
    "xor.b32  r,a,m;         \n\t" // conditionally invert bits
    "and.b32  m,m,0x00010001;\n\t" // msb ? 0x1 : 0
    "add.u32  r,r,m;         \n\t" // conditionally add 1
    "prmt.b32 m,r,0,0xbb99;  \n\t" // msb ? 0xffff : 0000
    "and.b32  m,m,0x00010001;\n\t" // msb ? 0x1 : 0
    "sub.u32  r,r,m;         \n\t" // subtract 1 if result wrapped around
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a));
#else /* __CUDA_ARCH__ >= 200 */
  asm(
    "{                       \n\t"
    ".reg .u32 a,m,r,s;      \n\t"
    "mov.b32  a,%1;          \n\t"
    "and.b32  m,a,0x80008000;\n\t" // extract msb
    "and.b32  r,a,0x7fff7fff;\n\t" // clear msb
    "shr.u32  s,m,15;        \n\t" // build lsb mask
    "sub.u32  m,m,s;         \n\t" //  from msb
    "xor.b32  r,r,m;         \n\t" // conditionally invert lsbs
    "add.u32  r,r,s;         \n\t" // conditionally add 1
    "and.b32  m,r,0x80008000;\n\t" // extract msb (1 if wrap-around)
    "shr.u32  s,m,15;        \n\t" // msb ? 1 : 0
    "sub.u32  r,r,s;         \n\t" // subtract 1 if result wrapped around
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a));
#endif /* __CUDA_ARCH__ >= 200 */
  return r; // halfword-wise absolute value with signed saturation
}

static __forceinline__ unsigned int __vadd2(unsigned int a, unsigned int b)
{
  unsigned int s, t;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vadd2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(t) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  s = a ^ b;          // sum bits
  t = a + b;          // actual sum
  s = s ^ t;          // determine carry-ins for each bit position
  s = s & 0x00010000; // carry-in to high word (= carry-out from low word)
  t = t - s;          // subtract out carry-out from low word
#endif /* __CUDA_ARCH__ >= 300 */
  return t; // halfword-wise sum, with wrap around
}

static __forceinline__ unsigned int __vaddss2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vadd2.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  int ahi, alo, blo, bhi, rhi, rlo;
  ahi = (int)((a & 0xffff0000U));
  bhi = (int)((b & 0xffff0000U));
#if __CUDA_ARCH__ < 350
  alo = (int)(a << 16);
  blo = (int)(b << 16);
#else /* __CUDA_ARCH__ < 350 */
  asm("shf.l.clamp.b32 %0,0,%1,16;" : "=r"(alo) : "r"(a));
  asm("shf.l.clamp.b32 %0,0,%1,16;" : "=r"(blo) : "r"(b));
#endif /* __CUDA_ARCH__ < 350 */
  asm("add.sat.s32 %0,%1,%2;" : "=r"(rlo) : "r"(alo), "r"(blo));
  asm("add.sat.s32 %0,%1,%2;" : "=r"(rhi) : "r"(ahi), "r"(bhi));
#if __CUDA_ARCH__ < 200
  r = ((unsigned int)rhi & 0xffff0000U) | ((unsigned int)rlo >> 16);
#else
  asm("prmt.b32 %0,%1,%2,0x7632;" : "=r"(r) : "r"(rlo), "r"(rhi));
#endif /* __CUDA_ARCH__ < 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise sum with signed saturation
}

static __forceinline__ unsigned int __vaddus2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vadd2.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  int alo, blo, rlo, ahi, bhi, rhi;
  asm(
    "{                              \n\t"
    "and.b32     %0, %4, 0xffff;    \n\t"
    "and.b32     %1, %5, 0xffff;    \n\t"
#if __CUDA_ARCH__ < 350
    "shr.u32     %2, %4, 16;        \n\t"
    "shr.u32     %3, %5, 16;        \n\t"
#else /* __CUDA_ARCH__ < 350 */
    "shf.r.clamp.b32  %2, %4, 0, 16;\n\t"
    "shf.r.clamp.b32  %3, %5, 0, 16;\n\t"
#endif /* __CUDA_ARCH__ < 350 */
    "}"
    : "=r"(alo), "=r"(blo), "=r"(ahi), "=r"(bhi)
    : "r"(a), "r"(b));
  rlo = min(alo + blo, 65535);
  rhi = min(ahi + bhi, 65535);
  r = (rhi << 16) + rlo;
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise sum with unsigned saturation
}

static __forceinline__ unsigned int __vavgs2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vavrg2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  // avgs (a + b) = ((a + b) < 0) ? ((a + b) >> 1) : ((a + b + 1) >> 1). The
  // two expressions can be re-written as follows to avoid needing additional
  // intermediate bits: ((a + b) >> 1) = (a >> 1) + (b >> 1) + ((a & b) & 1),
  // ((a + b + 1) >> 1) = (a >> 1) + (b >> 1) + ((a | b) & 1). The difference
  // between the two is ((a ^ b) & 1). Note that if (a + b) < 0, then also
  // ((a + b) >> 1) < 0, since right shift rounds to negative infinity. This
  // means we can compute ((a + b) >> 1) then conditionally add ((a ^ b) & 1)
  // depending on the sign bit of the shifted sum. By handling the msb sum
  // bit of the result separately, we avoid carry-out during summation and
  // also can use (potentially faster) logical right shifts.
  asm(
    "{                      \n\t"
    ".reg .u32 a,b,c,r,s,t,u,v;\n\t"
    "mov.b32 a,%1;          \n\t"
    "mov.b32 b,%2;          \n\t"
    "and.b32 u,a,0xfffefffe;\n\t" // prevent shift crossing chunk boundary
    "and.b32 v,b,0xfffefffe;\n\t" // prevent shift crossing chunk boundary
    "xor.b32 s,a,b;         \n\t" // a ^ b
    "and.b32 t,a,b;         \n\t" // a & b
    "shr.u32 u,u,1;         \n\t" // a >> 1
    "shr.u32 v,v,1;         \n\t" // b >> 1
    "and.b32 c,s,0x00010001;\n\t" // (a ^ b) & 1
    "and.b32 s,s,0x80008000;\n\t" // extract msb (a ^ b)
    "and.b32 t,t,0x00010001;\n\t" // (a & b) & 1
    "add.u32 r,u,v;         \n\t" // (a>>1)+(b>>1)
    "add.u32 r,r,t;         \n\t" // (a>>1)+(b>>1)+(a&b&1); rec. msb cy-in
    "xor.b32 r,r,s;         \n\t" // compute msb sum bit: a ^ b ^ cy-in
    "shr.u32 t,r,15;        \n\t" // sign ((a + b) >> 1)
    "not.b32 t,t;           \n\t" // ~sign ((a + b) >> 1)
    "and.b32 t,t,c;         \n\t" // ((a ^ b) & 1) & ~sign ((a + b) >> 1)
    "add.u32 r,r,t;         \n\t" // conditionally add ((a ^ b) & 1)
    "mov.b32 %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise average of signed integers
}

static __forceinline__ unsigned int __vavgu2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vavrg2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  // HAKMEM #23: a + b = 2 * (a | b) - (a ^ b) ==>
  // (a + b + 1) / 2 = (a | b) - ((a ^ b) >> 1)
  c = a ^ b;
  r = a | b;
  c = c & 0xfffefffe; // ensure shift doesn't cross half-word boundaries
  c = c >> 1;
  r = r - c;
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise average of unsigned integers
}

static __forceinline__ unsigned int __vhaddu2(unsigned int a, unsigned int b)
{
  // HAKMEM #23: a + b = 2 * (a & b) + (a ^ b) ==>
  // (a + b) / 2 = (a & b) + ((a ^ b) >> 1)
  unsigned int r, s;
  s = a ^ b;
  r = a & b;
  s = s & 0xfffefffe; // ensure shift doesn't cross halfword boundaries
  s = s >> 1;
  r = r + s;
  return r; // halfword-wise average of unsigned ints, rounded down
}

static __forceinline__ unsigned int __vcmpeq2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.eq %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 16; // convert bool
  r = c - r;   //  into mask
#else /* __CUDA_ARCH__ >= 300 */
  // inspired by Alan Mycroft's null-byte detection algorithm:
  // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
  r = a ^ b;          // 0x0000 if a == b
  c = r | 0x80008000; // set msbs, to catch carry out
  r = r ^ c;          // extract msbs, msb = 1 if r < 0x8000
  c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
  c = r & ~c;         // msb = 1, if r was 0x0000
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c)); // convert msbs to mask
#else /* __CUDA_ARCH__ >= 200 */
  asm("shr.u32 %0,%1,15;" : "=r"(r) : "r"(c)); // convert
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  msbs to
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   mask
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise (un)signed eq comparison, mask result
}

static __forceinline__ unsigned int __vcmpges2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset2.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 16; // convert bool
  r = c - r;   //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                             \n\t"
    ".reg .u32 a, b, r, s, t, u;   \n\t"
    "mov.b32        a,%1;          \n\t"
    "mov.b32        b,%2;          \n\t"
    "and.b32        s,a,0xffff0000;\n\t" // high word of a
    "and.b32        t,b,0xffff0000;\n\t" // high word of b
    "set.ge.s32.s32 u,s,t;         \n\t" // compare two high words
    "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
    "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
    "set.ge.s32.s32 s,s,t;         \n\t" // compare two low words
#if __CUDA_ARCH__ >= 200
    "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32        u,u,0xffff0000;\n\t" // mask comparison result hi word
    "and.b32        s,s,0x0000ffff;\n\t" // mask comparison result lo word
    "or.b32         r,s,u;         \n\t" // combine the two results
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32        %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise signed gt-eq comparison, mask result
}

static __forceinline__ unsigned int __vcmpgeu2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 16; // convert bool
  r = c - r;   //  into mask
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(b));
  c = __vavgu2(a, b); // (a + ~b + 1) / 2 = (a - b) / 2
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c)); // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
  asm("and.b32 %0,%0,0x80008000;" : "+r"(c));  // msb = carry-outs
  asm("shr.u32 %0,%1,15;" : "=r"(r) : "r"(c)); // build mask
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  from
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise unsigned gt-eq comparison, mask result
}

static __forceinline__ unsigned int __vcmpgts2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset2.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 16; // convert bool
  r = c - r;   //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                             \n\t"
    ".reg .u32 a, b, r, s, t, u;   \n\t"
    "mov.b32        a,%1;          \n\t"
    "mov.b32        b,%2;          \n\t"
    "and.b32        s,a,0xffff0000;\n\t" // high word of a
    "and.b32        t,b,0xffff0000;\n\t" // high word of b
    "set.gt.s32.s32 u,s,t;         \n\t" // compare two high words
    "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
    "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
    "set.gt.s32.s32 s,s,t;         \n\t" // compare two low words
#if __CUDA_ARCH__ >= 200
    "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32        u,u,0xffff0000;\n\t" // mask comparison result hi word
    "and.b32        s,s,0x0000ffff;\n\t" // mask comparison result lo word
    "or.b32         r,s,u;         \n\t" // combine the two results
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32        %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise signed gt comparison with mask result
}

static __forceinline__ unsigned int __vcmpgtu2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 16; // convert bool
  r = c - r;   //  into mask
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(b));
  c = __vhaddu2(a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c)); // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
  asm("and.b32 %0,%0,0x80008000;" : "+r"(c));  // msb = carry-outs
  asm("shr.u32 %0,%1,15;" : "=r"(r) : "r"(c)); // build mask
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  from
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise unsigned gt comparison, mask result
}

static __forceinline__ unsigned int __vcmples2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset2.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 16; // convert bool
  r = c - r;   //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                             \n\t"
    ".reg .u32 a, b, r, s, t, u;   \n\t"
    "mov.b32        a,%1;          \n\t"
    "mov.b32        b,%2;          \n\t"
    "and.b32        s,a,0xffff0000;\n\t" // high word of a
    "and.b32        t,b,0xffff0000;\n\t" // high word of b
    "set.le.s32.s32 u,s,t;         \n\t" // compare two high words
    "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
    "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
    "set.le.s32.s32 s,s,t;         \n\t" // compare two low words
#if __CUDA_ARCH__ >= 200
    "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32        u,u,0xffff0000;\n\t" // mask comparison result hi word
    "and.b32        s,s,0x0000ffff;\n\t" // mask comparison result lo word
    "or.b32         r,s,u;         \n\t" // combine the two results
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32        %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise signed lt-eq comparison, mask result
}

static __forceinline__ unsigned int __vcmpleu2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 16; // convert bool
  r = c - r;   //  into mask
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(a));
  c = __vavgu2(a, b); // (b + ~a + 1) / 2 = (b - a) / 2
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c)); // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
  asm("and.b32 %0,%0,0x80008000;" : "+r"(c));  // msb = carry-outs
  asm("shr.u32 %0,%1,15;" : "=r"(r) : "r"(c)); // build mask
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  from
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise unsigned lt-eq comparison, mask result
}

static __forceinline__ unsigned int __vcmplts2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset2.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 16; // convert bool
  r = c - r;   //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                             \n\t"
    ".reg .u32 a, b, r, s, t, u;   \n\t"
    "mov.b32        a,%1;          \n\t"
    "mov.b32        b,%2;          \n\t"
    "and.b32        s,a,0xffff0000;\n\t" // high word of a
    "and.b32        t,b,0xffff0000;\n\t" // high word of b
    "set.lt.s32.s32 u,s,t;         \n\t" // compare two high words
    "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
    "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
    "set.lt.s32.s32 s,s,t;         \n\t" // compare two low words
#if __CUDA_ARCH__ >= 200
    "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32        u,u,0xffff0000;\n\t" // mask comparison result hi word
    "and.b32        s,s,0x0000ffff;\n\t" // mask comparison result lo word
    "or.b32         r,s,u;         \n\t" // combine the two results
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32        %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise signed lt comparison with mask result
}

static __forceinline__ unsigned int __vcmpltu2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 16; // convert bool
  r = c - r;   //  into mask
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(a));
  c = __vhaddu2(a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c)); // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
  asm("and.b32 %0,%0,0x80008000;" : "+r"(c));  // msb = carry-outs
  asm("shr.u32 %0,%1,15;" : "=r"(r) : "r"(c)); // build mask
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  from
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise unsigned lt comparison, mask result
}

static __forceinline__ unsigned int __vcmpne2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 16; // convert bool
  r = c - r;   //  into mask
#else /* __CUDA_ARCH__ >= 300 */
  // inspired by Alan Mycroft's null-byte detection algorithm:
  // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
  r = a ^ b;          // 0x0000 if a == b
  c = r | 0x80008000; // set msbs, to catch carry out
  c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
  c = r | c;          // msb = 1, if r was not 0x0000
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xbb99;" : "=r"(r) : "r"(c)); // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
  asm("and.b32 %0,%0,0x80008000;" : "+r"(c));  // extract msbs
  asm("shr.u32 %0,%1,15;" : "=r"(r) : "r"(c)); // build mask
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  from
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise (un)signed ne comparison, mask result
}

static __forceinline__ unsigned int __vabsdiffu2(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vabsdiff2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  //this branch executes
  unsigned int t, u, v;
  s = a & 0x0000ffff; // extract low halfword
  r = b & 0x0000ffff; // extract low halfword
  u = umax(r, s);     // maximum of low halfwords
  v = umin(r, s);     // minimum of low halfwords
  s = a & 0xffff0000; // extract high halfword
  r = b & 0xffff0000; // extract high halfword
  t = umax(r, s);     // maximum of high halfwords
  s = umin(r, s);     // minimum of high halfwords
  r = u | t;          // maximum of both halfwords
  s = v | s;          // minimum of both halfwords
  r = r - s;          // |a - b| = max(a,b) - min(a,b);
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise absolute difference of unsigned ints
}

static __forceinline__ unsigned int __vmaxs2(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vmax2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  unsigned int t, u;
  asm("cvt.s32.s16 %0,%1;" : "=r"(r) : "r"(a)); // extract low halfword
  asm("cvt.s32.s16 %0,%1;" : "=r"(s) : "r"(b)); // extract low halfword
  t = max((int)r, (int)s);                      // maximum of low halfwords
  r = a & 0xffff0000;                           // extract high halfword
  s = b & 0xffff0000;                           // extract high halfword
  u = max((int)r, (int)s);                      // maximum of high halfwords
  r = u | (t & 0xffff);                         // combine halfword maximums
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise maximum of signed integers
}

static __forceinline__ unsigned int __vmaxu2(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vmax2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  unsigned int t, u;
  r = a & 0x0000ffff; // extract low halfword
  s = b & 0x0000ffff; // extract low halfword
  t = umax(r, s);     // maximum of low halfwords
  r = a & 0xffff0000; // extract high halfword
  s = b & 0xffff0000; // extract high halfword
  u = umax(r, s);     // maximum of high halfwords
  r = t | u;          // combine halfword maximums
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise maximum of unsigned integers
}

static __forceinline__ unsigned int __vmins2(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vmin2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  unsigned int t, u;
  asm("cvt.s32.s16 %0,%1;" : "=r"(r) : "r"(a)); // extract low halfword
  asm("cvt.s32.s16 %0,%1;" : "=r"(s) : "r"(b)); // extract low halfword
  t = min((int)r, (int)s);                      // minimum of low halfwords
  r = a & 0xffff0000;                           // extract high halfword
  s = b & 0xffff0000;                           // extract high halfword
  u = min((int)r, (int)s);                      // minimum of high halfwords
  r = u | (t & 0xffff);                         // combine halfword minimums
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise minimum of signed integers
}

static __forceinline__ unsigned int __vminu2(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vmin2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  unsigned int t, u;
  r = a & 0x0000ffff; // extract low halfword
  s = b & 0x0000ffff; // extract low halfword
  t = umin(r, s);     // minimum of low halfwords
  r = a & 0xffff0000; // extract high halfword
  s = b & 0xffff0000; // extract high halfword
  u = umin(r, s);     // minimum of high halfwords
  r = t | u;          // combine halfword minimums
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise minimum of unsigned integers
}

static __forceinline__ unsigned int __vseteq2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.eq %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  // inspired by Alan Mycroft's null-byte detection algorithm:
  // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
  r = a ^ b;          // 0x0000 if a == b
  c = r | 0x80008000; // set msbs, to catch carry out
  r = r ^ c;          // extract msbs, msb = 1 if r < 0x8000
  c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
  c = r & ~c;         // msb = 1, if r was 0x0000
  r = c >> 15;        // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise (un)signed eq comparison, bool result
}

static __forceinline__ unsigned int __vsetges2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset2.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                             \n\t"
    ".reg .u32 a, b, r, s, t, u;   \n\t"
    "mov.b32        a,%1;          \n\t"
    "mov.b32        b,%2;          \n\t"
    "and.b32        s,a,0xffff0000;\n\t" // high word of a
    "and.b32        t,b,0xffff0000;\n\t" // high word of b
    "set.ge.s32.s32 u,s,t;         \n\t" // compare two high words
    "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
    "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
    "set.ge.s32.s32 s,s,t;         \n\t" // compare two low words
#if __CUDA_ARCH__ >= 200
    "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
    "and.b32        r,r,0x00010001;\n\t" // convert from mask to bool
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32        u,u,0x00010000;\n\t" // extract bool result of hi word
    "and.b32        s,s,0x00000001;\n\t" // extract bool result of lo word
    "or.b32         r,s,u;         \n\t" // combine the two results
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32        %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise signed gt-eq comparison, bool result
}

static __forceinline__ unsigned int __vsetgeu2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(b));
  c = __vavgu2(a, b); // (a + ~b + 1) / 2 = (a - b) / 2
  c = c & 0x80008000; // msb = carry-outs
  r = c >> 15;        // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise unsigned gt-eq comparison, bool result
}

static __forceinline__ unsigned int __vsetgts2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset2.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                             \n\t"
    ".reg .u32 a, b, r, s, t, u;   \n\t"
    "mov.b32        a,%1;          \n\t"
    "mov.b32        b,%2;          \n\t"
    "and.b32        s,a,0xffff0000;\n\t" // high word of a
    "and.b32        t,b,0xffff0000;\n\t" // high word of b
    "set.gt.s32.s32 u,s,t;         \n\t" // compare two high words
    "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
    "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
    "set.gt.s32.s32 s,s,t;         \n\t" // compare two low words
#if __CUDA_ARCH__ >= 200
    "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
    "and.b32        r,r,0x00010001;\n\t" // convert from mask to bool
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32        u,u,0x00010000;\n\t" // extract bool result of hi word
    "and.b32        s,s,0x00000001;\n\t" // extract bool result of lo word
    "or.b32         r,s,u;         \n\t" // combine the two results
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32        %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise signed gt comparison with bool result
}

static __forceinline__ unsigned int __vsetgtu2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(b));
  c = __vhaddu2(a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
  c = c & 0x80008000;  // msbs = carry-outs
  r = c >> 15;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise unsigned gt comparison, bool result
}

static __forceinline__ unsigned int __vsetles2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset2.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                             \n\t"
    ".reg .u32 a, b, r, s, t, u;   \n\t"
    "mov.b32        a,%1;          \n\t"
    "mov.b32        b,%2;          \n\t"
    "and.b32        s,a,0xffff0000;\n\t" // high word of a
    "and.b32        t,b,0xffff0000;\n\t" // high word of b
    "set.le.s32.s32 u,s,t;         \n\t" // compare two high words
    "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
    "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
    "set.le.s32.s32 s,s,t;         \n\t" // compare two low words
#if __CUDA_ARCH__ >= 200
    "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
    "and.b32        r,r,0x00010001;\n\t" // convert from mask to bool
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32        u,u,0x00010000;\n\t" // extract bool result of hi word
    "and.b32        s,s,0x00000001;\n\t" // extract bool result of lo word
    "or.b32         r,s,u;         \n\t" // combine the two results
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32        %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise signed lt-eq comparison, bool result
}

static __forceinline__ unsigned int __vsetleu2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(a));
  c = __vavgu2(a, b); // (b + ~a + 1) / 2 = (b - a) / 2
  c = c & 0x80008000; // msb = carry-outs
  r = c >> 15;        // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise unsigned lt-eq comparison, bool result
}

static __forceinline__ unsigned int __vsetlts2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset2.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                             \n\t"
    ".reg .u32 a, b, r, s, t, u;   \n\t"
    "mov.b32        a,%1;          \n\t"
    "mov.b32        b,%2;          \n\t"
    "and.b32        s,a,0xffff0000;\n\t" // high word of a
    "and.b32        t,b,0xffff0000;\n\t" // high word of b
    "set.lt.s32.s32 u,s,t;         \n\t" // compare two high words
    "cvt.s32.s16    s,a;           \n\t" // sign-extend low word of a
    "cvt.s32.s16    t,b;           \n\t" // sign-extend low word of b
    "set.lt.s32.s32 s,s,t;         \n\t" // compare two low words
#if __CUDA_ARCH__ >= 200
    "prmt.b32       r,s,u,0x7632;  \n\t" // combine low and high results
    "and.b32        r,r,0x00010001;\n\t" // convert from mask to bool
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32        u,u,0x00010000;\n\t" // extract bool result of hi word
    "and.b32        s,s,0x00000001;\n\t" // extract bool result of lo word
    "or.b32         r,s,u;         \n\t" // combine the two results
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32        %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise signed lt comparison with bool result
}

static __forceinline__ unsigned int __vsetltu2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(a));
  c = __vhaddu2(a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
  c = c & 0x80008000;  // msb = carry-outs
  r = c >> 15;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise unsigned lt comparison, bool result
}

static __forceinline__ unsigned int __vsetne2(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset2.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  // inspired by Alan Mycroft's null-byte detection algorithm:
  // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
  r = a ^ b;          // 0x0000 if a == b
  c = r | 0x80008000; // set msbs, to catch carry out
  c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
  c = r | c;          // msb = 1, if r was not 0x0000
  c = c & 0x80008000; // extract msbs
  r = c >> 15;        // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise (un)signed ne comparison, bool result
}

static __forceinline__ unsigned int __vsadu2(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vabsdiff2.u32.u32.u32.add %0,%1,%2,%3;"
      : "=r"(r)
      : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  unsigned int t, u, v;
  s = a & 0x0000ffff; // extract low halfword
  r = b & 0x0000ffff; // extract low halfword
  u = umax(r, s);     // maximum of low halfwords
  v = umin(r, s);     // minimum of low halfwords
  s = a & 0xffff0000; // extract high halfword
  r = b & 0xffff0000; // extract high halfword
  t = umax(r, s);     // maximum of high halfwords
  s = umin(r, s);     // minimum of high halfwords
  u = u - v;          // low halfword: |a - b| = max(a,b) - min(a,b);
  t = t - s;          // high halfword: |a - b| = max(a,b) - min(a,b);
#if __CUDA_ARCH__ < 350
  asm("shr.u32 %0,%0,16;" : "+r"(t));
#else /*__CUDA_ARCH__ < 350 */
  asm("shf.r.clamp.b32  %0,%0,0,16;" : "+r"(t));
#endif /*__CUDA_ARCH__ < 350 */
  r = t + u; // sum absolute halfword differences
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise sum of abs differences of unsigned int
}

static __forceinline__ unsigned int __vsub2(unsigned int a, unsigned int b)
{
  unsigned int s, t;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vsub2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(t) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  s = a ^ b;          // sum bits
  t = a - b;          // actual sum
  s = s ^ t;          // determine carry-ins for each bit position
  s = s & 0x00010000; // borrow to high word
  t = t + s;          // compensate for borrow from low word
#endif /* __CUDA_ARCH__ >= 300 */
  return t; // halfword-wise difference
}

static __forceinline__ unsigned int __vsubss2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vsub2.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  int ahi, alo, blo, bhi, rhi, rlo;
  ahi = (int)((a & 0xffff0000U));
  bhi = (int)((b & 0xffff0000U));
#if __CUDA_ARCH__ < 200
  alo = (int)(a << 16);
  blo = (int)(b << 16);
#elif __CUDA_ARCH__ < 350
  // work around (we would want left shifts at least for sm_2x)
  asm("prmt.b32 %0,%1,0,0x1044;" : "=r"(alo) : "r"(a));
  asm("prmt.b32 %0,%1,0,0x1044;" : "=r"(blo) : "r"(b));
#else /* __CUDA_ARCH__ < 350 */
  asm("shf.l.clamp.b32 %0,0,%1,16;" : "=r"(alo) : "r"(a));
  asm("shf.l.clamp.b32 %0,0,%1,16;" : "=r"(blo) : "r"(b));
#endif /* __CUDA_ARCH__ < 350 */
  asm("sub.sat.s32 %0,%1,%2;" : "=r"(rlo) : "r"(alo), "r"(blo));
  asm("sub.sat.s32 %0,%1,%2;" : "=r"(rhi) : "r"(ahi), "r"(bhi));
#if __CUDA_ARCH__ < 200
  r = ((unsigned int)rhi & 0xffff0000U) | ((unsigned int)rlo >> 16);
#else /* __CUDA_ARCH__ < 200 */
  asm("prmt.b32 %0,%1,%2,0x7632;" : "=r"(r) : "r"(rlo), "r"(rhi));
#endif /* __CUDA_ARCH__ < 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise difference with signed saturation
}

static __forceinline__ unsigned int __vsubus2(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vsub2.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  int alo, blo, rlo, ahi, bhi, rhi;
  asm(
    "{                              \n\t"
    "and.b32     %0, %4, 0xffff;    \n\t"
    "and.b32     %1, %5, 0xffff;    \n\t"
#if __CUDA_ARCH__ < 350
    "shr.u32     %2, %4, 16;        \n\t"
    "shr.u32     %3, %5, 16;        \n\t"
#else /* __CUDA_ARCH__ < 350 */
    "shf.r.clamp.b32  %2, %4, 0, 16;\n\t"
    "shf.r.clamp.b32  %3, %5, 0, 16;\n\t"
#endif /* __CUDA_ARCH__ < 350 */
    "}"
    : "=r"(alo), "=r"(blo), "=r"(ahi), "=r"(bhi)
    : "r"(a), "r"(b));
  rlo = max((int)(alo - blo), 0);
  rhi = max((int)(ahi - bhi), 0);
  r = rhi * 65536 + rlo;
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise difference with unsigned saturation
}

static __forceinline__ unsigned int __vneg2(unsigned int a)
{
  return __vsub2(0, a); // halfword-wise negation with wrap-around
}

static __forceinline__ unsigned int __vnegss2(unsigned int a)
{
  return __vsubss2(0, a); // halfword-wise negation with signed saturation
}

static __forceinline__ unsigned int __vabsdiffs2(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vabsdiff2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  s = __vcmpges2(a, b); // mask = 0xff if a >= b
  r = a ^ b;            //
  s = (r & s) ^ b;      // select a when a >= b, else select b => max(a,b)
  r = s ^ r;            // select a when b >= a, else select b => min(a,b)
  r = __vsub2(s, r);    // |a - b| = max(a,b) - min(a,b);
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise absolute difference of signed integers
}

static __forceinline__ unsigned int __vsads2(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vabsdiff2.s32.s32.s32.add %0,%1,%2,%3;"
      : "=r"(r)
      : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  s = __vabsdiffs2(a, b);
  r = (s >> 16) + (s & 0x0000ffff);
#endif /*  __CUDA_ARCH__ >= 300 */
  return r; // halfword-wise sum of abs. differences of signed ints
}

static __forceinline__ unsigned int __vabs4(unsigned int a)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int b = 0, c = 0;
  asm("vabsdiff4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#elif __CUDA_ARCH__ >= 200
  asm(
    "{                       \n\t"
    ".reg .u32 a,m,r;        \n\t"
    "mov.b32  a,%1;          \n\t"
    "prmt.b32 m,a,0,0xba98;  \n\t" // msb ? 0xff : 00
    "xor.b32  r,a,m;         \n\t" // conditionally invert bits
    "and.b32  m,m,0x01010101;\n\t" // msb ? 0x1 : 0
    "add.u32  r,r,m;         \n\t" // conditionally add 1
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a));
#else /* __CUDA_ARCH__ >= 200 */
  asm(
    "{                  \n\t"
    ".reg .u32 a,m,r,s; \n\t"
    "mov.b32  a,%1;          \n\t"
    "and.b32  m,a,0x80808080;\n\t" // extract msb
    "and.b32  r,a,0x7f7f7f7f;\n\t" // clear msb
    "shr.u32  s,m,7;         \n\t" // build lsb mask
    "sub.u32  m,m,s;         \n\t" //  from msb
    "xor.b32  r,r,m;         \n\t" // conditionally invert lsbs
    "add.u32  r,r,s;         \n\t" // conditionally add 1
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a));
#endif /* __CUDA_ARCH__ >= 200 */
  return r; // byte-wise absolute value, with wrap-around
}

static __forceinline__ unsigned int __vabsss4(unsigned int a)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int b = 0, c = 0;
  asm("vabsdiff4.s32.s32.s32.sat %0,%1,%2,%3;"
      : "=r"(r)
      : "r"(a), "r"(b), "r"(c));
#elif __CUDA_ARCH__ >= 200
  asm(
    "{                       \n\t"
    ".reg .u32 a,m,r;        \n\t"
    "mov.b32  a,%1;          \n\t"
    "prmt.b32 m,a,0,0xba98;  \n\t" // msb ? 0xff : 00
    "xor.b32  r,a,m;         \n\t" // conditionally invert bits
    "and.b32  m,m,0x01010101;\n\t" // msb ? 0x1 : 0
    "add.u32  r,r,m;         \n\t" // conditionally add 1
    "prmt.b32 m,r,0,0xba98;  \n\t" // msb ? 0xff : 00
    "and.b32  m,m,0x01010101;\n\t" // msb ? 0x1 : 0
    "sub.u32  r,r,m;         \n\t" // subtract 1 if result wrapped around
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a));
#else /* __CUDA_ARCH__ >= 200 */
  asm(
    "{                       \n\t"
    ".reg .u32 a,m,r,s;      \n\t"
    "mov.b32  a,%1;          \n\t"
    "and.b32  m,a,0x80808080;\n\t" // extract msb
    "and.b32  r,a,0x7f7f7f7f;\n\t" // clear msb
    "shr.u32  s,m,7;         \n\t" // build lsb mask
    "sub.u32  m,m,s;         \n\t" //  from msb
    "xor.b32  r,r,m;         \n\t" // conditionally invert lsbs
    "add.u32  r,r,s;         \n\t" // conditionally add 1
    "and.b32  m,r,0x80808080;\n\t" // extract msb (1 if wrap-around)
    "shr.u32  s,m,7;         \n\t" // msb ? 1 : 0
    "sub.u32  r,r,s;         \n\t" // subtract 1 if result wrapped around
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a));
#endif /* __CUDA_ARCH__ >= 200 */
  return r; // byte-wise absolute value with signed saturation
}

static __forceinline__ unsigned int __vadd4(unsigned int a, unsigned int b)
{
#if __CUDA_ARCH__ >= 300
  unsigned int r, c = 0;
  asm("vadd4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  unsigned int r, s, t;
  s = a ^ b;          // sum bits
  r = a & 0x7f7f7f7f; // clear msbs
  t = b & 0x7f7f7f7f; // clear msbs
  s = s & 0x80808080; // msb sum bits
  r = r + t;          // add without msbs, record carry-out in msbs
  r = r ^ s;          // sum of msb sum and carry-in bits, w/o carry-out
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise sum, with wrap-around
}

static __forceinline__ unsigned int __vaddss4(unsigned int a, unsigned int b)
{
#if __CUDA_ARCH__ >= 300
  unsigned int r, c = 0;
  asm("vadd4.sat.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  /*
      For signed saturation, saturation is controlled by the overflow signal:
      ovfl = (carry-in to msb) XOR (carry-out from msb). Overflow can only
      occur when the msbs of both inputs are the same. The defined response to
      overflow is to deliver 0x7f when the addends are positive (bit 7 clear),
      and 0x80 when the addends are negative (bit 7 set). The truth table for
      the msb is

      a   b   cy_in  res  cy_out  ovfl
      --------------------------------
      0   0       0    0       0     0
      0   0       1    1       0     1
      0   1       0    1       0     0
      0   1       1    0       1     0
      1   0       0    1       0     0
      1   0       1    0       1     0
      1   1       0    0       1     1
      1   1       1    1       1     0

      The seven low-order bits can be handled by simple wrapping addition with
      the carry out from bit 6 recorded in the msb (thus corresponding to the
      cy_in in the truth table for the msb above). ovfl can be computed in many
      equivalent ways, here we use ovfl = (a ^ carry_in) & ~(a ^ b) since we
      already need to compute (a ^ b) for the msb sum bit computation. First we
      compute the normal, wrapped addition result. When overflow is detected,
      we mask off the msb of the result, then compute a mask covering the seven
      low order bits, which are all set to 1. This sets the byte to 0x7f as we
      previously cleared the msb. In the overflow case, the sign of the result
      matches the sign of either of the inputs, so we extract the sign of a and
      add it to the low order bits, which turns 0x7f into 0x80, the correct
      result for an overflowed negative result.
    */
  unsigned int r;
  asm(
    "{                         \n\t"
    ".reg .u32 a,b,r,s,t,u;    \n\t"
    "mov.b32  a, %1;           \n\t"
    "mov.b32  b, %2;           \n\t"
    "and.b32  r, a, 0x7f7f7f7f;\n\t" // clear msbs
    "and.b32  t, b, 0x7f7f7f7f;\n\t" // clear msbs
    "xor.b32  s, a, b;         \n\t" // sum bits = (a ^ b)
    "add.u32  r, r, t;         \n\t" // capture msb carry-in in bit 7
    "xor.b32  t, a, r;         \n\t" // a ^ carry_in
    "not.b32  u, s;            \n\t" // ~(a ^ b)
    "and.b32  t, t, u;         \n\t" // ovfl = (a ^ carry_in) & ~(a ^ b)
    "and.b32  s, s, 0x80808080;\n\t" // msb sum bits
    "xor.b32  r, r, s;         \n\t" // msb result = (a ^ b ^ carry_in)
#if __CUDA_ARCH__ >= 200
    "prmt.b32 s,a,0,0xba98;    \n\t" // sign(a) ? 0xff : 0
    "xor.b32  s,s,0x7f7f7f7f;  \n\t" // sign(a) ? 0x80 : 0x7f
    "prmt.b32 t,t,0,0xba98;    \n\t" // ovfl ? 0xff : 0
    "and.b32  s,s,t;           \n\t" // ovfl ? (sign(a) ? 0x80:0x7f) : 0
    "not.b32  t,t;             \n\t" // ~ovfl
    "and.b32  r,r,t;           \n\t" // ovfl ? 0 : a + b
    "or.b32   r,r,s;           \n\t" // ovfl ? (sign(a) ? 0x80:0x7f) : a+b
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32  t, t, 0x80808080;\n\t" // ovfl ? 0x80 : 0
    "shr.u32  s, t, 7;         \n\t" // ovfl ? 1 : 0
    "not.b32  u, t;            \n\t" // ovfl ? 0x7f : 0xff
    "and.b32  r, r, u;         \n\t" // ovfl ? (a + b) & 0x7f : a + b
    "and.b32  u, a, t;         \n\t" // ovfl ? a & 0x80 : 0
    "sub.u32  t, t, s;         \n\t" // ovfl ? 0x7f : 0
    "shr.u32  u, u, 7;         \n\t" // ovfl ? sign(a) : 0
    "or.b32   r, r, t;         \n\t" // ovfl ? 0x7f : a + b
    "add.u32  r, r, u;         \n\t" // ovfl ? 0x7f+sign(a) : a + b
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32  %0, r;           \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise sum with signed saturation
}

static __forceinline__ unsigned int __vaddus4(unsigned int a, unsigned int b)
{
#if __CUDA_ARCH__ >= 300
  unsigned int r, c = 0;
  asm("vadd4.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  // This code uses the same basic approach used for non-saturating addition.
  // The seven low-order bits in each byte are summed by regular addition,
  // with the carry-out from bit 6 (= carry-in for the msb) being recorded
  // in bit 7, while the msb is handled separately.
  //
  // The fact that this is a saturating addition simplifies the handling of
  // the msb. When carry-out from the msb occurs, the entire byte must be
  // written as 0xff, and the computed msb is overwritten in the process.
  // The corresponding entries in the truth table for the result msb thus
  // become "don't cares":
  //
  // a  b  cy-in  res  cy-out
  // ------------------------
  // 0  0    0     0     0
  // 0  0    1     1     0
  // 0  1    0     1     0
  // 0  1    1     X     1
  // 1  0    0     1     0
  // 1  0    1     X     1
  // 1  1    0     X     1
  // 1  1    1     X     1
  //
  // As is easily seen, the simplest implementation of the result msb bit is
  // simply (a | b | cy-in), with masking needed to isolate the msb. Note
  // that this computation also makes the msb handling redundant with the
  // clamping to 0xFF, because the msb is already set to 1 when saturation
  // occurs. This means we only need to apply saturation to the seven lsb
  // bits in each byte, by overwriting with 0x7F. Saturation is controlled
  // by carry-out from the msb, which can be represented by various Boolean
  // expressions. Since to compute (a | b | cy-in) we need to compute (a | b)
  // anyhow, most efficient of these is cy-out = ((a & b) | cy-in) & (a | b).
  unsigned int r;
  asm(
    "{                         \n\t"
    ".reg .u32 a,b,r,s,t,m;    \n\t"
    "mov.b32  a, %1;           \n\t"
    "mov.b32  b, %2;           \n\t"
    "or.b32   m, a, b;         \n\t" // (a | b)
    "and.b32  r, a, 0x7f7f7f7f;\n\t" // clear msbs
    "and.b32  t, b, 0x7f7f7f7f;\n\t" // clear msbs
    "and.b32  m, m, 0x80808080;\n\t" // (a | b), isolate msbs
    "add.u32  r, r, t;         \n\t" // add w/o msbs, record msb-carry-ins
    "and.b32  t, a, b;         \n\t" // (a & b)
    "or.b32   t, t, r;         \n\t" // (a & b) | cy-in)
    "or.b32   r, r, m;         \n\t" // msb = cy-in | (a | b)
    "and.b32  t, t, m;         \n\t" // cy-out=((a&b)|cy-in)&(a|b),in msbs
#if __CUDA_ARCH__ >= 200
    "prmt.b32 t, t, 0, 0xba98; \n\t" // cy-out ? 0xff : 0
#else /* __CUDA_ARCH__ >= 200 */
    "shr.u32  s, t, 7;         \n\t" // cy-out ? 1 : 0
    "sub.u32  t, t, s;         \n\t" // lsb-overwrite: cy-out ? 0x7F : 0
#endif /* __CUDA_ARCH__ >= 200 */
    "or.b32   r, r, t;         \n\t" // conditionally overwrite lsbs
    "mov.b32  %0, r;           \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise sum with unsigned saturation
}

static __forceinline__ unsigned int __vavgs4(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vavrg4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  // avgs (a + b) = ((a + b) < 0) ? ((a + b) >> 1) : ((a + b + 1) >> 1). The
  // two expressions can be re-written as follows to avoid needing additional
  // intermediate bits: ((a + b) >> 1) = (a >> 1) + (b >> 1) + ((a & b) & 1),
  // ((a + b + 1) >> 1) = (a >> 1) + (b >> 1) + ((a | b) & 1). The difference
  // between the two is ((a ^ b) & 1). Note that if (a + b) < 0, then also
  // ((a + b) >> 1) < 0, since right shift rounds to negative infinity. This
  // means we can compute ((a + b) >> 1) then conditionally add ((a ^ b) & 1)
  // depending on the sign bit of the shifted sum. By handling the msb sum
  // bit of the result separately, we avoid carry-out during summation and
  // also can use (potentially faster) logical right shifts.
  asm(
    "{                      \n\t"
    ".reg .u32 a,b,c,r,s,t,u,v;\n\t"
    "mov.b32 a,%1;          \n\t"
    "mov.b32 b,%2;          \n\t"
    "and.b32 u,a,0xfefefefe;\n\t" // prevent shift crossing chunk boundary
    "and.b32 v,b,0xfefefefe;\n\t" // prevent shift crossing chunk boundary
    "xor.b32 s,a,b;         \n\t" // a ^ b
    "and.b32 t,a,b;         \n\t" // a & b
    "shr.u32 u,u,1;         \n\t" // a >> 1
    "shr.u32 v,v,1;         \n\t" // b >> 1
    "and.b32 c,s,0x01010101;\n\t" // (a ^ b) & 1
    "and.b32 s,s,0x80808080;\n\t" // extract msb (a ^ b)
    "and.b32 t,t,0x01010101;\n\t" // (a & b) & 1
    "add.u32 r,u,v;         \n\t" // (a>>1)+(b>>1)
    "add.u32 r,r,t;         \n\t" // (a>>1)+(b>>1)+(a&b&1); rec. msb cy-in
    "xor.b32 r,r,s;         \n\t" // compute msb sum bit: a ^ b ^ cy-in
    "shr.u32 t,r,7;         \n\t" // sign ((a + b) >> 1)
    "not.b32 t,t;           \n\t" // ~sign ((a + b) >> 1)
    "and.b32 t,t,c;         \n\t" // ((a ^ b) & 1) & ~sign ((a + b) >> 1)
    "add.u32 r,r,t;         \n\t" // conditionally add ((a ^ b) & 1)
    "mov.b32 %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise average of signed integers
}

static __forceinline__ unsigned int __vavgu4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vavrg4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  // HAKMEM #23: a + b = 2 * (a | b) - (a ^ b) ==>
  // (a + b + 1) / 2 = (a | b) - ((a ^ b) >> 1)
  c = a ^ b;
  r = a | b;
  c = c & 0xfefefefe; // ensure following shift doesn't cross byte boundaries
  c = c >> 1;
  r = r - c;
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise average of unsigned integers
}

static __forceinline__ unsigned int __vhaddu4(unsigned int a, unsigned int b)
{
  // HAKMEM #23: a + b = 2 * (a & b) + (a ^ b) ==>
  // (a + b) / 2 = (a & b) + ((a ^ b) >> 1)
  unsigned int r, s;
  s = a ^ b;
  r = a & b;
  s = s & 0xfefefefe; // ensure following shift doesn't cross byte boundaries
  s = s >> 1;
  s = r + s;
  return s; // byte-wise average of unsigned integers, rounded down
}

static __forceinline__ unsigned int __vcmpeq4(unsigned int a, unsigned int b)
{
  unsigned int c, r;
#if __CUDA_ARCH__ >= 300
  r = 0;
  asm("vset4.u32.u32.eq %0,%1,%2,%3;" : "=r"(c) : "r"(a), "r"(b), "r"(r));
  r = c << 8; // convert bool
  r = r - c;  //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  // inspired by Alan Mycroft's null-byte detection algorithm:
  // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
  r = a ^ b;          // 0x00 if a == b
  c = r | 0x80808080; // set msbs, to catch carry out
  r = r ^ c;          // extract msbs, msb = 1 if r < 0x80
  c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
  c = r & ~c;         // msb = 1, if r was 0x00
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c)); // convert msbs to mask
#else /* __CUDA_ARCH__ >= 200 */
  asm("shr.u32 %0,%1,7;" : "=r"(r) : "r"(c));  // convert
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  msbs to
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   mask
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise (un)signed eq comparison with mask result
}

static __forceinline__ unsigned int __vcmpges4(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset4.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 8; // convert bool
  r = c - r;  //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                          \n\t"
    ".reg .u32 a, b, r, s, t, u;\n\t"
    "mov.b32     a,%1;          \n\t"
    "mov.b32     b,%2;          \n\t"
    "xor.b32     s,a,b;         \n\t" // a ^ b
    "or.b32      r,a,0x80808080;\n\t" // set msbs
    "and.b32     t,b,0x7f7f7f7f;\n\t" // clear msbs
    "sub.u32     r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
    "xor.b32     t,r,a;         \n\t" // msb: ~borrow-in ^ a
    "xor.b32     r,r,s;         \n\t" // msb: ~sign(res) = a^b^~borrow-in
    "and.b32     t,t,s;         \n\t" // msb: ovfl= (~bw-in ^ a) & (a ^ b)
    "xor.b32     t,t,r;         \n\t" // msb: ge = ovfl != ~sign(res)
#if __CUDA_ARCH__ >= 200
    "prmt.b32    r,t,0,0xba98;  \n\t" // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32     t,t,0x80808080;\n\t" // isolate msbs = ovfl
    "shr.u32     r,t,7;         \n\t" // build mask
    "sub.u32     r,t,r;         \n\t" //  from
    "or.b32      r,r,t;         \n\t" //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32     %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise signed gt-eq comparison with mask result
}

static __forceinline__ unsigned int __vcmpgeu4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset4.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 8; // convert bool
  r = c - r;  //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(b));
  c = __vavgu4(a, b); // (a + ~b + 1) / 2 = (a - b) / 2
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c)); // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
  asm("and.b32 %0,%0,0x80808080;" : "+r"(c));  // msb = carry-outs
  asm("shr.u32 %0,%1,7;" : "=r"(r) : "r"(c));  // build mask
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  from
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise unsigned gt-eq comparison with mask result
}

static __forceinline__ unsigned int __vcmpgts4(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset4.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 8; // convert bool
  r = c - r;  //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  /* a <= b <===> a + ~b < 0 */
  asm(
    "{                       \n\t"
    ".reg .u32 a,b,r,s,t,u;  \n\t"
    "mov.b32  a,%1;          \n\t"
    "mov.b32  b,%2;          \n\t"
    "not.b32  b,b;           \n\t"
    "and.b32  r,a,0x7f7f7f7f;\n\t" // clear msbs
    "and.b32  t,b,0x7f7f7f7f;\n\t" // clear msbs
    "xor.b32  s,a,b;         \n\t" // sum bits = (a ^ b)
    "add.u32  r,r,t;         \n\t" // capture msb carry-in in bit 7
    "xor.b32  t,a,r;         \n\t" // a ^ carry_in
    "not.b32  u,s;           \n\t" // ~(a ^ b)
    "and.b32  t,t,u;         \n\t" // msb: ovfl = (a ^ carry_in) & ~(a^b)
    "xor.b32  r,r,u;         \n\t" // msb: ~result = (~(a ^ b) ^ carry_in)
    "xor.b32  t,t,r;         \n\t" // msb: gt = ovfl != sign(~res)
#if __CUDA_ARCH__ >= 200
    "prmt.b32 r,t,0,0xba98;  \n\t" // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32  t,t,0x80808080;\n\t" // isolate msbs
    "shr.u32  r,t,7;         \n\t" // build mask
    "sub.u32  r,t,r;         \n\t" //  from
    "or.b32   r,r,t;         \n\t" //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise signed gt comparison with mask result
}

static __forceinline__ unsigned int __vcmpgtu4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset4.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 8; // convert bool
  r = c - r;  //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(b));
  c = __vhaddu4(a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c)); // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
  asm("and.b32 %0,%0,0x80808080;" : "+r"(c));  // msb = carry-outs
  asm("shr.u32 %0,%1,7;" : "=r"(r) : "r"(c));  // build mask
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  from
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise unsigned gt comparison with mask result
}

static __forceinline__ unsigned int __vcmples4(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset4.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 8; // convert bool
  r = c - r;  //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  /* a <= b <===> a + ~b < 0 */
  asm(
    "{                       \n\t"
    ".reg .u32 a,b,r,s,t,u;  \n\t"
    "mov.b32  a,%1;          \n\t"
    "mov.b32  b,%2;          \n\t"
    "not.b32  u,b;           \n\t" // ~b
    "and.b32  r,a,0x7f7f7f7f;\n\t" // clear msbs
    "and.b32  t,u,0x7f7f7f7f;\n\t" // clear msbs
    "xor.b32  u,a,b;         \n\t" // sum bits = (a ^ b)
    "add.u32  r,r,t;         \n\t" // capture msb carry-in in bit 7
    "xor.b32  t,a,r;         \n\t" // a ^ carry_in
    "not.b32  s,u;           \n\t" // ~(a ^ b)
    "and.b32  t,t,u;         \n\t" // msb: ovfl = (a ^ carry_in) & (a ^ b)
    "xor.b32  r,r,s;         \n\t" // msb: result = (a ^ ~b ^ carry_in)
    "xor.b32  t,t,r;         \n\t" // msb: le = ovfl != sign(res)
#if __CUDA_ARCH__ >= 200
    "prmt.b32 r,t,0,0xba98;  \n\t" // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32  t,t,0x80808080;\n\t" // isolate msbs
    "shr.u32  r,t,7;         \n\t" // build mask
    "sub.u32  r,t,r;         \n\t" //  from
    "or.b32   r,r,t;         \n\t" //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise signed lt-eq comparison with mask result
}

static __forceinline__ unsigned int __vcmpleu4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset4.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 8; // convert bool
  r = c - r;  //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(a));
  c = __vavgu4(a, b); // (b + ~a + 1) / 2 = (b - a) / 2
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c)); // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
  asm("and.b32 %0,%0,0x80808080;" : "+r"(c));  // msb = carry-outs
  asm("shr.u32 %0,%1,7;" : "=r"(r) : "r"(c));  // build mask
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  from
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise unsigned lt-eq comparison with mask result
}

static __forceinline__ unsigned int __vcmplts4(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset4.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 8; // convert bool
  r = c - r;  //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                          \n\t"
    ".reg .u32 a, b, r, s, t, u;\n\t"
    "mov.b32     a,%1;          \n\t"
    "mov.b32     b,%2;          \n\t"
    "not.b32     u,b;           \n\t" // ~b
    "xor.b32     s,u,a;         \n\t" // a ^ ~b
    "or.b32      r,a,0x80808080;\n\t" // set msbs
    "and.b32     t,b,0x7f7f7f7f;\n\t" // clear msbs
    "sub.u32     r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
    "xor.b32     t,r,a;         \n\t" // msb: ~borrow-in ^ a
    "not.b32     u,s;           \n\t" // msb: ~(a^~b)
    "xor.b32     r,r,s;         \n\t" // msb: res = a ^ ~b ^ ~borrow-in
    "and.b32     t,t,u;         \n\t" // msb: ovfl= (~bw-in ^ a) & ~(a^~b)
    "xor.b32     t,t,r;         \n\t" // msb: lt = ovfl != sign(res)
#if __CUDA_ARCH__ >= 200
    "prmt.b32    r,t,0,0xba98;  \n\t" // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32     t,t,0x80808080;\n\t" // isolate msbs
    "shr.u32     r,t,7;         \n\t" // build mask
    "sub.u32     r,t,r;         \n\t" //  from
    "or.b32      r,r,t;         \n\t" //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32     %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise signed lt comparison with mask result
}

static __forceinline__ unsigned int __vcmpltu4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset4.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 8; // convert bool
  r = c - r;  //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(a));
  c = __vhaddu4(a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c)); // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
  asm("and.b32 %0,%0,0x80808080;" : "+r"(c));  // msb = carry-outs
  asm("shr.u32 %0,%1,7;" : "=r"(r) : "r"(c));  // build mask
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  from
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise unsigned lt comparison with mask result
}

static __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset4.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  c = r << 8; // convert bool
  r = c - r;  //  to mask
#else /* __CUDA_ARCH__ >= 300 */
  // inspired by Alan Mycroft's null-byte detection algorithm:
  // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
  r = a ^ b;          // 0x00 if a == b
  c = r | 0x80808080; // set msbs, to catch carry out
  c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
  c = r | c;          // msb = 1, if r was not 0x00
#if __CUDA_ARCH__ >= 200
  asm("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c)); // build mask from msbs
#else /* __CUDA_ARCH__ >= 200 */
  asm("and.b32 %0,%0,0x80808080;" : "+r"(c));  // extract msbs
  asm("shr.u32 %0,%1,7;" : "=r"(r) : "r"(c));  // build mask
  asm("sub.u32 %0,%1,%0;" : "+r"(r) : "r"(c)); //  from
  asm("or.b32  %0,%1,%0;" : "+r"(r) : "r"(c)); //   msbs
#endif /* __CUDA_ARCH__ >= 200 */
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise (un)signed ne comparison with mask result
}

static __forceinline__ unsigned int __vabsdiffu4(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vabsdiff4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  s = __vcmpgeu4(a, b); // mask = 0xff if a >= b
  r = a ^ b;            //
  s = (r & s) ^ b;      // select a when a >= b, else select b => max(a,b)
  r = s ^ r;            // select a when b >= a, else select b => min(a,b)
  r = s - r;            // |a - b| = max(a,b) - min(a,b);
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise absolute difference of unsigned integers
}

static __forceinline__ unsigned int __vmaxs4(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vmax4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  s = __vcmpges4(a, b); // mask = 0xff if a >= b
  r = a & s;            // select a when b >= a
  s = b & ~s;           // select b when b < a
  r = r | s;            // combine byte selections
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise maximum of signed integers
}

static __forceinline__ unsigned int __vmaxu4(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vmax4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  s = __vcmpgeu4(a, b); // mask = 0xff if a >= b
  r = a & s;            // select a when b >= a
  s = b & ~s;           // select b when b < a
  r = r | s;            // combine byte selections
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise maximum of unsigned integers
}

static __forceinline__ unsigned int __vmins4(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vmin4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  s = __vcmpges4(b, a); // mask = 0xff if a >= b
  r = a & s;            // select a when b >= a
  s = b & ~s;           // select b when b < a
  r = r | s;            // combine byte selections
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise minimum of signed integers
}

static __forceinline__ unsigned int __vminu4(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vmin4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  s = __vcmpgeu4(b, a); // mask = 0xff if a >= b
  r = a & s;            // select a when b >= a
  s = b & ~s;           // select b when b < a
  r = r | s;            // combine byte selections
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise minimum of unsigned integers
}
static __forceinline__ unsigned int __vseteq4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset4.u32.u32.eq %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  // inspired by Alan Mycroft's null-byte detection algorithm:
  // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
  r = a ^ b;          // 0x00 if a == b
  c = r | 0x80808080; // set msbs, to catch carry out
  r = r ^ c;          // extract msbs, msb = 1 if r < 0x80
  c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
  c = r & ~c;         // msb = 1, if r was 0x00
  r = c >> 7;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise (un)signed eq comparison with bool result
}

static __forceinline__ unsigned int __vsetles4(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset4.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  /* a <= b <===> a + ~b < 0 */
  asm(
    "{                       \n\t"
    ".reg .u32 a,b,r,s,t,u;  \n\t"
    "mov.b32  a,%1;          \n\t"
    "mov.b32  b,%2;          \n\t"
    "not.b32  u,b;           \n\t" // ~b
    "and.b32  r,a,0x7f7f7f7f;\n\t" // clear msbs
    "and.b32  t,u,0x7f7f7f7f;\n\t" // clear msbs
    "xor.b32  u,a,b;         \n\t" // sum bits = (a ^ b)
    "add.u32  r,r,t;         \n\t" // capture msb carry-in in bit 7
    "xor.b32  t,a,r;         \n\t" // a ^ carry_in
    "not.b32  s,u;           \n\t" // ~(a ^ b)
    "and.b32  t,t,u;         \n\t" // msb: ovfl = (a ^ carry_in) & (a ^ b)
    "xor.b32  r,r,s;         \n\t" // msb: result = (a ^ ~b ^ carry_in)
    "xor.b32  t,t,r;         \n\t" // msb: le = ovfl != sign(res)
    "and.b32  t,t,0x80808080;\n\t" // isolate msbs
    "shr.u32  r,t,7;         \n\t" // convert to bool
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise signed lt-eq comparison with bool result
}

static __forceinline__ unsigned int __vsetleu4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset4.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(a));
  c = __vavgu4(a, b); // (b + ~a + 1) / 2 = (b - a) / 2
  c = c & 0x80808080; // msb = carry-outs
  r = c >> 7;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise unsigned lt-eq comparison with bool result
}

static __forceinline__ unsigned int __vsetlts4(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset4.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                          \n\t"
    ".reg .u32 a, b, r, s, t, u;\n\t"
    "mov.b32     a,%1;          \n\t"
    "mov.b32     b,%2;          \n\t"
    "not.b32     u,b;           \n\t" // ~b
    "or.b32      r,a,0x80808080;\n\t" // set msbs
    "and.b32     t,b,0x7f7f7f7f;\n\t" // clear msbs
    "xor.b32     s,u,a;         \n\t" // a ^ ~b
    "sub.u32     r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
    "xor.b32     t,r,a;         \n\t" // msb: ~borrow-in ^ a
    "not.b32     u,s;           \n\t" // msb: ~(a^~b)
    "xor.b32     r,r,s;         \n\t" // msb: res = a ^ ~b ^ ~borrow-in
    "and.b32     t,t,u;         \n\t" // msb: ovfl= (~bw-in ^ a) & ~(a^~b)
    "xor.b32     t,t,r;         \n\t" // msb: lt = ovfl != sign(res)
    "and.b32     t,t,0x80808080;\n\t" // isolate msbs
    "shr.u32     r,t,7;         \n\t" // convert to bool
    "mov.b32     %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise signed lt comparison with bool result
}

static __forceinline__ unsigned int __vsetltu4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset4.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(a));
  c = __vhaddu4(a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
  c = c & 0x80808080;  // msb = carry-outs
  r = c >> 7;          // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise unsigned lt comparison with bool result
}

static __forceinline__ unsigned int __vsetges4(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset4.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm(
    "{                          \n\t"
    ".reg .u32 a, b, r, s, t, u;\n\t"
    "mov.b32     a,%1;          \n\t"
    "mov.b32     b,%2;          \n\t"
    "xor.b32     s,a,b;         \n\t" // a ^ b
    "or.b32      r,a,0x80808080;\n\t" // set msbs
    "and.b32     t,b,0x7f7f7f7f;\n\t" // clear msbs
    "sub.u32     r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
    "xor.b32     t,r,a;         \n\t" // msb: ~borrow-in ^ a
    "xor.b32     r,r,s;         \n\t" // msb: ~sign(res) = a^b^~borrow-in
    "and.b32     t,t,s;         \n\t" // msb: ovfl= (~bw-in ^ a) & (a ^ b)
    "xor.b32     t,t,r;         \n\t" // msb: ge = ovfl != ~sign(res)
    "and.b32     t,t,0x80808080;\n\t" // isolate msbs
    "shr.u32     r,t,7;         \n\t" // convert to bool
    "mov.b32     %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise signed gt-eq comparison with bool result
}

static __forceinline__ unsigned int __vsetgeu4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset4.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(b));
  c = __vavgu4(a, b); // (a + ~b + 1) / 2 = (a - b) / 2
  c = c & 0x80808080; // msb = carry-outs
  r = c >> 7;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise unsigned gt-eq comparison with bool result
}

static __forceinline__ unsigned int __vsetgts4(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vset4.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  /* a <= b <===> a + ~b < 0 */
  asm(
    "{                       \n\t"
    ".reg .u32 a,b,r,s,t,u;  \n\t"
    "mov.b32  a,%1;          \n\t"
    "mov.b32  b,%2;          \n\t"
    "not.b32  b,b;           \n\t"
    "and.b32  r,a,0x7f7f7f7f;\n\t" // clear msbs
    "and.b32  t,b,0x7f7f7f7f;\n\t" // clear msbs
    "xor.b32  s,a,b;         \n\t" // sum bits = (a ^ b)
    "add.u32  r,r,t;         \n\t" // capture msb carry-in in bit 7
    "xor.b32  t,a,r;         \n\t" // a ^ carry_in
    "not.b32  u,s;           \n\t" // ~(a ^ b)
    "and.b32  t,t,u;         \n\t" // msb: ovfl = (a ^ carry_in) & ~(a^b)
    "xor.b32  r,r,u;         \n\t" // msb: ~result = (~(a ^ b) ^ carry_in)
    "xor.b32  t,t,r;         \n\t" // msb: gt = ovfl != sign(~res)
    "and.b32  t,t,0x80808080;\n\t" // isolate msbs
    "shr.u32  r,t,7;         \n\t" // convert to bool
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise signed gt comparison with mask result
}

static __forceinline__ unsigned int __vsetgtu4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset4.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  asm("not.b32 %0,%0;" : "+r"(b));
  c = __vhaddu4(a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
  c = c & 0x80808080;  // msb = carry-outs
  r = c >> 7;          // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise unsigned gt comparison with bool result
}

static __forceinline__ unsigned int __vsetne4(unsigned int a, unsigned int b)
{
  unsigned int r, c;
#if __CUDA_ARCH__ >= 300
  c = 0;
  asm("vset4.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  // inspired by Alan Mycroft's null-byte detection algorithm:
  // null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
  r = a ^ b;          // 0x00 if a == b
  c = r | 0x80808080; // set msbs, to catch carry out
  c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
  c = r | c;          // msb = 1, if r was not 0x00
  c = c & 0x80808080; // extract msbs
  r = c >> 7;         // convert to bool
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise (un)signed ne comparison with bool result
}

static __forceinline__ unsigned int __vsadu4(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vabsdiff4.u32.u32.u32.add %0,%1,%2,%3;"
      : "=r"(r)
      : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  r = __vabsdiffu4(a, b);
  s = r >> 8;
  r = (r & 0x00ff00ff) + (s & 0x00ff00ff);
  r = ((r << 16) + r) >> 16;
#endif /*  __CUDA_ARCH__ >= 300 */
  return r; // byte-wise sum of absol. differences of unsigned ints
}

static __forceinline__ unsigned int __vsub4(unsigned int a, unsigned int b)
{
#if __CUDA_ARCH__ >= 300
  unsigned int r, c = 0;
  asm("vsub4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  unsigned int r, s, t;
  s = a ^ ~b;         // inverted sum bits
  r = a | 0x80808080; // set msbs
  t = b & 0x7f7f7f7f; // clear msbs
  s = s & 0x80808080; // inverted msb sum bits
  r = r - t;          // subtract w/o msbs, record inverted borrows in msb
  r = r ^ s;          // combine inverted msb sum bits and borrows
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise difference
}

static __forceinline__ unsigned int __vsubss4(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vsub4.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  /*
      For signed saturation, saturation is controlled by the overflow signal:
      ovfl = (borrow-in to msb) XOR (borrow-out from msb). Overflow can only
      occur when the msbs of both inputs are differemt. The defined response to
      overflow is to deliver 0x7f when the addends are positive (bit 7 clear),
      and 0x80 when the addends are negative (bit 7 set). The truth table for
      the msb is

      a   b  bw_in  res  bw_out  ovfl  a^~bw_in  ~(a^~b) (a^~bw_in)&~(a^~b)
      ---------------------------------------------------------------------
      0   0      0    0       0     0         1        0                  0
      0   0      1    1       1     0         0        0                  0
      0   1      0    1       1     1         1        1                  1
      0   1      1    0       1     0         0        1                  0
      1   0      0    1       0     0         0        1                  0
      1   0      1    0       0     1         1        1                  1
      1   1      0    0       0     0         0        0                  0
      1   1      1    1       1     0         1        0                  0

      The seven low-order bits can be handled by wrapping subtraction with the
      borrow-out from bit 6 recorded in the msb (thus corresponding to the
      bw_in in the truth table for the msb above). ovfl can be computed in many
      equivalent ways, here we use ovfl = (a ^ ~borrow_in) & ~(a ^~b) since we
      already need to compute (a ^~b) and ~borrow-in for the msb result bit
      computation. First we compute the normal, wrapped subtraction result.
      When overflow is detected, we mask off the result's msb, then compute a
      mask covering the seven low order bits, which are all set to 1. This sets
      the byte to 0x7f as we previously cleared the msb. In the overflow case,
      the sign of the result matches the sign of input a, so we extract the
      sign of a and add it to the low order bits, which turns 0x7f into 0x80,
      the correct result for an overflowed negative result.
    */
  asm(
    "{                          \n\t"
    ".reg .u32 a,b,r,s,t,u,v,w; \n\t"
    "mov.b32     a,%1;          \n\t"
    "mov.b32     b,%2;          \n\t"
    "not.b32     u,b;           \n\t" // ~b
    "xor.b32     s,u,a;         \n\t" // a ^ ~b
    "or.b32      r,a,0x80808080;\n\t" // set msbs
    "and.b32     t,b,0x7f7f7f7f;\n\t" // clear msbs
    "sub.u32     r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
    "xor.b32     t,r,a;         \n\t" // msb: ~borrow-in ^ a
    "not.b32     u,s;           \n\t" // msb: ~(a^~b)
    "and.b32     s,s,0x80808080;\n\t" // msb: a ^ ~b
    "xor.b32     r,r,s;         \n\t" // msb: res = a ^ ~b ^ ~borrow-in
    "and.b32     t,t,u;         \n\t" // msb: ovfl= (~bw-in ^ a) & ~(a^~b)
#if __CUDA_ARCH__ >= 200
    "prmt.b32    s,a,0,0xba98;  \n\t" // sign(a) ? 0xff : 0
    "xor.b32     s,s,0x7f7f7f7f;\n\t" // sign(a) ? 0x80 : 0x7f
    "prmt.b32    t,t,0,0xba98;  \n\t" // ovfl ? 0xff : 0
    "and.b32     s,s,t;         \n\t" // ovfl ? (sign(a) ? 0x80:0x7f) : 0
    "not.b32     t,t;           \n\t" // ~ovfl
    "and.b32     r,r,t;         \n\t" // ovfl ? 0 : a + b
    "or.b32      r,r,s;         \n\t" // ovfl ? (sign(a) ? 0x80:0x7f) :a+b
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32     t,t,0x80808080;\n\t" // ovfl ? 0x80 : 0
    "shr.u32     s,t,7;         \n\t" // ovfl ? 1 : 0
    "not.b32     u,t;           \n\t" // ovfl ? 0x7f : 0xff
    "and.b32     r,r,u;         \n\t" // ovfl ? (a - b) & 0x7f : a - b
    "and.b32     u,a,t;         \n\t" // ovfl ? a & 0x80 : 0
    "sub.u32     t,t,s;         \n\t" // ovfl ? 0x7f : 0
    "shr.u32     u,u,7;         \n\t" // ovfl ? sign(a) : 0
    "or.b32      r,r,t;         \n\t" // ovfl ? 0x7f : a - b
    "add.u32     r,r,u;         \n\t" // ovfl ? 0x7f+sign(a) : a - b
#endif /* __CUDA_ARCH__ >= 200 */
    "mov.b32     %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise difference with signed saturation
}

static __forceinline__ unsigned int __vsubus4(unsigned int a, unsigned int b)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int c = 0;
  asm("vsub4.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else /* __CUDA_ARCH__ >= 300 */
  // This code uses the same basic approach used for the non-saturating
  // subtraction. The seven low-order bits in each byte are subtracted by
  // regular subtraction with the inverse of the borrow-out from bit 6 (=
  // inverse of borrow-in for the msb) being recorded in bit 7, while the
  // msb is handled separately.
  //
  // Clamping to 0 needs happens when there is a borrow-out from the msb.
  // This is simply accomplished by ANDing the normal addition result with
  // a mask based on the inverted msb borrow-out: ~borrow-out ? 0xff : 0x00.
  // The borrow-out information is generated from the msb. Since we already
  // have the msb's ~borrow-in and (a^~b) available from the computation of
  // the msb result bit, the most efficient way to compute msb ~borrow-out
  // is: ((a ^ ~b) & ~borrow-in) | (~b & a). The truth table for the msb is
  //
  // a b bw-in res ~bw-out a^~b (a^~b)&~bw-in (a&~b) ((a^~b)&~bw-in)|(a&~b)
  //
  // 0 0  0     0     1      1        1          0          1
  // 0 0  1     1     0      1        0          0          0
  // 0 1  0     1     0      0        0          0          0
  // 0 1  1     0     0      0        0          0          0
  // 1 0  0     1     1      0        0          1          1
  // 1 0  1     0     1      0        0          1          1
  // 1 1  0     0     1      1        1          0          1
  // 1 1  1     1     0      1        0          0          0
  //
  asm(
    "{                       \n\t"
    ".reg .u32 a,b,r,s,t,u;  \n\t"
    "mov.b32  a,%1;          \n\t"
    "mov.b32  b,%2;          \n\t"
    "not.b32  u,b;           \n\t" // ~b
    "xor.b32  s,u,a;         \n\t" // a ^ ~b
    "and.b32  u,u,a;         \n\t" // a & ~b
    "or.b32   r,a,0x80808080;\n\t" // set msbs
    "and.b32  t,b,0x7f7f7f7f;\n\t" // clear msbs
    "sub.u32  r,r,t;         \n\t" // subtract lsbs, msb: ~borrow-in
    "and.b32  t,r,s;         \n\t" // msb: (a ^ ~b) & ~borrow-in
    "and.b32  s,s,0x80808080;\n\t" // msb: a ^ ~b
    "xor.b32  r,r,s;         \n\t" // msb: res = a ^ ~b ^ ~borrow-in
    "or.b32   t,t,u;         \n\t" // msb: bw-out = ((a^~b)&~bw-in)|(a&~b)
#if __CUDA_ARCH__ >= 200
    "prmt.b32 t,t,0,0xba98;  \n\t" // ~borrow-out ? 0xff : 0
#else /* __CUDA_ARCH__ >= 200 */
    "and.b32  t,t,0x80808080;\n\t" // isolate msb: ~borrow-out
    "shr.u32  s,t,7;         \n\t" // build mask
    "sub.u32  s,t,s;         \n\t" //  from
    "or.b32   t,t,s;         \n\t" //   msb
#endif /* __CUDA_ARCH__ >= 200 */
    "and.b32  r,r,t;         \n\t" // cond. clear result if msb borrow-out
    "mov.b32  %0,r;          \n\t"
    "}"
    : "=r"(r)
    : "r"(a), "r"(b));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise difference with unsigned saturation
}

static __forceinline__ unsigned int __vneg4(unsigned int a)
{
  return __vsub4(0, a); // byte-wise negation with wrap-around
}

static __forceinline__ unsigned int __vnegss4(unsigned int a)
{
  unsigned int r;
#if __CUDA_ARCH__ >= 300
  unsigned int s = 0;
  asm("vsub4.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r) : "r"(s), "r"(a), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  r = __vsub4(0, a); //
  asm(
    "{                       \n\t"
    ".reg .u32 a, r, s;      \n\t"
    "mov.b32  r,%0;          \n\t"
    "mov.b32  a,%1;          \n\t"
    "and.b32  a,a,0x80808080;\n\t" // extract msb
    "and.b32  s,a,r;         \n\t" // wrap-around if msb set in a and -a
    "shr.u32  s,s,7;         \n\t" // msb ? 1 : 0
    "sub.u32  r,r,s;         \n\t" // subtract 1 if result wrapped around
    "mov.b32  %0,r;          \n\t"
    "}"
    : "+r"(r)
    : "r"(a));
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise negation with signed saturation
}

static __forceinline__ unsigned int __vabsdiffs4(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vabsdiff4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  s = __vcmpges4(a, b); // mask = 0xff if a >= b
  r = a ^ b;            //
  s = (r & s) ^ b;      // select a when a >= b, else select b => max(a,b)
  r = s ^ r;            // select a when b >= a, else select b => min(a,b)
  r = __vsub4(s, r);    // |a - b| = max(a,b) - min(a,b);
#endif /* __CUDA_ARCH__ >= 300 */
  return r; // byte-wise absolute difference of signed integers
}

static __forceinline__ unsigned int __vsads4(unsigned int a, unsigned int b)
{
  unsigned int r, s;
#if __CUDA_ARCH__ >= 300
  s = 0;
  asm("vabsdiff4.s32.s32.s32.add %0,%1,%2,%3;"
      : "=r"(r)
      : "r"(a), "r"(b), "r"(s));
#else /* __CUDA_ARCH__ >= 300 */
  r = __vabsdiffs4(a, b);
  s = r >> 8;
  r = (r & 0x00ff00ff) + (s & 0x00ff00ff);
  r = ((r << 16) + r) >> 16;
#endif /*  __CUDA_ARCH__ >= 300 */
  return r; // byte-wise sum of absolute differences of signed ints
}

/*******************************************************************************
 *                                                                             *
 *                             END SIMD functions                              *
 *                                                                             *
 *******************************************************************************/

#endif /* __cplusplus && __CUDACC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
/*
#include "sm_11_atomic_functions.h"
#include "sm_12_atomic_functions.h"
#include "sm_13_double_functions.h"
#include "sm_20_atomic_functions.h"
#include "sm_32_atomic_functions.h"
#include "sm_35_atomic_functions.h"
#include "sm_20_intrinsics.h"
#include "sm_30_intrinsics.h"
#include "sm_32_intrinsics.h"
#include "sm_35_intrinsics.h"
#include "surface_functions.h"
#include "texture_fetch_functions.h"
#include "texture_indirect_functions.h"
#include "surface_indirect_functions.h"
*/
#endif /* !__DEVICE_FUNCTIONS_H__ */
