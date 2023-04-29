#ifndef __HOST_DEFINES_H__
#define __HOST_DEFINES_H__

#ifdef __cplusplus

extern "C"
{
#endif

#if defined(__GNUC__) || defined(__CUDA_LIBDEVICE__)

#define __no_return__ __attribute__((noreturn))

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
/* gcc allows users to define attributes with underscores, 
   e.g., __attribute__((__noinline__)).
   Consider a non-CUDA source file (e.g. .cpp) that has the 
   above attribute specification, and includes this header file. In that case,
   defining __noinline__ as below  would cause a gcc compilation error.
   Hence, only define __noinline__ when the code is being processed
   by a  CUDA compiler component.
*/
#define __noinline__ __attribute__((noinline))
#endif /* __CUDACC__  || __CUDA_ARCH__ */

#define __forceinline__ __inline__ __attribute__((always_inline))
#define __align__(n) __attribute__((aligned(n)))
#define __thread__ __thread
#define __import__
#define __export__
#define __cdecl
#define __annotate__(a) __attribute__((a))
#define __location__(a) __annotate__(a)
#define CUDARTAPI

#elif defined(_MSC_VER)

#if _MSC_VER >= 1400

#define __restrict__ __restrict

#else /* _MSC_VER >= 1400 */

#define __restrict__

#endif /* _MSC_VER >= 1400 */

#define __inline__ __inline
#define __no_return__ __declspec(noreturn)
#define __noinline__ __declspec(noinline)
#define __forceinline__ __forceinline
#define __align__(n) __declspec(align(n))
#define __thread__ __declspec(thread)
#define __import__ __declspec(dllimport)
#define __export__ __declspec(dllexport)
#define __annotate__(a) __declspec(a)
#define __location__(a) __annotate__(__##a##__)
#define CUDARTAPI __stdcall

#else /* __GNUC__ || __CUDA_LIBDEVICE__ */

#define __inline__

#if !defined(__align__)

#error--- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for '__align__' !!! ---

#endif /* !__align__ */

#if !defined(CUDARTAPI)

#error--- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for 'CUDARTAPI' !!! ---

#endif /* !CUDARTAPI */

#endif /* !__GNUC__ */

#if !defined(__GNUC__) || __GNUC__ < 4 ||                                      \
  (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !defined(__clang__))

#define __specialization_static static

#else /* !__GNUC__ || __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3) */

#define __specialization_static

#endif /* !__GNUC__ || __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3) */

#if !defined(__CUDACC__) && !defined(__CUDABE__)

#undef __annotate__
#define __annotate__(a)

#else /* !__CUDACC__ && !__CUDABE__ */

#define __launch_bounds__(...) __annotate__(launch_bounds(__VA_ARGS__))

#endif /* !__CUDACC__ && !__CUDABE__ */

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(__GNUC__) ||         \
  defined(_WIN64)

#define __builtin_align__(a) __align__(a)

#else /* __CUDACC__ || __CUDABE__ || __GNUC__ || _WIN64 */

#define __builtin_align__(a)

#endif /* __CUDACC__ || __CUDABE__ || __GNUC__  || _WIN64 */

#define __host__ __location__(host)
#define __device__ __location__(device)
#define __global__ __location__(global)
//#define __shared__ \
        __location__(shared)
#define __constant__ __location__(constant)
#define __managed__ __location__(managed)

#if defined(__CUDABE__) || !defined(__CUDACC__)
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __cudart_builtin__
#else /* __CUDABE__  || !__CUDACC__ */
#define __device_builtin__ __location__(device_builtin)
#define __device_builtin_texture_type__                                        \
  __location__(device_builtin_texture_type)
#define __device_builtin_surface_type__                                        \
  __location__(device_builtin_surface_type)
#define __cudart_builtin__ __location__(cudart_builtin)
#endif /* __CUDABE__ || !__CUDACC__ */

#ifdef __cplusplus
}

#endif

#endif /* !__HOST_DEFINES_H__ */
