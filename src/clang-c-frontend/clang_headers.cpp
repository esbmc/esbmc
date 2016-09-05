#include "clang_c_language.h"

struct hooked_header {
  const char *basename;
  char *textstart;
  unsigned int *textsize;
};

extern "C" {
  extern char clang_Intrin_buf[];
  extern unsigned int clang_Intrin_buf_size;

  extern char clang___clang_cuda_cmath_buf[];
  extern unsigned int clang___clang_cuda_cmath_buf_size;

  extern char clang___clang_cuda_intrinsics_buf[];
  extern unsigned int clang___clang_cuda_intrinsics_buf_size;

  extern char clang___clang_cuda_math_forward_declares_buf[];
  extern unsigned int clang___clang_cuda_math_forward_declares_buf_size;

  extern char clang___clang_cuda_runtime_wrapper_buf[];
  extern unsigned int clang___clang_cuda_runtime_wrapper_buf_size;

  extern char clang___stddef_max_align_t_buf[];
  extern unsigned int clang___stddef_max_align_t_buf_size;

  extern char clang___wmmintrin_aes_buf[];
  extern unsigned int clang___wmmintrin_aes_buf_size;

  extern char clang___wmmintrin_pclmul_buf[];
  extern unsigned int clang___wmmintrin_pclmul_buf_size;

  extern char clang_adxintrin_buf[];
  extern unsigned int clang_adxintrin_buf_size;

  extern char clang_ammintrin_buf[];
  extern unsigned int clang_ammintrin_buf_size;

  extern char clang_arm_acle_buf[];
  extern unsigned int clang_arm_acle_buf_size;

  extern char clang_bmi2intrin_buf[];
  extern unsigned int clang_bmi2intrin_buf_size;

  extern char clang_bmiintrin_buf[];
  extern unsigned int clang_bmiintrin_buf_size;

  extern char clang_clflushoptintrin_buf[];
  extern unsigned int clang_clflushoptintrin_buf_size;

  extern char clang_cpuid_buf[];
  extern unsigned int clang_cpuid_buf_size;

  extern char clang_cuda_builtin_vars_buf[];
  extern unsigned int clang_cuda_builtin_vars_buf_size;

  extern char clang_emmintrin_buf[];
  extern unsigned int clang_emmintrin_buf_size;

  extern char clang_f16cintrin_buf[];
  extern unsigned int clang_f16cintrin_buf_size;

  extern char clang_float_buf[];
  extern unsigned int clang_float_buf_size;

  extern char clang_fma4intrin_buf[];
  extern unsigned int clang_fma4intrin_buf_size;

  extern char clang_fmaintrin_buf[];
  extern unsigned int clang_fmaintrin_buf_size;

  extern char clang_fxsrintrin_buf[];
  extern unsigned int clang_fxsrintrin_buf_size;

  extern char clang_htmintrin_buf[];
  extern unsigned int clang_htmintrin_buf_size;

  extern char clang_htmxlintrin_buf[];
  extern unsigned int clang_htmxlintrin_buf_size;

  extern char clang_ia32intrin_buf[];
  extern unsigned int clang_ia32intrin_buf_size;

  extern char clang_immintrin_buf[];
  extern unsigned int clang_immintrin_buf_size;

  extern char clang_intrin_buf[];
  extern unsigned int clang_intrin_buf_size;

  extern char clang_inttypes_buf[];
  extern unsigned int clang_inttypes_buf_size;

  extern char clang_iso646_buf[];
  extern unsigned int clang_iso646_buf_size;

  extern char clang_limits_buf[];
  extern unsigned int clang_limits_buf_size;

  extern char clang_lzcntintrin_buf[];
  extern unsigned int clang_lzcntintrin_buf_size;

  extern char clang_mm3dnow_buf[];
  extern unsigned int clang_mm3dnow_buf_size;

  extern char clang_mm_malloc_buf[];
  extern unsigned int clang_mm_malloc_buf_size;

  extern char clang_mmintrin_buf[];
  extern unsigned int clang_mmintrin_buf_size;

  extern char clang_mwaitxintrin_buf[];
  extern unsigned int clang_mwaitxintrin_buf_size;

  extern char clang_nmmintrin_buf[];
  extern unsigned int clang_nmmintrin_buf_size;

  extern char clang_omp_buf[];
  extern unsigned int clang_omp_buf_size;

  extern char clang_opencl_c_buf[];
  extern unsigned int clang_opencl_c_buf_size;

  extern char clang_pkuintrin_buf[];
  extern unsigned int clang_pkuintrin_buf_size;

  extern char clang_pmmintrin_buf[];
  extern unsigned int clang_pmmintrin_buf_size;

  extern char clang_popcntintrin_buf[];
  extern unsigned int clang_popcntintrin_buf_size;

  extern char clang_prfchwintrin_buf[];
  extern unsigned int clang_prfchwintrin_buf_size;

  extern char clang_rdseedintrin_buf[];
  extern unsigned int clang_rdseedintrin_buf_size;

  extern char clang_rtmintrin_buf[];
  extern unsigned int clang_rtmintrin_buf_size;

  extern char clang_s390intrin_buf[];
  extern unsigned int clang_s390intrin_buf_size;

  extern char clang_shaintrin_buf[];
  extern unsigned int clang_shaintrin_buf_size;

  extern char clang_smmintrin_buf[];
  extern unsigned int clang_smmintrin_buf_size;

  extern char clang_stdalign_buf[];
  extern unsigned int clang_stdalign_buf_size;

  extern char clang_stdarg_buf[];
  extern unsigned int clang_stdarg_buf_size;

  extern char clang_stdatomic_buf[];
  extern unsigned int clang_stdatomic_buf_size;

  extern char clang_stdbool_buf[];
  extern unsigned int clang_stdbool_buf_size;

  extern char clang_stddef_buf[];
  extern unsigned int clang_stddef_buf_size;

  extern char clang_stdint_buf[];
  extern unsigned int clang_stdint_buf_size;

  extern char clang_stdnoreturn_buf[];
  extern unsigned int clang_stdnoreturn_buf_size;

  extern char clang_tbmintrin_buf[];
  extern unsigned int clang_tbmintrin_buf_size;

  extern char clang_tgmath_buf[];
  extern unsigned int clang_tgmath_buf_size;

  extern char clang_tmmintrin_buf[];
  extern unsigned int clang_tmmintrin_buf_size;

  extern char clang_unwind_buf[];
  extern unsigned int clang_unwind_buf_size;

  extern char clang_vadefs_buf[];
  extern unsigned int clang_vadefs_buf_size;

  extern char clang_varargs_buf[];
  extern unsigned int clang_varargs_buf_size;

  extern char clang_wmmintrin_buf[];
  extern unsigned int clang_wmmintrin_buf_size;

  extern char clang_x86intrin_buf[];
  extern unsigned int clang_x86intrin_buf_size;

  extern char clang_xmmintrin_buf[];
  extern unsigned int clang_xmmintrin_buf_size;

  extern char clang_xopintrin_buf[];
  extern unsigned int clang_xopintrin_buf_size;

  extern char clang_xsavecintrin_buf[];
  extern unsigned int clang_xsavecintrin_buf_size;

  extern char clang_xsaveintrin_buf[];
  extern unsigned int clang_xsaveintrin_buf_size;

  extern char clang_xsaveoptintrin_buf[];
  extern unsigned int clang_xsaveoptintrin_buf_size;

  extern char clang_xsavesintrin_buf[];
  extern unsigned int clang_xsavesintrin_buf_size;

  extern char clang_xtestintrin_buf[];
  extern unsigned int clang_xtestintrin_buf_size;

  extern char clang_allocator_interface_buf[];
  extern unsigned int clang_allocator_interface_buf_size;

  extern char clang_asan_interface_buf[];
  extern unsigned int clang_asan_interface_buf_size;

  extern char clang_common_interface_defs_buf[];
  extern unsigned int clang_common_interface_defs_buf_size;

  extern char clang_coverage_interface_buf[];
  extern unsigned int clang_coverage_interface_buf_size;

  extern char clang_dfsan_interface_buf[];
  extern unsigned int clang_dfsan_interface_buf_size;

  extern char clang_esan_interface_buf[];
  extern unsigned int clang_esan_interface_buf_size;

  extern char clang_linux_syscall_hooks_buf[];
  extern unsigned int clang_linux_syscall_hooks_buf_size;

  extern char clang_lsan_interface_buf[];
  extern unsigned int clang_lsan_interface_buf_size;

  extern char clang_msan_interface_buf[];
  extern unsigned int clang_msan_interface_buf_size;

  extern char clang_tsan_interface_atomic_buf[];
  extern unsigned int clang_tsan_interface_atomic_buf_size;

  struct hooked_header clang_headers[] = {
    { "Intrin.h", clang_Intrin_buf, &clang_Intrin_buf_size},
    { "__clang_cuda_cmath.h", clang___clang_cuda_cmath_buf, &clang___clang_cuda_cmath_buf_size},
    { "__clang_cuda_intrinsics.h", clang___clang_cuda_intrinsics_buf, &clang___clang_cuda_intrinsics_buf_size},
    { "__clang_cuda_math_forward_declares.h", clang___clang_cuda_math_forward_declares_buf, &clang___clang_cuda_math_forward_declares_buf_size},
    { "__clang_cuda_runtime_wrapper.h", clang___clang_cuda_runtime_wrapper_buf, &clang___clang_cuda_runtime_wrapper_buf_size},
    { "__stddef_max_align_t.h", clang___stddef_max_align_t_buf, &clang___stddef_max_align_t_buf_size},
    { "__wmmintrin_aes.h", clang___wmmintrin_aes_buf, &clang___wmmintrin_aes_buf_size},
    { "__wmmintrin_pclmul.h", clang___wmmintrin_pclmul_buf, &clang___wmmintrin_pclmul_buf_size},
    { "adxintrin.h", clang_adxintrin_buf, &clang_adxintrin_buf_size},
    { "ammintrin.h", clang_ammintrin_buf, &clang_ammintrin_buf_size},
    { "arm_acle.h", clang_arm_acle_buf, &clang_arm_acle_buf_size},
    { "bmi2intrin.h", clang_bmi2intrin_buf, &clang_bmi2intrin_buf_size},
    { "bmiintrin.h", clang_bmiintrin_buf, &clang_bmiintrin_buf_size},
    { "clflushoptintrin.h", clang_clflushoptintrin_buf, &clang_clflushoptintrin_buf_size},
    { "cpuid.h", clang_cpuid_buf, &clang_cpuid_buf_size},
    { "cuda_builtin_vars.h", clang_cuda_builtin_vars_buf, &clang_cuda_builtin_vars_buf_size},
    { "emmintrin.h", clang_emmintrin_buf, &clang_emmintrin_buf_size},
    { "f16cintrin.h", clang_f16cintrin_buf, &clang_f16cintrin_buf_size},
    { "float.h", clang_float_buf, &clang_float_buf_size},
    { "fma4intrin.h", clang_fma4intrin_buf, &clang_fma4intrin_buf_size},
    { "fmaintrin.h", clang_fmaintrin_buf, &clang_fmaintrin_buf_size},
    { "fxsrintrin.h", clang_fxsrintrin_buf, &clang_fxsrintrin_buf_size},
    { "htmintrin.h", clang_htmintrin_buf, &clang_htmintrin_buf_size},
    { "htmxlintrin.h", clang_htmxlintrin_buf, &clang_htmxlintrin_buf_size},
    { "ia32intrin.h", clang_ia32intrin_buf, &clang_ia32intrin_buf_size},
    { "immintrin.h", clang_immintrin_buf, &clang_immintrin_buf_size},
    { "intrin.h", clang_intrin_buf, &clang_intrin_buf_size},
    { "inttypes.h", clang_inttypes_buf, &clang_inttypes_buf_size},
    { "iso646.h", clang_iso646_buf, &clang_iso646_buf_size},
    { "limits.h", clang_limits_buf, &clang_limits_buf_size},
    { "lzcntintrin.h", clang_lzcntintrin_buf, &clang_lzcntintrin_buf_size},
    { "mm3dnow.h", clang_mm3dnow_buf, &clang_mm3dnow_buf_size},
    { "mm_malloc.h", clang_mm_malloc_buf, &clang_mm_malloc_buf_size},
    { "mmintrin.h", clang_mmintrin_buf, &clang_mmintrin_buf_size},
    { "mwaitxintrin.h", clang_mwaitxintrin_buf, &clang_mwaitxintrin_buf_size},
    { "nmmintrin.h", clang_nmmintrin_buf, &clang_nmmintrin_buf_size},
    { "omp.h", clang_omp_buf, &clang_omp_buf_size},
    { "opencl-c.h", clang_opencl_c_buf, &clang_opencl_c_buf_size},
    { "pkuintrin.h", clang_pkuintrin_buf, &clang_pkuintrin_buf_size},
    { "pmmintrin.h", clang_pmmintrin_buf, &clang_pmmintrin_buf_size},
    { "popcntintrin.h", clang_popcntintrin_buf, &clang_popcntintrin_buf_size},
    { "prfchwintrin.h", clang_prfchwintrin_buf, &clang_prfchwintrin_buf_size},
    { "rdseedintrin.h", clang_rdseedintrin_buf, &clang_rdseedintrin_buf_size},
    { "rtmintrin.h", clang_rtmintrin_buf, &clang_rtmintrin_buf_size},
    { "s390intrin.h", clang_s390intrin_buf, &clang_s390intrin_buf_size},
    { "shaintrin.h", clang_shaintrin_buf, &clang_shaintrin_buf_size},
    { "smmintrin.h", clang_smmintrin_buf, &clang_smmintrin_buf_size},
    { "stdalign.h", clang_stdalign_buf, &clang_stdalign_buf_size},
    { "stdarg.h", clang_stdarg_buf, &clang_stdarg_buf_size},
    { "stdatomic.h", clang_stdatomic_buf, &clang_stdatomic_buf_size},
    { "stdbool.h", clang_stdbool_buf, &clang_stdbool_buf_size},
    { "stddef.h", clang_stddef_buf, &clang_stddef_buf_size},
    { "stdint.h", clang_stdint_buf, &clang_stdint_buf_size},
    { "stdnoreturn.h", clang_stdnoreturn_buf, &clang_stdnoreturn_buf_size},
    { "tbmintrin.h", clang_tbmintrin_buf, &clang_tbmintrin_buf_size},
    { "tgmath.h", clang_tgmath_buf, &clang_tgmath_buf_size},
    { "tmmintrin.h", clang_tmmintrin_buf, &clang_tmmintrin_buf_size},
    { "unwind.h", clang_unwind_buf, &clang_unwind_buf_size},
    { "vadefs.h", clang_vadefs_buf, &clang_vadefs_buf_size},
    { "varargs.h", clang_varargs_buf, &clang_varargs_buf_size},
    { "wmmintrin.h", clang_wmmintrin_buf, &clang_wmmintrin_buf_size},
    { "x86intrin.h", clang_x86intrin_buf, &clang_x86intrin_buf_size},
    { "xmmintrin.h", clang_xmmintrin_buf, &clang_xmmintrin_buf_size},
    { "xopintrin.h", clang_xopintrin_buf, &clang_xopintrin_buf_size},
    { "xsavecintrin.h", clang_xsavecintrin_buf, &clang_xsavecintrin_buf_size},
    { "xsaveintrin.h", clang_xsaveintrin_buf, &clang_xsaveintrin_buf_size},
    { "xsaveoptintrin.h", clang_xsaveoptintrin_buf, &clang_xsaveoptintrin_buf_size},
    { "xsavesintrin.h", clang_xsavesintrin_buf, &clang_xsavesintrin_buf_size},
    { "xtestintrin.h", clang_xtestintrin_buf, &clang_xtestintrin_buf_size},

    // Sanitizer headers, do we really need them?
    { "allocator_interface.h", clang_allocator_interface_buf, &clang_allocator_interface_buf_size},
    { "asan_interface.h", clang_asan_interface_buf, &clang_asan_interface_buf_size},
    { "common_interface_defs.h", clang_common_interface_defs_buf, &clang_common_interface_defs_buf_size},
    { "coverage_interface.h", clang_coverage_interface_buf, &clang_coverage_interface_buf_size},
    { "dfsan_interface.h", clang_dfsan_interface_buf, &clang_dfsan_interface_buf_size},
    { "esan_interface.h", clang_esan_interface_buf, &clang_esan_interface_buf_size},
    { "linux_syscall_hooks.h", clang_linux_syscall_hooks_buf, &clang_linux_syscall_hooks_buf_size},
    { "lsan_interface.h", clang_lsan_interface_buf, &clang_lsan_interface_buf_size},
    { "msan_interface.h", clang_msan_interface_buf, &clang_msan_interface_buf_size},
    { "tsan_interface_atomic.h", clang_tsan_interface_atomic_buf, &clang_tsan_interface_atomic_buf_size},

    { NULL, NULL, NULL}
  };
}

void clang_c_languaget::add_clang_headers()
{
  struct hooked_header *h;
  for (h = &clang_headers[0]; h->basename != NULL; h++) {
    clang_headers_name.emplace_back(h->basename);
    clang_headers_content.emplace_back(h->textstart, *h->textsize);
  }
}
