#include <clang-c-frontend/clang_c_language.h>
#include <fstream>

struct hooked_header
{
  const char *basename;
  char *textstart;
  unsigned int *textsize;
};

extern "C"
{
  extern char __clang_cuda_builtin_vars_buf[];
  extern unsigned int __clang_cuda_builtin_vars_buf_size;

  extern char __clang_cuda_cmath_buf[];
  extern unsigned int __clang_cuda_cmath_buf_size;

  extern char __clang_cuda_complex_builtins_buf[];
  extern unsigned int __clang_cuda_complex_builtins_buf_size;

  extern char __clang_cuda_device_functions_buf[];
  extern unsigned int __clang_cuda_device_functions_buf_size;

  extern char __clang_cuda_intrinsics_buf[];
  extern unsigned int __clang_cuda_intrinsics_buf_size;

  extern char __clang_cuda_libdevice_declares_buf[];
  extern unsigned int __clang_cuda_libdevice_declares_buf_size;

  extern char __clang_cuda_math_buf[];
  extern unsigned int __clang_cuda_math_buf_size;

  extern char __clang_cuda_math_forward_declares_buf[];
  extern unsigned int __clang_cuda_math_forward_declares_buf_size;

  extern char __clang_cuda_runtime_wrapper_buf[];
  extern unsigned int __clang_cuda_runtime_wrapper_buf_size;

  extern char __clang_hip_libdevice_declares_buf[];
  extern unsigned int __clang_hip_libdevice_declares_buf_size;

  extern char __clang_hip_math_buf[];
  extern unsigned int __clang_hip_math_buf_size;

  extern char __clang_hip_runtime_wrapper_buf[];
  extern unsigned int __clang_hip_runtime_wrapper_buf_size;

  extern char __stddef_max_align_t_buf[];
  extern unsigned int __stddef_max_align_t_buf_size;

  extern char __wmmintrin_aes_buf[];
  extern unsigned int __wmmintrin_aes_buf_size;

  extern char __wmmintrin_pclmul_buf[];
  extern unsigned int __wmmintrin_pclmul_buf_size;

  extern char adxintrin_buf[];
  extern unsigned int adxintrin_buf_size;

  extern char altivec_buf[];
  extern unsigned int altivec_buf_size;

  extern char ammintrin_buf[];
  extern unsigned int ammintrin_buf_size;

  extern char amxintrin_buf[];
  extern unsigned int amxintrin_buf_size;

  extern char arm64intr_buf[];
  extern unsigned int arm64intr_buf_size;

  extern char arm_acle_buf[];
  extern unsigned int arm_acle_buf_size;

  extern char arm_bf16_buf[];
  extern unsigned int arm_bf16_buf_size;

  extern char arm_cde_buf[];
  extern unsigned int arm_cde_buf_size;

  extern char arm_cmse_buf[];
  extern unsigned int arm_cmse_buf_size;

  extern char arm_fp16_buf[];
  extern unsigned int arm_fp16_buf_size;

  extern char arm_mve_buf[];
  extern unsigned int arm_mve_buf_size;

  extern char arm_neon_buf[];
  extern unsigned int arm_neon_buf_size;

  extern char arm_sve_buf[];
  extern unsigned int arm_sve_buf_size;

  extern char armintr_buf[];
  extern unsigned int armintr_buf_size;

  extern char avx2intrin_buf[];
  extern unsigned int avx2intrin_buf_size;

  extern char avx512bf16intrin_buf[];
  extern unsigned int avx512bf16intrin_buf_size;

  extern char avx512bitalgintrin_buf[];
  extern unsigned int avx512bitalgintrin_buf_size;

  extern char avx512bwintrin_buf[];
  extern unsigned int avx512bwintrin_buf_size;

  extern char avx512cdintrin_buf[];
  extern unsigned int avx512cdintrin_buf_size;

  extern char avx512dqintrin_buf[];
  extern unsigned int avx512dqintrin_buf_size;

  extern char avx512erintrin_buf[];
  extern unsigned int avx512erintrin_buf_size;

  extern char avx512fintrin_buf[];
  extern unsigned int avx512fintrin_buf_size;

  extern char avx512ifmaintrin_buf[];
  extern unsigned int avx512ifmaintrin_buf_size;

  extern char avx512ifmavlintrin_buf[];
  extern unsigned int avx512ifmavlintrin_buf_size;

  extern char avx512pfintrin_buf[];
  extern unsigned int avx512pfintrin_buf_size;

  extern char avx512vbmi2intrin_buf[];
  extern unsigned int avx512vbmi2intrin_buf_size;

  extern char avx512vbmiintrin_buf[];
  extern unsigned int avx512vbmiintrin_buf_size;

  extern char avx512vbmivlintrin_buf[];
  extern unsigned int avx512vbmivlintrin_buf_size;

  extern char avx512vlbf16intrin_buf[];
  extern unsigned int avx512vlbf16intrin_buf_size;

  extern char avx512vlbitalgintrin_buf[];
  extern unsigned int avx512vlbitalgintrin_buf_size;

  extern char avx512vlbwintrin_buf[];
  extern unsigned int avx512vlbwintrin_buf_size;

  extern char avx512vlcdintrin_buf[];
  extern unsigned int avx512vlcdintrin_buf_size;

  extern char avx512vldqintrin_buf[];
  extern unsigned int avx512vldqintrin_buf_size;

  extern char avx512vlintrin_buf[];
  extern unsigned int avx512vlintrin_buf_size;

  extern char avx512vlvbmi2intrin_buf[];
  extern unsigned int avx512vlvbmi2intrin_buf_size;

  extern char avx512vlvnniintrin_buf[];
  extern unsigned int avx512vlvnniintrin_buf_size;

  extern char avx512vlvp2intersectintrin_buf[];
  extern unsigned int avx512vlvp2intersectintrin_buf_size;

  extern char avx512vnniintrin_buf[];
  extern unsigned int avx512vnniintrin_buf_size;

  extern char avx512vp2intersectintrin_buf[];
  extern unsigned int avx512vp2intersectintrin_buf_size;

  extern char avx512vpopcntdqintrin_buf[];
  extern unsigned int avx512vpopcntdqintrin_buf_size;

  extern char avx512vpopcntdqvlintrin_buf[];
  extern unsigned int avx512vpopcntdqvlintrin_buf_size;

  extern char avxintrin_buf[];
  extern unsigned int avxintrin_buf_size;

  extern char bmi2intrin_buf[];
  extern unsigned int bmi2intrin_buf_size;

  extern char bmiintrin_buf[];
  extern unsigned int bmiintrin_buf_size;

  extern char cet_buf[];
  extern unsigned int cet_buf_size;

  extern char cetintrin_buf[];
  extern unsigned int cetintrin_buf_size;

  extern char cldemoteintrin_buf[];
  extern unsigned int cldemoteintrin_buf_size;

  extern char clflushoptintrin_buf[];
  extern unsigned int clflushoptintrin_buf_size;

  extern char clwbintrin_buf[];
  extern unsigned int clwbintrin_buf_size;

  extern char clzerointrin_buf[];
  extern unsigned int clzerointrin_buf_size;

  extern char cpuid_buf[];
  extern unsigned int cpuid_buf_size;

  extern char emmintrin_buf[];
  extern unsigned int emmintrin_buf_size;

  extern char enqcmdintrin_buf[];
  extern unsigned int enqcmdintrin_buf_size;

  extern char f16cintrin_buf[];
  extern unsigned int f16cintrin_buf_size;

  extern char float_buf[];
  extern unsigned int float_buf_size;

  extern char fma4intrin_buf[];
  extern unsigned int fma4intrin_buf_size;

  extern char fmaintrin_buf[];
  extern unsigned int fmaintrin_buf_size;

  extern char fxsrintrin_buf[];
  extern unsigned int fxsrintrin_buf_size;

  extern char gfniintrin_buf[];
  extern unsigned int gfniintrin_buf_size;

  extern char htmintrin_buf[];
  extern unsigned int htmintrin_buf_size;

  extern char htmxlintrin_buf[];
  extern unsigned int htmxlintrin_buf_size;

  extern char ia32intrin_buf[];
  extern unsigned int ia32intrin_buf_size;

  extern char immintrin_buf[];
  extern unsigned int immintrin_buf_size;

  extern char intrin_buf[];
  extern unsigned int intrin_buf_size;

  extern char inttypes_buf[];
  extern unsigned int inttypes_buf_size;

  extern char invpcidintrin_buf[];
  extern unsigned int invpcidintrin_buf_size;

  extern char iso646_buf[];
  extern unsigned int iso646_buf_size;

  extern char limits_buf[];
  extern unsigned int limits_buf_size;

  extern char lwpintrin_buf[];
  extern unsigned int lwpintrin_buf_size;

  extern char lzcntintrin_buf[];
  extern unsigned int lzcntintrin_buf_size;

  extern char mm3dnow_buf[];
  extern unsigned int mm3dnow_buf_size;

  extern char mm_malloc_buf[];
  extern unsigned int mm_malloc_buf_size;

  extern char mmintrin_buf[];
  extern unsigned int mmintrin_buf_size;

  extern char movdirintrin_buf[];
  extern unsigned int movdirintrin_buf_size;

  extern char msa_buf[];
  extern unsigned int msa_buf_size;

  extern char mwaitxintrin_buf[];
  extern unsigned int mwaitxintrin_buf_size;

  extern char nmmintrin_buf[];
  extern unsigned int nmmintrin_buf_size;

  extern char omp_tools_buf[];
  extern unsigned int omp_tools_buf_size;

  extern char omp_buf[];
  extern unsigned int omp_buf_size;

  extern char ompt_buf[];
  extern unsigned int ompt_buf_size;

  extern char opencl_c_base_buf[];
  extern unsigned int opencl_c_base_buf_size;

  extern char opencl_c_buf[];
  extern unsigned int opencl_c_buf_size;

  extern char pconfigintrin_buf[];
  extern unsigned int pconfigintrin_buf_size;

  extern char pkuintrin_buf[];
  extern unsigned int pkuintrin_buf_size;

  extern char pmmintrin_buf[];
  extern unsigned int pmmintrin_buf_size;

  extern char popcntintrin_buf[];
  extern unsigned int popcntintrin_buf_size;

  extern char prfchwintrin_buf[];
  extern unsigned int prfchwintrin_buf_size;

  extern char ptwriteintrin_buf[];
  extern unsigned int ptwriteintrin_buf_size;

  extern char rdseedintrin_buf[];
  extern unsigned int rdseedintrin_buf_size;

  extern char rtmintrin_buf[];
  extern unsigned int rtmintrin_buf_size;

  extern char s390intrin_buf[];
  extern unsigned int s390intrin_buf_size;

  extern char serializeintrin_buf[];
  extern unsigned int serializeintrin_buf_size;

  extern char sgxintrin_buf[];
  extern unsigned int sgxintrin_buf_size;

  extern char shaintrin_buf[];
  extern unsigned int shaintrin_buf_size;

  extern char smmintrin_buf[];
  extern unsigned int smmintrin_buf_size;

  extern char stdalign_buf[];
  extern unsigned int stdalign_buf_size;

  extern char stdarg_buf[];
  extern unsigned int stdarg_buf_size;

  extern char stdatomic_buf[];
  extern unsigned int stdatomic_buf_size;

  extern char stdbool_buf[];
  extern unsigned int stdbool_buf_size;

  extern char stddef_buf[];
  extern unsigned int stddef_buf_size;

  extern char stdint_buf[];
  extern unsigned int stdint_buf_size;

  extern char stdnoreturn_buf[];
  extern unsigned int stdnoreturn_buf_size;

  extern char tbmintrin_buf[];
  extern unsigned int tbmintrin_buf_size;

  extern char tgmath_buf[];
  extern unsigned int tgmath_buf_size;

  extern char tmmintrin_buf[];
  extern unsigned int tmmintrin_buf_size;

  extern char tsxldtrkintrin_buf[];
  extern unsigned int tsxldtrkintrin_buf_size;

  extern char unwind_buf[];
  extern unsigned int unwind_buf_size;

  extern char vadefs_buf[];
  extern unsigned int vadefs_buf_size;

  extern char vaesintrin_buf[];
  extern unsigned int vaesintrin_buf_size;

  extern char varargs_buf[];
  extern unsigned int varargs_buf_size;

  extern char vecintrin_buf[];
  extern unsigned int vecintrin_buf_size;

  extern char vpclmulqdqintrin_buf[];
  extern unsigned int vpclmulqdqintrin_buf_size;

  extern char waitpkgintrin_buf[];
  extern unsigned int waitpkgintrin_buf_size;

  extern char wasm_simd128_buf[];
  extern unsigned int wasm_simd128_buf_size;

  extern char wbnoinvdintrin_buf[];
  extern unsigned int wbnoinvdintrin_buf_size;

  extern char wmmintrin_buf[];
  extern unsigned int wmmintrin_buf_size;

  extern char x86intrin_buf[];
  extern unsigned int x86intrin_buf_size;

  extern char xmmintrin_buf[];
  extern unsigned int xmmintrin_buf_size;

  extern char xopintrin_buf[];
  extern unsigned int xopintrin_buf_size;

  extern char xsavecintrin_buf[];
  extern unsigned int xsavecintrin_buf_size;

  extern char xsaveintrin_buf[];
  extern unsigned int xsaveintrin_buf_size;

  extern char xsaveoptintrin_buf[];
  extern unsigned int xsaveoptintrin_buf_size;

  extern char xsavesintrin_buf[];
  extern unsigned int xsavesintrin_buf_size;

  extern char xtestintrin_buf[];
  extern unsigned int xtestintrin_buf_size;

  struct hooked_header clang_headers[] = {
    {"__clang_cuda_builtin_vars.h",
     __clang_cuda_builtin_vars_buf,
     &__clang_cuda_builtin_vars_buf_size},
    {"__clang_cuda_cmath.h",
     __clang_cuda_cmath_buf,
     &__clang_cuda_cmath_buf_size},
    {"__clang_cuda_complex_builtins.h",
     __clang_cuda_complex_builtins_buf,
     &__clang_cuda_complex_builtins_buf_size},
    {"__clang_cuda_device_functions.h",
     __clang_cuda_device_functions_buf,
     &__clang_cuda_device_functions_buf_size},
    {"__clang_cuda_intrinsics.h",
     __clang_cuda_intrinsics_buf,
     &__clang_cuda_intrinsics_buf_size},
    {"__clang_cuda_libdevice_declares.h",
     __clang_cuda_libdevice_declares_buf,
     &__clang_cuda_libdevice_declares_buf_size},
    {"__clang_cuda_math.h", __clang_cuda_math_buf, &__clang_cuda_math_buf_size},
    {"__clang_cuda_math_forward_declares.h",
     __clang_cuda_math_forward_declares_buf,
     &__clang_cuda_math_forward_declares_buf_size},
    {"__clang_cuda_runtime_wrapper.h",
     __clang_cuda_runtime_wrapper_buf,
     &__clang_cuda_runtime_wrapper_buf_size},
    {"__clang_hip_libdevice_declares.h",
     __clang_hip_libdevice_declares_buf,
     &__clang_hip_libdevice_declares_buf_size},
    {"__clang_hip_math.h", __clang_hip_math_buf, &__clang_hip_math_buf_size},
    {"__clang_hip_runtime_wrapper.h",
     __clang_hip_runtime_wrapper_buf,
     &__clang_hip_runtime_wrapper_buf_size},
    {"__stddef_max_align_t.h",
     __stddef_max_align_t_buf,
     &__stddef_max_align_t_buf_size},
    {"__wmmintrin_aes.h", __wmmintrin_aes_buf, &__wmmintrin_aes_buf_size},
    {"__wmmintrin_pclmul.h",
     __wmmintrin_pclmul_buf,
     &__wmmintrin_pclmul_buf_size},
    {"adxintrin.h", adxintrin_buf, &adxintrin_buf_size},
    {"altivec.h", altivec_buf, &altivec_buf_size},
    {"ammintrin.h", ammintrin_buf, &ammintrin_buf_size},
    {"amxintrin.h", amxintrin_buf, &amxintrin_buf_size},
    {"arm64intr.h", arm64intr_buf, &arm64intr_buf_size},
    {"arm_acle.h", arm_acle_buf, &arm_acle_buf_size},
    {"arm_bf16.h", arm_bf16_buf, &arm_bf16_buf_size},
    {"arm_cde.h", arm_cde_buf, &arm_cde_buf_size},
    {"arm_cmse.h", arm_cmse_buf, &arm_cmse_buf_size},
    {"arm_fp16.h", arm_fp16_buf, &arm_fp16_buf_size},
    {"arm_mve.h", arm_mve_buf, &arm_mve_buf_size},
    {"arm_neon.h", arm_neon_buf, &arm_neon_buf_size},
    {"arm_sve.h", arm_sve_buf, &arm_sve_buf_size},
    {"armintr.h", armintr_buf, &armintr_buf_size},
    {"avx2intrin.h", avx2intrin_buf, &avx2intrin_buf_size},
    {"avx512bf16intrin.h", avx512bf16intrin_buf, &avx512bf16intrin_buf_size},
    {"avx512bitalgintrin.h",
     avx512bitalgintrin_buf,
     &avx512bitalgintrin_buf_size},
    {"avx512bwintrin.h", avx512bwintrin_buf, &avx512bwintrin_buf_size},
    {"avx512cdintrin.h", avx512cdintrin_buf, &avx512cdintrin_buf_size},
    {"avx512dqintrin.h", avx512dqintrin_buf, &avx512dqintrin_buf_size},
    {"avx512erintrin.h", avx512erintrin_buf, &avx512erintrin_buf_size},
    {"avx512fintrin.h", avx512fintrin_buf, &avx512fintrin_buf_size},
    {"avx512ifmaintrin.h", avx512ifmaintrin_buf, &avx512ifmaintrin_buf_size},
    {"avx512ifmavlintrin.h",
     avx512ifmavlintrin_buf,
     &avx512ifmavlintrin_buf_size},
    {"avx512pfintrin.h", avx512pfintrin_buf, &avx512pfintrin_buf_size},
    {"avx512vbmi2intrin.h", avx512vbmi2intrin_buf, &avx512vbmi2intrin_buf_size},
    {"avx512vbmiintrin.h", avx512vbmiintrin_buf, &avx512vbmiintrin_buf_size},
    {"avx512vbmivlintrin.h",
     avx512vbmivlintrin_buf,
     &avx512vbmivlintrin_buf_size},
    {"avx512vlbf16intrin.h",
     avx512vlbf16intrin_buf,
     &avx512vlbf16intrin_buf_size},
    {"avx512vlbitalgintrin.h",
     avx512vlbitalgintrin_buf,
     &avx512vlbitalgintrin_buf_size},
    {"avx512vlbwintrin.h", avx512vlbwintrin_buf, &avx512vlbwintrin_buf_size},
    {"avx512vlcdintrin.h", avx512vlcdintrin_buf, &avx512vlcdintrin_buf_size},
    {"avx512vldqintrin.h", avx512vldqintrin_buf, &avx512vldqintrin_buf_size},
    {"avx512vlintrin.h", avx512vlintrin_buf, &avx512vlintrin_buf_size},
    {"avx512vlvbmi2intrin.h",
     avx512vlvbmi2intrin_buf,
     &avx512vlvbmi2intrin_buf_size},
    {"avx512vlvnniintrin.h",
     avx512vlvnniintrin_buf,
     &avx512vlvnniintrin_buf_size},
    {"avx512vlvp2intersectintrin.h",
     avx512vlvp2intersectintrin_buf,
     &avx512vlvp2intersectintrin_buf_size},
    {"avx512vnniintrin.h", avx512vnniintrin_buf, &avx512vnniintrin_buf_size},
    {"avx512vp2intersectintrin.h",
     avx512vp2intersectintrin_buf,
     &avx512vp2intersectintrin_buf_size},
    {"avx512vpopcntdqintrin.h",
     avx512vpopcntdqintrin_buf,
     &avx512vpopcntdqintrin_buf_size},
    {"avx512vpopcntdqvlintrin.h",
     avx512vpopcntdqvlintrin_buf,
     &avx512vpopcntdqvlintrin_buf_size},
    {"avxintrin.h", avxintrin_buf, &avxintrin_buf_size},
    {"bmi2intrin.h", bmi2intrin_buf, &bmi2intrin_buf_size},
    {"bmiintrin.h", bmiintrin_buf, &bmiintrin_buf_size},
    {"cet.h", cet_buf, &cet_buf_size},
    {"cetintrin.h", cetintrin_buf, &cetintrin_buf_size},
    {"cldemoteintrin.h", cldemoteintrin_buf, &cldemoteintrin_buf_size},
    {"clflushoptintrin.h", clflushoptintrin_buf, &clflushoptintrin_buf_size},
    {"clwbintrin.h", clwbintrin_buf, &clwbintrin_buf_size},
    {"clzerointrin.h", clzerointrin_buf, &clzerointrin_buf_size},
    {"cpuid.h", cpuid_buf, &cpuid_buf_size},
    {"emmintrin.h", emmintrin_buf, &emmintrin_buf_size},
    {"enqcmdintrin.h", enqcmdintrin_buf, &enqcmdintrin_buf_size},
    {"f16cintrin.h", f16cintrin_buf, &f16cintrin_buf_size},
    {"float.h", float_buf, &float_buf_size},
    {"fma4intrin.h", fma4intrin_buf, &fma4intrin_buf_size},
    {"fmaintrin.h", fmaintrin_buf, &fmaintrin_buf_size},
    {"fxsrintrin.h", fxsrintrin_buf, &fxsrintrin_buf_size},
    {"gfniintrin.h", gfniintrin_buf, &gfniintrin_buf_size},
    {"htmintrin.h", htmintrin_buf, &htmintrin_buf_size},
    {"htmxlintrin.h", htmxlintrin_buf, &htmxlintrin_buf_size},
    {"ia32intrin.h", ia32intrin_buf, &ia32intrin_buf_size},
    {"immintrin.h", immintrin_buf, &immintrin_buf_size},
    {"intrin.h", intrin_buf, &intrin_buf_size},
    {"inttypes.h", inttypes_buf, &inttypes_buf_size},
    {"invpcidintrin.h", invpcidintrin_buf, &invpcidintrin_buf_size},
    {"iso646.h", iso646_buf, &iso646_buf_size},
    {"limits.h", limits_buf, &limits_buf_size},
    {"lwpintrin.h", lwpintrin_buf, &lwpintrin_buf_size},
    {"lzcntintrin.h", lzcntintrin_buf, &lzcntintrin_buf_size},
    {"mm3dnow.h", mm3dnow_buf, &mm3dnow_buf_size},
    {"mm_malloc.h", mm_malloc_buf, &mm_malloc_buf_size},
    {"mmintrin.h", mmintrin_buf, &mmintrin_buf_size},
    {"movdirintrin.h", movdirintrin_buf, &movdirintrin_buf_size},
    {"msa.h", msa_buf, &msa_buf_size},
    {"mwaitxintrin.h", mwaitxintrin_buf, &mwaitxintrin_buf_size},
    {"nmmintrin.h", nmmintrin_buf, &nmmintrin_buf_size},
    {"omp-tools.h", omp_tools_buf, &omp_tools_buf_size},
    {"omp.h", omp_buf, &omp_buf_size},
    {"ompt.h", ompt_buf, &ompt_buf_size},
    {"opencl-c-base.h", opencl_c_base_buf, &opencl_c_base_buf_size},
    {"opencl-c.h", opencl_c_buf, &opencl_c_buf_size},
    {"pconfigintrin.h", pconfigintrin_buf, &pconfigintrin_buf_size},
    {"pkuintrin.h", pkuintrin_buf, &pkuintrin_buf_size},
    {"pmmintrin.h", pmmintrin_buf, &pmmintrin_buf_size},
    {"popcntintrin.h", popcntintrin_buf, &popcntintrin_buf_size},
    {"prfchwintrin.h", prfchwintrin_buf, &prfchwintrin_buf_size},
    {"ptwriteintrin.h", ptwriteintrin_buf, &ptwriteintrin_buf_size},
    {"rdseedintrin.h", rdseedintrin_buf, &rdseedintrin_buf_size},
    {"rtmintrin.h", rtmintrin_buf, &rtmintrin_buf_size},
    {"s390intrin.h", s390intrin_buf, &s390intrin_buf_size},
    {"serializeintrin.h", serializeintrin_buf, &serializeintrin_buf_size},
    {"sgxintrin.h", sgxintrin_buf, &sgxintrin_buf_size},
    {"shaintrin.h", shaintrin_buf, &shaintrin_buf_size},
    {"smmintrin.h", smmintrin_buf, &smmintrin_buf_size},
    {"stdalign.h", stdalign_buf, &stdalign_buf_size},
    {"stdarg.h", stdarg_buf, &stdarg_buf_size},
    {"stdatomic.h", stdatomic_buf, &stdatomic_buf_size},
    {"stdbool.h", stdbool_buf, &stdbool_buf_size},
    {"stddef.h", stddef_buf, &stddef_buf_size},
    {"stdint.h", stdint_buf, &stdint_buf_size},
    {"stdnoreturn.h", stdnoreturn_buf, &stdnoreturn_buf_size},
    {"tbmintrin.h", tbmintrin_buf, &tbmintrin_buf_size},
    {"tgmath.h", tgmath_buf, &tgmath_buf_size},
    {"tmmintrin.h", tmmintrin_buf, &tmmintrin_buf_size},
    {"tsxldtrkintrin.h", tsxldtrkintrin_buf, &tsxldtrkintrin_buf_size},
    {"unwind.h", unwind_buf, &unwind_buf_size},
    {"vadefs.h", vadefs_buf, &vadefs_buf_size},
    {"vaesintrin.h", vaesintrin_buf, &vaesintrin_buf_size},
    {"varargs.h", varargs_buf, &varargs_buf_size},
    {"vecintrin.h", vecintrin_buf, &vecintrin_buf_size},
    {"vpclmulqdqintrin.h", vpclmulqdqintrin_buf, &vpclmulqdqintrin_buf_size},
    {"waitpkgintrin.h", waitpkgintrin_buf, &waitpkgintrin_buf_size},
    {"wasm_simd128.h", wasm_simd128_buf, &wasm_simd128_buf_size},
    {"wbnoinvdintrin.h", wbnoinvdintrin_buf, &wbnoinvdintrin_buf_size},
    {"wmmintrin.h", wmmintrin_buf, &wmmintrin_buf_size},
    {"x86intrin.h", x86intrin_buf, &x86intrin_buf_size},
    {"xmmintrin.h", xmmintrin_buf, &xmmintrin_buf_size},
    {"xopintrin.h", xopintrin_buf, &xopintrin_buf_size},
    {"xsavecintrin.h", xsavecintrin_buf, &xsavecintrin_buf_size},
    {"xsaveintrin.h", xsaveintrin_buf, &xsaveintrin_buf_size},
    {"xsaveoptintrin.h", xsaveoptintrin_buf, &xsaveoptintrin_buf_size},
    {"xsavesintrin.h", xsavesintrin_buf, &xsavesintrin_buf_size},
    {"xtestintrin.h", xtestintrin_buf, &xtestintrin_buf_size},
    {nullptr, nullptr, nullptr}};
}

void clang_c_languaget::dump_clang_headers(const std::string &tmp_dir)
{
  static bool dumped = false;
  if(dumped)
    return;
  dumped = true;

  for(struct hooked_header *h = &clang_headers[0]; h->basename != nullptr; h++)
  {
    std::ofstream header;
    header.open(tmp_dir + "/" + std::string(h->basename));
    header << std::string(h->textstart, *h->textsize);
    header.close();
  }
}
