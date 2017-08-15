AUTOMAKE_OPTIONS=subdir-objects

noinst_LTLIBRARIES = libclangheaders.la
libclangheaders_la_SOURCES = clang___clang_cuda_builtin_vars.c \
      clang___clang_cuda_cmath.c clang___clang_cuda_complex_builtins.c \
      clang___clang_cuda_intrinsics.c \
      clang___clang_cuda_math_forward_declares.c \
      clang___clang_cuda_runtime_wrapper.c clang___stddef_max_align_t.c \
      clang___wmmintrin_aes.c clang___wmmintrin_pclmul.c \
      clang_adxintrin.c clang_altivec.c clang_ammintrin.c \
      clang_arm_acle.c clang_arm_neon.c clang_armintr.c \
      clang_avx2intrin.c clang_avx512bwintrin.c clang_avx512cdintrin.c \
      clang_avx512dqintrin.c clang_avx512erintrin.c clang_avx512fintrin.c \
      clang_avx512ifmaintrin.c clang_avx512ifmavlintrin.c \
      clang_avx512pfintrin.c clang_avx512vbmiintrin.c \
      clang_avx512vbmivlintrin.c clang_avx512vlbwintrin.c \
      clang_avx512vlcdintrin.c clang_avx512vldqintrin.c \
      clang_avx512vlintrin.c clang_avxintrin.c clang_bmi2intrin.c \
      clang_bmiintrin.c clang_clflushoptintrin.c clang_cpuid.c \
      clang_emmintrin.c clang_f16cintrin.c clang_float.c \
      clang_fma4intrin.c clang_fmaintrin.c clang_fxsrintrin.c \
      clang_htmintrin.c clang_htmxlintrin.c clang_ia32intrin.c \
      clang_immintrin.c clang_intrin.c clang_inttypes.c clang_iso646.c \
      clang_limits.c clang_lzcntintrin.c clang_mm3dnow.c clang_mm_malloc.c \
      clang_mmintrin.c clang_msa.c clang_mwaitxintrin.c clang_nmmintrin.c \
      clang_omp.c clang_opencl_c.c clang_pkuintrin.c clang_pmmintrin.c \
      clang_popcntintrin.c clang_prfchwintrin.c clang_rdseedintrin.c \
      clang_rtmintrin.c clang_s390intrin.c clang_shaintrin.c \
      clang_smmintrin.c clang_stdalign.c clang_stdarg.c clang_stdatomic.c \
      clang_stdbool.c clang_stddef.c clang_stdint.c clang_stdnoreturn.c \
      clang_tbmintrin.c clang_tgmath.c clang_tmmintrin.c clang_unwind.c \
      clang_vadefs.c clang_varargs.c clang_vecintrin.c clang_wmmintrin.c \
      clang_x86intrin.c clang_xmmintrin.c clang_xopintrin.c \
      clang_xsavecintrin.c clang_xsaveintrin.c clang_xsaveoptintrin.c \
      clang_xsavesintrin.c clang_xtestintrin.c

SUFFIXES = .hs

.hs.c:
	$(top_srcdir)/c2goto/headers/flail.sh $< $< > $@

clean-local:
	-rm -rf *.c

.NOTPARALLEL: all
