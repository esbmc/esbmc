AUTOMAKE_OPTIONS=subdir-objects

noinst_LTLIBRARIES = libclangheaders.la
libclangheaders_la_SOURCES = clang_Intrin.c clang___clang_cuda_cmath.c \
                            clang___clang_cuda_intrinsics.c \
                            clang___clang_cuda_math_forward_declares.c \
                            clang___clang_cuda_runtime_wrapper.c \
                            clang___stddef_max_align_t.c \
                            clang___wmmintrin_aes.c \
                            clang___wmmintrin_pclmul.c clang_adxintrin.c \
                            clang_ammintrin.c clang_arm_acle.c \
                            clang_bmi2intrin.c clang_bmiintrin.c \
                            clang_clflushoptintrin.c clang_cpuid.c \
                            clang_cuda_builtin_vars.c clang_emmintrin.c \
                            clang_f16cintrin.c clang_float.c \
                            clang_fma4intrin.c clang_fmaintrin.c \
                            clang_fxsrintrin.c clang_htmintrin.c \
                            clang_htmxlintrin.c clang_ia32intrin.c \
                            clang_immintrin.c clang_intrin.c \
                            clang_inttypes.c clang_iso646.c clang_limits.c \
                            clang_lzcntintrin.c clang_mm3dnow.c \
                            clang_mm_malloc.c clang_mmintrin.c \
                            clang_mwaitxintrin.c clang_nmmintrin.c \
                            clang_omp.c clang_opencl_c.c clang_pkuintrin.c \
                            clang_pmmintrin.c clang_popcntintrin.c \
                            clang_prfchwintrin.c clang_rdseedintrin.c \
                            clang_rtmintrin.c clang_s390intrin.c \
                            clang_shaintrin.c clang_smmintrin.c \
                            clang_stdalign.c clang_stdarg.c \
                            clang_stdatomic.c clang_stdbool.c \
                            clang_stddef.c clang_stdint.c \
                            clang_stdnoreturn.c clang_tbmintrin.c \
                            clang_tgmath.c clang_tmmintrin.c \
                            clang_unwind.c clang_vadefs.c clang_varargs.c \
                            clang_wmmintrin.c clang_x86intrin.c \
                            clang_xmmintrin.c clang_xopintrin.c \
                            clang_xsavecintrin.c clang_xsaveintrin.c \
                            clang_xsaveoptintrin.c clang_xsavesintrin.c \
                            clang_xtestintrin.c \
                            sanitizer/clang_allocator_interface.c \
                            sanitizer/clang_asan_interface.c \
                            sanitizer/clang_common_interface_defs.c \
                            sanitizer/clang_coverage_interface.c \
                            sanitizer/clang_dfsan_interface.c \
                            sanitizer/clang_esan_interface.c \
                            sanitizer/clang_linux_syscall_hooks.c \
                            sanitizer/clang_lsan_interface.c \
                            sanitizer/clang_msan_interface.c \
                            sanitizer/clang_tsan_interface_atomic.c

SUFFIXES = .hs

.hs.c:
	$(top_srcdir)/c2goto/headers/flail.sh $< $< > $@

clean-local:
	-rm -rf *.c
