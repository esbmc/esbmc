noinst_LIBRARIES = libclangheaders.a
libclangheaders_a_SOURCES = clang_rtmintrin.hs clang_pkuintrin.hs clang_smmintrin.hs \
                            clang_mm_malloc.hs clang_stdbool.hs clang_xsavecintrin.hs \
                            clang_xopintrin.hs clang_unwind.hs clang_popcntintrin.hs \
                            clang_htmintrin.hs clang_pmmintrin.hs clang_fmaintrin.hs \
                            clang___wmmintrin_aes.hs clang_immintrin.hs \
                            clang_emmintrin.hs clang_x86intrin.hs clang_stdint.hs \
                            clang_prfchwintrin.hs clang_omp.hs clang_stdalign.hs \
                            clang_fxsrintrin.hs clang_cpuid.hs clang_shaintrin.hs \
                            clang_stddef.hs clang_bmiintrin.hs clang_xsaveoptintrin.hs \
                            clang_xmmintrin.hs clang_adxintrin.hs clang_ia32intrin.hs \
                            clang_ammintrin.hs clang_bmi2intrin.hs clang_arm_acle.hs \
                            clang_varargs.hs clang_rdseedintrin.hs clang_Intrin.hs \
                            clang___clang_cuda_runtime_wrapper.hs \
                            clang_xtestintrin.hs clang___wmmintrin_pclmul.hs \
                            clang_wmmintrin.hs clang_stdatomic.hs clang_vadefs.hs \
                            clang_inttypes.hs clang_fma4intrin.hs clang_s390intrin.hs \
                            clang_nmmintrin.hs clang_xsavesintrin.hs clang_tmmintrin.hs \
                            clang_cuda_builtin_vars.hs \
                            clang_stdarg.hs clang_lzcntintrin.hs clang_float.hs \
                            clang_iso646.hs clang_tbmintrin.hs \
                            clang_mm3dnow.hs clang_stdnoreturn.hs clang_htmxlintrin.hs \
                            clang_limits.hs clang_xsaveintrin.hs \
                            clang_tgmath.hs clang___stddef_max_align_t.hs \
                            clang_mmintrin.hs clang_f16cintrin.hs \
                            sanitizer/clang_allocator_interface.hs \
                            sanitizer/clang_tsan_interface_atomic.hs \
                            sanitizer/clang_lsan_interface.hs \
                            sanitizer/clang_asan_interface.hs \
                            sanitizer/clang_dfsan_interface.hs \
                            sanitizer/clang_coverage_interface.hs \
                            sanitizer/clang_common_interface_defs.hs \
                            sanitizer/clang_msan_interface.hs \
                            sanitizer/clang_linux_syscall_hooks.hs

SUFFIXES = .hs

.hs.o:
	$(top_srcdir)/ansi-c/headers/flail.sh $< $< | $(CC) $(CFLAGS) -x c - -c -o $@
