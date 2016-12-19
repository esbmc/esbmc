noinst_LIBRARIES = libclangheaders.a
libclangheaders_a_SOURCES = clang_Intrin.hs clang___clang_cuda_cmath.hs \
                            clang___clang_cuda_intrinsics.hs \
                            clang___clang_cuda_math_forward_declares.hs \
                            clang___clang_cuda_runtime_wrapper.hs \
                            clang___stddef_max_align_t.hs \
                            clang___wmmintrin_aes.hs \
                            clang___wmmintrin_pclmul.hs clang_adxintrin.hs \
                            clang_ammintrin.hs clang_arm_acle.hs \
                            clang_bmi2intrin.hs clang_bmiintrin.hs \
                            clang_clflushoptintrin.hs clang_cpuid.hs \
                            clang_cuda_builtin_vars.hs clang_emmintrin.hs \
                            clang_f16cintrin.hs clang_float.hs \
                            clang_fma4intrin.hs clang_fmaintrin.hs \
                            clang_fxsrintrin.hs clang_htmintrin.hs \
                            clang_htmxlintrin.hs clang_ia32intrin.hs \
                            clang_immintrin.hs clang_intrin.hs \
                            clang_inttypes.hs clang_iso646.hs clang_limits.hs \
                            clang_lzcntintrin.hs clang_mm3dnow.hs \
                            clang_mm_malloc.hs clang_mmintrin.hs \
                            clang_mwaitxintrin.hs clang_nmmintrin.hs \
                            clang_omp.hs clang_opencl_c.hs clang_pkuintrin.hs \
                            clang_pmmintrin.hs clang_popcntintrin.hs \
                            clang_prfchwintrin.hs clang_rdseedintrin.hs \
                            clang_rtmintrin.hs clang_s390intrin.hs \
                            clang_shaintrin.hs clang_smmintrin.hs \
                            clang_stdalign.hs clang_stdarg.hs \
                            clang_stdatomic.hs clang_stdbool.hs \
                            clang_stddef.hs clang_stdint.hs \
                            clang_stdnoreturn.hs clang_tbmintrin.hs \
                            clang_tgmath.hs clang_tmmintrin.hs \
                            clang_unwind.hs clang_vadefs.hs clang_varargs.hs \
                            clang_wmmintrin.hs clang_x86intrin.hs \
                            clang_xmmintrin.hs clang_xopintrin.hs \
                            clang_xsavecintrin.hs clang_xsaveintrin.hs \
                            clang_xsaveoptintrin.hs clang_xsavesintrin.hs \
                            clang_xtestintrin.hs \
                            sanitizer/clang_allocator_interface.hs \
                            sanitizer/clang_asan_interface.hs \
                            sanitizer/clang_common_interface_defs.hs \
                            sanitizer/clang_coverage_interface.hs \
                            sanitizer/clang_dfsan_interface.hs \
                            sanitizer/clang_esan_interface.hs \
                            sanitizer/clang_linux_syscall_hooks.hs \
                            sanitizer/clang_lsan_interface.hs \
                            sanitizer/clang_msan_interface.hs \
                            sanitizer/clang_tsan_interface_atomic.hs

SUFFIXES = .hs

.hs.o:
	$(top_srcdir)/c2goto/headers/flail.sh $< $< | $(CC) $(CFLAGS) -x c - -c -o $@
