set (inputs clang___clang_cuda_builtin_vars.hs clang___clang_cuda_cmath.hs clang___clang_cuda_complex_builtins.hs clang___clang_cuda_device_functions.hs clang___clang_cuda_intrinsics.hs clang___clang_cuda_libdevice_declares.hs clang___clang_cuda_math_forward_declares.hs clang___clang_cuda_runtime_wrapper.hs clang___stddef_max_align_t.hs clang___wmmintrin_aes.hs clang___wmmintrin_pclmul.hs clang_adxintrin.hs clang_altivec.hs clang_ammintrin.hs clang_arm64intr.hs clang_arm_acle.hs clang_arm_fp16.hs clang_arm_neon.hs clang_armintr.hs clang_avx2intrin.hs clang_avx512bf16intrin.hs clang_avx512bitalgintrin.hs clang_avx512bwintrin.hs clang_avx512cdintrin.hs clang_avx512dqintrin.hs clang_avx512erintrin.hs clang_avx512fintrin.hs clang_avx512ifmaintrin.hs clang_avx512ifmavlintrin.hs clang_avx512pfintrin.hs clang_avx512vbmi2intrin.hs clang_avx512vbmiintrin.hs clang_avx512vbmivlintrin.hs clang_avx512vlbf16intrin.hs clang_avx512vlbitalgintrin.hs clang_avx512vlbwintrin.hs clang_avx512vlcdintrin.hs clang_avx512vldqintrin.hs clang_avx512vlintrin.hs clang_avx512vlvbmi2intrin.hs clang_avx512vlvnniintrin.hs clang_avx512vlvp2intersectintrin.hs clang_avx512vnniintrin.hs clang_avx512vp2intersectintrin.hs clang_avx512vpopcntdqintrin.hs clang_avx512vpopcntdqvlintrin.hs clang_avxintrin.hs clang_bmi2intrin.hs clang_bmiintrin.hs clang_cetintrin.hs clang_cldemoteintrin.hs clang_clflushoptintrin.hs clang_clwbintrin.hs clang_clzerointrin.hs clang_cpuid.hs clang_emmintrin.hs clang_enqcmdintrin.hs clang_f16cintrin.hs clang_float.hs clang_fma4intrin.hs clang_fmaintrin.hs clang_fxsrintrin.hs clang_gfniintrin.hs clang_htmintrin.hs clang_htmxlintrin.hs clang_ia32intrin.hs clang_immintrin.hs clang_intrin.hs clang_inttypes.hs clang_invpcidintrin.hs clang_iso646.hs clang_limits.hs clang_lwpintrin.hs clang_lzcntintrin.hs clang_mm3dnow.hs clang_mm_malloc.hs clang_mmintrin.hs clang_movdirintrin.hs clang_msa.hs clang_mwaitxintrin.hs clang_nmmintrin.hs clang_omp.hs clang_omp_tools.hs clang_ompt.hs clang_opencl_c.hs clang_opencl_c_base.hs clang_pconfigintrin.hs clang_pkuintrin.hs clang_pmmintrin.hs clang_popcntintrin.hs clang_prfchwintrin.hs clang_ptwriteintrin.hs clang_rdseedintrin.hs clang_rtmintrin.hs clang_s390intrin.hs clang_sgxintrin.hs clang_shaintrin.hs clang_smmintrin.hs clang_stdalign.hs clang_stdarg.hs clang_stdatomic.hs clang_stdbool.hs clang_stddef.hs clang_stdint.hs clang_stdnoreturn.hs clang_tbmintrin.hs clang_tgmath.hs clang_tmmintrin.hs clang_unwind.hs clang_vadefs.hs clang_vaesintrin.hs clang_varargs.hs clang_vecintrin.hs clang_vpclmulqdqintrin.hs clang_waitpkgintrin.hs clang_wbnoinvdintrin.hs clang_wmmintrin.hs clang_x86intrin.hs clang_xmmintrin.hs clang_xopintrin.hs clang_xsavecintrin.hs clang_xsaveintrin.hs clang_xsaveoptintrin.hs clang_xsavesintrin.hs clang_xtestintrin.hs)

function(mangle_hs output)
  set(result "")
  foreach(in_f ${ARGN})
    file(RELATIVE_PATH out_f ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${in_f})
    string(REGEX REPLACE .hs .c out_f "${out_f}")
    set(out_f "${CMAKE_CURRENT_BINARY_DIR}/${out_f}")
    add_custom_command(OUTPUT ${out_f}
      COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py ${CMAKE_CURRENT_SOURCE_DIR}/${in_f} ${out_f}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${in_f}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Converting clang headers ${in_f} to data"
      VERBATIM
      )
    list(APPEND result ${out_f})
  endforeach()
  set(${output} "${result}" PARENT_SCOPE)
endfunction()

mangle_hs(mangled_files ${inputs})
add_library(cheaders ${mangled_files})
set_target_properties(cheaders PROPERTIES LINKER_LANGUAGE C)
