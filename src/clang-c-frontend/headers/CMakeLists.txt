set (inputs __clang_cuda_builtin_vars.h __clang_cuda_cmath.h __clang_cuda_complex_builtins.h __clang_cuda_device_functions.h __clang_cuda_intrinsics.h __clang_cuda_libdevice_declares.h __clang_cuda_math.h __clang_cuda_math_forward_declares.h __clang_cuda_runtime_wrapper.h __clang_hip_libdevice_declares.h __clang_hip_math.h __clang_hip_runtime_wrapper.h __stddef_max_align_t.h __wmmintrin_aes.h __wmmintrin_pclmul.h adxintrin.h altivec.h ammintrin.h amxintrin.h arm64intr.h arm_acle.h arm_bf16.h arm_cde.h arm_cmse.h arm_fp16.h arm_mve.h arm_neon.h arm_sve.h armintr.h avx2intrin.h avx512bf16intrin.h avx512bitalgintrin.h avx512bwintrin.h avx512cdintrin.h avx512dqintrin.h avx512erintrin.h avx512fintrin.h avx512ifmaintrin.h avx512ifmavlintrin.h avx512pfintrin.h avx512vbmi2intrin.h avx512vbmiintrin.h avx512vbmivlintrin.h avx512vlbf16intrin.h avx512vlbitalgintrin.h avx512vlbwintrin.h avx512vlcdintrin.h avx512vldqintrin.h avx512vlintrin.h avx512vlvbmi2intrin.h avx512vlvnniintrin.h avx512vlvp2intersectintrin.h avx512vnniintrin.h avx512vp2intersectintrin.h avx512vpopcntdqintrin.h avx512vpopcntdqvlintrin.h avxintrin.h bmi2intrin.h bmiintrin.h cet.h cetintrin.h cldemoteintrin.h clflushoptintrin.h clwbintrin.h clzerointrin.h cpuid.h emmintrin.h enqcmdintrin.h f16cintrin.h float.h fma4intrin.h fmaintrin.h fxsrintrin.h gfniintrin.h htmintrin.h htmxlintrin.h ia32intrin.h immintrin.h intrin.h inttypes.h invpcidintrin.h iso646.h limits.h lwpintrin.h lzcntintrin.h mm3dnow.h mm_malloc.h mmintrin.h movdirintrin.h msa.h mwaitxintrin.h nmmintrin.h omp_tools.h omp.h ompt.h opencl_c_base.h opencl_c.h pconfigintrin.h pkuintrin.h pmmintrin.h popcntintrin.h prfchwintrin.h ptwriteintrin.h rdseedintrin.h rtmintrin.h s390intrin.h serializeintrin.h sgxintrin.h shaintrin.h smmintrin.h stdalign.h stdarg.h stdatomic.h stdbool.h stddef.h stdint.h stdnoreturn.h tbmintrin.h tgmath.h tmmintrin.h tsxldtrkintrin.h unwind.h vadefs.h vaesintrin.h varargs.h vecintrin.h vpclmulqdqintrin.h waitpkgintrin.h wasm_simd128.h wbnoinvdintrin.h wmmintrin.h x86intrin.h xmmintrin.h xopintrin.h xsavecintrin.h xsaveintrin.h xsaveoptintrin.h xsavesintrin.h xtestintrin.h)

function(mangle_hs output)
  set(result "")
  foreach(in_f ${ARGN})
    file(RELATIVE_PATH out_f ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${in_f})
    string(REGEX REPLACE .h .c out_f "${out_f}")
    set(out_f "${CMAKE_CURRENT_BINARY_DIR}/${out_f}")
    add_custom_command(OUTPUT ${out_f}
      COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/flail.py ${CMAKE_CURRENT_SOURCE_DIR}/${in_f} ${out_f}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${in_f} ${CMAKE_SOURCE_DIR}/scripts/flail.py
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
