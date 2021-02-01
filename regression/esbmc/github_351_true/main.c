#include <stdio.h>
#include <stdlib.h>
#ifndef uint32_t
#define uint32_t unsigned int
#endif

// Address space attributes
enum ADDRSPACE {
  GS = 256, FS, SS, ES, CS, DS, PHYSEG_SUPOVR,
  LINSEG_SUPOVR, LINSEG_NOSUPOVR, IDTR, GDTR, LDTR, TR
};

#define __gs __attribute__((address_space(GS)))
#define __fs __attribute__((address_space(FS)))
#define __ss __attribute__((address_space(SS)))
#define __es __attribute__((address_space(ES)))
#define __cs __attribute__((address_space(CS)))
#define __ds __attribute__((address_space(DS)))
#define __physeg_supovr __attribute__((address_space(PHYSEG_SUPOVR)))
#define __linseg_supovr __attribute__((address_space(LINSEG_SUPOVR)))
#define __linseg_nosupovr __attribute__((address_space(LINSEG_NOSUPOVR)))
#define __idtr __attribute__((address_space(IDTR)))
#define __gdtr __attribute__((address_space(GDTR)))
#define __ldtr __attribute__((address_space(LDTR)))
#define __tr __attribute__((address_space(TR)))

void foo(__linseg_supovr uint32_t* param1_p, uint32_t param2) {
  param1_p = (__linseg_supovr unsigned int *) malloc(sizeof(__linseg_supovr uint32_t));
  __ESBMC_assert(param2 >= 0, "Error");
  __ESBMC_assume(param1_p != NULL);
  *param1_p = param2;
  __ESBMC_assert(*param1_p == param2, "Value should be set correctly");
}