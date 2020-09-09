/*

ESBMC PARAMETERS

--no-unwinding-assertions --unwind 15 --ssa-guards --enable-core-dump --result-only --interval-analysis --show-cex --print-stack-traces --i386-linux --parse-tree-too --termination --no-library

 * This is a RANDOMLY GENERATED PROGRAM.
 *
 * Generator: csmith 2.4.0
 * Git version: 0bd545f
 * Options:   --output /home/sidia/release-esbmc/bin/corpus/esbmc-fuzzer-1596740192660.c --divs --int8 --comma-operators --consts --builtins --post-decr-operator --volatiles --pointers --longlong --math64
 * Seed:      1088334264
 */

#include "csmith.h"


static long __undefined;

/* --- Struct/Union Declarations --- */
/* --- GLOBAL VARIABLES --- */
static int16_t g_13 = 0L;
static int32_t g_29[1][8][1] = {{{0x8DC5336DL},{0x8DC5336DL},{0x8DC5336DL},{0x8DC5336DL},{0x8DC5336DL},{0x8DC5336DL},{0x8DC5336DL},{0x8DC5336DL}}};


/* --- FORWARD DECLARATIONS --- */
static int32_t  func_27(void);


/* --- FUNCTIONS --- */
/* ------------------------------------------ */
/* 
 * reads : g_29
 * writes:
 */
static int32_t  func_27(void)
{ /* block id: 36 */
    int32_t *l_28 = &g_29[0][4][0];
    int32_t *l_30 = &g_29[0][5][0];
    int32_t *l_31 = &g_29[0][0][0];
    int32_t *l_32 = &g_29[0][4][0];
    int32_t *l_33 = &g_29[0][4][0];
    int32_t *l_34 = (void*)0;
    int32_t *l_35 = &g_29[0][4][0];
    int32_t *l_36 = &g_29[0][4][0];
    int32_t *l_37 = &g_29[0][4][0];
    int32_t *l_38[1];
    int8_t l_39 = 0xF4L;
    int8_t l_40 = 0x5CL;
    uint8_t l_41 = 6UL;
    int i;
    for (i = 0; i < 1; i++)
        l_38[i] = &g_29[0][4][0];
    l_41++;
    return g_29[0][1][0];
}




/* ---------------------------------------- */
int main (int argc, char* argv[])
{
    int i, j, k;
    int print_hash_value = 0;
    if (argc == 2 && strcmp(argv[1], "1") == 0) print_hash_value = 1;
    platform_main_begin();
    crc32_gentab();
    func_27();
    transparent_crc(g_13, "g_13", print_hash_value);
    for (i = 0; i < 1; i++)
    {
        for (j = 0; j < 8; j++)
        {
            for (k = 0; k < 1; k++)
            {
                transparent_crc(g_29[i][j][k], "g_29[i][j][k]", print_hash_value);
                if (print_hash_value) printf("index = [%d][%d][%d]\n", i, j, k);

            }
        }
    }
    platform_main_end(crc32_context ^ 0xFFFFFFFFUL, print_hash_value);
    return 0;
}

/************************ statistics *************************
XXX max struct depth: 0
breakdown:
   depth: 0, occurrence: 10
XXX total union variables: 0

XXX non-zero bitfields defined in structs: 0
XXX zero bitfields defined in structs: 0
XXX const bitfields defined in structs: 0
XXX volatile bitfields defined in structs: 0
XXX structs with bitfields in the program: 0
breakdown:
XXX full-bitfields structs in the program: 0
breakdown:
XXX times a bitfields struct's address is taken: 0
XXX times a bitfields struct on LHS: 0
XXX times a bitfields struct on RHS: 0
XXX times a single bitfield on LHS: 0
XXX times a single bitfield on RHS: 0

XXX max expression depth: 1
breakdown:
   depth: 1, occurrence: 3

XXX total number of pointers: 10

XXX times a variable address is taken: 9
XXX times a pointer is dereferenced on RHS: 0
breakdown:
XXX times a pointer is dereferenced on LHS: 0
breakdown:
XXX times a pointer is compared with null: 0
XXX times a pointer is compared with address of another variable: 0
XXX times a pointer is compared with another pointer: 0
XXX times a pointer is qualified to be dereferenced: 68
XXX number of pointers point to pointers: 0
XXX number of pointers point to scalars: 10
XXX number of pointers point to structs: 0
XXX percent of pointers has null in alias set: 10
XXX average alias set size: 1

XXX times a non-volatile is read: 19
XXX times a non-volatile is write: 1
XXX times a volatile is read: 0
XXX    times read thru a pointer: 0
XXX times a volatile is write: 0
XXX    times written thru a pointer: 0
XXX times a volatile is available for access: 0
XXX percentage of non-volatile access: 100

XXX forward jumps: 0
XXX backward jumps: 0

XXX stmts: 2
XXX max block depth: 0
breakdown:
   depth: 0, occurrence: 2

XXX percentage a fresh-made variable is used: 37
XXX percentage an existing variable is used: 63
FYI: the random generator makes assumptions about the integer size. See platform.info for more details.
********************* end of statistics **********************/

