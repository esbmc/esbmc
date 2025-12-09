/*
$ gcc -fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer main.c -o main
./main
AddressSanitizer:DEADLYSIGNAL
=================================================================
==8392==ERROR: AddressSanitizer: SEGV on unknown address 0x00000000ea60 (pc 0x5630efcd819d bp 0x000000000001 sp 0x7ffd58c774d8 T0)
==8392==The signal is caused by a WRITE memory access.
    #0 0x5630efcd819d in main /home/lucas/ESBMC_Project/esbmc/regression/esbmc-unix/github_1498/main.c:7
    #1 0x7f1626736d8f in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:58
    #2 0x7f1626736e3f in __libc_start_main_impl ../csu/libc-start.c:392
    #3 0x5630efcd80c4 in _start (/home/lucas/ESBMC_Project/esbmc/regression/esbmc-unix/github_1498/main+0x10c4)

AddressSanitizer can not provide additional info.
SUMMARY: AddressSanitizer: SEGV /home/lucas/ESBMC_Project/esbmc/regression/esbmc-unix/github_1498/main.c:7 in main
==8392==ABORTING
*/

typedef struct {
  int a;
} b;

int main()
{
  ((b *)60000)->a = 0;
}
