#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef struct str1_t {
    unsigned int field11 : 16;
    unsigned int field12 : 8;
    uint8_t field13[59];
} str1_t;


typedef struct str2_t {
    str1_t field1;
    uint32_t field2;
    uint64_t field3;
    uint32_t field4;
    uint32_t field5[4];
    uint32_t field6;
} str2_t;

uint64_t global_data[100];

int main(void) {

    str2_t var_data;
    str2_t *var_data_p = &var_data;
    uint64_t var_offset[100]; // = global_data; //= 0xdeadbeef;
    
    printf("size=%d\n", sizeof(str1_t)/sizeof(uint64_t));

    str1_t  tmp;
    uint64_t *uint_p = (uint64_t *)&tmp;
    
    uint_p[0] = 5; // ESBMC fails on this line.
    
    uint_p[1] = 5;
 
    for (uint16_t i=1; i < sizeof(str1_t)/sizeof(uint64_t); i++) { // if the for statement starts fom zero ESBMC fails.
        //((uint64_t *)&var_data_p->field1)[i] = 5; // ((uint64_t *)var_offset)[i];
        uint_p[i] = 5;
    }
   return 0; 
}
