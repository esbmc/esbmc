#include <stdbool.h>
#include <stdint.h>


typedef struct str1_t {
    unsigned int field1 : 16;
    unsigned int field2 : 8;
    uint8_t field3[59];
} str1_t;


typedef struct str2_t {
    str1_t field1;
    uint32_t field2;
    uint64_t field3;
    uint32_t field4;
    uint32_t field5[4];
    uint32_t field6;
} str2_t;


int main(void) {

    str2_t var_data;
    str2_t *var_data_p = &var_data;
    uint64_t var_offset[1000]; // = 0xdeadbeef;
    
    for (uint16_t i=0; i < sizeof(str1_t)/sizeof(uint64_t); i++) {
        ((uint64_t *)&var_data_p->field1)[i] = ((uint64_t *)var_offset)[i];
    }
   return 0; 
}
