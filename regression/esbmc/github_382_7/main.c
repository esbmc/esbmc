// standard h files
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>

typedef union{
    uint64_t raw64;
    struct {
        uint32_t raw32lo;
        uint32_t raw32hi;
    };
    struct {
        uint64_t    AA:3;     // reg[2:0]
        uint64_t    BB:2;     // reg[4:3]
        uint64_t    CC:1;     // reg[5:5]
        uint64_t    DD:12;     // reg[17:6]
        uint64_t    EE:9;     // reg[26:18]
        uint64_t    YYYY:9;     // reg[35:27]
        uint64_t    FF:5;     // reg[40:36]
        uint64_t    GG:2;     // reg[42:41]
        uint64_t    HH:21;     // reg[63:43]
    };
} str_t;

typedef struct llc_flush_info_s {
    uint32_t A;
    uint32_t B;
    uint64_t C;
    uint32_t XXXX;
    uint32_t D;
    uint32_t E;//Note: SERVER won't use this counter.
    bool F;    
} str2_t;  

int main(void){
  str_t cfg;
  cfg.raw64 = 0;
  str2_t info;
  info.XXXX  =  cfg.YYYY ;
}

