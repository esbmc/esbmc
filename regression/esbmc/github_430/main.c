#include <stdint.h>

typedef union{
    uint64_t raw64;
    struct {
        uint32_t raw32lo;
        uint32_t raw32hi;
    };
    struct {
        uint64_t    field0:1;
        uint64_t    field1:1;   
        uint64_t    field2:1;   
        uint64_t    field3:1;   
        uint64_t    field4:1;   
        uint64_t    field5:1;   
        uint64_t    field6:1;   
        uint64_t    field7:1;
           
        uint64_t    field8:1;   
        uint64_t    field9:1;   
        uint64_t    field10:1;  
        uint64_t    field11:1;  
        uint64_t    field12:1;  
        uint64_t    field13:1;  
        uint64_t    field14:1;  
        uint64_t    field15:1;  

        uint64_t    field16:1;  
        uint64_t    field17:1;  
        uint64_t    field18:1;  
        uint64_t    field19:4;  
        uint64_t    field20:1;  
        uint64_t    field21:1;  
        uint64_t    field22:1;  
        uint64_t    field23:1;  

        uint64_t    field24:1;  
        uint64_t    field25:1;  
        uint64_t    field26:1;  
        uint64_t    field27:1;  
        uint64_t    field28:1;  
        uint64_t    field29:1;  
        uint64_t    field30:1;  
        uint64_t    field31:1;  

        uint64_t    field32:1;  
        uint64_t    field33:1;  
        uint64_t    field34:1;  
        uint64_t    field35:1;  
        uint64_t    field36:1;  
        uint64_t    field37:1;  
        uint64_t    field38:1;  
        uint64_t    field39:1;  

        uint64_t    field40:1;  
        uint64_t    field41:1;  
        uint64_t    field42:1;  
        uint64_t    field43:1;  
        uint64_t    field44:1;  
        uint64_t    field45:1;  
        uint64_t    field46:1;  
        uint64_t    field47:1;  

        uint64_t    field48:1;  
        uint64_t    field49:1;  
        uint64_t    field50:1;  
        uint64_t    field51:1;  
        uint64_t    field52:1;  
        uint64_t    field53:1;  
        uint64_t    field54:1;  

        uint64_t    field55:1;  
        uint64_t    field56:1;  
        uint64_t    field57:1;  
        uint64_t    field58:1;  
        uint64_t    field59:1;  
        uint64_t    field60:1;
    };
} str_t;

extern str_t external_var;

int main(void) {
    return external_var.field60;
}
