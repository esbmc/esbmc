#include <stdbool.h>
#include <stdint.h>
 
typedef union{
} union1;
 
typedef union{
    struct {
        uint32_t    union2_var:1;
    };
} union2;
extern union2 u2;
 
typedef struct {
    bool struct1_flag;
} struct1;
 
typedef struct {
    union1 u1;
    struct1 struct2_var1;
    struct1 struct2_var2;
 
} struct2;
 
typedef struct {
    struct2 struct3_var1;
} struct3;
 
extern struct3 struct3_extern;
 
void main(void) {
    bool flag;
 
    struct1 *struct1_p;
 
    if (flag) {
        struct1_p = &(struct3_extern.struct3_var1.struct2_var2);
    } else {
        struct1_p = &(struct3_extern.struct3_var1.struct2_var1);
    };
 
    struct1_p->struct1_flag = (u2.union2_var) != 0;
}
