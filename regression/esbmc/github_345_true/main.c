// standard h files
#include <stdbool.h>
#include <stdint.h>

#ifndef uint8_t
#define uint8_t unsigned char
#endif

#ifndef uint16_t
#define uint16_t unsigned short
#endif

#ifndef uint32_t
#define uint32_t unsigned int
#endif

#ifndef uint64_t
#define uint64_t unsigned long long
#endif

// structure types:
typedef union{
    uint32_t raw32;

    struct {
        uint32_t    field1:1;
        uint32_t    field2:1;
        uint32_t    field3:1;
        uint32_t    field4:1;
        uint32_t    field5:1;
        uint32_t    field6:1;
        uint32_t    field7:1;
        uint32_t    field8:1;
        uint32_t    field9:1;
        uint32_t    field10:1;
        uint32_t    field11:1;
        uint32_t    field12:1;
        uint32_t    field13:1;
        uint32_t    field14:1;
        uint32_t    field15:1;
        uint32_t    field16:1;
        uint32_t    field17:1;
        uint32_t    field18:1;
        uint32_t    field19:1;
        uint32_t    field20:1;
        uint32_t    field21:1;
        uint32_t    field22:1;
        uint32_t    field23:1;
        uint32_t    field24:1;
        uint32_t    field25:1;
        uint32_t    field26:1;
        uint32_t    field27:5;
        uint32_t    field28:1;
    };
} str1_t;


typedef union{
    uint32_t raw32;

    struct {
        uint32_t    field1:32;
    };
} str2_t;

extern str2_t extern_global_var1 ;


typedef union{
    uint32_t raw32;

    struct {
        uint32_t    saved_field1:2;
        uint32_t    field2:1;
    };
} str3_t;

extern str3_t extern_global_var2;


struct str4_s {

    bool bool_var;


};



typedef union {
    struct
    {
        uint16_t field1   : 2;
        uint16_t field2   : 1;
        uint16_t field3   : 13;
    };
    uint16_t raw;
} str4_t;


typedef union
{
   struct
   {
        uint16_t field1   : 4;
        uint16_t s_field2      : 1;
        uint16_t d_field3    : 2;
        uint16_t p_field4      : 1;
        uint16_t r_field5    : 3;
        uint16_t null_field6   : 1;
        uint16_t a_field7    : 1;
        uint16_t l_field8      : 1;
        uint16_t d_field9     : 1;
        uint16_t g_field10      : 1;
    };
    uint16_t ar_raw;
} str5_t;

typedef union
{
   struct
   {
        uint64_t   field1;
        str4_t     field2;
        str5_t     field3;
        uint32_t   field4;
    };
    struct
    {
        uint64_t low64;
        uint64_t high64;
    };
} str6_t;


//function protoypes:
void func1(bool retain_var);
void func2(bool return_var, uint32_t u_var);
void func3(struct str4_s* var1_p);
void func4(void);
void func5(struct str4_s* var1_p);
bool func6(bool bool_var);
void func7(struct str4_s* var1_p);



void func3(struct str4_s* var1_p){

    bool bool_var = var1_p->bool_var;

    if(func6(bool_var)){
        func5(var1_p);
    }
}


void func4(){
    extern_global_var1.raw32 += 1;
}

bool func6(bool bool_var){
    return ((extern_global_var2.field2 !=0) || !bool_var);
}


void func5(struct str4_s* var1_p){
    func7(var1_p);
}


void func7(struct str4_s* var1_p){
    str6_t var1;
    uint32_t var2 = var1.field3.d_field3;
}


int main(void) {
    struct str4_s var;
    var.bool_var = 1;
    __ESBMC_assert(var.bool_var != 1, "Bool_var should be 1");
    func3(&var);
    return 0;
}