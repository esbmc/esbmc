
typedef struct {

                int arr[33];

} struct_t;

 

extern struct_t tmp_struct;

 

static int func1(int i)

{

                return tmp_struct.arr[i];

}
