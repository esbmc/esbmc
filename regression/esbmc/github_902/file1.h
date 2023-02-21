
typedef struct {

                int arr[33];

} struct_t;

 

struct_t tmp_struct; /* error: duplicate definition of this symbol */

 

static int func1(int i)

{

                return tmp_struct.arr[i];

}
