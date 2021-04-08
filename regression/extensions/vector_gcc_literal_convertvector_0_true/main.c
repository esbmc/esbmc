typedef unsigned int vector4uint __attribute__((__vector_size__(16)));
typedef int  vector4int  __attribute__((__vector_size__(16)));

vector4uint vui; vector4int vi;

int main() {
   vui = (vector4uint){-2,-1,2,3};
   vector4int expected = (vector4int){(int)-2, (int)-1, (int)2, (int)3};
   vi = __builtin_convertvector(vui, vector4int);
   for(int i = 0; i < 4; i++) {
      assert(vi[i] == expected[i]);
   }
   return 0;
}