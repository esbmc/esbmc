typedef float vector4f __attribute__((__vector_size__(16)));
typedef int  vector4int  __attribute__((__vector_size__(16)));

vector4f vf; vector4int vi;

int main() {
   vf = (vector4f){0.1f, 0.5f, 2.5f, 3.4f};
   vector4int expected = (vector4int){(int)vf[0], (int)vf[1], (int)vf[2], (int)vf[3]};
   vi = __builtin_convertvector(vf, vector4int);
   for(int i = 0; i < 4; i++) {
      assert(vi[i] != expected[i]);
   }
   return 0;
}