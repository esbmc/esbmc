typedef union uni_s {
  int data;
} uni_t;

typedef struct arr_s {
  int data[2];
} arr_t;

typedef struct top_s {
  uni_t u; 
  arr_t a;
} top_s;

int main() {
  top_s t;
  arr_t* a = &t.a;

  for (int i = 0; i < 2 ; i++) {
    a->data[i] = i;
    assert(a->data[i] == i);
  }

  return 0;
}
