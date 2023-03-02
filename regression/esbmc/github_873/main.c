// --no-slice --incremental-bmc

typedef struct {
  int b
} d;
union {} f[];
d *a;
void main() 
  { a->b; }
