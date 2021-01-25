int comp(void * s1, void *s2, unsigned int n) {
  char *us1 = (char*) s1;
  char *us2 = (char*) s2;

  if(us1[0] != us2[0]) {  return -1; }
  if(us1[1] != us2[1]) {  return -1; }

  return 0;
}

int main() {

  char a[2][2];

  a[0][0] = 'a';
  a[0][1] = 'b';
  a[1][0] = 'g';
  a[1][1] = 'b';

  assert(comp(&a[0], &a[1], 2) == -1);

}
