char s[]="abc\001";

char *p="abc";

int input;

int main()
{
  assert(s[1]=='b');
  assert(s[4]==0);
  
  // write to s
  s[0]='x';
  
  assert(p[2]=='c');
  
  p=s;

  // write to p
  p[1]='y';
  
  assert(s[1]=='y'); 

  {
    const char local_string[]="asd123";

    assert(local_string[0]=='a');
    assert(sizeof(local_string)==7);
    assert(local_string[6]==0);
  }
}
