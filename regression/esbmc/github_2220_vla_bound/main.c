struct dirent
{
  char d_name[256];
};

unsigned long strlen(const char *);

void g(struct dirent *entry)
{
  char buf[strlen(entry->d_name) + 2];
  buf[0] = 0;
}

int main(void)
{
  return 0;
}
