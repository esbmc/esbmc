/* The IREP2 adjuster leaves a member expression inside a VLA size in a type
 * unadjusted (github_2220). A declaration's VLA size now lives in a local
 * (R63), so only a size that stays in a type, like this sizeof operand's,
 * still reaches the defect. */
struct dirent
{
  char d_name[256];
};

unsigned long strlen(const char *);

void g(struct dirent *entry)
{
  unsigned long s = sizeof(char[strlen(entry->d_name) + 2]);
  (void)s;
}

int main(void)
{
  return 0;
}
