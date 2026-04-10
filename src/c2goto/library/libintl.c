#include <stddef.h>
#include <stdbool.h>

char *bindtextdomain(const char *domainname, const char *dirname)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    domainname != NULL, "bindtextdomain called with NULL domainname");
  __ESBMC_assert(
    *domainname != '\0', "bindtextdomain called with empty domainname");

  // If dirname is NULL, query current binding
  if (dirname == NULL)
  {
    static char current_dir[256] = "/usr/share/locale";
    return current_dir;
  }

  // Non-deterministically model success or failure
  _Bool success = nondet_bool();

  if (!success)
    return NULL;

  // On success, return the directory name
  return (char *)dirname;
}

char *textdomain(const char *domainname)
{
__ESBMC_HIDE:;
  // If domainname is NULL, query current domain
  if (domainname == NULL)
  {
    static char current_domain[256] = "messages";
    return current_domain;
  }

  __ESBMC_assert(
    *domainname != '\0', "textdomain called with empty domainname");

  // Non-deterministically model success or failure
  _Bool success = nondet_bool();

  if (!success)
    return NULL;

  // On success, return the domain name
  return (char *)domainname;
}