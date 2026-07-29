#include <cstdint>
#include <esbmc/esbmc_parseoptions.h>
#include <esbmc/globals.h>
#include <langapi/mode.h>

#include <irep2/irep2.h>
#include <util/config/config.h>

int main(int argc, const char **argv)
{
  register_bundled_files();
  esbmc_parseoptionst parseoptions(argc, argv);
  return parseoptions.main();
}
