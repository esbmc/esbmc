#include <cstdint>
#include <esbmc/esbmc_parseoptions.h>
#include <langapi/mode.h>

#include <irep2/irep2.h>
#include <util/config.h>

int main(int argc, const char **argv)
{
  esbmc_parseoptionst parseoptions(argc, argv);
  return parseoptions.main();
}
