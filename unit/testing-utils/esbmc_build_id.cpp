#include <esbmc/globals.h>

// The real definition reads buildidstring_buf out of the object flail.py
// generates for the executable at link time (src/esbmc/CMakeLists.txt), so it
// exists only in a real esbmc binary. Tests that link esbmc-driver pull in
// objects that report the build id; none of them depend on its value.
// Use as an OBJECT library, like mode_table, so the .o is always in the link.
std::string esbmc_build_id()
{
  return "unit-test build";
}
