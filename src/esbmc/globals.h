#pragma once

#include <string>

/** @brief Registers every file bundled into the binary with file_operations.
 *         Must run before anything reads a bundled path. */
void register_bundled_files();

/** @brief The build ID linked into this binary: the commit it was built from,
 *         whether that tree was dirty, and who built it. Written on every
 *         build by scripts/buildidobj.py. */
std::string esbmc_build_id();
