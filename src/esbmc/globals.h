#pragma once

/** @brief Registers every file bundled into the binary with file_operations.
 *         Must run before anything reads a bundled path. */
void register_bundled_files();
