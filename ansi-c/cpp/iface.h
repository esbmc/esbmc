

/* Interface between ESBMC C parsing foo and pcc's cpp */

void record_define(const char *value); /* IE, cmdline -D, "bees=dangerous"; */
void record_include(const char *value); /* Similar, include path name */
void record_builtin_macros(); /* Insert builtin macros into sym table */
int open_output_file(const char *name); /* Obvious */
void fin(); /* Flushes buffers and closes file */
int pushfile(char *name);
int pushfile2(const char *fname, const char *fn, int idx, void *incs);
void fastscan();

void cpp_clear(void); // Clear some memory; doesn't free it though.
