void install_signal_catcher();
void signal_catcher(int sig);

/// Report a fatal memory-fault signal on stderr before dying from it. Without
/// this, ESBMC's only trace of a SIGSEGV is the shell's exit status, which is
/// indistinguishable from a killed process.
void install_fatal_signal_reporter();
