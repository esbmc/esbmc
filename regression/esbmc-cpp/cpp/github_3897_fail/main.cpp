// Same as regression/esbmc-cpp/cpp/github_3897, but run WITHOUT
// --mix-cpp-host-headers: -nostdinc++ keeps the host C++ library off the
// search path, proving the flag is genuinely opt-in.
#if !__has_include(<bits/c++config.h>) && !__has_include(<__config>)
#  error "the host C++ library headers are not on the include path"
#endif

int main()
{
  __ESBMC_assert(1, "host C++ library headers should not be reachable");
  return 0;
}
