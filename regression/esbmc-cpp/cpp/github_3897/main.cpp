// --mix-cpp-host-headers puts the host C++ library's include tree on the search
// path next to the bundled OMs. Probe for its configuration header (libstdc++,
// libc++) rather than include a standard header that may later be modelled.
#if !__has_include(<bits/c++config.h>) && !__has_include(<__config>)
#  error "the host C++ library headers are not on the include path"
#endif

int main()
{
  __ESBMC_assert(1, "host C++ library headers are reachable");
  return 0;
}
