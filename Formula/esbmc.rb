class Esbmc < Formula
  desc "Context-Bounded Model Checker for C/C++/Python programs"
  homepage "https://esbmc.org"
  url "https://github.com/esbmc/esbmc/archive/4e261e3ddfe7de0bd2ad681dacfcd7136b262e68.tar.gz"
  version "7.11-nightly-20241208"
  sha256 "d75e2656220ff123659658498da218f2d3905a0a6dfa054d24380ccb620bb7bc"
  license "Apache-2.0"
  head "https://github.com/esbmc/esbmc.git", branch: "master"

  depends_on "bison" => :build
  depends_on "cmake" => :build
  depends_on "csmith" => :build
  depends_on "flex" => :build
  depends_on "ninja" => :build

  depends_on "boost"
  depends_on "fmt"
  depends_on "gmp"
  depends_on "llvm@16"
  depends_on "nlohmann-json"
  depends_on "python@3.12"
  depends_on "yaml-cpp"
  depends_on "z3"

  def install
    python3 = Formula["python@3.12"].opt_bin/"python3.12"
    system python3, "-m", "pip", "install", "--break-system-packages",
           "--upgrade", "pip"
    system python3, "-m", "pip", "install", "--break-system-packages",
           "meson", "ast2json", "mypy", "pyparsing", "toml", "tomli"

    args = %W[
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
      -DCMAKE_INSTALL_PREFIX=#{prefix}
      -DLLVM_DIR=#{Formula["llvm@16"].opt_lib}/cmake/llvm
      -DClang_DIR=#{Formula["llvm@16"].opt_lib}/cmake/clang
      -DC2GOTO_SYSROOT=#{MacOS.sdk_path}
      -DPython3_EXECUTABLE=#{python3}
      -DDOWNLOAD_DEPENDENCIES=ON
      -DENABLE_CSMITH=ON
      -DENABLE_PYTHON_FRONTEND=ON
      -DENABLE_SOLIDITY_FRONTEND=ON
      -DENABLE_JIMPLE_FRONTEND=ON
      -DENABLE_REGRESSION=ON
      -DBUILD_TESTING=ON
      -DENABLE_FUZZER=OFF
      -DENABLE_Z3=ON
      -DZ3_DIR=#{Formula["z3"].opt_lib}/cmake/z3
      -DENABLE_BOOLECTOR=OFF
      -DENABLE_BITWUZLA=OFF
      -DENABLE_GOTO_CONTRACTOR=OFF
      -DBUILD_STATIC=OFF
      -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON
    ]

    system "cmake", "-S", ".", "-B", "build", "-G", "Ninja", *args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    assert_match "ESBMC version", shell_output("#{bin}/esbmc --version")

    # Test C verification
    (testpath/"test.c").write <<~EOS
      #include <assert.h>
      int main() {
        int x = 5;
        assert(x == 5);
        return 0;
      }
    EOS
    system bin/"esbmc", "test.c", "--no-bounds-check", "--no-pointer-check"

    # Test Python frontend
    (testpath/"test.py").write <<~EOS
      def main():
          x: int = 5
          assert x == 5

      if __name__ == "__main__":
          main()
    EOS
    system bin/"esbmc", "test.py"
  end
end
