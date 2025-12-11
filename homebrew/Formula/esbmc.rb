class Esbmc < Formula
  desc "Context-Bounded Model Checker for C/C++/Python programs"
  homepage "https://esbmc.org"
  url "https://github.com/esbmc/esbmc/archive/refs/tags/v7.11.tar.gz"
  sha256 "2485e7dcdb325e883f6432d90a0389f6321fdfcb34d7462f4f89766c4a8646da"
  license "Apache-2.0"
  head "https://github.com/esbmc/esbmc.git", branch: "master"

  depends_on "bison" => :build
  depends_on "cmake" => :build
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
    system python3, "-m", "pip", "install", "--break-system-packages", "--upgrade", "pip"
    system python3, "-m", "pip", "install", "--break-system-packages", "ast2json", "mypy"

    args = %W[
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
      -DCMAKE_INSTALL_PREFIX=#{prefix}
      -DLLVM_DIR=#{Formula["llvm@16"].opt_lib}/cmake/llvm
      -DClang_DIR=#{Formula["llvm@16"].opt_lib}/cmake/clang
      -DC2GOTO_SYSROOT=#{MacOS.sdk_path}
      -DENABLE_PYTHON=ON
      -DPython3_EXECUTABLE=#{python3}
      -DENABLE_FUZZER=OFF
      -DENABLE_Z3=ON
      -DZ3_DIR=#{Formula["z3"].opt_lib}/cmake/z3
      -DBUILD_TESTING=OFF
    ]

    system "cmake", "-S", ".", "-B", "build", "-G", "Ninja", *args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    assert_match "ESBMC version", shell_output("#{bin}/esbmc --version")

    (testpath/"test.c").write <<~EOS
      #include <assert.h>
      int main() {
        int x = 5;
        assert(x == 5);
        return 0;
      }
    EOS
    system bin/"esbmc", "test.c", "--no-bounds-check", "--no-pointer-check"
  end
end
