pipeline {
  agent any
  stages {
    stage('Build') {
      parallel {        
        stage('Autoconf') {
          environment {
            CC = 'gcc'
            CXX = 'g++'
            CFLAGS = '-DNDEBUG -O3'
            CXXFLAGS = '-DNDEBUG -O3'
            solvers = "--with-boolector=$BTOR_DIR --with-mathsat=$MATHSAT_DIR --with-z3=$Z3_DIR --with-yices=$YICES_DIR --with-cvc4=$CVC4_DIR"
            destiny = '--prefix=/home/jenkins/agent/workspace/esbmc-private_cmake/release-autoconf/'
            static_python = '--enable-python --enable-static-link --disable-shared'
            build_targets = '--enable-esbmc --disable-libesbmc'
            clang_dir = "--with-clang=$CLANG_HOME --with-llvm=$CLANG_HOME"
            flags = "$destiny $clang_dir $build_targets $static_python $solvers --disable-werror"
          }
          steps {
            echo 'Building with Autoconf'
            sh 'mkdir -p build-autoconf'
            dir(path: 'src') {
              sh './scripts/autoboot.sh'
            }

            dir(path: 'build-autoconf') {
              sh '../src/configure $flags'
              sh 'make -j`nproc`'
              stash(includes: 'esbmc/esbmc', name: 'build-autoconf')
              sh 'make install'
            }

            zip(zipFile: 'autoconf.zip', archive: true, dir: '/home/jenkins/agent/workspace/esbmc-private_cmake/release-autoconf/')
          }
        }
      }
    }
    stage('Test') {
      parallel {
        stage('ESBMC') {
          steps {
            dir(path: 'regression') {
              sh 'python3 testing_tool.py --tool="$PWD/../../build-autoconf/esbmc" --regression="./esbmc" --mode="CORE"'
            }
          }
        }
        stage('cstd - ctype') {
          steps {
            dir(path: 'regression') {
              sh 'python3 testing_tool.py --tool="$PWD/../../build-autoconf/esbmc" --regression="./cstd/ctype" --mode="CORE"'
            }
          }
        }
        stage('cstd - string') {
          steps {
            dir(path: 'regression') {
              sh 'python3 testing_tool.py --tool="$PWD/../../build-autoconf/esbmc" --regression="./cstd/string" --mode="CORE"'
            }
          }
        }
        stage('k-induction') {
          steps {
            dir(path: 'regression') {
              sh 'python3 testing_tool.py --tool="$PWD/../../build-autoconf/esbmc" --regression="./k-induction" --mode="CORE"'
            }
          }
        }
        stage('llvm') {
          steps {
            dir(path: 'regression') {
              sh 'python3 testing_tool.py --tool="$PWD/../../build-autoconf/esbmc" --regression="./llvm" --mode="CORE"'
            }
          }
        }
        stage('digital-filters') {
          steps {
            dir(path: 'regression') {
              sh 'python3 testing_tool.py --tool="$PWD/../../build-autoconf/esbmc" --regression="./digital-filters" --mode="CORE"'
            }
          }
        }
        stage('floats') {
          steps {
            dir(path: 'regression') {
              sh 'python3 testing_tool.py --tool="$PWD/../../build-autoconf/esbmc" --regression="floats" --mode="CORE"'
            }
          }
        }
        stage('floats regression') {
          steps {
            dir(path: 'regression') {
              sh 'python3 testing_tool.py --tool="$PWD/../../build-autoconf/esbmc" --regression="./floats-regression" --mode="CORE"'
            }
          }
        }
        stage('CPP') {
          steps {
            echo 'CPP testing is currently not working'
          }
        }
      }
    }
  }
  post {
        always {
            junit 'regression/test-reports/*.xml'
        }
    }
}