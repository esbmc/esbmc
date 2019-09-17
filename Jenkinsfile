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
    stage('Regression') {
      parallel {
        stage('ESBMC') {
          steps {
            dir(path: 'regression/esbmc') {
              sh 'PATH=$PWD/../../build-autoconf/esbmc:$PATH ../test.pl -c esbmc'
            }
          }
        }
        stage('cstd - ctype') {
          steps {
            dir(path: 'regression/cstd/ctype') {
              sh 'PATH=$PWD/../../../build-autoconf/esbmc:$PATH ../../test.pl -c esbmc'
            }
          }
        }
        stage('cstd - string') {
          steps {
            dir(path: 'regression/cstd/string') {
              sh 'PATH=$PWD/../../../build-autoconf/esbmc:$PATH ../../test.pl -c esbmc'
            }
          }
        }
        stage('k-induction') {
          steps {
            dir(path: 'regression/k-induction') {
              sh 'PATH=$PWD/../../build-autoconf/esbmc:$PATH ../test.pl -c esbmc'
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
}