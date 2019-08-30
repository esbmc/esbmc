pipeline {
  agent any
  stages {
    stage('Build') {
      parallel {        
        stage('Autoconf') {          
          environment {
            CC = "gcc"
            CXX = "g++"
            CFLAGS = "-DNDEBUG -O3"
            CXXFLAGS = "-DNDEBUG -O3"
            solvers = "--with-boolector=$BTOR_DIR --with-mathsat=$MATHSAT_DIR --with-z3=$Z3_DIR --with-yices=$YICES_DIR --with-cvc4=$CVC4_DIR"
            destiny = "--prefix=/home/jenkins/agent/workspace/esbmc-private_cmake/release-autoconf/"
            static_python = "--enable-python --enable-static-link --disable-shared"
            build_targets = "--enable-esbmc --disable-libesbmc"
            clang_dir = "--with-clang=$CLANG_HOME --with-llvm=$CLANG_HOME"
            flags = "$destiny $clang_dir $build_targets $static_python $solvers --disable-werror"
          }
          steps {
            echo 'Building with Autoconf'
            sh 'mkdir -p release-autoconf'
            sh 'mkdir -p build-autoconf'
            dir(path: 'src') {
              sh './scripts/autoboot.sh'
            }
            dir(path: 'build-autoconf') {
              sh '../src/configure $flags'
              sh 'make -j`nproc`'
              sh 'make install'              
            }
            sh 'ls /home/jenkins/agent/workspace/esbmc-private_cmake/release-autoconf/bin/esbmc'
            archiveArtifacts(artifacts: 'release-autoconf/bin/esbmc', onlyIfSuccessful: true)
            stash includes: 'release-autoconf/bin/esbmc', name: 'build-autoconf'
          }
        }
      }
    }
    stage('Regression') {
      parallel {
        stage('ESBMC') {
          steps {
            unstash 'build-autoconf'
            dir(path: "regression/esbmc") {
              echo "$PWD"
              sh 'PATH=$PWD/../../:$PATH make default || true'
            }            
          }
        }        
      }
    }
  }
}
