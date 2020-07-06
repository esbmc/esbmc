pipeline {
  agent {
    kubernetes {
      yaml '''
apiVersion: "v1"
kind: "Pod"
spec:
  containers:
    - name: "jnlp"
      image: "rafaelsamenezes/esbmc-build:latest"
      imagePullPolicy: "Always"
'''
    }

  }
  stages {
    stage('Build ESBMC') {
      when {
        expression {
          env.BRANCH_NAME.contains("benchexec")
        }

      }
      steps {
        sh 'mkdir build && cd build && cmake .. -GNinja -DClang_DIR=$CLANG_HOME -DLLVM_DIR=$CLANG_HOME -DBUILD_STATIC=On -DBoolector_DIR=$HOME/boolector-3.2.0 -DCMAKE_INSTALL_PREFIX:PATH=$PWD/../release'
        sh 'cd build && cmake --build . && cpack && mv ESBMC-*.sh ESBMC-Linux.sh'
        zip(zipFile: 'esbmc.zip', archive: true, glob: 'build/ESBMC-Linux.sh')
      }
    }

    stage('Run Benchexec') {
      when {
        expression {
          env.BRANCH_NAME.contains("benchexec")
        }

      }
      steps {
        script {
          def userInput = input(
            id: 'userInput', message: 'Type the category from benchmark file', parameters: [
              [$class: 'TextParameterDefinition', defaultValue: 'ConcurrencySafety-Main',  name: 'category']
            ])

          def built = build job: "benchexec-jenkins-job/high-res", parameters: [
            string(name: 'tool_url', value: "https://ssvlab.ddns.net/job/esbmc-master/job/${env.BRANCH_NAME}/$BUILD_NUMBER/artifact/esbmc.zip"),
            string(name: 'benchmark_url', value: "https://raw.githubusercontent.com/esbmc/esbmc/${env.BRANCH_NAME}/scripts/jenkins/benchmark.xml"),
            string(name: 'prepare_environment_url', value: "https://raw.githubusercontent.com/esbmc/esbmc/${env.BRANCH_NAME}/scripts/jenkins/prepare_environment.sh"),
            string(name: 'timeout', value: "900"),
            string(name: 'category', value: userInput)
          ]
        }

      }
    }

  }
}
