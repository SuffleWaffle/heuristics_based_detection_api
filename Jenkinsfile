def gitCredentialsId = "github-creds"
def gitRepoUrl = 'git@github.com:Drawer-Inc/heuristics_based_detection_service.git'
def pom = ''

def build_service = '''
        ls -al
        chmod +x ./build_service.sh
        ./build_service.sh
    '''
def build_update_version = '''
        chmod +x ./build_update_version.sh
        ./build_update_version.sh
    '''

properties(
  [
    parameters(
      [
        [
          $class            : 'GitParameterDefinition',
          branch            : '',
          branchFilter      : '.*',
          defaultValue      : '${env.BRANCH}',
          description       : 'Branch or Tag',
          name              : 'BRANCH',
          quickFilterEnabled: false,
          selectedValue     : 'DEFAULT',
          sortMode          : 'ASCENDING_SMART',
          tagFilter         : '*',
          type              : 'PT_BRANCH'
        ]
      ]
    ),

    pipelineTriggers([
      [
      $class: 'GenericTrigger',
      genericVariables: [
        [
          key: 'BRANCH',
          value: '$.ref',
          expressionType: 'JSONPath'
        ],
              
      ],
        causeString: 'Triggered by Github',
        token: '70Hv4pkKfxCImWjtesEQ50mXVPD892jemFP0JheZ1dL9KH4GDPYjdu4HWGld2WPM',
        printContributedVariables: true,
        printPostContent: true,
        silentResponse: false,
        regexpFilterText: '$BRANCH',
        regexpFilterExpression:  '^(refs/heads/stage-ci|refs/heads/main|refs/heads/dev-ci|refs/tags/.+)$'
      ]
      ])

  ]
)


pipeline {
  environment {
    SERVICE_NAME    = 'heuristics_based_detection_service'
    AWS_REGION      = 'us-east-1'
    AWS_ACCOUNT     = '064427434392'
    SLACK_DOMAIN    = 'drawerai'
    SLACK_CHANNEL   = "#ci-cd-ai"
    SLACK_TOKEN     = credentials("slack-token")
    PROD_PASSWORD   = credentials("ai-prod-password")
    STAGE_PASSWORD  = credentials("ai-stage-password")
  }

  agent any

  options {
    buildDiscarder(logRotator(numToKeepStr: '20'))
    ansiColor('xterm')
    timestamps()
  }

 
  stages {

    stage('Prepare') {
      steps {
        script {
          currentBuild.displayName = "#${env.BUILD_NUMBER}-${env.BRANCH}"
        }
      }
    }

    stage('Checkout') {
      steps {
        checkout(
          [
            $class           : 'GitSCM',
            branches         : [[name: "${BRANCH}"]],
            userRemoteConfigs: [[url: "${gitRepoUrl}", credentialsId: "${gitCredentialsId}"]],
          ]
        )
      }
    }

  
    stage ('Docker Build') {
      steps{
        script { sh build_service }
      }
    }

  stage ('Prepare deploy') {
    stages{
        stage('Deploy dev'){
          when {
            anyOf {
              expression {env.BRANCH =~ /^refs\/heads\/dev-ci$/}
              expression {env.BRANCH =~ /^refs\/tags\/dev-.+$/}
              expression {env.BRANCH =~ /^origin\/dev-ci$/}
            }
          }
          steps {
            script {
              sshagent(['ai-dev-creds']) {
                sh '''
                  scp -o StrictHostKeyChecking=no -P 9376 deploy_service.sh administrator@66.117.7.18:/home/administrator
                  ssh -o StrictHostKeyChecking=no -l administrator 66.117.7.18  -p 9376  chmod +x deploy_service.sh
                  ssh -o StrictHostKeyChecking=no -l administrator 66.117.7.18  -p 9376  ./deploy_service.sh
               '''
              }
            }
          }  
        }
        stage('Deploy stage'){
          when {
            anyOf {
              expression {env.BRANCH =~ /^refs\/heads\/stage-ci$/}
              expression {env.BRANCH =~ /^refs\/tags\/stage-.+$/}
              expression {env.BRANCH =~ /^origin\/stage-ci$/}
            }
          }
          steps {
            script {
              sh '''
                sshpass -p $STAGE_PASSWORD scp -o StrictHostKeyChecking=no -P 9376 deploy_service.sh administrator@205.134.224.136:/home/administrator
                sshpass -p $STAGE_PASSWORD ssh -o StrictHostKeyChecking=no administrator@205.134.224.136 -p 9376 "sed -i 's/BRANCH_NAME=\\"dev-ci\\"/BRANCH_NAME=\\"stage-ci\\"/g' deploy_service.sh"
                sshpass -p $STAGE_PASSWORD ssh -o StrictHostKeyChecking=no administrator@205.134.224.136 -p 9376 "sed -i 's/ENV_NAME=\\"dev\\"/ENV_NAME=\\"stage\\"/g' deploy_service.sh"
                sshpass -p $STAGE_PASSWORD ssh -o StrictHostKeyChecking=no administrator@205.134.224.136 -p 9376 chmod +x deploy_service.sh 
                sshpass -p $STAGE_PASSWORD ssh -o StrictHostKeyChecking=no administrator@205.134.224.136 -p 9376 ./deploy_service.sh
              '''
            }
          }  
        }
        stage('Deploy prod'){
          when {
            anyOf {
              expression {env.BRANCH =~ /^refs\/heads\/main$/}
              expression {env.BRANCH =~ /^refs\/tags\/prod-.+$/}
              expression {env.BRANCH =~ /^origin\/main$/}
            }
          }
          steps {
            slackSend(
              color: 'warning',
              channel: SLACK_CHANNEL,
              message: "*${env.JOB_NAME}* - <${env.RUN_DISPLAY_URL}|#${env.BUILD_NUMBER}> " +
                "\n:warning: *WARNING* :warning: it seems to be deploying on PROD environment! " +
                "\nPlease, approve this step in Jenkins via <${env.JOB_URL}|link> " +
                "\n*Additional info:*" +
                "\nRepository: *${gitRepoUrl}*" +
                "\nCommit Hash: *${env.GIT_COMMIT}*",
              teamDomain: SLACK_DOMAIN,
              token: SLACK_TOKEN
            )
            timeout(time: 10, unit: "MINUTES") {
	            input message: 'Do you want to approve this deployment on prod?', ok: 'Approve'
	          }
            slackSend(
              color: 'good',
              channel: SLACK_CHANNEL,
              message: "Job *${env.JOB_NAME}* (<${env.RUN_DISPLAY_URL}|#${env.BUILD_NUMBER}>) is *approved* to deploy on PROD" +
              "\n:thumbsup:",
              teamDomain: SLACK_DOMAIN,
              token: SLACK_TOKEN
            )
            script {
              sh '''
                sshpass -p $PROD_PASSWORD scp -o StrictHostKeyChecking=no -P 9376 deploy_service.sh administrator@205.134.233.2:/home/administrator
                sshpass -p $PROD_PASSWORD ssh -o StrictHostKeyChecking=no administrator@205.134.233.2 -p 9376 "sed -i 's/BRANCH_NAME=\\"dev-ci\\"/BRANCH_NAME=\\"main\\"/g' deploy_service.sh"
                sshpass -p $PROD_PASSWORD ssh -o StrictHostKeyChecking=no administrator@205.134.233.2 -p 9376 "sed -i 's/ENV_NAME=\\"dev\\"/ENV_NAME=\\"prod\\"/g' deploy_service.sh"
                sshpass -p $PROD_PASSWORD ssh -o StrictHostKeyChecking=no administrator@205.134.233.2 -p 9376 chmod +x deploy_service.sh 
                sshpass -p $PROD_PASSWORD ssh -o StrictHostKeyChecking=no administrator@205.134.233.2 -p 9376 ./deploy_service.sh
              '''
            }
          }  
        }
      } 
    }
  }




  

    post {
    always {
      junit allowEmptyResults: true, testResults: '**/*Test.xml'
      cleanWs()
    }

    aborted {
      wrap([$class: 'BuildUser']) {
        slackSend(
          color: '#808080',
          channel: SLACK_CHANNEL,
          message: "*${env.JOB_NAME}* - <${env.RUN_DISPLAY_URL}|#${env.BUILD_NUMBER}> " +
            "Aborted after ${currentBuild.durationString.replaceAll(' and counting', '')}" +
            "\nRepository: *${gitRepoUrl}*" +
            "\nBranch: *${BRANCH}*" +
            "\nCommit Hash: *${env.GIT_COMMIT}*" +
            // "\nLaunched by: *${env.BUILD_USER}*" +
            "\n:octagonal_sign:",
          teamDomain: SLACK_DOMAIN,
          token: SLACK_TOKEN
        )
      }
    }

    failure {
      wrap([$class: 'BuildUser']) {
        slackSend(
          color: 'danger',
          channel: SLACK_CHANNEL,
          message: "*${env.JOB_NAME}* - <${env.RUN_DISPLAY_URL}|#${env.BUILD_NUMBER}> " +
            "Failed after ${currentBuild.durationString.replaceAll(' and counting', '')}" +
            "\nRepository: *${gitRepoUrl}*" +
            "\nBranch: *${env.GIT_BRANCH}*" +
            "\nCommit Hash: *${env.GIT_COMMIT}*" +
            // "\nLaunched by: *${env.BUILD_USER}*" +
            "\n:poop:",
          teamDomain: SLACK_DOMAIN,
          token: SLACK_TOKEN
        )
      }
    }

    success {
      wrap([$class: 'BuildUser']) {
        slackSend(
          color: 'good',
          channel: SLACK_CHANNEL,
          message: "*${env.JOB_NAME}* - <${env.RUN_DISPLAY_URL}|#${env.BUILD_NUMBER}> " +
            "Success after ${currentBuild.durationString.replaceAll(' and counting', '')}" +
            "\nRepository: *${gitRepoUrl}*" +
            "\nBranch: *${env.GIT_BRANCH}*" +
            "\nCommit Hash: *${env.GIT_COMMIT}*" +
            // "\nLaunched by: *${env.BUILD_USER}*" +
            "\n:tada:",
          teamDomain: SLACK_DOMAIN,
          token: SLACK_TOKEN
        )
      }
    }
  }

}