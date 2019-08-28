# Homework 11

**This task is not mandatory.** Feel free to complete it for curiosity sake to get hands on Docker and Kubernetes.

In this homework your task is to build fully automated ML pipeline for the CV model from [HW9](https://ml-course-git.railsreactor.net/summer-19/summer-19/blob/master/tasks/09-cv.md).

Steps to follow:

1. Create a [Gitlab](https://gitlab.com) and [DockerHub](https://hub.docker.com/) accounts
2. Create a repo in your Gitlab account use [lecture demo repo](https://gitlab.com/tilast/pipeline-test) as an example. You will need only `runtime` directory (`train` should be skipped) since we will use pretrained model. Also, create repo in DockerHub that will contain Docker images for the project
3. Configure `kubectl` locally and connect to k8s cluster by using provided .yaml file with credentials (see discord)
4. Create a seldon service that wraps MobileNet network from the HW9
5. Write some unit and interation tests for your service
6. Implement gitlab ci/cd pipeline that builds the docker image, run tests and linter, deploys service to k8s. Please, pay attention to CI/CD settings section of the repo where you should locate your credentials
7. Perform basic load testing of the model (e.g. [ab](https://httpd.apache.org/docs/2.4/programs/ab.html))

Results must be provided as a:

1. Gitlab repo
2. Terminal screenshots that show deployed service and interaction with it

# Deadline

As it was mentioned, this task is optional, thus, deadline just outlines the date when the k8s cluster will be turned off so you won't be able to deploy anything to it:

**Due on 13.09.2019 23:59**
