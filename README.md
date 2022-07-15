# HiPar22

This repository contains all the necessary data to reproduce the experiments conducted for the article "" presented in the 3rd Workshop on Hierarchical Parallelism for Exascale Computing (HiPar22). 

## REPOSITORY ORGANIZATION
The repository is organized in two main folders: framework and tests. 

The framework folder contains the source code of the prototype implementation described in the article as well as all the necessary scripts to install and launch generic applications.

The tests folders contains all the information related to the tests presented on the article. There is a folder for each experiment presented in the article (random_forest, gridsearch and matmul). Within each folder there is the source code of the application, the datasets used and the necessary scripts and configuration files to run the test. Each of the folders contains a README file explaining its content and the test.



## ENVIRONMENT PREPARATION
To ease the execution of the tests, a container with all the environment set up has been published in docker hub with the tag `francesc.lordan/hipar22:latest`. It can be fetch with the following command
```bash
$ docker pull francesc.lordan/hipar22:latest
```
To set up the environment on a laptop, it is necessary to install the prototype implementation building it from the sources provided in this repository. The prototype is build on COMPSs v3.0; therefore, it inherits all its [dependencies](https://compss-doc.readthedocs.io/en/3.0/Sections/01_Installation/01_Dependencies.html).

To install COMPSs on a laptop from its sources run:
```bash
> cd framework/builders/
> sudo -E ./buildlocal  -K -J -T -D -C -M --skip-tests /opt/COMPSs
```

To install the runtime on supercomputers, there is additional information on the [Installing in Supercomputers](https://compss-doc.readthedocs.io/en/3.0/Sections/01_Installation/04_Supercomputers.html#) page from COMPSs' official documentation.

## LAUNCHING THE TESTS
The repository provides a script to launch all the experiments `launch_test`. The README file within each test's folder describes how to launch the test on a bare-metal installation, using the container or enqueuing the execution in a cluster managed with a queue system.

The `launch_test` script allows launching bare-metal executions indicating the test application (random_forest, gridsearch and matmul) and version as parameters. Once selected, each application accepts different parameters that are further detailed in the README file within the corresponding folder.
```bash
> launch_test <app> <version> [parameters]
```

The container automatically calls this script; therefor it is only necessary to indicate the application, version and parameters.
```bash
> docker run --rm francesc.lordan/hipar22:latest <app> <version> [parameters]
```

To run tests on supercomputers, each folder contains a script that enqueues the corresponding execution using the same parameters.
