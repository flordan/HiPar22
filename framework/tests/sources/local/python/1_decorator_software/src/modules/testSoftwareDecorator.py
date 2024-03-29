#!/usr/bin/python

# -*- coding: utf-8 -*-

"""
PyCOMPSs Testbench Tasks
========================
"""

# Imports
import unittest
import os

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import compss_barrier, compss_wait_on
from pycompss.api.software import software


@software(config_file=os.getcwd() + "/src/config/multinode.json")
@task(returns=int)
def multi_node_task():
    # Expected values
    expected_num_nodes = 2
    expected_num_threads = 2
    expected_hostnames = ["COMPSsWorker01", "COMPSsWorker02"]

    # Check the environment variables
    import os

    num_nodes = int(os.environ["COMPSS_NUM_NODES"])
    if num_nodes != expected_num_nodes:
        print("ERROR: Incorrect number of nodes")
        print("  - Expected: " + str(expected_num_nodes))
        print("  - Got: " + str(num_nodes))
        return 1

    num_threads = int(os.environ["COMPSS_NUM_THREADS"])
    if num_threads != expected_num_threads:
        print("ERROR: Incorrect number of threads")
        print("  - Expected: " + str(expected_num_threads))
        print("  - Got: " + str(num_threads))
        return 2

    hostnames = sorted(os.environ["COMPSS_HOSTNAMES"].split(","))
    if hostnames != expected_hostnames:
        print("ERROR: Incorrect hostnames")
        print("  - Expected: " + str(expected_hostnames))
        print("  - Got: " + str(hostnames))
        return 3

    omp_num_threads = int(os.environ["OMP_NUM_THREADS"])
    if omp_num_threads != expected_num_threads:
        print("ERROR: Incorrect number of OMP threads")
        print("  - Expected: " + str(expected_num_threads))
        print("  - Got: " + str(omp_num_threads))
        return 4

    # All ok
    return 0


@software(config_file=os.getcwd() + "/src/config/mpi_basic.json")
@task()
def my_date(d_prefix, param):
    pass


@software(config_file=os.getcwd() + "/src/config/pro_epi.json")
@task()
def prolog_epilog():
    pass


@software(config_file=os.getcwd() + "/src/config/binary_basic.json")
@task()
def my_date_binary(d_prefix, param):
    pass


@software(config_file=os.getcwd() + "/src/config/mpi_constrained.json")
@task()
def my_date_constrained(d_prefix, param):
    pass


@software(config_file=os.getcwd() + "/src/config/mpi_file_in.json")
@task(file=FILE_IN)
def my_sed_in(expression, file):
    pass


@software(config_file=os.getcwd() + "/src/config/mpi_param.json")
@task(returns=int)
def mpi_with_param(string_param):
    pass


@software(config_file=os.getcwd() + "/src/config/mpmd.json")
@task(returns=int)
def mpmd(a, b):
    pass


class TestSoftwareDecorator(unittest.TestCase):

    def testFunctionalUsageMPI(self):
        my_date("-d", "next friday")
        compss_barrier()

    def testFunctionalUsageBinary(self):
        my_date_binary("-d", "next saturday")
        compss_barrier()

    def testFunctionalUsageWithConstraint(self):
        my_date_constrained("-d", "next monday")
        compss_barrier()

    def testFileManagementIN(self):
        infile = "src/infile"
        my_sed_in('s/Hi/HELLO/g', infile)
        compss_barrier()

    def testPrologEpilog(self):
        prolog_epilog()
        compss_barrier()

    def testMpmdMPI(self):
        mpmd("last tuesday", "next tuesday")
        compss_barrier()

    def testStringParams(self):
        string_param = "this is a string with spaces"
        exit_value1 = mpi_with_param(string_param)
        exit_value2 = mpi_with_param(string_param)
        exit_value1 = compss_wait_on(exit_value1)
        exit_value2 = compss_wait_on(exit_value2)
        self.assertEqual(exit_value1, 0)
        self.assertEqual(exit_value2, 0)

    def testMultinode(self):
        ev = multi_node_task()
        ev = compss_wait_on(ev)
        self.assertEqual(ev, 0)
