#!/usr/bin/python

# -*- coding: utf-8 -*-

"""
PyCOMPSs Testbench Tasks
========================
"""

# Imports
import os
import time
from pycompss.api.api import compss_barrier_group, TaskGroup
from pycompss.api.parameter import FILE_INOUT, FILE_IN
from pycompss.api.task import task
from pycompss.api.exceptions import COMPSsException

NUM_TASKS = 3
NUM_GROUPS = 3
STORAGE_PATH = "/tmp/sharedDisk/"
TASK_SLEEP_TIME_FAST = 1
TASK_SLEEP_TIME_SLOW = 4

@task(file_path=FILE_INOUT)
def write_one(file_path):
    # Write value
    with open(file_path, 'a') as fos:
        new_value = str(1)
        fos.write(new_value)

@task(file_path=FILE_INOUT)
def write_one_slow(file_path):
    time.sleep(TASK_SLEEP_TIME_SLOW)
    # Write value
    with open(file_path, 'a') as fos:
        new_value = str(1)
        fos.write(new_value)

@task(file_path=FILE_INOUT)
def write_one_fast(file_path):
    time.sleep(TASK_SLEEP_TIME_FAST)
    # Write value
    with open(file_path, 'a') as fos:
        new_value = str(1)
        fos.write(new_value)

@task(file_path=FILE_INOUT)
def write_two(file_path):
    # Write value
    with open(file_path, 'a') as fos:
        new_value = str(2)
        fos.write(new_value)

@task(file_path=FILE_INOUT)
def write_three(file_path):
    # Write value
    with open(file_path, 'a') as fos:
        new_value = str(3)
        fos.write(new_value)
    raise COMPSsException("Exception has been raised!!")


def create_file(file_name):
    # Clean previous ocurrences of the file
    if os.path.exists(file_name):
        os.remove(file_name)
    # Create file
    if not os.path.exists(STORAGE_PATH):
        os.mkdir(STORAGE_PATH)
    open(file_name, 'w').close()


def test_exceptions(file_name):
    try:
        # Creation of group
        with TaskGroup('exceptionGroup1'):
            for i in range(NUM_TASKS):
                write_three(file_name)
    except COMPSsException:
        print("COMPSsException caught")
        write_two(file_name)
    write_one(file_name)


def test_exceptions_barrier(file_name):
    group_name = 'exceptionGroup2'
    # Creation of group
    with TaskGroup(group_name, False):
        for i in range(NUM_TASKS):
            write_three(file_name)
    try:
        # The barrier is not implicit and the exception is thrown
        compss_barrier_group(group_name)
    except COMPSsException:
        print("COMPSsException caught")
        write_two(file_name)
    write_one(file_name)


def test_exceptions_barrier_error(file_name):
    group_name = 'exceptionGroup3'
    # Creation of group
    with TaskGroup(group_name, False):
        for i in range(NUM_TASKS):
            write_three(file_name)

    # The barrier is not implicit and the exception is thrown
    compss_barrier_group(group_name)


def test_barrier_child():
    supergroup = 'SuperGroup_1'
    file_super = STORAGE_PATH + 'file_super_1'
    create_file(file_super)
    childgroup = 'childGroup_1'
    file_child = STORAGE_PATH + 'file_child_1'
    create_file(file_child)

    with TaskGroup(supergroup, False):
        for i in range(NUM_TASKS):
            write_one_slow(file_super)
            
        with TaskGroup(childgroup, False):
            for i in range(NUM_TASKS):
                write_one_fast(file_child)

    compss_barrier_group(childgroup)

def test_barrier_super():
    supergroup = 'SuperGroup_2'
    file_super = STORAGE_PATH + 'file_super_2'
    create_file(file_super)
    childgroup = 'childGroup_2'
    file_child = STORAGE_PATH + 'file_child_2'
    create_file(file_child)

    with TaskGroup(supergroup, False):
        for i in range(NUM_TASKS):
            write_one_fast(file_super)
            
        with TaskGroup(childgroup, False):
            for i in range(NUM_TASKS):
                write_one_slow(file_child)

    compss_barrier_group(supergroup)


def main():
    file_name1 = STORAGE_PATH + "taskGROUPS.txt"
    #create_file(file_name1)

    print("[LOG] Test EXCEPTIONS")
    #test_exceptions(file_name1)

    print("[LOG] Test EXCEPTIONS and BARRIERS")
    #test_exceptions_barrier(file_name1)

    print("[LOG] Test EXCEPTIONS and BARRIERS")
    #test_exceptions_barrier_error(file_name1)

    print("[LOG] Test Group hierarchy barrier - super")
    test_barrier_super()

    print("")
    print("")
    print("")
    print("[LOG] Test Group hierarchy barrier - internal")
    test_barrier_child()


if __name__ == '__main__':
    main()