#!/bin/bash

mkdir results
mkdir plots

rm results/*
rm plots/*

sbatch --partition=CPU --chdir=`pwd` --time=08:00:00 --ntasks=1 --cpus-per-task=1 --job-name=gold --mem=100G ./bench.sh