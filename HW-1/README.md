Data Science HW 1
===

Given a set of transactions and the minimum support, find the frequent patterns.

This repository is an implementation of FP-Growth

## Input

An `.txt` file about the information of the transaction set:

* Items are represented by numbers in 0~999.
* There are at most 100,000 transactions.
* For each transaction, there are at most 200 items.
* Each line indicates a transaction, and each item in the transaction are separated by a single `,`.
* Use `\n` as a new line (instead of `\r\n`).

## Output

A new `.txt` file of the frequent patterns with its minimum support:

* Each line shows a set of frequent patterns, followed by `:` and its minimum support rounded to 4 decimal places.

## Usage (Command Line)

* Environment
```
CPU: i7-8700k
RAM: 32G
OS: Ubuntu 20.04.3 LTS
gcc vesion: 11.3.0 
Python version: python 3.9.13 
```

* Complie
```
g++ -std=c++2a -pthread -fopenmp -O2 -o <sutdentID>_hw1 <studentID>_hw1.cpp
```
Example:
```
g++ -std=c++2a -pthread -fopenmp -O2 -o 108020017_hw1 108020017_hw1.cpp
```
* Run
```
./<studentID>_hw1 <min support> <input file> <output file>
```
Example: 
```
./108020017_hw1 0.2 sample.txt test.txt
```