# Medical Diagnosis System

A medical diagnosis system built from 10,000 broken (1 attribute missing) patient records using Bayesian Networks.

- **problem.pdf** - The problem statement
- **alarm.bif** - Describes the whole bayesian network of diseases and symptoms.
- **records.dat** - Contains all the patient records. 
- **gold_alarm.bif** - Contains ideal probability values.

To run -

```
bash compile.sh
bash run.sh alarm.bif <sample_data>.dat

```

It will generate a file **solved_alarm.bif**. 

To check the correctness - 

```
g++ -o format_check Format_checker.cpp
./format_check
```
It will output a score which is a measure of error from the gold values.
