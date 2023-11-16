# SLURM Job Submission Python Scripts

## experiments_in_json.py

***Description***\
Create a JSON file configuration of experiments' parameters.

***Options:***\
  -**h**, --**help**            show this help message and exit\
  --**clients** CLIENTS [CLIENTS ...]\
                        number of clients in the network\
  --**algorithms** ALGORITHMS [ALGORITHMS ...]\
                        algorithms used\
  --**models** MODELS [MODELS ...]\
                        ANN models used\
  --**epochs** EPOCHS       number of epochs that the experiment will run\
  --**thresholds** THRESHOLDS [THRESHOLDS ...]\
                        Thresholds to be used in FDA experiments. *First add the default threshold!*\
  --**batch_sizes** BATCH_SIZES [BATCH_SIZES ...]\
                        Batch sizes to be used in experiments. *First add the default batch size!*\
  --**repetitions** REPETITIONS\
                        How many times to repeat the same experiments.\
  --**print**           Print the parameters instead of saving in JSON 


***An example***
```bash
python3 experiments_in_json.py --clients 5 --algorithms synchronous naive --models lenet5 adv_cnn \
  --epochs 100 --thresholds 1.0 0.5 1.5 2.0  --batch_sizes 64 32 128 256 --repetitions 3
```
This will return 
```
 66 experiments have been saved in file: json_experiments/experiments_5.json
```

## run_experiments.py

***Description***\
Run multiple KungFu commands based on a JSON file of experiments parameters 

***Options:***\
  -**h**, --**help**         show this help message and exit\
  --**clients** CLIENTS  number of clients in the network\
  --**nodes** NODES      number of nodes in the network\
  --**ips** IPS          a list of ips in the form: IP1,IP2,...\
  --**nic** NIC          the NIC of the network\
  --**json** JSON        name of JSON file that contains the experiments' paramaters\
  --**print**            print commands instead of running them

***An example***
```bash
python3 run_experiments.py --clients 5 --nodes 2 --ips 192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4,192.0.2.5\
  --nic eth0 --json json_experiments/experiments_5.json
```
This will return 
```
Clients at each node: [3,2]
```
And run all the experiments consequently

## job_submitter.py

***Description***\
Generate and submit a SLURM job

***Options:***\
  -**h**, --**help**            show this help message and exit\
  --**special_name** SPECIAL_NAME
                        Special naming for the job name\
  --**clients** CLIENTS     Number of clients\
  --**clients_distr** clients_distr
                        Clients for each node.\
  --**partition** PARTITION
                        SLURM partition where the tests will run\
  --**account** ACCOUNT     SLURM account\
  --**time** TIME           Wall time of job\
  --**json** JSON           name of JSON file that contains the experiments' paramaters\
  --**script_name** SCRIPT_NAME
                        SLURM job script name

***An example***
```bash
python3 job_submitter.py --clients 5 clients_distr 3 --partition gpu\
   --account pa1243 --time 01:00:00 --json experiments_5.json --script_name slurm_job.sh
```
This will return
```
2 nodes will be used
Submitted batch job JOB_ID
```
