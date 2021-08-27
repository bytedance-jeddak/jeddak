# Quick Start

This is a guide on the deployment of this project.

## Deploy Preliminary Packages and Environment

### Anaconda

Download and install
```shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
#For macos, use: wget https://repo.anaconda.com/archive/Anaconda3-2020.11-MacOSX-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
```

Exit and re-login to configure

``conda config --set auto_activate_base false``

``conda create -n ${your_env_name} python=3.7``

``conda activate ${your_env_name}``

Set PYTHONPATH

``export PYTHONPATH=${project_directory}``

### Kafka

version = kafka-2.7.0

#### Download

``wget https://downloads.apache.org/kafka/2.7.0/kafka_2.13-2.7.0.tgz``

``tar -xvf kafka_2.13-2.7.0.tgz``

``cd kafka_2.13-2.7.0``

#### Run

``bin/zookeeper-server-start.sh -daemon config/zookeeper.properties``

``bin/kafka-server-start.sh -daemon config/server.properties``

## Deploy Jeddak Service

### Python Dependencies

```shell
apt-get install libgmp-dev libmpc-dev libmpfr-dev
#for macos : brew install gmp mpfr libmpc librdkafka
#run in the conda sandbox environment by `conda activate ${your_env_name}`
pip install -r requirements.txt
```

### Compile C library

``sh build.sh``

### Run Jeddak Service

Usage: python setup.py [-h] [--host HOST] [--port PORT] [--debug DEBUG]  [--threaded THREADED] [--online] --syncer_server SYNCER_SERVER --syncer_type SYNCER_TYPE --party_name PARTY_NAME

Examples: 
```bash
# start a party named "142". clear the remaining task queue
python setup.py \
    --host localhost \
    --port 8090 \
    --debug True \
    --syncer_server localhost:9092 \
    --syncer_type light_kafka \
    --party_name 142 \
    --debug True \
    --reset_task_queue
    
# start a party named "154". to sync task parameters by kafka
python setup.py \
    --host localhost \
    --port 8091 \
    --debug True \
    --syncer_server localhost:9092 \
    --syncer_type light_kafka \
    --party_name 154 \
    --debug True
```

|  Parameter   | Type  | Default | Description | Example |
|  ----  | ----  | ---- | ----  | ---- |
| host | str | "localhost" | hostname or ip address to deploy the service | |
| port | int | 5000 | port to deploy the service | |
| debug | bool | False | enables the debug mode on the Flask | |
| threaded | bool | False | served by a multithreaded mode | |
| syncer_server | str | None | required. syncer_server address | *ip*:*port*, *ip*:*port*:*password* |
| syncer_type | str | None | required. type of syncer_server | light_kafka|
| party_name | str | None | required. port to deploy the service | '' |
| reset_task_queue | bool | False | clear the task queue in syncer_server. we suggest to use it on at least one side of party (`guest` for example). |  |

Finally, post ``http://${active_party_ip}:8090/task_chain/submit`` （, where 8090 is the party port）with
````
[
    {
        "task_type": "task_chain",
        "party_names": ["142","154"],
        
        "messenger_server": [
            ["{messenger_ip}:9092"],
            ["{messenger_ip}:9092"]
        ],
        "messenger_type": ["kafka", "kafka"],
        "save_model": true
    },
    {
        "task_type": "data_loader",
        "task_role": ["guest", "host"],
        "input_data_source": ["csv", "csv"],
        "input_data_path": [
            "/data00/workspace/byte_federated_learning/example/data/test_guest.csv",
            "/data00/workspace/byte_federated_learning/example/data/test_host.csv"
        ]
    },
    {
        "task_type": "aligner",
        "task_role": ["guest", "host"],
        "align_mode": ["diffie_hellman", "diffie_hellman"],
        "output_id_only": [true, true],
        "sync_intersection": [true, true],
        "key_size": [1024, 1024]
    }
]
````

### Test in Jeddak Board (web page GUI)

Open `http://{host}:{port}` in a web browser.
Notes that "admin" is the default username and password.

### Test in Terminals

Please modify `KAFKA_DIR` dir and `ACT_SERVER_IP`, `PAS_SERVER_IP`, `ACT_SERVER_PORT`, `PAS_SERVER_PORT` 
in the files below, according to the real path and ip/ports of host, then run them by commands:

```
./example/standalone-start.sh
```

Modify task information in variable 'body3' of file 'example/standalone-test.sh', such as 

```
"input_data_path": [
            "{project_dir}/jeddak/example/data/test_guest.csv",
            "{project_dir}/jeddak/example/data/test_host.csv"
]
```

replace it by the real file path.

Then input the following command.

```
./example/standalone-test.sh
```

You can check the logs in `common/log/xxx/xxxx.log` files and find such records:

```
finish task xxxxxxx: aligner
finish task xxxxxxx: aligner
```
