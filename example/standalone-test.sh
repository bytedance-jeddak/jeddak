#!/bin/bash
set -e

ROOT_DIR=$HOME

ACT_SERVER_IP=localhost
PAS_SERVER_IP=localhost
ACT_SERVER_PORT=8090
PAS_SERVER_PORT=8091

SOURCE="$0"
while [ -h "$SOURCE"  ]; do
    DIR="$( cd -P "$( dirname "$SOURCE"  )" && pwd  )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /*  ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE"  )" && pwd  )"

PRJ_DIR="$DIR/.."

# aligner demo, copy and paste example/conf/aligner.json here
body3='[
    {
        "task_type": "task_chain",
        "party_names": ["142", "154"],
        "messenger_server": [
            ["127.0.0.1:9092"],
            ["127.0.0.1:9092"]
        ],
        "messenger_type": [
            "kafka",
            "kafka"
        ],
        "save_model": true
    },
    {
        "task_type": "data_loader",
        "task_role": ["guest", "host"],
        "input_data_source": ["csv", "csv"],
        "input_data_path": [
            "{project_root_dir}/example/data/party_a_train.data",
            "{project_root_dir}/example/data/party_b_train.data"
        ]
    },
    {
        "task_type": "aligner",
        "align_mode": ["cm20", "cm20"],
        "task_role": ["guest", "host"],
        "output_id_only": [true, true],
        "sync_intersection": [true, true],
        "key_size": [1024, 1024]
    }
]'

## dpgbdt demo, copy and paste example/conf/dpgbdt.json here
#body3='[
#    {
#        "task_type": "task_chain",
#        "party_names": ["142", "154"],
#        "messenger_server": [
#            ["127.0.0.1:9092"],
#            ["127.0.0.1:9092"]
#        ],
#        "messenger_type": [
#            "kafka",
#            "kafka"
#        ],
#        "save_model": true
#    },
#    {
#        "task_type": "data_loader",
#        "task_role": ["guest", "host"],
#        "input_data_source": ["csv", "csv"],
#        "train_data_path": [
#            "{project_root_dir}/example/data/party_a_train.data",
#            "{project_root_dir}/example/data/party_b_train.data"
#        ],
#        "validate_data_path": [null, null],
#        "convert_sparse_to_index": [true, true]
#    },
#    {
#        "task_type": "aligner",
#        "task_role": [
#            "guest",
#            "host"
#        ],
#        "align_mode": [
#            "cm20",
#            "cm20"
#        ],
#        "output_id_only": [
#            false,
#            false
#        ],
#        "sync_intersection": [
#            true,
#            true
#        ]
#    },
#    {
#        "task_type": "dpgbdt",
#        "task_role": ["guest", "host"],
#        "objective": ["binary_logistic", "binary_logistic"],
#        "base_score": [0.5, 0.5],
#        "num_round": [20, 20],
#        "eta": [0.3, 0.3],
#        "gamma": [0.0, 0.0],
#        "max_depth": [3, 3],
#        "min_child_weight": [1.0, 1.0],
#        "max_delta_step": [0, 0],
#        "sub_sample": [1.0, 1.0],
#        "lam": [1.0, 1.0],
#        "sketch_eps": [0.3, 0.3],
#        "train_validate_freq": [5, 5],
#        "homomorphism": ["cpaillier", "cpaillier"],
#        "key_size": [1024, 1024],
#        "privacy_mode": ["homo", "homo"]
#    },
#    {
#        "task_type": "evaluate",
#        "task_role": ["guest", "host"]
#    }
#]'

body3=$(echo $body3 | sed -e 's/\n//g')
body3=$(echo $body3 | sed -e 's/\ //g')
body3=${body3//"{project_root_dir}"/"$PRJ_DIR"} 

curl $ACT_SERVER_IP:$ACT_SERVER_PORT/task_chain/submit -X POST -d $body3