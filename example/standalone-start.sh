#!/bin/bash
# set -e

ROOT_DIR=~/workspace/jeddak
KAFKA_DIR=~/workspace/kafka_2.13-2.7.0

ACT_SERVER_IP=localhost
PAS_SERVER_IP=localhost
ACT_SERVER_PORT=8090
PAS_SERVER_PORT=8091


if [[ ! -d $KAFKA_DIR ]]; then
  echo "please set \$KAFKA_DIR var"
  exit 1
fi

SOURCE="$0"
while [ -h "$SOURCE"  ]; do
    DIR="$( cd -P "$( dirname "$SOURCE"  )" && pwd  )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /*  ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE"  )" && pwd  )"

PRJ_DIR="$DIR/.."

$(pgrep "python" | xargs kill -9)

pushd $KAFKA_DIR
  ./bin/zookeeper-server-start.sh config/zookeeper.properties > run1.log 2>&1 & 
  ./bin/kafka-server-start.sh config/server.properties > run2.log 2>&1 & 
popd

python $PRJ_DIR/setup.py --host=$ACT_SERVER_IP --port=$ACT_SERVER_PORT --debug=True --syncer_server 127.0.0.1:9092 --syncer_type light_kafka --party_name 142 --reset_task_queue > run1.log 2>&1 &
python $PRJ_DIR/setup.py --host=$PAS_SERVER_IP --port=$PAS_SERVER_PORT --debug=True --syncer_server 127.0.0.1:9092 --syncer_type light_kafka --party_name 154 > run2.log 2>&1 &