FROM ubuntu:latest

RUN sed -i "s/http:\/\/archive.ubuntu.com/http:\/\/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list
RUN apt-get update
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install -y openjdk-8-jdk wget mariadb-server mariadb-client \
  libgmp-dev libmpc-dev libmpfr-dev git \
  build-essential zlib1g-dev libbz2-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev


# create work dir
RUN mkdir /home/compute
ENV url /home/compute
WORKDIR ${url}
# install python
RUN wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz && tar -xzvf Python-3.7.4.tgz && rm Python-3.7.4.tgz && cd Python-3.7.4 && \
  ./configure --prefix=/usr/local/python3 && \
  make && \
  make install && make clean && \
  rm -rf Python-3.7.4 && rm -f /usr/bin/python /usr/bin/pip /usr/local/bin/pip
RUN ln -s /usr/local/python3/bin/python3 /usr/bin/python && \
  ln -s /usr/local/python3/bin/pip3 /usr/bin/pip


RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple






# install spark

RUN wget https://mirrors.tuna.tsinghua.edu.cn/apache/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz  && \
  tar -xvf spark-3.1.2-bin-hadoop3.2.tgz && \
  rm spark-3.1.2-bin-hadoop3.2.tgz && \
  mv spark-3.1.2-bin-hadoop3.2 spark-home
ENV SPARK_HOME "/home/compute/spark-home"
ENV PYSPARK_PYTHON "/usr/bin/python"
# RUN echo "export PATH=\"/home/compute/spark-home/bin:$PATH\"" >> ~/.bash_rc
ENV PATH "${url}/spark-home/bin:$PATH"

# install node.js
RUN wget https://nodejs.org/dist/v14.16.1/node-v14.16.1-linux-x64.tar.xz && \
  tar -xvf node-v14.16.1-linux-x64.tar.xz && \
  rm node-v14.16.1-linux-x64.tar.xz && \
  mv node-v14.16.1-linux-x64 node-js

ENV PATH "${url}/node-js/bin:$PATH"


# start mysql
RUN service mysql restart



# clone the project
COPY byte_federated_learning ${url}/byte_federated_learning
RUN cd ./byte_federated_learning && \
  sed -i 's/npm i/npm i --unsafe-perm=true --allow-root/g' build.sh && \
  sed -i 's/python3/python/g' build.sh && \
  pip install -r requirements.txt && chmod +x build.sh && \
  sh -c '/bin/echo -e "yes" | ./build.sh'
expose 1234




ENV host "0.0.0.0"
ENV debug "False"
ENV threaded "False"
ENV other ""
ENV server "None"
ENV type "None"
ENV name "None"
CMD cd byte_federated_learning && python setup.py --host ${host} --port 1234 --debug ${debug} --threaded ${threaded} --syncer_server ${server} --syncer_type ${type} --party_name ${name} ${other}
