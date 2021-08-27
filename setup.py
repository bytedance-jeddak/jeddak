import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Jeddak Web Console")

    if len(sys.argv) > 2:
        # setup via command line
        parser.add_argument('--host', type=str, default='localhost', help='host to run Jeddak')
        parser.add_argument('--port', type=int, default=5000, help='port to run Jeddak')
        parser.add_argument('--debug', type=bool, default=False, help='debug mode of flask server')
        parser.add_argument('--threaded', type=bool, default=False, help='threaded mode of flask server')
        parser.add_argument('--syncer_server', type=str, default=None, required=True, help='address of syncer')
        parser.add_argument('--syncer_type', type=str, default=None, required=True, help='type of syncer, redis/light_kafka')
        parser.add_argument('--party_name', type=str, default=None, required=True, help='name of this party')
        parser.add_argument('--reset_task_queue', action='store_true', help='whether to reset the task queue unfinished')
        parser.add_argument('--db_url', type=str, default=None, help='like user:password@ip:port/db_name')

        args = parser.parse_args()

    if len(sys.argv) == 2:
        # setup via config, such as config_demo.yaml
        # check the existence of configuration file
        if not os.path.exists(sys.argv[1]):
            print(" config is not exists!!!")
            exit()
        else:
            # load configuration
            file_path = str(sys.argv[1])
            with open(file_path, 'r') as f:
                args_dict = yaml.load(f.read())
                args = argparse.Namespace(**args_dict)

    from coordinator.web.console_route import web_console, coordinator_run, reset_task_queue
    if args.reset_task_queue:
        reset_task_queue(args.syncer_server, args.syncer_type, args.party_name)
    coordinator_run(args.syncer_server, args.syncer_type, args.party_name, db_url=args.db_url)

    web_console.run(host=args.host,
                    port=args.port,
                    debug=args.debug,
                    threaded=args.threaded,
                    use_reloader=False)
