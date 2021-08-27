import os
import traceback
import linecache
from base64 import b64encode

from flask import request, make_response, send_from_directory
from flask.helpers import safe_join
from flask_json import as_json
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename

from common.factory.response_factory import ResponseFactory
from common.util import constant
from fl.serving.model_loader import ModelLoader
from common.factory.serving_factory import ServingFactory
from coordinator.core.online_cache import OnlineCache
from common.util.random import Random
from coordinator.core.task_coordinator import TaskCoordinator
from coordinator.core.sqlite_manager import SqliteManager, STR2SHA256
from coordinator.web.console import web_console


# run coordinator when initializing the web_console, instead of using it as a route
def coordinator_run(syncer_server, syncer_type, party_name, db_url=None):
    try:
        SqliteManager.init(party_name, db_url=db_url)
        web_console.byte_task_coordinator = TaskCoordinator(syncer_server, syncer_type, party_name)
        web_console.byte_task_coordinator.start()
    except Exception as e:
        web_console.logger.info(e)
        raise Exception('error in running coordinator')


# reset_task_queue when initializing the web_console, instead of using it as a route
def reset_task_queue(syncer_server, syncer_type, party_name):
    try:
        TaskCoordinator.reset_task_queue(syncer_server, syncer_type, party_name)
    except Exception as e:
        web_console.logger.info(e)
        raise Exception('error in resetting task_queue')


@web_console.route('/task_chain/submit', methods=['POST'])
@as_json
def task_chain_submit():
    data = request.get_json(force=True)
    try:
        task_id = TaskCoordinator.generate_task_chain_id()
        parameter_chain = TaskCoordinator.generate(data)
        TaskCoordinator.submit(data, task_id, parameter_chain)
        return ResponseFactory.success_with_data({"task_id": task_id})

    except KeyError as e:
        web_console.logger.info(e)
        return ResponseFactory.key_error()

    except ValueError as e:
        web_console.logger.info(e)
        return ResponseFactory.value_error()

    except TypeError:
        return ResponseFactory.type_error()

    except Exception as e:
        traceback.print_exc()
        return ResponseFactory.internal_server_error()


# routers for board apis
@web_console.before_request
def before_request():
    if request.path == '/' or request.path.startswith('/img') \
            or request.path.startswith('/js') or request.path.startswith('/css') or request.path.startswith('/img')\
            or request.path.startswith('/fonts') or request.path.startswith('/fav'):
        return None

    # bypass the following requests only in the development environment
    if request.path.startswith('/task_chain/submit') \
            or request.path.startswith('/api/model_list')\
            or request.path.startswith('/api/model_load')\
            or request.path.startswith('/api/model_unload')\
            or request.path.startswith('/board/predict'):
        return None

    # add access control
    access_control_dict = {'/task_chain/submit': 16,
                           '/board/user': 4,
                           '/board/task': 16,
                           '/board/tasks': 128,
                           '/board/task_progress': 16,
                           '/board/model': 16,
                           '/board/dataset': 128,
                           '/board/datasets': 128,
                           '/board/models': 128,
                           '/board/party': 128,
                           '/board/predict': 128}
    # pass the access control in login and get salt
    if request.path.startswith('/board/user'):
        if request.get_json(force=True).get('req_type') == 'get_salt' \
                or request.get_json(force=True).get('req_type') == 'login'\
                or request.get_json(force=True).get('req_type') == 'get_change_password_salt'\
                or request.get_json(force=True).get('req_type') == 'get_add_user_salt':
            return None
    tok = request.headers.get('auth')
    if tok is None:
        return ResponseFactory.unauthorized()
    user_info = SqliteManager.user_dbo.get(args={
        'filter': {
            'token': tok,
            'deleted': False,
        }
    })
    if request.path.startswith("/board/user/admin_auth"):
        if user_info.get('role') < 4:
            return ResponseFactory.success_with_data({'role': user_info.get('role')})
        return ResponseFactory.unauthorized()
    path_split = request.path.split('?')[0].split('/')
    auth_role = int(access_control_dict['/' + path_split[1] + '/' + path_split[2]])
    # pass the check if the permission requirements are met
    if user_info is not None:
        if (user_info['role'] is not None) and (user_info['role'] < auth_role):
            return None

    return ResponseFactory.unauthorized()


class UserResource(Resource):
    @as_json
    def post(self):
        data = request.get_json(force=True)
        try:
            req_type = data.get('req_type')

            # if req_type=login, check user password with db
            # check in user login & return token and is_admin
            if req_type == 'login':
                if data.get('filter', {}).get('name') is None \
                        or data.get('filter', {}).get('password') is None:
                    return ResponseFactory.fail_with_message('invalid parameter')
                data['filter']['password'] = STR2SHA256(data['filter']['password'])
                user_info = SqliteManager.user_dbo.get(args=data)
                is_admin = False
                if user_info.get('role') < 4:
                    is_admin = True
                return ResponseFactory.success_with_data({'token': user_info.get('token'), 'isAdmin': is_admin}) \
                    if user_info is not None \
                    else ResponseFactory.fail_with_message('login fails')

            # get user list and show the users can be managed
            if req_type == 'get_user_list':
                tok = request.headers.get('auth')
                if tok is None:
                    return ResponseFactory.unauthorized()
                admin_info = SqliteManager.user_dbo.get(args={'filter': {'token': tok}})
                total_count, all_users = SqliteManager.user_dbo.list(data)
                users = []
                for au in all_users:
                    if au['role'] > admin_info.get('role'):
                        u = {'name': au['name'], 'role': au['role']}
                        users.append(u)
                    else:
                        total_count = total_count - 1
                return ResponseFactory.success_with_data({
                    'total_count': total_count,
                    'users': users
                })

            # get salt to compute the password before login
            if req_type == 'get_salt':
                if data.get('filter', {}).get('name') is None:
                    return ResponseFactory.fail_with_message('invalid parameter')
                user_info = SqliteManager.user_dbo.get(args=data)
                if user_info is not None:
                    return ResponseFactory.success_with_data({'salt': user_info.get('salt')})
                random_salt = b64encode(os.urandom(32)).decode('utf-8')[:-1]
                return ResponseFactory.success_with_data({'salt': random_salt})

            # get salt before change password
            if req_type == 'get_change_password_salt':
                if data.get('filter', {}).get('name') is None:
                    return ResponseFactory.fail_with_message('invalid parameter')
                user_info = SqliteManager.user_dbo.get(args={'filter': {'name': data['filter']['name']}})
                random_salt = b64encode(os.urandom(32)).decode('utf-8')[:-1]
                if user_info is not None:
                    return ResponseFactory.success_with_data({'salt': user_info.get('salt'), 'new_salt': random_salt})
                random_salt2 = b64encode(os.urandom(32)).decode('utf-8')[:-1]
                return ResponseFactory.success_with_data({'salt': random_salt, 'new_salt': random_salt2})

            # get salt before add user
            if req_type == 'get_add_user_salt':
                if data.get('filter', {}).get('name') is None \
                        or data.get('filter', {}).get('role') is None:
                    return ResponseFactory.fail_with_message('invalid parameter')
                tok = request.headers.get('auth')
                role = int(data['filter']['role'])
                user_info = SqliteManager.user_dbo.get(args={'filter': {'token': tok}})
                user_role = user_info.get('role')
                if not ((user_role < 2) or (user_role < 4 and role > 3)):
                    return ResponseFactory.unauthorized()
                random_salt = b64encode(os.urandom(32)).decode('utf-8')[:-1]
                add_user_info = SqliteManager.user_dbo.get(args={'filter': {'name': data['filter']['name']}})
                if add_user_info is None:
                    add_salt = {'name': data['filter']['name'], 'salt': random_salt, 'deleted': True}
                    SqliteManager.user_dbo.create(args=add_salt)
                    return ResponseFactory.success_with_data({'salt': random_salt})
                add_user_info = SqliteManager.user_dbo.get(
                    args={'filter': {'name': data['filter']['name'], 'deleted': True}})
                if add_user_info is not None:
                    update_salt = {'filter': {'name': data['filter']['name'], 'deleted': True},
                                   'field': {'salt': random_salt}}
                    ok = SqliteManager.user_dbo.update(args=update_salt)
                    return ResponseFactory.success_with_data({'salt': random_salt}) if ok \
                        else ResponseFactory.fail_with_message('err in add user')
                return ResponseFactory.fail_with_message('user name is exists')

            # add user in db
            if req_type == 'add_user':
                if data.get('filter', {}).get('name') is None \
                        or data.get('filter', {}).get('role') is None\
                        or data.get('filter', {}).get('password') is None\
                        or data.get('filter', {}).get('salt') is None:
                    return ResponseFactory.fail_with_message('invalid parameter')
                role = int(data['filter']['role'])
                tok = request.headers.get('auth')
                user_info = SqliteManager.user_dbo.get(args={'filter': {'token': tok}})
                user_role = user_info.get('role')
                if not ((user_role < 2) or (user_role < 4 and role > 3)):
                    return ResponseFactory.unauthorized()
                pass_hash = STR2SHA256(data['filter']['password'])
                plain_token = "{}.{}.{}.{}".format(
                    data['filter']['name'],
                    data['filter']['salt'],
                    pass_hash,
                    Random.generate_random_digits(length=10),
                    int(data['filter']['role'])
                )
                update_salt = {'filter': {
                    'name': data['filter']['name'],
                    'salt': data['filter']['salt'],
                    'deleted': True
                }, 'field': {
                    'password': pass_hash,
                    'role': int(data['filter']['role']),
                    'token': STR2SHA256(plain_token),
                    'deleted': False
                }}
                ok = SqliteManager.user_dbo.update(args=update_salt)
                return ResponseFactory.success_with_data('add success') if ok \
                    else ResponseFactory.fail_with_message('err in add user')

            # delete user in db
            if req_type == 'delete_user':
                if data.get('filter', {}).get('name') is None:
                    return ResponseFactory.fail_with_message('invalid parameter')
                tok = request.headers.get('auth')
                user_info = SqliteManager.user_dbo.get(args={'filter': {'name': data['filter']['name']}})
                admin_info = SqliteManager.user_dbo.get(args={'filter': {'token': tok}})
                admin_role = admin_info.get('role')
                user_role = user_info.get('role')
                if not ((admin_role < 2) or (admin_role < 4 and user_role > 3)):
                    return ResponseFactory.unauthorized()
                ok = SqliteManager.user_dbo.update(args={
                    'filter': {'name': data['filter']['name']},
                    'field': {'deleted': True}})
                return ResponseFactory.success_with_data('delete succeed') \
                    if ok else ResponseFactory.fail_with_message("delete fails")

            # default: return error message
            return ResponseFactory.fail_with_message('unknown request type')

        except Exception as e:
            traceback.print_exc()
            return ResponseFactory.internal_server_error()

    @as_json
    def patch(self):
        data = request.get_json(force=True)
        try:
            req_type = data.get('req_type')

            # if req_type=update_password, check old password and update in db
            # update password in db
            if req_type == 'update_password':
                if data.get('filter', {}).get('name') is None \
                        or data.get('filter', {}).get('password') is None \
                        or data.get('field', {}).get('password') is None\
                        or data.get('field', {}).get('salt') is None:
                    return ResponseFactory.fail_with_message('invalid parameter')
                data['filter']['password'] = STR2SHA256(data['filter']['password'])
                data['field']['password'] = STR2SHA256(data['field']['password'])
                plain_token = "{}.{}.{}.{}".format(
                    data['filter']['name'],
                    data['field']['password'],
                    Random.generate_random_digits(length=10),
                    data['field']['salt']
                )
                data['field']['token'] = STR2SHA256(plain_token)
                ok = SqliteManager.user_dbo.update(args=data)
                return ResponseFactory.success_with_data({'auth': data['field']['token']}) \
                    if ok else ResponseFactory.fail_with_message("fails")

            # admin can update user password and role in db
            if req_type == 'update_user':
                if data.get('filter', {}).get('name') is None:
                    return ResponseFactory.fail_with_message('invalid parameter')
                if (data.get('field', {}).get('password') is None) ^ (data.get('field', {}).get('salt') is None):
                    return ResponseFactory.fail_with_message('invalid parameter')
                tok = request.headers.get('auth')
                user_info = SqliteManager.user_dbo.get(args={'filter': {'name': data['filter']['name']}})
                admin_info = SqliteManager.user_dbo.get(args={'filter': {'token': tok}})
                admin_role = admin_info.get('role')
                user_role = user_info.get('role')
                if not ((admin_role < 2) or (admin_role < 4 and user_role > 3)):
                    return ResponseFactory.unauthorized()
                pw = Random.generate_random_digits(length=10)
                sa = Random.generate_random_digits(length=10)
                ro = user_role
                if data.get('field', {}).get('password') is None:
                    data['field']['password'] = user_info.get('password')
                    data['field']['salt'] = user_info.get('salt')
                else:
                    data['field']['password'] = STR2SHA256(data['field']['password'])
                    pw = data['field']['password']
                    sa = data['field']['salt']
                if data.get('field', {}).get('role') is None:
                    data['field']['role'] = user_role
                else:
                    ro = data['field']['role']
                plain_token = "{}.{}.{}.{}".format(data['filter']['name'], pw, sa, ro,
                                                   Random.generate_random_digits(length=10))
                data['field']['token'] = STR2SHA256(plain_token)
                ok = SqliteManager.user_dbo.update(args=data)
                return ResponseFactory.success_with_data('update user success') \
                    if ok else ResponseFactory.fail_with_message("fails in update user")

            # default: return error message
            return ResponseFactory.fail_with_message('unknown request type')

        except Exception as e:
            traceback.print_exc()
            return ResponseFactory.internal_server_error()


class PartyResource(Resource):
    def get(self):
        """
        get: return party information from coordinator (in-memory)
        """
        return ResponseFactory.success_with_data({
            "party_name": web_console.byte_task_coordinator.party_name,
            "syncer_type": web_console.byte_task_coordinator.syncer_type,
        })


class TaskResource(Resource):
    """
    resource for task
    """
    def get(self, task_id):
        try:
            get_type = request.args.get("type")

            # if get_type=log, read the task log from file system
            if get_type == 'log':
                file_name = "{}.log".format(task_id)
                file_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..', 'common', 'log',
                                         web_console.byte_task_coordinator.party_name)
                response = make_response(send_from_directory(file_path, file_name, as_attachment=True, cache_timeout=0))
                response.headers["Content-Disposition"] = "attachment; filename={}".format(file_name)
                return response

            # if get_type=log, read the data_saver output from file system
            elif get_type == 'output':
                file_name = "{}.csv".format(task_id)
                file_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..', 'common', 'data_saver',
                                         web_console.byte_task_coordinator.party_name)
                response = make_response(
                    send_from_directory(file_path, file_name, as_attachment=True, cache_timeout=0))
                response.headers["Content-Disposition"] = "attachment; filename={}".format(file_name)
                return response

            # if get_type=dlog, return dynamic logs based on request parameters
            elif get_type == 'dlog':
                start_offset = int(request.args.get('start_offset', 0))
                task_detail = SqliteManager.task_dbo.get(
                    unique_id=task_id,
                    args={'filter': {'deleted': False}}
                )
                if task_detail is None:
                    return ResponseFactory.fail_with_message("task_id not found")
                file_dir = os.path.join(
                    os.path.split(os.path.realpath(__file__))[0],
                    '../..',
                    'common',
                    'log',
                    web_console.byte_task_coordinator.party_name
                )
                file_name = "{}.log".format(task_id)
                file_path = safe_join(file_dir, file_name)
                linecache.updatecache(file_path)
                lines = linecache.getlines(file_path)
                log_list = lines[start_offset:]
                return ResponseFactory.success_with_data(dict(
                    log_list=log_list,
                    task_status=task_detail["status"]
                ))

            # default: get task detail from db
            task_detail = SqliteManager.task_dbo.get(
                unique_id=task_id,
                args={'filter': {'deleted': False}}
            )
            if task_detail is None:
                return ResponseFactory.fail_with_message("task_id not found")
            return ResponseFactory.success_with_data(task_detail)

        except Exception as e:
            traceback.print_exc()
            return ResponseFactory.internal_server_error()

    @as_json
    def patch(self, task_id):
        data = request.get_json(force=True)
        try:
            patch_type = data.get('type')

            # if patch_type=stop, stop the task with coordinator
            if patch_type == 'stop':
                syncer_type = web_console.byte_task_coordinator.syncer_type
                if syncer_type == constant.MessengerType.LIGHT_KAFKA:
                    return ResponseFactory.success_with_data({
                        'message': 'task termination is not supported'
                    })

            # default: update detail in db
            ok = SqliteManager.task_dbo.update(unique_id=task_id, args=data)
            return ResponseFactory.success() if ok else ResponseFactory.fail_with_message("task_id not found")

        except Exception as e:
            traceback.print_exc()
            return ResponseFactory.internal_server_error()

    def delete(self, task_id):
        """
        delete task detail in db
        """
        try:
            ok = SqliteManager.task_dbo.delete(unique_id=task_id)
            return ResponseFactory.success() if ok else ResponseFactory.fail_with_message('fails')

        except Exception as e:
            traceback.print_exc()
            return ResponseFactory.internal_server_error()


class TaskCollectionResource(Resource):
    def get(self):
        try:
            get_type = request.args.get("req_type")
            total_count, tasks = SqliteManager.task_dbo.list(request.args)

            # if get_type=stat, return brief statistics
            if get_type == 'stat':
                result = dict(task_num=0, ready=0, run=0, error=0, finish=0)
                for t in tasks:
                    result['task_num'] += 1
                    result[t['status']] = result.get(t['status'], 0) + 1
                return ResponseFactory.success_with_data(result)

            # if get_type=info, return task collection detail
            elif get_type == 'info':
                return ResponseFactory.success_with_data({
                    'total_count': total_count,
                    'tasks': tasks,
                })

            # default: return error message
            return ResponseFactory.fail_with_message("unknown request type")

        except Exception as e:
            traceback.print_exc()
            return ResponseFactory.internal_server_error()


class TaskProgressResource(Resource):
    """
    resource for task_progress
    """
    def get(self, task_id):
        """
        get task progress detail from db
        """
        try:
            # get the progress_list
            _, progress_list = SqliteManager.task_progress_dbo.list(args=dict(
                filter=dict(
                    task_id=task_id
                ),
            ))

            total_count = 0
            losses = []
            times = []
            train_auc = []
            validate_auc = []
            metrics = None
            error_msg = None
            for p in progress_list:
                if p['progress_type'] == constant.BoardConstants.TASK_PROGRESS_MODULE:
                    total_count += 1
                elif p['progress_type'] == constant.BoardConstants.TASK_PROGRESS_LOSS:
                    losses.append(p['progress_value']['loss'])
                    times.append(p['progress_value']['time'])
                    if not p['progress_value'].get('train_auc') is None:
                        train_auc.append(p['progress_value']['train_auc'])
                    if not p['progress_value'].get('validate_auc') is None:
                        validate_auc.append(p['progress_value']['validate_auc'])
                elif p['progress_type'] == constant.BoardConstants.TASK_PROGRESS_METRICS:
                    metrics = p['progress_value']
                elif p['progress_type'] == constant.BoardConstants.TASK_PROGRESS_ERROR_MSG:
                    error_msg = p['progress_value']

            return ResponseFactory.success_with_data(dict(
                progress=total_count,
                losses=losses,
                times=times,
                validate_auc=validate_auc,
                train_auc=train_auc,
                metrics=metrics,
                error_msg=error_msg
            ))

        except Exception as e:
            traceback.print_exc()
            return ResponseFactory.internal_server_error()


class ModelResource(Resource):
    """
    resource for model
    """
    @as_json
    def patch(self, model_id):
        """
        update model detail in db
        """
        data = request.get_json(force=True)
        try:
            ok = SqliteManager.model_dbo.update(unique_id=model_id, args=data)
            return ResponseFactory.success() if ok else ResponseFactory.fail_with_message("model_id not found")

        except Exception as e:
            traceback.print_exc()
            return ResponseFactory.internal_server_error()

    def delete(self, model_id):
        """
        delete model detail in db
        """
        try:
            ok = SqliteManager.model_dbo.delete(unique_id=model_id)
            return ResponseFactory.success() if ok else ResponseFactory.fail_with_message("model_id not found")

        except Exception as e:
            traceback.print_exc()
            return ResponseFactory.internal_server_error()


class ModelCollectionResource(Resource):
    """
    resource for model list
    """
    def get(self):
        """
        return model collection detail
        """
        try:
            total_count, models = SqliteManager.model_dbo.list(args=request.args)
            return ResponseFactory.success_with_data({
                'total_count': total_count,
                'models': models
            })

        except Exception as e:
            traceback.print_exc()
            return ResponseFactory.internal_server_error()


class DatasetResource(Resource):
    """
    resource for model list
    """
    def post(self):
        # save uploaded csv file
        f = request.files.get('file')
        if f is None:
            return ResponseFactory.fail_with_message('csv file not found')

        # check suffix
        file_name = secure_filename(f.filename)
        file_name_split = file_name.rsplit('.', 1)
        if len(file_name_split) < 2 or file_name_split[1] != 'csv':
            return ResponseFactory.fail_with_message('only csv supported')

        # saved on server
        root_dir = os.path.split(os.path.realpath(__file__))[0]
        local_data_dir = os.path.join(
            root_dir,
            '../../common/dataset',
            web_console.byte_task_coordinator.party_name
        )
        if not os.path.exists(local_data_dir):
            os.makedirs(local_data_dir)
        new_file_path = os.path.join(
            local_data_dir,
            "{}.{}.csv".format(file_name_split[0], Random.generate_random_digits())
        )
        f.save(new_file_path)
        return ResponseFactory.success_with_data({'file_path': os.path.abspath(new_file_path)})


class DatasetCollectionResource(Resource):
    @staticmethod
    def get_local_file_list(local_dir, file_type):
        """
        return: List<file_name, file_path>
        """
        file_path_list = []
        local_dir_abs = os.path.abspath(local_dir)
        for cur_dir, _, files in os.walk(local_dir_abs):
            for file in files:
                if file.endswith(".csv") or file.endswith(".data"):
                    file_abs_path = os.path.join(local_dir_abs, cur_dir, file)
                    file_rel_path = os.path.relpath(file_abs_path, local_dir_abs)
                    file_path_list.append({
                        'file_name': file,
                        'file_rel_path': file_rel_path,
                        'file_abs_path': file_abs_path,
                        'file_type': file_type
                    })
        return file_path_list

    def get(self):
        root_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..')
        data = DatasetCollectionResource.get_local_file_list(os.path.join(
                root_dir,
                'common/dataset',
                web_console.byte_task_coordinator.party_name
            ), 'user-uploaded dataset') + DatasetCollectionResource.get_local_file_list(os.path.join(
                root_dir,
                'dataset'
            ), 'common dataset') + DatasetCollectionResource.get_local_file_list(os.path.join(
                root_dir,
                'example/data'
            ), 'mini dataset')
        return ResponseFactory.success_with_data(data)


class ModelPredictResource(Resource):
    """
        resource for model online predict
    """
    @as_json
    def post(self):
        data = request.get_json(force=True)
        try:
            req_type = data.get('req_type')
            # if req_type=model_load, load the model
            if req_type == 'model_load':
                req_data = data.get('filter')
                if len(req_data) != 2 or req_data[0].get('task_type') != 'task_chain' \
                        or req_data[1].get('task_type') != 'model_loader':
                    return ResponseFactory.key_error()
                task_id = TaskCoordinator.generate_task_chain_id()
                parameter_chain = TaskCoordinator.generate(req_data)
                TaskCoordinator.submit(req_data, task_id, parameter_chain)
                OnlineCache.add_cache(req_data[1]['model_id'][0], req_data[0], req_data[1]['task_role'])
                return ResponseFactory.success()

            if req_type == 'model_unload':
                req_data = data.get('filter')
                model_id = req_data.get("model_id")
                del ModelLoader.models[model_id]
                del OnlineCache.model_cache[model_id]
                return ResponseFactory.success()

            if req_type == 'model_list':
                keys = ModelLoader.models.keys()
                return ResponseFactory.success_with_data(keys)

            if req_type == 'predict':
                req_data = data.get('filter')
                model_id = req_data.get("model_id")
                task_chain, task_role = OnlineCache.get_cache(model_id)
                if task_chain is None:
                    return ResponseFactory.fail_with_message("model_id is not ready. load it first")
                online_predict_task = [task_chain, {
                    'task_type': 'predict_online',
                    'task_role': task_role,
                    'model_id': [model_id] * len(task_role),
                    'input_data': req_data['input_data']
                }]
                task_id = TaskCoordinator.generate_task_chain_id()
                parameter_chain = TaskCoordinator.generate(online_predict_task)
                TaskCoordinator.submit(online_predict_task, task_id, parameter_chain)
                predict_data = ServingFactory.get_labels(task_chain_id=task_id)
                if predict_data is None:
                    return ResponseFactory.fail_with_message("timeout")
                return ResponseFactory.success_with_data({"predict_data": predict_data})

        except Exception as e:
            traceback.print_exc()
            return ResponseFactory.internal_server_error()


api = Api(web_console)
api.add_resource(UserResource, '/board/user')
api.add_resource(PartyResource, '/board/party')
api.add_resource(TaskResource, '/board/task/<string:task_id>')
api.add_resource(TaskCollectionResource, '/board/tasks')
api.add_resource(TaskProgressResource, '/board/task_progress/<string:task_id>')
api.add_resource(ModelResource, '/board/model/<string:model_id>')
api.add_resource(ModelCollectionResource, '/board/models')
api.add_resource(DatasetResource, '/board/dataset')
api.add_resource(DatasetCollectionResource, '/board/datasets')
api.add_resource(ModelPredictResource, '/board/predict')
