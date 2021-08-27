import os
import json
import hashlib
from base64 import b64encode

from common.util.constant import BoardConstants
from common.util.random import Random
from common.frame.message_frame.redis_message import RedisMessage

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, Text, DateTime, Boolean, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tenacity import retry, stop_after_attempt

Base = declarative_base()


def STR2SHA256(x): return hashlib.sha256(x.encode('utf-8')).hexdigest()
def retry_error_callback(retry_state): print("retry fails. calling retry_error_callback")


class DBO(object):
    """
    base class of database operator
    """
    def __init__(self, engine):
        self._engine = engine
        self._session = sessionmaker(bind=self._engine)
        self._base = None
        self._unique_field = None

    def list(self, args=None):
        if args is None:
            args = dict()
        session = self._session()
        data = session.query(self._base)
        data = DBO.filter(data, args, self._base)
        total_count = data.count()
        data = DBO.order_by(data, args, self._base)
        data = DBO.to_page(data, args)
        return total_count, [d.to_dict() for d in data]

    def get(self, unique_id=None, args=None):
        if args is None:
            args = dict()
        session = self._session()
        data = session.query(self._base)
        data = DBO.filter_unique(data, self._unique_field, unique_id, self._base)
        data = DBO.filter(data, args, self._base)
        data = DBO.order_by(data, args, self._base)
        data = DBO.order_by(data, args, self._base)
        d = data.first()
        return d.to_dict() if d is not None else None

    @retry(stop=stop_after_attempt(3), retry_error_callback=retry_error_callback)
    def create(self, args=None):
        if args is None:
            args = dict()
        session = self._session()
        data = self._base()
        for k in args:
            if hasattr(data, k):
                setattr(data, k, args.get(k))
        session.add(data)
        session.commit()

    @retry(stop=stop_after_attempt(3), retry_error_callback=retry_error_callback)
    def update(self, unique_id=None, args=None):
        """
        return True if update succeeds
        return False if record not found
        """
        if args is None:
            args = dict()
        session = self._session()
        data = session.query(self._base)
        data = DBO.filter_unique(data, self._unique_field, unique_id, self._base)
        data = DBO.filter(data, args, self._base)
        if data.first() is None:
            return False
        update_dict = DBO.parse_update_dict(args, self._base)
        if update_dict:
            data.update(update_dict)
        session.commit()
        return True

    @retry(stop=stop_after_attempt(3), retry_error_callback=retry_error_callback)
    def delete(self, unique_id=None):
        return self.update(
            unique_id=unique_id,
            args={
                'field': {
                    'deleted': True,
                }
            }
        )

    @staticmethod
    def filter(data, args, base):
        if 'filter' in args:
            filter_dict = args.get('filter', '{}')
            if type(filter_dict) == str:
                filter_dict = json.loads(filter_dict)
            for k in filter_dict:
                v = filter_dict[k]
                if hasattr(base, k):
                    data = data.filter(getattr(base, k) == v)
        return data

    @staticmethod
    def filter_unique(data, unique_field, unique_id, base):
        if unique_field is not None:
            data = data.filter(getattr(base, unique_field) == unique_id)
        return data

    @staticmethod
    def order_by(data, args, base):
        if 'order_by' in args:
            order_by_str = args.get('order_by')
            if order_by_str.startswith('-'):
                order_field = args.get('order_by')[1:]
                data = data.order_by(getattr(base, order_field).desc())
            else:
                data = data.order_by(getattr(base, args.get('order_by')))
        return data

    @staticmethod
    def to_page(data, args):
        if 'curr_page' in args and 'page_size' in args:
            curr_page_int = int(args.get('curr_page'))
            page_size_int = int(args.get('page_size'))
            data = data.offset((curr_page_int - 1) * page_size_int).limit(page_size_int)
        return data

    @staticmethod
    def parse_update_dict(args, base):
        update_dict = {}
        if 'field' in args:
            field_dict = args.get('field', '{}')
            if type(field_dict) == str:
                field_dict = json.loads(field_dict)
            for k in field_dict:
                v = field_dict[k]
                if hasattr(base, k):
                    update_dict[getattr(base, k)] = v
        return update_dict


class User(Base):
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text(), comment='用户名')
    password = Column(Text(), comment='用户密码')
    salt = Column(Text(), comment='盐值')
    token = Column(Text(), comment='用户登录token')
    role = Column(Integer, comment='角色')
    deleted = Column(Boolean(), default=False, comment='用户是否已删除')

    def to_dict(self):
        return dict(
            name=self.name,
            password=self.password,
            salt=self.salt,
            token=self.token,
            role=self.role,
            deleted=self.deleted
        )


class UserDbo(DBO):
    def __init__(self, engine):
        super(UserDbo, self).__init__(engine)
        self._base = User
        self._unique_field = None
        User.__table__.create(engine, checkfirst=True)
        self.init_admin()

    def init_admin(self):
        session = self._session()
        admin_user = session.query(User).filter(User.name == 'admin').first()
        if admin_user is None:
            # create admin user if not exist
            random_salt = b64encode(os.urandom(32)).decode('utf-8')[:-1]
            pass_salt = random_salt + BoardConstants.DEFAULT_PASS
            role_int = 1
            plain_token = "{}.{}.{}.{}".format(
                BoardConstants.DEFAULT_USER,
                pass_salt,
                Random.generate_random_digits(length=10),
                random_salt,
                role_int
            )
            ed_user = User(
                name=BoardConstants.DEFAULT_USER,
                password=STR2SHA256(STR2SHA256(pass_salt)),
                salt=random_salt,
                role=role_int,
                token=STR2SHA256(plain_token)
            )
            session.add(ed_user)
            session.commit()


class Task(Base):
    __tablename__ = 'task'

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Text(), comment='任务ID')
    party_name = Column(Text(), comment='发起方名称')
    party_names = Column(Text(), comment='参与方名称')
    status = Column(Text(), comment='任务状态')
    alias = Column(Text(), comment='任务备注名')
    task_chain = Column(Text(), comment='任务链JSON')
    deleted = Column(Boolean(), default=False, comment='任务是否已删除')
    create_time = Column(DateTime(timezone=True), server_default=func.now(), comment='创建时间')
    update_time = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment='修改时间')

    def to_dict(self):
        return dict(
            task_id=self.task_id,
            party_name=self.party_name,
            party_names=self.party_names,
            status=self.status,
            alias=self.alias,
            task_chain=self.task_chain,
            deleted=self.deleted,
            create_time=self.create_time.isoformat(),
            update_time=self.update_time.isoformat()
        )


class TaskDbo(DBO):
    def __init__(self, engine):
        super(TaskDbo, self).__init__(engine)
        self._base = Task
        self._unique_field = 'task_id'
        Task.__table__.create(engine, checkfirst=True)
        self.update_history()

    def update_history(self):
        session = self._session()
        session.query(Task).filter(
            Task.status != RedisMessage.FINISH,
            Task.status != RedisMessage.ERROR,
            Task.deleted == False
        ).update({
            Task.status: RedisMessage.ERROR
        })
        session.commit()


class Model(Base):
    __tablename__ = 'model'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Text(), comment='模型ID')
    alias = Column(Text(), comment='任务备注名')
    deleted = Column(Boolean(), default=False, comment='任务是否已删除')
    create_time = Column(DateTime(timezone=True), server_default=func.now(), comment='创建时间')

    def to_dict(self):
        return dict(
            model_id=self.model_id,
            alias=self.alias,
            deleted=self.deleted,
            create_time=self.create_time.isoformat()
        )


class ModelDbo(DBO):
    def __init__(self, engine):
        super(ModelDbo, self).__init__(engine)
        self._base = Model
        self._unique_field = 'model_id'
        Model.__table__.create(engine, checkfirst=True)


class TaskProgress(Base):
    __tablename__ = 'task_progress'

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Text(), comment='任务ID')
    progress_type = Column(Text(), comment='任务进度类型，取值module,loss,metrics,error')
    """
    progress_value: 
    - if progress_type=module, progress_value is the last finished module
    - if progress_type=loss, progress_value is `{'loss':xxx,'time':xxx}` 
                                            or `{'loss':xxx,'time':xxx, 'train_auc':xxx, 'validate_auc':xxx}`
    - if progress_type=metrics, progress_value is `{'xxx':xxx}`
    - if progress_type=error, progress_value is exception message in string format
    """
    progress_value = Column(Text(), comment='进度取值')
    create_time = Column(DateTime(timezone=True), server_default=func.now(), comment='创建时间')

    def to_dict(self):
        d = dict(
            task_id=self.task_id,
            progress_type=self.progress_type,
            progress_value=self.progress_value,
            create_time=self.create_time.isoformat()
        )
        if self.progress_type == BoardConstants.TASK_PROGRESS_LOSS or \
                self.progress_type == BoardConstants.TASK_PROGRESS_METRICS:
            d['progress_value'] = json.loads(self.progress_value)
        return d


class TaskProgressDbo(DBO):
    def __init__(self, engine):
        super(TaskProgressDbo, self).__init__(engine)
        self._base = TaskProgress
        self._unique_field = 'task_id'
        TaskProgress.__table__.create(engine, checkfirst=True)


class SqliteManager(object):
    party_name = None
    user_dbo = None
    task_dbo = None
    model_dbo = None
    task_progress_dbo = None

    @staticmethod
    def init(party_name, db_url=None):
        SqliteManager.party_name = party_name

        try:
            if db_url is not None:
                engine = SqliteManager.init_mysql_engine(db_url, party_name)
                SqliteManager.init_instance(engine)
            else:
                engine = SqliteManager.init_local_engine(party_name)
                SqliteManager.init_instance(engine)

        except Exception as e:
            print('[WARNING] Connection Refused by MySQL, now use local SQLite instead.')
            engine = SqliteManager.init_local_engine(party_name)

        SqliteManager.init_instance(engine)

    @staticmethod
    def init_instance(engine):
        SqliteManager.user_dbo = UserDbo(engine)
        SqliteManager.task_dbo = TaskDbo(engine)
        SqliteManager.model_dbo = ModelDbo(engine)
        SqliteManager.task_progress_dbo = TaskProgressDbo(engine)

    @staticmethod
    def init_local_engine(party_name, ):
        print('trying to connect to local SQLite...')
        root_dir = os.path.split(os.path.realpath(__file__))[0]
        db_dir = os.path.join(root_dir, '..', '..', 'common', 'sqlite', party_name)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        db_path = os.path.join(db_dir, 'jeddak.db')
        local_engine = create_engine('sqlite:///{}?check_same_thread=False'.format(db_path))
        return local_engine

    @staticmethod
    def init_mysql_engine(db_url, party_name):
        print('trying to connect to remote MySQL...')
        return create_engine(
            'mysql+pymysql://{}?charset=utf8'.format(db_url, party_name),
            connect_args={"init_command": "SET SESSION time_zone='+00:00'"}
        )
