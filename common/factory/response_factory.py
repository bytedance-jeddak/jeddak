from common.factory.factory import Factory


class ResponseFactory(Factory):
    @staticmethod
    def success():
        return dict(message='success'), 200

    @staticmethod
    def success_with_data(data):
        return dict(message='success', data=data), 200

    @staticmethod
    def fail_with_message(message):
        return dict(message=message), 400

    @staticmethod
    def unauthorized():
        return dict(message='unauthorized'), 401

    @staticmethod
    def key_error():
        return dict(message='invalid key'), 400

    @staticmethod
    def value_error():
        return dict(message='invalid value'), 400

    @staticmethod
    def type_error():
        return dict(message='invalid type, please check parameter list'), 400

    @staticmethod
    def internal_server_error():
        return dict(message="Internal Server Error"), 500
