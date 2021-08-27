import sys
import logging
from time import sleep
from flask import Flask, render_template, abort
from flask_json import FlaskJSON
from jinja2 import TemplateNotFound


class MyFlask(Flask):
    byte_task_coordinator = None


web_console = MyFlask(__name__, static_folder="templates", static_url_path="")
json = FlaskJSON(web_console)

web_console.logger.addHandler(logging.StreamHandler(sys.stdout))
web_console.logger.setLevel(logging.DEBUG)


@web_console.route('/', methods=['GET'])
def entry():
    try:
        return render_template('index.html')
    except TemplateNotFound:
        abort(404)
