#coding:utf-8
import sys
import time
from flask import Flask, url_for, redirect, flash, request
from config import DevConfig
from flask.ext.sqlalchemy import SQLAlchemy
from sqlalchemy import func
from flask import Flask, render_template
from flask.ext.bootstrap import Bootstrap
from flask.ext.wtf import Form
from flask.ext.pagedown import PageDown
from wtforms import StringField, SubmitField, TextAreaField, PasswordField, BooleanField
from flask.ext.pagedown.fields import PageDownField
from markdown import markdown
from markdown import Markdown
import bleach
import json

from flask.ext.login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user

db = SQLAlchemy()
bootstrap = Bootstrap()
pagedown = PageDown()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config.from_object(DevConfig)
    db.init_app(app)
    bootstrap.init_app(app)
    pagedown.init_app(app)
    login_manager.init_app(app)
    login_manager.session_protection = 'strong'
    login_manager.login_view = 'login'

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app

