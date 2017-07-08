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



#form
class ArticleForm(Form):
    title = StringField()
    tag = StringField()
    #content = TextAreaField()
    content = PageDownField()
    submit = SubmitField('提交')


class LoginForm(Form):
    username = StringField("Name")
    password = PasswordField("Password")
    submit = SubmitField('提交')



