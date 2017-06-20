class Config():
    pass

class ProdConfig():
    pass

class DevConfig():
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:root@localhost/flask"
    SECRET_KEY = "hard"


