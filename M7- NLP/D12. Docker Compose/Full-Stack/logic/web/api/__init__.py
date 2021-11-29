from enum import unique
from posix import EX_TEMPFAIL
from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import query

app = Flask(__name__)
api = Api(app)
app.config.from_object("api.config.Config")
db = SQLAlchemy(app)




class Striver(db.Model):

    __tablename__ = 'strivers'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    email= db.Column(db.String(64), unique=True, nullable=False)
    name= db.Column(db.String(64), unique=False, nullable=True)

    def __init__(self,email, name=""):
        
        self.email = email
        self.name = name

class Striver_api(Resource):

    def get(self):

        args_parser = reqparse.RequestParser()
        args_parser.add_argument('email', type = str)

        args = args_parser.parse_args()
        email_ = args['email']

        try:

            striver_info = db.session.query(Striver).filter_by(email=email_).first()
            return {'Name': striver_info.name, "Email": striver_info.email}

        except:
            return {'ERROR': "Couldn't find email"}


    def post(self):

        args_parser = reqparse.RequestParser()
        args_parser.add_argument('email', type = str)
        args_parser.add_argument('name', type = str)

        args = args_parser.parse_args()
        email_ = args['email']
        name_ = args['name']
        try:
            db.session.add( Striver(email=email_,name=name_) )
            db.session.commit()
            return {'email': email_, 'name': name_}
        except:
            return {'ERROR': "Couldn't insert email"}

api.add_resource(Striver_api, '/striver')