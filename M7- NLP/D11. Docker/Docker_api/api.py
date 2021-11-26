from flask import Flask
from flask_restful import Api, Resource, reqparse
import db

app1 = Flask(__name__)
api = Api(app1)

class Students(Resource):

    def get(self):

        return db.get_students()

class Name(Resource):

    def get(self):

        parser = reqparse.RequestParser()
        parser.add_argument("email", type=str)

        args = parser.parse_args()
        email = args['email']

        return db.get_name(email)


api.add_resource(Students, "/students")
api.add_resource(Name, "/name")

if __name__ == "__main__":
    app1.run('0.0.0.0')