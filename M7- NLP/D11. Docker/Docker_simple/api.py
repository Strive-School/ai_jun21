from flask_restful import Resource, Api, reqparse
from flask import Flask
import db
app1 = Flask(__name__)
api = Api(app1)


class Student(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('name', type=str)
        arg = parser.parse_args()
        name = arg['name']
        print(name)
        return db.get_student(name)

class Students(Resource):
    def get(self):
        return db.get_students()


api.add_resource(Student, '/student')

api.add_resource(Students, '/students')

if __name__ == '__main__':
   app1.run(host='0.0.0.0')