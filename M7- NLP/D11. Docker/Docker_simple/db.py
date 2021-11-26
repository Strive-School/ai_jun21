from sqlalchemy import create_engine
import pandas as pd



def create_db():
    try:
        engine = create_engine('sqlite:///student.db')
        conn = engine.connect()

        df = pd.read_csv("students.csv")

        df.to_sql("students",conn, if_exists='replace')
        conn.close()
        return 0
    except Exception as e:
        print(e)
        return 1


def get_student(name):
    try:
        engine = create_engine('sqlite:///student.db')
        conn = engine.connect()

        df = pd.read_sql_query("SELECT name FROM students WHERE email = '{}'".format(name),conn)
        conn.close()
        print(df)
        return df.to_dict()

    except Exception as e:
        print(e)
        return {"Error": "No such name in DB"}

def get_students():
    try:
        engine = create_engine('sqlite:///student.db')
        conn = engine.connect()

        df = pd.read_sql_query("SELECT * FROM students",conn)
        conn.close()
        return df.to_dict()

    except Exception as e:
        print(e)
        return {"Error": "No such name in DB"}

create_db()