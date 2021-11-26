from sqlalchemy import create_engine
import pandas as pd


def create_db():

    try:
        engine = create_engine("sqlite:///student.db")
        conn = engine.connect()

        df = pd.read_csv("students.csv")

        df.to_sql("students", conn, if_exists='replace')
        conn.close()
        return 0
    except:
        return 1

def get_students():

    try:
        engine = create_engine("sqlite:///student.db")
        conn = engine.connect()

        df = pd.read_sql_query("SELECT * FROM students", conn)
        conn.close()
        return df.to_dict()

    except:

        return {"ERROR": "No table was found"}

def get_name(email):

    try:
        engine = create_engine("sqlite:///student.db")
        conn = engine.connect()

        df = pd.read_sql_query("SELECT name FROM students WHERE email = '{}'".format(email), conn)
        conn.close()

        return df.to_dict()

    except:

        return {"ERROR": "No email was found"}


create_db()