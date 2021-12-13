import pandas as pd
import mysql.connector as mysql  # connecting with mysql
from mysql.connector import Error
import sys  # sys module

# import csv file into python by using pandas library
# create data frame--attendancedata

# create database named student_Attendance
# database code#

# import mysql.connector as msql
# from mysql.connector import Error
# try:
# conn = msql.connect(host='localhost', user='root',   password='')#user= username, password=password
# if conn.is_connected():
# cursor = conn.cursor()
# cursor.execute("CREATE DATABASE student_Attendance")
#  print("Created")
#  except Error as e:
# print("Error while connecting to MySQL", e)
# end of the code


def read_csv(filename):
    attendancedata = pd.read_csv(filename, index_col=False, delimiter=',')
    print(attendancedata.head())
    return attendancedata
    # if you want to access another csv file , please change the file name


def database(attendance_table):
    try:

        # connecting  with student_attendance database
        conn = mysql.connect(host='localhost', database='student_Attendance', user='root', password='')
        if conn.is_connected():
            cursor = conn.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)

            # creating  table :attendance_data
            cursor.execute('DROP TABLE IF EXISTS attendance_data;')
            print('Creating table...')

            # passing the create table statement which  want (column names)
            cursor.execute(
                "CREATE TABLE attendance_data(index_no varchar(255),student varchar(255),date varchar(255),attendance varchar(255))")
            print("Table is created....")

            # looping  through the data frame(attendance data)
            for x, row in attendance_table.iterrows():
                sql = "INSERT INTO student_Attendance.attendance_data VALUES (%s,%s,%s,%s)"
                cursor.execute(sql, tuple(row))
                print("Record inserted to the database table")
                # commit changes
                conn.commit()

            # Execute query
            sql = "SELECT * FROM student_Attendance. attendance_data"
            cursor.execute(sql)

            # Fetching  all the records in the student_Attendance_25-08-2021.csv
            result = cursor.fetchall()
            for x in result:
                print(x)

            # returning message
            return 'Student Attendance Table successfully created'
    except Error as e:
        print("Error while connecting to MySQL", e)


# calling main method and command line arguments
if __name__ == '__main__':
    csv_filename = sys.argv[1]
    print(sys.argv[0])
    print(sys.argv[1])
    attendance_data = read_csv(csv_filename)
    status = database(attendance_data)
    print(status)

# python Database.py Attendance_Sheet27-08-2021.csv



