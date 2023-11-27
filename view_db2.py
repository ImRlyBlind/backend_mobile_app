from flask import render_template, Flask, url_for, redirect, request, session
import mysql.connector
import csv
import pandas as pd
from io import StringIO

app = Flask(__name__)
app.secret_key = 'key'

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '12345678',
    'database': 'app_record'
}

# Dummy
staff_credentials = {'username': 'admin', 'password': 'admin123'}

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/view')
def view():
    if 'username' in session:

        connection = mysql.connector.connect(**db_config)
        cur = connection.cursor()

        cur.execute("SELECT id FROM app_rec")
        student_ids = [str(student_id) for (student_id,) in cur.fetchall()]
        session['student_ids'] = student_ids

        cur.execute("SELECT id, name, picture, result, picture_result FROM app_rec")


        data = cur.fetchall()

        cur.close()
        connection.close()

        num_students = len(student_ids)

        subject_name = session.get('subject_name', None)

        return render_template('index2.html', data=data, subject_name=subject_name, num_students=num_students, student_ids=student_ids)

    else:
        return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    error_message = None

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        #subject_name = request.form['subject']

        if username == staff_credentials['username'] and password == staff_credentials['password']:
            session['username'] = username
            #session['subject_name'] = subject_name
            print("Login successful")
            return redirect(url_for('add_student_subject'))

        else:
            error_message = 'Invalid username or password.'

        #return render_template('login.html', error=error_message)

    return render_template('login.html', error=error_message)
    #         if 'file' in request.files:
    #             file = request.files['file']
    #
    #             if file.filename:
    #                 if file.filename.endswith(('.xlsx', '.csv')):
    #                     if file.filename.endswith('.xlsx'):
    #                         students_df = pd.read_excel(file)
    #                     elif file.filename.endswith('.csv'):
    #                         students_df = pd.read_csv(file)
    #
    #                     students = students_df.to_dict(orient='records')
    #                     student_ids = students_df['Student ID'].tolist()
    #                     insert_into_db(students)
    #
    #                     # Update session data
    #                     session['num_students'] = len(student_ids)
    #                     session['student_ids'] = student_ids
    #
    #                     print("number ", len(student_ids))
    #
    #                     print("List of Students:", students)
    #                     session['username'] = 'user'
    #
    #                     return redirect(url_for('view'))
    #                 else:
    #                     error_message = 'Please upload a valid Excel (.xlsx) or CSV (.csv) file.'
    #             else:
    #                 error_message = 'No file uploaded.'
    #         else:
    #             error_message = 'No file uploaded.'
    #
    #     else:
    #         error_message = 'Invalid username or password.'
    #
    # return render_template('login.html', error=error_message)

@app.route('/add_student_subject', methods=['GET', 'POST'])
def add_student_subject():
    if 'username' in session:
        if request.method == 'POST':

            subject_name = request.form['subject']
            session['subject_name'] = subject_name

            if 'file' in request.files:
                file = request.files['file']

                if file.filename:
                    if file.filename.endswith(('.xlsx', '.csv')):
                        if file.filename.endswith('.xlsx'):
                            students_df = pd.read_excel(file)
                        elif file.filename.endswith('.csv'):
                            students_df = pd.read_csv(file)

                        students = students_df.to_dict(orient='records')
                        student_ids = students_df['Student ID'].tolist()
                        insert_into_db_from_index(students)

                        num_students = len(student_ids)
                        session['num_students'] = num_students
                        session['student_ids'] = student_ids

                        print("number ", num_students)
                        print("List of Students:", students)
                        return redirect(url_for('view'))
                    else:
                        error_message = 'Please upload a valid Excel (.xlsx) or CSV (.csv) file.'
                        return render_template('add_student_subject.html', error=error_message)
                else:
                    error_message = 'No file uploaded.'
                    return render_template('add_student_subject.html', error=error_message)
            else:
                error_message = 'No file uploaded.'
                return render_template('add_student_subject.html', error=error_message)

        return render_template('add_student_subject.html', error=None)

    else:
        return redirect(url_for('login'))

def insert_into_db(student_ids):
    connection = mysql.connector.connect(**db_config)
    cur = connection.cursor()
    sql = ("INSERT INTO app_rec (id) "
           "VALUES (%s) "
           "ON DUPLICATE KEY UPDATE id = id"
           )
    print(student_ids)
    for sid in student_ids:
        value = (sid['Student ID'],)
        print(sid)
        cur.execute(sql, value)

    connection.commit()
    cur.close()
    connection.close()

@app.route("/delete_confirm/<int:id>")
def delete_confirm(id):
    connection = mysql.connector.connect(**db_config)
    cur = connection.cursor()
    cur.execute("SELECT * FROM app_rec WHERE id = %s", (id,))
    record = cur.fetchone()
    dict_rec = {'id': record[0], 'name': record[1], 'picture': record[2], 'result': record[3], 'picture_result': record[4]}
    cur.close()

    return render_template('delete_record.html', record=dict_rec)

@app.route("/delete_record/<int:id>", methods=['POST'])
def delete_record(id):
    num_students = session.get('num_students', 0)
    student_ids = session.get('student_ids', [])
    print(student_ids)
    connection = mysql.connector.connect(**db_config)
    cur = connection.cursor()
    student_ids.remove(str(id))
    cur.execute("DELETE FROM app_rec WHERE id = %s", (id,))
    connection.commit()
    cur.close()
    num_students-=1
    session['num_students'] = num_students
    session['student_ids'] = student_ids

    return redirect(url_for('view'))

@app.route('/update_add_student', methods=['POST'])
def update_add_student():
    if 'username' in session:
        if request.method == 'POST':
            # Assuming you have a form with input fields for student information
            student_id = request.form.get('student_id')
            student_name = request.form.get('student_name')

            connection = mysql.connector.connect(**db_config)
            cur = connection.cursor()

            # Check if the student already exists in the database
            cur.execute("SELECT id FROM app_rec WHERE id = %s", (student_id,))
            existing_student = cur.fetchone()

            if existing_student:
                # Update the existing student
                cur.execute("UPDATE app_rec SET name = %s WHERE id = %s", (student_name, student_id))
            else:
                # Add a new student
                cur.execute("INSERT INTO app_rec (id, name) VALUES (%s, %s)", (student_id, student_name))

            connection.commit()
            cur.close()
            connection.close()

        return redirect(url_for('view'))

    else:
        return redirect(url_for('login'))


@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        if 'manual_student_id' in request.form and 'manual_student_name' in request.form:
            # Handle manual entry
            student_id = request.form['manual_student_id']
            student_name = request.form['manual_student_name']

            # Insert student into the database
            insert_into_db_from_index((int(student_id), student_name))

            return redirect(url_for('view'))

        elif 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file.filename:
                if file.filename.endswith(('.xlsx', '.xls', '.csv')):
                    # Process file based on its format
                    if file.filename.endswith(('.xlsx', '.xls')):
                        student_df = pd.read_excel(file)
                    elif file.filename.endswith('.csv'):
                        student_df = pd.read_csv(StringIO(file.read().decode('utf-8')))

                    # Insert students into the database
                    insert_into_db_from_index(student_df)
                    return redirect(url_for('view'))

        # If the file format is not supported or no file is uploaded, handle accordingly
        return render_template('add_student.html', error="Invalid file format or no file uploaded")

    # Render the add student page for GET requests
    return render_template('add_student.html', error=None)

def insert_into_db_from_index(students):
    num_students = session.get('num_students', 0)
    student_ids = session.get('student_ids', [])

    connection = mysql.connector.connect(**db_config)
    cur = connection.cursor()
    sql_select = "SELECT id FROM app_rec WHERE id = %s"
    sql_insert_update = ("INSERT INTO app_rec (id, name) "
                         "VALUES (%s, %s) "
                         "ON DUPLICATE KEY UPDATE id = id, name = VALUES(name)"
                         )


    if isinstance(students, pd.DataFrame):
        for index, student in students.iterrows():
            student_id = int(student['Student ID'])
            student_name = student.get('Name', '')

            value = (student_id, student_name)

            cur.execute(sql_select, (student_id,))
            exist_id = cur.fetchone()

            if exist_id:
                cur.execute(sql_insert_update, value)
            else:
                cur.execute(sql_insert_update, value)
                student_ids.append(student_id)
                num_students += 1

    elif isinstance(students, tuple):
        student_id, student_name = students
        value = (student_id, student_name)

        cur.execute(sql_select, (student_id,))
        exist_id = cur.fetchone()

        if exist_id:
            cur.execute(sql_insert_update, value)
        else:
            cur.execute(sql_insert_update, value)
            student_ids.append(student_id)
            num_students += 1

    session['num_students'] = num_students
    session['student_ids'] = student_ids

    connection.commit()
    cur.close()
    connection.close()

@app.route('/edit_student/<int:id>', methods=['GET', 'POST'])
def edit_student(id):
    connection = mysql.connector.connect(**db_config)
    cur = connection.cursor()
    cur.execute("SELECT * FROM app_rec WHERE id = %s", (id,))
    student = cur.fetchone()
    cur.close()

    if request.method == 'POST':
        new_id = request.form['new_id']
        new_name = request.form['new_name']

        # Update the student information in the database
        connection = mysql.connector.connect(**db_config)
        cur = connection.cursor()
        cur.execute("UPDATE app_rec SET id = %s, name = %s WHERE id = %s", (new_id, new_name, id))
        connection.commit()
        cur.close()

        # Redirect to the index page after updating
        return redirect(url_for('view'))

    return render_template('edit.html', student=student)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)