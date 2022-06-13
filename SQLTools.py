import mysql.connector
from mysql.connector import Error


# create SQL connection
def create_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database='stock_data'
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


# execute a SQL query
def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

    return cursor


if __name__ == "__main__":
    conn = create_connection("localhost", "root", "Plasma13sword!")
    data = execute_query(conn, "SELECT * FROM stock_data.stock_history_ibm")

    for dat in data:
        print(str(dat[0]))
