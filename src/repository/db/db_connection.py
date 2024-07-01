# Copyright 2024 ONYX Aikya. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from src.repository.db.db_connector import DBConnector
import mysql.connector


class DBConnection(object):
    connection = None

    @classmethod
    def get_connection(cls, new=False):
        """Creates return new Singleton database connection"""
        if new or not cls.connection:
            cls.connection = DBConnector().create_connection()
        return cls.connection

    @classmethod
    def execute_query(cls, query):
        """execute query on singleton db connection"""
        connection = cls.get_connection()
        try:
            cursor = connection.cursor()
        except mysql.connector.Error as err:
            print("db connection error: {}".format(err))
            connection = cls.get_connection(new=True)  # Create new connection
            cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result

    @classmethod
    def execute_fetch_one(cls, query):
        """execute query on singleton db connection"""
        connection = cls.get_connection()
        try:
            cursor = connection.cursor()
        except mysql.connector.Error as err:
            print("db connection error: {}".format(err))
            connection = cls.get_connection(new=True)  # Create new connection
            cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        return result
    @classmethod
    def execute_update(cls, query):
        """execute query on singleton db connection"""
        connection = cls.get_connection()
        try:
            cursor = connection.cursor()
        except mysql.connector.Error as err:
            print("db connection error: {}".format(err))
            connection = cls.get_connection(new=True)  # Create new connection
            cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()
        row_count = cursor.rowcount
        print(f"{row_count} records inserted/updated successfully into table")
        cursor.close()
        return row_count
    # Function to execute a batch insert

    @classmethod
    def execute_batch_insert(cls, query, values):
        connection = cls.get_connection()
        try:
            cursor = connection.cursor()
        except mysql.connector.Error as err:
            print("db connection error: {}".format(err))
            connection = cls.get_connection(new=True)  # Create new connection
            cursor = connection.cursor()
        # Prepare a list of tuples from the data attributes
        # Execute the batch insert
        cursor.executemany(query, values)
        connection.commit()
        row_count = cursor.rowcount
        print(f"{row_count} records inserted successfully into table")
        cursor.close()
        return row_count
